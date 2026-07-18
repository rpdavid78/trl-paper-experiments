#!/usr/bin/env python3
"""Paired SWAG-Diag FixBN audit for the ICLR artifact.

This script never trains or mutates the published checkpoints.  Each SWAG draw
and each augmented calibration batch is shared by the legacy rolling arm and
the independent reset/cumulative arm.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts" / "cifar100_all_methods_iclr.py")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_iclr" / "swag_diag_fixbn_ab_seed0.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--fixbn-batches", type=int, default=20)
    parser.add_argument("--swag-stats", default="c100_swag_stats.pth")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, repo_root: Path):
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("trl_cifar100_audit_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_state(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
    raise TypeError(f"Unrecognized model checkpoint: {path}")


def load_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def materialize_batches(loader, count: int, seed: int):
    seed_everything(seed)
    iterator = iter(loader)
    batches = []
    for _ in range(count):
        try:
            x, y = next(iterator)
        except StopIteration:
            break
        batches.append((x.contiguous(), y.contiguous()))
    del iterator
    if len(batches) != count:
        raise RuntimeError(f"Requested {count} FixBN batches, got {len(batches)}")
    return batches


def bn_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def fixbn_legacy_rolling(model: nn.Module, batches, device: torch.device) -> None:
    model.train()
    for x, _ in batches:
        model(x.to(device, non_blocking=True))
    model.eval()


@torch.no_grad()
def fixbn_independent_reset_cumulative(model: nn.Module, batches, device: torch.device) -> None:
    modules = bn_modules(model)
    momenta = [m.momentum for m in modules]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    for x, _ in batches:
        model(x.to(device, non_blocking=True))
    model.eval()
    for module, momentum in zip(modules, momenta):
        module.momentum = momentum


@torch.no_grad()
def predict_probs(model: nn.Module, loader, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    for x, _ in loader:
        chunks.append(torch.softmax(model(x.to(device, non_blocking=True)), dim=1).cpu())
    return torch.cat(chunks)


def bn_buffer_rms(left: nn.Module, right: nn.Module) -> float:
    total = 0.0
    count = 0
    for lhs, rhs in zip(bn_modules(left), bn_modules(right)):
        for name in ("running_mean", "running_var"):
            delta = getattr(lhs, name).double() - getattr(rhs, name).double()
            total += delta.square().sum().item()
            count += delta.numel()
    return math.sqrt(total / count)


def metrics(runner, probs_id, probs_ood, targets, num_classes: int):
    acc, nll, ece, brier = runner.calc_metrics(probs_id, targets, num_classes)
    auroc = runner.auroc_entropy(probs_id, probs_ood)
    return {
        "acc": float(acc),
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
        "entropy_auroc": float(auroc),
    }


def main() -> None:
    args = parse_args()
    args.runner = args.runner.resolve()
    args.repo_root = args.repo_root.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.output = args.output.resolve()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    map_path = args.ckpt_dir / "resnet18_cifar100_map.pth"
    stats_path = args.ckpt_dir / args.swag_stats
    for path in (args.runner, map_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.chdir(args.repo_root)
    runner = load_module(args.runner, args.repo_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runner.DEVICE = device

    cfg = runner.CFG()
    cfg.seed = args.seed
    cfg.ckpt_dir = str(args.ckpt_dir)
    cfg.num_workers = args.num_workers
    cfg.swag_samples = args.samples
    cfg.swag_fixbn_batches = args.fixbn_batches

    seed_everything(args.seed)
    train_aug, _, _, test_loader, ood_loader = runner.get_data(cfg)
    targets = runner.get_targets(test_loader)

    map_model = runner.ResNetCIFAR(cfg.num_classes, use_dropout=False).to(device)
    map_model.load_state_dict(load_state(map_path))
    map_model.eval()
    rolling_model = copy.deepcopy(map_model).to(device)
    reset_model = copy.deepcopy(map_model).to(device)

    stats = load_payload(stats_path)
    means = stats.get("mean")
    square_means = stats.get("sq_mean")
    if not isinstance(means, list) or not isinstance(square_means, list):
        raise TypeError("SWAG payload must contain list fields 'mean' and 'sq_mean'")
    if len(means) != len(list(map_model.parameters())) or len(square_means) != len(means):
        raise ValueError("SWAG moment count does not match model parameter count")

    rolling_id = []
    rolling_ood = []
    reset_id = []
    reset_ood = []
    per_draw = []
    start = time.perf_counter()

    draw_generator = torch.Generator(device=device)
    draw_generator.manual_seed(1_000_000 + args.seed)

    for draw in range(args.samples):
        max_parameter_diff = 0.0
        with torch.no_grad():
            for p_roll, p_reset, mean_cpu, square_mean_cpu in zip(
                rolling_model.parameters(), reset_model.parameters(), means, square_means
            ):
                mean = mean_cpu.to(device=device, dtype=p_roll.dtype)
                square_mean = square_mean_cpu.to(device=device, dtype=p_roll.dtype)
                variance = torch.clamp(square_mean - mean.square(), min=1e-30)
                noise = torch.randn(
                    p_roll.shape,
                    dtype=p_roll.dtype,
                    device=device,
                    generator=draw_generator,
                )
                sampled = mean + math.sqrt(float(cfg.swag_sample_scale)) * torch.sqrt(variance) * noise
                p_roll.copy_(sampled)
                p_reset.copy_(sampled)
                max_parameter_diff = max(
                    max_parameter_diff,
                    float((p_roll - p_reset).abs().max().item()),
                )

        calibration_seed = 2_000_000 + args.seed * 10_000 + draw
        batches = materialize_batches(train_aug, args.fixbn_batches, calibration_seed)
        fixbn_legacy_rolling(rolling_model, batches, device)
        fixbn_independent_reset_cumulative(reset_model, batches, device)

        p_roll_id = predict_probs(rolling_model, test_loader, device)
        p_roll_ood = predict_probs(rolling_model, ood_loader, device)
        p_reset_id = predict_probs(reset_model, test_loader, device)
        p_reset_ood = predict_probs(reset_model, ood_loader, device)
        rolling_id.append(p_roll_id)
        rolling_ood.append(p_roll_ood)
        reset_id.append(p_reset_id)
        reset_ood.append(p_reset_ood)

        draw_row = {
            "draw": draw,
            "calibration_seed": calibration_seed,
            "parameter_max_abs_diff": max_parameter_diff,
            "bn_buffer_rms": bn_buffer_rms(rolling_model, reset_model),
            "id_probability_mae": float((p_roll_id - p_reset_id).abs().mean().item()),
        }
        per_draw.append(draw_row)
        print(json.dumps(draw_row), flush=True)
        del batches, p_roll_id, p_roll_ood, p_reset_id, p_reset_ood

    p_rolling_id = torch.stack(rolling_id).mean(0)
    p_rolling_ood = torch.stack(rolling_ood).mean(0)
    p_reset_id = torch.stack(reset_id).mean(0)
    p_reset_ood = torch.stack(reset_ood).mean(0)
    rolling_metrics = metrics(runner, p_rolling_id, p_rolling_ood, targets, cfg.num_classes)
    reset_metrics = metrics(runner, p_reset_id, p_reset_ood, targets, cfg.num_classes)
    delta = {key: reset_metrics[key] - rolling_metrics[key] for key in rolling_metrics}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result = {
        "experiment": "swag_diag_fixbn_rolling_vs_independent_reset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "swag_samples": args.samples,
        "fixbn_batches": args.fixbn_batches,
        "pairing": "identical SWAG parameter draws and identical augmented batches per draw",
        "rolling_definition": "no BN reset; checkpoint/sample buffers roll forward; original momentum",
        "reset_definition": "reset_running_stats per draw; momentum=None cumulative average; momentum restored",
        "runner": str(args.runner),
        "runner_sha256": sha256_file(args.runner),
        "map_checkpoint": str(map_path),
        "map_checkpoint_sha256": sha256_file(map_path),
        "swag_stats": str(stats_path),
        "swag_stats_sha256": sha256_file(stats_path),
        "swag_snapshots": int(stats.get("n", -1)),
        "rolling": rolling_metrics,
        "independent_reset": reset_metrics,
        "delta_reset_minus_rolling": delta,
        "predictive_probability_mae": float((p_rolling_id - p_reset_id).abs().mean().item()),
        "elapsed_sec": time.perf_counter() - start,
        "per_draw": per_draw,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
