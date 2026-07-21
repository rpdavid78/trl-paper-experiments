#!/usr/bin/env python3
"""Paired TRL FixBN audit for the CIFAR-100 ICLR experiment.

The audit reuses an existing MAP checkpoint and saved TRL spine; it never
trains and never mutates either artifact.  For every posterior draw, the
legacy ``rolling`` and corrected ``reset`` arms receive exactly the same
spine anchor, transverse Gaussian vector, and augmented FixBN batches.

Three checks are produced in one JSON artifact:

* beta=4 (configurable) test/OOD metrics for rolling versus reset;
* the full validation beta sweep, with a separately selected beta per arm;
* forward versus reverse posterior-draw order at the fixed beta.

The fixed-beta comparison starts both arms from the MAP BatchNorm buffers.
It is therefore a controlled causal audit of FixBN semantics, not an exact
replay of any HVP-contaminated buffers that were not stored in the spine.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
PHASES = (
    "fixed-forward",
    "fixed-reverse",
    "sweep-forward",
    "sweep-reverse",
    "pipeline-forward",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        type=Path,
        default=ROOT / "scripts" / "cifar100_all_methods_iclr.py",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing data/; defaults to --repo-root.",
    )
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--map-checkpoint", default="resnet18_cifar100_map.pth")
    parser.add_argument("--trl-spine", default="c100_trl_stage2_spine.pth")
    parser.add_argument(
        "--historical-runner",
        type=Path,
        default=None,
        help="Optional unmodified runner used for the published run; recorded for lineage only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_iclr" / "trl_fixbn_ab_seed0.json",
    )
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--fixbn-batches", type=int, default=25)
    parser.add_argument("--fixed-beta", type=float, default=4.0)
    parser.add_argument(
        "--tube-scales",
        type=float,
        nargs="+",
        default=(2.0, 3.0, 4.0, 6.0, 10.0, 20.0),
    )
    parser.add_argument("--posterior-seed", type=int, default=1000)
    parser.add_argument("--calibration-seed-base", type=int, default=2_000_000)
    parser.add_argument(
        "--final-posterior-seed",
        type=int,
        default=1001,
        help="Distinct posterior bank for the post-sweep test/OOD phase.",
    )
    parser.add_argument(
        "--final-calibration-seed-base",
        type=int,
        default=3_000_000,
        help="Distinct augmented-batch bank for the post-sweep test/OOD phase.",
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help="Optional source commit recorded for audit lineage.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PHASES,
        default=PHASES,
        help="Select independent audit phases; useful for splitting work across GPUs.",
    )
    parser.add_argument(
        "--hash-large-artifacts",
        action="store_true",
        help="Also SHA-256 the (potentially tens-of-GB) spine; size+mtime are always recorded.",
    )
    parser.add_argument(
        "--no-reset-ema",
        action="store_true",
        help="Skip the diagnostic reset+EMA arm (rolling/reset-cumulative remain mandatory).",
    )
    parser.add_argument(
        "--no-reverse-beta-sweep",
        action="store_true",
        help="Skip the reverse beta-grid order check.",
    )
    parser.add_argument("--nll-threshold", type=float, default=0.01)
    parser.add_argument("--ece-threshold", type=float, default=0.005)
    parser.add_argument("--brier-threshold", type=float, default=0.005)
    parser.add_argument("--acc-threshold", type=float, default=0.002)
    parser.add_argument("--auroc-threshold", type=float, default=0.01)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: Path, *, include_sha256: bool) -> Dict[str, object]:
    stat = path.stat()
    identity: Dict[str, object] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        identity["sha256"] = sha256_file(path)
    else:
        identity["sha256"] = None
        identity["sha256_note"] = "omitted; pass --hash-large-artifacts to compute"
    return identity


def load_module(path: Path, repo_root: Path):
    # The canonical runner keeps ``trl_iclr_utils`` beside itself under
    # ``scripts/``.  Insert both locations so the diagnostic works when
    # invoked as a script, not only when a test harness preloads scripts/.
    for candidate in (repo_root, path.parent):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("trl_fixbn_audit_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_payload(path: Path, *, mmap: bool = False):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except TypeError:
        if mmap:
            raise RuntimeError(
                "This diagnostic requires a torch.load implementation with mmap support "
                "for the TRL spine."
            )
        return torch.load(path, map_location="cpu")


def load_state(path: Path) -> Mapping[str, torch.Tensor]:
    payload = load_payload(path)
    if isinstance(payload, nn.Module):
        state = payload.state_dict()
    elif isinstance(payload, Mapping) and payload and all(
        torch.is_tensor(value) for value in payload.values()
    ):
        state = payload
    elif isinstance(payload, Mapping):
        state = None
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, nn.Module):
                state = candidate.state_dict()
                break
            if isinstance(candidate, Mapping):
                state = candidate
                break
        if state is None:
            raise TypeError(f"Unrecognized model checkpoint mapping: {path}")
    else:
        raise TypeError(f"Unrecognized model checkpoint: {path}")

    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return state


def resolve_artifact(ckpt_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ckpt_dir / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def materialize_batches(loader, count: int, seed: int):
    """Materialize one deterministic augmented calibration bank on CPU."""
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


def bn_modules(model: nn.Module) -> List[nn.Module]:
    return [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def snapshot_bn_buffers(model: nn.Module) -> torch.Tensor:
    chunks = []
    for module in bn_modules(model):
        chunks.extend(
            [module.running_mean.detach().cpu().float(), module.running_var.detach().cpu().float()]
        )
    if not chunks:
        return torch.empty(0)
    return torch.cat([chunk.reshape(-1) for chunk in chunks])


def tensor_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"RMS shape mismatch: {left.shape} != {right.shape}")
    if left.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean((left.double() - right.double()).square())).item())


def canonical_draw_mean(draws: Mapping[int, torch.Tensor]) -> torch.Tensor:
    if not draws:
        raise ValueError("Cannot aggregate an empty draw mapping")
    canonical = sorted(draws)
    total = draws[canonical[0]].clone()
    for draw_id in canonical[1:]:
        total.add_(draws[draw_id])
    return total.div_(len(canonical))


@torch.no_grad()
def predict_probs(model: nn.Module, loader, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    for x, _ in loader:
        chunks.append(torch.softmax(model(x.to(device, non_blocking=True)), dim=1).cpu())
    return torch.cat(chunks)


@dataclass(frozen=True)
class PosteriorDraw:
    draw_id: int
    anchor_index: int
    z: torch.Tensor


def build_draw_bank(spine: Sequence[Mapping[str, torch.Tensor]], samples: int, seed: int):
    if samples < 1:
        raise ValueError("samples must be positive")
    if not spine:
        raise ValueError("TRL spine is empty")
    ranks = {int(point["N"].shape[1]) for point in spine}
    if len(ranks) != 1:
        raise ValueError(f"Inconsistent transverse ranks in spine: {sorted(ranks)}")
    rank = ranks.pop()
    anchor_rng = np.random.RandomState(seed)
    noise_rng = torch.Generator(device="cpu")
    noise_rng.manual_seed(seed)
    return [
        PosteriorDraw(
            draw_id=draw_id,
            anchor_index=int(anchor_rng.randint(len(spine))),
            z=torch.randn(rank, generator=noise_rng),
        )
        for draw_id in range(samples)
    ]


def sampled_theta(
    spine: Sequence[Mapping[str, torch.Tensor]],
    draw: PosteriorDraw,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    point = spine[draw.anchor_index]
    theta = point["theta"].to(device)
    basis = point["N"].to(device)
    scale = point["inv_sqrt_prec"].to(device)
    z = draw.z.to(device=device, dtype=theta.dtype)
    if basis.shape[0] != theta.numel() or basis.shape[1] != z.numel():
        raise ValueError("Spine theta/N/z dimensions are inconsistent")
    return theta + basis @ (float(beta) * (scale * z))


def init_arm_models(
    base_model: nn.Module, device: torch.device, *, include_reset_ema: bool
) -> Dict[str, nn.Module]:
    models = {
        "rolling": copy.deepcopy(base_model).to(device).eval(),
        "reset": copy.deepcopy(base_model).to(device).eval(),
    }
    if include_reset_ema:
        models["reset_ema"] = copy.deepcopy(base_model).to(device).eval()
    return models


def _assign_theta(runner, theta: torch.Tensor, model: nn.Module) -> None:
    expected = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if theta.numel() != expected:
        raise ValueError(
            f"Spine theta has {theta.numel()} values, model trainable subspace has {expected}"
        )
    runner._trl_vector_to_trainable_parameters(theta, model)


def evaluate_paired(
    *,
    runner,
    base_model: nn.Module,
    spine: Sequence[Mapping[str, torch.Tensor]],
    beta,
    draw_bank: Sequence[PosteriorDraw],
    draw_order: Sequence[int],
    calibration_loader,
    eval_loaders: Mapping[str, object],
    fixbn_batches: int,
    calibration_seed_base: int,
    device: torch.device,
    include_reset_ema: bool = True,
    arm_models: Optional[MutableMapping[str, nn.Module]] = None,
):
    """Evaluate paired arms and return ensemble probabilities plus audit traces.

    ``arm_models`` may be supplied to carry legacy rolling BN buffers across
    beta values during the validation sweep.  Fixed-beta/order runs omit it
    and therefore start from identical MAP buffers.
    """
    models = (
        arm_models
        if arm_models is not None
        else init_arm_models(base_model, device, include_reset_ema=include_reset_ema)
    )
    draw_by_id = {draw.draw_id: draw for draw in draw_bank}
    if set(draw_order) != set(draw_by_id):
        raise ValueError("draw_order must contain every draw_id exactly once")

    draw_probabilities: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {
        arm: {split: {} for split in eval_loaders} for arm in models
    }
    bn_snapshots: Dict[str, Dict[int, torch.Tensor]] = {arm: {} for arm in models}
    per_draw = []
    start = time.perf_counter()

    for position, draw_id in enumerate(draw_order):
        draw = draw_by_id[draw_id]
        beta_by_arm = (
            {arm: float(beta[arm]) for arm in models}
            if isinstance(beta, Mapping)
            else {arm: float(beta) for arm in models}
        )
        theta_by_beta = {
            arm_beta: sampled_theta(spine, draw, arm_beta, device)
            for arm_beta in sorted(set(beta_by_arm.values()))
        }
        for arm, model in models.items():
            _assign_theta(runner, theta_by_beta[beta_by_arm[arm]], model)

        calibration_seed = int(calibration_seed_base + draw_id)
        batches = materialize_batches(calibration_loader, fixbn_batches, calibration_seed)
        runner.fix_bn(models["rolling"], batches, device, fixbn_batches, mode="rolling")
        runner.fix_bn(models["reset"], batches, device, fixbn_batches, mode="reset")
        if "reset_ema" in models:
            for module in bn_modules(models["reset_ema"]):
                module.reset_running_stats()
            runner.fix_bn(
                models["reset_ema"], batches, device, fixbn_batches, mode="rolling"
            )

        for arm, model in models.items():
            bn_snapshots[arm][draw_id] = snapshot_bn_buffers(model)

        draw_row = {
            "position": int(position),
            "draw_id": int(draw_id),
            "anchor_index": int(draw.anchor_index),
            "calibration_seed": calibration_seed,
            "beta_by_arm": beta_by_arm,
            "bn_buffer_rms_rolling_vs_reset": tensor_rms(
                bn_snapshots["rolling"][draw_id], bn_snapshots["reset"][draw_id]
            ),
        }
        if "reset_ema" in models:
            draw_row["bn_buffer_rms_rolling_vs_reset_ema"] = tensor_rms(
                bn_snapshots["rolling"][draw_id], bn_snapshots["reset_ema"][draw_id]
            )
            draw_row["bn_buffer_rms_reset_ema_vs_reset"] = tensor_rms(
                bn_snapshots["reset_ema"][draw_id], bn_snapshots["reset"][draw_id]
            )
        for split, loader in eval_loaders.items():
            paired_probs = {}
            for arm, model in models.items():
                probs = predict_probs(model, loader, device)
                draw_probabilities[arm][split][draw_id] = probs
                paired_probs[arm] = probs
            draw_row[f"{split}_probability_mae_rolling_vs_reset"] = float(
                (paired_probs["rolling"] - paired_probs["reset"]).abs().mean().item()
            )
            if "reset_ema" in paired_probs:
                draw_row[f"{split}_probability_mae_rolling_vs_reset_ema"] = float(
                    (paired_probs["rolling"] - paired_probs["reset_ema"]).abs().mean().item()
                )
                draw_row[f"{split}_probability_mae_reset_ema_vs_reset"] = float(
                    (paired_probs["reset_ema"] - paired_probs["reset"]).abs().mean().item()
                )
            del paired_probs

        per_draw.append(draw_row)
        print(json.dumps({"beta": beta_by_arm, **draw_row}), flush=True)
        del batches, theta_by_beta

    # Aggregate in canonical draw_id order, independent of execution order.
    # This prevents floating-point summation order from masquerading as a
    # forward/reverse dependency in the reset arms.
    probabilities = {}
    for arm, split_draws in draw_probabilities.items():
        probabilities[arm] = {}
        for split, draws in split_draws.items():
            probabilities[arm][split] = canonical_draw_mean(draws)
    return {
        "probabilities": probabilities,
        "bn_snapshots": bn_snapshots,
        "per_draw": per_draw,
        "elapsed_sec": time.perf_counter() - start,
        "models": models,
    }


def classification_metrics(runner, probs: torch.Tensor, targets: torch.Tensor, classes: int):
    acc, nll, ece, brier = runner.calc_metrics(probs, targets, classes)
    return {
        "acc": float(acc),
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
    }


def summarize_fixed(runner, evaluated, targets: torch.Tensor, classes: int):
    arms = {}
    for arm, probs in evaluated["probabilities"].items():
        row = classification_metrics(runner, probs["id"], targets, classes)
        row["entropy_auroc"] = float(runner.auroc_entropy(probs["id"], probs["ood"]))
        arms[arm] = row
    contrasts = {
        "reset_minus_rolling": {
            key: arms["reset"][key] - arms["rolling"][key] for key in arms["rolling"]
        }
    }
    if "reset_ema" in arms:
        contrasts["reset_ema_minus_rolling"] = {
            key: arms["reset_ema"][key] - arms["rolling"][key] for key in arms["rolling"]
        }
        contrasts["reset_minus_reset_ema"] = {
            key: arms["reset"][key] - arms["reset_ema"][key] for key in arms["rolling"]
        }
    bn_values = [row["bn_buffer_rms_rolling_vs_reset"] for row in evaluated["per_draw"]]
    return {
        "rolling": arms["rolling"],
        "reset": arms["reset"],
        **({"reset_ema": arms["reset_ema"]} if "reset_ema" in arms else {}),
        "delta_reset_minus_rolling": contrasts["reset_minus_rolling"],
        "contrasts": contrasts,
        "id_probability_mae": float(
            (
                evaluated["probabilities"]["rolling"]["id"]
                - evaluated["probabilities"]["reset"]["id"]
            )
            .abs()
            .mean()
            .item()
        ),
        "ood_probability_mae": float(
            (
                evaluated["probabilities"]["rolling"]["ood"]
                - evaluated["probabilities"]["reset"]["ood"]
            )
            .abs()
            .mean()
            .item()
        ),
        "probability_mae_contrasts": {
            "reset_minus_rolling_id": float(
                (
                    evaluated["probabilities"]["rolling"]["id"]
                    - evaluated["probabilities"]["reset"]["id"]
                ).abs().mean().item()
            ),
            **(
                {
                    "reset_ema_minus_rolling_id": float(
                        (
                            evaluated["probabilities"]["rolling"]["id"]
                            - evaluated["probabilities"]["reset_ema"]["id"]
                        ).abs().mean().item()
                    ),
                    "reset_minus_reset_ema_id": float(
                        (
                            evaluated["probabilities"]["reset_ema"]["id"]
                            - evaluated["probabilities"]["reset"]["id"]
                        ).abs().mean().item()
                    ),
                }
                if "reset_ema" in evaluated["probabilities"] else {}
            ),
        },
        "bn_buffer_rms": {
            "mean_across_draws": float(np.mean(bn_values)),
            "max_across_draws": float(np.max(bn_values)),
            "last_draw": float(bn_values[-1]),
        },
        "elapsed_sec": float(evaluated["elapsed_sec"]),
        "per_draw": evaluated["per_draw"],
    }


def summarize_order_effect(runner, forward, reverse, targets: torch.Tensor, classes: int):
    result = {}
    for arm in forward["probabilities"]:
        forward_id = forward["probabilities"][arm]["id"]
        reverse_id = reverse["probabilities"][arm]["id"]
        forward_ood = forward["probabilities"][arm]["ood"]
        reverse_ood = reverse["probabilities"][arm]["ood"]
        forward_metrics = classification_metrics(runner, forward_id, targets, classes)
        reverse_metrics = classification_metrics(runner, reverse_id, targets, classes)
        forward_metrics["entropy_auroc"] = float(runner.auroc_entropy(forward_id, forward_ood))
        reverse_metrics["entropy_auroc"] = float(runner.auroc_entropy(reverse_id, reverse_ood))
        bn_by_draw = {
            str(draw_id): tensor_rms(
                forward["bn_snapshots"][arm][draw_id], reverse["bn_snapshots"][arm][draw_id]
            )
            for draw_id in forward["bn_snapshots"][arm]
        }
        result[arm] = {
            "forward": forward_metrics,
            "reverse": reverse_metrics,
            "delta_reverse_minus_forward": {
                key: reverse_metrics[key] - forward_metrics[key] for key in forward_metrics
            },
            "id_probability_mae": float((forward_id - reverse_id).abs().mean().item()),
            "ood_probability_mae": float((forward_ood - reverse_ood).abs().mean().item()),
            "bn_buffer_rms_same_draw": {
                "mean": float(np.mean(list(bn_by_draw.values()))),
                "max": float(np.max(list(bn_by_draw.values()))),
                "by_draw": bn_by_draw,
            },
        }
    return result


def run_sweep(
    *, runner, base_model, spine, draw_bank, calibration_loader, val_loader,
    targets, tube_scales, fixbn_batches, calibration_seed_base, device, classes,
    include_reset_ema,
):
    """Run the canonical beta order while carrying legacy rolling buffers."""
    models = init_arm_models(base_model, device, include_reset_ema=include_reset_ema)
    rows = []
    order = [draw.draw_id for draw in draw_bank]
    for beta in tube_scales:
        evaluated = evaluate_paired(
            runner=runner,
            base_model=base_model,
            spine=spine,
            beta=float(beta),
            draw_bank=draw_bank,
            draw_order=order,
            calibration_loader=calibration_loader,
            eval_loaders={"val": val_loader},
            fixbn_batches=fixbn_batches,
            calibration_seed_base=calibration_seed_base,
            device=device,
            include_reset_ema=include_reset_ema,
            arm_models=models,
        )
        models = evaluated["models"]
        row = {"beta": float(beta)}
        for arm in models:
            row[arm] = classification_metrics(
                runner, evaluated["probabilities"][arm]["val"], targets, classes
            )
        row["val_probability_mae"] = float(
            (
                evaluated["probabilities"]["rolling"]["val"]
                - evaluated["probabilities"]["reset"]["val"]
            )
            .abs()
            .mean()
            .item()
        )
        row["elapsed_sec"] = float(evaluated["elapsed_sec"])
        rows.append(row)

    selected = {}
    for arm in models:
        finite = [row for row in rows if math.isfinite(row[arm]["nll"])]
        if not finite:
            raise RuntimeError(f"No finite validation NLL for {arm}")
        winner = min(finite, key=lambda row: row[arm]["nll"])
        selected[arm] = {
            "beta": float(winner["beta"]),
            "val_nll": float(winner[arm]["nll"]),
        }
    return {"rows": rows, "selected": selected, "_models": models}


def _json_objects(path: Path) -> Iterable[object]:
    text = path.read_text(encoding="utf-8")
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _walk_dicts(value) -> Iterable[Mapping]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def find_reference_trl(path: Optional[Path], seed: int):
    if path is None:
        return None
    for obj in _json_objects(path):
        for row in _walk_dicts(obj):
            if str(row.get("method", "")).lower() == "trl" and int(row.get("seed", -1)) == seed:
                keys = ("acc", "nll", "ece", "brier", "auroc", "best_tube_scale")
                return {key: row[key] for key in keys if key in row}
    raise LookupError(f"No TRL seed={seed} row found in {path}")


def contextual_reference_comparison(reference, fixed_forward, pipeline_selected):
    if reference is None:
        return None
    key_map = {
        "acc": "acc",
        "nll": "nll",
        "ece": "ece",
        "brier": "brier",
        "auroc": "entropy_auroc",
    }
    comparisons = {}
    for phase, summary in (
        ("fixed_beta_from_map", fixed_forward),
        ("post_sweep_selected_beta", pipeline_selected),
    ):
        if summary is None:
            continue
        comparisons[phase] = {}
        for arm in ("rolling", "reset"):
            comparisons[phase][f"{arm}_minus_reported"] = {
                reported_key: float(summary[arm][audit_key]) - float(reference[reported_key])
                for reported_key, audit_key in key_map.items()
                if reported_key in reference
            }
    return {
        "comparisons": comparisons,
        "interpretation": (
            "Context only, not an exact reproduction check: post-HVP BN buffers were not "
            "saved and the audit uses an explicitly paired CPU-generated posterior bank, "
            "whereas the historical runner drew z on CUDA. Causal decisions use reset-minus-"
            "rolling contrasts within the audit, not these reported-result deltas."
        ),
    }


def beta_grid_order_effect(forward, reverse):
    if reverse is None:
        return None
    forward_by_beta = {row["beta"]: row for row in forward["rows"]}
    reverse_by_beta = {row["beta"]: row for row in reverse["rows"]}
    if forward_by_beta.keys() != reverse_by_beta.keys():
        raise ValueError("Forward/reverse beta sweeps contain different grids")
    arms = forward["selected"].keys()
    metric_keys = ("acc", "nll", "ece", "brier")
    per_beta = {
        str(beta): {
            arm: {
                f"{metric}_reverse_minus_forward": (
                    reverse_by_beta[beta][arm][metric]
                    - forward_by_beta[beta][arm][metric]
                )
                for metric in metric_keys
            }
            for arm in arms
        }
        for beta in sorted(forward_by_beta)
    }
    reset_deltas = [
        abs(float(delta))
        for row in per_beta.values()
        for delta in row["reset"].values()
    ]
    return {
        "selected_beta_changed": {
            arm: forward["selected"][arm]["beta"] != reverse["selected"][arm]["beta"]
            for arm in arms
        },
        "selected_forward": forward["selected"],
        "selected_reverse": reverse["selected"],
        "per_beta": per_beta,
        "reset_all_metrics_finite": all(math.isfinite(value) for value in reset_deltas),
        "reset_max_abs_metric_delta": max(reset_deltas),
    }


def threshold_decision(
    fixed_summary, pipeline_summary, sweep, reverse_sweep, order_effect, args
):
    thresholds = {
        "acc": args.acc_threshold,
        "nll": args.nll_threshold,
        "ece": args.ece_threshold,
        "brier": args.brier_threshold,
        "entropy_auroc": args.auroc_threshold,
    }
    def checks_for(summary):
        if summary is None:
            return None
        delta = summary["delta_reset_minus_rolling"]
        return {
            key: {
                "absolute_delta": abs(float(delta[key])),
                "threshold": float(limit),
                "finite": math.isfinite(float(delta[key])),
                "triggered": (
                    math.isfinite(float(delta[key]))
                    and abs(float(delta[key])) > float(limit)
                ),
            }
            for key, limit in thresholds.items()
        }

    fixed_checks = checks_for(fixed_summary)
    pipeline_checks = checks_for(pipeline_summary)
    triggered = []
    metric_triggers = []
    audit_valid = True
    for source, checks in (("fixed", fixed_checks), ("pipeline", pipeline_checks)):
        if checks is None:
            continue
        for key, value in checks.items():
            if not value["finite"]:
                triggered.append(f"{source}:{key}:nonfinite")
                audit_valid = False
                continue
            if value["triggered"]:
                label = f"{source}:{key}"
                triggered.append(label)
                metric_triggers.append(label)
    beta_changed = None
    if sweep is not None:
        beta_changed = (
            sweep["selected"]["rolling"]["beta"] != sweep["selected"]["reset"]["beta"]
        )
        if beta_changed:
            triggered.append("selected_beta")
    grid_order = (
        beta_grid_order_effect(sweep, reverse_sweep)
        if sweep is not None and reverse_sweep is not None else None
    )
    reset_grid_invariant = None
    if grid_order is not None:
        if grid_order["selected_beta_changed"]["rolling"]:
            triggered.append("rolling_beta_grid_order")
        reset_grid_invariant = (
            not grid_order["selected_beta_changed"]["reset"]
            and grid_order["reset_all_metrics_finite"]
            and float(grid_order["reset_max_abs_metric_delta"]) <= 1e-7
        )
        if not reset_grid_invariant:
            triggered.append("reset_beta_grid_order_invariance_failed")

    order_checks = None
    if order_effect is not None:
        reset_effect = order_effect["reset"]
        reset_metric_deltas = [
            abs(float(value))
            for value in reset_effect["delta_reverse_minus_forward"].values()
        ]
        reset_probability_deltas = [
            abs(float(reset_effect["id_probability_mae"])),
            abs(float(reset_effect["ood_probability_mae"])),
        ]
        reset_all_finite = all(
            math.isfinite(value)
            for value in reset_metric_deltas + reset_probability_deltas
        )
        reset_metric_max = max(reset_metric_deltas)
        reset_probability_max = max(reset_probability_deltas)
        reset_invariant = (
            reset_all_finite
            and reset_metric_max <= 1e-7
            and reset_probability_max <= 1e-8
        )
        rolling_delta = order_effect["rolling"]["delta_reverse_minus_forward"]
        rolling_metric_checks = {
            key: {
                "absolute_delta": abs(float(rolling_delta[key])),
                "threshold": float(limit),
                "finite": math.isfinite(float(rolling_delta[key])),
                "triggered": (
                    math.isfinite(float(rolling_delta[key]))
                    and abs(float(rolling_delta[key])) > float(limit)
                ),
            }
            for key, limit in thresholds.items()
        }
        order_checks = {
            "reset_invariant": reset_invariant,
            "reset_all_finite": reset_all_finite,
            "reset_metric_max_abs_delta": reset_metric_max,
            "reset_probability_max_mae": reset_probability_max,
            "rolling_metric_checks": rolling_metric_checks,
        }
        if not reset_invariant:
            audit_valid = False
            triggered.append("reset_order_invariance_failed")
        for key, value in rolling_metric_checks.items():
            if not value["finite"]:
                audit_valid = False
                triggered.append(f"rolling_order:{key}:nonfinite")
                continue
            if value["triggered"]:
                label = f"rolling_order:{key}"
                triggered.append(label)
                metric_triggers.append(label)

    phases_complete = all(
        value is not None
        for value in (
            fixed_summary, pipeline_summary, sweep, reverse_sweep, order_effect
        )
    )
    if reset_grid_invariant is False:
        audit_valid = False
    beta_only_trigger = bool(beta_changed) or (
        grid_order is not None and grid_order["selected_beta_changed"]["rolling"]
    )
    repeat_seed0 = bool(
        phases_complete and audit_valid and beta_only_trigger and not metric_triggers
    )
    complete = bool(phases_complete and audit_valid and not repeat_seed0)
    return {
        "fixed_metric_checks": fixed_checks,
        "pipeline_metric_checks": pipeline_checks,
        "draw_order_checks": order_checks,
        "selected_beta_changed": beta_changed,
        "beta_grid_order_effect": grid_order,
        "reset_beta_grid_invariant": reset_grid_invariant,
        "triggered": triggered,
        "audit_valid": audit_valid,
        "phases_complete": phases_complete,
        "complete_for_final_decision": complete,
        "repeat_seed0_posterior_banks": repeat_seed0,
        "rerun_seeds_0_to_4": (
            None if not phases_complete or not audit_valid or repeat_seed0
            else bool(metric_triggers)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.fixbn_batches < 1:
        raise ValueError("--samples and --fixbn-batches must be positive")
    args.runner = args.runner.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.data_root = (args.data_root or args.repo_root).expanduser().resolve()
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.reference_results is not None:
        args.reference_results = args.reference_results.expanduser().resolve()
    if args.historical_runner is not None:
        args.historical_runner = args.historical_runner.expanduser().resolve()
    map_path = resolve_artifact(args.ckpt_dir, args.map_checkpoint)
    spine_path = resolve_artifact(args.ckpt_dir, args.trl_spine)
    required = [args.runner, map_path, spine_path]
    if args.historical_runner is not None:
        required.append(args.historical_runner)
    if args.reference_results is not None:
        required.append(args.reference_results)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    runner = load_module(args.runner, args.repo_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runner.DEVICE = device

    cfg = runner.CFG()
    cfg.seed = args.seed
    cfg.ckpt_dir = str(args.ckpt_dir)
    cfg.num_workers = args.num_workers
    cfg.trl_val_samples = args.samples
    cfg.trl_fixbn_batches = args.fixbn_batches
    cfg.trl_tube_scales = tuple(args.tube_scales)

    seed_everything(args.seed)
    os.chdir(args.data_root)
    train_aug, _, val_loader, test_loader, ood_loader = runner.get_data(cfg)
    val_targets = runner.get_targets(val_loader)
    test_targets = runner.get_targets(test_loader)

    map_model = runner.ResNetCIFAR(cfg.num_classes, use_dropout=False).to(device)
    map_model.load_state_dict(load_state(map_path), strict=True)
    map_model.eval()

    spine_payload = load_payload(spine_path, mmap=True)
    if not isinstance(spine_payload, Mapping) or not isinstance(spine_payload.get("spine"), list):
        raise TypeError("TRL spine payload must contain a list field named 'spine'")
    spine = spine_payload["spine"]
    for index, point in enumerate(spine):
        missing = {"theta", "N", "inv_sqrt_prec"} - set(point)
        if missing:
            raise ValueError(f"Spine point {index} missing fields: {sorted(missing)}")
    saved_cfg = spine_payload.get("cfg", {})
    if isinstance(saved_cfg, Mapping) and "seed" in saved_cfg and int(saved_cfg["seed"]) != args.seed:
        raise ValueError(f"Spine seed {saved_cfg['seed']} does not match requested seed {args.seed}")
    lineage_fields = {
        "batch_size": int(cfg.batch_size),
        "num_classes": int(cfg.num_classes),
    }
    for field, expected in lineage_fields.items():
        if isinstance(saved_cfg, Mapping) and field in saved_cfg:
            observed = int(saved_cfg[field])
            if observed != expected:
                raise ValueError(
                    f"Spine {field} {observed} does not match requested {field} {expected}"
                )

    map_theta = runner.parameters_to_vector(
        [parameter for parameter in map_model.parameters() if parameter.requires_grad]
    ).detach().cpu()
    spine_map_theta = spine[0]["theta"].detach().cpu()
    if map_theta.shape != spine_map_theta.shape:
        raise ValueError(
            f"MAP/spine theta shape mismatch: {map_theta.shape} != {spine_map_theta.shape}"
        )
    map_spine_max_abs_diff = float((map_theta - spine_map_theta).abs().max().item())
    map_spine_theta_equal = bool(torch.equal(map_theta, spine_map_theta))
    if not map_spine_theta_equal:
        raise ValueError(
            "The selected MAP checkpoint is not the MAP endpoint stored at spine[0] "
            f"(max_abs_diff={map_spine_max_abs_diff:.6e})"
        )
    del map_theta, spine_map_theta

    draw_bank = build_draw_bank(spine, args.samples, args.posterior_seed)
    forward_order = [draw.draw_id for draw in draw_bank]
    reverse_order = list(reversed(forward_order))
    phases = set(args.phases)
    if args.no_reverse_beta_sweep:
        phases.discard("sweep-reverse")

    fixed_forward_raw = None
    fixed_reverse_raw = None
    fixed_forward = None
    fixed_reverse = None
    if "fixed-forward" in phases:
        fixed_forward_raw = evaluate_paired(
            runner=runner,
            base_model=map_model,
            spine=spine,
            beta=args.fixed_beta,
            draw_bank=draw_bank,
            draw_order=forward_order,
            calibration_loader=train_aug,
            eval_loaders={"id": test_loader, "ood": ood_loader},
            fixbn_batches=args.fixbn_batches,
            calibration_seed_base=args.calibration_seed_base,
            device=device,
            include_reset_ema=not args.no_reset_ema,
        )
        fixed_forward_raw.pop("models", None)
        fixed_forward = summarize_fixed(
            runner, fixed_forward_raw, test_targets, cfg.num_classes
        )
        runner.cleanup()
    if "fixed-reverse" in phases:
        fixed_reverse_raw = evaluate_paired(
            runner=runner,
            base_model=map_model,
            spine=spine,
            beta=args.fixed_beta,
            draw_bank=draw_bank,
            draw_order=reverse_order,
            calibration_loader=train_aug,
            eval_loaders={"id": test_loader, "ood": ood_loader},
            fixbn_batches=args.fixbn_batches,
            calibration_seed_base=args.calibration_seed_base,
            device=device,
            include_reset_ema=not args.no_reset_ema,
        )
        fixed_reverse_raw.pop("models", None)
        fixed_reverse = summarize_fixed(
            runner, fixed_reverse_raw, test_targets, cfg.num_classes
        )
        runner.cleanup()
    order_effect = None
    if fixed_forward_raw is not None and fixed_reverse_raw is not None:
        order_effect = summarize_order_effect(
            runner, fixed_forward_raw, fixed_reverse_raw, test_targets, cfg.num_classes
        )

    sweep = None
    sweep_models = None
    if "sweep-forward" in phases or "pipeline-forward" in phases:
        sweep = run_sweep(
            runner=runner,
            base_model=map_model,
            spine=spine,
            draw_bank=draw_bank,
            calibration_loader=train_aug,
            val_loader=val_loader,
            targets=val_targets,
            tube_scales=args.tube_scales,
            fixbn_batches=args.fixbn_batches,
            calibration_seed_base=args.calibration_seed_base,
            device=device,
            classes=cfg.num_classes,
            include_reset_ema=not args.no_reset_ema,
        )
        sweep_models = sweep.pop("_models")
        runner.cleanup()
    reverse_sweep = None
    if "sweep-reverse" in phases:
        reverse_sweep = run_sweep(
            runner=runner,
            base_model=map_model,
            spine=spine,
            draw_bank=draw_bank,
            calibration_loader=train_aug,
            val_loader=val_loader,
            targets=val_targets,
            tube_scales=list(reversed(args.tube_scales)),
            fixbn_batches=args.fixbn_batches,
            calibration_seed_base=args.calibration_seed_base,
            device=device,
            classes=cfg.num_classes,
            include_reset_ema=not args.no_reset_ema,
        )
        reverse_sweep.pop("_models", None)
        runner.cleanup()

    pipeline_selected = None
    if "pipeline-forward" in phases:
        if sweep is None or sweep_models is None:
            raise RuntimeError("pipeline-forward requires the canonical validation sweep")
        final_draw_bank = build_draw_bank(
            spine, args.samples, args.final_posterior_seed
        )
        final_order = [draw.draw_id for draw in final_draw_bank]
        beta_by_arm = {
            arm: float(selection["beta"])
            for arm, selection in sweep["selected"].items()
        }
        pipeline_raw = evaluate_paired(
            runner=runner,
            base_model=map_model,
            spine=spine,
            beta=beta_by_arm,
            draw_bank=final_draw_bank,
            draw_order=final_order,
            calibration_loader=train_aug,
            eval_loaders={"id": test_loader, "ood": ood_loader},
            fixbn_batches=args.fixbn_batches,
            calibration_seed_base=args.final_calibration_seed_base,
            device=device,
            include_reset_ema=not args.no_reset_ema,
            arm_models=sweep_models,
        )
        pipeline_raw.pop("models", None)
        pipeline_selected = summarize_fixed(
            runner, pipeline_raw, test_targets, cfg.num_classes
        )
        pipeline_selected["selected_beta_by_arm"] = beta_by_arm
        pipeline_selected["posterior_seed"] = int(args.final_posterior_seed)
        pipeline_selected["calibration_seed_base"] = int(
            args.final_calibration_seed_base
        )
        runner.cleanup()

    decision = threshold_decision(
        fixed_forward, pipeline_selected, sweep, reverse_sweep, order_effect, args
    )
    published_reference = find_reference_trl(args.reference_results, args.seed)

    result = {
        "experiment": "trl_fixbn_rolling_vs_independent_reset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": runner.torchvision.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": int(args.seed),
        "posterior_seed": int(args.posterior_seed),
        "calibration_seed_base": int(args.calibration_seed_base),
        "final_posterior_seed": int(args.final_posterior_seed),
        "final_calibration_seed_base": int(args.final_calibration_seed_base),
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "data_root": str(args.data_root),
        "samples": int(args.samples),
        "fixbn_batches": int(args.fixbn_batches),
        "fixed_beta": float(args.fixed_beta),
        "reset_ema_enabled": not args.no_reset_ema,
        "tube_scales": [float(value) for value in args.tube_scales],
        "requested_phases": list(args.phases),
        "completed_phases": [
            phase
            for phase in PHASES
            if phase in phases
            or (phase == "sweep-forward" and "pipeline-forward" in phases)
        ],
        "pairing": (
            "identical spine anchors, transverse z vectors, and augmented FixBN batches "
            "for rolling/reset; ID and OOD are evaluated on the same sampled network"
        ),
        "initial_bn_state": "MAP checkpoint buffers for each fixed-beta/order run",
        "scope_note": (
            "Controlled FixBN-semantics audit; it cannot replay HVP-updated BN buffers "
            "because those buffers are not stored in the spine artifact. Its paired "
            "posterior bank is generated on CPU for device-independent arm matching; "
            "the historical runner generated z on CUDA."
        ),
        "posterior_bank_definition": (
            "NumPy RandomState anchors plus torch.Generator(device='cpu') Gaussian z; "
            "complete (anchor, z, augmented-batch seed) records are shared by arms"
        ),
        "rolling_definition": "no BN reset; original momentum and buffers carry across draws",
        "reset_definition": (
            "reset_running_stats per draw; momentum=None cumulative average; momentum restored"
        ),
        "reset_ema_definition": (
            "reset_running_stats per draw; original EMA momentum retained; isolates carry-over "
            "from cumulative-versus-EMA averaging"
            if not args.no_reset_ema else None
        ),
        "audit_runner": artifact_identity(args.runner, include_sha256=True),
        "diagnostic_script": artifact_identity(THIS, include_sha256=True),
        "source_commit": args.source_commit,
        "historical_runner_lineage": (
            artifact_identity(args.historical_runner, include_sha256=True)
            if args.historical_runner is not None else None
        ),
        "map_checkpoint": artifact_identity(map_path, include_sha256=True),
        "map_spine_theta_identity": {
            "exactly_equal": map_spine_theta_equal,
            "max_abs_diff": map_spine_max_abs_diff,
        },
        "trl_spine": artifact_identity(
            spine_path, include_sha256=args.hash_large_artifacts
        ),
        "saved_spine_cfg": dict(saved_cfg) if isinstance(saved_cfg, Mapping) else saved_cfg,
        "saved_best_tube_scale": spine_payload.get("best_tube_scale"),
        "published_reference": published_reference,
        "reference_results_artifact": (
            artifact_identity(args.reference_results, include_sha256=True)
            if args.reference_results is not None else None
        ),
        "contextual_reference_comparison": contextual_reference_comparison(
            published_reference, fixed_forward, pipeline_selected
        ),
        "fixed_beta_forward": fixed_forward,
        "fixed_beta_reverse": fixed_reverse,
        "draw_order_effect": order_effect,
        "validation_beta_sweep": sweep,
        "validation_beta_sweep_reverse_order": reverse_sweep,
        "post_sweep_selected_beta": pipeline_selected,
        "escalation_decision": decision,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
