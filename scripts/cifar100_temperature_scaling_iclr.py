#!/usr/bin/env python
"""Scalar temperature-scaling baseline for CIFAR-100 and CIFAR-100-C.

This script fits one scalar temperature on the same held-out clean validation
split used by the TRL pipeline, using validation NLL. It then evaluates MAP+TS
on the clean CIFAR-100 test set and, optionally, CIFAR-100-C corruptions.

Temperature scaling preserves the MAP argmax predictions/ranking; it is a
calibration-only baseline and does not produce posterior samples or functional
variance diagnostics.

Examples:
    python scripts/cifar100_temperature_scaling_iclr.py --seed 0 \
        --results results_iclr/cifar100_temperature_scaling.jsonl

    python scripts/cifar100_temperature_scaling_iclr.py --seed 0 \
        --cifar100c-root /path/to/CIFAR-100-C \
        --results results_iclr/cifar100_temperature_scaling_with_c.jsonl

Quick smoke test:
    python scripts/cifar100_temperature_scaling_iclr.py --seed 0 --quick
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))

from trl_iclr_utils.experiment_io import StageTimer, append_jsonl, flatten_timings  # noqa: E402
from cifar100_all_methods_iclr import (  # noqa: E402
    CFG,
    DEVICE,
    auroc_entropy,
    calc_metrics,
    get_data,
    get_targets,
    load_or_train_map,
    set_seed,
)


def collect_logits(model: torch.nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(DEVICE)).detach().cpu())
            targets.append(y.detach().cpu())
    return torch.cat(logits, dim=0), torch.cat(targets, dim=0).long()


def probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / float(temperature), dim=1)


def fit_temperature(
    logits_val: torch.Tensor,
    targets_val: torch.Tensor,
    *,
    init_temperature: float = 1.0,
    max_iter: int = 50,
) -> tuple[float, float]:
    """Fit scalar temperature by minimizing validation NLL."""
    logits = logits_val.to(DEVICE)
    targets = targets_val.to(DEVICE)
    log_temperature = torch.nn.Parameter(torch.tensor(float(init_temperature), device=DEVICE).log())
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp()
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().cpu().item())
    with torch.no_grad():
        val_nll = float(F.cross_entropy(logits / temperature, targets).detach().cpu().item())
    return temperature, val_nll


def make_row(
    *,
    dataset: str,
    method: str,
    seed: int,
    probs: torch.Tensor,
    targets: torch.Tensor,
    cfg: CFG,
    temperature: float,
    selected_val_nll: float,
    selected_val_metrics: Dict[str, float],
    timings: Dict[str, Dict[str, float]],
    probs_ood: Optional[torch.Tensor] = None,
    corruption: Optional[str] = None,
    severity: Optional[int] = None,
):
    acc, nll, ece, brier = calc_metrics(probs, targets, cfg.num_classes)
    auroc = auroc_entropy(probs, probs_ood) if probs_ood is not None else float("nan")
    row = {
        "dataset": dataset,
        "architecture": "ResNet-18-CIFAR",
        "method": method,
        "seed": int(seed),
        "acc": float(acc),
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
        "auroc": float(auroc),
        "temperature": float(temperature),
        "temperature_selection": "validation_nll",
        "selection_val_nll": float(selected_val_nll),
        "selection_val_acc": float(selected_val_metrics["acc"]),
        "selection_val_ece": float(selected_val_metrics["ece"]),
        "selection_val_brier": float(selected_val_metrics["brier"]),
    }
    if corruption is not None:
        row["corruption"] = corruption
    if severity is not None:
        row["severity"] = int(severity)
    row.update(flatten_timings("time", timings))
    return row


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-100 scalar temperature scaling baseline.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=str, default="results_iclr/cifar100_temperature_scaling.jsonl")
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--init-temperature", type=float, default=1.0)
    p.add_argument("--cifar100c-root", type=str, default=None)
    p.add_argument("--corruptions", nargs="*", default=None)
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--quick", action="store_true", help="Smoke-test mode; not for paper tables.")
    return p.parse_args()


def cfg_from_args(args) -> CFG:
    cfg = CFG()
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.ckpt_dir = args.ckpt_dir or f"./checkpoints_c100_seed{args.seed}"
    if args.quick:
        cfg.epochs_map = 1
    return cfg


def main():
    args = parse_args()
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, Dict[str, float]] = {}
    with StageTimer("data", timings):
        tr_aug, _bn_clean, val_loader, test_loader, ood_loader = get_data(cfg)
    with StageTimer("map_train_or_load", timings):
        model_map = load_or_train_map(tr_aug, cfg)

    with StageTimer("collect_val_logits", timings):
        logits_val, targets_val = collect_logits(model_map, val_loader)
    with StageTimer("fit_temperature", timings):
        temperature, selected_val_nll = fit_temperature(
            logits_val,
            targets_val,
            init_temperature=args.init_temperature,
            max_iter=args.max_iter,
        )

    p_val = probs_from_logits(logits_val, temperature)
    val_acc, val_nll_check, val_ece, val_brier = calc_metrics(p_val, targets_val, cfg.num_classes)
    selected_val_metrics = {
        "acc": float(val_acc),
        "nll": float(val_nll_check),
        "ece": float(val_ece),
        "brier": float(val_brier),
    }

    with StageTimer("collect_test_logits", timings):
        logits_test, targets_test = collect_logits(model_map, test_loader)
    with StageTimer("collect_ood_logits", timings):
        logits_ood, _targets_ood = collect_logits(model_map, ood_loader)

    p_test = probs_from_logits(logits_test, temperature)
    p_ood = probs_from_logits(logits_ood, temperature)
    row = make_row(
        dataset="CIFAR-100",
        method="MAP+TS",
        seed=cfg.seed,
        probs=p_test,
        targets=targets_test,
        cfg=cfg,
        temperature=temperature,
        selected_val_nll=selected_val_nll,
        selected_val_metrics=selected_val_metrics,
        timings=timings,
        probs_ood=p_ood,
    )
    append_jsonl(args.results, row)
    print(f"MAP+TS clean acc={100*row['acc']:.2f} nll={row['nll']:.4f} ece={row['ece']:.4f} "
          f"brier={row['brier']:.4f} T={temperature:.4f} val_nll={selected_val_nll:.4f}")

    if args.cifar100c_root:
        from cifar100c_eval_iclr import CIFAR100CDataset  # noqa: E402

        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        t_clean = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        c_root = Path(args.cifar100c_root)
        corruptions = args.corruptions
        if corruptions is None or len(corruptions) == 0:
            corruptions = sorted([p.stem for p in c_root.glob("*.npy") if p.name != "labels.npy"])

        for corr in corruptions:
            for sev in args.severities:
                ds = CIFAR100CDataset(str(c_root), corr, sev, transform=t_clean)
                loader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                )
                with StageTimer(f"collect_c100c_logits_{corr}_{sev}", timings):
                    logits_c, targets_c = collect_logits(model_map, loader)
                probs_c = probs_from_logits(logits_c, temperature)
                row_c = make_row(
                    dataset="CIFAR-100-C",
                    method="MAP+TS",
                    seed=cfg.seed,
                    probs=probs_c,
                    targets=targets_c,
                    cfg=cfg,
                    temperature=temperature,
                    selected_val_nll=selected_val_nll,
                    selected_val_metrics=selected_val_metrics,
                    timings=timings,
                    corruption=corr,
                    severity=sev,
                )
                append_jsonl(args.results, row_c)
                print(f"MAP+TS {corr:20s} sev={sev} acc={100*row_c['acc']:.2f} "
                      f"nll={row_c['nll']:.4f} ece={row_c['ece']:.4f} brier={row_c['brier']:.4f}")

    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
