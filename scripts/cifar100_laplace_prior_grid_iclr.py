#!/usr/bin/env python
"""CIFAR-100 last-layer Laplace prior-selection sanity check.

This script reruns the ELA/LLA last-layer Laplace baselines under two prior
selection rules:

1. Standard empirical-Bayes marginal-likelihood optimization, matching the
   original paper pipeline.
2. Validation-NLL grid search, using the same held-out clean validation split
   used for TRL hyperparameter selection.

The goal is to make the Laplace baseline comparison explicit: if validation
NLL prior tuning does not rescue ELA/LLA, the calibration gap is not an artifact
of giving TRL a validation-selected hyperparameter while leaving Laplace tuned
only by evidence.

Example:
    python scripts/cifar100_laplace_prior_grid_iclr.py --seed 0 \
        --results results_iclr/cifar100_laplace_prior_grid.jsonl

Quick smoke test:
    python scripts/cifar100_laplace_prior_grid_iclr.py --seed 0 --quick
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from laplace import Laplace

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))

from trl_iclr_utils.experiment_io import StageTimer, append_jsonl, flatten_timings  # noqa: E402
from cifar100_all_methods_iclr import (  # noqa: E402
    CFG,
    DEVICE,
    ResNetCIFAR,
    auroc_entropy,
    calc_metrics,
    cleanup,
    get_data,
    get_targets,
    load_or_train_map,
    set_seed,
)


def build_laplace_fit_loader(bn_loader_clean, cfg: CFG):
    """Return the same clean-train subset used by the main Laplace pipeline."""
    n = len(bn_loader_clean.dataset)
    subset_size = min(int(cfg.laplace_subset), n)
    gen = torch.Generator().manual_seed(cfg.seed)
    subset_idx = torch.randperm(n, generator=gen)[:subset_size]
    sub_tr = torch.utils.data.Subset(bn_loader_clean.dataset, subset_idx)
    return torch.utils.data.DataLoader(
        sub_tr,
        batch_size=cfg.laplace_fit_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
    )


def prior_precision_summary(la: Laplace) -> Dict[str, float]:
    pp = la.prior_precision.detach().float().cpu()
    return {
        "prior_precision_mean": float(pp.mean().item()),
        "prior_precision_min": float(pp.min().item()),
        "prior_precision_max": float(pp.max().item()),
    }


def fit_laplace(model_map, bn_loader_clean, cfg: CFG, timings: Optional[Dict[str, Dict[str, float]]] = None) -> Laplace:
    """Fit a last-layer KRON Laplace object, without optimizing prior precision."""
    fit_loader = build_laplace_fit_loader(bn_loader_clean, cfg)
    model = copy.deepcopy(model_map).to(DEVICE)
    model.eval()
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    if timings is None:
        la.fit(fit_loader)
    else:
        with StageTimer("laplace_fit", timings):
            la.fit(fit_loader)
    return la


def optimize_prior(la: Laplace, rule: str, val_loader=None, pred_type: Optional[str] = None,
                   timings: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    """Optimize prior precision using either marglik or validation grid search."""
    if rule == "marglik":
        if timings is None:
            la.optimize_prior_precision(method="marglik")
        else:
            with StageTimer("laplace_prior_marglik", timings):
                la.optimize_prior_precision(method="marglik")
        return

    if rule != "gridsearch":
        raise ValueError(f"Unsupported prior selection rule: {rule}")
    if val_loader is None:
        raise ValueError("gridsearch requires val_loader")

    kwargs = {"method": "gridsearch", "val_loader": val_loader}
    if pred_type is not None:
        kwargs["pred_type"] = pred_type
        if pred_type == "nn":
            kwargs["link_approx"] = "mc"
        elif pred_type == "glm":
            kwargs["link_approx"] = "probit"

    def _run_gridsearch():
        try:
            la.optimize_prior_precision(**kwargs)
        except TypeError:
            # Older laplace-torch versions may not accept pred_type in this call.
            # Fall back to the default grid-search predictive rule rather than
            # failing after the expensive curvature fit.
            kwargs_no_pred = {k: v for k, v in kwargs.items() if k != "pred_type"}
            la.optimize_prior_precision(**kwargs_no_pred)

    if timings is None:
        _run_gridsearch()
    else:
        stage = f"laplace_prior_gridsearch_{pred_type or 'default'}"
        with StageTimer(stage, timings):
            _run_gridsearch()


def laplace_predict_loader(la: Laplace, loader, pred_type: str, n_samples: int = 25):
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            if pred_type == "nn":
                try:
                    p = la(x, pred_type="nn", link_approx="mc", n_samples=n_samples)
                except TypeError:
                    p = la(x, pred_type="nn", n_samples=n_samples)
            elif pred_type == "glm":
                try:
                    p = la(x, pred_type="glm", link_approx="probit")
                except TypeError:
                    p = la(x, pred_type="glm")
            else:
                raise ValueError(f"Unsupported pred_type: {pred_type}")
            outs.append(p.detach().cpu())
    return torch.cat(outs, dim=0)


def eval_laplace_variant(
    la: Laplace,
    *,
    pred_type: str,
    label: str,
    seed: int,
    val_loader,
    test_loader,
    ood_loader,
    cfg: CFG,
    prior_rule: str,
    prior_rule_pred_type: Optional[str],
    timings: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    val_targets = get_targets(val_loader)
    test_targets = get_targets(test_loader)

    with StageTimer(f"predict_val_{label}", timings):
        p_val = laplace_predict_loader(la, val_loader, pred_type, cfg.laplace_n_samples_ela)
    with StageTimer(f"predict_test_{label}", timings):
        p_test = laplace_predict_loader(la, test_loader, pred_type, cfg.laplace_n_samples_ela)
    with StageTimer(f"predict_ood_{label}", timings):
        p_ood = laplace_predict_loader(la, ood_loader, pred_type, cfg.laplace_n_samples_ela)

    val_acc, val_nll, val_ece, val_brier = calc_metrics(p_val, val_targets, cfg.num_classes)
    acc, nll, ece, brier = calc_metrics(p_test, test_targets, cfg.num_classes)
    auroc = auroc_entropy(p_test, p_ood)

    row = {
        "dataset": "CIFAR-100",
        "architecture": "ResNet-18-CIFAR",
        "method": label,
        "base_method": "ELA" if pred_type == "nn" else "LLA",
        "seed": int(seed),
        "acc": float(acc),
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
        "auroc": float(auroc),
        "prior_selection": prior_rule,
        "prior_selection_pred_type": prior_rule_pred_type or "none",
        "laplace_pred_type": pred_type,
        "selection_val_acc": float(val_acc),
        "selection_val_nll": float(val_nll),
        "selection_val_ece": float(val_ece),
        "selection_val_brier": float(val_brier),
        "laplace_subset": int(cfg.laplace_subset),
        "laplace_fit_bs": int(cfg.laplace_fit_bs),
        "laplace_n_samples_ela": int(cfg.laplace_n_samples_ela),
    }
    row.update(prior_precision_summary(la))
    row.update(flatten_timings("time", timings))
    return row


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-100 ELA/LLA prior-selection sanity check.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=str, default="results_iclr/cifar100_laplace_prior_grid.jsonl")
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--laplace-subset", type=int, default=None)
    p.add_argument("--laplace-fit-bs", type=int, default=None)
    p.add_argument("--laplace-samples", type=int, default=None)
    p.add_argument("--rules", nargs="+", default=["marglik", "gridsearch"], choices=["marglik", "gridsearch"])
    p.add_argument("--grid-pred-types", nargs="+", default=["glm", "nn"], choices=["glm", "nn"],
                   help="Predictive rule(s) used by laplace-torch validation grid search.")
    p.add_argument("--quick", action="store_true", help="Smoke-test mode; not for paper tables.")
    return p.parse_args()


def cfg_from_args(args) -> CFG:
    cfg = CFG()
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.ckpt_dir = args.ckpt_dir or f"./checkpoints_c100_seed{args.seed}"
    if args.laplace_subset is not None:
        cfg.laplace_subset = args.laplace_subset
    if args.laplace_fit_bs is not None:
        cfg.laplace_fit_bs = args.laplace_fit_bs
    if args.laplace_samples is not None:
        cfg.laplace_n_samples_ela = args.laplace_samples
    if args.quick:
        cfg.epochs_map = 1
        cfg.laplace_subset = min(cfg.laplace_subset, 256)
        cfg.laplace_fit_bs = min(cfg.laplace_fit_bs, 64)
        cfg.laplace_n_samples_ela = min(cfg.laplace_n_samples_ela, 3)
    return cfg


def main():
    args = parse_args()
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)

    timings_global: Dict[str, Dict[str, float]] = {}
    with StageTimer("data", timings_global):
        tr_aug, bn_clean, val_loader, test_loader, ood_loader = get_data(cfg)
    with StageTimer("map_train_or_load", timings_global):
        model_map = load_or_train_map(tr_aug, cfg)

    rows: List[Dict[str, float]] = []

    if "marglik" in args.rules:
        timings = dict(timings_global)
        print("\n>>> Fitting ELA/LLA with marginal-likelihood prior optimization")
        la = fit_laplace(model_map, bn_clean, cfg, timings=timings)
        optimize_prior(la, "marglik", timings=timings)
        for pred_type, label in [("nn", "ELA-marglik"), ("glm", "LLA-marglik")]:
            row = eval_laplace_variant(
                la,
                pred_type=pred_type,
                label=label,
                seed=cfg.seed,
                val_loader=val_loader,
                test_loader=test_loader,
                ood_loader=ood_loader,
                cfg=cfg,
                prior_rule="marglik",
                prior_rule_pred_type=None,
                timings=dict(timings),
            )
            append_jsonl(args.results, row)
            rows.append(row)
            print(f"{label:12s} acc={100*row['acc']:.2f} nll={row['nll']:.4f} ece={row['ece']:.4f} "
                  f"brier={row['brier']:.4f} val_nll={row['selection_val_nll']:.4f} "
                  f"prior_mean={row['prior_precision_mean']:.4g}")
        cleanup()

    if "gridsearch" in args.rules:
        for grid_pred_type in args.grid_pred_types:
            timings = dict(timings_global)
            print(f"\n>>> Fitting Laplace with validation-NLL grid search, pred_type={grid_pred_type}")
            la = fit_laplace(model_map, bn_clean, cfg, timings=timings)
            optimize_prior(la, "gridsearch", val_loader=val_loader, pred_type=grid_pred_type, timings=timings)

            # Evaluate the grid-selected object using the corresponding predictive rule.
            # This gives ELA-grid when pred_type=nn and LLA-grid when pred_type=glm.
            label = "ELA-grid" if grid_pred_type == "nn" else "LLA-grid"
            row = eval_laplace_variant(
                la,
                pred_type=grid_pred_type,
                label=label,
                seed=cfg.seed,
                val_loader=val_loader,
                test_loader=test_loader,
                ood_loader=ood_loader,
                cfg=cfg,
                prior_rule="val_nll_gridsearch",
                prior_rule_pred_type=grid_pred_type,
                timings=timings,
            )
            append_jsonl(args.results, row)
            rows.append(row)
            print(f"{label:12s} acc={100*row['acc']:.2f} nll={row['nll']:.4f} ece={row['ece']:.4f} "
                  f"brier={row['brier']:.4f} val_nll={row['selection_val_nll']:.4f} "
                  f"prior_mean={row['prior_precision_mean']:.4g}")
            cleanup()

    print(f"\nWrote {len(rows)} rows to {args.results}")


if __name__ == "__main__":
    main()
