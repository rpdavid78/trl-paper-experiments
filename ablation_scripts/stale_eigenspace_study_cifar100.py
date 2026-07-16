#!/usr/bin/env python
"""Study whether transported transverse structure drifts along the TRL spine.

This script directly targets the reviewer concern about stale eigenvalues / stale
transverse curvature. It builds a TRL spine, then at selected stored checkpoints
it recomputes a fresh local top-K Hessian eigenspace and compares it against the
transported basis stored by TRL.

Outputs JSONL rows with:
  - mean/max principal angle between transported and fresh eigenspaces
  - mean singular value / subspace overlap
  - relative top-eigenvalue drift compared with the MAP spectrum

Run example:
  python ablation_scripts/stale_eigenspace_study_cifar100.py \
    --seed 0 --k 30 --steps 40 --fractions 0 0.25 0.5 0.75 1.0 \
    --results results_iclr/stale_eigenspace_cifar100.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse.linalg as sla
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# Import the instrumented CIFAR-100 script.
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
for path in (ROOT / "scripts", ROOT, THIS.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trl_iclr_utils.experiment_io import StageTimer, append_jsonl  # noqa: E402
from cifar100_all_methods_iclr import (  # noqa: E402
    CFG,
    DEVICE,
    PracticalTRLStage2,
    build_trl_prior_from_laplace,
    get_data,
    get_hvp_function_ablation,
    laplace_fit_and_predict,
    load_or_train_map,
    set_seed,
)


def eigenspace_at(model, theta: torch.Tensor, loader, k: int, hvp_batches: int):
    vector_to_parameters(theta.to(DEVICE), model.parameters())
    model.to(DEVICE)
    hvp_fn = get_hvp_function_ablation(model, loader, DEVICE, num_batches=hvp_batches)
    n = theta.numel()
    op = sla.LinearOperator((n, n), matvec=hvp_fn)
    vals, vecs = sla.eigsh(op, k=k, which="LA")
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    # torch QR for numerical cleanup
    Q = torch.from_numpy(vecs.copy()).float().to(DEVICE)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    return vals, Q.detach().cpu()


def subspace_stats(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    """Subspace overlap and principal-angle statistics.

    A and B are expected to have orthonormal columns.  The singular values of
    A.T @ B are cosines of the principal angles between span(A) and span(B).
    The normalized overlap ||A.T @ B||_F^2 / k equals mean(s_i^2).
    """
    A = A.float()
    B = B.float()
    s = torch.linalg.svdvals(A.T @ B).clamp(0, 1).cpu().numpy()
    angles = np.degrees(np.arccos(s))
    return {
        "mean_cosine": float(np.mean(s)),
        "min_cosine": float(np.min(s)),
        "subspace_overlap": float(np.mean(s ** 2)),
        "mean_principal_angle_deg": float(np.mean(angles)),
        "max_principal_angle_deg": float(np.max(angles)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=str, default="results_iclr/stale_eigenspace_cifar100.jsonl")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--hvp-batches", type=int, default=5)
    p.add_argument("--fractions", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    cfg.seed = args.seed
    cfg.trl_k_perp = args.k
    cfg.trl_steps = args.steps
    cfg.trl_hvp_batches = args.hvp_batches
    cfg.ckpt_dir = args.ckpt_dir or f"./checkpoints_c100_seed{args.seed}"
    if args.quick:
        cfg.epochs_map = 1
        cfg.trl_k_perp = min(cfg.trl_k_perp, 3)
        cfg.trl_steps = min(cfg.trl_steps, 3)
        cfg.trl_hvp_batches = 1
        args.fractions = [0, 1]

    set_seed(cfg.seed)
    timings = {}
    with StageTimer("data", timings):
        tr_aug, bn_clean, val, ts, ood = get_data(cfg)
    with StageTimer("map", timings):
        model_map = load_or_train_map(tr_aug, cfg)
    with StageTimer("laplace_for_prior", timings):
        _, base_val, *_ = laplace_fit_and_predict(model_map, bn_clean, ts, ood, cfg)
    prior_vec = build_trl_prior_from_laplace(base_val, model_map)

    trl = PracticalTRLStage2(
        map_model=model_map,
        prior_vec=prior_vec,
        clean_loader=bn_clean,
        steps=cfg.trl_steps,
        k_perp=cfg.trl_k_perp,
        step_size=cfg.trl_step_size,
        eta=cfg.trl_eta,
        tube_scale=4.0,
        max_delta_norm=cfg.trl_max_delta_norm,
        hvp_batches=cfg.trl_hvp_batches,
        store_every=1,
    )
    with StageTimer("trl_build", timings):
        trl.build()

    # Fresh MAP eigenspace for eigenvalue drift reference.
    model_tmp = model_map.__class__(num_classes=cfg.num_classes).to(DEVICE)
    model_tmp.load_state_dict(model_map.state_dict())
    map_theta = parameters_to_vector([p for p in model_tmp.parameters() if p.requires_grad]).detach().cpu()
    with StageTimer("fresh_map_eigenspace", timings):
        vals0, Q0 = eigenspace_at(model_tmp, map_theta, bn_clean, cfg.trl_k_perp, cfg.trl_hvp_batches)

    n_spine = len(trl.spine)
    chosen = sorted(set(max(0, min(n_spine - 1, int(round(f * (n_spine - 1))))) for f in args.fractions))

    for idx in chosen:
        pt = trl.spine[idx]
        theta_t = pt["theta"]
        N_trans = pt["N"].cpu()
        with StageTimer(f"fresh_eigenspace_idx_{idx}", timings):
            vals_t, Q_t = eigenspace_at(model_tmp, theta_t, bn_clean, cfg.trl_k_perp, cfg.trl_hvp_batches)
        stats_trans_vs_fresh = subspace_stats(N_trans, Q_t)
        stats_map_vs_fresh = subspace_stats(Q0, Q_t)
        eps = 1e-12
        rel_drift = np.abs(vals_t - vals0) / (np.abs(vals0) + eps)
        row = {
            "dataset": "CIFAR-100",
            "architecture": "ResNet-18-CIFAR",
            "study": "stale_eigenspace",
            "seed": cfg.seed,
            "k": cfg.trl_k_perp,
            "steps": cfg.trl_steps,
            "spine_index": int(idx),
            "spine_fraction": float(idx / max(1, n_spine - 1)),
            "transport_vs_fresh_mean_cosine": stats_trans_vs_fresh["mean_cosine"],
            "transport_vs_fresh_min_cosine": stats_trans_vs_fresh["min_cosine"],
            "transport_vs_fresh_mean_angle_deg": stats_trans_vs_fresh["mean_principal_angle_deg"],
            "transport_vs_fresh_max_angle_deg": stats_trans_vs_fresh["max_principal_angle_deg"],
            "transport_vs_fresh_subspace_overlap": stats_trans_vs_fresh["subspace_overlap"],
            "map_vs_fresh_mean_cosine": stats_map_vs_fresh["mean_cosine"],
            "map_vs_fresh_mean_angle_deg": stats_map_vs_fresh["mean_principal_angle_deg"],
            "map_vs_fresh_subspace_overlap": stats_map_vs_fresh["subspace_overlap"],
            "eigenvalue_relative_drift_mean": float(np.mean(rel_drift)),
            "eigenvalue_relative_drift_max": float(np.max(rel_drift)),
        }
        append_jsonl(args.results, row)
        print(row)

    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
