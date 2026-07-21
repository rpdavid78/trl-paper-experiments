#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse.linalg as sla
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import parameters_to_vector, vector_to_parameters

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
for p in [THIS.parent, ROOT / "scripts", ROOT, Path.cwd(), Path.cwd() / "scripts", Path.cwd() / "code"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from trl_iclr_utils.experiment_io import append_jsonl  # noqa: E402
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


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clone_spine_light(spine: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    out = []
    for pt in spine:
        out.append({
            "theta": pt["theta"],
            "N": pt["N"],
            "inv_sqrt_prec": pt["inv_sqrt_prec"],
        })
    return out


def fixed_map_basis_spine(spine: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    n0 = spine[0]["N"]
    isp0 = spine[0]["inv_sqrt_prec"]
    return [{"theta": pt["theta"], "N": n0, "inv_sqrt_prec": isp0} for pt in spine]


def single_checkpoint_spine(spine: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    # T_eff = 0: posterior samples only around the MAP endpoint / first stored point.
    pt0 = spine[0]
    return [{"theta": pt0["theta"], "N": pt0["N"], "inv_sqrt_prec": pt0["inv_sqrt_prec"]}]


def compute_metrics(probs: torch.Tensor, targets: torch.Tensor, num_classes: int, n_bins: int = 15) -> Dict[str, float]:
    probs = probs.float().cpu()
    targets = targets.long().cpu()
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    acc = (pred == targets).float().mean().item()
    eps = 1e-12
    nll = -torch.log(probs[torch.arange(len(targets)), targets].clamp_min(eps)).mean().item()
    y_onehot = F.one_hot(targets, num_classes=num_classes).float()
    brier = ((probs - y_onehot) ** 2).sum(dim=1).mean().item()
    ece = 0.0
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if mask.any():
            bin_acc = (pred[mask] == targets[mask]).float().mean().item()
            bin_conf = conf[mask].mean().item()
            ece += mask.float().mean().item() * abs(bin_acc - bin_conf)
    return {"acc": float(acc), "nll": float(nll), "ece": float(ece), "brier": float(brier)}


def compute_auroc_from_msp(id_probs: torch.Tensor, ood_probs: torch.Tensor) -> float:
    id_score = id_probs.max(dim=1).values.cpu().numpy()
    ood_score = ood_probs.max(dim=1).values.cpu().numpy()
    y = np.concatenate([np.ones_like(id_score), np.zeros_like(ood_score)])
    s = np.concatenate([id_score, ood_score])
    return float(roc_auc_score(y, s))


def eigenspace_at(model, theta: torch.Tensor, loader, k: int, hvp_batches: int):
    theta = theta.to(DEVICE)
    vector_to_parameters(theta, model.parameters())
    model.to(DEVICE)
    hvp_fn = get_hvp_function_ablation(model, loader, DEVICE, num_batches=hvp_batches)
    n = int(theta.numel())
    op = sla.LinearOperator((n, n), matvec=hvp_fn)
    vals, vecs = sla.eigsh(op, k=k, which="LA")
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    Q = torch.from_numpy(vecs.copy()).float().to(DEVICE)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    vals_t = torch.maximum(torch.from_numpy(vals.copy()).float().to(DEVICE), torch.tensor(0.0, device=DEVICE))
    return vals_t.detach(), Q.detach()


def make_fresh_refresh_spine(
    spine_transport: List[Dict[str, torch.Tensor]],
    model_map,
    prior_vec: torch.Tensor,
    loader,
    k: int,
    hvp_batches: int,
    cache_path: Path,
    force: bool,
    max_points: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    if cache_path.exists() and not force:
        print(f">>> Loading cached fresh-refresh spine: {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu", mmap=True)
        return payload["spine"]

    model = copy.deepcopy(model_map).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    n_total = len(spine_transport)
    if max_points and max_points > 0 and max_points < n_total:
        idxs = sorted(set(int(round(i * (n_total - 1) / (max_points - 1))) for i in range(max_points)))
        print(f">>> Fresh-refresh diagnostic uses subset of spine points: {idxs}", flush=True)
    else:
        idxs = list(range(n_total))

    fresh_by_idx = {}
    t0_all = time.perf_counter()
    for c, idx in enumerate(idxs, 1):
        pt = spine_transport[idx]
        theta = pt["theta"].float().cpu()
        print(f">>> Fresh eigenspace {c}/{len(idxs)} at idx={idx}", flush=True)
        t0 = time.perf_counter()
        evals, Q = eigenspace_at(model, theta, loader, k, hvp_batches)
        prior = prior_vec.to(DEVICE)
        prior_proj = torch.sum((Q ** 2) * prior.unsqueeze(1), dim=0)
        prec = torch.clamp(evals + prior_proj, min=1e-6)
        inv_sqrt_prec = torch.rsqrt(prec)
        fresh_by_idx[idx] = {
            "theta": theta,
            "N": Q.detach().cpu(),
            "inv_sqrt_prec": inv_sqrt_prec.detach().cpu(),
        }
        print(f"    done idx={idx} in {time.perf_counter() - t0:.1f}s", flush=True)
        del evals, Q, prior_proj, prec, inv_sqrt_prec
        cleanup()

    # If only a subset was recomputed, evaluate a reduced longitudinal mixture over those points.
    fresh_spine = [fresh_by_idx[idx] for idx in idxs]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spine": fresh_spine, "source_indices": idxs, "k": k, "hvp_batches": hvp_batches}, cache_path)
    print(f">>> Saved fresh-refresh spine cache to {cache_path}", flush=True)
    print(f">>> Total fresh-refresh construction time: {time.perf_counter() - t0_all:.1f}s", flush=True)
    del model
    cleanup()
    return fresh_spine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--results", type=str, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--tube-scale", type=float, default=4.0)
    p.add_argument("--n-samples", type=int, default=25)
    p.add_argument("--fixbn-batches", type=int, default=25)
    p.add_argument("--fixbn-mode", choices=["rolling", "reset"], default="rolling",
                   help="rolling reproduces the published ablation; reset is independent.")
    p.add_argument("--hvp-batches", type=int, default=5)
    p.add_argument("--eval-seed-offset", type=int, default=12345)
    p.add_argument("--modes", nargs="+", default=["single", "fresh"],
                   choices=["transported", "fixed", "single", "fresh"])
    p.add_argument("--fresh-cache", type=str, default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--fresh-max-points", type=int, default=0,
                   help="0 = recompute all spine points; e.g. 5 for a smoke test.")
    p.add_argument("--include-ood", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    cfg.seed = int(args.seed)
    cfg.ckpt_dir = args.ckpt_dir
    cfg.trl_hvp_batches = int(args.hvp_batches)

    set_seed(cfg.seed)
    print(">>> Loading data...", flush=True)
    tr_aug, bn_clean, val_loader, ts_loader, ood_loader = get_data(cfg)

    print(">>> Loading MAP...", flush=True)
    model_map = load_or_train_map(tr_aug, cfg)

    spine_path = Path(cfg.ckpt_dir) / cfg.trl_spine
    print(">>> Loading transported spine:", spine_path, flush=True)
    payload = torch.load(spine_path, map_location="cpu", mmap=True)
    spine_transport = payload["spine"]
    best_ts = payload.get("best_tube_scale", None)
    if best_ts is not None and args.tube_scale is None:
        args.tube_scale = float(best_ts)

    k = int(spine_transport[0]["N"].shape[1])
    cfg.trl_k_perp = k
    cfg.trl_steps = len(spine_transport)

    params = [p for p in model_map.parameters() if p.requires_grad]
    dummy_prior = torch.zeros_like(parameters_to_vector(params)).to(DEVICE)
    trl = PracticalTRLStage2(
        map_model=model_map,
        prior_vec=dummy_prior,
        clean_loader=bn_clean,
        steps=len(spine_transport),
        k_perp=k,
        step_size=cfg.trl_step_size,
        eta=cfg.trl_eta,
        tube_scale=float(args.tube_scale),
        max_delta_norm=cfg.trl_max_delta_norm,
        hvp_batches=cfg.trl_hvp_batches,
        store_every=1,
    )

    mode_to_spine = {}
    if "transported" in args.modes:
        mode_to_spine["TRL-transported-basis"] = clone_spine_light(spine_transport)
    if "fixed" in args.modes:
        mode_to_spine["TRL-fixed-MAP-basis"] = fixed_map_basis_spine(spine_transport)
    if "single" in args.modes:
        mode_to_spine["TRL-single-checkpoint"] = single_checkpoint_spine(spine_transport)

    if "fresh" in args.modes:
        print(">>> Fitting Laplace only to recover prior precision for fresh-refresh.", flush=True)
        # This mirrors the stale-eigenspace diagnostic and the original TRL build.
        _, base_val, *_ = laplace_fit_and_predict(model_map, bn_clean, ts_loader, ood_loader, cfg)
        prior_vec = build_trl_prior_from_laplace(base_val, model_map)
        fresh_cache = Path(args.fresh_cache) if args.fresh_cache else Path(cfg.ckpt_dir) / "c100_trl_fresh_refresh_spine.pth"
        fresh_spine = make_fresh_refresh_spine(
            spine_transport=spine_transport,
            model_map=model_map,
            prior_vec=prior_vec,
            loader=bn_clean,
            k=k,
            hvp_batches=cfg.trl_hvp_batches,
            cache_path=fresh_cache,
            force=bool(args.force_refresh),
            max_points=int(args.fresh_max_points),
        )
        mode_to_spine["TRL-fresh-refresh-basis"] = fresh_spine

    for method, spine in mode_to_spine.items():
        print(f"\n>>> Evaluating {method} with {len(spine)} spine point(s)...", flush=True)
        set_seed(cfg.seed + args.eval_seed_offset)
        trl.spine = spine
        trl.beta = float(args.tube_scale)
        trl.k = int(spine[0]["N"].shape[1])
        trl.reset_accounting()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs, targets = trl.predict(
            ts_loader, tr_aug, n_samples=args.n_samples,
            fix_bn_batches=args.fixbn_batches, fix_bn_mode=args.fixbn_mode,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        m = compute_metrics(probs, targets, cfg.num_classes)

        auroc = None
        if args.include_ood:
            set_seed(cfg.seed + args.eval_seed_offset)
            ood_probs, _ = trl.predict(
                ood_loader, tr_aug, n_samples=args.n_samples,
                fix_bn_batches=args.fixbn_batches, fix_bn_mode=args.fixbn_mode,
            )
            auroc = compute_auroc_from_msp(probs, ood_probs)
            del ood_probs

        row = {
            "dataset": "CIFAR-100",
            "architecture": "ResNet-18-CIFAR",
            "study": "trl_refresh_single_ablation",
            "method": method,
            "seed": int(cfg.seed),
            "tube_scale": float(args.tube_scale),
            "n_samples": int(args.n_samples),
            "fixbn_batches": int(args.fixbn_batches),
            "fixbn_mode": args.fixbn_mode,
            "k": int(trl.k),
            "n_spine_points": int(len(spine)),
            "fresh_max_points": int(args.fresh_max_points),
            "runtime_total_sec": float(wall),
            "fixbn_overhead_sec": float(trl.last_predict_fixbn_sec),
            **m,
        }
        if auroc is not None:
            row["auroc"] = float(auroc)
        append_jsonl(args.results, row)
        print(row, flush=True)
        del probs, targets
        cleanup()

    print("Wrote", args.results, flush=True)


if __name__ == "__main__":
    main()
