#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
for path in (ROOT / "scripts", ROOT, THIS.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trl_iclr_utils.experiment_io import append_jsonl  # noqa: E402
from cifar100_all_methods_iclr import (  # noqa: E402
    CFG,
    DEVICE,
    PracticalTRLStage2,
    get_data,
    load_or_train_map,
    set_seed,
)


def clone_spine(spine):
    out = []
    for pt in spine:
        q = {}
        for k, v in pt.items():
            if torch.is_tensor(v):
                q[k] = v.clone()
            else:
                q[k] = copy.deepcopy(v)
        out.append(q)
    return out


def fixed_map_basis_spine(spine):
    """Memory-light fixed-MAP basis spine.

    Keep each theta along the spine, but use the MAP/initial transverse basis
    and initial transverse scale at every spine point.  This intentionally avoids
    cloning the full P x K basis at every point.
    """
    n0 = spine[0]["N"]
    isp0 = spine[0]["inv_sqrt_prec"]

    out = []
    for pt in spine:
        out.append({
            "theta": pt["theta"],
            "N": n0,
            "inv_sqrt_prec": isp0,
        })
    return out


def compute_metrics(probs: torch.Tensor, targets: torch.Tensor, num_classes: int, n_bins: int = 15):
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
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bin_acc = (pred[mask] == targets[mask]).float().mean().item()
            bin_conf = conf[mask].mean().item()
            ece += mask.float().mean().item() * abs(bin_acc - bin_conf)

    return {
        "acc": float(acc),
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--results", type=str, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--tube-scale", type=float, default=4.0)
    p.add_argument("--n-samples", type=int, default=25)
    p.add_argument("--fixbn-batches", type=int, default=25)
    p.add_argument("--eval-seed-offset", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = CFG()
    cfg.seed = args.seed
    cfg.ckpt_dir = args.ckpt_dir

    set_seed(cfg.seed)

    print(">>> Loading data...")
    tr_aug, bn_clean, val_loader, ts_loader, ood_loader = get_data(cfg)

    print(">>> Loading MAP...")
    model_map = load_or_train_map(tr_aug, cfg)

    spine_path = Path(cfg.ckpt_dir) / cfg.trl_spine
    print(">>> Loading spine:", spine_path)
    payload = torch.load(spine_path, map_location="cpu")
    spine_transport = payload["spine"]

    k = int(spine_transport[0]["N"].shape[1])
    cfg.trl_k_perp = k

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
        tube_scale=args.tube_scale,
        max_delta_norm=cfg.trl_max_delta_norm,
        hvp_batches=cfg.trl_hvp_batches,
        store_every=1,
    )

    def evaluate_mode(method, spine):
        print(f">>> Evaluating {method}...")
        set_seed(cfg.seed + args.eval_seed_offset)

        trl.spine = spine
        trl.beta = float(args.tube_scale)
        trl.k = k
        trl.reset_accounting()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        probs, targets = trl.predict(
            ts_loader,
            tr_aug,
            n_samples=args.n_samples,
            fix_bn_batches=args.fixbn_batches,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        m = compute_metrics(probs, targets, cfg.num_classes)
        row = {
            "dataset": "CIFAR-100",
            "architecture": "ResNet-18-CIFAR",
            "study": "trl_basis_ablation",
            "method": method,
            "seed": int(cfg.seed),
            "tube_scale": float(args.tube_scale),
            "n_samples": int(args.n_samples),
            "fixbn_batches": int(args.fixbn_batches),
            "k": int(k),
            "n_spine_points": int(len(spine)),
            "runtime_total_sec": float(wall),
            "fixbn_overhead_sec": float(trl.last_predict_fixbn_sec),
            **m,
        }

        append_jsonl(args.results, row)
        print(row)

        del probs, targets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate transported basis using the original spine.
    evaluate_mode("TRL-transported-basis", spine_transport)

    # Build a memory-light fixed-MAP-basis spine, then release transported N's.
    spine_fixed = fixed_map_basis_spine(spine_transport)
    del spine_transport
    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    evaluate_mode("TRL-fixed-MAP-basis", spine_fixed)

    print("Wrote", args.results)


if __name__ == "__main__":
    main()
