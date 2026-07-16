#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
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


def binary_auroc(pos_scores, neg_scores):
    """AUROC with higher score meaning more ID-like."""
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    order = np.argsort(scores)
    sorted_scores = scores[order]

    ranks_sorted = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ranks_sorted

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--spine-file", type=str, default=None)
    p.add_argument("--results", type=str, required=True)
    p.add_argument("--tube-scales", nargs="+", type=float, default=[2.0, 3.0, 4.0, 6.0, 10.0, 20.0])
    p.add_argument("--n-samples", type=int, default=25)
    p.add_argument("--fixbn-batches", type=int, default=25)
    p.add_argument("--eval-seed-base", type=int, default=1000)
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

    if args.spine_file is not None:
        spine_path = Path(args.spine_file)
    else:
        backup = Path(cfg.ckpt_dir) / "c100_trl_stage2_spine_MAIN_BACKUP.pth"
        regular = Path(cfg.ckpt_dir) / cfg.trl_spine
        spine_path = backup if backup.exists() else regular

    print(">>> Loading spine:", spine_path)
    payload = torch.load(spine_path, map_location="cpu")
    spine = payload["spine"] if isinstance(payload, dict) and "spine" in payload else payload

    k = int(spine[0]["N"].shape[1])
    cfg.trl_k_perp = k

    params = [p for p in model_map.parameters() if p.requires_grad]
    dummy_prior = torch.zeros_like(parameters_to_vector(params)).to(DEVICE)

    trl = PracticalTRLStage2(
        map_model=model_map,
        prior_vec=dummy_prior,
        clean_loader=bn_clean,
        steps=len(spine),
        k_perp=k,
        step_size=cfg.trl_step_size,
        eta=cfg.trl_eta,
        tube_scale=1.0,
        max_delta_norm=cfg.trl_max_delta_norm,
        hvp_batches=cfg.trl_hvp_batches,
        store_every=1,
    )
    trl.spine = spine
    trl.k = k

    for beta in args.tube_scales:
        print(f">>> Evaluating tube_scale={beta}...")
        trl.beta = float(beta)
        trl.reset_accounting()

        # Same posterior samples across tube scales, to reduce Monte Carlo noise.
        set_seed(args.eval_seed_base + cfg.seed)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        probs_id, targets_id = trl.predict(
            ts_loader,
            tr_aug,
            n_samples=args.n_samples,
            fix_bn_batches=args.fixbn_batches,
        )
        fixbn_id = float(trl.last_predict_fixbn_sec)

        set_seed(args.eval_seed_base + cfg.seed)
        probs_ood, _ = trl.predict(
            ood_loader,
            tr_aug,
            n_samples=args.n_samples,
            fix_bn_batches=args.fixbn_batches,
        )
        fixbn_ood = float(trl.last_predict_fixbn_sec)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        m = compute_metrics(probs_id, targets_id, cfg.num_classes)

        id_score = probs_id.max(dim=1).values.cpu().numpy()
        ood_score = probs_ood.max(dim=1).values.cpu().numpy()
        auroc = binary_auroc(id_score, ood_score)

        row = {
            "dataset": "CIFAR-100",
            "architecture": "ResNet-18-CIFAR",
            "study": "trl_tube_scale_sensitivity",
            "method": "TRL",
            "seed": int(cfg.seed),
            "tube_scale": float(beta),
            "n_samples": int(args.n_samples),
            "fixbn_batches": int(args.fixbn_batches),
            "k": int(k),
            "n_spine_points": int(len(spine)),
            "runtime_total_sec": float(wall),
            "fixbn_overhead_sec": float(fixbn_id + fixbn_ood),
            "auroc": float(auroc),
            **m,
        }

        append_jsonl(args.results, row)
        print(row)

        del probs_id, probs_ood, targets_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Wrote", args.results)


if __name__ == "__main__":
    main()
