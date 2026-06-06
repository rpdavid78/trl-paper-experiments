import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import vector_to_parameters

try:
    from cifar100_all_methods_iclr import ResNetCIFAR, fix_bn, set_seed
except ModuleNotFoundError as e:
    raise SystemExit(
        "Could not import cifar100_all_methods_iclr. Set PYTHONPATH to include the code directory, e.g.\n"
        "export PYTHONPATH=.:./scripts:$PYTHONPATH"
    ) from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


def make_loaders(seed, batch_size, num_workers, data_root):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    t_train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    t_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_aug = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=t_train_aug)
    train_clean = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=t_clean)

    indices = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx = indices[:45000]
    val_idx = indices[45000:]

    train_aug_subset = torch.utils.data.Subset(train_aug, train_idx)
    val_clean_subset = torch.utils.data.Subset(train_clean, val_idx)

    train_aug_loader = torch.utils.data.DataLoader(
        train_aug_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_clean_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_aug_loader, val_loader


def choose_indices(n, max_points):
    if max_points is None or max_points <= 0 or max_points >= n:
        return list(range(n))
    if max_points == 1:
        return [0]
    return sorted(set(int(round(i * (n - 1) / (max_points - 1))) for i in range(max_points)))


def load_spine(path):
    print(f"Loading spine with mmap: {path}", flush=True)
    try:
        payload = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    spine = payload["spine"] if isinstance(payload, dict) and "spine" in payload else payload
    print(f"Loaded spine with {len(spine)} points", flush=True)
    return spine


@torch.no_grad()
def collect_probs_and_labels(model, loader):
    model.eval()
    probs = []
    labels = []
    ce_sum = 0.0
    total = 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu())
        labels.append(y.cpu())
        ce_sum += float(crit(logits, y).item())
        total += int(y.numel())
    return torch.cat(probs, dim=0), torch.cat(labels, dim=0), ce_sum / max(total, 1)


def metrics_against_base(p0, pt, y):
    p0 = p0.float().clamp_min(EPS)
    pt = pt.float().clamp_min(EPS)
    p0 = p0 / p0.sum(dim=1, keepdim=True)
    pt = pt / pt.sum(dim=1, keepdim=True)
    m = 0.5 * (p0 + pt)

    pred0 = p0.argmax(dim=1)
    predt = pt.argmax(dim=1)
    top1_disagreement = (pred0 != predt).float().mean().item()

    kl_0t = (p0 * (p0.log() - pt.log())).sum(dim=1).mean().item()
    kl_t0 = (pt * (pt.log() - p0.log())).sum(dim=1).mean().item()
    js = 0.5 * (p0 * (p0.log() - m.log())).sum(dim=1) + 0.5 * (pt * (pt.log() - m.log())).sum(dim=1)
    js = js.mean().item()

    cos_sim = F.cosine_similarity(p0, pt, dim=1).mean().item()
    cos_dist = 1.0 - cos_sim
    l1_dist = (p0 - pt).abs().sum(dim=1).mean().item()
    l2_dist = torch.linalg.vector_norm(p0 - pt, ord=2, dim=1).mean().item()
    maxprob_diff = (p0.max(dim=1).values - pt.max(dim=1).values).abs().mean().item()
    acc0 = (pred0 == y).float().mean().item()
    acct = (predt == y).float().mean().item()

    return {
        "top1_disagreement": top1_disagreement,
        "mean_js": js,
        "mean_kl_0_to_t": kl_0t,
        "mean_kl_t_to_0": kl_t0,
        "mean_cosine_distance": cos_dist,
        "mean_l1_distance": l1_dist,
        "mean_l2_distance": l2_dist,
        "mean_maxprob_absdiff": maxprob_diff,
        "base_acc": acc0,
        "point_acc": acct,
        "acc_delta": acct - acc0,
    }


def set_model_to_theta(model, map_state_cpu, theta_cpu):
    model.load_state_dict(map_state_cpu)
    theta = theta_cpu.to(DEVICE)
    vector_to_parameters(theta, model.parameters())
    del theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ckpt-root", type=str, default="checkpoints")
    ap.add_argument("--out-dir", type=str, default="results/spine_functional_disagreement")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--fixbn-batches", type=int, default=25)
    ap.add_argument("--max-points", type=int, default=0, help="0 means all stored spine points; e.g. 5 for quick diagnostic")
    ap.add_argument("--modes", nargs="+", default=["fixbn"], choices=["raw", "fixbn"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    detail_path = Path(args.out_dir) / "spine_functional_disagreement_cifar100_resnet18_detail.csv"
    per_seed_path = Path(args.out_dir) / "spine_functional_disagreement_cifar100_resnet18_per_seed.csv"
    summary_path = Path(args.out_dir) / "spine_functional_disagreement_cifar100_resnet18_summary.csv"

    detail_rows = []
    per_seed_rows = []

    for seed in args.seeds:
        print("\n" + "=" * 100, flush=True)
        print(f"Seed {seed}", flush=True)
        print("=" * 100, flush=True)
        set_seed(seed)

        ckpt_dir = Path(args.ckpt_root) / f"checkpoints_c100_seed{seed}"
        map_path = ckpt_dir / "resnet18_cifar100_map.pth"
        spine_path = ckpt_dir / "c100_trl_stage2_spine.pth"
        if not map_path.exists():
            raise FileNotFoundError(map_path)
        if not spine_path.exists():
            raise FileNotFoundError(spine_path)

        train_aug_loader, val_loader = make_loaders(seed, args.batch_size, args.num_workers, args.data_root)
        map_state_cpu = torch.load(map_path, map_location="cpu")
        spine = load_spine(spine_path)
        idxs = choose_indices(len(spine), args.max_points)
        print("Evaluating indices:", idxs, flush=True)

        model = ResNetCIFAR(num_classes=100, use_dropout=False).to(DEVICE)

        for mode in args.modes:
            print(f"\nMode: {mode}", flush=True)
            base_probs = None
            labels = None
            base_ce = None
            mode_rows = []
            for j, idx in enumerate(idxs):
                t0 = time.perf_counter()
                theta_cpu = spine[idx]["theta"] if isinstance(spine[idx], dict) else spine[idx]
                set_model_to_theta(model, map_state_cpu, theta_cpu)
                if mode == "fixbn":
                    fix_bn(model, train_aug_loader, DEVICE, num_batches=args.fixbn_batches, return_elapsed=False)
                probs, y, ce = collect_probs_and_labels(model, val_loader)
                if j == 0:
                    base_probs = probs
                    labels = y
                    base_ce = ce
                metrics = metrics_against_base(base_probs, probs, labels)
                frac = idx / max(1, len(spine) - 1)
                elapsed = time.perf_counter() - t0
                row = {
                    "seed": seed,
                    "mode": mode,
                    "spine_index": idx,
                    "spine_fraction": frac,
                    "val_ce": ce,
                    "delta_ce": ce - base_ce,
                    "elapsed_sec": elapsed,
                    "n_spine_points": len(spine),
                    **metrics,
                }
                detail_rows.append(row)
                mode_rows.append(row)
                print(
                    f"seed={seed} mode={mode} idx={idx:03d} frac={frac:.3f} "
                    f"ce={ce:.6f} dCE={ce - base_ce:.6f} "
                    f"dis={metrics['top1_disagreement']:.6f} js={metrics['mean_js']:.6e} "
                    f"cosdist={metrics['mean_cosine_distance']:.6e} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # exclude first point for mean disagreement if desired? We report both all and nonzero path means.
            nonzero = [r for r in mode_rows if r["spine_index"] != idxs[0]] or mode_rows
            for rows, suffix in [(mode_rows, "all"), (nonzero, "nonzero")]:
                out = {"seed": seed, "mode": mode, "subset": suffix, "n_points_eval": len(rows)}
                for m in [
                    "top1_disagreement",
                    "mean_js",
                    "mean_cosine_distance",
                    "mean_l1_distance",
                    "mean_maxprob_absdiff",
                    "delta_ce",
                ]:
                    vals = np.array([float(r[m]) for r in rows], dtype=float)
                    out[m + "_mean_over_spine"] = float(vals.mean())
                    out[m + "_max_over_spine"] = float(vals.max())
                    out[m + "_endpoint"] = float(rows[-1][m])
                out["base_val_ce"] = float(base_ce)
                out["endpoint_val_ce"] = float(mode_rows[-1]["val_ce"])
                per_seed_rows.append(out)

        del spine
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(per_seed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_rows = []
    grouped = defaultdict(list)
    for r in per_seed_rows:
        grouped[(r["mode"], r["subset"])].append(r)
    metrics = [k for k in per_seed_rows[0].keys() if k not in ["seed", "mode", "subset", "n_points_eval"]]
    for (mode, subset), rows in grouped.items():
        out = {"mode": mode, "subset": subset, "n": len(rows)}
        for m in metrics:
            vals = np.array([float(r[m]) for r in rows], dtype=float)
            out[m + "_mean"] = float(vals.mean())
            out[m + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(out)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nWrote:")
    print(detail_path)
    print(per_seed_path)
    print(summary_path)
    print("\nSummary rows:")
    for row in summary_rows:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
