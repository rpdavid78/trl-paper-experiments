#!/usr/bin/env python3
"""
Fine-tuning spine diagnostic for TRL.

Goal: find a realistic small-data fine-tuning regime where the longitudinal
spine contributes to uncertainty, without changing the submitted paper unless
this diagnostic passes a pre-specified gate.

This script reuses the released CIFAR-100 ResNetCIFAR and PracticalTRLStage2
implementation. It adapts a CIFAR-100 MAP checkpoint to few-shot CIFAR-10,
constructs a TRL spine on the target data, and compares:
  - MAP fine-tuned point estimate
  - TRL-single-checkpoint: T_eff=0, transverse sampler at gamma_0 only
  - TRL-full-spine: uniform over stored spine checkpoints + transverse sampler
  - TRL-endpoint-single: transverse sampler at final stored spine point
  - TRL-best-single-val: best single spine checkpoint chosen by validation NLL

The important gate is paired, not marginal: full-spine should raise functional
variance and improve NLL/ECE/Brier relative to single-checkpoint across seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset


# ----------------------------- import project code -----------------------------

def import_c100_module(module_dir: str):
    module_dir = os.path.abspath(module_dir)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    try:
        import cifar100_all_methods_iclr as c100
    except Exception as e:
        raise RuntimeError(
            f"Could not import cifar100_all_methods_iclr from {module_dir}. "
            "Pass --cifar100-code-dir or set PYTHONPATH."
        ) from e
    return c100


# ----------------------------- utilities -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str | Path, row: Dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def subset_per_class(dataset, n_per_class: int, seed: int, exclude: Optional[set[int]] = None) -> Tuple[List[int], set[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.targets)
    indices: List[int] = []
    used: set[int] = set(exclude or set())
    for c in sorted(np.unique(labels)):
        cls = np.where(labels == c)[0]
        cls = np.asarray([int(i) for i in cls if int(i) not in used])
        rng.shuffle(cls)
        take = cls[:n_per_class].tolist()
        indices.extend(take)
        used.update(take)
    rng.shuffle(indices)
    return indices, used


def collect_targets(loader: DataLoader) -> torch.Tensor:
    ys = []
    for _, y in loader:
        ys.append(y.cpu())
    return torch.cat(ys, dim=0)


# ----------------------------- data -----------------------------

def make_cifar10_loaders(
    data_root: str,
    seed: int,
    train_per_class: int,
    val_per_class: int,
    batch_size: int,
    num_workers: int,
):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    t_aug = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    t_clean = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_aug_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=t_aug)
    train_clean_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=t_clean)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=t_clean)

    train_idx, used = subset_per_class(train_clean_full, train_per_class, seed=seed, exclude=None)
    val_idx, _ = subset_per_class(train_clean_full, val_per_class, seed=seed + 12345, exclude=used)

    train_aug = Subset(train_aug_full, train_idx)
    train_clean = Subset(train_clean_full, train_idx)
    val_clean = Subset(train_clean_full, val_idx)

    tr_aug_loader = DataLoader(train_aug, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    tr_clean_loader = DataLoader(train_clean, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_clean, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return tr_aug_loader, tr_clean_loader, val_loader, test_loader


# ----------------------------- model and fine-tuning -----------------------------

def load_c100_backbone_into_c10(model_c10: nn.Module, c100_ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(c100_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    current = model_c10.state_dict()
    matched = {}
    skipped = []
    for k, v in ckpt.items():
        kk = k.replace("module.", "")
        if kk in current and tuple(current[kk].shape) == tuple(v.shape):
            matched[kk] = v
        else:
            skipped.append(kk)
    current.update(matched)
    model_c10.load_state_dict(current)
    print(f">>> Loaded {len(matched)} tensors from CIFAR-100 checkpoint; skipped {len(skipped)} tensors.")
    if skipped:
        print("    skipped examples:", skipped[:8])
    model_c10.to(device)


def fine_tune(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float, wd: float, momentum: float) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * y.numel()
            n += y.numel()
        sched.step()
        print(f"  fine-tune epoch {ep+1:03d}/{epochs}: loss={total_loss/max(1,n):.4f}", flush=True)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(probs, dim=0)


def calc_metrics(probs: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> Dict[str, float]:
    p = probs.clamp(1e-7, 1.0)
    targets = targets.long().cpu()
    nll = F.nll_loss(torch.log(p), targets).item()
    acc = p.argmax(1).eq(targets).float().mean().item()
    confs, preds = p.max(1)
    bins = torch.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() > 0:
            ece += torch.abs(confs[mask].mean() - preds[mask].eq(targets[mask]).float().mean()) * (mask.sum() / len(p))
    oh = F.one_hot(targets, num_classes).float()
    brier = ((p - oh) ** 2).sum(1).mean().item()
    ent = (-(p * torch.log(p.clamp_min(1e-8))).sum(1)).mean().item()
    return {"acc": float(acc), "nll": float(nll), "ece": float(ece), "brier": float(brier), "entropy": float(ent)}


# ----------------------------- TRL diagnostics -----------------------------


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _vector_to_trainable_parameters(vec: torch.Tensor, model: nn.Module) -> None:
    """Load a vector into only the trainable parameters.

    This matters for head-only / last-block fine-tuning, where the TRL state
    lives in the active subspace rather than in all model parameters.
    """
    vector_to_parameters(vec, trainable_parameters(model))


def set_finetune_mode(model: nn.Module, mode: str) -> None:
    """Select the parameter subspace used for CIFAR-10 fine-tuning and TRL.

    full:      all parameters
    head:      classifier/head only
    lastblock: layer4 + classifier/head
    mid_block: layer3 + classifier/head
    """
    valid = {"full", "head", "lastblock", "mid_block"}
    if mode not in valid:
        raise ValueError(f"Unknown ft_mode={mode!r}; expected one of {sorted(valid)}.")

    def is_head(name: str) -> bool:
        return (
            name.startswith("fc.")
            or name.startswith("linear.")
            or name.startswith("classifier.")
            or name in {"fc.weight", "fc.bias", "linear.weight", "linear.bias"}
        )

    for name, p in model.named_parameters():
        if mode == "full":
            p.requires_grad = True
        elif mode == "head":
            p.requires_grad = is_head(name)
        elif mode == "lastblock":
            p.requires_grad = name.startswith("layer4.") or is_head(name)
        elif mode == "mid_block":
            p.requires_grad = name.startswith("layer3.") or is_head(name)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f">>> ft_mode={mode}: trainable params {n_trainable}/{n_total} "
        f"({100.0 * n_trainable / max(1, n_total):.2f}%)",
        flush=True,
    )
    if n_trainable == 0:
        raise RuntimeError(f"No trainable parameters selected for ft_mode={mode!r}.")


def make_prior_vec(model: nn.Module, prior_base: float, conv_boost: float, device: torch.device) -> torch.Tensor:
    chunks = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        val = prior_base if ("linear." in name or "fc." in name) else max(float(prior_base) * float(conv_boost), float(prior_base))
        chunks.append(torch.full((p.numel(),), float(val), device=device))
    return torch.cat(chunks)


def posterior_predict_samples(
    trl,
    loader: DataLoader,
    bn_loader_aug: DataLoader,
    device: torch.device,
    n_samples: int,
    fix_bn_batches: int,
    mode: str,
    spine_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Return mean probs, sample probs [S,N,C], and FixBN wall time."""
    if len(trl.spine) == 0:
        raise RuntimeError("Empty spine; call build() first.")

    # Reset parameters and BN buffers to MAP before each diagnostic mode.
    trl.model.load_state_dict(trl.map_state)
    trl.model.to(device)
    trl.model.eval()

    if mode == "full":
        choices = None
    elif mode == "single":
        choices = [0]
    elif mode == "endpoint":
        choices = [len(trl.spine) - 1]
    elif mode == "idx":
        if spine_idx is None:
            raise ValueError("spine_idx required when mode='idx'")
        choices = [int(spine_idx)]
    else:
        raise ValueError(f"Unknown mode {mode}")

    ens_probs = []
    fixbn_total = 0.0

    for _ in range(n_samples):
        if choices is None:
            pt = trl.spine[np.random.randint(len(trl.spine))]
        else:
            pt = trl.spine[choices[np.random.randint(len(choices))]]

        th_loc = pt["theta"].to(device)
        N_loc = pt["N"].to(device)
        isp_loc = pt["inv_sqrt_prec"].to(device)
        z = torch.randn(trl.k, device=device)
        theta_sample = th_loc + N_loc @ (trl.beta * (isp_loc * z))
        _vector_to_trainable_parameters(theta_sample, trl.model)

        elapsed = trl.fix_bn(trl.model, bn_loader_aug, device, num_batches=fix_bn_batches, return_elapsed=True) if hasattr(trl, "fix_bn") else None
        # PracticalTRLStage2 does not expose fix_bn as method; caller attaches it below.
        if elapsed is None:
            raise RuntimeError("fix_bn was not attached to TRL object")
        fixbn_total += float(elapsed)

        probs = []
        with torch.no_grad():
            trl.model.eval()
            for x, _ in loader:
                probs.append(torch.softmax(trl.model(x.to(device)), dim=1).cpu())
        ens_probs.append(torch.cat(probs, dim=0))

        del th_loc, N_loc, isp_loc, z, theta_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    samples = torch.stack(ens_probs, dim=0)
    return samples.mean(0), samples, fixbn_total



@torch.no_grad()
def _refresh_bn_for_spine_signal(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> None:
    """For the deterministic spine-signal diagnostic, keep the checkpoint BN buffers.

    The posterior sampler has its own FixBN path. For this diagnostic we want the
    pure functional drift of the stored spine parameters relative to the MAP
    buffers. Resetting BN stats on the tiny target set can collapse validation CE
    toward random predictions and obscure the longitudinal signal.
    """
    model.eval()


def _js_to_base(probs: torch.Tensor, base_probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-8)
    q = base_probs.clamp_min(1e-8)
    m = (0.5 * (p + q)).clamp_min(1e-8)
    js = 0.5 * (p * (p.log() - m.log())).sum(1) + 0.5 * (q * (q.log() - m.log())).sum(1)
    return float(js.mean().item())


def _functional_metrics_vs_base(probs: torch.Tensor, base_probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = probs.cpu()
    base_probs = base_probs.cpu()
    targets = targets.cpu().long()

    ce = calc_metrics(probs, targets, num_classes=probs.shape[1])["nll"]
    base_ce = calc_metrics(base_probs, targets, num_classes=base_probs.shape[1])["nll"]

    pred = probs.argmax(1)
    base_pred = base_probs.argmax(1)

    cosine = F.cosine_similarity(probs, base_probs, dim=1)
    maxprob = probs.max(1).values
    base_maxprob = base_probs.max(1).values

    return {
        "ce": float(ce),
        "delta_ce": float(ce - base_ce),
        "top1_disagreement": float(pred.ne(base_pred).float().mean().item()),
        "mean_js": _js_to_base(probs, base_probs),
        "mean_cosine_distance": float((1.0 - cosine).mean().item()),
        "mean_l1_distance": float((probs - base_probs).abs().sum(1).mean().item()),
        "mean_maxprob_absdiff": float((maxprob - base_maxprob).abs().mean().item()),
    }


def spine_longitudinal_signal(
    trl,
    val_loader: DataLoader,
    val_targets: torch.Tensor,
    bn_loader_aug: DataLoader,
    device: torch.device,
    fix_bn_batches: int,
    seed: int,
    out_csv: str | Path,
) -> None:
    """Measure functional drift of deterministic spine points on validation data."""
    if len(trl.spine) == 0:
        raise RuntimeError("Empty spine; call build() first.")

    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)

    rows = []
    base_probs = None
    n_points = len(trl.spine)

    print(">>> Measuring longitudinal spine signal on validation set...", flush=True)

    for idx, pt in enumerate(trl.spine):
        trl.model.load_state_dict(trl.map_state)
        trl.model.to(device)
        _vector_to_trainable_parameters(pt["theta"].to(device), trl.model)

        _refresh_bn_for_spine_signal(trl.model, bn_loader_aug, device, fix_bn_batches)
        probs = predict_probs(trl.model, val_loader, device)

        if base_probs is None:
            base_probs = probs.clone()

        metrics = _functional_metrics_vs_base(probs, base_probs, val_targets)
        row = {
            "seed": int(seed),
            "idx": int(idx),
            "frac": float(idx / max(1, n_points - 1)),
            **metrics,
        }
        rows.append(row)

        print(
            f"spine-signal seed={seed} idx={idx:03d} frac={row['frac']:.3f} "
            f"ce={row['ce']:.6f} dCE={row['delta_ce']:.6f} "
            f"dis={row['top1_disagreement']:.6f} js={row['mean_js']:.6e}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    nonzero = df[df["idx"] > 0]
    endpoint = df.iloc[-1]
    summary = {
        "seed": int(seed),
        "n_points": int(n_points),
        "top1_disagreement_mean_over_spine": float(df["top1_disagreement"].mean()),
        "top1_disagreement_max_over_spine": float(df["top1_disagreement"].max()),
        "top1_disagreement_endpoint": float(endpoint["top1_disagreement"]),
        "mean_js_mean_over_spine": float(df["mean_js"].mean()),
        "mean_js_max_over_spine": float(df["mean_js"].max()),
        "mean_js_endpoint": float(endpoint["mean_js"]),
        "delta_ce_mean_over_spine": float(df["delta_ce"].mean()),
        "delta_ce_max_over_spine": float(df["delta_ce"].max()),
        "delta_ce_endpoint": float(endpoint["delta_ce"]),
        "top1_disagreement_nonzero_mean": float(nonzero["top1_disagreement"].mean()) if len(nonzero) else 0.0,
        "mean_js_nonzero_mean": float(nonzero["mean_js"].mean()) if len(nonzero) else 0.0,
        "delta_ce_nonzero_mean": float(nonzero["delta_ce"].mean()) if len(nonzero) else 0.0,
    }

    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print(f">>> Wrote spine signal detail: {out_csv}", flush=True)
    print(f">>> Wrote spine signal summary: {summary_csv}", flush=True)

def avg_function_var(samples: torch.Tensor) -> float:
    # samples: [S, N, C]
    return float(samples.var(dim=0, unbiased=False).sum(dim=1).mean().item())


def evaluate_posterior_method(
    method_name: str,
    trl,
    loader: DataLoader,
    targets: torch.Tensor,
    bn_loader_aug: DataLoader,
    device: torch.device,
    n_samples: int,
    fix_bn_batches: int,
    mode: str,
    spine_idx: Optional[int] = None,
) -> Dict[str, float | int | str]:
    t0 = time.perf_counter()
    p, samples, fixbn_sec = posterior_predict_samples(
        trl, loader, bn_loader_aug, device, n_samples, fix_bn_batches, mode=mode, spine_idx=spine_idx
    )
    metrics = calc_metrics(p, targets, num_classes=10)
    metrics.update({
        "method": method_name,
        "avg_function_var": avg_function_var(samples),
        "runtime_sec": time.perf_counter() - t0,
        "fixbn_sec": float(fixbn_sec),
        "n_samples": int(n_samples),
        "spine_idx": -1 if spine_idx is None else int(spine_idx),
    })
    return metrics


def select_best_single_by_val(
    trl,
    val_loader: DataLoader,
    val_targets: torch.Tensor,
    bn_loader_aug: DataLoader,
    device: torch.device,
    n_samples: int,
    fix_bn_batches: int,
    stride: int,
) -> Tuple[int, pd.DataFrame]:
    rows = []
    candidate_indices = list(range(0, len(trl.spine), max(1, stride)))
    if (len(trl.spine) - 1) not in candidate_indices:
        candidate_indices.append(len(trl.spine) - 1)
    for idx in candidate_indices:
        row = evaluate_posterior_method(
            f"candidate_idx_{idx}", trl, val_loader, val_targets, bn_loader_aug, device,
            n_samples=n_samples, fix_bn_batches=fix_bn_batches, mode="idx", spine_idx=idx
        )
        row["candidate_idx"] = int(idx)
        rows.append(row)
        print(f"    val candidate idx={idx:03d}: nll={row['nll']:.4f} acc={row['acc']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    best_idx = int(df.sort_values("nll").iloc[0]["candidate_idx"])
    return best_idx, df


# ----------------------------- main -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cifar100-code-dir", default=".")
    p.add_argument("--c100-ckpt", default="checkpoints/checkpoints_c100_seed0/resnet18_cifar100_map.pth")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="results/finetune_spine/finetune_c10_spine_smoke.jsonl")
    p.add_argument("--ckpt-dir", default="results/finetune_spine/checkpoints")
    p.add_argument("--train-per-class", type=int, default=100)
    p.add_argument("--val-per-class", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ft-epochs", type=int, default=5)
    p.add_argument("--ft-lr", type=float, default=0.01)
    p.add_argument("--ft-wd", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--ft-mode", choices=["full", "head", "lastblock", "mid_block"], default="full", help="Which parameter subset to fine-tune and use for TRL.")
    p.add_argument("--force-ft", action="store_true")
    p.add_argument("--trl-steps", type=int, default=20)
    p.add_argument("--trl-k", type=int, default=30)
    p.add_argument("--trl-step-size", type=float, default=0.01)
    p.add_argument("--trl-eta", type=float, default=1e-3)
    p.add_argument("--trl-tube-scale", type=float, default=4.0)
    p.add_argument("--trl-max-delta-norm", type=float, default=0.02)
    p.add_argument("--trl-hvp-batches", type=int, default=5)
    p.add_argument("--trl-store-every", type=int, default=1)
    p.add_argument("--prior-base", type=float, default=5.0)
    p.add_argument("--prior-conv-boost", type=float, default=50.0)
    p.add_argument("--pred-samples", type=int, default=25)
    p.add_argument("--fixbn-batches", type=int, default=10)
    p.add_argument("--best-val-samples", type=int, default=8)
    p.add_argument("--best-val-stride", type=int, default=4)
    p.add_argument("--signal-only", action="store_true", help="Only build spine and write longitudinal signal CSV, then exit.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    c100 = import_c100_module(args.cifar100_code_dir)
    # Attach fix_bn to TRL instances for custom prediction helper.

    ensure_dir(args.ckpt_dir)
    ensure_dir(Path(args.out).parent)

    print("=" * 100)
    print(f"Fine-tuning TRL spine diagnostic | seed={args.seed} | device={device}")
    print("=" * 100)

    tr_aug, tr_clean, val_loader, test_loader = make_cifar10_loaders(
        args.data_root, args.seed, args.train_per_class, args.val_per_class,
        args.batch_size, args.num_workers,
    )
    val_targets = collect_targets(val_loader)
    test_targets = collect_targets(test_loader)

    model = c100.ResNetCIFAR(num_classes=10, use_dropout=False).to(device)
    load_c100_backbone_into_c10(model, args.c100_ckpt, device)
    set_finetune_mode(model, args.ft_mode)

    ft_ckpt = Path(args.ckpt_dir) / f"c10_finetune_from_c100_seed{args.seed}_mode{args.ft_mode}_n{args.train_per_class}_ep{args.ft_epochs}.pth"
    if ft_ckpt.exists() and not args.force_ft:
        print(f">>> Loading fine-tuned checkpoint: {ft_ckpt}")
        model.load_state_dict(torch.load(ft_ckpt, map_location=device))
    else:
        print(">>> Fine-tuning on few-shot CIFAR-10 target...")
        fine_tune(model, tr_aug, device, args.ft_epochs, args.ft_lr, args.ft_wd, args.momentum)
        torch.save(model.state_dict(), ft_ckpt)
        print(f">>> Saved fine-tuned checkpoint: {ft_ckpt}")

    p_map = predict_probs(model, test_loader, device)
    map_metrics = calc_metrics(p_map, test_targets, num_classes=10)
    map_row = {
        "dataset": "CIFAR-10-fewshot-finetune",
        "architecture": "ResNet-18-CIFAR",
        "study": "finetune_spine_smoke",
        "method": "MAP-finetuned",
        "seed": args.seed,
        "source_ckpt": args.c100_ckpt,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "ft_epochs": args.ft_epochs,
        "ft_lr": args.ft_lr,
        "ft_wd": args.ft_wd,
            "ft_mode": args.ft_mode,
        **map_metrics,
    }
    print(">>> MAP fine-tuned test:", map_row)
    append_jsonl(args.out, map_row)

    prior_vec = make_prior_vec(model, args.prior_base, args.prior_conv_boost, device)
    trl = c100.PracticalTRLStage2(
        map_model=model,
        prior_vec=prior_vec,
        clean_loader=tr_clean,
        steps=args.trl_steps,
        k_perp=args.trl_k,
        step_size=args.trl_step_size,
        eta=args.trl_eta,
        tube_scale=args.trl_tube_scale,
        max_delta_norm=args.trl_max_delta_norm,
        hvp_batches=args.trl_hvp_batches,
        store_every=args.trl_store_every,
    )
    trl.fix_bn = c100.fix_bn

    print(">>> Building TRL spine on target few-shot data...")
    t_build = time.perf_counter()
    trl.build()
    build_sec = time.perf_counter() - t_build
    print(f">>> Spine built with {len(trl.spine)} stored points in {build_sec:.1f}s")
    spine_signal_path = Path(args.out).with_name(Path(args.out).stem + f"_seed{args.seed}_spine_signal.csv")
    val_targets_for_signal = collect_targets(val_loader)
    spine_longitudinal_signal(
        trl=trl,
        val_loader=val_loader,
        val_targets=val_targets_for_signal,
        bn_loader_aug=tr_aug,
        device=device,
        fix_bn_batches=args.fixbn_batches,
        seed=args.seed,
        out_csv=spine_signal_path,
    )

    if args.signal_only:
        print(">>> signal-only requested; exiting after spine longitudinal diagnostic.", flush=True)
        return

    # Main diagnostics on test set.
    rows = []
    for method_name, mode in [
        ("TRL-single-checkpoint", "single"),
        ("TRL-full-spine", "full"),
        ("TRL-endpoint-single", "endpoint"),
    ]:
        print(f">>> Evaluating {method_name}...")
        row = evaluate_posterior_method(
            method_name, trl, test_loader, test_targets, tr_aug, device,
            n_samples=args.pred_samples, fix_bn_batches=args.fixbn_batches, mode=mode
        )
        row.update({
            "dataset": "CIFAR-10-fewshot-finetune",
            "architecture": "ResNet-18-CIFAR",
            "study": "finetune_spine_smoke",
            "seed": args.seed,
            "source_ckpt": args.c100_ckpt,
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "ft_epochs": args.ft_epochs,
            "ft_lr": args.ft_lr,
            "ft_wd": args.ft_wd,
            "ft_mode": args.ft_mode,
            "trl_steps": args.trl_steps,
            "trl_k": args.trl_k,
            "tube_scale": args.trl_tube_scale,
            "spine_points": len(trl.spine),
            "spine_build_sec": build_sec,
            "prior_base": args.prior_base,
        })
        print(row)
        append_jsonl(args.out, row)
        rows.append(row)

    # Best single checkpoint on spine chosen by validation NLL.
    print(">>> Selecting best single spine checkpoint by validation NLL...")
    best_idx, val_df = select_best_single_by_val(
        trl, val_loader, val_targets, tr_aug, device,
        n_samples=args.best_val_samples, fix_bn_batches=args.fixbn_batches,
        stride=args.best_val_stride,
    )
    val_path = Path(args.out).with_name(Path(args.out).stem + f"_seed{args.seed}_val_candidates.csv")
    val_df.to_csv(val_path, index=False)
    print(f">>> Best val checkpoint idx={best_idx}; wrote {val_path}")

    best_row = evaluate_posterior_method(
        "TRL-best-single-val", trl, test_loader, test_targets, tr_aug, device,
        n_samples=args.pred_samples, fix_bn_batches=args.fixbn_batches, mode="idx", spine_idx=best_idx
    )
    best_val_nll = float(val_df[val_df["candidate_idx"] == best_idx]["nll"].iloc[0])
    best_row.update({
        "dataset": "CIFAR-10-fewshot-finetune",
        "architecture": "ResNet-18-CIFAR",
        "study": "finetune_spine_smoke",
        "seed": args.seed,
        "source_ckpt": args.c100_ckpt,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "ft_epochs": args.ft_epochs,
        "ft_lr": args.ft_lr,
        "ft_wd": args.ft_wd,
            "ft_mode": args.ft_mode,
        "trl_steps": args.trl_steps,
        "trl_k": args.trl_k,
        "tube_scale": args.trl_tube_scale,
        "spine_points": len(trl.spine),
        "spine_build_sec": build_sec,
        "prior_base": args.prior_base,
        "best_val_nll": best_val_nll,
        "best_val_idx": int(best_idx),
    })
    print(best_row)
    append_jsonl(args.out, best_row)
    rows.append(best_row)

    # Quick gate printed at end.
    df = pd.DataFrame(rows)
    try:
        single = df[df.method == "TRL-single-checkpoint"].iloc[0]
        full = df[df.method == "TRL-full-spine"].iloc[0]
        best = df[df.method == "TRL-best-single-val"].iloc[0]
        print("\n" + "=" * 100)
        print("SMOKE GATE")
        print("=" * 100)
        print(f"Delta avg_function_var full-single: {full.avg_function_var - single.avg_function_var:+.6f}")
        print(f"Delta NLL full-single: {full.nll - single.nll:+.6f}")
        print(f"Delta ECE full-single: {full.ece - single.ece:+.6f}")
        print(f"Delta Brier full-single: {full.brier - single.brier:+.6f}")
        print(f"Delta NLL full-best-single-val: {full.nll - best.nll:+.6f}")
        print("Continue to 10 seeds only if functional variance rises clearly and NLL/ECE/Brier do not degrade materially.")
    except Exception as e:
        print("Could not print gate:", e)

    print(f"\nWrote JSONL rows to {args.out}")


if __name__ == "__main__":
    main()
