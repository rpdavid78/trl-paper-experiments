#!/usr/bin/env python3
"""
Legacy toy diagnostic: TRL single-checkpoint (T_eff=0) vs TRL full-spine.

This script is intentionally self-contained. It builds low-loss spines for two
small toy tasks using a full-network Hessian, then compares posterior predictive
sampling from:
  1) a single checkpoint around the MAP endpoint, and
  2) a uniform mixture over the stored spine points.

This is the original toy-spine diagnostic. The final paper Tables 3--5 are
reproduced by toy/rerun_toy_tables.py and toy/run_final_toy_tables.sh.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


# ----------------------------- utilities -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def param_meta(model: nn.Module):
    names, shapes, numels = [], [], []
    for name, p in model.named_parameters():
        names.append(name)
        shapes.append(tuple(p.shape))
        numels.append(p.numel())
    return names, shapes, numels


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def vector_to_param_dict(theta: torch.Tensor, names, shapes, numels) -> Dict[str, torch.Tensor]:
    out = {}
    idx = 0
    for name, shape, numel in zip(names, shapes, numels):
        out[name] = theta[idx : idx + numel].reshape(shape)
        idx += numel
    return out


def make_sine(seed: int, n_train: int = 80, n_test: int = 400, noise: float = 0.15, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_train = torch.empty(n_train, 1).uniform_(-3.0, 3.0, generator=g)
    y_train = torch.sin(2.0 * x_train) + 0.3 * torch.cos(3.0 * x_train) + noise * torch.randn(n_train, 1, generator=g)
    x_test = torch.linspace(-4.0, 4.0, n_test).reshape(-1, 1)
    y_test = torch.sin(2.0 * x_test) + 0.3 * torch.cos(3.0 * x_test)
    return x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)


def make_moons(seed: int, n_train: int = 500, n_test: int = 2000, noise: float = 0.10, device="cpu"):
    rng = np.random.default_rng(seed)

    def sample(n):
        n0 = n // 2
        n1 = n - n0
        t0 = rng.uniform(0, math.pi, size=n0)
        t1 = rng.uniform(0, math.pi, size=n1)
        x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
        x1 = np.stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)], axis=1)
        X = np.concatenate([x0, x1], axis=0)
        y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)], axis=0)
        X = X + noise * rng.standard_normal(X.shape)
        perm = rng.permutation(n)
        return X[perm], y[perm]

    Xtr, ytr = sample(n_train)
    Xte, yte = sample(n_test)
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    return (
        torch.tensor(Xtr, dtype=torch.float32, device=device),
        torch.tensor(ytr, dtype=torch.long, device=device),
        torch.tensor(Xte, dtype=torch.float32, device=device),
        torch.tensor(yte, dtype=torch.long, device=device),
    )


# ----------------------------- training -----------------------------


def train_map_regression(model, X, y, steps=5000, lr=2e-3, weight_decay=1e-4, noise=0.15, verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(X)
        loss = 0.5 * ((pred - y) ** 2).mean() / (noise**2)
        loss.backward()
        opt.step()
        if verbose and (step + 1) % 1000 == 0:
            print(f"  train step={step+1} loss={loss.item():.6f}", flush=True)


def train_map_classification(model, X, y, steps=5000, lr=2e-3, weight_decay=1e-4, verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if verbose and (step + 1) % 1000 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  train step={step+1} loss={loss.item():.6f} acc={acc:.3f}", flush=True)


# ----------------------------- full Hessian geometry -----------------------------


def build_loss_fn(task, model, names, shapes, numels, X, y, noise, weight_decay):
    def loss_fn(theta):
        params = vector_to_param_dict(theta, names, shapes, numels)
        out = functional_call(model, params, (X,))
        if task == "sine":
            data_loss = 0.5 * ((out - y) ** 2).mean() / (noise**2)
        else:
            data_loss = F.cross_entropy(out, y)
        prior = 0.5 * weight_decay * (theta @ theta) / max(1, theta.numel())
        return data_loss + prior

    return loss_fn


def hessian_eigh(loss_fn, theta, device, jitter=1e-5):
    theta = theta.detach().clone().to(device).requires_grad_(True)
    H = torch.autograd.functional.hessian(loss_fn, theta, vectorize=True)
    H = H.detach()
    H = 0.5 * (H + H.T)
    H = H + jitter * torch.eye(H.shape[0], device=device, dtype=H.dtype)
    evals, evecs = torch.linalg.eigh(H)
    return evals.detach(), evecs.detach()


def choose_flat_direction(task, model, names, shapes, numels, loss_fn, theta, evals, evecs, X_probe, prev_dir=None, step_size=0.02, n_candidates=20):
    # Candidates among the lowest-curvature eigenvectors. Choose the one that produces
    # the largest functional displacement for a small step, while staying local.
    n = evals.numel()
    n_candidates = min(n_candidates, n)
    idxs = torch.arange(n_candidates, device=theta.device)
    best_score = -1.0
    best_v = evecs[:, 0]
    with torch.no_grad():
        base_params = vector_to_param_dict(theta, names, shapes, numels)
        base_out = functional_call(model, base_params, (X_probe,))
        for idx in idxs:
            v = evecs[:, int(idx.item())]
            if prev_dir is not None and torch.dot(v, prev_dir) < 0:
                v = -v
            theta_try = theta + step_size * v
            params_try = vector_to_param_dict(theta_try, names, shapes, numels)
            out_try = functional_call(model, params_try, (X_probe,))
            if task == "sine":
                score = (out_try - base_out).abs().mean().item()
            else:
                p0 = base_out.softmax(-1)
                p1 = out_try.softmax(-1)
                score = (p1 - p0).abs().sum(-1).mean().item()
            if score > best_score:
                best_score = score
                best_v = v
    if prev_dir is not None and torch.dot(best_v, prev_dir) < 0:
        best_v = -best_v
    return best_v / (best_v.norm() + 1e-12), best_score


def correct_to_low_loss(loss_fn, theta, tangent, target_loss, steps=50, lr=0.05, tolerance=0.02):
    # Projected gradient correction: reduce loss without undoing progress along tangent.
    z = theta.detach().clone().requires_grad_(True)
    for _ in range(steps):
        loss = loss_fn(z)
        if float(loss.detach().cpu()) <= target_loss * (1.0 + tolerance) + 1e-6:
            break
        (grad,) = torch.autograd.grad(loss, z)
        grad = grad - torch.dot(grad, tangent) * tangent
        with torch.no_grad():
            z -= lr * grad / (grad.norm() + 1e-12)
        z.requires_grad_(True)
    return z.detach()


def build_spine(task, model, names, shapes, numels, loss_fn, theta0, X_probe, T, step_size, hessian_refresh, correct_steps, correct_lr, verbose=False):
    spine = [theta0.detach().clone()]
    theta = theta0.detach().clone()
    prev_dir = None
    evals = evecs = None
    base_loss = float(loss_fn(theta).detach().cpu())
    target_loss = base_loss
    geom_time = 0.0
    for t in range(1, T + 1):
        if evals is None or ((t - 1) % hessian_refresh == 0):
            t0 = time.perf_counter()
            evals, evecs = hessian_eigh(loss_fn, theta, theta.device)
            geom_time += time.perf_counter() - t0
        tangent, score = choose_flat_direction(
            task, model, names, shapes, numels, loss_fn, theta, evals, evecs, X_probe,
            prev_dir=prev_dir, step_size=step_size, n_candidates=min(30, theta.numel())
        )
        theta_pred = theta + step_size * tangent
        theta_corr = correct_to_low_loss(loss_fn, theta_pred, tangent, target_loss, steps=correct_steps, lr=correct_lr)
        theta = theta_corr.detach()
        spine.append(theta.clone())
        prev_dir = tangent.detach()
        if verbose:
            print(f"    spine {t:03d}/{T} loss={float(loss_fn(theta).detach().cpu()):.6f} func_step={score:.4e}", flush=True)
    return spine, geom_time


def transverse_basis(evals, evecs, k, beta, prior_prec=1e-3):
    # Use top-curvature directions with inverse-sqrt scaling. This mirrors the
    # intended transverse uncertainty: data-sensitive directions get small but
    # calibrated perturbations. beta controls the tube width.
    k = min(k, evals.numel())
    idx = torch.argsort(evals, descending=True)[:k]
    vals = torch.clamp(evals[idx], min=0.0)
    V = evecs[:, idx]
    scales = beta / torch.sqrt(vals + prior_prec)
    return V, scales


def sample_thetas(points: List[torch.Tensor], V, scales, n_samples: int, seed: int):
    g = torch.Generator(device=V.device).manual_seed(seed)
    out = []
    n_points = len(points)
    for s in range(n_samples):
        j = int(torch.randint(0, n_points, (1,), generator=g, device=V.device).item())
        eps = torch.randn(scales.numel(), generator=g, device=V.device, dtype=V.dtype)
        out.append(points[j] + V @ (eps * scales))
    return out


# ----------------------------- evaluation -----------------------------


def regression_predictive(model, names, shapes, numels, thetas, X):
    preds = []
    with torch.no_grad():
        for theta in thetas:
            params = vector_to_param_dict(theta, names, shapes, numels)
            preds.append(functional_call(model, params, (X,)).reshape(-1))
    return torch.stack(preds, dim=0)


def classification_predictive(model, names, shapes, numels, thetas, X):
    probs = []
    with torch.no_grad():
        for theta in thetas:
            params = vector_to_param_dict(theta, names, shapes, numels)
            probs.append(functional_call(model, params, (X,)).softmax(-1))
    return torch.stack(probs, dim=0).mean(dim=0)


def eval_sine(model, names, shapes, numels, points, V, scales, X_test, y_test, n_samples, seed, noise):
    thetas = sample_thetas(points, V, scales, n_samples, seed)
    preds = regression_predictive(model, names, shapes, numels, thetas, X_test)
    mu = preds.mean(dim=0)
    var = preds.var(dim=0, unbiased=False) + noise**2 + 1e-8
    y = y_test.reshape(-1)
    rmse = torch.sqrt(((mu - y) ** 2).mean()).item()
    nll = (0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((y - mu) ** 2) / var).mean().item()
    z = (y - mu) / torch.sqrt(var)
    z_var = z.var(unbiased=False).item()
    cov1 = (z.abs() <= 1.0).float().mean().item()
    avg_pred_var = preds.var(dim=0, unbiased=False).mean().item()
    return {"rmse": rmse, "nll": nll, "z_var": z_var, "cov_1sigma": cov1, "avg_function_var": avg_pred_var}


def eval_moons(model, names, shapes, numels, points, V, scales, X_test, y_test, n_samples, seed):
    thetas = sample_thetas(points, V, scales, n_samples, seed)

    # Keep per-sample predictive probabilities to measure functional variation.
    probs = []
    with torch.no_grad():
        for theta in thetas:
            params = vector_to_param_dict(theta, names, shapes, numels)
            probs.append(functional_call(model, params, (X_test,)).softmax(-1))
    probs_samples = torch.stack(probs, dim=0)  # [S, N, C]
    p = probs_samples.mean(dim=0)

    # Average predictive-function variance across posterior samples.
    avg_function_var = probs_samples.var(dim=0, unbiased=False).sum(dim=1).mean().item()

    eps = 1e-8
    nll = (-torch.log(p[torch.arange(y_test.numel(), device=y_test.device), y_test] + eps)).mean().item()
    onehot = F.one_hot(y_test, num_classes=2).float()
    brier = ((p - onehot) ** 2).sum(dim=1).mean().item()
    acc = (p.argmax(dim=1) == y_test).float().mean().item()
    entropy = (-(p * torch.log(p + eps)).sum(dim=1)).mean().item()
    return {
        "acc": acc,
        "nll": nll,
        "brier": brier,
        "entropy": entropy,
        "avg_function_var": avg_function_var,
    }


# ----------------------------- runs -----------------------------


@dataclass
class ToyConfig:
    task: str
    in_dim: int
    out_dim: int
    T: int
    step_size: float
    beta: float
    k: int
    train_steps: int
    hessian_refresh: int
    correct_steps: int
    correct_lr: float
    noise: float = 0.15


def run_task(cfg: ToyConfig, seed: int, args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(seed)
    model = MLP(cfg.in_dim, cfg.out_dim, hidden=args.hidden).to(device)
    weight_decay = args.weight_decay

    if cfg.task == "sine":
        Xtr, ytr, Xte, yte = make_sine(seed, device=device, noise=cfg.noise)
        train_map_regression(model, Xtr, ytr, steps=cfg.train_steps, lr=args.lr, weight_decay=weight_decay, noise=cfg.noise, verbose=args.verbose)
    else:
        Xtr, ytr, Xte, yte = make_moons(seed, device=device)
        train_map_classification(model, Xtr, ytr, steps=cfg.train_steps, lr=args.lr, weight_decay=weight_decay, verbose=args.verbose)

    names, shapes, numels = param_meta(model)
    theta0 = flatten_params(model).to(device)
    loss_fn = build_loss_fn(cfg.task, model, names, shapes, numels, Xtr, ytr, cfg.noise, weight_decay)
    map_loss = float(loss_fn(theta0).detach().cpu())

    print(f"\n>>> {cfg.task} seed={seed}: P={theta0.numel()} MAP-loss={map_loss:.6f}", flush=True)
    t0 = time.perf_counter()
    evals0, evecs0 = hessian_eigh(loss_fn, theta0, device)
    hess0_time = time.perf_counter() - t0
    print(f"    MAP Hessian done in {hess0_time:.1f}s | min={evals0.min().item():.3e} max={evals0.max().item():.3e}", flush=True)

    V, scales = transverse_basis(evals0, evecs0, cfg.k, cfg.beta, prior_prec=args.prior_prec)

    spine, geom_time = build_spine(
        cfg.task, model, names, shapes, numels, loss_fn, theta0, Xtr,
        T=cfg.T, step_size=cfg.step_size, hessian_refresh=cfg.hessian_refresh,
        correct_steps=cfg.correct_steps, correct_lr=cfg.correct_lr, verbose=args.verbose,
    )
    end_loss = float(loss_fn(spine[-1]).detach().cpu())
    print(f"    spine built: points={len(spine)} end-loss={end_loss:.6f} delta={end_loss-map_loss:.6f} geom_time={geom_time:.1f}s", flush=True)

    rows = []
    variants = [
        ("TRL-single-checkpoint", [theta0]),
        ("TRL-full-spine", spine),
    ]
    for method, points in variants:
        t1 = time.perf_counter()
        if cfg.task == "sine":
            metrics = eval_sine(model, names, shapes, numels, points, V, scales, Xte, yte, args.n_pred_samples, seed + 1000, cfg.noise)
        else:
            metrics = eval_moons(model, names, shapes, numels, points, V, scales, Xte, yte, args.n_pred_samples, seed + 1000)
        runtime = time.perf_counter() - t1
        row = {
            "task": cfg.task,
            "method": method,
            "seed": seed,
            "T": 0 if method == "TRL-single-checkpoint" else cfg.T,
            "beta_perp": cfg.beta,
            "k_perp": cfg.k,
            "n_pred_samples": args.n_pred_samples,
            "map_loss": map_loss,
            "endpoint_loss": end_loss,
            "endpoint_delta_loss": end_loss - map_loss,
            "hessian_time_sec": hess0_time,
            "spine_geom_time_sec": geom_time,
            "pred_runtime_sec": runtime,
            **metrics,
        }
        print("    " + json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
    return rows


def aggregate(rows, out_csv):
    # Pure standard library aggregation.
    groups = {}
    for r in rows:
        key = (r["task"], r["method"])
        groups.setdefault(key, []).append(r)
    metrics = sorted({k for r in rows for k, v in r.items() if isinstance(v, (float, int)) and k not in {"seed", "T", "k_perp", "n_pred_samples"}})
    out_rows = []
    for (task, method), rs in groups.items():
        o = {"task": task, "method": method, "n": len(rs)}
        for m in metrics:
            vals = np.array([float(r[m]) for r in rs if m in r and r[m] is not None], dtype=float)
            if vals.size:
                o[m + "_mean"] = float(vals.mean())
                o[m + "_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out_rows.append(o)
    fieldnames = sorted({k for r in out_rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["sine", "moons"], choices=["sine", "moons"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir", type=str, default="../results_toy_spine")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--prior-prec", type=float, default=1e-3)
    p.add_argument("--n-pred-samples", type=int, default=250)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    detail_jsonl = os.path.join(args.out_dir, "toy_spine_single_vs_full_detail.jsonl")
    summary_csv = os.path.join(args.out_dir, "toy_spine_single_vs_full_summary.csv")

    if args.quick:
        configs = {
            "sine": ToyConfig("sine", 1, 1, T=3, step_size=0.02, beta=0.005, k=8, train_steps=500, hessian_refresh=3, correct_steps=10, correct_lr=0.05),
            "moons": ToyConfig("moons", 2, 2, T=3, step_size=0.08, beta=0.05, k=8, train_steps=500, hessian_refresh=3, correct_steps=10, correct_lr=0.05),
        }
        args.n_pred_samples = min(args.n_pred_samples, 50)
    else:
        configs = {
            # Legacy fixed toy-spine diagnostic configuration.
            "sine": ToyConfig("sine", 1, 1, T=30, step_size=0.02, beta=0.005, k=30, train_steps=3000, hessian_refresh=10, correct_steps=30, correct_lr=0.05),
            "moons": ToyConfig("moons", 2, 2, T=50, step_size=0.08, beta=0.05, k=30, train_steps=3000, hessian_refresh=10, correct_steps=30, correct_lr=0.05),
        }

    rows = []
    with open(detail_jsonl, "w") as f:
        for seed in args.seeds:
            for task in args.tasks:
                task_rows = run_task(configs[task], seed, args)
                for r in task_rows:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    rows.append(r)

    summary_rows = aggregate(rows, summary_csv)
    print("\nWrote:")
    print(detail_jsonl)
    print(summary_csv)
    print("\nSummary:")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
