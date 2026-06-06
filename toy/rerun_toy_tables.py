#!/usr/bin/env python3
"""
Re-run the toy experiments that generated the paper toy tables.

This script is a cleaned, multi-seed version of the original notebook.  It runs
both toy tasks under one protocol and writes consistent per-seed, aggregate and
LaTeX tables for:

  * Table 3-style sine regression comparison: ELA / LLA / TRL-full-spine
  * Table 4-style two-moons comparison: ELA / LLA / TRL-full-spine
  * Table 5-style spine isolation: TRL-single-checkpoint vs TRL-full-spine

Key fixes relative to the notebook:

  * all metrics come from the same saved posterior samples;
  * ELA/LLA samples are not accidentally overwritten;
  * standard deviations are computed across seeds;
  * TRL-full-spine and TRL-single-checkpoint use the same selected configuration;
  * toy ELA/LLA are full-parameter/full-Hessian baselines.

Example:

  source /mnt/hd2/rpdavid/envs/trl-iclr/bin/activate
  cd /mnt/hd2/rpdavid
  python rerun_toy_tables.py \
    --out-dir /mnt/hd2/rpdavid/results_toy_tables_rerun \
    --seeds 0 1 2 3 4 5 6 7 8 9

For a quick smoke test:

  python rerun_toy_tables.py --out-dir /tmp/toy_smoke --seeds 0 --quick
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons

try:
    from laplace import Laplace
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import laplace. Activate the TRL environment and install laplace-torch."
    ) from exc


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep deterministic flags off by default for speed; set them externally if needed.


def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def flatten_trainable(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector([p for p in model.parameters() if p.requires_grad])


def set_trainable_from_vector(model: nn.Module, theta: torch.Tensor) -> None:
    vector_to_parameters(theta, [p for p in model.parameters() if p.requires_grad])


def grads_to_vector(grads: Iterable[Optional[torch.Tensor]], params: List[torch.nn.Parameter]) -> torch.Tensor:
    vecs = []
    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.reshape(-1))
    return torch.cat(vecs)


def model_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stable_cov_factor_from_precision(H: torch.Tensor, jitter: float = 1e-4) -> torch.Tensor:
    """Return L with covariance approximately L L^T = inv(H).

    Uses symmetric eigendecomposition with eigenvalue clipping. This is more stable
    than Cholesky for small toy Hessians whose empirical curvature may be nearly
    singular or mildly indefinite after numerical noise.
    """
    device = H.device
    dtype = H.dtype
    H = 0.5 * (H + H.T)
    H = H + jitter * torch.eye(H.shape[0], device=device, dtype=dtype)
    vals, vecs = torch.linalg.eigh(H)
    vals = torch.clamp(vals, min=jitter)
    return vecs @ torch.diag(torch.rsqrt(vals))


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def fmt_ms(mean: float, std: float, ndigits: int = 3) -> str:
    if math.isnan(mean):
        return "--"
    return f"{mean:.{ndigits}f} $\\pm$ {std:.{ndigits}f}"


# -----------------------------------------------------------------------------
# TRL toy engine: cleaned version of notebook's trl_from_laplace.py
# -----------------------------------------------------------------------------


@dataclass
class TRLPathPoint:
    theta: torch.Tensor
    v_parallel: torch.Tensor
    N: torch.Tensor
    L_perp: torch.Tensor
    lambdas_perp: torch.Tensor


@dataclass
class TRLConfig:
    n_steps: int = 50
    step_size: float = 0.02
    correction_lr: float = 0.10
    k_perp: int = 30
    jitter: float = 1e-4
    perp_scale: float = 1.0


def decompose_hessian(H: torch.Tensor, k_perp: int, jitter: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valley direction, top transverse eigenvectors, and transverse eigenvalues."""
    device = H.device
    H_sym = 0.5 * (H + H.T) + jitter * torch.eye(H.shape[0], device=device, dtype=H.dtype)
    vals, vecs = torch.linalg.eigh(H_sym)
    vals = torch.clamp(vals, min=jitter)

    v_parallel = vecs[:, 0]
    n_params = vals.numel()
    k = min(int(k_perp), n_params - 1)
    top_idx = torch.arange(n_params - 1, n_params - 1 - k, -1, device=device)
    lambdas = vals[top_idx]
    N = vecs[:, top_idx]
    return v_parallel, N, lambdas


def full_loss_and_grad(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, torch.Tensor]:
    """Average loss and gradient over the loader."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_n = 0
    total_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if yb.ndim > 1 and yb.shape[-1] == 1:
            yb = yb.squeeze(-1)
        loss = loss_fn(out, yb)
        loss.backward()
        total_loss += float(loss.item()) * xb.shape[0]
        total_n += xb.shape[0]
        total_batches += 1

    # The losses above are batch means. The original notebook divides gradients by
    # the number of batches. We keep that convention for consistency.
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad.detach().clone() / max(1, total_batches))
    return total_loss / max(1, total_n), parameters_to_vector(grads)


class HessianTRL:
    """Toy full-Hessian TRL approximation."""

    def __init__(
        self,
        base_model: nn.Module,
        la,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        config: TRLConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.model = copy.deepcopy(base_model).to(device)
        self.la = la
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.config = config
        self.theta0 = la.mean.detach().to(device) if hasattr(la, "mean") else flatten_trainable(base_model).detach().to(device)
        self.H_map = la.posterior_precision.detach().to(device)
        self.path: List[TRLPathPoint] = []
        self._build_path()

    def _build_path(self) -> None:
        cfg = self.config
        v_curr, N_curr, lambdas = decompose_hessian(self.H_map, cfg.k_perp, cfg.jitter)
        theta_curr = self.theta0.clone()
        for _ in range(cfg.n_steps):
            L_curr = torch.diag(cfg.perp_scale * torch.rsqrt(lambdas + 1e-6))
            self.path.append(
                TRLPathPoint(
                    theta=theta_curr.detach().clone(),
                    v_parallel=v_curr.detach().clone(),
                    N=N_curr.detach().clone(),
                    L_perp=L_curr.detach().clone(),
                    lambdas_perp=lambdas.detach().clone(),
                )
            )

            pred = theta_curr + cfg.step_size * v_curr
            set_trainable_from_vector(self.model, pred)
            _, grad = full_loss_and_grad(self.model, self.train_loader, self.device, self.loss_fn)
            g_perp = grad - torch.dot(grad, v_curr) * v_curr
            theta_next = pred - cfg.correction_lr * g_perp

            delta = theta_next - theta_curr
            dist = delta.norm()
            v_new = v_curr if dist < 1e-9 else delta / dist
            overlaps = v_new @ N_curr
            N_transported = N_curr - torch.outer(v_new, overlaps)
            N_curr, _ = torch.linalg.qr(N_transported, mode="reduced")
            theta_curr = theta_next.detach().clone()
            v_curr = v_new.detach().clone()

    @torch.no_grad()
    def predict_samples(
        self,
        x: torch.Tensor,
        n_samples: int,
        mode: str = "full",
        idx: int = 0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Return samples of model output/logit with shape (S,N).

        mode="full" samples a checkpoint index uniformly from the stored spine.
        mode="single" uses path[idx] only but still samples transverse directions.
        mode="spine_only" samples checkpoints without transverse perturbation.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        x = x.to(self.device)
        samples = []
        if mode == "full":
            indices = torch.randint(0, len(self.path), (n_samples,))
        elif mode in {"single", "spine_only"}:
            indices = torch.full((n_samples,), int(idx), dtype=torch.long)
        else:
            raise ValueError(f"Unknown TRL prediction mode: {mode}")

        for i in indices:
            pt = self.path[int(i)]
            if mode == "spine_only":
                theta = pt.theta.to(self.device)
            else:
                z = torch.randn(pt.L_perp.shape[0], device=self.device)
                theta = pt.theta.to(self.device) + pt.N.to(self.device) @ (pt.L_perp.to(self.device) @ z)
            set_trainable_from_vector(self.model, theta)
            out = self.model(x)
            if out.ndim > 1 and out.shape[-1] == 1:
                out = out.squeeze(-1)
            samples.append(out.detach().clone())
        return torch.stack(samples, dim=0)


# -----------------------------------------------------------------------------
# Sine regression
# -----------------------------------------------------------------------------


class MLP1D(nn.Module):
    def __init__(self, hidden_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_sine_data(seed: int, n_train: int = 80, n_test: int = 200, noise_std: float = 0.3) -> Tuple[TensorDataset, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x_train = torch.linspace(-3.0, 3.0, n_train).unsqueeze(-1)
    y_train = torch.sin(x_train) + noise_std * torch.randn(x_train.shape, generator=gen)
    x_test = torch.linspace(-6.0, 6.0, n_test).unsqueeze(-1)
    y_test = torch.sin(x_test).squeeze(-1)
    return TensorDataset(x_train, y_train), x_test, y_test


def train_regression_map(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float = 1e-3, wd: float = 1e-4) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).squeeze(-1), yb.squeeze(-1))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def sample_ela_regression(la, base_model: nn.Module, x: torch.Tensor, n_samples: int, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean = la.mean.detach().to(device)
    H = la.posterior_precision.detach().to(device)
    L = stable_cov_factor_from_precision(H, jitter=1e-6)
    model = copy.deepcopy(base_model).to(device)
    preds = []
    x = x.to(device)
    for _ in range(n_samples):
        eps = torch.randn(len(mean), device=device)
        theta = mean + L @ eps
        set_trainable_from_vector(model, theta)
        preds.append(model(x).squeeze(-1).detach().clone())
    return torch.stack(preds, dim=0)


def sample_lla_regression(la, base_model: nn.Module, x: torch.Tensor, n_samples: int, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean = la.mean.detach().to(device)
    H = la.posterior_precision.detach().to(device)
    L = stable_cov_factor_from_precision(H, jitter=1e-6)
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        f_map = model(x).squeeze(-1)
    params = [p for p in model.parameters() if p.requires_grad]
    J = []
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        yi = model(x[i:i+1]).squeeze(-1)
        yi.backward(retain_graph=True)
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        J.append(parameters_to_vector(grads))
    J = torch.stack(J, dim=0)
    preds = []
    for _ in range(n_samples):
        eps = torch.randn(len(mean), device=device)
        delta = L @ eps
        preds.append((f_map + J @ delta).detach().clone())
    return torch.stack(preds, dim=0)


def regression_metrics(pred_samples: torch.Tensor, y_true: torch.Tensor, obs_noise_std: float = 0.0) -> Dict[str, float]:
    pred_samples = pred_samples.float()
    y = y_true.to(pred_samples.device).float()
    mean = pred_samples.mean(dim=0)

    # Epistemic/function variance from posterior samples.
    function_var = pred_samples.var(dim=0, unbiased=False)

    # Predictive variance for regression NLL/coverage includes observation noise.
    var = function_var + float(obs_noise_std) ** 2 + 1e-6

    rmse = torch.sqrt(torch.mean((mean - y).pow(2))).item()
    nll = (0.5 * (torch.log(2 * math.pi * var) + (y - mean).pow(2) / var)).mean().item()
    z = (y - mean) / torch.sqrt(var)
    return {
        "rmse": rmse,
        "nll": nll,
        "z_mean": z.mean().item(),
        "z_var": z.var(unbiased=False).item(),
        "coverage_1": (z.abs() <= 1.0).float().mean().item(),
        "coverage_2": (z.abs() <= 2.0).float().mean().item(),
        "coverage_3": (z.abs() <= 3.0).float().mean().item(),
        "avg_function_var": function_var.mean().item(),
    }


def run_sine_seed(args, seed: int, device: torch.device) -> List[Dict]:
    set_seed(seed)
    train_ds, x_test, y_test = make_sine_data(seed, noise_std=args.sine_noise)
    loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = train_regression_map(MLP1D(), loader, device, epochs=args.sine_epochs)
    la = Laplace(model, likelihood="regression", subset_of_weights="all", hessian_structure="full")
    la.fit(loader)
    la.optimize_prior_precision(method="marglik")

    cfg = TRLConfig(
        n_steps=args.sine_trl_steps,
        step_size=args.sine_trl_step_size,
        correction_lr=args.sine_trl_correction_lr,
        k_perp=args.sine_trl_k,
        perp_scale=args.sine_trl_perp_scale,
        jitter=args.trl_jitter,
    )
    trl = HessianTRL(model, la, loader, nn.MSELoss(reduction="mean"), cfg, device)

    rows = []
    samples_by_method = {
        "ELA": sample_ela_regression(la, model, x_test, args.samples, device, seed * 1000 + 11),
        "LLA": sample_lla_regression(la, model, x_test, args.samples, device, seed * 1000 + 12),
        "TRL-full-spine": trl.predict_samples(x_test, args.samples, mode="full", seed=seed * 1000 + 13),
        "TRL-single-checkpoint": trl.predict_samples(x_test, args.samples, mode="single", idx=0, seed=seed * 1000 + 14),
    }
    for method, samples in samples_by_method.items():
        m = regression_metrics(samples.cpu(), y_test.cpu(), obs_noise_std=args.sine_noise)
        m.update({"task": "sine", "method": method, "seed": seed})
        rows.append(m)
    return rows


# -----------------------------------------------------------------------------
# Two moons classification
# -----------------------------------------------------------------------------


class MLPBinary(nn.Module):
    def __init__(self, hidden_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_twomoons_data(
    seed: int,
    n_train: int = 500,
    n_test: int = 2000,
    noise: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Xtr, ytr = make_moons(n_samples=n_train, noise=noise, random_state=seed)
    Xte, yte = make_moons(n_samples=n_test, noise=noise, random_state=seed + 12345)

    Xtr = Xtr.astype("float32")
    Xte = Xte.astype("float32")

    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    return (
        torch.from_numpy(Xtr).float(),
        torch.from_numpy(ytr).float(),
        torch.from_numpy(Xte).float(),
        torch.from_numpy(yte).float(),
    )


def train_binary_map(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float = 1e-3, wd: float = 1e-4) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def sample_ela_probs(la, base_model: nn.Module, x: torch.Tensor, n_samples: int, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean = la.mean.detach().to(device)
    H = la.posterior_precision.detach().to(device)
    L = stable_cov_factor_from_precision(H, jitter=1e-4)
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    x = x.to(device)
    probs = []
    for _ in range(n_samples):
        eps = torch.randn(len(mean), device=device)
        theta = mean + L @ eps
        set_trainable_from_vector(model, theta)
        probs.append(torch.sigmoid(model(x)).detach().clone())
    return torch.stack(probs, dim=0)


def sample_lla_probs(la, base_model: nn.Module, x: torch.Tensor, n_samples: int, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean = la.mean.detach().to(device)
    H = la.posterior_precision.detach().to(device)
    L = stable_cov_factor_from_precision(H, jitter=1e-4)
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        logits_map = model(x).squeeze(-1)
    params = [p for p in model.parameters() if p.requires_grad]
    J = []
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        out = model(x[i:i+1]).squeeze(-1)
        out.backward(retain_graph=True)
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        J.append(parameters_to_vector(grads))
    J = torch.stack(J, dim=0)
    probs = []
    for _ in range(n_samples):
        eps = torch.randn(len(mean), device=device)
        delta = L @ eps
        probs.append(torch.sigmoid(logits_map + J @ delta).detach().clone())
    return torch.stack(probs, dim=0)


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = p.double().clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def classification_metrics(prob_samples: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    eps = 1e-6
    prob_samples = prob_samples.double()
    y = y_true.to(prob_samples.device).double()
    p_raw = prob_samples.mean(dim=0)
    p = p_raw.clamp(eps, 1.0 - eps)

    nll = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).mean().item()
    brier = ((p_raw - y).pow(2)).mean().item()
    acc = ((p_raw >= 0.5).double() == y).double().mean().item()

    return {
        "nll": nll,
        "brier": brier,
        "acc": acc,
        "entropy": binary_entropy(p_raw).mean().item(),
        "avg_function_var": prob_samples.var(dim=0, unbiased=False).mean().item(),
    }


def run_twomoons_seed(args, seed: int, device: torch.device) -> List[Dict]:
    set_seed(seed)
    Xtr, ytr, Xte, yte = make_twomoons_data(
        seed,
        n_train=args.moons_n_train,
        n_test=args.moons_n_test,
        noise=args.moons_noise,
    )
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=min(args.moons_batch_size, len(Xtr)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = train_binary_map(
        MLPBinary(hidden_dim=args.moons_hidden),
        loader,
        device,
        epochs=args.moons_epochs,
    )
    la = Laplace(model, likelihood="classification", subset_of_weights="all", hessian_structure="full")
    la.fit(loader)
    la.optimize_prior_precision(method="marglik")

    cfg = TRLConfig(
        n_steps=args.moons_trl_steps,
        step_size=args.moons_trl_step_size,
        correction_lr=args.moons_trl_correction_lr,
        k_perp=args.moons_trl_k,
        perp_scale=args.moons_trl_perp_scale,
        jitter=args.trl_jitter,
    )
    trl = HessianTRL(model, la, loader, nn.BCEWithLogitsLoss(reduction="mean"), cfg, device)

    logits_full = trl.predict_samples(Xte, args.samples, mode="full", seed=seed * 1000 + 23)
    logits_single = trl.predict_samples(Xte, args.samples, mode="single", idx=0, seed=seed * 1000 + 24)

    samples_by_method = {
        "ELA": sample_ela_probs(la, model, Xte, args.samples, device, seed * 1000 + 21),
        "LLA": sample_lla_probs(la, model, Xte, args.samples, device, seed * 1000 + 22),
        "TRL-full-spine": torch.sigmoid(logits_full).detach().cpu(),
        "TRL-single-checkpoint": torch.sigmoid(logits_single).detach().cpu(),
    }
    rows = []
    for method, samples in samples_by_method.items():
        m = classification_metrics(samples.cpu(), yte.cpu())
        m.update({"task": "two_moons", "method": method, "seed": seed})
        rows.append(m)
    return rows


# -----------------------------------------------------------------------------
# Aggregation and LaTeX tables
# -----------------------------------------------------------------------------


def summarize(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        grouped.setdefault((r["task"], r["method"]), []).append(r)
    metrics = sorted(k for k in {k for r in rows for k in r.keys()} if k not in {"task", "method", "seed"})
    out = []
    for (task, method), rs in sorted(grouped.items()):
        row = {"task": task, "method": method, "n_seeds": len(rs)}
        for m in metrics:
            vals = [float(r[m]) for r in rs if m in r and r[m] not in (None, "") and math.isfinite(float(r[m]))]
            if vals:
                row[f"{m}_mean"], row[f"{m}_std"] = mean_std(vals)
        out.append(row)
    return out


def get_summary(summary: List[Dict], task: str, method: str, metric: str) -> Tuple[float, float]:
    for r in summary:
        if r["task"] == task and r["method"] == method:
            return r.get(f"{metric}_mean", float("nan")), r.get(f"{metric}_std", float("nan"))
    return float("nan"), float("nan")


def latex_table3_sine(summary: List[Dict]) -> str:
    methods = ["ELA", "LLA", "TRL-full-spine"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Noisy sine regression toy. Values are mean $\pm$ standard deviation over seeds.}",
        r"\label{tab:toy-sine-rerun}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & RMSE $\downarrow$ & NLL $\downarrow$ & $\mathrm{Var}(z)$ & Cov. 1$\sigma$ & Cov. 2$\sigma$ & Func. var. $\uparrow$ \\",
        r"\midrule",
    ]
    for method in methods:
        rmse = fmt_ms(*get_summary(summary, "sine", method, "rmse"), 3)
        nll = fmt_ms(*get_summary(summary, "sine", method, "nll"), 3)
        zvar = fmt_ms(*get_summary(summary, "sine", method, "z_var"), 3)
        c1 = fmt_ms(*get_summary(summary, "sine", method, "coverage_1"), 3)
        c2 = fmt_ms(*get_summary(summary, "sine", method, "coverage_2"), 3)
        fv = fmt_ms(*get_summary(summary, "sine", method, "avg_function_var"), 4)
        label = method.replace("TRL-full-spine", "TRL")
        lines.append(f"{label} & {rmse} & {nll} & {zvar} & {c1} & {c2} & {fv} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def latex_table4_twomoons(summary: List[Dict]) -> str:
    methods = ["ELA", "LLA", "TRL-full-spine"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Two-moons classification toy. Values are mean $\pm$ standard deviation over seeds.}",
        r"\label{tab:toy-twomoons-rerun}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Acc. $\uparrow$ & NLL $\downarrow$ & Brier $\downarrow$ & Entropy & Func. var. $\uparrow$ \\",
        r"\midrule",
    ]
    for method in methods:
        acc = fmt_ms(*get_summary(summary, "two_moons", method, "acc"), 3)
        nll = fmt_ms(*get_summary(summary, "two_moons", method, "nll"), 3)
        brier = fmt_ms(*get_summary(summary, "two_moons", method, "brier"), 3)
        ent = fmt_ms(*get_summary(summary, "two_moons", method, "entropy"), 3)
        fv = fmt_ms(*get_summary(summary, "two_moons", method, "avg_function_var"), 4)
        label = method.replace("TRL-full-spine", "TRL")
        lines.append(f"{label} & {acc} & {nll} & {brier} & {ent} & {fv} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def latex_table5_spine(summary: List[Dict]) -> str:
    rows = [
        ("Sine", "TRL-single-checkpoint", "sine", "TRL-single-checkpoint"),
        ("Sine", "TRL-full-spine", "sine", "TRL-full-spine"),
        ("Two-moons", "TRL-single-checkpoint", "two_moons", "TRL-single-checkpoint"),
        ("Two-moons", "TRL-full-spine", "two_moons", "TRL-full-spine"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Toy spine isolation rerun. Values are mean $\pm$ standard deviation over seeds.}",
        r"\label{tab:toy-spine-isolation-rerun}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Task & Method & NLL $\downarrow$ & RMSE / Acc. & Brier $\downarrow$ & Func. var. $\uparrow$ \\",
        r"\midrule",
    ]
    for task_label, method_label, task, method in rows:
        nll = fmt_ms(*get_summary(summary, task, method, "nll"), 3)
        if task == "sine":
            main = fmt_ms(*get_summary(summary, task, method, "rmse"), 3)
            brier = "--"
        else:
            main = fmt_ms(*get_summary(summary, task, method, "acc"), 3)
            brier = fmt_ms(*get_summary(summary, task, method, "brier"), 3)
        fv = fmt_ms(*get_summary(summary, task, method, "avg_function_var"), 4)
        lines.append(f"{task_label} & {method_label} & {nll} & {main} & {brier} & {fv} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def paired_counts(rows: List[Dict]) -> List[Dict]:
    out = []
    by_seed_task: Dict[Tuple[str, int], Dict[str, Dict]] = {}
    for r in rows:
        by_seed_task.setdefault((r["task"], int(r["seed"])), {})[r["method"]] = r
    comparisons = [
        ("sine", "TRL-full-spine", "TRL-single-checkpoint", ["nll", "rmse", "avg_function_var"]),
        ("two_moons", "TRL-full-spine", "TRL-single-checkpoint", ["nll", "acc", "brier", "entropy", "avg_function_var"]),
    ]
    for task, a, b, metrics in comparisons:
        seeds = sorted(seed for (t, seed), d in by_seed_task.items() if t == task and a in d and b in d)
        for metric in metrics:
            vals = []
            better = 0
            higher_is_better = metric in {"acc", "entropy", "avg_function_var"}
            for seed in seeds:
                ra = by_seed_task[(task, seed)][a]
                rb = by_seed_task[(task, seed)][b]
                diff = float(ra[metric]) - float(rb[metric])
                vals.append(diff)
                if (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
                    better += 1
            m, s = mean_std(vals)
            out.append({
                "task": task,
                "method_a": a,
                "method_b": b,
                "metric": metric,
                "diff_a_minus_b_mean": m,
                "diff_a_minus_b_std": s,
                "a_better_count": better,
                "n": len(seeds),
            })
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-run toy ELA/LLA/TRL tables with seed-level aggregation.")
    p.add_argument("--out-dir", type=str, default="results_toy_tables_rerun")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--task", choices=["all", "sine", "two_moons"], default="all")
    p.add_argument("--device", default="auto")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--quick", action="store_true", help="Fast smoke test with fewer epochs/samples.")

    p.add_argument("--sine-noise", type=float, default=0.3)
    p.add_argument("--sine-epochs", type=int, default=600)
    p.add_argument("--sine-trl-steps", type=int, default=30)
    p.add_argument("--sine-trl-step-size", type=float, default=0.02)
    p.add_argument("--sine-trl-correction-lr", type=float, default=0.10)
    p.add_argument("--sine-trl-perp-scale", type=float, default=0.005)
    p.add_argument("--sine-trl-k", type=int, default=30)

    p.add_argument("--moons-n", type=int, default=300)  # deprecated; kept for compatibility
    p.add_argument("--moons-n-train", type=int, default=500)
    p.add_argument("--moons-n-test", type=int, default=2000)
    p.add_argument("--moons-hidden", type=int, default=50)
    p.add_argument("--moons-noise", type=float, default=0.1)
    p.add_argument("--moons-batch-size", type=int, default=128)
    p.add_argument("--moons-epochs", type=int, default=500)
    p.add_argument("--moons-trl-steps", type=int, default=50)
    p.add_argument("--moons-trl-step-size", type=float, default=0.08)
    p.add_argument("--moons-trl-correction-lr", type=float, default=0.10)
    p.add_argument("--moons-trl-perp-scale", type=float, default=0.05)
    p.add_argument("--moons-trl-k", type=int, default=30)

    p.add_argument("--trl-jitter", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.samples = min(args.samples, 25)
        args.sine_epochs = min(args.sine_epochs, 25)
        args.moons_epochs = min(args.moons_epochs, 25)
        args.sine_trl_steps = min(args.sine_trl_steps, 5)
        args.moons_trl_steps = min(args.moons_trl_steps, 5)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {out_dir.resolve()}")

    all_rows: List[Dict] = []
    for seed in args.seeds:
        print("\n" + "=" * 90)
        print(f"Seed {seed}")
        print("=" * 90, flush=True)
        if args.task in {"all", "sine"}:
            print("Running sine regression...", flush=True)
            rows = run_sine_seed(args, seed, device)
            write_csv(out_dir / "toy_results_per_seed.partial.csv", all_rows + rows)
            all_rows.extend(rows)
            for r in rows:
                print(json.dumps(r, sort_keys=True), flush=True)
        if args.task in {"all", "two_moons"}:
            print("Running two moons classification...", flush=True)
            rows = run_twomoons_seed(args, seed, device)
            write_csv(out_dir / "toy_results_per_seed.partial.csv", all_rows + rows)
            all_rows.extend(rows)
            for r in rows:
                print(json.dumps(r, sort_keys=True), flush=True)

    write_csv(out_dir / "toy_results_per_seed.csv", all_rows)
    summary = summarize(all_rows)
    write_csv(out_dir / "toy_results_summary.csv", summary)
    pc = paired_counts(all_rows)
    write_csv(out_dir / "toy_paired_counts.csv", pc)

    (out_dir / "table3_sine.tex").write_text(latex_table3_sine(summary))
    (out_dir / "table4_twomoons.tex").write_text(latex_table4_twomoons(summary))
    (out_dir / "table5_spine_isolation.tex").write_text(latex_table5_spine(summary))

    metadata = {
        "args": vars(args),
        "device": str(device),
        "note": "Generated by rerun_toy_tables.py. Tables 3/4/5 use the same seed-level samples and aggregation protocol.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print("\nWrote:")
    for fn in [
        "toy_results_per_seed.csv",
        "toy_results_summary.csv",
        "toy_paired_counts.csv",
        "table3_sine.tex",
        "table4_twomoons.tex",
        "table5_spine_isolation.tex",
        "run_metadata.json",
    ]:
        print(f"  {out_dir / fn}")


if __name__ == "__main__":
    main()
