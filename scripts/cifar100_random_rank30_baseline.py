import argparse
import copy
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cifar100_all_methods_iclr as base


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def vector_to_trainable_params(vec, model):
    base._trl_vector_to_trainable_parameters(vec, model)


def make_random_orthonormal_basis(num_params, rank, seed, device):
    print(f">>> Building random orthonormal basis: P={num_params}, rank={rank}, device={device}")

    if device.type == "cuda":
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()
    gen.manual_seed(int(seed))

    Q = torch.empty((num_params, rank), device=device, dtype=torch.float32)

    for j in range(rank):
        v = torch.randn(num_params, device=device, dtype=torch.float32, generator=gen)

        for _ in range(2):
            if j > 0:
                coeff = Q[:, :j].T @ v
                v = v - Q[:, :j] @ coeff

        nrm = torch.linalg.norm(v)
        if not torch.isfinite(nrm) or float(nrm) < 1e-12:
            raise RuntimeError(f"Failed to construct basis vector {j}; norm={nrm}")

        Q[:, j] = v / nrm
        print(f"  basis vector {j + 1:02d}/{rank} done")

        del v
        base.cleanup()

    return Q


def compute_random_rayleighs(model, basis, hvp_loader, hvp_batches):
    print(f">>> Computing Rayleigh quotients with {hvp_batches} HVP batch(es)")
    hvp_fn = base.get_hvp_function_ablation(
        model=model,
        loader=hvp_loader,
        device=base.DEVICE,
        num_batches=hvp_batches,
    )

    vals = []
    rank = basis.shape[1]

    for j in range(rank):
        q_cpu = basis[:, j].detach().float().cpu().numpy()
        hq = hvp_fn(q_cpu)
        rq = float(np.dot(q_cpu, hq))
        rq = max(rq, 0.0)
        vals.append(rq)
        print(f"  Rayleigh {j + 1:02d}/{rank}: {rq:.6e}")

        del q_cpu, hq
        base.cleanup()

    return torch.tensor(vals, dtype=torch.float32)


def project_prior_diag(basis, prior_vec):
    device = basis.device
    prior = prior_vec.detach().to(device=device, dtype=basis.dtype)

    vals = []
    for j in range(basis.shape[1]):
        q = basis[:, j]
        vals.append(torch.dot(q * q, prior).detach())
    return torch.stack(vals).float().cpu()


def build_random_posterior(
    cfg,
    model_map,
    base_val,
    clean_loader,
    rank,
    basis_seed,
    basis_device,
    prior_boost,
    prior_floor,
    cache_path,
):
    if cache_path and os.path.exists(cache_path):
        print(f">>> Loading cached random posterior from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        return {
            "basis": payload["basis"].float(),
            "rayleigh": payload["rayleigh"].float(),
            "prior_proj": payload["prior_proj"].float(),
            "inv_sqrt_prec": payload["inv_sqrt_prec"].float(),
        }

    params = trainable_params(model_map)
    theta_map = parameters_to_vector(params).detach()
    num_params = int(theta_map.numel())

    dev = torch.device(basis_device)
    basis = make_random_orthonormal_basis(
        num_params=num_params,
        rank=rank,
        seed=basis_seed,
        device=dev,
    )

    rayleigh = compute_random_rayleighs(
        model=model_map,
        basis=basis,
        hvp_loader=clean_loader,
        hvp_batches=cfg.trl_hvp_batches,
    )

    prior_vec = base.build_trl_prior_from_laplace(
        base_val=base_val,
        model=model_map,
        boost_factor=prior_boost,
        boost_floor=prior_floor,
    )
    prior_proj = project_prior_diag(basis=basis, prior_vec=prior_vec)

    prec = torch.clamp(rayleigh + prior_proj, min=1e-6)
    inv_sqrt_prec = torch.rsqrt(prec)

    payload = {
        "basis": basis.detach().float().cpu(),
        "rayleigh": rayleigh.detach().float().cpu(),
        "prior_proj": prior_proj.detach().float().cpu(),
        "inv_sqrt_prec": inv_sqrt_prec.detach().float().cpu(),
        "rank": int(rank),
        "basis_seed": int(basis_seed),
        "prior_boost": float(prior_boost),
        "prior_floor": float(prior_floor),
        "hvp_batches": int(cfg.trl_hvp_batches),
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        print(f">>> Saving random posterior cache to {cache_path}")
        torch.save(payload, cache_path)

    return {
        "basis": payload["basis"],
        "rayleigh": payload["rayleigh"],
        "prior_proj": payload["prior_proj"],
        "inv_sqrt_prec": payload["inv_sqrt_prec"],
    }


@torch.no_grad()
def predict_random_lowrank(
    model_map,
    posterior,
    loader,
    bn_loader_aug,
    beta,
    n_samples,
    fix_bn_batches,
    fix_bn_mode,
    mc_seed,
    basis_device,
):
    base.set_seed(int(mc_seed))

    model = copy.deepcopy(model_map).to(base.DEVICE)
    model.eval()
    map_vec = parameters_to_vector(trainable_params(model)).detach().to(base.DEVICE)

    basis_cpu = posterior["basis"].float()
    inv_sqrt_cpu = posterior["inv_sqrt_prec"].float()

    if basis_device == "cuda" and torch.cuda.is_available():
        basis = basis_cpu.to(base.DEVICE)
        inv_sqrt = inv_sqrt_cpu.to(base.DEVICE)
    else:
        basis = basis_cpu
        inv_sqrt = inv_sqrt_cpu

    probs_all = []
    targets_all = []
    fixbn_total = 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range(n_samples):
        coeff = torch.randn(inv_sqrt.numel(), device=inv_sqrt.device, dtype=torch.float32)
        coeff = float(beta) * inv_sqrt * coeff

        if basis.device.type == "cuda":
            delta = basis @ coeff
            theta_sample = map_vec + delta.to(base.DEVICE)
        else:
            delta = basis @ coeff.cpu()
            theta_sample = map_vec + delta.to(base.DEVICE)

        vector_to_trainable_params(theta_sample, model)

        elapsed = base.fix_bn(
            model,
            bn_loader_aug,
            base.DEVICE,
            num_batches=fix_bn_batches,
            return_elapsed=True,
            mode=fix_bn_mode,
        )
        fixbn_total += float(elapsed or 0.0)

        batch_probs = []
        for x, y in loader:
            batch_probs.append(torch.softmax(model(x.to(base.DEVICE)), dim=1).cpu())
            if i == 0:
                targets_all.append(y.cpu())

        probs_all.append(torch.cat(batch_probs, dim=0))
        print(".", end="", flush=True)

        del coeff, delta, theta_sample, batch_probs
        base.cleanup()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    print(f"\n  prediction wall={total:.1f}s fixbn={fixbn_total:.1f}s")

    return torch.stack(probs_all, dim=0).mean(0), (torch.cat(targets_all) if targets_all else None), fixbn_total


def nll_from_probs(probs, targets):
    p = probs.clamp(1e-7, 1.0)
    return nn.NLLLoss()(torch.log(p), targets.long()).item()


def run_random_rank_baseline(cfg, args):
    base.set_seed(cfg.seed)
    base.ensure_dir(cfg.ckpt_dir)
    base.ensure_dir(os.path.dirname(args.results) or ".")

    timings = {}

    with base.StageTimer("data", timings):
        tr_aug, bn_clean, val_loader, test_loader, ood_loader = base.get_data(cfg)

    targets_test = base.get_targets(test_loader)

    with base.StageTimer("map_train_or_load", timings):
        model_map = base.load_or_train_map(tr_aug, cfg)

    if args.base_val is None:
        lap_t = {}
        with base.StageTimer("laplace_fit_for_prior", lap_t):
            _, base_val, _, _, _, _ = base.laplace_fit_and_predict(
                model_map=model_map,
                bn_loader_clean=bn_clean,
                ts_loader=test_loader,
                ood_loader=ood_loader,
                cfg=cfg,
                timings=lap_t,
            )
        timings.update({f"prior_{k}": v for k, v in lap_t.items()})
    else:
        base_val = float(args.base_val)
        print(f">>> Using user-provided base_val={base_val:.6f}")

    cache_path = args.cache_path
    if cache_path is None:
        cache_path = os.path.join(
            cfg.ckpt_dir,
            f"random_rank{args.rank}_basis{args.basis_seed}_boost{args.prior_boost:g}_hvp{cfg.trl_hvp_batches}.pt",
        )

    with base.StageTimer("random_basis_and_precision", timings):
        posterior = build_random_posterior(
            cfg=cfg,
            model_map=model_map,
            base_val=base_val,
            clean_loader=bn_clean,
            rank=args.rank,
            basis_seed=args.basis_seed,
            basis_device=args.basis_device,
            prior_boost=args.prior_boost,
            prior_floor=args.prior_floor,
            cache_path=cache_path,
        )

    print("\n>>> Random-rank validation beta sweep")
    val_targets = base.get_targets(val_loader)
    sweep = []
    best_beta = None
    best_val_nll = float("inf")

    with base.StageTimer("random_validation_scale_sweep", timings):
        for beta in cfg.trl_tube_scales:
            print(f"  beta={float(beta):g}")
            probs_val, _, fixbn_sec = predict_random_lowrank(
                model_map=model_map,
                posterior=posterior,
                loader=val_loader,
                bn_loader_aug=tr_aug,
                beta=float(beta),
                n_samples=cfg.trl_val_samples,
                fix_bn_batches=cfg.trl_fixbn_batches,
                fix_bn_mode=cfg.trl_fixbn_mode,
                mc_seed=args.val_mc_seed,
                basis_device=args.basis_device,
            )
            val_nll = nll_from_probs(probs_val, val_targets)
            sweep.append((float(beta), float(val_nll)))
            print(f"  beta={float(beta):g} val_nll={val_nll:.6f}")

            if np.isfinite(val_nll) and val_nll < best_val_nll:
                best_val_nll = float(val_nll)
                best_beta = float(beta)

            timings[f"random_val_fixbn_beta_{float(beta):g}"] = {
                "wall_sec": float(fixbn_sec),
                "peak_vram_gb": 0.0,
            }
            base.cleanup()

    if best_beta is None:
        raise RuntimeError("No valid beta selected in random-rank sweep.")

    print(f"\n>>> Selected random-rank beta={best_beta:g} by val NLL={best_val_nll:.6f}")

    with base.StageTimer("random_test_prediction", timings):
        probs_test, _, fixbn_test = predict_random_lowrank(
            model_map=model_map,
            posterior=posterior,
            loader=test_loader,
            bn_loader_aug=tr_aug,
            beta=best_beta,
            n_samples=cfg.trl_val_samples,
            fix_bn_batches=cfg.trl_fixbn_batches,
            fix_bn_mode=cfg.trl_fixbn_mode,
            mc_seed=args.test_mc_seed,
            basis_device=args.basis_device,
        )
    timings["random_test_fixbn_overhead"] = {
        "wall_sec": float(fixbn_test),
        "peak_vram_gb": 0.0,
    }

    with base.StageTimer("random_ood_prediction", timings):
        probs_ood, _, fixbn_ood = predict_random_lowrank(
            model_map=model_map,
            posterior=posterior,
            loader=ood_loader,
            bn_loader_aug=tr_aug,
            beta=best_beta,
            n_samples=cfg.trl_val_samples,
            fix_bn_batches=cfg.trl_fixbn_batches,
            fix_bn_mode=cfg.trl_fixbn_mode,
            mc_seed=args.ood_mc_seed,
            basis_device=args.basis_device,
        )
    timings["random_ood_fixbn_overhead"] = {
        "wall_sec": float(fixbn_ood),
        "peak_vram_gb": 0.0,
    }

    acc, nll, ece, brier = base.calc_metrics(probs_test, targets_test, cfg.num_classes)
    auroc = base.auroc_entropy(probs_test, probs_ood)

    row = {
        "dataset": "CIFAR-100",
        "architecture": "ResNet-18-CIFAR",
        "method": f"RandomRank{args.rank}",
        "seed": int(cfg.seed),
        "acc": float(acc),
        "nll": float(nll),
        "ece": float(ece),
        "brier": float(brier),
        "auroc": float(auroc),
        "random_rank": int(args.rank),
        "random_basis_seed": int(args.basis_seed),
        "best_tube_scale": float(best_beta),
        "best_val_nll": float(best_val_nll),
        "tube_scale_sweep": sweep,
        "prior_base_val": float(base_val),
        "prior_boost": float(args.prior_boost),
        "prior_floor": float(args.prior_floor),
        "samples": int(cfg.trl_val_samples),
        "fixbn_batches": int(cfg.trl_fixbn_batches),
        "trl_fixbn_mode": cfg.trl_fixbn_mode,
        "hvp_batches": int(cfg.trl_hvp_batches),
        "rayleigh_mean": float(posterior["rayleigh"].mean().item()),
        "rayleigh_max": float(posterior["rayleigh"].max().item()),
        "prior_proj_mean": float(posterior["prior_proj"].mean().item()),
        "prior_proj_max": float(posterior["prior_proj"].max().item()),
        "runtime_total_sec": float(sum(v.get("wall_sec", 0.0) for v in timings.values())),
        "peak_vram_gb": float(max([v.get("peak_vram_gb", 0.0) for v in timings.values()] + [0.0])),
    }
    row.update(base.flatten_timings("time", timings))

    print("\n" + "=" * 100)
    print("RANDOM FULL-NETWORK LOW-RANK BASELINE")
    print("=" * 100)
    print(
        f"{row['method']:14s} seed={row['seed']} "
        f"acc={100*row['acc']:.2f} nll={row['nll']:.4f} "
        f"ece={row['ece']:.4f} brier={row['brier']:.4f} "
        f"auroc={row['auroc']:.4f} beta={row['best_tube_scale']}"
    )

    base.append_jsonl(args.results, row)
    print(f"\nWrote JSONL row to {args.results}")

    return [row]


def parse_args():
    p = argparse.ArgumentParser(description="Random rank-k full-network posterior baseline.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=str, default="results/cifar100_random_rank30.jsonl")
    p.add_argument("--ckpt-dir", type=str, default=None)

    p.add_argument("--rank", type=int, default=30)
    p.add_argument("--basis-seed", type=int, default=None)
    p.add_argument(
        "--basis-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    p.add_argument("--cache-path", type=str, default=None)

    p.add_argument("--prior-boost", type=float, default=50.0)
    p.add_argument("--prior-floor", type=float, default=5.0)
    p.add_argument("--base-val", type=float, default=None)

    p.add_argument("--trl-tube-scales", type=float, nargs="*", default=None)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--fixbn-batches", type=int, default=25)
    p.add_argument(
        "--fixbn-mode", choices=["rolling", "reset"], default="rolling",
        help="BatchNorm refresh mode; rolling reproduces the reported control.",
    )
    p.add_argument("--hvp-batches", type=int, default=5)

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--epochs-map", type=int, default=None)

    p.add_argument("--val-mc-seed", type=int, default=1000)
    p.add_argument("--test-mc-seed", type=int, default=2000)
    p.add_argument("--ood-mc-seed", type=int, default=3000)

    p.add_argument("--quick", action="store_true", help="Smoke test only; not for paper.")
    return p.parse_args()


def cfg_from_args(args):
    cfg = base.CFG()
    cfg.seed = int(args.seed)
    cfg.ckpt_dir = args.ckpt_dir or f"./checkpoints_c100_seed{args.seed}"

    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.epochs_map is not None:
        cfg.epochs_map = int(args.epochs_map)

    cfg.trl_k_perp = int(args.rank)
    cfg.trl_val_samples = int(args.samples)
    cfg.trl_fixbn_batches = int(args.fixbn_batches)
    cfg.trl_fixbn_mode = args.fixbn_mode
    cfg.trl_hvp_batches = int(args.hvp_batches)

    if args.trl_tube_scales is not None and len(args.trl_tube_scales) > 0:
        cfg.trl_tube_scales = tuple(float(x) for x in args.trl_tube_scales)

    if args.quick:
        cfg.epochs_map = min(cfg.epochs_map, 1)
        cfg.trl_k_perp = min(cfg.trl_k_perp, 3)
        cfg.trl_val_samples = min(cfg.trl_val_samples, 2)
        cfg.trl_fixbn_batches = min(cfg.trl_fixbn_batches, 1)
        cfg.trl_hvp_batches = min(cfg.trl_hvp_batches, 1)
        cfg.trl_tube_scales = (1.0,)

    return cfg


if __name__ == "__main__":
    args = parse_args()
    if args.basis_seed is None:
        args.basis_seed = 9000 + int(args.seed)

    if args.basis_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--basis-device cuda requested but CUDA is not available.")

    cfg = cfg_from_args(args)
    run_random_rank_baseline(cfg, args)
