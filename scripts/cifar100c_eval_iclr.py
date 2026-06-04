#!/usr/bin/env python
"""Evaluate CIFAR-100-C robustness/calibration for the ICLR revision.

This script is intentionally aligned with the revised paper narrative: the TRL
implementation is evaluated as a practical discrete-spine approximation with a
transported low-rank transverse structure, not as a smooth continuous tubular
pushforward.

Expected CIFAR-100-C layout:
  <cifar100c_root>/labels.npy
  <cifar100c_root>/gaussian_noise.npy
  <cifar100c_root>/shot_noise.npy
  ...
Each corruption array contains 5 severities concatenated along axis 0.

Recommended paper table methods: map trl deepens swag. You can also include
ela/lla/mcdo for completeness.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))

from trl_iclr_utils.experiment_io import StageTimer, append_jsonl, flatten_timings  # noqa: E402
from cifar100_all_methods_iclr import (  # noqa: E402
    CFG,
    DEVICE,
    ResNetCIFAR,
    SWAG,
    build_trl_prior_from_laplace,
    calc_metrics,
    fix_bn,
    get_data,
    laplace_fit_and_predict,
    load_or_train_map,
    load_or_train_mcdo,
    mc_dropout_predict,
    predict_probs,
    PracticalTRLStage2,
    run_swag,
    set_seed,
)


class CIFAR100CDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, corruption: str, severity: int, transform=None):
        root = Path(root)
        arr_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        if not arr_path.exists():
            raise FileNotFoundError(arr_path)
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        x = np.load(arr_path)
        y = np.load(labels_path)
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in {1,2,3,4,5}")
        start = (severity - 1) * 10000
        end = severity * 10000
        self.x = x[start:end]
        self.y = y[:10000] if len(y) == 10000 else y[start:end]
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.fromarray(self.x[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.y[idx])


def laplace_predict_loader(la, loader, pred_type: str, n_samples: int = 25):
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(next(la.model.parameters()).device)
            if pred_type == "nn":
                try:
                    p = la(x, pred_type=pred_type, link_approx="mc", n_samples=n_samples)
                except TypeError:
                    p = la(x, pred_type=pred_type, n_samples=n_samples)
            else:
                try:
                    p = la(x, pred_type=pred_type, link_approx="probit")
                except TypeError:
                    p = la(x, pred_type=pred_type)
            outs.append(p.detach().cpu())
    return torch.cat(outs)


def load_deep_ensemble_models(cfg: CFG, tr_loader_aug=None) -> List[torch.nn.Module]:
    """Load ensemble members; train missing ones only if a train loader is given."""
    models = []
    for i in range(cfg.ens_M):
        path = os.path.join(cfg.ckpt_dir, f"{cfg.ens_prefix}_{i}.pth")
        m = ResNetCIFAR(cfg.num_classes, use_dropout=False).to(DEVICE)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEVICE))
        elif tr_loader_aug is not None:
            from cifar100_all_methods_iclr import train_model
            m = train_model(m, tr_loader_aug, cfg, epochs=cfg.epochs_map, lr=cfg.lr_map, wd=cfg.wd_map, ckpt_path=path)
        else:
            raise FileNotFoundError(f"Missing ensemble checkpoint {path}. Run the main script with --methods deepens first, or pass --train-missing-baselines.")
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def deep_ensemble_predict_loader(models: List[torch.nn.Module], loader):
    preds = [predict_probs(m, loader) for m in models]
    return torch.stack(preds).mean(0)


def load_or_train_swag_for_c(cfg: CFG, tr_loader_aug, base_model):
    """Return a SWAG object for CIFAR-100-C evaluation.

    If stats are missing, this calls run_swag once on clean test/OOD loaders in
    the caller's setup before this function should be called. For direct usage,
    this function can load existing stats only.
    """
    stats_path = os.path.join(cfg.ckpt_dir, cfg.swag_stats)
    if not os.path.exists(stats_path):
        return None
    payload = torch.load(stats_path, map_location=DEVICE)
    swag = SWAG(base_model, cfg)
    swag.n = payload["n"]
    swag.mean = [t.to(DEVICE) for t in payload["mean"]]
    swag.sq_mean = [t.to(DEVICE) for t in payload["sq_mean"]]
    return swag


def swag_predict_loader(swag: SWAG, loader, tr_loader_aug, cfg: CFG):
    preds = []
    fixbn_total = 0.0
    for _ in range(cfg.swag_samples):
        swag.sample(scale=cfg.swag_sample_scale)
        elapsed = fix_bn(swag.base_model, tr_loader_aug, DEVICE, num_batches=cfg.swag_fixbn_batches, return_elapsed=True)
        fixbn_total += float(elapsed or 0.0)
        preds.append(predict_probs(swag.base_model, loader))
    return torch.stack(preds).mean(0), fixbn_total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cifar100c-root", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=["map", "trl", "deepens", "swag"],
                   help="map ela lla trl deepens swag mcdo")
    p.add_argument("--results", default="results_iclr/cifar100c.jsonl")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--corruptions", nargs="*", default=None)
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ens-M", type=int, default=None)
    p.add_argument("--trl-k-perp", type=int, default=30)
    p.add_argument("--trl-steps", type=int, default=40)
    p.add_argument("--trl-fixbn-batches", type=int, default=25)
    p.add_argument("--trl-samples", type=int, default=25)
    p.add_argument("--trl-tube-scale", type=float, default=None,
                   help="Use the clean validation-selected TRL tube scale. If omitted, uses first CFG scale.")
    p.add_argument("--train-missing-baselines", action="store_true",
                   help="Train missing ensemble/SWAG/MC-Dropout checkpoints if needed. Otherwise require existing checkpoints.")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.ckpt_dir = args.ckpt_dir or f"./checkpoints_c100_seed{args.seed}"
    cfg.trl_k_perp = args.trl_k_perp
    cfg.trl_steps = args.trl_steps
    cfg.trl_fixbn_batches = args.trl_fixbn_batches
    cfg.trl_val_samples = args.trl_samples
    if args.ens_M is not None:
        cfg.ens_M = args.ens_M
    if args.quick:
        cfg.epochs_map = 1
        cfg.ens_M = min(cfg.ens_M, 2)
        cfg.swag_epochs = 1
        cfg.swag_samples = 2
        cfg.mcdo_samples = 2
        cfg.trl_steps = min(cfg.trl_steps, 3)
        cfg.trl_k_perp = min(cfg.trl_k_perp, 3)
        cfg.trl_val_samples = min(cfg.trl_val_samples, 2)
        cfg.trl_hvp_batches = 1
    set_seed(cfg.seed)

    methods = [m.lower() for m in args.methods]
    corruptions = args.corruptions
    if corruptions is None or len(corruptions) == 0:
        corruptions = sorted([p.stem for p in Path(args.cifar100c_root).glob("*.npy") if p.name != "labels.npy"])

    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    t_clean = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    timings: Dict[str, Dict[str, float]] = {}
    with StageTimer("data_clean", timings):
        tr_aug, bn_clean, val, ts, ood = get_data(cfg)
    with StageTimer("map_train_or_load", timings):
        model_map = load_or_train_map(tr_aug, cfg)

    la = base_val = None
    if any(m in methods for m in ["ela", "lla", "trl"]):
        la, base_val, *_ = laplace_fit_and_predict(model_map, bn_clean, ts, ood, cfg, timings=timings)

    trl = None
    best_ts = None
    if "trl" in methods:
        prior_vec = build_trl_prior_from_laplace(base_val, model_map)
        trl = PracticalTRLStage2(model_map, prior_vec, bn_clean, cfg.trl_steps, cfg.trl_k_perp,
                                 cfg.trl_step_size, cfg.trl_eta, tube_scale=1.0,
                                 max_delta_norm=cfg.trl_max_delta_norm,
                                 hvp_batches=cfg.trl_hvp_batches,
                                 store_every=cfg.trl_store_every)
        with StageTimer("trl_spine_construction", timings):
            trl.build()
        best_ts = float(args.trl_tube_scale) if args.trl_tube_scale is not None else float(cfg.trl_tube_scales[0])
        trl.beta = best_ts

    ensemble_models: Optional[List[torch.nn.Module]] = None
    if "deepens" in methods or "deep_ensemble" in methods or "ensemble" in methods:
        with StageTimer("deep_ensemble_load_or_train", timings):
            ensemble_models = load_deep_ensemble_models(cfg, tr_loader_aug=tr_aug if args.train_missing_baselines else None)

    mcdo_model = None
    if "mcdo" in methods or "mc_dropout" in methods:
        with StageTimer("mcdo_load_or_train", timings):
            if args.train_missing_baselines:
                mcdo_model = load_or_train_mcdo(tr_aug, cfg)
            else:
                path = os.path.join(cfg.ckpt_dir, cfg.mcdo_ckpt)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing MC-Dropout checkpoint {path}. Run main script first or pass --train-missing-baselines.")
                mcdo_model = load_or_train_mcdo(tr_aug, cfg)

    swag_obj = None
    if "swag" in methods:
        with StageTimer("swag_load_or_train", timings):
            stats_path = os.path.join(cfg.ckpt_dir, cfg.swag_stats)
            if not os.path.exists(stats_path):
                if not args.train_missing_baselines:
                    raise FileNotFoundError(f"Missing SWAG stats {stats_path}. Run main script with --methods swag first, or pass --train-missing-baselines.")
                # Build stats once using clean test/OOD loaders; predictions here are discarded.
                run_swag(tr_aug, ts, ood, copy.deepcopy(model_map), cfg, timings=timings)
            swag_obj = load_or_train_swag_for_c(cfg, tr_aug, copy.deepcopy(model_map))
            if swag_obj is None:
                raise RuntimeError("Could not load or build SWAG stats.")

    for corr in corruptions:
        for sev in args.severities:
            ds = CIFAR100CDataset(args.cifar100c_root, corr, sev, transform=t_clean)
            loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
            targets = torch.tensor(ds.y).long()
            for method in methods:
                method_norm = method.lower()
                t = dict(timings)
                fixbn_overhead = 0.0
                if method_norm == "map":
                    with StageTimer(f"eval_{corr}_{sev}_map", t):
                        probs = predict_probs(model_map, loader)
                    label = "MAP"
                elif method_norm == "ela":
                    with StageTimer(f"eval_{corr}_{sev}_ela", t):
                        probs = laplace_predict_loader(la, loader, "nn", cfg.laplace_n_samples_ela)
                    label = "ELA"
                elif method_norm == "lla":
                    with StageTimer(f"eval_{corr}_{sev}_lla", t):
                        probs = laplace_predict_loader(la, loader, "glm", cfg.laplace_n_samples_ela)
                    label = "LLA"
                elif method_norm == "trl":
                    trl.reset_accounting()
                    with StageTimer(f"eval_{corr}_{sev}_trl", t):
                        probs, _ = trl.predict(loader, bn_loader_aug=tr_aug, n_samples=cfg.trl_val_samples,
                                               fix_bn_batches=cfg.trl_fixbn_batches)
                    fixbn_overhead = trl.last_predict_fixbn_sec
                    t[f"eval_{corr}_{sev}_trl_fixbn_overhead"] = {"wall_sec": float(fixbn_overhead), "peak_vram_gb": 0.0}
                    label = "TRL"
                elif method_norm in ["deepens", "deep_ensemble", "ensemble"]:
                    with StageTimer(f"eval_{corr}_{sev}_deepens", t):
                        probs = deep_ensemble_predict_loader(ensemble_models, loader)
                    label = "DeepEns"
                elif method_norm == "swag":
                    with StageTimer(f"eval_{corr}_{sev}_swag", t):
                        probs, fixbn_overhead = swag_predict_loader(swag_obj, loader, tr_aug, cfg)
                    t[f"eval_{corr}_{sev}_swag_fixbn_overhead"] = {"wall_sec": float(fixbn_overhead), "peak_vram_gb": 0.0}
                    label = "SWAG"
                elif method_norm in ["mcdo", "mc_dropout"]:
                    with StageTimer(f"eval_{corr}_{sev}_mcdo", t):
                        probs = mc_dropout_predict(mcdo_model, loader, tr_aug, cfg, timings=t, stage_prefix=f"eval_{corr}_{sev}_mcdo")
                    label = "MC-Dropout"
                else:
                    print(f"Skipping unsupported method: {method}")
                    continue
                acc, nll, ece, brier = calc_metrics(probs, targets, cfg.num_classes)
                row = {
                    "dataset": "CIFAR-100-C",
                    "architecture": "ResNet-18-CIFAR",
                    "method": label,
                    "seed": cfg.seed,
                    "corruption": corr,
                    "severity": sev,
                    "acc": float(acc),
                    "nll": float(nll),
                    "ece": float(ece),
                    "brier": float(brier),
                    "best_tube_scale": best_ts,
                    "fixbn_overhead_sec": float(fixbn_overhead),
                }
                row.update(flatten_timings("time", t))
                append_jsonl(args.results, row)
                print(row)
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
