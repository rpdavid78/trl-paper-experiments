# cifar100_all_baselines_trl_laplace_de_swag_mcdropout.py
# A single script for CIFAR-100:
# - MAP
# - Laplace (Last-layer, KRON) + marglik => ELA/LLA
# - Deep Ensemble (DE)
# - SWAG-Diag
# - MC Dropout
# - TRL Stage-2 (HVP ablation) with tube_scale sweep
#
# Maintains the essential logic of your scripts:
# - Fixed split 45k/5k with SEED
# - Train with AUG; CLEAN loaders available
# - TRL: HVP with BN in TRAIN; FixBN with AUG loader
# - Laplace: last-layer + kron + optimize_prior_precision(method="marglik")
#
# Requisitos:
# pip install laplace-torch scipy scikit-learn torchvision

import os
import time
import copy
import gc
import hashlib
import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse.linalg as sla
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.nn.utils import parameters_to_vector, vector_to_parameters


def _trl_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def _trl_vector_to_trainable_parameters(vec, model):
    """Load a flattened vector into trainable parameters only.

    For ordinary full-model TRL this is identical to vector_to_parameters(vec,
    model.parameters()). For head-only or last-block fine-tuning, PracticalTRLStage2
    constructs theta in the active trainable subspace, so writing into all
    parameters causes a shape mismatch.
    """
    vector_to_parameters(vec, _trl_trainable_parameters(model))
from laplace import Laplace

try:
    from trl_iclr_utils.experiment_io import StageTimer, append_jsonl, flatten_timings
except Exception:
    # allow running from repository root without installing as package
    import sys as _sys, os as _os
    _sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
    from trl_iclr_utils.experiment_io import StageTimer, append_jsonl, flatten_timings


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class CFG:
    # system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # data
    batch_size: int = 128
    num_workers: int = 2
    num_classes: int = 100
    ood_svhn_n: int = 2000

    # MAP training (base recipe)
    epochs_map: int = 50
    lr_map: float = 0.1
    wd_map: float = 5e-4
    momentum: float = 0.9

    # checkpoints
    ckpt_dir: str = "./checkpoints_c100"
    map_ckpt: str = "resnet18_cifar100_map.pth"
    mcdo_ckpt: str = "resnet18_cifar100_mcdo.pth"
    ens_prefix: str = "c100_ens"
    swag_stats: str = "c100_swag_diag_map_stats_v2.pth"
    trl_spine: str = "c100_trl_stage2_spine.pth"

    # Laplace
    laplace_subset: int = 5000
    laplace_fit_bs: int = 32
    laplace_n_samples_ela: int = 25  # ELA
    # LLA does not use n_samples in its code

    # Deep Ensemble
    ens_M: int = 5

    # SWAG
    swag_epochs: int = 10
    swag_lr: float = 1e-3
    swag_samples: int = 20
    swag_sample_scale: float = 1.0
    swag_collect_momentum: float = 0.9
    swag_collect_alpha: float = 0.1
    swag_fixbn_batches: int = 20
    allow_legacy_swag_cache: bool = False

    # SWAG-only BatchNorm refresh. ``reset`` is the corrected independent
    # recalibration; use ``rolling`` to reproduce the originally reported run.
    swag_fixbn_mode: str = "reset"

    # MC Dropout
    mcdo_p: float = 0.2
    mcdo_samples: int = 25
    mcdo_fixbn_batches: int = 20

    # TRL Stage-2 fixed parameters
    trl_k_perp: int = 30
    trl_steps: int = 40
    trl_step_size: float = 0.01
    trl_eta: float = 1e-3
    trl_val_samples: int = 25
    trl_fixbn_batches: int = 25
    trl_tube_scales: Tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 10.0, 20.0)

    # TRL numerics / safety
    trl_max_delta_norm: float = 0.02
    trl_hvp_batches: int = 5

    # TRL spine storage (memory)
    trl_store_every: int = 1  # 1 = saves every step (like your script). e.g.: 2 = saves every 2 steps.

    # Dedicated prior-boost diagnostics. These modes reuse one constructed spine
    # and terminate after writing their rows, so they should be run with
    # ``--methods trl`` rather than as part of the full baseline suite.
    run_boost_ablation: bool = False
    run_boost_betaperp_sweep: bool = False
    boost_results: str = "results/boost_ablation.jsonl"
    boost_values: Tuple[float, ...] = (0.0, 10.0, 50.0, 100.0)
    boost_beta_fixed: float = 4.0
    boost_beta_grid: Tuple[float, ...] = (2.0, 4.0, 8.0)
    boost_sampling_seeds: Tuple[int, ...] = (0, 1, 2)
    boost_n_samples: int = 25
    boost_fixbn_batches: int = 25


CFG_ = CFG()
DEVICE = torch.device(CFG_.device)


# ==============================================================================
# UTILS
# ==============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


SWAG_CACHE_SCHEMA_VERSION = 2


def model_state_sha256(model: nn.Module) -> str:
    """Return a deterministic fingerprint of parameters and persistent buffers."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        metadata = (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            repr(tuple(tensor.shape)).encode("ascii"),
        )
        for field in metadata:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def validate_swag_cache_provenance(payload: Dict, map_model: nn.Module,
                                   cfg: CFG, allow_legacy: bool = False) -> str:
    """Validate the MAP state and collection protocol behind cached moments."""
    map_sha256 = model_state_sha256(map_model)
    expected = {
        "schema_version": SWAG_CACHE_SCHEMA_VERSION,
        "base_model_source": "MAP",
        "base_model_state_sha256": map_sha256,
        "swag_variant": "diagonal",
        "map_seed": int(cfg.seed),
        "swag_epochs": int(cfg.swag_epochs),
        "swag_lr": float(cfg.swag_lr),
        "swag_momentum": float(cfg.momentum),
        "swag_batch_size": int(cfg.batch_size),
        "swag_num_workers": int(cfg.num_workers),
    }
    missing = [key for key in expected if key not in payload]
    if missing:
        if not allow_legacy:
            raise RuntimeError(
                "Legacy or incomplete SWAG cache provenance "
                f"(missing: {', '.join(missing)}). Regenerate it or "
                "pass --allow-legacy-swag-cache for exact historical reproduction."
            )
        warnings.warn(
            "Accepting legacy SWAG cache with incomplete provenance metadata.",
            RuntimeWarning,
            stacklevel=2,
        )

    # A legacy override waives absent metadata only. Any metadata that is
    # present must still agree with the selected MAP and collection protocol.
    mismatches = [
        f"{key}={payload[key]!r} (expected {value!r})"
        for key, value in expected.items()
        if key in payload and payload[key] != value
    ]
    if mismatches:
        raise RuntimeError(
            "SWAG cache provenance mismatch: " + "; ".join(mismatches)
        )
    return map_sha256


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_grads(grads, params):
    vec = []
    for g, p in zip(grads, params):
        if g is not None:
            vec.append(g.contiguous().view(-1))
        else:
            vec.append(torch.zeros_like(p).view(-1))
    return torch.cat(vec)


def get_targets(loader) -> torch.Tensor:
    ys = []
    for _, y in loader:
        ys.append(y)
    return torch.cat(ys)


# ==============================================================================
# DATA
# ==============================================================================
def get_data(cfg: CFG):
    print(">>> Preparando Dados CIFAR-100...")

    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    t_train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    t_clean = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    trainset_aug = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=t_train_aug)
    trainset_clean = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=t_clean)

    # fixed split with seed
    indices = torch.randperm(len(trainset_aug), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    train_idx = indices[:45000]
    val_idx = indices[45000:]

    tr_sub_aug = torch.utils.data.Subset(trainset_aug, train_idx)
    tr_sub_clean = torch.utils.data.Subset(trainset_clean, train_idx)
    val_sub_clean = torch.utils.data.Subset(trainset_clean, val_idx)

    tr_loader_aug = torch.utils.data.DataLoader(
        tr_sub_aug, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    bn_loader_clean = torch.utils.data.DataLoader(
        tr_sub_clean, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_sub_clean, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=t_clean)
    ts_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # OOD: SVHN
    svhn = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=t_clean)
    ood_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(svhn, range(cfg.ood_svhn_n)),
        batch_size=cfg.batch_size,
        shuffle=False
    )

    return tr_loader_aug, bn_loader_clean, val_loader, ts_loader, ood_loader


# ==============================================================================
# MODEL (ResNet-18 for CIFAR - "CIFAR-style" version as its TRL/Laplace)
# ==============================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_dropout=False, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.use_dropout = use_dropout
        self.drop = nn.Dropout(p_drop) if use_dropout and p_drop > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=100, use_dropout=False, p_drop=0.0):
        super().__init__()
        self.in_planes = 64
        self.use_dropout = use_dropout
        self.p_drop = p_drop

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_head = nn.Dropout(p_drop) if use_dropout and p_drop > 0 else nn.Identity()
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(BasicBlock(self.in_planes, planes, st, use_dropout=self.use_dropout, p_drop=self.p_drop))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avgpool(out).flatten(1)
        out = self.drop_head(out)
        return self.linear(out)


# ==============================================================================
# TRAIN / LOAD (MAP e afins)
# ==============================================================================
def train_model(model: nn.Module, train_loader, cfg: CFG, epochs: int, lr: float, wd: float, ckpt_path: Optional[str] = None):
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch+1}/{epochs} done.")
        cleanup()

    if ckpt_path is not None:
        torch.save(model.state_dict(), ckpt_path)

    return model


def load_or_train_map(tr_loader_aug, cfg: CFG) -> nn.Module:
    ensure_dir(cfg.ckpt_dir)
    path = os.path.join(cfg.ckpt_dir, cfg.map_ckpt)
    model = ResNetCIFAR(cfg.num_classes, use_dropout=False).to(DEVICE)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(">>> MAP Carregado.")
        return model

    print(f">>> Treinando MAP ({cfg.epochs_map} epochs)...")
    model = train_model(model, tr_loader_aug, cfg, epochs=cfg.epochs_map, lr=cfg.lr_map, wd=cfg.wd_map, ckpt_path=path)
    return model


# ==============================================================================
# PRED / METRICS
# ==============================================================================
@torch.no_grad()
def predict_probs(model: nn.Module, loader) -> torch.Tensor:
    model.eval()
    probs = []
    for x, _ in loader:
        p = torch.softmax(model(x.to(DEVICE)), dim=1).cpu()
        probs.append(p)
    return torch.cat(probs, dim=0)


def calc_metrics(probs: torch.Tensor, targets: torch.Tensor, num_classes: int):
    p = probs.clamp(1e-7, 1 - 1e-7)
    nll = nn.NLLLoss()(torch.log(p), targets.long()).item()
    acc = p.argmax(1).eq(targets).float().mean().item()

    confs, preds = p.max(1)
    ece = 0.0
    bins = torch.linspace(0, 1, 16)
    for i in range(15):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() > 0:
            ece += torch.abs(
                confs[mask].mean() - preds[mask].eq(targets[mask]).float().mean()
            ) * (mask.sum() / len(p))

    oh = F.one_hot(targets.long(), num_classes).float()
    brier = ((p - oh) ** 2).sum(1).mean().item()
    return acc, nll, float(ece), brier


def auroc_entropy(p_id: torch.Tensor, p_ood: torch.Tensor) -> float:
    p_id = p_id.clamp(1e-9, 1.0)
    p_ood = p_ood.clamp(1e-9, 1.0)
    ent_id = -(p_id * torch.log(p_id)).sum(1).numpy()
    ent_ood = -(p_ood * torch.log(p_ood)).sum(1).numpy()
    y = np.concatenate([np.zeros(len(ent_id)), np.ones(len(ent_ood))])
    s = np.concatenate([ent_id, ent_ood])
    return float(roc_auc_score(y, s))


# ==============================================================================
# Laplace baseline (ELA/LLA)
# ==============================================================================
def laplace_fit_and_predict(model_map: nn.Module, bn_loader_clean, ts_loader, ood_loader, cfg: CFG, timings: Optional[Dict[str, Dict[str, float]]] = None):
    print("\n>>> [Laplace] Ajustando Laplace (Last-Layer, KRON) + marglik...")
    # subset for fit
    subset_idx = torch.randperm(len(bn_loader_clean.dataset), generator=torch.Generator().manual_seed(cfg.seed))[:cfg.laplace_subset]
    sub_tr = torch.utils.data.Subset(bn_loader_clean.dataset, subset_idx)

    la = Laplace(model_map, likelihood="classification", subset_of_weights="last_layer", hessian_structure="kron")
    if timings is None:
        la.fit(torch.utils.data.DataLoader(sub_tr, batch_size=cfg.laplace_fit_bs, shuffle=True))
        la.optimize_prior_precision(method="marglik")
    else:
        with StageTimer("laplace_fit", timings):
            la.fit(torch.utils.data.DataLoader(sub_tr, batch_size=cfg.laplace_fit_bs, shuffle=True))
        with StageTimer("laplace_prior_optimization", timings):
            la.optimize_prior_precision(method="marglik")

    # prior base (LL)
    if la.prior_precision.ndim > 1:
        base_val = la.prior_precision.diag().mean().item()
    else:
        base_val = la.prior_precision.mean().item()
    print(f"    Prior Base (LL): {base_val:.4f}")

    # ELA / LLA prediction (maintains its behavior)
    def laplace_pred(la_obj, loader, pred_type):
        out = []
        link = "mc" if pred_type == "nn" else "probit"
        for x, _ in loader:
            if pred_type == "nn":
                out.append(
                    la_obj(x.to(DEVICE), pred_type=pred_type, link_approx=link, n_samples=cfg.laplace_n_samples_ela)
                    .detach().cpu()
                )
            else:
                out.append(
                    la_obj(x.to(DEVICE), pred_type=pred_type, link_approx=link)
                    .detach().cpu()
                )
        return torch.cat(out)

    print(">>> [Laplace] Pred ELA/LLA...")
    if timings is None:
        p_ela = laplace_pred(la, ts_loader, "nn")
        p_lla = laplace_pred(la, ts_loader, "glm")
        p_ela_ood = laplace_pred(la, ood_loader, "nn")
        p_lla_ood = laplace_pred(la, ood_loader, "glm")
    else:
        with StageTimer("ela_predict_id", timings):
            p_ela = laplace_pred(la, ts_loader, "nn")
        with StageTimer("lla_predict_id", timings):
            p_lla = laplace_pred(la, ts_loader, "glm")
        with StageTimer("ela_predict_ood", timings):
            p_ela_ood = laplace_pred(la, ood_loader, "nn")
        with StageTimer("lla_predict_ood", timings):
            p_lla_ood = laplace_pred(la, ood_loader, "glm")

    return la, base_val, p_ela, p_lla, p_ela_ood, p_lla_ood


# ==============================================================================
# Deep Ensemble
# ==============================================================================
def deep_ensemble(tr_loader_aug, ts_loader, ood_loader, cfg: CFG):
    print(f"\n>>> [Deep Ensemble] Training/Loading M={cfg.ens_M}...")
    ensure_dir(cfg.ckpt_dir)

    preds_id = []
    preds_ood = []
    last_model = None

    targets_ts = get_targets(ts_loader)

    for i in range(cfg.ens_M):
        m = ResNetCIFAR(cfg.num_classes, use_dropout=False).to(DEVICE)
        path = os.path.join(cfg.ckpt_dir, f"{cfg.ens_prefix}_{i}.pth")

        if os.path.exists(path):
            print(f"  Loading '{os.path.basename(path)}'...")
            m.load_state_dict(torch.load(path, map_location=DEVICE))
        else:
            print(f"  Training '{os.path.basename(path)}'...")
            m = train_model(m, tr_loader_aug, cfg, epochs=cfg.epochs_map, lr=cfg.lr_map, wd=cfg.wd_map, ckpt_path=path)

        p_id = predict_probs(m, ts_loader)
        p_ood = predict_probs(m, ood_loader)

        preds_id.append(p_id)
        preds_ood.append(p_ood)
        last_model = copy.deepcopy(m)

        cleanup()

    p_ens = torch.stack(preds_id).mean(0)
    p_ens_ood = torch.stack(preds_ood).mean(0)

    # The last member is returned for legacy callers only. The canonical runner
    # always initializes SWAG-Diag from the MAP checkpoint.
    return p_ens, p_ens_ood, last_model, targets_ts


# ==============================================================================
# SWAG-Diag (diagonal moments only; no low-rank deviation matrix)
# ==============================================================================
class SWAG:
    def __init__(self, base_model: nn.Module, cfg: CFG):
        self.base_model = copy.deepcopy(base_model).to(DEVICE)
        self.n = 0
        self.mean = [torch.zeros_like(p, device=DEVICE) for p in self.base_model.parameters()]
        self.sq_mean = [torch.zeros_like(p, device=DEVICE) for p in self.base_model.parameters()]
        self.cfg = cfg

    @torch.no_grad()
    def collect(self, model: nn.Module):
        """Collect SWAG moments using an unbiased online arithmetic mean.

        The previous implementation used an exponential moving average
        initialized at zero, which shrinks the SWAG mean toward zero when only
        a finite number of snapshots is collected. That can severely damage
        sampled models. Here we estimate E[theta] and E[theta^2] directly.
        """
        self.n += 1
        for i, p in enumerate(model.parameters()):
            p_data = p.data.to(DEVICE)
            delta = p_data - self.mean[i]
            self.mean[i].add_(delta / self.n)

            p2 = p_data ** 2
            delta2 = p2 - self.sq_mean[i]
            self.sq_mean[i].add_(delta2 / self.n)

    @torch.no_grad()
    def sample(self, scale: float):
        for i, p in enumerate(self.base_model.parameters()):
            mu = self.mean[i]
            var = torch.clamp(self.sq_mean[i] - mu ** 2, min=1e-30)
            z = torch.randn_like(p, device=DEVICE)
            p.data = mu + math.sqrt(scale) * torch.sqrt(var) * z


def fix_bn(model: nn.Module, loader, device: torch.device, num_batches: int,
           return_elapsed: bool = False, *, mode: str = "rolling"):
    """Refresh BN statistics with either independent or legacy semantics.

    ``reset`` clears every BatchNorm running buffer for the current sampled
    network and uses cumulative averaging over the requested batches. This is
    the corrected independent recalibration. ``rolling`` preserves the legacy
    behavior used by the published runs: buffers and their original momentum
    carry over between posterior samples.
    """
    if mode not in {"reset", "rolling"}:
        raise ValueError(f"Unsupported FixBN mode: {mode!r}")
    if num_batches < 1:
        raise ValueError(f"num_batches must be positive, got {num_batches}")

    bn_layers = [
        module for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    original_momenta = [module.momentum for module in bn_layers]

    use_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        if mode == "reset":
            for module in bn_layers:
                module.reset_running_stats()
                module.momentum = None
        model.train()
        with torch.no_grad():
            it = iter(loader)
            for _ in range(num_batches):
                try:
                    x, _ = next(it)
                except StopIteration:
                    break
                model(x.to(device))
    finally:
        model.eval()
        if mode == "reset":
            for module, momentum in zip(bn_layers, original_momenta):
                module.momentum = momentum

    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if return_elapsed:
        return elapsed
    return None


def run_swag(tr_loader_aug, ts_loader, ood_loader, map_model: nn.Module, cfg: CFG,
             timings: Optional[Dict[str, Dict[str, float]]] = None):
    print("\n>>> [SWAG-Diag] Training SWAG-Diag (fine-tune + collect)...")
    ensure_dir(cfg.ckpt_dir)
    stats_path = os.path.join(cfg.ckpt_dir, cfg.swag_stats)
    map_sha256 = model_state_sha256(map_model)

    # se stats existem, carrega
    if os.path.exists(stats_path):
        print("  Loading SWAG-Diag stats...")
        payload = torch.load(stats_path, map_location=DEVICE)
        validate_swag_cache_provenance(
            payload,
            map_model,
            cfg,
            allow_legacy=cfg.allow_legacy_swag_cache,
        )
        swag = SWAG(map_model, cfg)
        swag.n = payload["n"]
        swag.mean = [t.to(DEVICE) for t in payload["mean"]]
        swag.sq_mean = [t.to(DEVICE) for t in payload["sq_mean"]]
    else:
        swag = SWAG(map_model, cfg)
        m = copy.deepcopy(map_model).to(DEVICE)
        opt = torch.optim.SGD(m.parameters(), lr=cfg.swag_lr, momentum=cfg.momentum)

        ctx = StageTimer("swag_finetune_collect", timings) if timings is not None else None
        if ctx is not None: ctx.__enter__()
        try:
            for ep in range(cfg.swag_epochs):
                m.train()
                for x, y in tr_loader_aug:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad(set_to_none=True)
                    nn.CrossEntropyLoss()(m(x), y).backward()
                    opt.step()
                swag.collect(m)
                cleanup()
        finally:
            if ctx is not None: ctx.__exit__(None, None, None)

        torch.save({
            "n": swag.n,
            "mean": [t.detach().cpu() for t in swag.mean],
            "sq_mean": [t.detach().cpu() for t in swag.sq_mean],
            "schema_version": SWAG_CACHE_SCHEMA_VERSION,
            "base_model_source": "MAP",
            "base_model_state_sha256": map_sha256,
            "swag_variant": "diagonal",
            "map_seed": int(cfg.seed),
            "swag_epochs": int(cfg.swag_epochs),
            "swag_lr": float(cfg.swag_lr),
            "swag_momentum": float(cfg.momentum),
            "swag_batch_size": int(cfg.batch_size),
            "swag_num_workers": int(cfg.num_workers),
        }, stats_path)

    print(">>> [SWAG-Diag] Sampling...")
    preds_id = []
    preds_ood = []
    swag_fixbn_sec = 0.0
    for _ in range(cfg.swag_samples):
        swag.sample(scale=cfg.swag_sample_scale)
        elapsed = fix_bn(
            swag.base_model,
            tr_loader_aug,
            DEVICE,
            num_batches=cfg.swag_fixbn_batches,
            return_elapsed=True,
            mode=cfg.swag_fixbn_mode,
        )
        swag_fixbn_sec += float(elapsed or 0.0)
        preds_id.append(predict_probs(swag.base_model, ts_loader))
        preds_ood.append(predict_probs(swag.base_model, ood_loader))
        cleanup()
    if timings is not None:
        timings["swag_fixbn_overhead"] = {"wall_sec": swag_fixbn_sec, "peak_vram_gb": 0.0}

    p_swag = torch.stack(preds_id).mean(0)
    p_swag_ood = torch.stack(preds_ood).mean(0)
    return p_swag, p_swag_ood


# ==============================================================================
# MC Dropout
# ==============================================================================
def enable_dropout_only(model: nn.Module):
    # Keeps BN in eval, but activates Dropout
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, loader, tr_loader_for_bn, cfg: CFG, timings: Optional[Dict[str, Dict[str, float]]] = None, stage_prefix: str = "mcdo"):
    # calibra BN once (optional; here follows the "fix_bn" style of your baselines)
    elapsed = fix_bn(
        model,
        tr_loader_for_bn,
        DEVICE,
        num_batches=cfg.mcdo_fixbn_batches,
        return_elapsed=True,
        mode="rolling",
    )
    if timings is not None:
        timings[f"{stage_prefix}_fixbn_overhead"] = {"wall_sec": float(elapsed or 0.0), "peak_vram_gb": 0.0}

    preds = []
    for _ in range(cfg.mcdo_samples):
        enable_dropout_only(model)
        probs = []
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(DEVICE)), dim=1).cpu())
        preds.append(torch.cat(probs))
        cleanup()
    return torch.stack(preds).mean(0)


def load_or_train_mcdo(tr_loader_aug, cfg: CFG):
    ensure_dir(cfg.ckpt_dir)
    path = os.path.join(cfg.ckpt_dir, cfg.mcdo_ckpt)

    model = ResNetCIFAR(cfg.num_classes, use_dropout=True, p_drop=cfg.mcdo_p).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(">>> MC Dropout model carregado.")
        return model

    print(f">>> Treinando MC Dropout model (dropout={cfg.mcdo_p}, {cfg.epochs_map} epochs)...")
    model = train_model(model, tr_loader_aug, cfg, epochs=cfg.epochs_map, lr=cfg.lr_map, wd=cfg.wd_map, ckpt_path=path)
    return model


# ==============================================================================
# TRL Stage-2 (HVP ablation)
# ==============================================================================
def get_hvp_function_ablation(model: nn.Module, loader, device: torch.device, num_batches: int):
    # ABLATION: BN in training during HVP
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    data_cache = []
    it = iter(loader)
    for _ in range(num_batches):
        try:
            data_cache.append(next(it))
        except StopIteration:
            break
    if len(data_cache) == 0:
        raise RuntimeError("HVP: data_cache vazio.")

    def hvp(v_numpy):
        v = torch.from_numpy(v_numpy).float().to(device)
        model.zero_grad(set_to_none=True)

        loss_avg = torch.tensor(0.0, device=device)
        for x, y in data_cache:
            x, y = x.to(device), y.to(device)
            loss_avg = loss_avg + nn.CrossEntropyLoss()(model(x), y)
        loss_avg = loss_avg / len(data_cache)

        grads = torch.autograd.grad(loss_avg, params, create_graph=True, allow_unused=True)
        g_vec = flatten_grads(grads, params)

        prod = torch.dot(g_vec, v)
        hv_grads = torch.autograd.grad(prod, params, retain_graph=False, allow_unused=True)
        hv_vec = flatten_grads(hv_grads, params)

        del loss_avg, grads, g_vec, prod, hv_grads
        return hv_vec.detach().cpu().numpy()

    _ = num_params
    return hvp


class PracticalTRLStage2:
    def __init__(
        self,
        map_model: nn.Module,
        prior_vec: torch.Tensor,
        clean_loader,
        steps: int,
        k_perp: int,
        step_size: float,
        eta: float,
        tube_scale: float,
        max_delta_norm: float,
        hvp_batches: int,
        store_every: int = 1,
    ):
        self.map_state = copy.deepcopy(map_model.state_dict())
        self.model = copy.deepcopy(map_model).to(DEVICE)
        self.prior = prior_vec.detach().to(DEVICE)
        self.loader = clean_loader

        self.T = steps
        self.k = k_perp
        self.ds = step_size
        self.eta = eta
        self.beta = tube_scale
        self.max_delta_norm = max_delta_norm
        self.hvp_batches = hvp_batches
        self.store_every = max(1, int(store_every))

        self.spine: List[Dict[str, torch.Tensor]] = []
        # Accounting for the practical inference cost. These counters are reset
        # around validation/test/OOD phases in trl_stage2_run.
        self.cumulative_fixbn_sec: float = 0.0
        self.last_predict_fixbn_sec: float = 0.0
        self.last_predict_wall_sec: float = 0.0

    def _reset(self):
        self.model.load_state_dict(self.map_state)
        self.model.eval()

    def reset_accounting(self):
        self.cumulative_fixbn_sec = 0.0
        self.last_predict_fixbn_sec = 0.0
        self.last_predict_wall_sec = 0.0

    def build(self):
        self._reset()

        params = [p for p in self.model.parameters() if p.requires_grad]
        curr_theta = parameters_to_vector(params).detach()
        num_params = curr_theta.numel()

        print(f"    [TRL] Build Start: P={num_params}, T={self.T}, K={self.k}")
        print("    [TRL] Mode: EFFICIENT (Geometric Transport) | HVP(train-BN)")

        hvp_fn = get_hvp_function_ablation(self.model, self.loader, DEVICE, num_batches=self.hvp_batches)
        op = sla.LinearOperator((num_params, num_params), matvec=hvp_fn)
        vals, vecs = sla.eigsh(op, k=self.k + 1, which="LA")

        idx = np.argsort(vals)[::-1][:self.k]
        N = torch.from_numpy(vecs[:, idx].copy()).float().to(DEVICE)
        evals = torch.maximum(
            torch.from_numpy(vals[idx].copy()).float().to(DEVICE),
            torch.tensor(0.0, device=DEVICE),
        )

        # tangente inicial
        vr = torch.randn(num_params, device=DEVICE)
        vr = vr - N @ (N.T @ vr)
        v = vr / (vr.norm() + 1e-9)

        data_iterator = iter(self.loader)

        for t in range(self.T):
            # geometria transversal + prior
            prior_proj = torch.sum((N ** 2) * self.prior.unsqueeze(1), dim=0)
            prec = torch.clamp(evals + prior_proj, min=1e-6)
            inv_sqrt_prec = torch.rsqrt(prec)

            # spine guard (with stride)
            if (t % self.store_every) == 0:
                self.spine.append({
                    "theta": curr_theta.detach().cpu(),
                    "N": N.detach().cpu(),
            "evals": evals.detach().cpu(),
                    "inv_sqrt_prec": inv_sqrt_prec.detach().cpu(),
                })

            # step in the trajectory
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            _trl_vector_to_trainable_parameters(curr_theta, self.model)

            try:
                xb, yb = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.loader)
                xb, yb = next(data_iterator)

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = self.model(xb)
            if not torch.isfinite(logits).all():
                raise RuntimeError("TRL build: logits NaN/Inf")

            loss = nn.CrossEntropyLoss()(logits, yb)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            g = flatten_grads(grads, params)

            g_perp = g - torch.dot(g, v) * v
            delta = self.ds * v - self.eta * g_perp

            dnorm = delta.norm()
            if dnorm > self.max_delta_norm:
                delta = delta * (self.max_delta_norm / (dnorm + 1e-12))

            theta_next = curr_theta + delta

            # transporte
            d = theta_next - curr_theta
            d_norm = d.norm()
            if d_norm > 1e-9:
                v_new = d / d_norm
                proj = v_new @ N
                N_ortho = N - torch.outer(v_new, proj)
                N, _ = torch.linalg.qr(N_ortho, mode="reduced")
                v = v_new

            curr_theta = theta_next

            del loss, grads, g, g_perp, delta
            if (t % 5) == 0:
                cleanup()

    def predict(self, loader, bn_loader_aug, n_samples: int, fix_bn_batches: int):
        """Posterior prediction for the practical discrete TRL approximation.

        The implemented longitudinal rule samples uniformly over stored spine
        checkpoints. Conditional on a checkpoint, it draws a low-rank Gaussian
        transverse perturbation in the transported frame. FixBN time is tracked
        separately so paper tables can report it as an inference overhead.
        """
        if not self.spine:
            raise RuntimeError("Spine vazia. Rode build() antes.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pred_start = time.perf_counter()
        fixbn_total = 0.0

        ens_probs = []
        targets_all = []

        for i in range(n_samples):
            pt = self.spine[np.random.randint(len(self.spine))]
            th_loc = pt["theta"].to(DEVICE)
            N_loc = pt["N"].to(DEVICE)
            isp_loc = pt["inv_sqrt_prec"].to(DEVICE)

            z = torch.randn(self.k, device=DEVICE)
            theta_sample = th_loc + N_loc @ (self.beta * (isp_loc * z))

            _trl_vector_to_trainable_parameters(theta_sample, self.model)

            # Practical FixBN uses the augmented training loader, matching the
            # released implementation. This is not part of the continuous
            # idealization; it is a practical inference step.
            elapsed = fix_bn(
                self.model,
                bn_loader_aug,
                DEVICE,
                num_batches=fix_bn_batches,
                return_elapsed=True,
                mode="rolling",
            )
            fixbn_total += float(elapsed or 0.0)

            preds_batch = []
            with torch.no_grad():
                for x, y in loader:
                    preds_batch.append(torch.softmax(self.model(x.to(DEVICE)), 1).cpu())
                    if i == 0:
                        targets_all.append(y)

            ens_probs.append(torch.cat(preds_batch))
            print(".", end="", flush=True)

            del theta_sample, th_loc, N_loc, isp_loc
            cleanup()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_predict_wall_sec = time.perf_counter() - pred_start
        self.last_predict_fixbn_sec = fixbn_total
        self.cumulative_fixbn_sec += fixbn_total

        print()
        return torch.stack(ens_probs).mean(0), (torch.cat(targets_all) if targets_all else None)


def build_trl_prior_from_laplace(base_val: float, model: nn.Module, boost_factor: float = 50.0, boost_floor: float = 5.0) -> torch.Tensor:
    # keeps its logic: conv prior = max(base*50, 5), head prior = base
    boost_val = max(base_val * boost_factor, boost_floor)

    prior_list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # our head is "linear."
        val_to_use = base_val if ("linear." in name or "fc." in name) else boost_val
        prior_list.append(torch.full((param.numel(),), float(val_to_use), device=DEVICE))
    return torch.cat(prior_list)


def trl_stage2_run(model_map, base_val, bn_loader_clean, tr_loader_aug, val_loader, ts_loader, ood_loader, cfg: CFG):
    print("\n>>> [TRL Stage-2] Construindo prior e rodando build + sweep tube_scale...")

    prior_vec = build_trl_prior_from_laplace(base_val, model_map)

    trl = PracticalTRLStage2(
        map_model=model_map,
        prior_vec=prior_vec,
        clean_loader=bn_loader_clean,  # build with clean (stable)
        steps=cfg.trl_steps,
        k_perp=cfg.trl_k_perp,
        step_size=cfg.trl_step_size,
        eta=cfg.trl_eta,
        tube_scale=0.1,  # placeholder
        max_delta_norm=cfg.trl_max_delta_norm,
        hvp_batches=cfg.trl_hvp_batches,
        store_every=cfg.trl_store_every,
    )

    trl_timings = {}

    print("\n>>> [TRL Stage-2 BUILD] (HVP)...")
    with StageTimer("trl_spine_construction", trl_timings):
        trl.build()
    if getattr(cfg, "run_boost_ablation", False):
        rows = []
        rows.extend(boost_ablation(
            trl, model_map, base_val,
            eval_loader=val_loader,
            bn_loader_aug=tr_loader_aug,
            cfg=cfg,
            boosts=cfg.boost_values,
            beta_fixed=cfg.boost_beta_fixed,
            seeds=cfg.boost_sampling_seeds,
            n_samples=cfg.boost_n_samples,
            fix_bn_batches=cfg.boost_fixbn_batches,
            split_tag="val",
        ))
        rows.extend(boost_ablation(
            trl, model_map, base_val,
            eval_loader=ts_loader,
            bn_loader_aug=tr_loader_aug,
            cfg=cfg,
            boosts=cfg.boost_values,
            beta_fixed=cfg.boost_beta_fixed,
            seeds=cfg.boost_sampling_seeds,
            n_samples=cfg.boost_n_samples,
            fix_bn_batches=cfg.boost_fixbn_batches,
            split_tag="test",
        ))
        for row in rows:
            append_jsonl(cfg.boost_results, row)
        print(f"\nWrote {len(rows)} boost-ablation rows to {cfg.boost_results}")
        import sys; sys.exit(0)
    if getattr(cfg, "run_boost_betaperp_sweep", False):
        rows = boost_betaperp_sweep_2d(
            trl, model_map, base_val,
            test_loader=ts_loader,
            bn_loader_aug=tr_loader_aug,
            cfg=cfg,
            c_grid=cfg.boost_values,
            beta_grid=cfg.boost_beta_grid,
            seeds=cfg.boost_sampling_seeds,
            n_samples=cfg.boost_n_samples,
            fix_bn_batches=cfg.boost_fixbn_batches,
        )
        for row in rows:
            append_jsonl(cfg.boost_results, row)
        print(f"\nWrote {len(rows)} joint-sweep rows to {cfg.boost_results}")
        import sys; sys.exit(0)

    # sweep in VAL with fixed seed by tube_scale (same as yours)
    print(f"\n>>> [TRL Stage-2 SWEEP] n_samples={cfg.trl_val_samples}, fix_bn_batches={cfg.trl_fixbn_batches}")
    best_nll = float("inf")
    best_ts = None
    sweep = []

    t_val = get_targets(val_loader)

    trl.reset_accounting()
    with StageTimer("trl_validation_scale_sweep", trl_timings):
        for i, ts in enumerate(cfg.trl_tube_scales, 1):
            print(f"[{i}/{len(cfg.trl_tube_scales)}] tube_scale={ts} ... ", end="")
            cleanup()

            try:
                set_seed(1000)  # fixo -> mesmas amostras
                trl.beta = float(ts)

                p_val, _ = trl.predict(
                    val_loader,
                    bn_loader_aug=tr_loader_aug,
                    n_samples=cfg.trl_val_samples,
                    fix_bn_batches=cfg.trl_fixbn_batches,
                )
                p_val = p_val.clamp(1e-7, 1.0)
                nll = nn.NLLLoss()(torch.log(p_val), t_val.long()).item()

                print(f"Val NLL={nll:.4f}")
                sweep.append((float(ts), float(nll)))
                if np.isfinite(nll) and nll < best_nll:
                    best_nll = nll
                    best_ts = float(ts)
            except Exception as e:
                print(f"Fail: {e}")

            cleanup()
    trl_timings["trl_validation_fixbn_overhead"] = {"wall_sec": float(trl.cumulative_fixbn_sec), "peak_vram_gb": 0.0}

    print(f">>> Vencedor TRL Stage-2 (Val NLL): tube_scale={best_ts} (Val NLL={best_nll:.4f})")

    # salva spine + config
    ensure_dir(cfg.ckpt_dir)
    spine_path = os.path.join(cfg.ckpt_dir, cfg.trl_spine)
    torch.save({
        "cfg": cfg.__dict__,
        "best_tube_scale": best_ts,
        "best_val_nll": best_nll,
        "sweep": sweep,
        "spine": trl.spine,
    }, spine_path)
    print(f">>> Spine salva em: {spine_path}")

    # final no TEST + OOD
    print(f"\n>>> [TRL Stage-2 FINAL] Test/OOD com tube_scale={best_ts}, n_samples={cfg.trl_val_samples}")
    trl.beta = float(best_ts)

    trl.reset_accounting()
    with StageTimer("trl_test_posterior_prediction", trl_timings):
        p_trl, t_ts = trl.predict(ts_loader, bn_loader_aug=tr_loader_aug, n_samples=cfg.trl_val_samples, fix_bn_batches=cfg.trl_fixbn_batches)
    trl_timings["trl_test_fixbn_overhead"] = {"wall_sec": float(trl.last_predict_fixbn_sec), "peak_vram_gb": 0.0}

    trl.reset_accounting()
    with StageTimer("trl_ood_posterior_prediction", trl_timings):
        p_trl_ood, _ = trl.predict(ood_loader, bn_loader_aug=tr_loader_aug, n_samples=cfg.trl_val_samples, fix_bn_batches=cfg.trl_fixbn_batches)
    trl_timings["trl_ood_fixbn_overhead"] = {"wall_sec": float(trl.last_predict_fixbn_sec), "peak_vram_gb": 0.0}

    return p_trl, p_trl_ood, sweep, best_ts, best_nll, trl_timings



# ==============================================================================
# ICLR EXPERIMENT MAIN WITH CLI, JSONL LOGGING, TIMING, AND PEAK VRAM
# ==============================================================================
def _metrics_row(dataset: str, architecture: str, method: str, seed: int, probs_id: torch.Tensor,
                 probs_ood: torch.Tensor, targets: torch.Tensor, cfg: CFG, timings: Dict[str, Dict[str, float]],
                 extra: Optional[Dict] = None):
    acc, nll, ece, brier = calc_metrics(probs_id, targets, cfg.num_classes)
    auroc = auroc_entropy(probs_id, probs_ood) if probs_ood is not None else float('nan')
    total_sec = float(sum(v.get('wall_sec', 0.0) for v in timings.values()))
    peak_vram = float(max([v.get('peak_vram_gb', 0.0) for v in timings.values()] + [0.0]))
    row = {
        'dataset': dataset,
        'architecture': 'ResNet-18-CIFAR',
        'method': method,
        'seed': int(seed),
        'acc': float(acc),
        'nll': float(nll),
        'ece': float(ece),
        'brier': float(brier),
        'auroc': float(auroc),
        'runtime_total_sec': total_sec,
        'peak_vram_gb': peak_vram,
        'trl_k_perp': int(cfg.trl_k_perp),
        'trl_steps': int(cfg.trl_steps),
        'trl_step_size': float(cfg.trl_step_size),
        'trl_fixbn_batches': int(cfg.trl_fixbn_batches),
        'trl_hvp_batches': int(cfg.trl_hvp_batches),
        'ood_score': 'predictive_entropy',
    }
    row.update(flatten_timings('time', timings))
    if extra:
        row.update(extra)
    return row


def main_iclr(cfg: CFG, methods: List[str], results_path: Optional[str] = None):
    set_seed(cfg.seed)
    ensure_dir(cfg.ckpt_dir)
    cleanup()

    methods = [m.lower() for m in methods]
    run_all = 'all' in methods
    def wants(name):
        return run_all or name.lower() in methods

    timings_global = {}
    with StageTimer('data', timings_global):
        tr_aug, bn_clean, val, ts, ood = get_data(cfg)
    targets_ts = get_targets(ts)

    with StageTimer('map_train_or_load', timings_global):
        print("\n>>> [MAP]...")
        model_map = load_or_train_map(tr_aug, cfg)

    rows = []

    if wants('map'):
        t = dict(timings_global)
        with StageTimer('map_predict', t):
            print("\n>>> [MAP] Pred...")
            p_map = predict_probs(model_map, ts)
            p_map_ood = predict_probs(model_map, ood)
        rows.append(_metrics_row('CIFAR-100', 'ResNet-18-CIFAR', 'MAP', cfg.seed, p_map, p_map_ood, targets_ts, cfg, t))

    need_laplace = wants('ela') or wants('lla') or wants('trl')
    la = base_val = p_ela = p_lla = p_ela_ood = p_lla_ood = None
    if need_laplace:
        t = dict(timings_global)
        with StageTimer('laplace_fit_predict', t):
            la, base_val, p_ela, p_lla, p_ela_ood, p_lla_ood = laplace_fit_and_predict(
                model_map=model_map, bn_loader_clean=bn_clean, ts_loader=ts, ood_loader=ood, cfg=cfg, timings=t
            )
        if wants('ela'):
            rows.append(_metrics_row('CIFAR-100', 'ResNet-18-CIFAR', 'ELA', cfg.seed, p_ela, p_ela_ood, targets_ts, cfg, t))
        if wants('lla'):
            rows.append(_metrics_row('CIFAR-100', 'ResNet-18-CIFAR', 'LLA', cfg.seed, p_lla, p_lla_ood, targets_ts, cfg, t))

    if wants('deepens') or wants('deep_ensemble') or wants('ensemble'):
        t = dict(timings_global)
        with StageTimer('deep_ensemble_train_predict', t):
            p_ens, p_ens_ood, _last_ensemble_member, _ = deep_ensemble(tr_aug, ts, ood, cfg)
        rows.append(_metrics_row('CIFAR-100', 'ResNet-18-CIFAR', 'DeepEns', cfg.seed, p_ens, p_ens_ood, targets_ts, cfg, t,
                                 extra={'ensemble_M': int(cfg.ens_M)}))

    if wants('swag'):
        t = dict(timings_global)
        with StageTimer('swag_train_predict', t):
            p_swag, p_swag_ood = run_swag(tr_aug, ts, ood, model_map, cfg, timings=t)
        rows.append(_metrics_row(
            'CIFAR-100', 'ResNet-18-CIFAR', 'SWAG-Diag', cfg.seed,
            p_swag, p_swag_ood, targets_ts, cfg, t,
            extra={
                'swag_variant': 'diagonal',
                'swag_samples': int(cfg.swag_samples),
                'swag_fixbn_batches': int(cfg.swag_fixbn_batches),
                'swag_fixbn_mode': cfg.swag_fixbn_mode,
                'swag_stats': cfg.swag_stats,
            },
        ))

    if wants('mcdo') or wants('mc_dropout'):
        t = dict(timings_global)
        with StageTimer('mc_dropout_train_or_load_predict', t):
            mc_model = load_or_train_mcdo(tr_aug, cfg)
            p_mc = mc_dropout_predict(mc_model, ts, tr_aug, cfg, timings=t, stage_prefix="mcdo_id")
            p_mc_ood = mc_dropout_predict(mc_model, ood, tr_aug, cfg, timings=t, stage_prefix="mcdo_ood")
        rows.append(_metrics_row('CIFAR-100', 'ResNet-18-CIFAR', 'MC-Dropout', cfg.seed, p_mc, p_mc_ood, targets_ts, cfg, t))

    if wants('trl'):
        if base_val is None:
            raise RuntimeError('TRL requires Laplace prior base value. Include ELA/LLA or let need_laplace run.')
        t = dict(timings_global)
        p_trl, p_trl_ood, trl_sweep, best_ts, best_val_nll, trl_timings = trl_stage2_run(
            model_map=model_map, base_val=base_val, bn_loader_clean=bn_clean,
            tr_loader_aug=tr_aug, val_loader=val, ts_loader=ts, ood_loader=ood, cfg=cfg
        )
        t.update(trl_timings)
        rows.append(_metrics_row('CIFAR-100', 'ResNet-18-CIFAR', 'TRL', cfg.seed, p_trl, p_trl_ood, targets_ts, cfg, t,
                                 extra={'best_tube_scale': float(best_ts), 'best_val_nll': float(best_val_nll),
                                        'trl_sweep': trl_sweep}))

    print("\n" + "=" * 100)
    print("ICLR JSONL RESULTS")
    print("=" * 100)
    for r in rows:
        print(f"{r['method']:12s} seed={r['seed']} acc={100*r['acc']:.2f} nll={r['nll']:.4f} ece={r['ece']:.4f} brier={r['brier']:.4f} auroc={r['auroc']:.4f} total_sec={r['runtime_total_sec']:.1f}")
        if results_path:
            append_jsonl(results_path, r)
    if results_path:
        print(f"\nWrote JSONL rows to {results_path}")
    return rows


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Instrumented CIFAR-100 TRL / baselines runner for ICLR revision.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--results', type=str, default='results/cifar100.jsonl')
    p.add_argument('--methods', nargs='+', default=['all'], help='all, map, ela, lla, trl, deepens, swag, mcdo')
    p.add_argument('--ckpt-dir', type=str, default=None)
    p.add_argument('--epochs-map', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--ens-M', type=int, default=None)
    p.add_argument('--trl-k-perp', type=int, default=None)
    p.add_argument('--trl-steps', type=int, default=None)
    p.add_argument('--trl-step-size', type=float, default=None)
    p.add_argument('--trl-hvp-batches', type=int, default=None)
    p.add_argument('--trl-fixbn-batches', type=int, default=None)
    p.add_argument('--trl-val-samples', type=int, default=None)
    p.add_argument('--trl-tube-scales', type=float, nargs='*', default=None)
    p.add_argument('--swag-epochs', type=int, default=None)
    p.add_argument('--swag-samples', type=int, default=None)
    p.add_argument('--swag-fixbn-batches', type=int, default=None)
    p.add_argument('--swag-fixbn-mode', choices=['rolling', 'reset'], default=None,
                   help='SWAG-Diag BN refresh: reset (corrected default) or rolling (published run).')
    p.add_argument('--swag-stats', type=str, default=None,
                   help='SWAG-Diag cache filename inside --ckpt-dir.')
    p.add_argument('--allow-legacy-swag-cache', action='store_true',
                   help='Allow a SWAG cache without MAP fingerprint metadata.')
    boost_mode = p.add_mutually_exclusive_group()
    boost_mode.add_argument('--run-boost-ablation', action='store_true',
                            help='Run the validation/test 1D backbone-prior boost sweep and exit.')
    boost_mode.add_argument('--run-boost-betaperp-sweep', action='store_true',
                            help='Run the joint boost x beta_perp test sensitivity sweep and exit.')
    p.add_argument('--boost-results', type=str, default='results/boost_ablation.jsonl')
    p.add_argument('--boost-values', type=float, nargs='+', default=None,
                   help='Backbone-prior boost values c used by either boost diagnostic.')
    p.add_argument('--boost-beta-fixed', type=float, default=4.0,
                   help='Fixed beta_perp for the 1D boost ablation.')
    p.add_argument('--boost-beta-grid', type=float, nargs='+', default=None,
                   help='beta_perp values for the joint 2D sensitivity sweep.')
    p.add_argument('--boost-sampling-seeds', type=int, nargs='+', default=None,
                   help='Posterior-sampling seeds within one MAP checkpoint.')
    p.add_argument('--boost-n-samples', type=int, default=25)
    p.add_argument('--boost-fixbn-batches', type=int, default=25)
    p.add_argument('--quick', action='store_true', help='Debug mode: fewer epochs/samples. Do not use for paper tables.')
    args = p.parse_args()
    if (args.run_boost_ablation or args.run_boost_betaperp_sweep):
        methods = {m.lower() for m in args.methods}
        if not ({'trl', 'all'} & methods):
            p.error('boost diagnostics require --methods trl (or all)')
    return args


def cfg_from_args(args) -> CFG:
    cfg = CFG()
    cfg.seed = args.seed
    cfg.ckpt_dir = args.ckpt_dir or f'./checkpoints_c100_seed{args.seed}'
    if args.epochs_map is not None: cfg.epochs_map = args.epochs_map
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.ens_M is not None: cfg.ens_M = args.ens_M
    if args.trl_k_perp is not None: cfg.trl_k_perp = args.trl_k_perp
    if args.trl_steps is not None: cfg.trl_steps = args.trl_steps
    if args.trl_step_size is not None: cfg.trl_step_size = args.trl_step_size
    if args.trl_hvp_batches is not None: cfg.trl_hvp_batches = args.trl_hvp_batches
    if args.trl_fixbn_batches is not None: cfg.trl_fixbn_batches = args.trl_fixbn_batches
    if args.trl_val_samples is not None: cfg.trl_val_samples = args.trl_val_samples
    if args.trl_tube_scales is not None and len(args.trl_tube_scales) > 0:
        cfg.trl_tube_scales = tuple(args.trl_tube_scales)
    if args.swag_epochs is not None: cfg.swag_epochs = args.swag_epochs
    if args.swag_samples is not None: cfg.swag_samples = args.swag_samples
    if args.swag_fixbn_batches is not None: cfg.swag_fixbn_batches = args.swag_fixbn_batches
    if args.swag_fixbn_mode is not None: cfg.swag_fixbn_mode = args.swag_fixbn_mode
    if args.swag_stats is not None: cfg.swag_stats = args.swag_stats
    cfg.allow_legacy_swag_cache = args.allow_legacy_swag_cache
    if cfg.swag_samples < 1:
        raise ValueError("--swag-samples must be positive")
    if cfg.swag_fixbn_batches < 1:
        raise ValueError("--swag-fixbn-batches must be positive")
    cfg.run_boost_ablation = args.run_boost_ablation
    cfg.run_boost_betaperp_sweep = args.run_boost_betaperp_sweep
    cfg.boost_results = args.boost_results
    cfg.boost_beta_fixed = args.boost_beta_fixed
    cfg.boost_n_samples = args.boost_n_samples
    cfg.boost_fixbn_batches = args.boost_fixbn_batches
    if args.boost_values is not None:
        cfg.boost_values = tuple(args.boost_values)
    if args.boost_beta_grid is not None:
        cfg.boost_beta_grid = tuple(args.boost_beta_grid)
    if args.boost_sampling_seeds is not None:
        cfg.boost_sampling_seeds = tuple(args.boost_sampling_seeds)
    if args.quick:
        cfg.epochs_map = min(cfg.epochs_map, 1)
        cfg.ens_M = min(cfg.ens_M, 2)
        cfg.swag_epochs = min(cfg.swag_epochs, 1)
        cfg.swag_samples = min(cfg.swag_samples, 2)
        cfg.swag_fixbn_batches = min(cfg.swag_fixbn_batches, 1)
        cfg.trl_steps = min(cfg.trl_steps, 3)
        cfg.trl_k_perp = min(cfg.trl_k_perp, 3)
        cfg.trl_val_samples = min(cfg.trl_val_samples, 2)
        cfg.trl_fixbn_batches = min(cfg.trl_fixbn_batches, 1)
        cfg.trl_hvp_batches = min(cfg.trl_hvp_batches, 1)
        cfg.trl_tube_scales = (1.0,)
        cfg.boost_values = cfg.boost_values[:1]
        cfg.boost_beta_grid = cfg.boost_beta_grid[:1]
        cfg.boost_sampling_seeds = cfg.boost_sampling_seeds[:1]
        cfg.boost_n_samples = min(cfg.boost_n_samples, 2)
        cfg.boost_fixbn_batches = min(cfg.boost_fixbn_batches, 1)
    return cfg



def boost_ablation(trl, model, base_val, eval_loader, bn_loader_aug, cfg,
                   boosts=(0.0, 10.0, 50.0, 100.0),
                   beta_fixed=4.0, seeds=(0, 1, 2),
                   n_samples=25, fix_bn_batches=25, split_tag="test"):
    import numpy as np, collections, torch
    if not trl.spine:
        raise RuntimeError("Spine vazia. Rode trl.build() antes da ablacao.")
    if "evals" not in trl.spine[0]:
        raise RuntimeError("Spine sem 'evals'. Reconstrua o spine 1x apos editar build().")
    trl.beta = float(beta_fixed)
    rows = []
    for bf in boosts:
        prior_vec = build_trl_prior_from_laplace(base_val, model,
                                                 boost_factor=bf, boost_floor=5.0)
        for cp in trl.spine:
            N = cp["N"].to(DEVICE)
            evals = cp["evals"].to(DEVICE)
            prior_proj = torch.sum((N ** 2) * prior_vec.unsqueeze(1), dim=0)
            prec = torch.clamp(evals + prior_proj, min=1e-6)
            cp["inv_sqrt_prec"] = torch.rsqrt(prec).detach().cpu()
            del N, evals, prior_proj, prec
        cleanup()
        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            probs, targets = trl.predict(loader=eval_loader,
                                         bn_loader_aug=bn_loader_aug,
                                         n_samples=n_samples,
                                         fix_bn_batches=fix_bn_batches)
            acc, nll, ece, brier = calc_metrics(probs, targets, cfg.num_classes)
            rows.append({
                "experiment": "boost_ablation_1d",
                "split": split_tag,
                "map_seed": int(cfg.seed),
                "sampling_seed": int(seed),
                "boost": float(bf),
                "beta_perp": float(beta_fixed),
                "lambda_base": float(base_val),
                "prior_floor": 5.0,
                "trl_k_perp": int(cfg.trl_k_perp),
                "trl_steps": int(cfg.trl_steps),
                "n_samples": int(n_samples),
                "fixbn_batches": int(fix_bn_batches),
                "acc": acc, "nll": nll, "ece": ece, "brier": brier,
            })
            print(f"[{split_tag} boost={bf:>6} sampling_seed={seed}] acc={acc:.4f} nll={nll:.4f} "
                  f"ece={ece:.4f} brier={brier:.4f}")
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for m in ("acc", "nll", "ece", "brier"):
            agg[r["boost"]][m].append(r[m])
    print(f"\n=== Boost ablation [{split_tag}] (mean +/- std over {len(seeds)} sampling seeds, beta fixed at {beta_fixed}) ===")
    for bf in boosts:
        line = f"boost={bf:>6}: "
        for m in ("acc", "nll", "ece", "brier"):
            v = np.array(agg[bf][m]); line += f"{m}={v.mean():.4f}+/-{v.std():.4f}  "
        print(line)
    return rows



# ============================================================================
# boost_betaperp_sweep_2d.py
#
# Sweep 2D conjunto (c, beta_perp) para o prior block-isotropic do TRL.
# Objetivo: testar se o "otimo agudo em c=50" do F.5 e um pico genuino no
# plano (c, beta_perp) ou apenas um ponto numa cordilheira c*beta_perp ~ const.
#
# Run with --run-boost-betaperp-sweep. The CLI constructs one spine and reuses
# it across all cells in the requested (c, beta_perp) grid.
#
# Custo: ~1 spine + 9*3 = 27 amostragens em 1 checkpoint (seed 0). Barato,
# porque spine e N0 NAO dependem de c nem de beta_perp (so a curvatura/HVPs
# e o complemento stiff os definem). c entra so em L_perp (prior projetado),
# beta_perp e o multiplicador da escala. Por isso: construir UMA vez, varrer
# so a amostragem.
#
# The implementation below directly reuses the canonical prior projection,
# transverse factors, sampler, FixBN path, and metric computation.
# ============================================================================

def boost_betaperp_sweep_2d(trl, model, base_val, test_loader, bn_loader_aug, cfg,
                            c_grid=(25.0, 50.0, 100.0),
                            beta_grid=(2.0, 4.0, 8.0),
                            seeds=(0,),
                            n_samples=25, fix_bn_batches=25):
    """Sweep conjunto (c, beta_perp). Adaptado de boost_ablation():
    - c entra em inv_sqrt_prec (via prior_proj), recomputado 1x por c.
    - beta_perp entra em trl.beta, no loop interno (nao mexe na spine nem no prior).
    Spine e N construidos 1x (trl.build() ja rodou antes). Custo: |c|*|beta|*|seeds| predicts.
    """
    import numpy as np, collections, torch
    if not trl.spine:
        raise RuntimeError("Spine vazia. Rode trl.build() antes do sweep.")
    if "evals" not in trl.spine[0]:
        raise RuntimeError("Spine sem 'evals'. Reconstrua o spine 1x apos editar build().")

    rows = []
    for c in c_grid:
        prior_vec = build_trl_prior_from_laplace(base_val, model,
                                                 boost_factor=c, boost_floor=5.0)
        for cp in trl.spine:
            N = cp["N"].to(DEVICE)
            evals = cp["evals"].to(DEVICE)
            prior_proj = torch.sum((N ** 2) * prior_vec.unsqueeze(1), dim=0)
            prec = torch.clamp(evals + prior_proj, min=1e-6)
            cp["inv_sqrt_prec"] = torch.rsqrt(prec).detach().cpu()
            del N, evals, prior_proj, prec
        cleanup()

        for beta in beta_grid:
            trl.beta = float(beta)
            for seed in seeds:
                np.random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                probs, targets = trl.predict(loader=test_loader,
                                             bn_loader_aug=bn_loader_aug,
                                             n_samples=n_samples,
                                             fix_bn_batches=fix_bn_batches)
                acc, nll, ece, brier = calc_metrics(probs, targets, cfg.num_classes)
                rows.append({
                    "experiment": "boost_beta_sweep_2d",
                    "split": "test",
                    "map_seed": int(cfg.seed),
                    "sampling_seed": int(seed),
                    "c": float(c),
                    "beta_perp": float(beta),
                    "product": float(c * beta),
                    "lambda_base": float(base_val),
                    "prior_floor": 5.0,
                    "trl_k_perp": int(cfg.trl_k_perp),
                    "trl_steps": int(cfg.trl_steps),
                    "n_samples": int(n_samples),
                    "fixbn_batches": int(fix_bn_batches),
                    "acc": acc, "nll": nll, "ece": ece, "brier": brier,
                })
                print(f"[c={c:>6} beta={beta:>5} prod={c*beta:>6.0f} sampling_seed={seed}] "
                      f"acc={acc:.4f} nll={nll:.4f} ece={ece:.4f} brier={brier:.4f}")

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for m in ("acc", "nll", "ece", "brier"):
            agg[(r["c"], r["beta_perp"])][m].append(r[m])

    print(f"\n=== Sweep 2D (mean +/- std over {len(seeds)} seed(s)) -- grade ECE ===")
    print("   c \\ beta |" + "".join(f"  {b:>6}" for b in beta_grid))
    for c in c_grid:
        line = f"   {c:>7} |"
        for b in beta_grid:
            v = np.array(agg[(c, b)]["ece"]); line += f"  {v.mean():.4f}"
        print(line)

    by_prod = collections.defaultdict(list)
    for (c, b), d in agg.items():
        by_prod[c * b].append(((c, b), float(np.mean(d["ece"]))))
    print("\n=== Cordilheira (agrupado por c*beta) ===")
    within, prod_means = [], []
    for prod in sorted(by_prod):
        cells = by_prod[prod]; eces = [e for _, e in cells]
        prod_means.append(np.mean(eces))
        if len(cells) > 1:
            spread = max(eces) - min(eces); within.append(spread)
            cs = "  ".join(f"{cb}->ECE {e:.4f}" for cb, e in cells)
            print(f"  produto={prod:>6.0f} (n={len(cells)}): spread={spread:.4f} | {cs}")
        else:
            cb, e = cells[0]
            print(f"  produto={prod:>6.0f} (n=1): {cb}->ECE {e:.4f}")
    across = (max(prod_means) - min(prod_means)) if prod_means else 0.0
    mw = np.mean(within) if within else float("nan")
    print(f"\n  spread MEDIO dentro de produto-igual : {mw:.4f}")
    print(f"  spread ENTRE produtos                : {across:.4f}")
    if within and mw < 0.3 * across:
        print("  --> CORDILHEIRA: ECE segue o produto; 50 nao e pico isolado. Recuar texto do F.5.")
    else:
        print("  --> PICO CONJUNTO provavel: (50,4) nao reproduzido por produto igual. F.5 pode afirmar otimo.")
        print("      (Cheque se (25,8) colapsou: se sim, refuta multiplicatividade -> evidencia PRO estrutura.)")

    return rows

if __name__ == '__main__':
    args = parse_args()
    cfg = cfg_from_args(args)
    main_iclr(cfg, args.methods, args.results)
