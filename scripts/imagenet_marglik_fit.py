#!/usr/bin/env python3
"""
ImageNet scale-check — Etapa 1: split persistido + marglik fit (last-layer, KRON).

Espelha o protocolo CIFAR canônico (cifar100_all_methods_iclr.py, laplace_fit_and_predict):
  - Laplace(model, "classification", subset_of_weights="last_layer", hessian_structure="kron")
  - fit em subset de 5000 do train_idx, batch 32, shuffle, transform clean
  - optimize_prior_precision(method="marglik")
  - lambda_base = prior_precision.diag().mean() se matriz KRON, senão .mean()

Decisões de protocolo (Rodrigo, 11/jun/2026):
  - MAP fixo = torchvision ResNet50_Weights.IMAGENET1K_V1 (sem retreino por seed).
  - Seed controla APENAS split train_idx/tuning_idx e amostragens (marglik/HVP/FixBN/sweep).
  - val oficial 50k = test intocado (este script nem o toca).
  - Split persistido em imagenet_split_seed{seed}.pt = fonte única da verdade
    pros scripts downstream (Lanczos/FixBN/sweep), com guard-rail de classes.

Uso:
  python imagenet_marglik_fit.py --seeds 0
  python imagenet_marglik_fit.py --seeds 0 1 2
"""

import argparse
import hashlib
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from laplace import Laplace

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_eval_transform():
    """Transform eval/clean: Resize 256 -> CenterCrop 224 -> normalize ImageNet."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_or_create_split(dataset, seed, tuning_size, out_dir):
    """Split seedado train_idx/tuning_idx, persistido como fonte única da verdade.

    Guard-rail: se o split já existe, exige classes idênticas às do ImageFolder
    atual (protege contra re-extração que mude a ordem das classes).
    """
    split_path = os.path.join(out_dir, f"imagenet_split_seed{seed}.pt")
    n = len(dataset)

    if os.path.exists(split_path):
        loaded = torch.load(split_path, map_location="cpu", weights_only=False)
        assert loaded["classes"] == dataset.classes, (
            f"GUARD-RAIL: classes do split persistido ({split_path}) divergem do "
            f"ImageFolder atual. Dataset foi re-extraído/alterado? ABORTANDO."
        )
        assert loaded["n_dataset"] == n, (
            f"GUARD-RAIL: tamanho do dataset mudou "
            f"({loaded['n_dataset']} -> {n}). ABORTANDO."
        )
        assert loaded["tuning_size"] == tuning_size, (
            f"GUARD-RAIL: tuning_size do split persistido "
            f"({loaded['tuning_size']}) != solicitado ({tuning_size}). ABORTANDO."
        )
        print(f"    [split] reusando {split_path} "
              f"(train={len(loaded['train_idx'])}, tuning={len(loaded['tuning_idx'])})")
        return loaded["train_idx"], loaded["tuning_idx"], split_path

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tuning_idx = perm[:tuning_size].clone()
    train_idx = perm[tuning_size:].clone()

    payload = {
        "seed": seed,
        "n_dataset": n,
        "tuning_size": tuning_size,
        "train_idx": train_idx,
        "tuning_idx": tuning_idx,
        "classes": dataset.classes,
        "created_unix": time.time(),
    }
    torch.save(payload, split_path)
    print(f"    [split] criado {split_path} "
          f"(train={len(train_idx)}, tuning={len(tuning_idx)})")
    return train_idx, tuning_idx, split_path


def sample_marglik_subset(train_idx, seed, subset_size):
    """Subset do marglik tirado SÓ de train_idx, reprodutível por seed."""
    g = torch.Generator().manual_seed(seed + 10_000)  # offset: não colide com o split
    sel = torch.randperm(len(train_idx), generator=g)[:subset_size]
    return train_idx[sel]


def fit_marglik_for_seed(dataset, seed, args):
    print(f"\n>>> [seed {seed}] split + marglik fit...")
    t0 = time.time()

    train_idx, tuning_idx, split_path = get_or_create_split(
        dataset, seed, args.tuning_size, args.out_dir
    )

    fit_idx = sample_marglik_subset(train_idx, seed, args.laplace_subset)
    fit_loader = DataLoader(
        Subset(dataset, fit_idx.tolist()),
        batch_size=args.laplace_fit_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("    [model] ResNet-50 IMAGENET1K_V1 (MAP fixo)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(DEVICE).eval()

    print(f"    [laplace] fit last-layer KRON em {len(fit_idx)} amostras "
          f"(bs={args.laplace_fit_bs})...")
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la.fit(fit_loader)

    print("    [laplace] optimize_prior_precision(marglik)...")
    la.optimize_prior_precision(method="marglik")

    pp = la.prior_precision
    if pp.ndim > 1:
        lambda_base = pp.diag().mean().item()
    else:
        lambda_base = pp.mean().item()

    elapsed = time.time() - t0
    print(f"    Prior Base (LL): {lambda_base:.4f}  [{elapsed:.0f}s]")

    record = {
        "experiment": "imagenet_marglik_fit",
        "seed": seed,
        "split_seed": seed,
        "marglik_subset_seed": seed + 10_000,
        "lambda_base": lambda_base,
        "prior_precision_shape": list(pp.shape),
        "checkpoint_source": "torchvision",
        "weights": "IMAGENET1K_V1",
        "map_is_fixed": True,
        "seed_controls": "split_and_sampling_only",
        "data_root": os.path.abspath(args.train_root),
        "n_dataset": len(dataset),
        "n_classes": len(dataset.classes),
        "tuning_size": args.tuning_size,
        "laplace_subset": args.laplace_subset,
        "laplace_fit_bs": args.laplace_fit_bs,
        "subset_of_weights": "last_layer",
        "hessian_structure": "kron",
        "transform": "Resize256_CenterCrop224_ImageNetNorm",
        "split_file": split_path,
        "split_sha256": sha256_of(split_path),
        "elapsed_s": round(elapsed, 1),
        "timestamp_unix": time.time(),
    }
    jsonl_path = os.path.join(args.out_dir, "imagenet_marglik.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"    [jsonl] -> {jsonl_path}")

    return lambda_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tuning-size", type=int, default=50000)
    ap.add_argument("--laplace-subset", type=int, default=5000)
    ap.add_argument("--laplace-fit-bs", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--train-root",
        required=True,
    )
    ap.add_argument("--out-dir", default="results/imagenet_resnet50_scalecheck")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(">>> Carregando ImageFolder (pode levar minutos no primeiro scan)...")
    dataset = datasets.ImageFolder(args.train_root, transform=build_eval_transform())
    print(f"    {len(dataset)} imagens, {len(dataset.classes)} classes")
    assert len(dataset.classes) == 1000, (
        f"Esperado 1000 synsets, encontrado {len(dataset.classes)} — "
        f"extração incompleta?"
    )

    results = {}
    for seed in args.seeds:
        results[seed] = fit_marglik_for_seed(dataset, seed, args)

    print("\n=== Resumo lambda_base por seed ===")
    for seed, lb in results.items():
        print(f"  seed {seed}: {lb:.4f}")


if __name__ == "__main__":
    main()
