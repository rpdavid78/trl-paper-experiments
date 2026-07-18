# ImageNet / ResNet-50 scale-check

This note documents Tables 25-27, the ImageNet / ResNet-50 scale-check reported
in the paper appendix.

The ImageNet experiment is a scale-check, not the main benchmark. It uses a fixed torchvision ResNet-50 pretrained checkpoint (`IMAGENET1K_V1`) and evaluates whether the practical TRL transverse posterior remains operational at ImageNet scale.

## Protocol

- Model: torchvision ResNet-50 with `IMAGENET1K_V1` weights.
- MAP checkpoint: fixed pretrained torchvision checkpoint; no ImageNet retraining.
- Data:
  - ImageNet train split for last-layer marglik, HVP batches, and FixBN batches.
  - Official ImageNet validation split, mechanically divided into:
    - `val_tuning_25k` for selecting `(c, beta_perp)`;
    - `val_test_25k` for held-out evaluation.
- Prior:
  - classifier head precision: `lambda_base`;
  - backbone precision: `max(c * lambda_base, prior_floor)`.
- Main transverse rank: `rank = 30`.
- Posterior samples: `S = 25`.
- FixBN batches: `25`.

## Scripts

First estimate the last-layer Laplace marginal-likelihood base precision:

```bash
python scripts/imagenet_marglik_fit.py \
  --train-root <imagenet_train_root> \
  --out-dir results/imagenet_resnet50_scalecheck \
  --seeds 0 1 2
```

Then run the ImageNet / ResNet-50 TRL scale-check:

```bash
python scripts/imagenet_resnet50_scalecheck.py \
  --train-root <imagenet_train_root> \
  --val-root <imagenet_val_root> \
  --out-dir results/imagenet_resnet50_scalecheck \
  --seeds 0 1 2 \
  --rank 30 \
  --samples 25 \
  --fixbn-batches 25 \
  --hvp-batches 5 \
  --boost-c 50 150 450 \
  --betas 0.5 1 1.5 2 3 4 \
  --spine-steps 0
```

`--spine-steps 0` is the single-checkpoint transverse scale-check. Positive `--spine-steps` values enable the optional post-hoc spine diagnostic.

## Table 26: seed-0 spine-length diagnostic

Keep `--seeds 0 --boost-c 50 --betas 2 --samples 25` fixed and run these
`(--spine-steps, --spine-step)` pairs in separate output directories:

```text
(0,  0.01)  length 0.00
(8,  0.01)  length 0.08
(24, 0.01)  length 0.24
(24, 0.02)  length 0.48
```

For example:

```bash
python scripts/imagenet_resnet50_scalecheck.py \
  --train-root <imagenet_train_root> --val-root <imagenet_val_root> \
  --out-dir results/imagenet_resnet50_scalecheck/spine_T8_ds001 \
  --seeds 0 --rank 30 --samples 25 --fixbn-batches 25 --hvp-batches 5 \
  --boost-c 50 --betas 2 --spine-steps 8 --spine-step 0.01
```

## Table 27: seed-0 transverse-rank diagnostic

Run ranks 30, 50, and 100 with `--seeds 0 --boost-c 50 --betas 2 3 4` and a
fresh basis for each rank. Set `--lanczos-iters` to three times the rank (90,
150, and 300). The rank-100 run can use `--basis-device cpu` to reduce GPU
storage pressure.

Large checkpoints, cached bases, raw JSONL result files, and datasets are not included in the release.
