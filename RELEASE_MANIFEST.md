# TRL experiment code release manifest

This repository contains scripts and documentation for the TRL paper experiments.

## Main folders

- `scripts/`: main CIFAR-100 pipelines, architecture checks, utilities.
- `ablation_scripts/`: TRL ablations and sensitivity scripts.
- `toy/`: toy spine-isolation experiments.
- `finetune/`: CIFAR-100 to CIFAR-10 few-shot fine-tuning diagnostic.
- `diagnostics/`: deterministic spine functional-disagreement diagnostics.
- `phase1_prereg/`: preregistered Phase 1 materials when available.
- `assets/`: small paper figure assets when available.

## Not included

Large checkpoints, raw result files, datasets, and logs are intentionally excluded.
See `README.md` for original server paths and reproduction notes.

## ImageNet / ResNet-50 scale-check

- `scripts/imagenet_marglik_fit.py`: last-layer Laplace marginal-likelihood fit used to estimate `lambda_base`.
- `scripts/imagenet_resnet50_scalecheck.py`: ImageNet / ResNet-50 TRL scale-check pipeline.
- `docs/imagenet_resnet50_scalecheck.md`: protocol and reproduction notes for Appendix I.

Large ImageNet datasets, checkpoints, cached bases, raw JSONL files, and generated result files are intentionally excluded.
