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
