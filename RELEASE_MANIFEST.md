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
See `README.md` for portable reproduction notes; machine-specific server paths
are intentionally omitted.

## Canonical CIFAR-100 runner and exported snapshot

- `scripts/cifar100_all_methods_iclr.py`: canonical instrumented CIFAR-100 runner.
- `scripts/all_exported_code_snapshot/cifar100_all_methods_iclr.py`: byte-for-byte
  exported mirror of the canonical runner.
- `scripts/cifar100c_eval_iclr.py`: canonical CIFAR-100-C evaluator.
- `docs/swag_diag_protocol.md`: SWAG-Diag MAP-initializer, cache-provenance,
  and FixBN rolling-versus-reset audit.
- `diagnostics/swag_fixbn_ab_cifar100.py`: paired seed-level FixBN audit using
  identical SWAG-Diag draws and calibration batches in both arms.

In the canonical `scripts/cifar100_all_methods_iclr.py` runner and
`scripts/cifar100c_eval_iclr.py` evaluator, SWAG-Diag uses the corrected
independent-reset FixBN mode by default. The published `S=20`, 20-batch result
used rolling buffers; that historical mode is available there through the
explicit `--swag-fixbn-mode rolling` option. Their versioned SWAG-Diag caches
are tied to a SHA-256 fingerprint of the complete MAP state_dict including BN
buffers and are provenance-checked when loaded.

The exported mirrors of those two canonical entrypoints carry the same
behavior. Secondary architecture, VGG, and base runners, together with their
noncanonical historical snapshot counterparts, received the MAP-initializer
correction and `SWAG-Diag` label only. They retain their own historical
sampling, FixBN, and cache protocols; the canonical reset mode,
versioned-cache format, and provenance guarantees must not be inferred for
those runners.

## ImageNet / ResNet-50 scale-check

- `scripts/imagenet_marglik_fit.py`: last-layer Laplace marginal-likelihood fit used to estimate `lambda_base`.
- `scripts/imagenet_resnet50_scalecheck.py`: ImageNet / ResNet-50 TRL scale-check pipeline.
- `docs/imagenet_resnet50_scalecheck.md`: protocol and reproduction notes for Appendix I.

Large ImageNet datasets, checkpoints, cached bases, raw JSONL files, and generated result files are intentionally excluded.

## Random rank-30 subspace control

- `scripts/cifar100_random_rank30_baseline.py`: random low-rank subspace baseline used as a control against the TRL transverse subspace.
