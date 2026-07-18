# TRL experiment code release manifest

This manifest defines the contents and boundaries of the double-blind TRL code
release. `README.md` is the runnable reproduction guide and maps commands to the
current paper tables and figures.

## Included

- `scripts/cifar100_all_methods_iclr.py`: canonical ResNet-18 CIFAR-100 runner.
- `scripts/cifar100_arch_sensitivity_iclr.py`: WRN-16-4 architecture runner.
- `scripts/vgg_all_methods_iclr.py` and `scripts/vgg_bn_cifar.py`: VGG-11-BN runner and exact model definition.
- `scripts/cifar100c_eval_iclr.py`: CIFAR-100-C evaluation.
- `scripts/cifar100_laplace_prior_grid_iclr.py`: validation-NLL Laplace prior-grid check.
- `scripts/cifar100_temperature_scaling_iclr.py`: scalar temperature-scaling check.
- `scripts/cifar100_random_rank30_baseline.py`: random full-network rank-30 control.
- `scripts/imagenet_marglik_fit.py` and `scripts/imagenet_resnet50_scalecheck.py`: ImageNet/ResNet-50 scale-check.
- `scripts/aggregate_results.py` and `scripts/make_paper_assets.py`: aggregation and generated-asset utilities.
- `ablation_scripts/`: rank/spine/tube/FixBN, fixed-basis, stale-eigenspace, and fresh-refresh diagnostics.
- `diagnostics/`: deterministic spine loss and functional-disagreement analysis, including the paired SWAG-Diag FixBN audit.
- `finetune/`: CIFAR-100 to CIFAR-10 small-data fine-tuning diagnostics.
- `toy/`: final toy-table driver and underlying toy implementations.
- `docs/`: protocol notes, grids, commands, and reported sanity-check outcomes, including `docs/swag_diag_protocol.md`.
- `phase1_prereg/`: frozen Phase 1 pre-registration and provenance note.
- `scripts/all_exported_code_snapshot/`: historical flattened-export snapshot; not the canonical execution surface.
- `requirements.txt`: pinned audited Python dependency versions.

## Intentionally excluded

- CIFAR, SVHN, CIFAR-100-C, and ImageNet data.
- MAP, ensemble, SWAG-Diag, MC-Dropout, and fine-tuned checkpoints.
- Cached TRL spines, random/fresh bases, Lanczos workspaces, and ImageNet caches.
- Raw JSONL/CSV result files, logs, and generated paper assets.
- Python virtual environments and package caches.

These omissions are expected for an academic code release. Standard small
datasets are downloaded by torchvision; externally distributed datasets are
provided through explicit CLI roots. Missing CIFAR checkpoints can be trained
by the canonical runners. ImageNet uses the fixed torchvision
`IMAGENET1K_V1` checkpoint and user-provided ImageNet directories.

## SWAG-Diag reproducibility boundary

The released baseline is diagonal SWAG, not full low-rank-plus-diagonal SWAG.
The canonical `scripts/cifar100_all_methods_iclr.py` runner always initializes
it from MAP, including under `--methods all`. Together with
`scripts/cifar100c_eval_iclr.py`, it uses independent-reset FixBN by default and
validates versioned caches against the complete MAP state_dict, including
BatchNorm buffers. The published five-seed row used 20 samples, 20 FixBN
batches, and rolling buffers; that historical path remains available through
the explicit `--swag-fixbn-mode rolling` option.

The historical flattened snapshot received the corresponding SWAG-Diag
initializer, FixBN, and cache corrections needed for provenance, but it is not
a byte-for-byte mirror and is not the recommended execution surface. Secondary
architecture, VGG, and base runners received only the MAP-initializer correction
and `SWAG-Diag` label; they retain their historical sampling, FixBN, and cache
protocols. Exact commands, paired seed-0 metrics, and artifact hashes are in
`docs/swag_diag_protocol.md`; the audit runner is
`diagnostics/swag_fixbn_ab_cifar100.py`.

## Reproduction boundary

The repository supplies code and protocol sufficient to regenerate the
reported artifacts, but it does not claim that a full paper rerun is cheap.
Deep Ensembles require repeated training, TRL construction uses HVP/Lanczos
operations, fresh-refresh repeats eigendecomposition along the spine, and the
ImageNet diagnostics require substantial GPU memory and data I/O.

Nested experiments must preserve their independent unit during aggregation.
In particular, Table 17 averages posterior-sampling repeats within each MAP
checkpoint before computing mean and sample standard deviation across three MAP
checkpoints. `scripts/aggregate_results.py --independent-unit` implements this
rule.

## Audited environment

- Python 3.12.3
- PyTorch 2.5.1+cu121
- torchvision 0.20.1+cu121
- laplace-torch 0.2.2.2
- NumPy 1.26.4
- pandas 3.0.3
- SciPy 1.17.1
- scikit-learn 1.8.0
- matplotlib 3.10.9
- Pillow 12.2.0
- 2 x NVIDIA RTX 6000 Ada Generation, 49,140 MiB each
- NVIDIA driver 610.43.02

## Review status

The manuscript is under double-blind review. Author-identifying citation
metadata and an archival release tag are deferred until de-anonymization. The
current `LICENSE` remains all rights reserved; public visibility alone does not
make the release OSI open source.
