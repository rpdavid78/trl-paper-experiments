# TRL Paper Experiments

Experiment code release for **Tubular Riemannian Laplace (TRL)**, a post-hoc posterior approximation for Bayesian neural networks.

This repository contains the code used for the CIFAR-100, CIFAR-100-C, toy, fine-tuning, architecture-sensitivity, and diagnostic experiments reported in the paper.

## Repository structure

```text
scripts/
  Main CIFAR-100 pipelines, architecture checks, CIFAR-100-C evaluation,
  result aggregation, and paper-asset utilities.

ablation_scripts/
  Sensitivity and ablation scripts for TRL hyperparameters and implementation choices.

toy/
  Small full-Hessian toy experiments isolating the longitudinal spine contribution.

finetune/
  CIFAR-100 to CIFAR-10 few-shot fine-tuning diagnostic.

diagnostics/
  Deterministic spine functional-disagreement diagnostics.

phase1_prereg/
  Pre-registered Phase 1 diagnostic materials, when included.

assets/
  Small figure assets used in the paper.

docs/
  Notes for paper tables and ablation protocols.

requirements.txt
  Minimal Python dependency list.

RELEASE_MANIFEST.md
  Short manifest describing the release contents.
```

## Main scripts

### CIFAR-100 / ResNet-18 main benchmark

```bash
python scripts/cifar100_all_methods_iclr.py --methods all --seed 0
```

This script contains the main CIFAR-100 experiment pipeline, including MAP, Laplace baselines, TRL, Deep Ensembles, SWAG, and MC-Dropout.

### CIFAR-100-C robustness

```bash
python scripts/cifar100c_eval_iclr.py
```

Evaluates trained methods on CIFAR-100-C corruptions and severities.

### Architecture-sensitivity checks

```bash
python scripts/cifar100_arch_sensitivity_iclr.py --arch wrn16_4
python scripts/vgg_all_methods_iclr.py
```

These scripts reproduce the WideResNet-16-4 and VGG-11-BN architecture checks.

### Toy spine-isolation experiments

```bash
python toy/toy_spine_single_vs_full.py
```

Runs the original toy experiment comparing a MAP-centered single-checkpoint posterior with the full TRL spine. This script is kept as a legacy diagnostic. The final Tables 3--5 are reproduced by `toy/rerun_toy_tables.py` and `toy/run_final_toy_tables.sh`; see the final toy-table reproduction notes below.

### Fine-tuning diagnostic

```bash
python finetune/finetune_cifar10_spine_smoke.py
```

Runs the CIFAR-100 to CIFAR-10 few-shot fine-tuning diagnostic used to study when longitudinal spine movement contributes predictive variation.

### Spine functional-disagreement diagnostic

```bash
python diagnostics/spine_functional_disagreement_cifar100.py
```

Measures deterministic functional drift along the stored TRL spine using validation-set disagreement, Jensen--Shannon divergence, and loss drift.

## Environment

A minimal dependency list is provided in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments were run with PyTorch, torchvision, scipy, scikit-learn, pandas, matplotlib, and laplace-torch. GPU execution is recommended for all CIFAR-scale experiments.

## Data

The scripts use standard public datasets such as CIFAR-100, CIFAR-10, SVHN, and CIFAR-100-C. Dataset downloads are handled by the scripts or expected to be placed in a local data directory. Large datasets are not included in this repository.

## Checkpoints and results

Large checkpoints, raw logs, cached TRL spines, and raw result files are intentionally not included in this code release.

The repository is intended to provide:

* the experiment code,
* the implementation details,
* the ablation and diagnostic scripts,
* and the structure needed to reproduce the reported experiments.

## TRL implementation notes

The practical TRL implementation uses:

* a MAP checkpoint as the base solution,
* stochastic Hessian-vector products for curvature access,
* a low-rank transverse subspace,
* a transported discrete spine,
* validation-selected transverse scale,
* and BatchNorm recalibration when applicable.

The large-network TRL prior is block-isotropic: classifier-head parameters receive the base precision inherited from the last-layer Laplace fit, while non-head parameters receive a boosted backbone precision. The backbone prior-boost coefficient `c=50` is selected on a held-out clean validation split by validation NLL from a small sweep over `c`, and is then confirmed on the test split. The tube scale `beta_perp` is likewise selected on the held-out clean validation split by validation NLL in the CIFAR-scale pipeline. The Table 16 sweeps report both validation and test sensitivity for the boost, showing that the no-boost setting collapses and that the boost factor and tube scale do not reduce to a single effective product.

## Table 14 tube-scale sensitivity rerun

The final Table 14 / Figure 9 tube-scale sensitivity values are documented in:

```text
docs/tube_scale_sameood_rerun.md
```

This note records the rerun using the same test/OOD evaluation path as the other CIFAR-100 ablations. The final sweep uses:

```text
beta_perp in {2.0, 3.0, 4.0, 6.0, 10.0, 20.0}
seed in {0, 1, 2}
k_perp = 30, T = 40, step_size = 0.01, FixBN batches = 25, S = 25
```

For each `beta_perp`, the script is run with a single-value `--trl-tube-scales <beta_perp>` argument so that the reported row is the final test/OOD evaluation for that forced scale. The selected main scale remains `beta_perp = 4.0`, selected by validation NLL; it also gives the best ECE and Brier in this sweep.

## Random rank-30 full-network control

The random full-network rank-30 diagnostic control used in the appendix text is documented in:

```text
docs/random_rank30_full_network_control.md
```

This control replaces the Fisher/GGN-selected TRL transverse subspace by a random rank-30 orthonormal full-network subspace, while keeping the block prior, FixBN protocol, `S=25`, and validation-selected tube scale unchanged. Across five random bases on one MAP checkpoint and one-basis controls on two additional MAP checkpoints, the random posterior stays MAP-like and selects the largest value in the original beta grid. A provenance-clean extension to `beta_perp=40` remains MAP-like and only weakly improves validation NLL, supporting the interpretation that the calibration gain comes from Fisher/GGN-selected transverse directions rather than from full-network access alone.

## Table 16 boost-prior ablation

The code and notes for the TRL backbone-prior boost ablation are provided in:

```text
docs/table16_boost_ablation.md
```

This documents both parts of Table 16:

* the 1D boost sweep at fixed validation-selected `beta_perp = 4`, including validation and test NLL/ECE and the no-boost `c=0` case;
* the joint `c x beta_perp` sweep showing that the prior boost and tube scale are not reducible to a single effective product.

The relevant implementation is in:

```text
scripts/cifar100_all_methods_iclr.py
```

with the functions:

```text
boost_ablation(...)
boost_betaperp_sweep_2d(...)
```

For Panel A, `c=50` is selected by validation NLL on the held-out clean validation split and confirmed on the test split; it is not selected from test metrics. Panel B is a sensitivity analysis demonstrating that `c` and `beta_perp` do not collapse to a single effective product.

## Reproducibility notes

The paper experiments use multiple regimes:

1. CIFAR-100 from scratch,
2. CIFAR-100-C robustness,
3. toy full-Hessian diagnostics,
4. CIFAR-100 to CIFAR-10 few-shot fine-tuning,
5. WideResNet-16-4 and VGG-11-BN architecture checks,
6. spine functional-disagreement diagnostics,
7. TRL hyperparameter and implementation ablations.

Exact command lines depend on local data and checkpoint paths. The scripts expose CLI arguments for seeds, checkpoint directories, data roots, TRL rank, spine length, tube scale, FixBN batches, and output paths.

## Final toy-table reproduction notes

The final toy Tables 3--5 are reproduced by the consolidated toy runner:

```text
toy/rerun_toy_tables.py
toy/run_final_toy_tables.sh
```

## ImageNet / ResNet-50 scale-check

The ImageNet / ResNet-50 experiment is included as a scale-check, not as the main benchmark. It uses a fixed torchvision ResNet-50 pretrained checkpoint (`IMAGENET1K_V1`), ImageNet train batches for last-layer marglik, HVP, and FixBN, and a mechanical split of the official validation set into `val_tuning_25k` and `val_test_25k`.

The protocol is documented in:

```text
docs/imagenet_resnet50_scalecheck.md
```

Main scripts:

```bash
python scripts/imagenet_marglik_fit.py \
  --train-root <imagenet_train_root> \
  --out-dir results/imagenet_resnet50_scalecheck \
  --seeds 0 1 2

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

Large ImageNet datasets, cached bases, raw JSONL files, checkpoints, and generated result files are not included in the release.
