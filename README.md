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

Runs the toy experiments comparing a MAP-centered single checkpoint posterior with the full TRL spine.

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

The experiments were run with PyTorch, torchvision, scipy, scikit-learn, pandas, matplotlib, and laplace-torch.

GPU execution is recommended for all CIFAR-scale experiments.

## Data

The scripts use standard public datasets such as CIFAR-100, CIFAR-10, SVHN, and CIFAR-100-C. Dataset downloads are handled by the scripts or expected to be placed in a local data directory.

Large datasets are not included in this repository.

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

The large-network TRL prior is block-isotropic: classifier-head parameters receive the base precision inherited from the last-layer Laplace fit, while non-head parameters receive a boosted backbone precision. The prior-boost ablation is included in the release scripts and documented in the paper appendix.

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

## Not included

The following are intentionally excluded:

```text
checkpoints/
results/
logs/
datasets/
*.pth
*.pt
*.jsonl
*.csv
*.tar.gz
```

These files can be large and are ignored by `.gitignore`.

## Citation

If you use this code, please cite the associated TRL paper.

```bibtex
@article{trl2026,
  title   = {Tubular Riemannian Laplace Approximations for Bayesian Neural Networks},
  author  = {Rodrigo Pereira David},
  year    = {2026}
}
```

## License

See `LICENSE`.
