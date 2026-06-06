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

The large-network TRL prior is block-isotropic: classifier-head parameters receive the base precision inherited from the last-layer Laplace fit, while non-head parameters receive a boosted backbone precision. The backbone prior-boost coefficient `c=50` is selected on a held-out clean validation split by validation NLL from a small sweep over `c`, and is then confirmed on the test split. The tube scale `beta_perp` is likewise selected on the held-out clean validation split by validation NLL in the CIFAR-scale pipeline. The Table 16 sweeps report both validation and test sensitivity for the boost, showing that the no-boost setting collapses and that the boost factor and tube scale do not reduce to a single effective product.


## Table 16 boost-prior ablation

The code and notes for the TRL backbone-prior boost ablation are provided in:

```text
docs/table16_boost_ablation.md
```

This documents both parts of Table 16:

- the 1D boost sweep at fixed validation-selected `beta_perp = 4`, including validation and test NLL/ECE and the no-boost `c=0` case;
- the joint `c x beta_perp` sweep showing that the prior boost and tube scale are not reducible to a single effective product.

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

The older `toy/toy_spine_single_vs_full.py` is retained as a legacy original spine diagnostic; it is not the final Tables 3--5 protocol.

In the toy experiments, ELA and LLA use full-network/full-Hessian Laplace approximations with prior precision optimized by marginal likelihood via `optimize_prior_precision(method="marglik")`. In the CIFAR-scale experiments, ELA and LLA are last-layer KFAC/KRON approximations for scalability.

TRL tube scales in the toys are fixed/adopted for the final toy protocol rather than validation-selected inside `rerun_toy_tables.py`: `beta_perp=0.005` for sine regression and `beta_perp=0.05` for two-moons. This differs from CIFAR-scale, where `beta_perp` is selected by validation NLL.

### Table 3: sine regression

Final protocol:

```text
sine noise_std = 0.15
seeds = 0--29
```

Command:

```bash
CUDA_VISIBLE_DEVICES=1 python toy/rerun_toy_tables.py \
  --task sine \
  --out-dir results/results_sine_noise015_30seeds_final \
  --seeds $(seq 0 29) \
  --sine-noise 0.15 \
  2>&1 | tee results_sine_noise015_30seeds_final.log
```

Important note: the default `--sine-noise` value in `rerun_toy_tables.py` is `0.3`; the final Table 3 numbers require the explicit `--sine-noise 0.15` flag shown above. Predictive NLL is computed with posterior functional variance plus observation variance `0.15^2`, applied identically to ELA, LLA, and TRL.

### Table 4: two-moons classification

Final protocol:

```text
two-moons noise = 0.30
n_train = 500
n_test = 1000
hidden = 16
seeds = 0--9
samples = 250
TRL: T=50, step_size=0.08, beta_perp=0.05, k=30
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python toy/rerun_toy_tables.py \
  --task two_moons \
  --out-dir results/results_twomoons_noise03_500_1000_h16_10seeds_final \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --samples 250 \
  --moons-noise 0.30 \
  --moons-epochs 3000 \
  --moons-n-train 500 \
  --moons-n-test 1000 \
  --moons-hidden 16 \
  --moons-trl-steps 50 \
  --moons-trl-step-size 0.08 \
  --moons-trl-perp-scale 0.05 \
  --moons-trl-k 30 \
  2>&1 | tee results_twomoons_noise03_500_1000_h16_10seeds_final.log
```

### Table 5: spine isolation

Final same-protocol paired diagnostic:

```text
sine noise_std = 0.15
two-moons noise = 0.30
two-moons n_train = 500
two-moons n_test = 1000
two-moons hidden = 16
seeds = 0--9
samples = 250
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python toy/rerun_toy_tables.py \
  --task all \
  --out-dir results/results_table5_spine_isolation_final_10seeds \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --samples 250 \
  --sine-noise 0.15 \
  --moons-noise 0.30 \
  --moons-epochs 3000 \
  --moons-n-train 500 \
  --moons-n-test 1000 \
  --moons-hidden 16 \
  --moons-trl-steps 50 \
  --moons-trl-step-size 0.08 \
  --moons-trl-perp-scale 0.05 \
  --moons-trl-k 30 \
  2>&1 | tee results_table5_spine_isolation_final_10seeds.log
```

Interpretation: Table 3 shows that TRL improves over LLA in both RMSE and NLL on sine regression. Table 5 isolates the spine contribution: on sine regression, the full spine substantially improves NLL and increases functional variation while RMSE remains comparable; on two-moons, full-spine and single-checkpoint are nearly tied, indicating that most of the gain in that classification regime comes from the transverse subspace.

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
