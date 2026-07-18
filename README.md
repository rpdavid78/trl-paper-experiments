# Tubular Riemannian Laplace experiments

Code release for **Tubular Riemannian Laplace Approximations for Bayesian
Neural Networks (TRL)**. The repository contains the runnable experiment code,
ablation and diagnostic entry points, aggregation utilities, and frozen Phase 1
pre-registration used by the paper.

The release intentionally excludes datasets, trained checkpoints, cached
spines/bases, raw logs, and generated result directories. Every required input
and regeneration path is documented below and in `RELEASE_MANIFEST.md`.

## Quick start

The audited environment used Python 3.12.3. Create an environment from the
repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The server used PyTorch 2.5.1 and torchvision 0.20.1 CUDA 12.1 builds. On a
CUDA 12.1 machine, install the matching wheels first if the default resolver
does not select them:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

All commands below are run from the repository root. `--quick` is available on
the main runners for smoke tests but must not be used for paper values.

## Data

| Dataset | Acquisition and expected placement | Downloaded by code? |
| --- | --- | --- |
| CIFAR-100 | `./data/cifar-100-python/` through torchvision | Yes |
| CIFAR-10 | `./data/cifar-10-batches-py/` through torchvision | Yes |
| SVHN | `./data/` through torchvision | Yes |
| CIFAR-100-C | Extract the official archive anywhere; pass the extracted directory with `--cifar100c-root`. It must contain `labels.npy` and files such as `gaussian_noise.npy`. | No |
| ImageNet-1k | Provide ImageFolder-style train and validation directories through `--train-root` and `--val-root`. | No |

ImageNet is not redistributed because access is governed by the ImageNet terms
and the dataset is very large. The code expects one class/synset directory per
class under both roots. CIFAR-100-C is likewise kept outside Git because its
arrays are large; the release contains no modified dataset copies.

## Checkpoints and outputs

The main CIFAR-100 runners train a missing MAP checkpoint and place all model
artifacts below the selected `--ckpt-dir`. With the documented convention this
is `checkpoints_c100_seed<seed>/`. The same directory subsequently supplies:

- `resnet18_cifar100_map.pth` for MAP, Laplace, diagnostics, and fine-tuning;
- ensemble, SWAG-Diag, and MC-Dropout checkpoints for their evaluations;
- `c100_trl_stage2_spine.pth` for CIFAR-100-C and spine diagnostics.

CIFAR-100-C evaluation normally reuses those checkpoints. Add
`--train-missing-baselines` only when you deliberately want the evaluator to
train a missing ensemble, SWAG-Diag, or MC-Dropout model.

ImageNet uses the fixed torchvision `ResNet50_Weights.IMAGENET1K_V1` checkpoint;
torchvision downloads it to its normal cache. The ImageNet scripts generate the
last-layer prior fit, transverse bases, caches, and JSONL outputs locally.

Generated files belong under `results/`, `checkpoints*/`, or another local path
and are ignored by Git. Re-running a command never requires a checkpoint from
the authors, but full regeneration is compute intensive.

## Paper artifact map

The table numbers refer to the current double-blind manuscript. Commands and
exact protocols are expanded in the sections that follow.

| Paper artifact | Entry point | Seeds / independent units | Inputs | Cost |
| --- | --- | --- | --- | --- |
| Tables 1, 6, and 10: clean CIFAR-100 and runtime | `scripts/cifar100_all_methods_iclr.py` | MAP seeds 0-4 | CIFAR-100, SVHN | Very heavy |
| Tables 2, 11, and 12: CIFAR-100-C | `scripts/cifar100c_eval_iclr.py` | MAP seeds 0-2 | CIFAR-100-C plus CIFAR checkpoints | Heavy evaluation |
| Tables 3-5 and toy plots | `toy/run_final_toy_tables.sh` | 30 sine seeds; 10 two-moons/spine seeds | Synthetic | Light to moderate |
| Table 7: calibration sanity checks | `scripts/cifar100_laplace_prior_grid_iclr.py`, `scripts/cifar100_temperature_scaling_iclr.py` | MAP seeds 0-2 | CIFAR checkpoints | Moderate to heavy |
| Tables 8-9: WRN-16-4 and VGG-11-BN | `scripts/cifar100_arch_sensitivity_iclr.py`, `scripts/vgg_all_methods_iclr.py` | seeds 42, 43, 44 | CIFAR-100, SVHN | Very heavy |
| Tables 13-16 and sensitivity figures | main TRL runner plus `ablation_scripts/` | three reported repeats | CIFAR checkpoints/spines | Heavy to very heavy |
| Table 15 / Figure 10: tube scale | `ablation_scripts/trl_tube_scale_sensitivity_cifar100.py` | seeds 0-2 in the released protocol | Stored TRL spine | Heavy evaluation |
| Table 17: boost prior | main runner boost flags; `docs/table17_boost_ablation.md` | MAP seeds 0-2; repeats averaged within checkpoint | CIFAR checkpoints | Very heavy |
| Tables 18-19: spine loss and functional drift | `diagnostics/spine_functional_disagreement_cifar100.py` | MAP seeds 0-4 | Stored MAP and spine | Moderate |
| Table 20: stale eigenspace | `ablation_scripts/stale_eigenspace_study_cifar100.py` | seeds 0-2 | CIFAR checkpoints | Very heavy |
| Table 21: transported vs fixed basis | `ablation_scripts/trl_fixed_basis_ablation_cifar100.py` | seeds 0-2 | Stored TRL spine | Heavy evaluation |
| Table 22: single and fresh-refresh checks | `ablation_scripts/trl_refresh_single_ablation_cifar100.py` | seeds 0-2 | Stored TRL spine | Extremely heavy for fresh refresh |
| Tables 23-24: CIFAR-10 fine-tuning/spine signal | `finetune/finetune_cifar10_spine_smoke.py` | seeds 0-9 | CIFAR-100 MAP checkpoints | Very heavy |
| Tables 25-27: ImageNet scale-check | ImageNet scripts in `scripts/` | Table 25: seeds 0-2; Tables 26-27: seed-0 diagnostics | ImageNet train/val | Extremely heavy |
| Table 28: diagnostic-control summary | synthesis of random-rank, FixBN, stale/fixed/fresh, and spine diagnostics | As above | As above | No separate run |

Figure 1 is conceptual artwork rather than a generated experimental plot.
Small generated LaTeX/PDF assets are rebuilt from local summaries with
`scripts/make_paper_assets.py`; generated assets are not committed.

## Main CIFAR-100 run

Tables 1, 6, and 10 share the five-seed core runs:

```bash
for s in 0 1 2 3 4; do
  python scripts/cifar100_all_methods_iclr.py \
    --methods all \
    --seed "$s" \
    --ckpt-dir "checkpoints_c100_seed${s}" \
    --results "results/cifar100_seed${s}_core.jsonl"
done
```

This trains five Deep Ensemble members per MAP seed in addition to MAP,
Laplace, SWAG-Diag, MC-Dropout, and TRL. To reproduce only a subset, replace
`all` with one or more of `map ela lla trl deepens swag mcdo`.

```bash
python scripts/aggregate_results.py \
  --input 'results/cifar100_seed*_core.jsonl' \
  --group dataset architecture method \
  --metrics acc nll ece brier auroc runtime_total_sec peak_vram_gb \
  --out results/cifar100_clean_5seeds_summary.csv
```

### SWAG-Diag protocol and FixBN audit

The released baseline is **SWAG-Diag**: it stores per-parameter arithmetic
means and second moments and samples a diagonal Gaussian. It does not include
the low-rank deviation matrix of full SWAG. The canonical clean-benchmark
runner, `scripts/cifar100_all_methods_iclr.py`, initializes SWAG-Diag from the
MAP checkpoint even when Deep Ensemble ran earlier under `--methods all`.

The canonical runner and `scripts/cifar100c_eval_iclr.py` use the corrected
independent-reset FixBN default, a versioned cache, and provenance validation
against the complete MAP state_dict including BatchNorm buffers. The historical
flattened snapshot received the corresponding SWAG-Diag corrections but is not
a byte-for-byte mirror or the recommended execution surface. Secondary
architecture, VGG, and base runners received the MAP-initializer correction and
`SWAG-Diag` result label, but retain their historical sampling, FixBN, and cache
protocols.

The published five-seed SWAG-Diag row used 20 posterior samples, 20 FixBN
batches, and rolling BatchNorm buffers. Reproduce that exact path only with a
known historical cache:

```bash
python scripts/cifar100_all_methods_iclr.py \
  --methods swag \
  --seed 0 \
  --swag-samples 20 \
  --swag-fixbn-batches 20 \
  --swag-fixbn-mode rolling \
  --swag-stats c100_swag_stats.pth \
  --allow-legacy-swag-cache
```

Do not enable `--allow-legacy-swag-cache` for an unknown cache; regenerate the
versioned default cache instead. A paired seed-0 audit found differences far
below the predeclared escalation thresholds, so the FixBN correction does not
require replacing the reported five-seed row. See
`docs/swag_diag_protocol.md` for exact metrics and artifact hashes, and
`diagnostics/swag_fixbn_ab_cifar100.py` for the paired audit runner.

## CIFAR-100-C

Run this after the corresponding clean checkpoints exist:

```bash
for s in 0 1 2; do
  python scripts/cifar100c_eval_iclr.py \
    --cifar100c-root /path/to/CIFAR-100-C \
    --seed "$s" \
    --ckpt-dir "checkpoints_c100_seed${s}" \
    --methods map ela lla trl deepens swag mcdo \
    --trl-tube-scale 4 \
    --results "results/cifar100c_seed${s}_main_ts4.jsonl"
done
```

The clean-validation-selected `beta_perp=4` is reused unchanged; no corrupted
test data are used for tuning. Aggregate corruption/severity rows within each
seed before taking the across-seed standard deviation:

```bash
python scripts/aggregate_results.py \
  --input 'results/cifar100c_seed*_main_ts4.jsonl' \
  --group method \
  --independent-unit seed \
  --metrics acc nll ece brier \
  --out results/cifar100c_main_ts4_3seeds_summary.csv
```

## Toy Tables 3-5

The shell driver fixes every final-paper setting and is the canonical command:

```bash
bash toy/run_final_toy_tables.sh
```

It runs 30 seeds for noisy sine regression and 10 seeds for both two-moons and
the spine-isolation comparison. Running `toy/rerun_toy_tables.py` without those
arguments uses general-purpose defaults and is not the final-paper protocol.

## Calibration Table 7

For each `s` in `0 1 2`, run both checks against the same MAP checkpoint:

```bash
python scripts/cifar100_laplace_prior_grid_iclr.py \
  --seed "$s" \
  --ckpt-dir "checkpoints_c100_seed${s}" \
  --results results/cifar100_laplace_prior_grid.jsonl

python scripts/cifar100_temperature_scaling_iclr.py \
  --seed "$s" \
  --ckpt-dir "checkpoints_c100_seed${s}" \
  --results results/cifar100_temperature_scaling.jsonl
```

See `docs/laplace_grid_temperature_scaling.md` for the selection rules and
reported interpretation. Temperature scaling can optionally evaluate
CIFAR-100-C with `--cifar100c-root`.

## Architecture Tables 8-9

```bash
for s in 42 43 44; do
  python scripts/cifar100_arch_sensitivity_iclr.py \
    --arch wrn16_4 --methods map ela lla trl swag mcdo \
    --seed "$s" --results "results/wrn16_4_seed${s}.jsonl"

  python scripts/vgg_all_methods_iclr.py \
    --methods map ela lla trl swag mcdo \
    --seed "$s" --results "results/vgg11_bn_seed${s}.jsonl"
done
```

`scripts/vgg_bn_cifar.py` contains the exact VGG-11-BN architecture used by the
VGG runner, including its initialization and dropout placement.

## TRL ablations

Rank, spine length, and FixBN grids used by Tables 13, 14, and 16 are:

```text
k_perp:       5, 10, 20, 30, 50
spine steps:  5, 10, 20, 40
FixBN batches: 1, 5, 10, 25
```

Run one forced value per output file with the main runner, keeping all other
settings at their defaults. The tube-scale rerun for Table 15 / Figure 10 is
fully specified in `docs/tube_scale_sameood_rerun.md` and uses:

```bash
python ablation_scripts/trl_tube_scale_sensitivity_cifar100.py \
  --seed 0 \
  --ckpt-dir checkpoints_c100_seed0 \
  --tube-scales 2 3 4 6 10 20 \
  --n-samples 25 \
  --fixbn-batches 25 \
  --results results/tube_scale_sensitivity.jsonl
```

Table 17 has dedicated, runnable 1D and 2D boost commands plus the required
nested aggregation protocol in `docs/table17_boost_ablation.md`.

## Spine diagnostics

Tables 18-19 use all points from the five stored spines:

```bash
python diagnostics/spine_functional_disagreement_cifar100.py \
  --seeds 0 1 2 3 4 \
  --ckpt-root . \
  --fixbn-batches 25 \
  --out-dir results/spine_functional_disagreement
```

Tables 20-22 use the following entry points for each seed in `0 1 2`:

```bash
python ablation_scripts/stale_eigenspace_study_cifar100.py \
  --seed 0 --ckpt-dir checkpoints_c100_seed0 \
  --k 30 --steps 40 --fractions 0 0.25 0.5 0.75 1 \
  --results results/stale_eigenspace.jsonl

python ablation_scripts/trl_fixed_basis_ablation_cifar100.py \
  --seed 0 --ckpt-dir checkpoints_c100_seed0 \
  --results results/trl_basis_ablation.jsonl

python ablation_scripts/trl_refresh_single_ablation_cifar100.py \
  --seed 0 --ckpt-dir checkpoints_c100_seed0 \
  --modes single fresh --include-ood \
  --results results/trl_refresh_single.jsonl
```

Fresh refresh recomputes an eigenspace at every selected spine point. Use
`--fresh-max-points 5` only for a smoke test; the paper run uses all points.

The random rank-30 negative control is documented in
`docs/random_rank30_full_network_control.md` and implemented by
`scripts/cifar100_random_rank30_baseline.py`.

## Fine-tuning Tables 23-24

The paper protocol fine-tunes for 50 epochs at learning rate 0.003 and uses a
20-step, rank-30 spine with tube scale 0.1. Run seeds 0-9, supplying the matching
CIFAR-100 source checkpoint:

```bash
python finetune/finetune_cifar10_spine_smoke.py \
  --seed 0 \
  --c100-ckpt checkpoints_c100_seed0/resnet18_cifar100_map.pth \
  --ft-mode full --ft-epochs 50 --ft-lr 0.003 \
  --train-per-class 100 --val-per-class 100 \
  --trl-steps 20 --trl-k 30 --trl-tube-scale 0.1 \
  --pred-samples 25 --fixbn-batches 25 \
  --out results/finetune_spine/seed0.jsonl
```

The frozen Phase 1 hypothesis and decision rule are preserved under
`phase1_prereg/`. Raw Phase 1 outputs and caches are intentionally excluded.

## ImageNet Tables 25-27

First estimate the last-layer marginal-likelihood precision:

```bash
python scripts/imagenet_marglik_fit.py \
  --train-root /path/to/imagenet/train \
  --out-dir results/imagenet_resnet50_scalecheck \
  --seeds 0 1 2
```

Then reproduce the three-seed single-checkpoint scale-check (Table 25):

```bash
python scripts/imagenet_resnet50_scalecheck.py \
  --train-root /path/to/imagenet/train \
  --val-root /path/to/imagenet/val \
  --out-dir results/imagenet_resnet50_scalecheck \
  --seeds 0 1 2 --rank 30 --samples 25 \
  --fixbn-batches 25 --hvp-batches 5 \
  --boost-c 50 150 450 --betas 0.5 1 1.5 2 3 4 \
  --spine-steps 0
```

Tables 26-27 are seed-0 spine-length and rank diagnostics. Their exact grids,
selection split, cache behavior, and memory notes are in
`docs/imagenet_resnet50_scalecheck.md`.

## Aggregation and paper assets

`scripts/aggregate_results.py` computes sample standard deviations (`ddof=1`).
For nested runs such as Table 17 or CIFAR-100-C, use `--independent-unit` to
average stochastic/corruption rows inside each independent seed first.

After creating the expected summary CSV names documented above, generate the
small LaTeX tables and PDF plots with:

```bash
python scripts/make_paper_assets.py \
  --results-root results \
  --out-dir results/paper_assets
```

The command reports missing optional inputs and builds every asset whose source
summary is present. `--help` is side-effect free.

## Compute expectations

The audited server had two NVIDIA RTX 6000 Ada Generation GPUs with 49,140 MiB
VRAM each (driver 610.43.02). Individual scripts generally use one GPU unless
the surrounding launcher schedules independent seeds across devices.

| Regime | Practical expectation |
| --- | --- |
| Toy experiments | CPU or one GPU; minutes to hours depending on final seeds |
| Temperature scaling | One GPU; mostly checkpoint inference |
| Main CIFAR-100 `all` | Multiple long training/post-hoc stages per seed; budget GPU-days for all five seeds |
| TRL rank/spine/fresh diagnostics | HVP/Lanczos dominated; fresh refresh is the most expensive CIFAR diagnostic |
| VGG/WRN architecture checks | Full independent training and TRL construction per seed |
| ImageNet rank/spine diagnostics | High memory and I/O; rank-30 basis is about 3.1 GB in fp32 and high-rank Lanczos needs substantially more workspace |

Exact runtime is hardware and storage dependent. Table 10 is generated from
the instrumented `runtime_total_sec`, stage timings, FixBN overhead, and
`peak_vram_gb` fields rather than from estimates in this README.

## Repository layout

```text
scripts/             Canonical CIFAR/ImageNet runners and aggregation tools
ablation_scripts/    TRL sensitivity and transported/fresh-basis diagnostics
diagnostics/         Deterministic spine loss and functional-drift evaluation
finetune/            CIFAR-100 to CIFAR-10 small-data fine-tuning experiment
toy/                 Final toy table runner and legacy spine-isolation runner
docs/                Protocol notes and reported sanity-check outcomes
phase1_prereg/       Frozen Phase 1 pre-registration provenance
```

`scripts/all_exported_code_snapshot/` is a provenance snapshot of the earlier
flattened export, not the recommended execution surface. Use the top-level
scripts listed above; compatibility shims in the snapshot point back to the
canonical implementations.

## Release and citation status

This is a double-blind review release, so author-identifying citation metadata
is intentionally not committed. A `CITATION.cff` and archival release tag should
be added after de-anonymization. See `LICENSE` for the current reuse terms; the
repository is publicly readable but the present all-rights-reserved license is
not an OSI open-source license.
