# EXPERIMENTS README

This file summarizes the experiment code, outputs, checkpoints, and interpretation used for the TRL paper revisions.

Last updated: 2026-06-04, after the backbone prior-boost ablation (Section 16 below; paper Appendix F.5 / Table 16, documenting sentence in Appendix B.4). This ablation showed the block-isotropic prior's two-block structure is necessary (c=0 collapses) and the default boost c=50 is a clear interior optimum, not cherry-picked; it also verified that the same prior construction is used unchanged across ResNet-18, WideResNet-16-4, and VGG-11-BN (head correctly detected as `self.linear` in each). Prior update: 2026-06-02, pre-registered Phase 1 longitudinal-predictability test (Section 15; one short sentence added to paper Appendix H). Earlier: WideResNet-16-4 and VGG-11-BN architecture checks (Sections 13--14; paper Appendix D.1--D.2).

## 0. Environment

Activate the main environment:

```bash
source /mnt/hd2/rpdavid/envs/trl-iclr/bin/activate
```

Common exports:

```bash
export PYTHONPATH=/mnt/hd2/rpdavid/trl_export/code:/mnt/hd2/rpdavid/trl_export:/home/rpdavid/projects/trl_iclr_code/trl_iclr_code:/home/rpdavid/projects/trl_iclr_code/trl_iclr_code/scripts:$PYTHONPATH
export TMPDIR=/mnt/hd2/rpdavid/tmp
export PIP_CACHE_DIR=/mnt/hd2/rpdavid/pipcache
```

Main directories:

```text
/mnt/hd2/rpdavid/trl_export/code
/mnt/hd2/rpdavid/trl_export/results
/mnt/hd2/rpdavid/trl_checkpoints
/mnt/hd2/rpdavid/code
/mnt/hd2/rpdavid/results_toy_spine
/mnt/hd2/rpdavid/results_finetune_spine
/mnt/hd2/rpdavid/trl_spine_disagreement_update
/home/rpdavid/projects/trl_iclr_code/trl_iclr_code
```
### Import shim used during the spine-disagreement diagnostic

The exported code expects `trl_iclr_utils.experiment_io`. On the server this package directory was absent, so a minimal shim was created under the exported code tree:

```bash
mkdir -p /mnt/hd2/rpdavid/trl_export/code/trl_iclr_utils

cat > /mnt/hd2/rpdavid/trl_export/code/trl_iclr_utils/__init__.py <<'PY'
PY

cat > /mnt/hd2/rpdavid/trl_export/code/trl_iclr_utils/experiment_io.py <<'PY'
from experiment_io import StageTimer, append_jsonl, flatten_timings
PY
```

This only forwards the already exported `experiment_io.py` utilities and was used to make `cifar100_all_methods_iclr.py` importable by the spine-disagreement script.


## 1. Main CIFAR-100 / ResNet-18 experiments

### Purpose

Main benchmark comparing MAP, Laplace, TRL, Deep Ensemble, SWAG, MC Dropout, and related baselines on CIFAR-100 with ResNet-18-CIFAR.

### Code

```text
/mnt/hd2/rpdavid/trl_export/code/cifar100_all_methods_iclr.py
/mnt/hd2/rpdavid/trl_export/code/cifar100_all_methods_base.py
```

### Core results

Typical result files:

```text
/mnt/hd2/rpdavid/trl_export/results/cifar100_seed0_core.jsonl
/mnt/hd2/rpdavid/trl_export/results/cifar100_seed1_core.jsonl
/mnt/hd2/rpdavid/trl_export/results/cifar100_seed2_core.jsonl
```

### MAP checkpoints

```text
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed0/resnet18_cifar100_map.pth
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed1/resnet18_cifar100_map.pth
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed2/resnet18_cifar100_map.pth
```

### TRL spine caches

```text
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed0/c100_trl_stage2_spine.pth
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed1/c100_trl_stage2_spine.pth
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed2/c100_trl_stage2_spine.pth
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed3/c100_trl_stage2_spine.pth
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed4/c100_trl_stage2_spine.pth
```

### Timing note

In the main CIFAR-100 core files, `runtime_total_sec` for TRL includes the shared `map_train_or_load` stage. Do not add MAP runtime again when comparing end-to-end wall-clock totals.

Important timing keys:

```text
runtime_total_sec
peak_vram_gb
time_map_train_or_load_wall_sec
time_trl_spine_construction_wall_sec
time_trl_validation_scale_sweep_wall_sec
time_trl_test_posterior_prediction_wall_sec
time_trl_ood_posterior_prediction_wall_sec
```

## 2. CIFAR-100-C robustness experiments

### Purpose

Evaluate methods on CIFAR-100-C corruptions and severities.

### Code

```text
/mnt/hd2/rpdavid/trl_export/code/cifar100c_eval_iclr.py
```

### Results

```text
/mnt/hd2/rpdavid/trl_export/results/cifar100c_seed0_snapshot.jsonl
```

This file contains one row per method/corruption/severity combination, with detailed timing keys such as:

```text
time_eval_glass_blur_1_trl_wall_sec
time_eval_glass_blur_1_trl_fixbn_overhead_wall_sec
time_eval_impulse_noise_1_deepens_wall_sec
```

## 3. CIFAR-100 fresh-refresh and single-checkpoint ablation

### Purpose

Tests whether recomputing local curvature along the spine matters, and whether the full spine adds value beyond a single checkpoint.

Compared methods:

```text
TRL-single-checkpoint
TRL-fresh-refresh-basis
```

### Code

```text
/mnt/hd2/rpdavid/trl_extra_ablation_update/code/trl_refresh_single_ablation_cifar100.py
/mnt/hd2/rpdavid/trl_extra_ablation_update/code/run_refresh_single_smoke.sh
```

### Example command

```bash
bash /mnt/hd2/rpdavid/trl_extra_ablation_update/code/run_refresh_single_smoke.sh 0 0
```

Arguments:

```text
1st argument: seed
2nd argument: GPU id
```

### Results

```text
/mnt/hd2/rpdavid/trl_results/trl_refresh_single_ablation_smoke_seed0.jsonl
/mnt/hd2/rpdavid/trl_results/trl_refresh_single_ablation_3seeds_clean.jsonl
/mnt/hd2/rpdavid/trl_results/trl_refresh_single_ablation_3seeds_clean_summary.csv
```

### Fresh-refresh cache

```text
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed0/c100_trl_fresh_refresh_spine.pth
```

### Interpretation

Fresh-refresh gives at most small gains in CIFAR-100. This supports the claim that the MAP-amortized transverse Fisher/GGN subspace is sufficient in this regime, and that recomputing local curvature along the spine adds construction cost without commensurate predictive gain.

## 4. Toy spine isolation experiments

### Purpose

Isolate the longitudinal contribution of the spine in small full-Hessian toy regimes.

Tasks:

```text
sine regression
two-moons classification
```

Compared methods:

```text
TRL-single-checkpoint
TRL-full-spine
```

### Code

```text
/mnt/hd2/rpdavid/code/toy_spine_single_vs_full.py
/mnt/hd2/rpdavid/code/run_toy_spine_10seeds.sh
```

### Results

```text
/mnt/hd2/rpdavid/results_toy_spine/toy_spine_single_vs_full_detail.jsonl
```

### Metric added

`avg_function_var` was added for both tasks.

For sine regression, it is predictive output variance across posterior samples:

```python
avg_pred_var = preds.var(dim=0, unbiased=False).mean().item()
```

For two-moons classification, it is class-probability variance across posterior samples:

```python
avg_function_var = probs_samples.var(dim=0, unbiased=False).sum(dim=1).mean().item()
```

### Interpretation

Sine:

```text
Full-spine increases avg_function_var in 10/10 seeds.
NLL improves on average but only in 6/10 seeds.
RMSE slightly worsens in 9/10 seeds.
```

Two-moons:

```text
Full-spine increases avg_function_var, entropy, and accuracy in 10/10 seeds.
NLL and Brier are close.
```

This appears in Appendix C as the controlled toy diagnostic for longitudinal dispersion.

## 5. Fine-tuning diagnostic: CIFAR-100 to CIFAR-10 few-shot

### Purpose

Find a realistic non-toy regime where the longitudinal spine contributes measurably.

Setup:

```text
Source model: ResNet-18-CIFAR MAP trained on CIFAR-100
Target: CIFAR-10 few-shot
train_per_class = 100
val_per_class = 100
test = official CIFAR-10 test set
```

### Code

```text
/mnt/hd2/rpdavid/code/finetune_cifar10_spine_smoke.py
```

Original package:

```text
/mnt/hd2/rpdavid/trl_finetune_spine_diagnostic.tar.gz
```

### Split logic

The code uses separate train, validation, and test sets:

```text
train_idx: 100 examples per class from CIFAR-10 train
val_idx: 100 examples per class from CIFAR-10 train, disjoint from train_idx
test_loader: official CIFAR-10 test set
```

`TRL-best-single-val` is selected by validation NLL, then evaluated on the official test set.

### Sample-budget logic

The posterior sample budget is matched across methods. With `n_samples=25`, full-spine uses 25 total posterior draws. For each draw, it samples a stored checkpoint index and then samples a transverse perturbation. It is not 25 samples per spine checkpoint.

### Final configuration used for 10 seeds

```text
ft_epochs = 50
ft_lr = 0.003
trl_tube_scale = 0.1
trl_steps = 20
prior_base = 5
pred_samples = 25
```

### Fine-tuned checkpoints

```text
/mnt/hd2/rpdavid/results_finetune_spine/checkpoints_ft50_lr003_10seeds
```

Expected pattern:

```text
/mnt/hd2/rpdavid/results_finetune_spine/checkpoints_ft50_lr003_10seeds/c10_finetune_from_c100_seed0_n100_ep50.pth
/mnt/hd2/rpdavid/results_finetune_spine/checkpoints_ft50_lr003_10seeds/c10_finetune_from_c100_seed1_n100_ep50.pth
...
```

### Exploratory smokes and sweeps

Initial smokes:

```text
/mnt/hd2/rpdavid/results_finetune_spine/smoke_seed0.log
/mnt/hd2/rpdavid/results_finetune_spine/smoke_seed0_tube1.log
/mnt/hd2/rpdavid/results_finetune_spine/smoke_seed0_tube05.log
/mnt/hd2/rpdavid/results_finetune_spine/smoke_seed0_tube025.log
/mnt/hd2/rpdavid/results_finetune_spine/smoke_seed0_tube01.log
```

Tube/prior/steps sweep:

```text
/mnt/hd2/rpdavid/results_finetune_spine/sweep_tube01
```

Convergence tests:

```text
/mnt/hd2/rpdavid/results_finetune_spine/convergence_test
```

Important convergence-test example:

```text
/mnt/hd2/rpdavid/results_finetune_spine/convergence_test/seed0_ft50_lr003_tube01_T20_prior5.jsonl
```

### Final 10-seed results

Directory:

```text
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds
```

Files:

```text
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed0.jsonl
...
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed9.jsonl
```

Aggregates:

```text
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/summary.csv
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/success_counts.csv
```

### Methods evaluated

```text
MAP-finetuned
TRL-single-checkpoint
TRL-full-spine
TRL-endpoint-single
TRL-best-single-val
```

### Summary used in the article

Mean over 10 seeds:

```text
MAP-finetuned:
acc 0.77668
nll 0.648900
ece 0.018337
brier 0.315211

TRL-single-checkpoint:
acc 0.77646
nll 0.648862
ece 0.018670
brier 0.315264
avg_function_var 0.000776

TRL-best-single-val:
acc 0.77719
nll 0.647131
ece 0.019034
brier 0.314214
avg_function_var 0.000937

TRL-endpoint-single:
acc 0.77752
nll 0.647006
ece 0.019105
brier 0.314129
avg_function_var 0.000963

TRL-full-spine:
acc 0.77754
nll 0.646752
ece 0.017231
brier 0.314221
avg_function_var 0.002296
```

Paired outcomes:

```text
Full-spine vs TRL-single-checkpoint:
NLL better: 10/10
Brier better: 10/10
ECE better: 8/10
avg_function_var higher: 10/10

Full-spine vs TRL-best-single-val:
NLL better: 7/10
ECE better: 8/10
Brier better: 5/10
```

### Longitudinal spine-signal diagnostic added after the 10-seed run

Purpose: measure whether the deterministic points along the stored spine move functionally, before posterior sampling. This mirrors the CIFAR-100 functional-disagreement diagnostic and supports the article's regime-diagnostic claim.

The script was patched to add:

```text
spine_longitudinal_signal(...)
--signal-only
```

Important implementation note:

```text
Do not reset BatchNorm running statistics for this deterministic diagnostic.
The first attempted version reset BN and produced CE around 2.31 on CIFAR-10,
which is near random. The corrected version keeps the checkpoint BN buffers,
giving idx=0 validation CE around 0.62, consistent with validation NLL.
```

Signal-only command for seeds 0--4 used each seed's CIFAR-100 MAP checkpoint:

```bash
for s in 0 1 2 3 4; do
  python /mnt/hd2/rpdavid/code/finetune_cifar10_spine_smoke.py \
    --seed "$s" \
    --out /mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed${s}_spine_signal_only.jsonl \
    --ckpt-dir /mnt/hd2/rpdavid/results_finetune_spine/checkpoints_ft50_lr003_10seeds \
    --cifar100-code-dir /mnt/hd2/rpdavid/trl_export/code \
    --c100-ckpt /mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed${s}/resnet18_cifar100_map.pth \
    --data-root /mnt/hd2/rpdavid/data \
    --train-per-class 100 \
    --val-per-class 100 \
    --ft-epochs 50 \
    --ft-lr 0.003 \
    --trl-tube-scale 0.1 \
    --trl-steps 20 \
    --trl-k 30 \
    --prior-base 5.0 \
    --signal-only \
    2>&1 | tee /mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed${s}_spine_signal_only.log
done
```

For seeds 5--9, the CIFAR-100 MAP checkpoints for seeds 5--9 were not present. The command used the seed-0 CIFAR-100 MAP checkpoint only to satisfy initialization; the script then loaded the already existing fine-tuned checkpoint for the target seed before building the spine.

```bash
for s in 5 6 7 8 9; do
  python /mnt/hd2/rpdavid/code/finetune_cifar10_spine_smoke.py \
    --seed "$s" \
    --out /mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed${s}_spine_signal_only.jsonl \
    --ckpt-dir /mnt/hd2/rpdavid/results_finetune_spine/checkpoints_ft50_lr003_10seeds \
    --cifar100-code-dir /mnt/hd2/rpdavid/trl_export/code \
    --c100-ckpt /mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed0/resnet18_cifar100_map.pth \
    --data-root /mnt/hd2/rpdavid/data \
    --train-per-class 100 \
    --val-per-class 100 \
    --ft-epochs 50 \
    --ft-lr 0.003 \
    --trl-tube-scale 0.1 \
    --trl-steps 20 \
    --trl-k 30 \
    --prior-base 5.0 \
    --signal-only \
    2>&1 | tee /mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed${s}_spine_signal_only.log
done
```

Per-seed outputs:

```text
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed0_spine_signal_only_seed0_spine_signal.csv
...
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed9_spine_signal_only_seed9_spine_signal.csv

/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed0_spine_signal_only_seed0_spine_signal_summary.csv
...
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/seed9_spine_signal_only_seed9_spine_signal_summary.csv
```

Aggregates written after the run:

```text
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/spine_signal_finetune_10seeds_per_seed.csv
/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds/spine_signal_finetune_10seeds_summary.csv
```

Aggregate fine-tuning spine-signal values over 10 seeds:

```text
Endpoint:
top1_disagreement_endpoint  mean 0.099500  std 0.018032
mean_js_endpoint            mean 0.017530  std 0.003625
delta_ce_endpoint           mean 0.059439  std 0.023901

Mean over spine:
top1_disagreement_mean_over_spine  mean 0.051890  std 0.010427
mean_js_mean_over_spine            mean 0.006469  std 0.001533
delta_ce_mean_over_spine           mean 0.019065  std 0.008657
```

Interpretation for the article:

```text
Compared with CIFAR-100 from scratch, fine-tuning has about 2x higher endpoint
top-1 disagreement and about 2.8x higher endpoint JS. It also has larger CE
drift. The result should be framed as a regime diagnostic, not as proof that
longitudinal movement is uniformly beneficial.
```

### Interpretation

This is an exploratory existence diagnostic, not a benchmark protocol. The correct interpretation is:

```text
The spine helps against the MAP-centered local tube.
Part of the gain comes from relocation to better centers along the spine.
Full-spine adds about 3x more functional variation.
Against endpoint-single and best-single-val, full-spine is competitive, not uniformly superior.
```

## 6. K, FixBN, tube-scale, and basis ablations

### Purpose

Analyze sensitivity to TRL hyperparameters and implementation choices.

Main axes:

```text
subspace dimension k
FixBN batches
tube scale
fixed vs transported basis
stale or refreshed eigenspaces
```

### Code

```text
/mnt/hd2/rpdavid/trl_export/code/trl_tube_scale_sensitivity_cifar100.py
/mnt/hd2/rpdavid/trl_export/code/trl_fixed_basis_ablation_cifar100.py
/mnt/hd2/rpdavid/trl_export/code/stale_eigenspace_study_cifar100.py
```

### Example results

```text
/mnt/hd2/rpdavid/trl_export/results/ablation_k_seed0_k5.jsonl
/mnt/hd2/rpdavid/trl_export/results/ablation_k_seed1_k50.jsonl
/mnt/hd2/rpdavid/trl_export/results/ablation_fixbn_seed0_fb25.jsonl
/mnt/hd2/rpdavid/trl_export/results/ablation_fixbn_seed1_fb5.jsonl
```

## 7. Spine loss profile and functional disagreement

### Purpose

Determine whether the transported spine is nearly functionally invariant, and whether the same diagnostic can distinguish regimes where longitudinal mixing helps.

The key diagnostic compares deterministic spine points to the initial spine point using validation data:

```text
top-1 disagreement
mean Jensen-Shannon divergence
delta cross-entropy / delta CE
```

`delta CE` is the change in validation cross-entropy relative to the initial spine point. For classification it is the same quantity as validation NLL on the labels, up to naming.

### CIFAR-100 from scratch / ResNet-18

### Code

```text
/mnt/hd2/rpdavid/trl_spine_disagreement_update/code/spine_functional_disagreement_cifar100.py
```

### Import dependency

The script imports `cifar100_all_methods_iclr.py`, which in turn imports `trl_iclr_utils.experiment_io`. See the environment shim above if that package directory is absent.

### Inputs

```text
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed*/resnet18_cifar100_map.pth
/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed*/c100_trl_stage2_spine.pth
```

### Command used for the final 5-seed diagnostic

```bash
export PYTHONPATH=/mnt/hd2/rpdavid/trl_export/code:/mnt/hd2/rpdavid/trl_spine_disagreement_update/code:$PYTHONPATH

python /mnt/hd2/rpdavid/trl_spine_disagreement_update/code/spine_functional_disagreement_cifar100.py \
  --seeds 0 1 2 3 4 \
  --ckpt-root /mnt/hd2/rpdavid/trl_checkpoints \
  --out-dir /mnt/hd2/rpdavid/trl_spine_disagreement_update/results_full \
  --data-root /mnt/hd2/rpdavid/data \
  --batch-size 256 \
  --num-workers 4 \
  --fixbn-batches 25 \
  --max-points 0 \
  --modes fixbn
```

### Outputs

```text
/mnt/hd2/rpdavid/trl_spine_disagreement_update/results_full/spine_functional_disagreement_cifar100_resnet18_detail.csv
/mnt/hd2/rpdavid/trl_spine_disagreement_update/results_full/spine_functional_disagreement_cifar100_resnet18_per_seed.csv
/mnt/hd2/rpdavid/trl_spine_disagreement_update/results_full/spine_functional_disagreement_cifar100_resnet18_summary.csv
```

### Summary used in the article

Final FixBN, 5-seed CIFAR-100 values:

```text
Endpoint:
top1_disagreement_endpoint  0.05084 +/- 0.00536
mean_js_endpoint            0.00631 +/- 0.00087
delta_ce_endpoint           0.01044 +/- 0.00275

Mean over spine:
top1_disagreement_mean_over_spine  0.03161 +/- 0.00280
mean_js_mean_over_spine            0.00280 +/- 0.00042
delta_ce_mean_over_spine           0.00295 +/- 0.00145
```

Per-seed endpoint ranges in this run:

```text
top1_disagreement_endpoint  0.0432 to 0.0580
mean_js_endpoint            0.00549 to 0.00776
delta_ce_endpoint           0.00691 to 0.01320
```

### Fine-tuning spine-signal diagnostic

The fine-tuning diagnostic in Section 5 applies the same deterministic-spine signal to CIFAR-10 small-data fine-tuning. The final 10-seed values are:

```text
Endpoint:
top1_disagreement_endpoint  0.09950 +/- 0.01803
mean_js_endpoint            0.01753 +/- 0.00363
delta_ce_endpoint           0.05944 +/- 0.02390

Mean over spine:
top1_disagreement_mean_over_spine  0.05189 +/- 0.01043
mean_js_mean_over_spine            0.00647 +/- 0.00153
delta_ce_mean_over_spine           0.01907 +/- 0.00866
```

Per-seed endpoint ranges in the fine-tuning run:

```text
top1_disagreement_endpoint  0.079 to 0.133
mean_js_endpoint            0.01278 to 0.02441
delta_ce_endpoint           0.02350 to 0.09350
```

### Interpretation

CIFAR-100 from scratch has a low-loss spine but only limited functional drift. This reconciles why full-spine mixing is nearly redundant with local transverse sampling in the main benchmark.

CIFAR-10 small-data fine-tuning has a substantially larger longitudinal functional signal: about 2x higher endpoint top-1 disagreement and about 2.8x higher endpoint JS than CIFAR-100 from scratch. It also has larger validation-loss drift, so the correct claim is not that longitudinal motion is uniformly beneficial. The correct claim is that TRL separates two effects:

```text
1. whether the spine remains in a low-loss corridor;
2. whether movement along that corridor induces meaningful predictive variation.
```

This diagnostic supports the hybrid-instrumental framing used in the final article revision.



## 8. Paper asset generation

### Code

```text
/mnt/hd2/rpdavid/trl_export/code/make_paper_assets.py
/mnt/hd2/rpdavid/trl_export/code/aggregate_results.py
/home/rpdavid/projects/trl_iclr_code/trl_iclr_code/trl_iclr_utils/aggregate_results.py
```

### Assets

Common assets in the article zip:

```text
fig1_discrete_spine.pdf
fig1_transport_frame.pdf
fig_tangent_transverse.pdf
fig_sine_regression_trl.png
fig_twomoons_trl.png
efficiency_pareto.pdf
eigenvalue_spectrum.pdf
final_hybrid_plot.pdf
trl_map_tubular.pdf
valley_cartoon.pdf
```

## 9. Roy / Dold / geometric Laplace repository attempt

### Purpose

Check whether an external comparison with Roy/Dold-style geometric Laplace / loss-tunnel code could be run.

### Directory

```text
/mnt/hd2/rpdavid/roy_geometric_laplace
```

### Separate environment

```text
/mnt/hd2/rpdavid/envs/roy-jax
```

### Status

JAX was installed and detected GPUs:

```text
jax 0.10.1
devices [CudaDevice(id=0), CudaDevice(id=1)]
```

Missing or problematic dependencies included:

```text
matplotlib
tree_math
flax
torch
matfree
```

A `matfree` incompatibility appeared:

```text
cannot import name 'lanczos' from 'matfree'
```

Expected external checkpoints were also absent:

```text
./checkpoints/CIFAR-10/ResNet/good_params_seed3.pickle
./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle
```

### Decision

No cross-stack empirical comparison was included. The paper instead explains that direct comparison with tunnel/diffusion samplers requires matching MAP checkpoints, architectures, data splits, posterior scales, and sampling budgets.

## 10. Article files

### Final artifacts in this conversation

Original final artifacts before the spine-signal update:

```text
/mnt/data/trl_article_final.pdf
/mnt/data/trl_article_final.zip
```

Final artifacts after adding the longitudinal spine-signal diagnostic and Appendix H table:

```text
/mnt/data/trl_article_final_spine_signal.pdf
/mnt/data/trl_article_final_spine_signal.zip
```

### Important source files inside the zip

```text
main.tex
main.bbl
trl_icml2026.bib
paper_assets/
CHANGELOG_NEW_RESULTS.md
EXPERIMENTS_README.md
```

### Recently edited sections

```text
Section 6.3
Conclusion
Appendix C
Appendix G
Appendix H
Subspace inference and loss tunnels paragraph
Cost/efficiency discussion
Longitudinal spine-signal diagnostic table in Appendix H
```

### Spine-signal article update

A small Appendix H table was added to mirror the CIFAR-100 functional-disagreement diagnostic from Appendix G. It reports the fine-tuning deterministic-spine metrics:

```text
endpoint top-1 disagreement
endpoint JS
endpoint delta CE
mean-over-spine top-1 disagreement
mean-over-spine JS
mean-over-spine delta CE
```

The article text now uses these results to support the regime-diagnostic framing:

```text
CIFAR-100 from scratch: weak longitudinal signal, full-spine mostly redundant.
CIFAR-10 small-data fine-tuning: stronger longitudinal signal, full-spine improves over single-checkpoint.
```



## 11. Useful inspection commands

### Search for important numbers and methods

```bash
grep -R -n "1985\.8\|2444\.1\|504\.6\|runtime\|TRL-full-spine\|best-single-val" \
  /mnt/hd2/rpdavid /home/rpdavid/projects/trl_iclr_code \
  --include="*.tex" --include="*.csv" --include="*.jsonl" --include="*.log" --include="*.md" \
  2>/dev/null | head -200
```

### Search for timers

```bash
grep -R -n "StageTimer\|runtime_total_sec\|time_map_train_or_load\|time_trl_spine_construction\|append_jsonl" \
  /mnt/hd2/rpdavid/trl_export/code \
  /home/rpdavid/projects/trl_iclr_code/trl_iclr_code \
  2>/dev/null | head -300
```

### Verify fine-tuning split

```bash
sed -n '70,145p' /mnt/hd2/rpdavid/code/finetune_cifar10_spine_smoke.py
sed -n '500,555p' /mnt/hd2/rpdavid/code/finetune_cifar10_spine_smoke.py
```

### Verify posterior sample budget in fine-tuning diagnostic

```bash
sed -n '226,330p' /mnt/hd2/rpdavid/code/finetune_cifar10_spine_smoke.py
```

Expected logic:

```text
n_samples is the total number of posterior draws.
Full-spine samples a checkpoint index inside that total budget.
It is not n_samples per checkpoint.
```


### Aggregate fine-tuning spine-signal CSVs

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/mnt/hd2/rpdavid/results_finetune_spine/ft50_lr003_tube01_10seeds")
files = sorted(root.glob("seed*_spine_signal_only_seed*_spine_signal_summary.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["file"] = f.name
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
cols = [
    "top1_disagreement_endpoint",
    "mean_js_endpoint",
    "delta_ce_endpoint",
    "top1_disagreement_mean_over_spine",
    "mean_js_mean_over_spine",
    "delta_ce_mean_over_spine",
]

summary = []
for c in cols:
    summary.append({
        "metric": c,
        "mean": all_df[c].mean(),
        "std": all_df[c].std(ddof=1),
    })

summary_df = pd.DataFrame(summary)
print(all_df[["seed"] + cols])
print(summary_df.to_string(index=False))

all_df.to_csv(root / "spine_signal_finetune_10seeds_per_seed.csv", index=False)
summary_df.to_csv(root / "spine_signal_finetune_10seeds_summary.csv", index=False)
PY
```

## 12. High-level experimental narrative

The final paper uses three regimes and a spine-signal diagnostic.

### Toy/full-Hessian

The longitudinal spine can be the main source of functional dispersion.

### CIFAR-100 from scratch

Most predictive gain comes from the MAP-amortized transverse Fisher/GGN subspace. The spine is useful as geometric structure and diagnostic, but it does not dominate.

The final functional-disagreement diagnostic shows:

```text
endpoint top-1 disagreement  5.08% +/- 0.54%
endpoint JS                  0.00631 +/- 0.00087
endpoint delta CE            0.01044 +/- 0.00275
```

This is a weak but nonzero longitudinal signal.

### Small-data fine-tuning

The spine becomes useful again: it improves the MAP-centered local tube and increases functional variation. However, part of the gain comes from relocation to better centers; full-spine is competitive with endpoint/best-single controls, not uniformly superior.

The final deterministic-spine diagnostic shows:

```text
endpoint top-1 disagreement  9.95% +/- 1.80%
endpoint JS                  0.01753 +/- 0.00363
endpoint delta CE            0.05944 +/- 0.02390
```

Compared with CIFAR-100 from scratch, fine-tuning has a stronger longitudinal functional signal, and this is the regime where full-spine mixing improves over the single-checkpoint posterior.

### Final framing

The article should avoid claiming that the spine is always beneficial. The stronger claim is:

```text
TRL provides a scalable posterior approximation and a diagnostic factorization
of posterior uncertainty into longitudinal and transverse components. The
relative importance of the longitudinal component is regime-dependent.
```

The spine-signal diagnostic operationalizes this by reporting both functional drift and validation-loss drift.


## 13. WideResNet-16-4 architecture-sensitivity check

### Purpose

First supplementary CNN architecture check in the paper. The goal is to test whether the CIFAR-100 calibration behavior of TRL is tied specifically to ResNet-18-CIFAR. WideResNet-16-4 keeps the same classification likelihood and CIFAR-100 protocol, but changes the convolutional architecture and normalization behavior. This is a sensitivity check, not a full architecture-scaling study.

The key design choice is that this experiment stays inside the main classification pipeline:

```text
cross-entropy HVPs
softmax posterior predictive averaging
validation-NLL model selection
accuracy / NLL / ECE / Brier / MSP-AUROC metrics
BatchNorm recalibration when applicable
```

This is why the WideResNet check is methodologically cleaner than the exploratory UCI regression route, which would require changing the likelihood, HVP loss, posterior predictive sampler, NLL definition, and metrics.

### Code / protocol

The experiment reuses the CIFAR-100 TRL machinery with a WideResNet-16-4-CIFAR model.

Relevant main pipeline:

```text
/mnt/hd2/rpdavid/trl_export/code/cifar100_all_methods_iclr.py
```

The paper reports this as Appendix D.1, "Architecture-sensitivity check: WideResNet-16-4".

### Hyperparameter selection

The main TRL structural configuration is reused, and only the tube scale is selected on the clean validation split by validation NLL using the same criterion as in the ResNet-18 experiments.

Selected value:

```text
beta_perp = 3.0 for all three WideResNet seeds
```

This is close to the ResNet-18 value:

```text
ResNet-18 beta_perp = 4.0
WideResNet-16-4 beta_perp = 3.0
```

The difference is interpreted as normal architecture-specific curvature / normalization dependence, not as a protocol change.

### Results

CIFAR-100, WideResNet-16-4-CIFAR, mean +/- std over three seeds:

```text
Method        Acc              NLL              ECE              Brier            AUROC
MAP           0.739 +/- 0.002  1.026 +/- 0.017  0.095 +/- 0.002  0.373 +/- 0.004  0.798 +/- 0.021
SWAG          0.740 +/- 0.001  1.032 +/- 0.014  0.097 +/- 0.001  0.373 +/- 0.004  0.798 +/- 0.018
MC-Dropout    0.747 +/- 0.002  0.910 +/- 0.005  0.053 +/- 0.002  0.350 +/- 0.001  0.839 +/- 0.020
ELA           0.728 +/- 0.002  1.124 +/- 0.010  0.150 +/- 0.003  0.407 +/- 0.003  0.852 +/- 0.021
LLA           0.729 +/- 0.002  1.172 +/- 0.010  0.207 +/- 0.006  0.428 +/- 0.003  0.888 +/- 0.017
TRL           0.740 +/- 0.001  0.944 +/- 0.007  0.028 +/- 0.004  0.359 +/- 0.002  0.817 +/- 0.029
```

Deep Ensembles are omitted from the WideResNet table in the paper. The table is intended to compare single-training and post-hoc baselines under an architecture change, not to repeat the full main benchmark.

### Interpretation

The result is deliberately mixed and should not be framed as a dominance claim.

TRL substantially improves over MAP, SWAG, ELA, and LLA in NLL, ECE, and Brier score. It also achieves the best ECE among the evaluated WideResNet methods:

```text
TRL ECE = 0.028 +/- 0.004
MC-Dropout ECE = 0.053 +/- 0.002
MAP ECE = 0.095 +/- 0.002
```

This supports the main paper's calibration claim beyond ResNet-18.

However, MC-Dropout is stronger in accuracy, NLL, and Brier score on WideResNet:

```text
MC-Dropout NLL = 0.910 +/- 0.005
TRL NLL        = 0.944 +/- 0.007

MC-Dropout Brier = 0.350 +/- 0.001
TRL Brier        = 0.359 +/- 0.002
```

Thus the correct reading is:

```text
The calibration effect of TRL is not specific to ResNet-18, but the relative
ordering of single-training baselines depends on architecture and regularization.
```

This is consistent with the final paper framing:

```text
TRL is a strong post-hoc calibration and likelihood correction around one MAP
solution, but it is not uniformly superior to training-time stochastic baselines
such as MC-Dropout across all architectures.
```

### Relation to the later VGG check

WideResNet is the first architecture-sensitivity check. VGG-11-BN was added later as a second check with a non-residual architecture. Together they support the cleaner strategy of using additional CNN classification architectures rather than UCI regression when testing generalization of the TRL implementation.


## 14. VGG-11-BN architecture-sensitivity check

### Purpose

Second supplementary architecture check, following the WideResNet-16-4 check in Section 13, added during the ICLR revision. The goal is to test whether TRL's calibration behavior transfers to a non-residual architecture. VGG-11-BN has no residual skip connections, which is relevant because the near-exact functional invariances that shape low-loss valleys are partly induced by residual structure. This is a generalization check, not a significance lever: it reuses the main TRL machinery verbatim and only swaps the model class.

### Code

The TRL core is embedded in the main CIFAR-100 script, not a separate module:

```text
~/projects/trl_iclr_code/trl_iclr_code/scripts/cifar100_all_methods_iclr.py
```

The VGG variant was produced by adding a VGG-11-BN model definition (head named `self.linear` to match the prior-split and last-layer Laplace plumbing, dropout present for MC-Dropout fairness) and generating a sibling script via surgical edits (swap model factory ResNetCIFAR -> VGGCIFAR, add a `cfg.architecture` field, separate checkpoint paths):

```text
~/projects/trl_iclr_code/trl_iclr_code/scripts/vgg_all_methods_iclr.py
```

### Environment / exports

```bash
export TMPDIR=/mnt/hd2/rpdavid/tmp
```

### Command used (per seed)

Run once per seed (42, 43, 44). The tube-scale grid was extended downward to (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0), a superset of the paper grid {2,3,4,6,10,20}, after an initial border-hit that turned out to be an artifact of a contaminated MAP (see operational note below). The extended grid is a superset of the main-paper grid, so selection remains comparable across architectures.

```bash
python scripts/vgg_all_methods_iclr.py --methods all --seed 42 \
  --ckpt-dir /mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_vgg_seed42 \
  --results results/cifar100_vgg.jsonl \
  --trl-tube-scales 0.5 1.0 1.5 2.0 3.0 4.0 6.0 10.0 20.0
# repeat with --seed 43 / 44 and matching --ckpt-dir checkpoints_c100_vgg_seed43 / seed44
```

MAP, the 5 ensemble members, SWAG, and MC-Dropout are trained per seed (50 epochs each); on a re-run within the same seed directory they are loaded rather than retrained. Each seed takes roughly one hour; the ensemble dominates wall-clock.

### Tube-scale selection

All three seeds selected beta_perp = 4.0 by validation NLL, with a clean interior optimum (curve descends to 4.0 and rises on both sides). This matches the ResNet-18 selection (4.0) and is within one grid step of WideResNet (3.0), indicating the transverse tempering scale is not strongly architecture-sensitive. Representative validation-NLL sweep (seed 42):

```text
0.5 -> 1.3698   3.0 -> 1.2711
1.0 -> 1.3499   4.0 -> 1.2608   <- selected
1.5 -> 1.3260   6.0 -> 1.2897
2.0 -> 1.3031  10.0 -> 1.4677
               20.0 -> 2.2636
```

### Results

```text
results/cifar100_vgg.jsonl   (3 seeds x 7 methods)
```

Aggregated mean +/- std over seeds 42, 43, 44 (CIFAR-100, VGG-11-BN):

```text
Method        Acc     NLL     ECE     Brier   AUROC
DeepEns      0.716   1.067   0.038   0.387   0.832
MAP          0.678   1.377   0.134   0.462   0.780
SWAG         0.679   1.368   0.135   0.460   0.805
MC-Dropout   0.686   1.135   0.023   0.419   0.808
ELA          0.656   1.587   0.220   0.528   0.815
LLA          0.674   1.670   0.324   0.568   0.852
TRL          0.676   1.248   0.028   0.438   0.803
```

(Std on these is small, ~0.001-0.014 depending on metric; the per-seed JSONL has exact values.)

### Interpretation

Among the strictly post-hoc methods (MAP, SWAG, ELA, LLA, TRL: applicable to an already-trained checkpoint), TRL is clearly the strongest in calibration: ECE 0.028 vs ~0.134 for MAP/SWAG (~5x better), and best NLL/Brier among them. This reproduces the central paper pattern on a non-residual architecture.

MC-Dropout, however, beats TRL on all four probabilistic axes (NLL, ECE, Brier, accuracy) on this architecture, with non-overlapping three-seed intervals (consistent, not noise). This is read through the post-hoc-vs-training-time taxonomy: MC-Dropout is single-training but NOT post-hoc (it acquires uncertainty during 50 epochs of dropout training and cannot be applied to a deterministic checkpoint), whereas TRL attaches uncertainty to a fixed checkpoint at no training-time cost. On VGG the distributed training-time uncertainty of MC-Dropout extracts more useful functional diversity than TRL's post-hoc transverse subspace. TRL is also ~2.5x the wall-clock cost of MC-Dropout here. This was checked for fairness (same S=25 sample budget; FixBN 25 vs MC-Dropout 20 is the paper-wide protocol and does not explain the gap; rank k=30 is the measured optimum per Table 11, not a guess) and confirmed to be a genuine method property on this architecture, not an implementation artifact.

In the paper this appears as Appendix D.2, as a single-training-only table mirroring the WideResNet check (Deep Ensembles omitted from the table there; the aggregate above includes the ensemble row for reference only). The post-hoc-vs-training-time taxonomy is introduced once in Section 5 and referenced by both architecture checks.

### Operational note: smoke-test checkpoint contamination

On the first real run of seed 42, MAP/ELA/LLA/MC-Dropout/TRL all reported ~5% accuracy (chance level for CIFAR-100) while DeepEns/SWAG were correct (~70%). Cause: the `--ckpt-dir` pointed at a directory that still contained 1-epoch MAP and MC-Dropout checkpoints from an earlier smoke test (`--epochs-map 1`). The real run loaded those instead of training. The fix was to delete the contaminated `vgg11bn_cifar100_map.pth`, `vgg11bn_cifar100_mcdo.pth`, and the spine built on top of them (`c100_vgg_trl_stage2_spine.pth`), keeping the correctly-trained ensemble and SWAG, then re-run. Diagnosis was by timestamp (smoke files predated the run) and accuracy (5% = 1-epoch network), not file size (state_dict byte size is identical regardless of epochs). Seeds 43/44 used fresh empty ckpt-dirs and were not at risk. Lesson: never point a real run at a ckpt-dir that has held smoke-test checkpoints.

## 15. Pre-registered Phase 1: does functional drift predict longitudinal-mixing utility?

### Purpose

A pre-registered test of an extension hypothesis, run during the ICLR revision.
The fine-tuning diagnostic (Section 5) showed that full-spine longitudinal mixing
helps in the small-data regime. Phase 1 asked a sharper question: is the *size* of
the mixing benefit predictable from the endpoint JS functional-drift signal? The
hope was that "high JS => spine mixing helps more" could become a usable criterion
and elevate the spine story to a central contribution. This was a deliberately
falsifiable hypothesis, pre-registered before any performance number was computed.

### Integrity protocol (followed)

The prediction, control, primary metric, and falsification criterion were written,
dated, and git-committed BEFORE running the performance comparison. Files:
`PREREGISTRATION_phase1.md`, `phase1_step1_geometry_only.py` (measure JS only),
`phase1_step2_performance.py` (performance; guarded to refuse running until the
pre-registration date+hash are filled and the spine cache from step 1 exists),
`phase1_step3_test_prediction.py` (applies the frozen ordinal decision rule).
Sequence enforced: geometry -> freeze ordinal ranking -> commit -> performance.

### Design (frozen)

- Four fine-tuning regimes, ONLY the trainable subspace varies, everything else
  identical to Section 5 (same backbone, CIFAR-10 100/class split, val split, T=20,
  k=30, S=25, FixBN):
    head_only (head)        JS ~0.0002
    last_block (layer4+head) JS ~0.0045
    mid_block (layer3+head)  JS measured in step 1 (intermediate)
    full (all)              JS ~0.0198 (3-seed aux); 0.0175 +/- 0.0036 (10-seed)
- R1 from-scratch = contextual reference only, NOT in the ordinal test (different
  adaptation setting; controls not identical).
- Primary metric: GAIN_NLL = NLL(val-selected single ckpt) - NLL(full-spine).
  Positive = mixing helps. Control is the validation-selected single checkpoint
  (the HARDEST control: it already captures center relocation, so any residual
  gain isolates mixing).
- Pre-registered prediction: GAIN_NLL increases monotonically with endpoint JS.
- Falsification: a disjoint inversion (non-overlapping intervals in the wrong
  direction) among the four fine-tuning regimes falsifies the ordinal prediction.

### Result: prediction FALSIFIED (reported honestly, not elevated)

The ordinal prediction did not hold. The incremental NLL gain of full-spine mixing
over the validation-selected control was small in every regime and non-monotone in
JS: the mid_block (intermediate-drift) regime showed the LEAST favorable incremental
gain, not an intermediate one. Per-regime GAIN_NLL (best-single-val minus full-spine,
positive = mixing helps), 5 seeds:

    head_only   -0.0012 +/- 0.0013   (mixing slightly hurts)
    last_block  -0.0023 +/- 0.0004   (slightly hurts)
    mid_block   -0.0043 +/- 0.0010   (hurts most -- breaks monotonicity)
    full        +0.0005 +/- 0.0010   (marginal, within noise)

### Cross-check on the 10-seed clean full-FT data (relocation vs mixing)

To confirm the mechanism, the full regime was decomposed against BOTH controls on
the clean 10-seed data (`ft50_lr003_tube01_10seeds`):

    full-spine - MAP-centered single:  NLL -0.0021 +/- 0.0006  (wins 10/10)
    full-spine - val-selected single:  NLL -0.0004 +/- 0.0008  (wins 7/10, ~noise)

A validation-selected single checkpoint captures ~82% of the NLL gain that
full-spine has over the MAP-centered (initial) checkpoint. Conclusion: the spine's
benefit is PREDOMINANTLY center relocation; longitudinal mixing adds a small
incremental gain (~18%), and that increment is NOT predicted by JS. avg_function_var
confirms mixing does inject ~2.5-2.8x more posterior variance, but that extra
variance does not translate into better mean NLL beyond a well-chosen center.

### Decision (per pre-registration rule)

DO NOT ELEVATE. The central contribution stays "TRL is the best calibrated post-hoc
correction among checkpoint-applicable methods." Appendix H was NOT wrong: it already
attributed the spine benefit mostly to relocation (Section 6.3 of the paper) and its
10-seed numbers are confirmed here.

### What went into the paper

Conservative choice: NOT a new subsection, NOT the phrase "pre-registered negative
result" in the paper. Only a single short sentence added to Appendix H, stating that
endpoint functional drift does not monotonically predict the incremental mixing gain,
so endpoint JS is best viewed as a drift diagnostic rather than a standalone
criterion. Rationale: a borderline paper should not spend a visible page on a null
result for a hypothesis that was never needed to defend TRL. The pre-registration is
kept in the repository as rebuttal ammunition, not spent in the body.

### Lesson recorded

An early 5-seed cross-protocol pull (mixing `results_finetune_spine` + `phase1`
files) over-stated the effect as "relocation, not mixing." The clean 10-seed data
corrected this to "predominantly relocation + small incremental mixing (~18%)."
Lesson: do not draw mechanism conclusions from cross-protocol seed mixes; confirm on
a single consistent protocol before writing the claim.

## 16. Backbone prior-boost ablation (block-isotropic prior)

### Purpose

Ablate the TRL backbone prior-boost factor. The TRL prior is block-isotropic:
classifier-head parameters receive the marginal-likelihood last-layer Laplace
precision `base_val`, while all non-head (backbone) parameters receive a boosted
precision `max(base_val * c, 5)`, with default boost factor `c = 50`. This factor
and the floor were previously hardcoded, undocumented, and un-ablated. This section
documents the ablation that shows (i) the two-block structure is necessary
(an approximately isotropic prior, `c = 0`, collapses), and (ii) `c = 50` is not
cherry-picked (calibration has a clear interior optimum at 50, with `c = 100`
already worsening). In the paper this is Appendix F.5 / Table 16, with a documenting
sentence in Appendix B.4.

### Code

The ablation is implemented inside the main CIFAR-100 script, reusing the existing
TRL Stage-2 spine. Five edits were made (backup at
`cifar100_all_methods_iclr.py.bak`):

```text
1. build_trl_prior_from_laplace(...)  -> parametrized with boost_factor=50.0,
   boost_floor=5.0 (defaults preserve original behavior).
2. PracticalTRLStage2.build(), spine append dict -> added "evals":
   evals.detach().cpu() so inv_sqrt_prec can be recomputed per boost without
   rebuilding the spine.
3. boost_ablation(...) function appended (placed BEFORE the `if __name__` block;
   it NameErrors if defined after the main_iclr call).
4. trl_stage2_run(), immediately after trl.build(): hook
   `if getattr(cfg, "run_boost_ablation", False): boost_ablation(...); sys.exit(0)`.
5. main, after cfg = cfg_from_args(args): `cfg.run_boost_ablation = True`
   (TEMP flag; must be removed for normal runs).
```

Key implementation fact: the prior enters ONLY in `inv_sqrt_prec` (the transverse
sampling scale); the spine trajectory `theta` and the transverse basis `N` come
from Lanczos/HVP and are independent of the prior. The ablation therefore builds
the spine once and recomputes `inv_sqrt_prec = rsqrt(clamp(evals + prior_proj))`
per boost value, reusing `N` and `evals`. This is exact, not an approximation:
reusing the spine across boosts reflects the real dependency structure.

```text
/mnt/hd2/rpdavid/trl_export/code/cifar100_all_methods_iclr.py
```

### Command (per checkpoint)

The ablation runs the full boost grid {0, 10, 50, 100} x 3 posterior samplings
internally; one process per MAP checkpoint. `--methods trl` auto-triggers the
Laplace fit (need_laplace) to produce `base_val`, and skips ensemble/SWAG/Dropout.

```bash
# checkpoint seed0
python cifar100_all_methods_iclr.py --seed 0 \
  --ckpt-dir /mnt/hd2/rpdavid/results_article_trl/checkpoints_resnet18_seed0 \
  --methods trl

# checkpoints seed1 / seed4 (parallel on two GPUs)
CUDA_VISIBLE_DEVICES=0 python cifar100_all_methods_iclr.py --seed 1 \
  --ckpt-dir /mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed1 \
  --methods trl > boost_seed1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python cifar100_all_methods_iclr.py --seed 4 \
  --ckpt-dir /mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_seed4 \
  --methods trl > boost_seed4.txt 2>&1 &
```

Note: the from-scratch MAP checkpoints available on disk were seed0, seed1, seed4
(seed2/seed3 MAPs were not present), so the three-checkpoint aggregate uses these.
beta_perp is held fixed at 4.0 inside boost_ablation to isolate the boost effect
from the tube-scale sweep. `base_val` per checkpoint: 4.9016 (seed0),
4.9494 (seed1), 4.7611 (seed4); at c=50 this gives backbone precision ~238-247.

### Results

Aggregated over the three MAP checkpoints (each entry is the mean over 3 posterior
samplings; mean +/- std is across the three checkpoints), beta_perp = 4.0:

```text
boost c   backbone prec.   Acc              NLL              ECE              Brier
0 (~iso)  ~5               0.552 +/- 0.017  1.989 +/- 0.054  0.283 +/- 0.006  0.697 +/- 0.018
10        ~49              0.726 +/- 0.001  1.135 +/- 0.011  0.161 +/- 0.006  0.419 +/- 0.005
50 (def)  ~244             0.746 +/- 0.002  0.956 +/- 0.002  0.015 +/- 0.001  0.354 +/- 0.002
100       ~487             0.747 +/- 0.001  0.964 +/- 0.005  0.040 +/- 0.003  0.354 +/- 0.003
```

Per-checkpoint ECE (shows cross-checkpoint stability of the optimum):

```text
boost c   seed0   seed1   seed4
0         0.289   0.276   0.286
10        0.153   0.167   0.164
50        0.017   0.014   0.015
100       0.044   0.038   0.038
```

### Interpretation

Two patterns, consistent across all three checkpoints. (1) The approximately
isotropic prior (c=0, where the floor of 5 makes backbone precision ~= head
precision ~5) COLLAPSES: ECE 0.283, NLL 1.989, accuracy 0.552. A uniform-strength
prior under-regularizes the backbone transverse directions and the sampled networks
degrade. This is direct evidence that the two-block structure (backbone precision
>> head precision) is necessary; an isotropic prior does not suffice. (2) Calibration
improves up to c=50 then degrades: ECE 0.283 -> 0.161 -> 0.015 -> 0.040, a clear
interior optimum at 50, with c=100 already worse. The optimum is broad in accuracy
and Brier but sharp in ECE, and its location is stable across checkpoints (cross-
checkpoint std at c=50 is 0.001 in ECE). The boost is therefore a genuine, ablated
hyperparameter, not a magic constant; c=50 is a fixed structural choice used in all
TRL runs, and the only validation-selected TRL parameter remains the tube scale
beta_perp.

### Prior consistency across architectures (verified, not a separate run)

The same block-isotropic construction (`max(base_val * 50, 5)` backbone, `base_val`
head) is used UNCHANGED across all three architecture experiments. Each of the three
tables was produced by a DIFFERENT script, and `build_trl_prior_from_laplace` was
confirmed identical (boost 50, floor 5, head-detection `"linear." in name or
"fc." in name`) in all three:

```text
ResNet-18 (Tables 1/6):  cifar100_all_methods_iclr.py
                         /mnt/hd2/rpdavid/trl_export/code/   seeds 0..4
WRN-16-4  (Table 7):     cifar100_arch_sensitivity_iclr.py  (--arch wrn16_4)
                         /mnt/hd2/rpdavid/trl_arch_sensitivity_update/code/  seeds 0,1,2
VGG-11-BN (Table 8):     vgg_all_methods_iclr.py
                         ~/projects/trl_iclr_code/trl_iclr_code/scripts/  seeds 42,43,44
```

IMPORTANT canonical-script note (corrects an earlier draft of this section):
`cifar100_arch_sensitivity_iclr.py` is the WRN script (its own
ARCHITECTURE_SENSITIVITY_README recommends `run_arch_sensitivity_wrn16_4.sh`, output
`cifar100_wrn16_4_arch_sensitivity_3seeds.jsonl`; it accepts `--arch vgg11_bn` but
that path was NOT used for the paper VGG numbers). The VGG numbers in Table 8 come
from `vgg_all_methods_iclr.py`, results at
`~/projects/trl_iclr_code/trl_iclr_code/results/cifar100_vgg.jsonl`, MAP/spine
checkpoints at `/mnt/hd2/rpdavid/trl_checkpoints/checkpoints_c100_vgg_seed42|43|44/`
(head named `self.linear`, `build_trl_prior_from_laplace` at line ~882, boost 50 /
floor 5 confirmed). A stray `checkpoints_c100_vgg11_bn_seed42/` under the
arch_sensitivity tree is a `--arch vgg11_bn` smoke, not the Table 8 run; a
`cifar100_vgg_BADMAP.jsonl` alongside the real results is the contaminated-checkpoint
artifact described in Section 14.

The head-detection regex matches all three architectures because each model class
(ResNetCIFAR, WRN, VGG11BNCIFAR) names its head `self.linear`. So the VGG head is
correctly detected and receives `base_val`, not the boost; there is no silent
"boost-on-everything" bug on VGG, and Table 8 is not compromised. Verified by
inspecting `build_trl_prior_from_laplace` and the model class definitions in all
three scripts, not by a re-run.

A one-sentence note stating this cross-architecture consistency should be added to
paper Appendix B.4 (STILL PENDING as of this update). Suggested wording: "The same
block-isotropic prior construction (head precision lambda_base, backbone precision
max(50 lambda_base, 5)) is used unchanged in the WideResNet-16-4 and VGG-11-BN
checks, with the classifier head identified as the final linear layer in each
architecture."

### Supplementary-material reproducibility note

The paper states code is available in the supplementary material. Because the three
architecture tables come from three different scripts, ALL THREE must be in the
supplement or the tables will not reproduce: `cifar100_all_methods_iclr.py` (ResNet,
also carries the boost-ablation edits of this section), `cifar100_arch_sensitivity_iclr.py`
(WRN), and `vgg_all_methods_iclr.py` (VGG). Confirm all three are packaged.

### Cleanup required

The TEMP flag must be removed before any normal run, or every run becomes an
ablation:

```bash
sed -i '/cfg.run_boost_ablation = True/d' \
  /mnt/hd2/rpdavid/trl_export/code/cifar100_all_methods_iclr.py
```

The hook at the build() site is inert without the flag (getattr default False);
edits 1-3 are harmless to keep and are reusable for re-running the ablation.
