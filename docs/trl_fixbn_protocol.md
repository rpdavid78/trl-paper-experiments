# TRL FixBN protocol and seed-0 audit

This note records the BatchNorm-refresh semantics used by the CIFAR-100 TRL
posterior and the controlled seed-0 audit performed before release.

## Historical and corrected modes

The reported TRL experiments used 25 posterior samples and 25 augmented
training batches per sample. The historical implementation refreshed
BatchNorm without clearing its buffers, so running means and variances carried
from one posterior sample to the next. This mode is called `rolling` in the
released code. It is order dependent and remains available for historical
reproduction:

```bash
python scripts/cifar100_all_methods_iclr.py \
  --methods trl \
  --seed 0 \
  --trl-fixbn-batches 25 \
  --trl-fixbn-mode rolling
```

The corrected `reset` mode is the canonical default. For every posterior
sample it:

1. calls `reset_running_stats()` on each BatchNorm module;
2. temporarily sets `momentum=None`, producing a cumulative average over the
   25 calibration batches;
3. restores the module's original momentum after calibration.

```bash
python scripts/cifar100_all_methods_iclr.py \
  --methods trl \
  --seed 0 \
  --trl-fixbn-batches 25 \
  --trl-fixbn-mode reset
```

The mode is passed explicitly and recorded by the canonical runner and the
CIFAR-100-C evaluator. Historical ablation/diagnostic entry points expose an
explicit `--fixbn-mode` option and retain `rolling` as their default so their
reported rows remain reproducible.

## Why reset uses cumulative averaging

Resetting the buffers while retaining the usual exponential-moving-average
momentum is not a valid independent refresh with only 25 batches. With
`momentum=0.1`, the artificial initial running variance still has weight
`0.9^25 = 0.0718`. An auxiliary seed-0 arm confirmed the resulting collapse:

| Mode | Accuracy | NLL | ECE | Brier | Entropy AUROC |
|---|---:|---:|---:|---:|---:|
| Historical rolling | 0.742200 | 0.957253 | 0.016627 | 0.354154 | 0.863862 |
| Reset + cumulative average | 0.741900 | 0.958121 | 0.016361 | 0.354103 | 0.867579 |
| Reset + EMA momentum 0.1 | 0.010000 | 4.633780 | 0.005131 | 0.990538 | 0.524167 |

The reset-plus-EMA arm is therefore a diagnostic failure mode, not a candidate
protocol.

## Controlled seed-0 audit

The released audit is `diagnostics/trl_fixbn_ab_cifar100.py`. It uses the
historical seed-0 MAP and stored 40-point TRL spine. Rolling and reset receive
the same spine anchors, transverse Gaussian vectors, augmented calibration
batches, and final ID/OOD evaluation network for every draw.

The definitive run used:

```bash
python diagnostics/trl_fixbn_ab_cifar100.py \
  --runner scripts/cifar100_all_methods_iclr.py \
  --repo-root . \
  --data-root . \
  --ckpt-dir /mnt/hd2/rpdavid/results_article_trl/checkpoints_resnet18_seed0 \
  --historical-runner /mnt/hd2/rpdavid/trl_export/code/cifar100_all_methods_iclr_article_trl.py \
  --reference-results /mnt/hd2/rpdavid/results_article_trl/article_trl_resnet18_seed0_full_tangent.jsonl \
  --source-commit 96698aad918e83a7b6789acc6f9af36da4ee7d35 \
  --seed 0 \
  --samples 25 \
  --fixbn-batches 25 \
  --fixed-beta 4 \
  --phases fixed-forward fixed-reverse pipeline-forward sweep-reverse \
  --no-reset-ema \
  --hash-large-artifacts \
  --device cuda:0 \
  --output results/trl_fixbn_seed0_definitive.json
```

The `pipeline-forward` phase preserves the rolling state left by the complete
validation sweep, selects beta separately for each arm, then evaluates a
distinct paired posterior bank on test and OOD. It therefore tests the actual
sweep-to-final dependency, not only an isolated beta-4 call initialized from
MAP.

### Fixed beta from MAP

| Metric | Rolling | Reset | Reset minus rolling | Escalation threshold |
|---|---:|---:|---:|---:|
| Accuracy | 0.742200 | 0.741900 | -0.000300 | 0.002 |
| NLL | 0.957253 | 0.958121 | +0.000868 | 0.010 |
| ECE | 0.016627 | 0.016361 | -0.000266 | 0.005 |
| Brier | 0.354154 | 0.354103 | -0.000051 | 0.005 |
| Predictive-entropy AUROC | 0.863862 | 0.867579 | +0.003717 | 0.010 |

The mean absolute ID probability difference was `1.461e-4`; the OOD value was
`2.266e-4`.

### Complete sweep-to-final pipeline

Both arms selected `beta=4`. Their validation NLLs at the selected scale were
`0.940895` (rolling) and `0.941124` (reset).

| Metric | Rolling | Reset | Reset minus rolling | Escalation threshold |
|---|---:|---:|---:|---:|
| Accuracy | 0.743000 | 0.742500 | -0.000500 | 0.002 |
| NLL | 0.955980 | 0.956806 | +0.000827 | 0.010 |
| ECE | 0.018594 | 0.020915 | +0.002321 | 0.005 |
| Brier | 0.354701 | 0.354677 | -0.000024 | 0.005 |
| Predictive-entropy AUROC | 0.872073 | 0.874560 | +0.002487 | 0.010 |

The mean absolute ID probability difference was `1.708e-4`; the OOD value was
`2.785e-4`.

## Order controls and escalation decision

For independent reset, reversing all 25 posterior draws reproduced the
probability tensors, BN buffers, and every metric exactly. Reversing the beta
grid likewise produced zero reset-arm change for accuracy, NLL, ECE, and Brier
at every beta, and selected beta remained 4.

The historical rolling arm was measurably order dependent, but the largest
fixed-beta metric change was ECE `0.001383`, below the `0.005` threshold. Its
selected beta also remained 4 in both grid orders.

The predeclared triggers were absolute changes greater than `0.002` accuracy,
`0.010` NLL, `0.005` ECE, `0.005` Brier, or `0.010` entropy AUROC. No trigger
fired in either the isolated or complete-pipeline comparison. The audit is
valid and complete, no repeat posterior bank is required, and the reported
five-seed TRL row does **not** require rerunning seeds 0--4.

## Scope and provenance

This is a controlled implementation-sensitivity audit, not a bitwise replay of
the historical seed-0 predictor. The stored spine contains parameters and
transverse bases but not the BN buffers left by HVP construction. The paired
audit also generates Gaussian vectors on CPU to guarantee identical vectors
between arms; the historical runner generated them on CUDA. Causal conclusions
therefore use reset-minus-rolling contrasts inside the audit. The comparison to
the historical seed-0 result is context only.

The executed diagnostic was subsequently hardened to reject non-finite values,
validate reset invariance across every reverse-grid metric, require the reverse
sweep for a final decision, and validate saved batch-size/class-count lineage.
Those changes affect validation and decision metadata only; the sampling and
metric-generating path is unchanged. Reapplying the hardened decision code to
the definitive JSON returns `audit_valid=true`, no triggers, and
`rerun_seeds_0_to_4=false`.

```text
Definitive audit JSON SHA-256:
  2d8aa12c9fc6c032f063ed042d9323acc35bed6e7de605d4515486385673f024
Definitive log SHA-256:
  3b108b3ef9d3f80634eac5c3ef25be7b7447bc63c6ce8dfb15c0d78d873bd9ea
Executed diagnostic SHA-256:
  3c5fdd6e6919feaf0d6b9827ff0e2d44b9c9098b967ffe079b6ca67425fbba92
Hardened released diagnostic SHA-256 at audit close:
  a5b56ed7c62c50d1e8bd6261f40a0c459a625181c83986850db7b2d33f011c23
Canonical audit runner SHA-256:
  bd819fdfd4888cbad12242573ef5db0c8efa4471127bccf37daf041fb1c06c22
Historical runner SHA-256:
  74ef1efbe8cb939f2767b3259e0bdf07919fed829234c6787d02f2d4d656d35a
MAP checkpoint SHA-256:
  cf0a7868e8f1950fbf997f45208e6f7bf37397946d786eb3a0268f943720ead2
TRL spine SHA-256:
  d6e24832ffec05179982f02a41b7dfe2ddf43b8688a0a6f96ac75d85ba57ea7b
Historical seed-0 result SHA-256:
  f4a2a532db1dbd9772e7ecf311af0e43fb139eb7317b506f1bd295112f8f56a8
Auxiliary reset-plus-EMA JSON SHA-256:
  78f6e6169ed63ba4388b92e1464a16b5cd31fdc9b3227f9d06c40f6e8f394e71
PyTorch: 2.5.1+cu121
torchvision: 0.20.1+cu121
GPU: NVIDIA RTX 6000 Ada Generation
```
