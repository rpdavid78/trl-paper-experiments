# SWAG-Diag initializer and FixBN protocol

This note records the implementation and audit decisions for the CIFAR-100
SWAG baseline.

## Baseline identity

The released implementation is **SWAG-Diag**, not full SWAG. It collects ten
parameter snapshots and stores only per-parameter arithmetic means and second
moments. Posterior sampling therefore uses a diagonal Gaussian; no low-rank
matrix of snapshot deviations is retained.

The canonical clean-benchmark runner is
`scripts/cifar100_all_methods_iclr.py`; its paired robustness evaluator is
`scripts/cifar100c_eval_iclr.py`. The runner passes the MAP checkpoint to
`run_swag` directly, so running Deep Ensemble first under `--methods all`
cannot change the SWAG-Diag initializer. The CIFAR-100-C evaluator loads (or,
when explicitly requested, creates) statistics through the same canonical
cache/provenance path. Regression tests cover this initializer invariant.
This is an initializer guarantee only, not a claim of stochastic equality
between `--methods all` and `--methods swag`: methods run earlier in the former
command can advance the random-number-generator state.

The independent-reset FixBN implementation, versioned cache, provenance
metadata, and legacy-cache gate documented below apply to that canonical runner
and the CIFAR-100-C evaluator (including their exported mirrors). Secondary
architecture-sensitivity, VGG, and base runners, together with their
noncanonical historical snapshot counterparts, received only the
MAP-initializer correction and `SWAG-Diag` output label. They retain their own
sampling, FixBN, and cache protocols for historical reproduction.

For the canonical runner and CIFAR-100-C evaluator, new statistics use the
versioned cache name
`c100_swag_diag_map_stats_v2.pth`. Its payload records:

- `schema_version = 2`;
- `base_model_source = "MAP"`;
- `base_model_state_sha256`, a SHA-256 fingerprint of the complete MAP
  state_dict including BN buffers;
- `map_seed`;
- `swag_variant = "diagonal"`;
- `swag_epochs`;
- `swag_lr`;
- `swag_momentum`, the optimizer momentum used during collection;
- `swag_batch_size` and `swag_num_workers`, the collection-loader settings.

The loader validates every provenance and collection-protocol field listed
above, not only the MAP fingerprint. It rejects missing fields by default and
always rejects any present field whose value disagrees with the selected MAP
or current configuration. `--allow-legacy-swag-cache` waives missing legacy
metadata only; it does not waive mismatches.

## Published and corrected FixBN modes

The published five-seed SWAG-Diag row used:

```text
posterior samples: 20
FixBN batches per sample: 20
FixBN mode: rolling
initializer: MAP
```

In the canonical runner and CIFAR-100-C evaluator, `rolling` retains the
BatchNorm buffers and their original exponential-moving average momentum
between posterior samples. It remains available only for exact reproduction.
The corrected default, `reset`, calls
`reset_running_stats()` for each sampled network, uses `momentum=None` for a
cumulative average over its calibration batches, and restores the original
momentum afterward.

Exact historical path with a known legacy cache:

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

Corrected path with a newly generated, provenance-checked cache:

```bash
python scripts/cifar100_all_methods_iclr.py \
  --methods swag \
  --seed 0 \
  --swag-samples 20 \
  --swag-fixbn-batches 20 \
  --swag-fixbn-mode reset
```

## Paired seed-0 FixBN audit

The audit loaded the historical seed-0 MAP and diagonal-moment cache without
training. For each of 20 posterior draws, both arms received exactly the same
sampled parameters and the same materialized 20 augmented calibration batches.
Only the BatchNorm refresh rule differed.

The released diagnostic is `diagnostics/swag_fixbn_ab_cifar100.py`. For a
checkpoint directory containing the MAP and SWAG-Diag statistics:

```bash
python diagnostics/swag_fixbn_ab_cifar100.py \
  --ckpt-dir /path/to/checkpoints_c100_seed0 \
  --seed 0 \
  --samples 20 \
  --fixbn-batches 20
```

| Metric | Legacy rolling | Independent reset | Reset minus rolling |
|---|---:|---:|---:|
| Accuracy | 0.740800 | 0.741600 | +0.000800 |
| NLL | 1.049775 | 1.049606 | -0.000169 |
| ECE | 0.100839 | 0.100064 | -0.000774 |
| Brier | 0.370873 | 0.370815 | -0.000058 |
| Predictive-entropy AUROC | 0.871516 | 0.871344 | -0.000172 |

The mean absolute difference between the final predictive probability tensors
was `4.67e-5`. The rolling arm also reproduced the historical seed-0 result to
within approximately `0.001` on every reported metric. This is a one-seed
implementation sensitivity check, not a replacement five-seed table; its
effect is far below the predeclared escalation thresholds of `0.01` NLL and
`0.005` ECE.

Audit provenance:

```text
historical runner SHA-256:
  eb1ea717ba4d18cfe7f0fb828cf0cbde24fce3b2485de5decbf208acd6b5366d
MAP checkpoint SHA-256:
  cf0a7868e8f1950fbf997f45208e6f7bf37397946d786eb3a0268f943720ead2
SWAG-Diag statistics SHA-256:
  a307922881ee25db3f1a481845ecc7d1e5869d6c6ab5cd08bc16c9625331c7bc
audit JSON SHA-256:
  c30cfe019390b20fb9f87a4c5bfa5bf77ba2011a4aaabfd5d782d82c1ea6dac3
PyTorch: 2.5.1+cu121
GPU: NVIDIA RTX 6000 Ada Generation
```

## Consequence

The initializer bug in the former `--methods all` control flow was a release
reproducibility defect, but it did not affect the published SWAG-Diag table: the
published caches were created in separate MAP-started SWAG/MC-Dropout runs
before the ensemble checkpoints. The FixBN audit likewise shows no material
change to the seed-0 predictive result. No five-seed SWAG-Diag rerun is required
for these two corrections.
