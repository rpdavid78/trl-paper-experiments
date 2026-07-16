# Table 7: Laplace prior-grid and temperature-scaling checks

This note documents two reviewer-facing calibration sanity checks for the CIFAR-100 experiments.

## 1. Last-layer Laplace prior-selection check

The main CIFAR-100 pipeline uses last-layer Laplace with Kronecker-factored curvature and empirical-Bayes prior precision optimization:

```python
la.optimize_prior_precision(method="marglik")
```

This is the standard Laplace-Redux style protocol. However, TRL selects its tube scale and backbone prior boost using held-out validation NLL. To remove the possible confound that TRL is selected by validation NLL while ELA/LLA are selected by marginal likelihood, run the validation-grid prior check:

```bash
python scripts/cifar100_laplace_prior_grid_iclr.py \
  --seed 0 \
  --ckpt-dir /path/to/checkpoints_c100_seed0 \
  --results results_iclr/cifar100_laplace_prior_grid_seed0.jsonl
```

The script reports four rows on the clean CIFAR-100 test set:

```text
ELA-marglik
LLA-marglik
ELA-grid
LLA-grid
```

where `ELA-grid` uses validation-grid prior tuning with `pred_type="nn"` and `link_approx="mc"`, and `LLA-grid` uses `pred_type="glm"` and `link_approx="probit"`. Both grid-search variants use the same held-out clean validation split as TRL.

Run several seeds with the same checkpoint convention used by the main script, for example:

```bash
for s in 0 1 2; do
  python scripts/cifar100_laplace_prior_grid_iclr.py \
    --seed $s \
    --ckpt-dir /path/to/checkpoints_c100_seed${s} \
    --results results_iclr/cifar100_laplace_prior_grid.jsonl
done
```

### Reported clean CIFAR-100 outcome

Three-seed sanity-check means:

```text
Method        Acc      NLL      ECE      Brier    Prior precision
ELA-marglik   0.7235   1.3486   0.2590   0.4669   4.93
LLA-marglik   0.7380   1.4875   0.3644   0.5194   4.93
ELA-grid      0.7417   1.0427   0.0956   0.3715   1e4
LLA-grid      0.7415   1.0421   0.0958   0.3716   1e4
```

The default `laplace-torch` grid-search interval is `torch.logspace(-4, 4, 100)`, so `lambda = 1e4` is the upper boundary of the grid. Validation-NLL tuning therefore drives the last-layer posterior toward the MAP limit rather than producing a stronger non-degenerate last-layer posterior. This explains why ELA-grid and LLA-grid are nearly identical: once the posterior variance is nearly collapsed, the nonlinear Monte Carlo and probit-GLM predictive approximations become effectively MAP-like.

This check addresses the tuning-protocol concern for last-layer Laplace. It should not be read as a new main baseline: in this CIFAR-100 regime, validation-tuned last-layer Laplace recovers MAP-like predictions rather than an improved calibrated posterior.

## 2. Scalar temperature scaling

Temperature scaling is a calibration-only MAP baseline. It fits a single scalar temperature on the held-out clean validation split by validation NLL, then evaluates the temperature-scaled MAP probabilities on the clean test set and, optionally, CIFAR-100-C.

Clean CIFAR-100:

```bash
python scripts/cifar100_temperature_scaling_iclr.py \
  --seed 0 \
  --ckpt-dir /path/to/checkpoints_c100_seed0 \
  --results results_iclr/cifar100_temperature_scaling_seed0.jsonl
```

Clean CIFAR-100 plus CIFAR-100-C:

```bash
python scripts/cifar100_temperature_scaling_iclr.py \
  --seed 0 \
  --ckpt-dir /path/to/checkpoints_c100_seed0 \
  --cifar100c-root /path/to/CIFAR-100-C \
  --results results_iclr/cifar100_temperature_scaling_with_c_seed0.jsonl
```

Across seeds:

```bash
for s in 0 1 2; do
  python scripts/cifar100_temperature_scaling_iclr.py \
    --seed $s \
    --ckpt-dir /path/to/checkpoints_c100_seed${s} \
    --results results_iclr/cifar100_temperature_scaling.jsonl
done
```

### Reported clean CIFAR-100 outcome

Three-seed sanity-check mean:

```text
Method   Acc      NLL      ECE      Brier    Temperature
MAP+TS   0.7415   0.9635   0.0296   0.3570   1.38
```

Temperature scaling is the strongest scalar calibration control. It substantially improves MAP calibration and closes the clean NLL/Brier gap to TRL within seed-to-seed variability. TRL's robust advantage over temperature scaling is lower ECE together with posterior samples and functional-variance diagnostics.

## Interpretation

These checks separate three claims:

1. Giving ELA/LLA the same validation-NLL prior-selection criterion does not produce a stronger non-degenerate last-layer posterior; it drives last-layer Laplace toward MAP-like collapse.
2. Temperature scaling is a strong scalar calibration baseline and should be reported as a sanity check for clean calibration claims.
3. Temperature scaling should not be interpreted as a posterior approximation: it preserves MAP rankings and argmax predictions and does not provide posterior samples, functional variance, or geometric support diagnostics.
