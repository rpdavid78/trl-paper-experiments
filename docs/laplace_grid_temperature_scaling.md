# Laplace prior-grid and temperature-scaling checks

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
  --results results_iclr/cifar100_laplace_prior_grid_seed0.jsonl
```

The script reports four rows on the clean CIFAR-100 test set:

```text
ELA-marglik
LLA-marglik
ELA-grid
LLA-grid
```

where `ELA-grid` uses `optimize_prior_precision(method="gridsearch", val_loader=val_loader, pred_type="nn")`, and `LLA-grid` uses `pred_type="glm"`. Both grid-search variants use the same held-out clean validation split as TRL.

Run several seeds with the same checkpoint convention used by the main script, for example:

```bash
for s in 0 1 2; do
  python scripts/cifar100_laplace_prior_grid_iclr.py \
    --seed $s \
    --results results_iclr/cifar100_laplace_prior_grid.jsonl
done
```

## 2. Scalar temperature scaling

Temperature scaling is a calibration-only MAP baseline. It fits a single scalar temperature on the held-out clean validation split by validation NLL, then evaluates the temperature-scaled MAP probabilities on the clean test set and, optionally, CIFAR-100-C.

Clean CIFAR-100:

```bash
python scripts/cifar100_temperature_scaling_iclr.py \
  --seed 0 \
  --results results_iclr/cifar100_temperature_scaling_seed0.jsonl
```

Clean CIFAR-100 plus CIFAR-100-C:

```bash
python scripts/cifar100_temperature_scaling_iclr.py \
  --seed 0 \
  --cifar100c-root /path/to/CIFAR-100-C \
  --results results_iclr/cifar100_temperature_scaling_with_c_seed0.jsonl
```

Across seeds:

```bash
for s in 0 1 2; do
  python scripts/cifar100_temperature_scaling_iclr.py \
    --seed $s \
    --cifar100c-root /path/to/CIFAR-100-C \
    --results results_iclr/cifar100_temperature_scaling_with_c.jsonl
done
```

## Interpretation

These checks separate two claims:

1. Whether ELA/LLA remain worse than TRL after giving last-layer Laplace the same validation-NLL selection criterion.
2. Whether TRL's calibration gains exceed a cheap scalar calibration baseline.

Temperature scaling should not be interpreted as a posterior approximation: it preserves MAP rankings and argmax predictions and does not provide posterior samples, functional variance, or geometric support diagnostics.
