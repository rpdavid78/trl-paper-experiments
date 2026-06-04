# Table 16: TRL backbone-prior boost ablation

Table 16 reports two boost-prior ablations for CIFAR-100 / ResNet-18.

## Panel A: 1D boost sweep

Fixed transverse scale:

```text
beta_perp = 4
c in {0, 10, 50, 100}
```

This panel tests whether an approximately isotropic head/backbone prior is sufficient. The no-boost setting `c=0` collapses, showing that the two-block prior structure is necessary.

## Panel B: joint c x beta_perp sweep

Grid:

```text
c in {25, 50, 100}
beta_perp in {2, 4, 8}
```

This panel tests whether `c` and `beta_perp` reduce to a single effective product. They do not: configurations with the same product can have very different calibration.

## Implementation notes

The ablations are implemented in:

```text
scripts/cifar100_all_methods_iclr.py
```

Relevant functions:

```text
boost_ablation(...)
boost_betaperp_sweep_2d(...)
```

The spine and transverse basis are built once. The prior boost changes the projected prior precision and hence the transverse sampling factor; `beta_perp` changes the explicit tube scale.

Do not leave temporary hardcoded triggers such as:

```python
cfg.run_boost_ablation = True
cfg.run_boost_betaperp_sweep = True
```

in the released script. Use an explicit flag, a separate ablation entry point, or a documented local edit before launching the sweep.
