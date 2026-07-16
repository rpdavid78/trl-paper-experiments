# Table 17: TRL backbone-prior boost ablation

Table 17 reports two boost-prior ablations for CIFAR-100 / ResNet-18. The
reported uncertainty is across three independently trained MAP checkpoints
(`map_seed` 0, 1, and 2). Posterior-sampling repeats are averaged within each
checkpoint before the across-checkpoint mean and sample standard deviation.

## Panel A: 1D boost sweep

Fixed transverse scale:

```text
beta_perp = 4
c in {0, 10, 50, 100}
```

This panel tests whether an approximately isotropic head/backbone prior is
sufficient. The `c=0` setting remains subject to the documented prior floor of
5 and collapses, showing that a substantially boosted backbone block is needed.

Run all three MAP checkpoints from the repository root:

```bash
for s in 0 1 2; do
  python scripts/cifar100_all_methods_iclr.py \
    --methods trl \
    --seed "$s" \
    --ckpt-dir "checkpoints_c100_seed${s}" \
    --run-boost-ablation \
    --boost-values 0 10 50 100 \
    --boost-beta-fixed 4 \
    --boost-sampling-seeds 0 1 2 \
    --boost-results results/boost_table17_panel_a.jsonl
done
```

The script evaluates every `c` on the clean validation and test splits. Select
`c` by validation NLL; the test rows are confirmation only.

## Panel B: joint c x beta_perp sweep

Grid:

```text
c in {25, 50, 100}
beta_perp in {2, 4, 8}
```

This panel tests whether `c` and `beta_perp` reduce to one effective product.
They do not: configurations with the same product can have very different
calibration.

```bash
for s in 0 1 2; do
  python scripts/cifar100_all_methods_iclr.py \
    --methods trl \
    --seed "$s" \
    --ckpt-dir "checkpoints_c100_seed${s}" \
    --run-boost-betaperp-sweep \
    --boost-values 25 50 100 \
    --boost-beta-grid 2 4 8 \
    --boost-sampling-seeds 0 1 2 \
    --boost-results results/boost_table17_panel_b.jsonl
done
```

## Aggregation

Average stochastic repeats within each MAP checkpoint before reporting the
three-checkpoint mean and sample standard deviation:

```bash
python scripts/aggregate_results.py \
  --input results/boost_table17_panel_a.jsonl \
  --group experiment split boost beta_perp \
  --independent-unit map_seed \
  --metrics acc nll ece brier \
  --out results/boost_table17_panel_a_summary.csv

python scripts/aggregate_results.py \
  --input results/boost_table17_panel_b.jsonl \
  --group experiment split c beta_perp product \
  --independent-unit map_seed \
  --metrics acc nll ece brier \
  --out results/boost_table17_panel_b_summary.csv
```

## Implementation notes

The executable implementation is in
`scripts/cifar100_all_methods_iclr.py`, in `boost_ablation(...)` and
`boost_betaperp_sweep_2d(...)`. The spine and transverse basis are built once.
The prior boost changes projected precision and hence the transverse sampling
factor; `beta_perp` changes the explicit tube scale.

Both modes write self-describing JSONL rows containing `map_seed`,
`sampling_seed`, split, posterior sample count, and FixBN count.
