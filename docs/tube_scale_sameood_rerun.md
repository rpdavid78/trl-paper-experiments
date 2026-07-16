# Table 15 / Figure 10: tube-scale sensitivity rerun

This note documents the final CIFAR-100 / ResNet-18 tube-scale sensitivity protocol used for Table 15 and Figure 10.

## Purpose

The tube-scale sweep studies the effect of the transverse scale `beta_perp` while holding the other TRL settings fixed:

```text
k_perp = 30
T = 40
step_size = 0.01
FixBN batches = 25
posterior samples S = 25
```

The final table uses the same test/OOD evaluation path as the other CIFAR-100 ablations. In practice, each value of `beta_perp` is run as a single-value validation sweep via `--trl-tube-scales <value>`, so the reported row is the final test/OOD evaluation for that forced scale.

## Command pattern

For each seed and tube scale:

```bash
python scripts/cifar100_all_methods_iclr.py \
  --methods trl \
  --seed <seed> \
  --ckpt-dir <checkpoint_dir_for_seed> \
  --trl-tube-scales <beta_perp> \
  --results results/tube_scale_sameood/beta<beta_tag>_seed<seed>.jsonl
```

Final sweep:

```text
beta_perp in {2.0, 3.0, 4.0, 6.0, 10.0, 20.0}
seed in {0, 1, 2}
```

## Final aggregated values

Mean plus/minus standard deviation over three seeds:

| beta_perp | Acc | NLL | ECE | Brier | AUROC |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.744 ± 0.002 | 0.988 ± 0.007 | 0.067 ± 0.003 | 0.361 ± 0.002 | 0.861 ± 0.020 |
| 3.0 | 0.744 ± 0.002 | 0.965 ± 0.007 | 0.041 ± 0.004 | 0.356 ± 0.001 | 0.865 ± 0.021 |
| 4.0 | 0.743 ± 0.003 | 0.955 ± 0.006 | 0.014 ± 0.002 | 0.356 ± 0.001 | 0.870 ± 0.022 |
| 6.0 | 0.738 ± 0.004 | 0.989 ± 0.005 | 0.063 ± 0.009 | 0.370 ± 0.001 | 0.876 ± 0.022 |
| 10.0 | 0.703 ± 0.004 | 1.257 ± 0.015 | 0.206 ± 0.014 | 0.464 ± 0.006 | 0.862 ± 0.021 |
| 20.0 | 0.387 ± 0.022 | 2.551 ± 0.056 | 0.195 ± 0.016 | 0.823 ± 0.012 | 0.581 ± 0.095 |

The selected scale for the main CIFAR-100 TRL configuration is `beta_perp = 4.0`, selected by validation NLL. It also gives the best ECE and Brier in this sweep. AUROC is relatively flat across moderate scales and peaks at `beta_perp = 6.0`, indicating that the OOD-ranking optimum need not coincide with the ID likelihood/calibration optimum.

## Paper assets

The corresponding table and figure assets are generated from `results/tube_scale_sensitivity_3seeds_summary.csv` by:

```bash
python scripts/make_paper_assets.py \
  --results-root results \
  --out-dir results/paper_assets
```

This writes:

```text
results/paper_assets/tables/table_ablation_tube_scale.tex
results/paper_assets/figures/fig_ablation_tube_scale_nll.pdf
results/paper_assets/figures/fig_ablation_tube_scale_ece.pdf
```

Large raw JSONL files and generated result directories are intentionally not included in the GitHub release.
