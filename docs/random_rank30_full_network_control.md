# Random rank-30 full-network control

This note documents the diagnostic control added to the article appendix after the support-selection versus covariance-shape discussion.

## Purpose

The control tests whether the TRL calibration gain could be explained only by full-network access, low-rank sampling, the block prior, FixBN, and validation-selected tube scale, rather than by Fisher/GGN-selected transverse directions.

## Protocol

The control replaces the Fisher/GGN-selected TRL transverse subspace with a random rank-30 orthonormal full-network subspace while keeping the TRL evaluation protocol fixed:

- ResNet-18 CIFAR-100 MAP checkpoint protocol;
- same block-isotropic prior used by TRL;
- prior boost `50` and prior floor `5`;
- rank `30`;
- `S=25` predictive samples;
- FixBN with 25 batches;
- explicit `--fixbn-mode rolling` for reproduction of the reported control
  (`reset` is available for the corrected independent-refresh protocol);
- validation-selected tube scale;
- original scale grid `beta_perp in {2,3,4,6,10,20}` plus one provenance-clean extension to `beta_perp=40` for basis seed 9000 on MAP seed 0.

The random basis is built from i.i.d. Gaussian draws and orthonormalized by two modified Gram-Schmidt reorthogonalization passes in parameter space. The five-basis sweep is across random bases on a fixed MAP checkpoint, not an across-MAP-seed estimate. The two additional one-basis controls use separate MAP checkpoints.

Example for one basis on MAP seed 0:

```bash
python scripts/cifar100_random_rank30_baseline.py \
  --seed 0 \
  --ckpt-dir checkpoints_c100_seed0 \
  --rank 30 \
  --basis-seed 9000 \
  --basis-device cuda \
  --cache-path results/random_rank30/cache_map0_basis9000.pt \
  --trl-tube-scales 2 3 4 6 10 20 \
  --samples 25 \
  --fixbn-batches 25 \
  --hvp-batches 5 \
  --results results/random_rank30/map0_basis9000.jsonl
```

Repeat with basis seeds 9000-9004 for the fixed MAP-seed-0 control. The two
additional checkpoint controls use MAP seeds 1 and 2 with basis seeds 9001 and
9002 respectively. Add `40` to `--trl-tube-scales` only for the documented
provenance-clean extension.

## Clean controls

### Five random bases on MAP seed 0

```text
basis 9000 acc 0.7413 nll 1.0377 ece 0.0951 brier 0.3693 auroc 0.8702 beta 20.0
basis 9001 acc 0.7414 nll 1.0377 ece 0.0950 brier 0.3693 auroc 0.8701 beta 20.0
basis 9002 acc 0.7412 nll 1.0376 ece 0.0951 brier 0.3694 auroc 0.8703 beta 20.0
basis 9003 acc 0.7412 nll 1.0379 ece 0.0952 brier 0.3694 auroc 0.8708 beta 20.0
basis 9004 acc 0.7414 nll 1.0382 ece 0.0951 brier 0.3695 auroc 0.8699 beta 20.0
```

Summary:

```text
Acc              0.7413 +/- 0.0001
NLL              1.0378 +/- 0.0002
ECE              0.0951 +/- 0.0001
Brier            0.3694 +/- 0.0001
AUROC            0.8703 +/- 0.0003
best_tube_scale  20.0000 +/- 0.0000
```

The deviations are population standard deviations across basis seeds on a fixed MAP checkpoint.

### One-basis controls on MAP seeds 1 and 2

```text
seed 1 basis 9001 acc 0.7432 nll 1.0185 ece 0.0867 brier 0.3659 auroc 0.8334 beta 20.0
seed 2 basis 9002 acc 0.7430 nll 1.0241 ece 0.0855 brier 0.3701 auroc 0.8679 beta 20.0
```

### Aggregate over seven controls

```text
n=7
Acc              0.7418 +/- 0.0008
NLL              1.0331 +/- 0.0076
ECE              0.0925 +/- 0.0041
Brier            0.3690 +/- 0.0013
AUROC            0.8647 +/- 0.0128
best_tube_scale  20.0000 +/- 0.0000
```

All seven original-grid controls select `beta_perp=20` and remain MAP-like in NLL and calibration. MSP-AUROC overlaps the MAP and TRL ranges in the main CIFAR-100 table, consistent with AUROC being less sensitive than NLL/ECE to the calibration mechanism isolated by this diagnostic.

## Provenance-clean beta=40 extension

The beta=40 row was rerun after the earlier disk-full append failure. The final JSONL row was written successfully.

Validation sweep:

```text
beta=2   val_nll 1.0160642862
beta=3   val_nll 1.0160053968
beta=4   val_nll 1.0159319639
beta=6   val_nll 1.0157116652
beta=10  val_nll 1.0150005817
beta=20  val_nll 1.0117554665
beta=40  val_nll 0.9998667836
```

Selected test metrics:

```text
Acc   0.7418000102
NLL   1.0230919123
ECE   0.0887127295
Brier 0.3664085865
AUROC 0.87134835
beta  40.0
```

The correct interpretation is that the random full-network rank-30 posterior is weakly responsive to scale, not completely flat. Even across a 20x scale range from beta=2 to beta=40, validation NLL improves by only about 0.016 and the test predictor remains MAP-like. By contrast, the Fisher/GGN-selected TRL subspace reaches the calibrated regime around beta=4 and collapses at beta=20, indicating that the selected TRL directions are functionally loaded.

## Article-text consequence

The appendix text should support the following conclusion:

```text
Full-network access, low-rank sampling, block prior, FixBN, and validation-selected beta are jointly insufficient. Random rank-30 full-network sampling stays MAP-like. The calibration gain comes from Fisher/GGN-selected transverse directions.
```
