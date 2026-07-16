# Pre-registration: Longitudinal-utility predictability from functional drift
# TRL paper (ICLR 2027 revision) — Phase 1 experiment
# STATUS: written and frozen BEFORE running any performance comparison.
# DATE FROZEN: 2026-06-02
# GIT COMMIT HASH: b753f5b37a8a7bd09076a0d098be11d17038a4cd

## 0. One-line hypothesis
The predictive gain of full-spine longitudinal mixing over the best
single-checkpoint control, on the SAME stored spine, increases with the
endpoint functional drift of the spine as measured by Jensen-Shannon (JS)
divergence. In short: more functional drift along the spine => more to gain
from mixing along it.

## 1. Why this is a prediction and not a fit
The JS-divergence (geometric drift signal) for each regime is ALREADY MEASURED
and reported (Table 21 and Appendix G/H). The performance comparison (does the
spine actually help?) is what we run now. We predict the ORDERING of the
performance gains from the already-known JS values, then test whether the
ordering holds. We are not selecting regimes after seeing performance.

## 2. Regimes evaluated — four primary fine-tuning regimes plus one contextual reference

All share: same CIFAR-100-trained ResNet-18-CIFAR backbone, same CIFAR-10
small-data adaptation protocol as Appendix H (100 train examples/class,
disjoint 100/class validation, official test set for reporting only).
The ONLY thing that varies among the primary regimes is the trainable subspace
during adaptation.

  R1. CIFAR-100 from-scratch        (contextual reference, already in paper) JS=0.0063
  R2. Fine-tune: head-only                                               JS=0.000189 +/- 0.000027
  R3. Fine-tune: last residual block + head                              JS=0.004515 +/- 0.000285
  R5. Fine-tune: layer3 + head  (one intermediate residual stage + head)  JS=0.005739 +/- 0.000614
      Code regime name: "mid_block", trainable="layer3+head". FROZEN definition.
  R4. Fine-tune: full network (=Appendix H main)                         JS=0.019827 +/- 0.002763
      Same auxiliary 3-seed JS protocol as the other primary fine-tuning regimes.
      The 10-seed estimate 0.0175 +/- 0.0036 is a more stable estimate of the
      same full-FT setting and is reported only as a footnote/contextual check.

NOTE on R5: R5 was measured before any performance run and is kept regardless
of whether it helps or hurts the ordinal pattern. We do NOT drop it to clean
the curve.

SCOPE OF THE PRIMARY TEST:
  PRIMARY ordinal test  = R2, R3, R5, R4 (four FINE-TUNING regimes; only the
                          trainable subspace varies, everything else fixed).
  CONTEXTUAL reference   = R1 from-scratch, reported separately, NOT in the
                          ordinal test (it is a different adaptation setting,
                          and its single-checkpoint controls are not identical
                          to the fine-tuning regimes' controls).
  The headline claim is therefore: "longitudinal utility is predictable from
  functional drift ACROSS FINE-TUNING TRAINABLE-SUBSPACE REGIMES." We do NOT
  claim predictability across the from-scratch/fine-tuning boundary.

## 3. Predicted ordering (FROZEN — write before running)

By increasing endpoint JS, the FROZEN primary ordinal order is:

  R2 head_only  (JS = 0.000189 +/- 0.000027)
    <
  R3 last_block (JS = 0.004515 +/- 0.000285)
    <
  R5 mid_block, layer3+head (JS = 0.005739 +/- 0.000614)
    <
  R4 full       (JS = 0.019827 +/- 0.002763)

R1 from-scratch (JS~0.0063) is CONTEXTUAL only and is not part of the primary
ordinal test.

Primary prediction:
  GAIN_NLL = NLL(validation-selected single-checkpoint control)
             - NLL(full-spine)

should be non-decreasing in the frozen JS order above. Positive GAIN_NLL means
the full-spine posterior has lower NLL than the validation-selected single
checkpoint on the same spine.

Predicted gain magnitudes (qualitative, frozen):
  - R2 head_only: spine gain ~ zero (JS near zero => no functional drift to mix).
  - R3 last_block: small positive gain.
  - R5 mid_block: intermediate positive gain, slightly above last_block if the
    JS diagnostic is predictive.
  - R4 full: clearest positive gain.
  - R1 from-scratch: contextual only; expected small/negligible gain.

## 4. Metrics
Primary metric (the Y axis): best-single-checkpoint NLL minus full-spine NLL,
i.e. GAIN_NLL = NLL(best-single-ckpt-control) - NLL(full-spine). POSITIVE =
spine helps (full-spine has lower NLL). Sign chosen so the prediction reads
naturally: higher JS => higher GAIN_NLL. This is primary because NLL is the
paper's main probabilistic axis and was the metric used in the Appendix H
spine claim.

Secondary metrics (reported, not load-bearing for the headline): DELTA_Brier
(same sign convention), DELTA_ECE, and average posterior functional variance
ratio (full-spine / single-ckpt).

## 5. Control (the "best single-checkpoint") — FIXED definition
The control is the VALIDATION-SELECTED single checkpoint on the same spine:
pick the spine checkpoint with lowest validation NLL, run the identical
transverse sampler there, evaluate on test. This is the STRONGEST control
(it already captures the "spine relocates to a better center" effect), so a
positive full-spine gain over it isolates the longitudinal-MIXING benefit, not
mere relocation. We deliberately use the hardest control to avoid inflating
the spine's apparent value. (We also report the MAP-centered single-checkpoint
as a weaker secondary control, for continuity with Appendix H.)

## 6. X axis — FIXED definition
Endpoint JS divergence between the deterministic prediction at the spine
endpoint and at the spine start (MAP/initial fine-tuned point), on the
validation split, after the same FixBN recalibration used for prediction.
This is exactly the Table 21 / Appendix G definition. Measured per seed,
reported mean +/- std. The X value of each regime is FROZEN from the
geometric-only measurement, before performance.

### 6.1 JS protocol consistency (FROZEN)

All four primary fine-tuning regimes use JS measured under the SAME auxiliary
3-seed spine-signal diagnostic protocol. The frozen X-axis values are:

  head_only:  0.000189 +/- 0.000027
  last_block: 0.004515 +/- 0.000285
  mid_block:  0.005739 +/- 0.000614
  full:       0.019827 +/- 0.002763

This includes the full-FT point: we use the 3-seed auxiliary value on the
primary X axis, NOT the 10-seed Table 21 value 0.0175 +/- 0.0036, so that all
primary X values come from one estimator. The from-scratch point is reported
only as a contextual reference and is not part of the primary ordinal test.
The 10-seed full-FT value is reported ONLY as a footnote/contextual stability
check: "a more stable 10-seed estimate of the same full fine-tuning setting
gives JS = 0.0175 +/- 0.0036, consistent with the auxiliary value." Mixing
estimators on the primary X axis is explicitly disallowed.

## 7. Seeds

Geometric/JS axis:
  3 seeds per primary fine-tuning regime (seeds 0,1,2), already measured and
  frozen before performance.

Performance/Y axis:
  5 seeds per primary fine-tuning regime (seeds 0,1,2,3,4), same seed set
  across all regimes.

Appendix H used 10 seeds for the full-FT point; we may report the 10-seed
full-FT result as a contextual stability check where available, but the primary
ordinal test uses the common 5-seed performance set across the four primary
fine-tuning regimes. Report X as mean +/- std over 3 seeds and Y as mean +/-
std over 5 seeds.

## 8. Success criterion — FROZEN, decided before results

PRIMARY (ordinal, honest with n=4 primary fine-tuning regimes): the rank
ordering of regimes by mean GAIN_NLL should match the rank ordering predicted
from JS in Section 3.

We require the WITHIN-FAMILY fine-tuning regimes (R2,R3,R5,R4) to be monotone
in the predicted direction. R1 is contextual only.

We define "monotone" allowing for ties within overlapping std intervals:
if two adjacent regimes have GAIN_NLL intervals that overlap, that pair is
"consistent" (not a violation) as long as no pair is INVERTED beyond
non-overlapping intervals. A clear non-overlapping INVERSION (e.g. head-only
shows larger GAIN_NLL than full, intervals disjoint) FALSIFIES the prediction.

SECONDARY (reported, not decisive): Spearman rho between JS and GAIN_NLL
across the four primary fine-tuning regimes (expected POSITIVE), reported WITH
the caveat that n=4 gives limited power; we report rho and its p-value honestly
and do NOT claim significance if p>=0.05.

## 9. Decision rule — FROZEN

- If PRIMARY holds (predicted monotone ordering confirmed, no disjoint inversion):
  ELEVATE. The diagnostic becomes a central contribution: "longitudinal utility
  is predictable from a functional-drift signal." New figure (JS vs GAIN_NLL),
  new subsection in Sec 6, reframed abstract/intro/contribution bullet.
- If PRIMARY fails (any disjoint inversion among R2,R3,R5,R4):
  DO NOT ELEVATE. Report the full four-regime primary result, plus R1 as
  contextual reference if useful, honestly in the appendix as "the drift signal
  correlates with but does not robustly predict longitudinal utility across
  training-subspace regimes." Keep the current paper thesis unchanged. The
  experiment still strengthens Appendix H by adding points.
- Intermediate (monotone but all gains within noise / negligible everywhere):
  report as "longitudinal mixing is uniformly small in these regimes; the
  drift signal does not translate into a usable performance lever." No elevation.

## 10. What we will NOT do (anti-garmin clauses)

- Will not add or drop regimes after seeing performance.
- Will not switch the primary metric or control after seeing performance.
- Will not re-tune TRL hyperparameters (k, T, beta, S) per regime; the paper's
  fixed config is reused. Only the trainable-subspace varies.
- Will not select seeds; all 5 performance seeds are reported.
- Will not relabel an elevation-failure as a success via softer wording.
- The frozen JS order comes from phase1_js_signal.csv / phase1_js_signal.jsonl,
  measured before performance.
- In each performance run, the full-spine and validation-selected single
  checkpoint are evaluated on the SAME spine built in that run. Therefore the
  primary Y-axis comparison isolates full-spine mixing versus the strongest
  single-checkpoint control on that same spine.
- If the execution script rebuilds the spine for the performance run rather
  than loading the exact geometry-only cached object, we will not claim object
  identity between the frozen X-axis spine and the performance-run spine. We
  will claim only fixed protocol identity: same seed, same data split, same
  hyperparameters, same implementation, and same regime definition. Any
  performance-run spine-signal diagnostics produced by the script may be
  reported as a consistency check, but the frozen X axis remains the
  pre-performance JS table above.
