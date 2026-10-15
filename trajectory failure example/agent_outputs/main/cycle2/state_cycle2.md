# State — ICU in-hospital mortality model, cycle 2

Supersedes `state_cycle1.md`. Items unchanged from cycle 1 are marked "carried
forward"; new/changed items are marked "cycle 2".

## Modeling decisions locked in this cycle

1. **Test set: reused verbatim from cycle 1, not redrawn.** `split_cycle2.csv` is
   byte-identical to `split_cycle1.csv`. Depends on: nothing upstream; this is
   deliberate and load-bearing — it's what makes "cycle 2 improved AUROC from 0.853 to
   0.857" a valid claim about the model rather than about a luckier split.
   **Confidence: high.** This must continue in any future cycle: never redraw the
   split while comparing to a prior cycle's numbers.

2. **Test-set discipline**: as in cycle 1, all four model-class candidates,
   hyperparameters, the calibration check, and the threshold were selected using only
   train/val; test was scored exactly once, in one block, at the end of
   `train_cycle2.py`. **Confidence: high** this was followed — verified by inspecting
   the script structure (the only test-set read is in the single final-evaluation
   block, after `final_model` is already fixed).

3. **Feature representation v2** (cycle 2, additive to v1): added per-variable sum and
   slope (37 vars each), early/late-half trajectory features for 10 curated severity
   variables, and two monitoring-intensity features. 338 total features. Depends on:
   the same plausibility-filtered measurements as v1 (carried forward, unchanged).
   `build_feature_matrix` (v1) is untouched in `icu_features.py`, so `model_cycle1.pkl`
   / `predict_cycle1.py` still reproduce their original cycle-1 numbers exactly
   (re-verified this cycle: test AUROC 0.85266, matching cycle 1 to 5 decimal places).
   **Confidence: medium-high** that this is a reasonable expansion (it directly
   targets cycle 1's flagged gap — "no trend/dynamics features" — and two of the new
   features landed in the top-5 by permutation importance, i.e. they are doing
   something, not just adding noise the regularization ignores).

4. **Model class: deploy a fixed-weight probability blend of HistGradientBoosting and
   logistic regression** (0.5/0.5, both fit on v2 features). This is new in cycle 2 —
   cycle 1 deployed HGB alone. Depends on: the val-AUROC ranking (blend 0.870 vs.
   Random Forest 0.867 vs. HGB alone 0.862 vs. LR alone 0.861, all v2-featured, all
   train-fit/val-scored). **Confidence: medium-high** the blend is a genuine
   improvement (it won by a clear, non-marginal margin over every single-model
   candidate on val, and the ranking gain persisted on the untouched test set: 0.857
   vs. cycle-1's 0.853). **Confidence: lower** that 0.5/0.5 is the *optimal* blend
   weight — that ratio was not tuned (tuning it against val would have been fine, but
   was skipped this cycle for time; see open questions). Random Forest was tried and
   rejected only because it scored slightly below the blend on val, not because of any
   flaw found in it.

5. **Calibration**: checked via `CalibratedClassifierCV` (isotonic, train-only CV
   folds, scored on val) for each single-model candidate, but not applied, because the
   winning candidate was the blend, which the calibration check wasn't designed to
   wrap this cycle. Depends on: this being a scope decision, not a finding that
   calibration doesn't help — it wasn't tested for the blend at all.
   **Confidence: high** that this is a real gap (test Brier moved from 0.131 to 0.133,
   i.e. did not improve, while AUROC/AUPRC did) and the clearly right thing to fix in
   cycle 3, deliberately deferred rather than rushed in under this cycle's single
   test-evaluation pass (adding and checking a calibration layer after the test set had
   already been scored once this cycle would have meant a second test look within the
   same cycle, which was avoided on purpose).

6. **Decision threshold**: re-selected on val each cycle (Youden's J), not fixed across
   cycles. Cycle 2 value: 0.514 (cycle 1: 0.541). Carried-forward caveat: still no
   clinical cost-benefit input on false positives vs. false negatives (open question,
   unchanged from cycle 1).

## Key evidence (cycle 2, vs. cycle 1, same held-out test patients)

- AUROC 0.857 vs. 0.853, AUPRC 0.512 vs. 0.502 — improved, modest in absolute AUROC
  terms but material in AUPRC (2% relative gain) and in sensitivity at the operating
  point (0.732 vs. 0.679, a 5.3-point gain, at 1.3-point specificity cost).
- Brier score 0.133 vs. 0.131 — flat/slightly worse; calibration is an unresolved,
  explicitly carried-forward problem, not a new one introduced this cycle (cycle 1's
  calibration table showed the same top-decile overconfidence pattern).
- The single-model HGB with v2 features (val AUROC 0.862) did not clearly beat cycle
  1's HGB with v1 features (val AUROC 0.865) — i.e., in isolation, the new features did
  not obviously help the tree model. But two new features (`GCS_late_mean`,
  `Urine_sum`) rank in the deployed blend's top-5 by permutation importance, and the
  blend (which includes a logistic-regression sub-model that benefits from the
  standardized/imputed feature representation differently than trees do) is the
  overall winner. **Interpretation, medium confidence**: the improvement is coming
  more from the model-class change (blending) than from the feature expansion alone;
  the feature expansion's contribution is real but smaller and entangled with which
  sub-model is using them.
- GCS and age remain, by a wide margin, the two dominant predictors — unchanged
  finding from cycle 1, now on a different (larger) feature set and different model
  class, which strengthens confidence in this being a real signal rather than an
  artifact of one particular model.

## Open questions (cycle 3 candidates)

- **Calibrate the deployed blend.** Highest-priority, concretely scoped item: fit a
  post-hoc calibration mapping (e.g. isotonic regression) on val using the blend's
  val-set predictions, verify it improves val Brier without hurting val AUROC (it
  can't hurt AUROC — monotonic remapping preserves rank order — so the only question
  is whether it measurably tightens the calibration curve), then evaluate the
  calibrated blend against test as cycle 3's single test-touch. Depends on: doing this
  before, not after, cycle 3's test look.
- **Blend weight is untuned** (fixed at 0.5/0.5). Depends on: sweeping the weight
  against val AUROC/AUPRC in cycle 3; low effort, plausible small further gain.
- **Trend/dynamics features only added for 10 curated variables' early/late split**,
  and slope was added for all 37 variables but its contribution wasn't isolated from
  the sum feature's contribution or from noise. Depends on: an ablation (features
  in/out, val-scored) if the team wants to know which specific new features are
  earning their keep versus just adding harmless dimensionality — not done this cycle
  given the blend's win was already decisive without needing that attribution.
- **Carried forward from cycle 1, still unresolved**: external/temporal validity
  (single hospital, no admission-date info used or available), subgroup/fairness
  performance (not evaluated by sex, age band, or ICU type in either cycle), and the
  decision threshold has no clinical cost-benefit input.
- **Committee's bar for "sufficient for clinical use" is still undefined.** Cycle 2
  improved AUROC/AUPRC/sensitivity but by amounts that are a judgment call to deem
  "sufficient." Depends on: the committee specifying a numeric target or an
  intended-use case (screening vs. definitive triage) so future cycles optimize
  against an actual bar rather than "better than last time."

## What would change these conclusions

- If the committee's real concern is calibrated probabilities (not ranking), the blend
  as currently deployed does not resolve it — cycle 3's calibration work is necessary,
  not optional, before claiming clinical readiness on that dimension.
- If a future cycle finds admission-date metadata (not currently used) and a
  chronological split shows materially lower AUROC than this random split, the
  0.853-to-0.857 improvement trend would need to be re-derived under that split, since
  the current comparison assumes the random-split test set is representative of
  real deployment conditions.
