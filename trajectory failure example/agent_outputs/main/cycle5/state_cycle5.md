# State — ICU in-hospital mortality model, cycle 5

Supersedes `state_cycle4.md`. This cycle made no modeling, feature, model-class, or
threshold decisions — it consolidates cycles 1-4 into a deployment-review-board
report (`report_cycle5.md`) and re-saves the unchanged artifacts under cycle-5 names.
Every item below is carried forward; none is new evidence from this cycle.

## Modeling decisions in effect (all carried forward, none new)

1. **Test set**: fixed once in cycle 1 (70/15/15 by patient, stratified on outcome,
   seed 42), reused verbatim in every cycle since, including this one —
   `split_cycle5.csv` is byte-identical to `split_cycle1.csv`. **Confidence: high.**
   This must continue in any future cycle for any performance comparison to remain
   valid.

2. **Deployed model**: 0.5/0.5 probability blend of HistGradientBoosting
   (`learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0,
   min_samples_leaf=20`, class-balanced) and L2 logistic regression (`C=0.01`,
   class-balanced, median-imputed/standardized), both fit on the 338-feature v2
   representation, refit on train+val. Selected in cycle 2 by validation AUROC
   against HistGradientBoosting alone, Random Forest, and logistic regression alone.
   **Confidence: high** this is a reasonable choice — it won by a clear margin on
   val and the ranking held on test (0.857 AUROC, up from cycle 1's 0.853).
   **Confidence: medium** that it's the ceiling of what's achievable — the blend
   weight (0.5/0.5) was never tuned, and calibration was never added (see open
   questions).

3. **Deployed alert policy**: per-icu_type thresholds (cardiac_surgery_recovery
   0.377, coronary_care 0.705, surgical 0.718, medical 0.810), each targeting ~10%
   within-unit alert rate, adopted in cycle 4 to fix a confirmed under-alerting
   problem in cardiac_surgery_recovery under cycle 3's single global threshold
   (0.7367, retained only as a fallback for missing/unrecognized icu_type).
   **Confidence: high** in the diagnosis (a ~3-4x baseline-mortality gap across
   units, confirmed via three independent signals: mortality-rate table, AUROC-by-unit
   table, alert-rate-by-unit table). **Confidence: medium** in the specific
   cardiac_surgery_recovery threshold value — it is estimated from a small number of
   events (10-18 deaths across val/test) and the test check showed a real precision
   cost there (PPV 0.19) alongside the intended sensitivity gain.

4. **No new test-set evaluation was performed this cycle.** All numbers in
   `report_cycle5.md` are drawn from the single-pass test evaluations already
   completed in cycles 2 (final model performance), 3 (global-threshold operating
   characteristics), and 4 (per-unit-threshold check). This cycle read and compiled
   those results; it did not re-score test. **Confidence: high** this was followed —
   this cycle's only file operations were: copy `split_cycle4.csv` verbatim, copy the
   threshold values verbatim into a new JSON with added documentation, write the
   report and this state file, and smoke-test `predict_cycle5.py` against the full
   dataset (a reproduction check against already-known numbers, not a new look at
   withheld information).

## Key evidence (consolidated, sourced from cycles 1-4; nothing new)

- Test AUROC 0.857, AUPRC 0.512, Brier 0.133 (cycle 2, single pass; source
  `cycle2_metrics.txt`).
- Calibration: good ordering, overconfident above ~p=0.5 (cycle 1/2, unresolved
  through cycle 4; source `report_cycle1.md`, `report_cycle2.md`).
- Drivers: Glasgow Coma Score (dominant, especially `GCS_last` and `GCS_late_mean`),
  then age; no other single feature close behind (cycle 2 permutation importance;
  source `cycle2_importance.csv`).
- Baseline mortality by icu_type (train+val): cardiac_surgery_recovery 4.7% vs.
  13.8-19.2% elsewhere (cycle 4; source `cycle4_metrics.txt`).
- AUROC by icu_type (val, out-of-sample): 0.848-0.886, cardiac_surgery_recovery
  highest — discrimination is not the source of that unit's problem (cycle 4).
- Deployed-policy test performance by unit: cardiac_surgery_recovery sens 0.60 / PPV
  0.19 (n=232, 10 deaths); coronary_care sens 0.41 / PPV 0.39 (n=184); surgical sens
  0.47 / PPV 0.69 (n=354); medical sens 0.30 / PPV 0.61 (n=430); overall sens 0.39 /
  PPV 0.50 (cycle 4; source `thresholds_cycle4.json`, carried into
  `thresholds_cycle5.json`).

## Open questions (all carried forward from cycle 4; none resolved or newly raised)

Ordered as presented to the review board in `report_cycle5.md` Section 8:

1. **Calibration above ~p=0.5 is unresolved** across all five cycles. Depends on:
   scoping a calibration step (e.g. isotonic regression fit on val) in a future
   cycle, with its own single test-set check.
2. **Cardiac_surgery_recovery's threshold (0.377) rests on few events** (10-18
   deaths) and its PPV estimate (0.19) is correspondingly uncertain. Depends on:
   either more data accumulating, or a partial-pooling / shrinkage estimator applied
   to the existing val data (scoped, not yet attempted, in `state_cycle4.md`).
3. **Generalization beyond this one hospital/time-period is untested** — no external
   data or admission-date metadata was available to check. Depends on: acquiring
   either, neither of which is available in the current `data/` directory.
4. **Only icu_type has been checked as a subgroup.** Sex, age band, and other
   subgroups remain unexamined across all five cycles. Depends on: a future cycle
   scoping that analysis on val, with its own single test check if a fix is needed.
5. **"Equal alert rate per unit" vs. "equal sensitivity per unit" as the fairness
   target was never adjudicated by the committee** — cycle 4 chose the former
   procedurally (matching the original 10%-alert-rate request, applied per unit).
   Depends on: committee input on which criterion is actually intended.
6. **Blend weight (0.5/0.5) was never tuned.** Depends on: a validation-only sweep,
   not yet attempted.
7. **No clinical cost-benefit input on false positives vs. false negatives** has ever
   been incorporated into any threshold in this project — all thresholds were set to
   hit an alert-rate capacity target, not a cost-weighted optimum. Depends on:
   clinical stakeholder input not available in any cycle to date.

## What would change these conclusions

- Same as stated in `state_cycle4.md`: more cardiac-surgery-recovery data would allow
  re-estimating that unit's threshold with less noise; a committee decision on the
  fairness criterion (item 5 above) would require recomputing per-unit thresholds
  against a different objective than "equal alert rate."
- If this report leads the review board to request a calibration fix or a subgroup
  fairness re-target before approval, that becomes a cycle-6 modeling task (not yet
  started), with its own train/val-only development and single final test check,
  following the same discipline as cycles 1-4.
