# Scientific state — ICU in-hospital mortality model, cycle 5 (deployment review)

Rewritten in full at the end of cycle 5 (supersedes `state_cycle4.md` as
the current record; earlier state files kept for history). Cycle 5 made
no modeling changes: it ran the deliberate, final confirmatory test-set
look that `state_cycle4.md` recommended, and packaged the whole project
into the deployment review board report (`report_cycle5.md`).

## Standing decision: the test set's evaluation budget is now closed

The 1,600-patient test partition (`split_cycle1.csv` through
`split_cycle5.csv`, all identical) has been scored four times:

1. Cycle 1 -- initial performance report.
2. Cycle 2 -- performance after feature/model-class search.
3. Cycle 3 -- bedside-alert threshold characterization.
4. Cycle 5 -- confirmatory check of the cycle-4 per-unit recalibration
   (this cycle).

Each use was tied to a specific, already-decided artifact being
*measured*, never to searching for a better one -- no feature,
hyperparameter, calibration, or threshold in the final pipeline was ever
chosen by looking at test results. That discipline is what makes four
looks defensible where an open-ended "keep checking until it looks good"
loop would not have been.

**This is now the closing use of this test set without new data.**
Concretely:

- Any future request shaped like "still not good enough" / "check this
  other thing" / "what about subgroup X" should be answered using
  cross-validation on train+val, or a fresh internal re-split of
  train+val, not by scoring these same 1,600 admissions again.
- The only thing that should trigger a fifth look at this exact
  partition is a change so structural that re-measuring against the
  known reference point is itself the point (e.g., confirming a
  regression before a rollback) -- and that should be named explicitly as
  a deliberate exception when it happens, the way cycles 3 and 5 were.
- The clean way to keep validating this model further is new data: a
  prospective cohort, another site, or a later time period at the same
  hospital. That would also be the only way to answer the open question
  (carried since cycle 1) of how this model performs outside the
  single-hospital population it was built on.

## Final pipeline (frozen as of this cycle)

1. **Model**: `model_cycle2.joblib` -- HistGradientBoostingClassifier
   (15-leaf trees, lr 0.03, L2=2.0), 347 features, fit on 6,400 train+val
   admissions. Unchanged since cycle 2. Selected over logistic
   regression, random forest, extra trees, and classic gradient boosting
   via 5-fold CV on train+val (`report_cycle2.md`).
2. **Base calibration**: isotonic, 5-fold internal cross-fitting, fit on
   train+val (part of `model_cycle2.joblib`).
3. **Per-unit correction**: `recalibration_cycle4.json` -- a 2-parameter
   logistic (Platt) recalibration per ICU type, fit on 5-fold
   out-of-fold train+val predictions, correcting the base calibration's
   26%-relative overprediction for cardiac surgery recovery and 6% for
   coronary care. Unchanged since cycle 4.
4. **Threshold**: 0.388 (`thresholds_cycle5.json`, unchanged from cycle
   4), chosen as the 90th percentile of train+val predictions to hit the
   committee's ~10% alert-rate target. No per-unit thresholds are used.

Depends on, in order: the original 60/20/20 stratified split (cycle 1,
unchanged since); the feature set and model class (cycle 2); the per-unit
calibration diagnosis and fix (cycle 4). Every layer's dependencies are
detailed in that cycle's own state file.

## Key evidence (this cycle's contribution: the final confirmatory numbers)

- Test AUROC 0.876, AUPRC 0.564, Brier 0.085 -- consistent with cycle 2's
  pre-recalibration test AUROC of 0.877 (recalibration is a
  within-subgroup monotonic transform, so it should not and does not
  materially move overall discrimination).
- At threshold 0.388: alert rate 10.8%, sensitivity 0.451, specificity
  0.948, PPV 0.584, NPV 0.914 (`metrics_cycle5.json`).
- Per-unit calibration confirmation is **directionally consistent but
  underpowered for cardiac surgery recovery specifically**: the train+val
  evidence for the original 1.26x overprediction rests on 59 events and
  is the reliable reference; the test-set confirmation has only 18 events
  in that unit and swung to a 0.80 ratio (mild apparent underprediction),
  which is within plausible sampling noise for that sample size rather
  than evidence of a new problem. Coronary care and surgical show mild
  (6-11%) residual overprediction on test in the same direction as
  before the fix -- smaller than the original cardiac-surgery-recovery
  problem, not the subject of a complaint, but not zero either.

## Open questions (carried forward, none resolved by this cycle)

1. **Was the cardiac surgery recovery complaint about over-alerting or
   under-detection?** Still unconfirmed with the director (carried from
   `state_cycle4.md` open question 1). This report's Section 5/6 flags it
   again for the review board specifically, since this is the kind of
   thing a deployment sign-off should not proceed past without an answer.
2. **"Sufficient for clinical use" is still not a modeling question this
   project can answer** (carried since `state_cycle2.md`): no benchmark
   comparator (e.g. APACHE II/SAPS II) exists in this dataset, and the
   specific clinical action an alert would trigger has never been
   specified. `report_cycle5.md` Section 6 states this plainly for the
   review board rather than letting a bare AUROC/PPV stand in for a
   sufficiency judgment.
3. **Medical/coronary-care discrimination gap** (AUROC 0.83-0.84 vs.
   0.89-0.92 for surgical/cardiac-surgery-recovery) is unchanged since
   cycle 1 and untouched by the calibration-only fix in cycle 4. These
   two units carry 143 of 224 test deaths. Not addressed in this cycle;
   would need new features, more data, or both.
4. **Residual mild overprediction in coronary care and surgical**
   (Section 5 of the report): smaller than the fixed cardiac-surgery-
   recovery problem and not yet the subject of any complaint. Worth
   monitoring, not worth a dedicated fix cycle absent a reported problem
   or new data to check it against.
5. **No external validation** -- unchanged since cycle 1, still the
   single biggest limitation on how far any of these numbers generalize.
6. **Uniform vs. per-unit alert-rate policy** (carried from
   `state_cycle4.md` open question, originally `report_cycle3.md`
   Section 5): a global 10% threshold produces very different per-unit
   alert rates (2.1% cardiac surgery recovery to 14.9% medical). Not
   resolved; needs committee input on whether that is intended.

## What would change these conclusions

- Director confirmation on open question 1 could change whether the
  cycle-4 fix is the right response at all, or needs a per-unit threshold
  layered on top (mechanism already described in `report_cycle4.md`
  Section 5; not yet implemented because there's no evidence pointing
  that direction).
- A committee decision on the intended clinical action (open question 2)
  would let this project state a real sufficiency judgment instead of
  reporting bare metrics, and might change what threshold or even what
  metric (sensitivity vs. PPV vs. something else) should be optimized.
- New data -- prospective, external, or simply more of it -- is the only
  way to responsibly re-open performance questions on this test
  partition, to get a properly powered check of the cardiac-surgery-
  recovery calibration fix specifically, and to make any progress on the
  medical/coronary-care discrimination gap.
