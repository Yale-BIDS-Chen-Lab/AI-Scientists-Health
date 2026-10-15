# Scientific state — ICU in-hospital mortality model, cycle 4

Rewritten in full at the end of cycle 4 (supersedes `state_cycle3.md` as
the current record; earlier state files kept for history). Cycle 4
diagnosed and fixed a subgroup calibration problem; it did not retrain
the ranking model and, deliberately, did not touch the test set.

## Standing decision: test set still has zero spent-without-justification looks remaining

`state_cycle3.md` closed the test set's "iterate then recheck" budget
after three legitimate, individually-justified looks (cycle 1 report,
cycle 2 report, cycle 3 threshold characterization). This cycle's
trigger — a unit director reporting a problem, asking for a fix — is
exactly the shape of request that policy anticipated and excluded. **This
cycle honored that: the diagnosis, the fix, and the validation of the fix
are all train+val-only (5-fold out-of-fold, and a doubly-cross-validated
check for the fix itself). Test labels were not read anywhere in
`train_cycle4.py`.**

This matters going forward: the fix in this cycle is validated only on
train+val. It has not been confirmed on data the model+recalibration
pipeline has never touched at all (test data was used to fit the original
model's hyperparameters' *comparison* in cycles 1-2, so even test isn't
fully "never touched" by the broader pipeline -- but it has at least never
been used to fit or check this specific recalibration). **Recommendation,
not yet acted on**: before this recalibration goes into clinical use, spend
one deliberate, final look at the test set specifically to confirm the
per-icu_type calibration ratios hold up there too -- this would be a
legitimate fourth use of the test partition because it is a genuine
pre-deployment confirmation of a specific, already-decided fix (like
cycle 3's threshold characterization), not another round of "try again."
Do not spend this look casually or bundle it with unrelated changes.
Whoever runs it should (a) compute the same predicted/observed ratio
table by icu_type on test, using the frozen recalibration parameters in
`recalibration_cycle4.json`, and (b) stop there -- if the numbers look
wrong, that is new information to bring back for a cycle 5 designed
around it, not a license to keep adjusting and re-checking against that
same test set.

## Decisions made this cycle

1. **Diagnosis, not a retrain**: confirmed via train+val out-of-fold
   predictions that cardiac_surgery_recovery (predicted/observed ratio
   1.26) and, to a lesser extent, coronary_care (ratio 1.06) are
   systematically overpredicted, while medical and surgical are already
   well calibrated (ratios 1.00 and 0.95). Depends on: the cycle-2 model
   and its isotonic calibration being fit on the pooled population
   (confirmed in `train_cycle2.py`) rather than per subgroup -- the
   direct mechanism for why a low-base-rate subgroup like cardiac surgery
   recovery (4.5% true mortality vs. 14.0% overall) would be
   overpredicted by a population-level calibration curve. Confidence:
   high -- the effect is large (26% relative), consistent in direction
   with the subgroup's known low base rate, and reproduces on
   out-of-fold (not in-sample) predictions.
2. **Fix = per-icu_type Platt recalibration in logit space**, not
   retraining the classifier. Depends on: the diagnosis being a
   calibration problem specifically -- subgroup AUROC was already good
   (0.86-0.92 across units, `report_cycle2.md`), so there was no ranking
   problem to fix by retraining. Confidence: high that this was the right
   class of fix; medium-high on the specific parameterization (2-parameter
   logistic recalibration per subgroup, chosen over full nonparametric
   isotonic per subgroup because the smallest subgroup, cardiac surgery
   recovery, has only 59 train+val events -- too few to trust a
   nonparametric fit without more smoothing than a 2-parameter model needs).
3. **Validated via double cross-validation, train+val only**: refit the
   per-subgroup correction on 4/5 of the out-of-fold data and applied it
   to the held-out 1/5, cycling folds, to check the fix generalizes
   in-family before touching anything resembling test. Confidence: high
   that this specific check is honest (it never reuses a fold's own data
   to correct itself); medium on how far it generalizes beyond train+val,
   since it is still the same 6,400-patient population the recalibration
   was designed against.
4. **New global threshold (0.388) recomputed on corrected probabilities**,
   same method as cycle 3 (90th percentile on train+val), same ~10%
   target. Depends on: decision 2. Confidence: same caveats as cycle 3's
   threshold (single-percentile design estimate, modest transfer gap seen
   historically between design and test-set alert rate).
5. **No per-unit thresholds adopted.** A single global threshold on
   corrected probabilities was chosen over unit-specific thresholds.
   Depends on: the judgment that the director's complaint is about
   probability *accuracy*, not about wanting a different sensitivity/PPV
   trade-off for that unit specifically -- see open question 1, this is
   the least certain judgment call in this cycle.

## Key evidence

- Predicted/observed ratio by icu_type, train+val out-of-fold: cardiac
  surgery recovery 1.26 -> 0.99 after the fix; coronary care 1.06 -> 1.00;
  medical 1.00 -> 1.00; surgical 0.95 -> 1.00 (doubly-cross-validated,
  `metrics_cycle4.json`).
- Overall train+val AUROC 0.8645 -> 0.8647 and Brier 0.0881 -> 0.0881
  after the fix -- no discrimination cost from recalibrating.
- Mechanical trade-off: at the fixed ~10% global alert budget, cardiac
  surgery recovery's design-set alert rate drops 2.4% -> 1.3% and
  sensitivity drops 0.32 -> 0.17 (PPV roughly unchanged, 0.59). This is
  the direct, expected consequence of correcting overprediction for a
  genuinely low-risk population, not a new problem introduced by the fix.

## Open questions

1. **Which direction was the director's complaint?** (`report_cycle4.md`
   Section 5). The data strongly support an overprediction/over-alerting
   story, but "seems inaccurate" as reported secondhand is compatible
   with the opposite concern (missing real risk), which this fix does not
   address and in fact moves further from. Depends on: direct
   clarification from the director, which this project does not have.
   This is the single biggest risk that the fix in this cycle is solving
   the wrong-signed problem. Flagged prominently in the report;
   recommendation is to confirm before deployment, not to guess further.
2. **Test-set confirmation not yet done** (see "Standing decision" above)
   -- recommended as a deliberate, final, narrowly-scoped check, not yet
   executed in this cycle.
3. **Small-subgroup precision**: cardiac surgery recovery's numbers rest
   on 59 (train+val) events; both the original diagnosis and the
   corrected performance figures for this unit should be read as
   directionally reliable, not precise to the percentage point. A larger
   or multi-site cohort (open question carried since cycle 1) would
   firm this up.
4. **Coronary care's smaller overprediction (ratio 1.06) was corrected as
   a matter of consistency**, but was never itself the subject of a
   complaint. Worth confirming this correction is neutral-to-positive for
   that unit too if/when it's ever reviewed (design-set numbers in
   Section 4 of the report suggest yes: alert rate and sensitivity both
   essentially unchanged there).
5. **All open questions from `state_cycle2.md`/`state_cycle3.md` not
   touched by this cycle remain open**: the general "is this sufficient
   for clinical use" question, medical/coronary-care's lower absolute
   discrimination (unaffected by a calibration-only fix), and the lack of
   external validation.

## What would change these conclusions

- Direct word from the director on complaint direction (open question 1)
  could invalidate the chosen fix's framing entirely, even though the
  calibration bug itself is real and worth fixing either way -- in that
  case the response would be a per-unit threshold adjustment layered on
  top of (not instead of) this cycle's calibration correction, since the
  correction and the alert-budget policy are separate, independently
  adjustable levers.
- A deliberate, final test-set look (recommended, not yet done) that
  contradicts the train+val calibration story would mean the fix
  overfit train+val's specific subgroup composition and needs
  reworking -- in which case that becomes the evidence base for a cycle 5,
  not a reason to keep re-checking the same test set repeatedly.
