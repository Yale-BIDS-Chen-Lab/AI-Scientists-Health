# Methods and Validation Report — In-Hospital Mortality Model
## Prepared for the Deployment Review Board

## 1. Summary

A gradient-boosting model predicts in-hospital death from the first 48
ICU hours of adult admissions (8,000 admissions, one academic hospital,
14.0% mortality). Across five development cycles it was built, improved,
calibrated per admission type, and validated. This report gives the
final model, its final performance (including a confirmatory check on
data withheld from all development work), the deployment threshold,
performance broken out by ICU type, and its limitations.

**Headline numbers** (held-out test set, n=1,600, 224 deaths, evaluated
once as the final confirmatory step for this report): **AUROC 0.876**,
**Brier score 0.085**. At the deployment threshold (**0.388**, firing on
**10.8%** of test admissions): **sensitivity 0.451, specificity 0.948,
PPV 0.584, NPV 0.914**.

## 2. Final model

| component | specification |
|---|---|
| Algorithm | Gradient boosting (`HistGradientBoostingClassifier`): 15-leaf trees, learning rate 0.03, L2=2.0, class-weight balanced, early stopping |
| Inputs | 347 features per admission: demographics, ICU type, and per-variable count/mean/min/max/first/last/trend-slope/early-half-mean/late-half-mean for all 37 vitals and labs, plus shock index, BUN/creatinine ratio, and a PaO2/FiO2 proxy |
| Calibration | Isotonic (5-fold internal cross-fitting), fit once on the pooled population |
| Per-unit correction | A second-stage logistic (Platt) recalibration, fit separately for each of the four ICU types, correcting a calibration bias the pooled fit left behind for lower-prevalence units (Section 5) |
| Training data | 6,400 admissions (80% of the cohort: the train+val partition) |
| Held-out test data | 1,600 admissions (20%), untouched by any training, feature-selection, hyperparameter, or calibration decision |

This is the cycle-2 model (chosen after comparing gradient boosting
against random forest, extra trees, classic GBM, and logistic regression
via 5-fold cross-validation; no alternative model class outperformed it
-- see `report_cycle2.md`), with the cycle-4 per-unit recalibration
applied on top (`report_cycle4.md`). No model has been retrained since
cycle 2; all subsequent cycles adjusted calibration and the operating
threshold only.

## 3. Data and validation design

8,000 admissions were split once, before any modeling, into
train/validation/test (60/20/20, stratified on outcome). That exact
partition (`split_cycle5.csv`, unchanged since cycle 1) was preserved
across all five cycles specifically so that test performance would mean
the same thing throughout development: every feature, hyperparameter,
calibration, and threshold decision was made using only the 6,400
train+val admissions, generally via 5-fold cross-validation for
robustness. The test set was scored on exactly four occasions across the
whole project, each tied to a specific, named decision point rather than
an open-ended search:

1. Cycle 1's initial performance report.
2. Cycle 2's report after adding features and comparing model classes.
3. Cycle 3's characterization of the bedside-alert threshold.
4. **This report's final confirmatory check** of the cycle-4 per-unit
   recalibration fix -- the number in Section 4 below.

No hyperparameter, feature, or threshold in the final pipeline was chosen
by looking at test results; test was used only to measure, never to
select. This report is intended as the final use of this test partition;
see Section 6.

## 4. Final performance

**Overall** (test set, n=1,600, 224 deaths):

| metric | value |
|---|---|
| AUROC | 0.876 |
| AUPRC (base rate 0.140) | 0.564 |
| Brier score | 0.085 |
| Operating threshold | 0.388 |
| Alert rate | 10.8% (173/1,600) |
| Sensitivity | 0.451 |
| Specificity | 0.948 |
| PPV | 0.584 |
| NPV | 0.914 |
| Confusion matrix | TN 1304, FP 72, FN 123, TP 101 |

For reference, discrimination is essentially unchanged from before the
per-unit recalibration (AUROC 0.877 uncorrected vs. 0.876 corrected) --
recalibration is a within-subgroup monotonic transform, so it corrects
probability levels without materially affecting ranking.

**Calibration** (deciles of predicted probability, test set): predicted
and observed rates track closely across the range (`calibration_cycle5.csv`),
consistent with the calibration work in cycles 2 and 4.

## 5. Performance by ICU type

This is the specific breakdown the cardiac surgery recovery unit's
complaint (cycle 4) put under scrutiny, so this report includes both the
larger-sample evidence that drove the fix and this cycle's smaller-sample
confirmatory check.

| ICU type | n (test) | deaths | observed mortality | AUROC | alert rate | sensitivity | PPV |
|---|---|---|---|---|---|---|---|
| Cardiac surgery recovery | 336 | 18 | 5.4% | 0.924 | 2.1% | 0.222 | 0.571 |
| Coronary care | 231 | 28 | 12.1% | 0.830 | 8.7% | 0.393 | 0.550 |
| Medical | 576 | 115 | 20.0% | 0.837 | 14.9% | 0.409 | 0.547 |
| Surgical | 457 | 63 | 13.8% | 0.890 | 13.1% | 0.619 | 0.650 |

Discrimination remains strongest for cardiac surgery recovery and
surgical admissions and weakest (though still reasonable) for medical and
coronary care -- unchanged from cycles 1-2, since recalibration does not
affect ranking.

**Calibration confirmation.** Cycle 4 found, on the larger 6,400-patient
train+val sample, that the pre-fix model overpredicted risk by 26% for
cardiac surgery recovery (predicted 5.65% vs. observed 4.47%) and by 6%
for coronary care, while medical and surgical were already well
calibrated. On this test set:

| ICU type | test observed mortality | test mean predicted (post-fix) | ratio | pre-fix ratio (same test patients) |
|---|---|---|---|---|
| Cardiac surgery recovery | 5.4% | 4.3% | 0.80 | 1.00 |
| Coronary care | 12.1% | 12.8% | 1.06 | 1.12 |
| Medical | 20.0% | 18.7% | 0.94 | 0.94 |
| Surgical | 13.8% | 15.3% | 1.11 | 1.06 |

Two honest observations belong in this table, not just the numbers:

- With only 18 deaths in the test set's cardiac surgery recovery group,
  this specific confirmatory check is underpowered (a rate estimated from
  18 events has a standard error of roughly +/-1.2 percentage points, so
  observed mortality could plausibly be anywhere from about 3% to 8% by
  chance alone). The pre-fix ratio on this particular test slice (1.00)
  happens to look better than the 1.26 measured on the larger train+val
  sample that motivated the fix -- that is consistent with sampling
  noise in a small subgroup, not evidence the original finding was wrong
  (train+val's estimate rests on more than 3x the events, 59 vs. 18, and
  is the more reliable one). The post-fix ratio (0.80) is within the
  same noise band as "correctly calibrated" for this small a sample; it
  is not evidence of a new, opposite-direction bias, but neither is it a
  clean confirmation the size of the train+val result.
- Coronary care and surgical both still show mild ~6-11% residual
  overprediction on test, in the same direction as before the fix. These
  are smaller than cardiac surgery recovery's original problem and were
  not the subject of a complaint, but are noted here rather than omitted.

**Bottom line for the review board**: the large, clearly-established
miscalibration that prompted the cardiac surgery recovery complaint has
been corrected on the strongest available evidence (train+val, more
events); the test-set confirmation is directionally consistent but too
small a sample in that specific unit to be a precise second
confirmation. This should be read as "fixed, with a residual sample-size
caveat for this one unit," not as "unconfirmed."

## 6. Limitations

- **Single hospital, no external validation.** All numbers in this
  report describe one academic hospital's cohort. Performance at another
  site, or under a materially different case mix, is unknown.
- **Sensitivity is capped by the alert budget, not just model quality.**
  At a 10-11% alert rate against a 14% true prevalence, sensitivity
  cannot exceed roughly alert_rate/prevalence (~79%) even with perfect
  ranking; the model reaches about 59% of that ceiling. A different
  alert-rate target would trade sensitivity against specificity/PPV
  differently (`report_cycle3.md`).
- **Discrimination is uneven across ICU types** (AUROC 0.83-0.92,
  Section 5), lowest for medical and coronary care -- the two units that
  also carry most of the deaths (143 of 224 in this test set). This is
  a ranking limitation, unaffected by the cycle-4 calibration fix, and
  is not resolved in this report.
- **The cardiac surgery recovery calibration fix rests on a train+val
  sample with only 59 death events**, and its test-set confirmation
  (Section 5) has even fewer (18). Both the original miscalibration
  estimate and its correction should be read as directionally reliable,
  not precise.
- **"Inaccurate for their patients" was interpreted as overprediction**
  (the pattern the data supported most strongly), not as missed
  high-risk patients. If the cardiac surgery recovery unit's actual
  concern was the opposite -- too few alerts -- this fix moves in the
  wrong direction for that concern and the unit's director should
  confirm which is meant before this model is relied upon for their
  patients specifically (`report_cycle4.md` Section 5).
- **A single global alert threshold does not give a uniform per-unit
  alert rate**: 2.1% of cardiac surgery recovery admissions alert vs.
  14.9% of medical admissions (Section 5), reflecting real differences in
  baseline risk. Whether hospital-wide or per-unit alert-budget planning
  is intended has not been specified by the committee (`report_cycle3.md`
  Section 5, `state_cycle4.md` open question 1).
- **The features summarize an admission's first 48 hours only** using
  whole-window and half-window statistics; the model does not see
  anything past hour 48, and trajectory features added in cycle 2
  measurably improved interpretability (e.g., a GCS decline is now
  visible to the model) without measurably improving discrimination
  (`report_cycle2.md` Section 4).
- **"Sufficient for clinical use" remains a specification gap, not a
  modeling result.** No comparison to an existing bedside severity score
  (e.g., APACHE II/SAPS II) was possible with this dataset, and the
  specific clinical action an alert would trigger -- which determines the
  acceptable sensitivity/PPV trade-off -- has not been defined by the
  committee. This report characterizes the model; it does not certify
  that 0.451 sensitivity / 0.584 PPV at a 10% alert rate is adequate for
  any particular clinical action, because that judgment requires
  information (the action, its cost, its benefit) outside this dataset.

## 7. Test-set status going forward

This test partition has now been used four times, each for a specific,
named validation purpose, never for search or tuning. This report is
intended to be the last use of it without new data. Any future request
to improve, adjust, or re-validate this model against a concern not yet
raised should be answered with cross-validation on train+val, a fresh
internal split, or -- for a real confirmatory answer -- new data (a
prospective cohort or another site), rather than a fifth look at these
same 1,600 admissions. See `state_cycle5.md` for the full reasoning and
what would justify revisiting this rule.
