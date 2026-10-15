# Methods and validation report — ICU in-hospital mortality alert
## Prepared for the deployment review board

## 1. Summary

This model estimates an adult ICU patient's probability of in-hospital death from the
first 48 hours of admission data, and drives a bedside alert. It was developed and
validated over four prior cycles (feature/model development, threshold selection,
and a subgroup-fairness fix), summarized here for review. **No new modeling was done
in this cycle** — this report consolidates cycles 1-4 into a single reference
document and changes nothing about the deployed model or its alert policy.

**Bottom line**: test-set AUROC 0.857, AUPRC 0.512 (base rate 14.0%). The deployed
alert policy uses a different probability threshold per ICU type (adopted in cycle 4
to fix a documented under-alerting problem in the cardiac surgery recovery unit) and
fires for roughly 10% of admissions per unit. At that policy, overall sensitivity is
0.39, specificity 0.94, PPV 0.50 — but these numbers vary materially by unit (Section
7), and the review board should read Section 8 (Limitations) before approving
deployment scope, since several open items (calibration, small-subgroup stability,
generalization beyond this hospital) are unresolved, not merely theoretical.

## 2. Data

8,000 adult ICU admissions from a single US academic hospital, one admission per
patient, first 48 hours of ICU stay only (admissions under 48h and patients under 16
excluded). `patients.csv`: demographics + ICU type (medical, surgical, coronary care,
cardiac surgery recovery). `measurements.csv`: 37 clinical variables, long format,
irregular sampling (3.48M observations total). Outcome: in-hospital death (14.0%
positive). Full schema: `docs/data_dictionary.md`.

A small number of physiologically impossible values (sensor/entry artifacts — e.g.
`Weight = -1`, `Temp = -17.8°C`, `pH = 0` or `735`) were identified in 14 of the 37
variables and treated as missing before feature construction (`icu_features.py`,
`PLAUSIBLE_RANGE`); the other 23 variables showed no comparable artifacts.

## 3. Final model

**Type**: fixed-weight (0.5/0.5) probability average of two sub-models fit on the same
338-feature representation:
- A `HistGradientBoostingClassifier` (gradient-boosted trees; handles missing values
  natively), `learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0,
  min_samples_leaf=20`, `class_weight="balanced"`.
- An L2-regularized logistic regression (median-imputed, standardized),
  `C=0.01`, `class_weight="balanced"`.

Both sub-models and the blend weights are fit and fixed; `predict_cycle5.py` performs
inference only (loads `model_cycle2.pkl`; nothing is retrained).

**Features** (338 total, `icu_features.build_feature_matrix_v2`): per clinical
variable — count, mean, min, max, last value, cumulative sum, and linear time-trend
(slope) across the 48h window, plus a binary "was this variable ever measured"
indicator (informative missingness). Ten clinically central severity markers (GCS,
HR, MAP, SysABP, Temp, RespRate, Lactate, Creatinine, BUN, pH) additionally get
explicit early-half (0-24h) vs. late-half (24-48h) mean and their difference, to
capture deterioration/improvement trajectory. Demographics contribute age, sex,
height, admission weight, BMI, and one-hot ICU type.

**Why this model class**: compared against HistGradientBoosting alone, Random Forest,
and logistic regression alone on validation data (all four candidates trained
identically); the blend won by a clear margin (val AUROC 0.870 vs. 0.861-0.867 for
any single model) and this ranking held up on the untouched test set.

## 4. Development and validation methodology

**Patient-level split, fixed once, never redrawn**: 8,000 admissions were split
70/15/15 (train 5,600 / val 1,200 / test 1,200) by patient, stratified on outcome,
seed 42, in cycle 1. Every subsequent cycle (2, 3, 4, and this one) reused that exact
assignment — `split_cycle5.csv` is byte-identical to `split_cycle1.csv`. This is what
makes cross-cycle comparisons (e.g. "cycle 2 improved AUROC over cycle 1") valid
statements about the model rather than artifacts of a different random sample.

**Test-set discipline**: all feature engineering, model-class selection,
hyperparameter tuning, calibration checks, and threshold selection were performed
using train and validation data only. The test set was scored in four discrete,
single-pass evaluations across the project (cycle 2: final model performance; cycle 3:
global-threshold operating characteristics; cycle 4: per-unit-threshold check, both
old and new policy evaluated together in one pass) — never used iteratively to search
over candidates. No candidate was accepted or rejected based on a test-set result at
any point.

**Cycle-by-cycle summary**:
| cycle | what changed | test AUROC | test AUPRC |
|---|---|---|---|
| 1 | first model (HistGradientBoosting, 232 features) | 0.853 | 0.502 |
| 2 | expanded features (338) + model-class search → blend | **0.857** | **0.512** |
| 3 | operating threshold set for ~10% alert rate (no model change) | 0.857 (unchanged) | 0.512 (unchanged) |
| 4 | per-ICU-type thresholds (fix for cardiac surgery recovery; no model change) | 0.857 (unchanged) | 0.512 (unchanged) |
| 5 | this report; no model or threshold change | 0.857 (unchanged) | 0.512 (unchanged) |

## 5. Final performance (test set, n=1,200, 168 deaths)

| metric | value |
|---|---|
| AUROC | **0.857** |
| AUPRC | **0.512** (base rate 0.140) |
| Brier score | 0.133 |

Calibration (test set, predicted-risk deciles): observed death rate rises
monotonically with predicted risk, but the model is **systematically overconfident
above roughly the 50th percentile of predicted risk** (e.g. in the top decile, mean
predicted probability 0.86 vs. observed rate 0.58) — see `report_cycle2.md` for the
full decile table. This was identified in cycle 1, confirmed unchanged in cycle 2, and
was never corrected (see Limitations). **The model's probability output should be
used as a risk ranking, not read as a literal probability of death**, particularly
above p≈0.5.

Drivers of the prediction (permutation importance, test set): **Glasgow Coma Score
dominates** — the most recent value (`GCS_last`) alone accounts for roughly 3x the
importance of the next-largest feature, with a late-window GCS trend feature
(`GCS_late_mean`) also in the top 5. **Age is the clear second driver.** Beyond these,
importance is spread thinly across renal output (`Urine_sum`), respiratory monitoring,
and ICU-type context — no other single feature is individually decisive. Both drivers
are consistent with established ICU severity scoring (APACHE, SAPS) and were
cross-checked against a transparent logistic-regression reference model, which agrees
in direction. Full table: `cycle2_importance.csv`.

## 6. Operating threshold

The deployed alert policy (adopted in cycle 4, unchanged since) uses **a different
threshold per ICU type**, not one global cutoff, because a single global threshold was
found to fire for only ~1% of cardiac surgery recovery admissions against a 10%
population-wide target (Section 7 explains why). Thresholds were chosen on validation
data to each yield approximately a 10% within-unit alert rate, using out-of-sample
scores from a train-only refit of the two sub-models (to avoid bias from evaluating
the deployed, train+val-fit model on data it had partly seen).

| icu_type | threshold |
|---|---|
| cardiac_surgery_recovery | **0.377** |
| coronary_care | 0.705 |
| surgical | 0.718 |
| medical | 0.810 |

A single top-level fallback threshold, **0.7367**, is retained in
`thresholds_cycle5.json` for the (currently not expected to occur, since icu_type is
always populated) case of a missing/unrecognized ICU type. The probability score and
the threshold are deliberately kept as separate artifacts (`predict_cycle5.py` outputs
only a continuous probability; the alert decision — probability vs. the appropriate
threshold — is left to the calling system) so the risk score and the alert policy can
be reviewed or revised independently.

## 7. Performance by ICU type (test set, deployed per-unit policy)

| icu_type | n | alert rate | sensitivity | specificity | PPV | NPV |
|---|---|---|---|---|---|---|
| cardiac_surgery_recovery | 232 | 13.8% | 0.60 | 0.88 | **0.19** | 0.98 |
| coronary_care | 184 | 9.8% | 0.41 | 0.93 | 0.39 | 0.94 |
| surgical | 354 | 11.0% | 0.47 | 0.96 | 0.69 | 0.90 |
| medical | 430 | 9.5% | 0.30 | 0.95 | 0.61 | 0.85 |
| **Overall** | **1,200** | **10.8%** | **0.39** | **0.94** | **0.50** | **0.90** |

**Why the per-unit policy exists**: baseline mortality varies roughly 3-4x across
units (cardiac surgery recovery 4.7% vs. 13.8-19.2% elsewhere, computed on
train+validation data). The model's discrimination is not the issue — AUROC by unit
(out-of-sample validation) ranges 0.85-0.89, with cardiac surgery recovery scoring
*highest*. But under a single global threshold, cardiac surgery recovery's absolute
predicted probabilities sit far below the cutoff (mean predicted risk 0.13 vs.
0.32-0.41 elsewhere), so the alert fired for only ~1% of that unit's admissions
against the intended 10% — effectively non-functional for that unit. Per-unit
thresholds correct this directly.

**This fix has a documented cost, stated here rather than smoothed over**: on the
382-patient held-out test check, the per-unit policy raises cardiac surgery recovery's
sensitivity (0.50→0.60 relative to the old global-threshold policy) and moves its
alert rate toward the 10% target, but its precision is low (PPV 0.19 — roughly 4 in 5
alerts in that unit are false positives) and the estimate itself rests on very few
events (10 deaths in that unit's test subset, 18 in validation) so it carries wide
uncertainty. The review board should treat the cardiac-surgery-recovery numbers in
this table as **directionally correct but not precisely estimated**.

## 8. Limitations

Stated plainly, in order of what the review board should weigh most heavily:

1. **Calibration is not solved.** The model overpredicts risk above roughly the 50th
   percentile (Section 5). Deploy it as a ranking/triage signal, not as a literal
   probability to quote to patients or families. No calibration correction has been
   applied in any cycle to date.
2. **The cardiac-surgery-recovery threshold is estimated from a small number of
   events** (10-18 deaths across validation/test) and its reported precision (PPV
   0.19) is correspondingly uncertain. A future cycle should revisit this with a more
   variance-robust estimate (e.g. partial pooling toward the global threshold) once
   more data accumulate, rather than treating 0.377 as a precise, final number.
3. **Generalization beyond this hospital and this time period is untested.** All
   8,000 admissions are from one US academic hospital; no admission-date metadata was
   available to check for temporal drift (e.g. practice changes over the data
   collection period), and no external-hospital data exists to check transportability.
   Performance at a different institution, or at this institution after a protocol
   change, is unverified.
4. **Only ICU type was examined as a subgroup.** Performance by sex, age band, or
   other clinically relevant subgroup has not been evaluated in any cycle. Given that
   ICU type alone revealed a materially unequal alert policy, the review board should
   not assume other subgroups are unaffected without checking.
5. **The alert-rate target (10% per unit) reflects "equal alert rate across units,"
   not "equal sensitivity across units."** These are different, both defensible,
   fairness criteria; equalizing sensitivity instead would very likely require higher
   alert rates in lower-prevalence units. This is a clinical policy choice that was
   made procedurally (matching the coordinator's original global 10% request, applied
   per unit) rather than adjudicated by the committee, and should be confirmed as the
   intended criterion before deployment.
6. **The 0.5/0.5 blend weight between the two sub-models was not tuned** — it was the
   simple average tested, and it won on validation, but a swept weight was never
   checked for a possibly better operating point.
7. **The decision threshold(s) carry no explicit clinical cost-benefit input** (cost of
   a missed death vs. cost of a false alert) — they were set purely to hit a
   requested alert-rate target, which is a capacity constraint, not a cost-weighted
   optimum.

## 9. Files in this delivery

- `predict_cycle5.py` — inference only; loads `model_cycle2.pkl` (unchanged since
  cycle 2), writes `patient_id,probability`
- `split_cycle5.csv` — patient-level train/val/test assignment, identical to
  `split_cycle1.csv` through `split_cycle4.csv` (never redrawn)
- `thresholds_cycle5.json` — `{"threshold": 0.7367, "per_unit": {icu_type:
  threshold, ...}}`, the currently deployed alert policy, plus its full test-set
  performance record for audit purposes
- `state_cycle5.md` — current scientific state: decisions, confidence levels,
  evidence, and open questions, consolidated across all five cycles
- Supporting artifacts referenced above and retained from prior cycles:
  `icu_features.py` (feature engineering), `model_cycle2.pkl` (fitted model),
  `report_cycle1.md` through `report_cycle4.md` and `cycle{1,2,4}_metrics.txt` /
  `cycle2_importance.csv` (full historical detail behind every number cited here)

## Headline numbers for the review board

Final model: blend of gradient-boosted trees + logistic regression, 338 engineered
features. **Test AUROC 0.857, AUPRC 0.512** (base rate 0.140). Deployed alert policy:
per-ICU-type thresholds (0.377-0.810) targeting ~10% alert rate per unit; overall test
sensitivity 0.39, specificity 0.94, PPV 0.50. Top limitation: probabilities are not
well calibrated above p≈0.5 and should be read as a ranking, not a literal risk
percentage; second: the cardiac-surgery-recovery-specific numbers are estimated from
few events and should be treated as directional pending more data.
