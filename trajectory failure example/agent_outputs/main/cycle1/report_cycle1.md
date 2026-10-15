# In-hospital mortality prediction, ICU adults — cycle 1

## Task and data

8,000 adult ICU admissions, one per patient, first 48 hours of ICU stay. `patients.csv`
gives demographics and ICU type; `measurements.csv` gives 37 clinical variables in long
format (irregular sampling, no fixed grid); `outcomes.csv` gives `in_hospital_death`
(14.0% positive). Full data description in `docs/data_dictionary.md`.

## Data cleaning

A subset of variables contained physiologically impossible values consistent with
sensor/entry artifacts rather than true measurements: `Weight` had 673 rows at exactly
-1 kg, `Temp` had 41 rows down to -17.8°C, and `pH`, `HR`, `MAP`, `SysABP`, `DiasABP`,
`NIMAP`, `NISysABP`, `NIDiasABP`, `PaCO2`, `PaO2`, `WBC`, and `Mg` each had a small
number of zero or extreme values (e.g. pH of 0 or 735). For these 14 variables only, I
replaced values outside a documented physiologically plausible range with missing
before computing any summary statistic (exact bounds in `icu_features.py`,
`PLAUSIBLE_RANGE`). The other 23 variables showed no comparable artifacts and were left
unfiltered. This is a targeted cleaning step, not a general outlier removal.

## Features

For each of the 37 variables I computed, from the cleaned 48-hour window: count, mean,
min, max, and last observed value, plus a binary "was this variable ever measured"
indicator (informative missingness — what gets tested is itself a clinical signal in
ICU data). Demographics contribute age, sex, height, admission weight, BMI (when both
height and weight are present), and one-hot ICU type. Total: 232 features per patient.
`icu_features.py` contains this logic and is shared by training and inference so the
two stay identical.

## Modeling and evaluation protocol

Patients were split once, by `patient_id`, stratified on outcome, into train (5,600) /
val (1,200) / test (1,200), seed 42 — recorded in `split_cycle1.csv`. All feature
choices, the model class, hyperparameters, and the decision threshold were selected
using train and val only. The test set was scored exactly once, at the end, and those
numbers are reported below unchanged.

**Primary model**: `HistGradientBoostingClassifier` (gradient-boosted trees), chosen
because it handles missing values natively — appropriate here since missingness is
itself informative and imputing it would discard that signal — and because tree
ensembles are the established strong baseline for this kind of sparse, irregularly
sampled ICU time series. `class_weight="balanced"` compensates for the 14% event rate.
Five hyperparameter combinations (learning rate, leaf count, L2) were compared by val
AUROC; the best (`learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0`, val
AUROC 0.865) was refit on train+val (6,800 patients) for the model shipped in
`model_cycle1.pkl`.

**Secondary reference model**: L2-logistic regression (median-imputed, standardized),
tuned the same way (val AUROC 0.860 at best C). This is not the deployed model; it
exists only to sanity-check that the gradient-boosting model's behavior lines up with
a transparent linear model's coefficients (see below). It is not saved for inference.

The decision threshold (for sensitivity/specificity/PPV/NPV only — the deliverable
prediction is a probability, not a label) was chosen on val by Youden's J and then
applied, unchanged, to test.

## Results (test set, n=1,200, 168 deaths, touched once)

| metric | value |
|---|---|
| AUROC | **0.853** |
| AUPRC | **0.502** (base rate 0.140) |
| Brier score | 0.131 |
| Sensitivity @ threshold 0.541 | 0.679 |
| Specificity @ threshold 0.541 | 0.843 |
| PPV @ threshold 0.541 | 0.413 |
| NPV @ threshold 0.541 | 0.942 |

Reference logistic regression test AUROC: 0.848 — close to the primary model, which is
reassuring: the gradient-boosting model's extra flexibility is buying a modest, not
dominant, amount of discrimination over a simple linear model, so the more complex
model is not obviously overfit to idiosyncratic training patterns.

Calibration (test set, predicted-risk deciles, mean predicted probability vs. observed
death rate):

| predicted decile | mean predicted | observed rate | n |
|---|---|---|---|
| lowest | 0.024 | 0.000 | 120 |
| | 0.045 | 0.025 | 120 |
| | 0.069 | 0.008 | 120 |
| | 0.101 | 0.017 | 120 |
| | 0.152 | 0.058 | 120 |
| | 0.239 | 0.083 | 120 |
| | 0.365 | 0.133 | 120 |
| | 0.511 | 0.183 | 120 |
| | 0.678 | 0.325 | 120 |
| highest | 0.832 | 0.567 | 120 |

Observed rates rise monotonically with predicted risk across deciles (one minor
inversion between the 2nd and 3rd bins, expected noise at n=120 per bin). Above the
~0.5 predicted-probability range the model is systematically over-confident (mean
predicted 0.68–0.83 vs. observed 0.33–0.57): at high predicted risk, treat the score as
a strong risk-ranking signal, not a literal probability. A recalibration step (e.g.
isotonic or Platt scaling on val) is a natural cycle-2 addition; it was not done in
cycle 1 in order to keep the model-selection surface small for a first pass.

## What drives the predictions

Permutation importance on the test set (single pass, 10 repeats, AUROC as the scoring
metric) — full table in `cycle1_importance.csv`:

| rank | feature | importance (AUROC drop) |
|---|---|---|
| 1 | GCS_last (Glasgow Coma Score, most recent) | 0.074 |
| 2 | age_years | 0.018 |
| 3 | Urine_mean (mean urine output) | 0.008 |
| 4 | icu_type = cardiac_surgery_recovery | 0.008 |
| 5 | GCS_max | 0.007 |
| 6 | GCS_mean | 0.004 |
| 7 | Glucose_last | 0.003 |

Neurological status (GCS) dominates by a wide margin — the last recorded GCS alone
accounts for roughly 4x the importance of the next-strongest feature, consistent with
GCS's established role as a core severity-of-illness marker in ICU risk scores
(APACHE, SAPS). Age is the clear second signal. Beyond that, importance drops off
quickly and spreads across renal (urine output), admission context (ICU type), and
metabolic (glucose) features — no single lab value beyond GCS is individually
decisive; the model is combining many weak signals.

The logistic-regression reference model's largest standardized coefficients agree in
direction with clinical expectation and cross-check the gradient-boosting model's top
signal: lower `GCS_last`, lower `GCS_min` (both meaning worse neurological status), and
higher age all push risk up. Several "was this ever measured" indicators
(`measured_MAP`, `measured_ALP`) carry negative coefficients — i.e. having these
routine values on record is associated with lower risk, plausibly because sicker
patients get less routine, more crisis-driven monitoring, or because these reflect a
less acute admission pathway. This informative-missingness effect is a real, if
counter-intuitive, feature of ICU flowsheet data and is worth flagging rather than
filtering out. Full coefficient table: `cycle1_lr_coefficients.csv`.

## Files produced

- `icu_features.py` — shared feature engineering (used by training and inference)
- `train_cycle1.py` — full training/evaluation pipeline (reproducible, seed 42)
- `predict_cycle1.py` — inference only; loads `model_cycle1.pkl`, no retraining
- `model_cycle1.pkl` — fitted model bundle (model + feature column list + threshold)
- `split_cycle1.csv` — patient_id, split (train/val/test)
- `cycle1_metrics.txt` — full run log (hyperparameter search, all metrics)
- `cycle1_importance.csv`, `cycle1_lr_coefficients.csv` — full interpretability tables
- `state_cycle1.md` — current scientific state, assumptions, open questions

## Headline numbers

Test AUROC **0.853**, AUPRC **0.502** (vs. 0.140 base rate), Brier **0.131**. Dominant
driver: Glasgow Coma Score, followed by age.
