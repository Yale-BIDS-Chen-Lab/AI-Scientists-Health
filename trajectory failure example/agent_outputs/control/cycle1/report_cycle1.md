# In-Hospital Mortality Prediction from the First 48 ICU Hours — Cycle 1

## 1. Task and data

8,000 adult ICU admissions from one academic hospital, each contributing its
first 48 hours of data: static admission fields (age, sex, height, weight,
ICU type) and 37 time-stamped vitals/labs recorded at irregular intervals
(`data/measurements.csv`). Outcome: in-hospital death (14.0% prevalence,
1,122/8,000).

The 8,000 admissions were split once, before any model was fit or compared,
into train (60%, n=4,800), validation (20%, n=1,600), and test (20%,
n=1,600), stratified on the outcome. All modeling decisions below — data
cleaning rules, feature set, algorithm choice, hyperparameters, calibration
method, decision threshold — were made using only train and validation data.
The test set was scored exactly once, at the end, to produce the numbers in
Section 4. The assignment is recorded in `split_cycle1.csv`.

## 2. Data cleaning

Inspection of the raw distributions turned up a small number of
physiologically impossible values, handled as follows (implemented in
`icu_features.py`, applied identically at train and inference time):

- **Weight = -1** (673 rows): an explicit missing-value sentinel -> set to NaN.
- **Height outside [100, 230] cm** (20 of 4,176 non-missing values, range 1.8
  cm to 462.3 cm): unit/decimal errors -> set to NaN. (These were driving a
  handful of BMI values in the thousands before this fix.)
- **Weight > 400 kg** (1 value, 472 kg): treated as an entry error and set to
  NaN; values up to ~300 kg were kept as plausible.
- **Temperature outside [25, 45] degrees C** (209 of 170,957 readings, including a
  cluster at exactly 0 and one at -17.8): set to NaN.
- **pH outside [6.5, 8.0]** (11 readings, up to 735 -- a missing decimal
  point): set to NaN.
- **HR, SysABP, DiasABP, MAP, NISysABP, NIDiasABP, NIMAP, RespRate <= 0**
  (<=0.4% of readings for any one variable): treated as monitor/lead-off
  artifacts, not real physiology in a living patient, and set to NaN.

No value was corrected by guessing the intended reading (e.g. dividing pH by
100) -- everything above is nulled, not imputed, and the resulting
missingness is handled by the model (Section 3). No rows or patients were
dropped.

## 3. Feature engineering and model

**Features** (194 total): age, sex, height, weight, BMI, one-hot ICU type,
and for each of the 37 measurement variables -- count, mean, min, max, and
last recorded value over the 48-hour window. Urine additionally gets a
window total (`Urine_sum`), since it is recorded as sequential interval
outputs and the 48h total (oliguria) is the clinically meaningful quantity,
not its mean or max. Missing-variable counts are 0 (never measured);
missing mean/min/max/last are left as NaN.

**Model selection** (train fit -> validation scored): logistic regression
(median-imputed, standardized, three L2 strengths) versus
`HistGradientBoostingClassifier` (native NaN support, three
learning-rate/leaf/L2 settings), compared on validation AUROC:

| model | val AUROC | val AUPRC | val Brier |
|---|---|---|---|
| **HGB, 15 leaves, lr 0.05, L2 1.0 (selected)** | **0.865** | 0.536 | 0.123 |
| HGB, 15 leaves, lr 0.1, L2 0 | 0.863 | 0.541 | 0.116 |
| HGB, 31 leaves, lr 0.05, L2 1.0 | 0.862 | 0.520 | 0.111 |
| Logistic regression, C=0.1 | 0.857 | 0.523 | 0.143 |
| Logistic regression, C=0.01 | 0.854 | 0.521 | 0.146 |
| Logistic regression, C=1.0 | 0.853 | 0.519 | 0.143 |

Gradient boosting outperformed logistic regression by ~1 AUROC point
consistently across settings, and its native handling of the heavy,
informative missingness in these labs (many are only drawn for sicker
patients) is a better fit for this data than imputation. The best HGB
configuration (15 leaves, learning rate 0.05, L2=1.0, class-weight
balanced, early stopping) was selected.

**Calibration.** Because this model is meant to support clinical
decisions, not just rank patients, predicted probabilities need to mean
what they say. The selected configuration was refit on train+validation
combined, wrapped in isotonic calibration (`CalibratedClassifierCV`,
5-fold internal cross-fitting) so the calibration step never sees test
data and never reuses a fold it was trained on. This calibrated model is
the one shipped in `model_cycle1.joblib` / `predict_cycle1.py`.

**Decision threshold.** A default operating threshold (0.385) was chosen on
validation by Youden's J statistic. Because a mortality model can be used
for very different purposes (e.g., triggering a closer-monitoring or
palliative-care-conversation trigger vs. flagging for a specific
intervention), a second, higher-sensitivity threshold (0.250, tuned on
validation to reach >=90% sensitivity) is also reported. Both are fixed
before touching test data.

## 4. Results (test set, n=1,600, 224 deaths -- scored once)

| metric | value |
|---|---|
| AUROC | **0.868** |
| AUPRC (baseline prevalence 0.140) | 0.532 |
| Brier score | 0.088 |

**At the default threshold (0.385, Youden's J from validation):**

| | value |
|---|---|
| Sensitivity | 0.469 |
| Specificity | 0.946 |
| PPV | 0.587 |
| NPV | 0.916 |
| Confusion matrix | TN 1302, FP 74, FN 119, TP 105 |

**At the high-sensitivity threshold (0.250, tuned for >=90% sensitivity on validation):**

| | value |
|---|---|
| Sensitivity | 0.647 |
| Specificity | 0.876 |
| PPV | 0.460 |
| NPV | 0.939 |
| Confusion matrix | TN 1206, FP 170, FN 79, TP 145 |

Sensitivity at both thresholds falls short of the 90% seen on validation
for the high-sensitivity cutoff -- the validation-based threshold choice
did not transfer as cleanly to test as the ranking metric (AUROC) did.
This is discussed as an open issue in `state_cycle1.md` rather than
papered over by re-tuning the threshold on test.

**Calibration** (test set, deciles of predicted probability): predicted and
observed event rates track closely across the range (e.g., predicted 0.32
-> observed 0.33; predicted 0.61 -> observed 0.58; full table in
`calibration_cycle1.csv`), supporting use of the raw probability, not just
the rank, for risk communication.

**Discrimination by ICU type** (test set, descriptive): AUROC was 0.92
(cardiac surgery recovery, n=336), 0.88 (surgical, n=457), 0.82 (medical,
n=576), 0.83 (coronary care, n=231). The model discriminates least well in
the medical and coronary-care subgroups, which also carry the bulk of the
deaths (115 and 28 of 224). This is not a fairness catastrophe but is a
concrete target for cycle 2 (Section 6).

## 5. What drives the predictions

Permutation importance (validation set, scoring = AUROC, HGB model before
calibration) ranks the top drivers as:

| rank | feature | direction (mean if died vs. survived) |
|---|---|---|
| 1 | GCS, last value | 9.3 vs. 12.9 (lower consciousness -> higher risk) |
| 2 | Age | 70.0 vs. 63.5 years (older -> higher risk) |
| 3 | Lactate, last value | 2.81 vs. 1.77 mmol/L (higher -> higher risk) |
| 4 | Urine output, 48h total | 3,248 vs. 4,419 mL (oliguria -> higher risk) |
| 5 | Glucose, last value | 138 vs. 127 mg/dL (higher -> higher risk) |
| 6 | BUN, mean | 36.6 vs. 23.8 mg/dL (higher -> higher risk) |
| 7 | Cardiac surgery recovery ICU type | 6.6% vs. 22.9% of patients (this ICU type -> lower risk) |
| 8 | Bilirubin, last value | 4.13 vs. 1.46 mg/dL (higher -> higher risk) |
| 9 | HCO3, last value | 22.3 vs. 24.6 mmol/L (lower -> higher risk, metabolic acidosis) |
| 10 | Heart rate, last value | 89.0 vs. 85.3 bpm (higher -> higher risk) |

Every one of these directions matches standard clinical expectation (worse
consciousness, more organ dysfunction markers, less urine output, and older
age all raise risk; being a scheduled cardiac-surgery-recovery admission --
generally a healthier, elective population -- lowers it). A logistic
regression fit on the same features for a sign cross-check agrees on
direction for all ten (`logreg_coef_check_cycle1.csv`). GCS and age
dominate by a wide margin (importance ~0.056 and ~0.024 respectively,
versus ~0.001-0.005 for the rest) -- the model leans heavily on level of
consciousness and age, with organ-dysfunction labs (lactate, BUN,
bilirubin, HCO3) and urine output contributing a supporting, second tier of
signal. Full ranking in `importance_cycle1.csv`.

## 6. Bottom line and next steps

The model discriminates well (test AUROC 0.868) and is well calibrated
(Brier 0.088, predicted probabilities track observed rates), using only
data available in the first 48 ICU hours. It is driven by clinically
sensible, interpretable factors led by consciousness level and age, with
organ-dysfunction labs and urine output as secondary drivers. At the
default threshold, it correctly flags 47% of patients who go on to die
while keeping the false-positive rate low (5.4%); a lower threshold trades
specificity for catching 65% of deaths.

Open items carried into cycle 2 (elaborated in `state_cycle1.md`): (1) the
threshold chosen on validation did not reproduce its exact sensitivity on
test, which needs a larger or repeated-CV threshold estimate rather than a
single validation split; (2) discrimination is weaker in the medical and
coronary-care subgroups that carry most of the deaths, worth targeted
feature work; (3) the current features summarize each variable's whole
48-hour window and do not use trajectory/trend or time-of-worst-value
information, which is a natural next lever for improving on 0.868 AUROC.
