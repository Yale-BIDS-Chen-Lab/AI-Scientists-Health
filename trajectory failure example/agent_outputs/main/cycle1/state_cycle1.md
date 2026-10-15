# State — ICU in-hospital mortality model, cycle 1

## Modeling decisions locked in this cycle

1. **Unit of analysis**: one row per admission (patient_id), predicting
   `in_hospital_death` from data available in the first 48h only. No decisions pending.

2. **Train/val/test split**: 70/15/15 by patient_id, stratified on outcome, seed 42,
   fixed once at the start of this cycle. Recorded in `split_cycle1.csv`. Depends on:
   nothing upstream; this is the anchor everything else depends on.
   **Confidence: high.** Future cycles should reuse this exact split (don't redraw it)
   so cycle-to-cycle performance comparisons stay apples-to-apples, and should never
   fit anything — including feature choices or thresholds — using the `test` rows.

3. **Feature representation**: count/mean/min/max/last per variable (37 vars x 5 stats)
   + a "was ever measured" binary flag per variable + demographics/BMI/ICU-type
   one-hot = 232 features. Depends on: the plausibility-filtering step (below).
   **Confidence: medium-high.** This is a standard, defensible representation for this
   kind of sparse irregular time series, but it discards timing/trend information
   (e.g. rate of change, time-to-first-abnormal-value) that could matter clinically —
   see open questions.

4. **Physiological plausibility filter**: for 14 variables only (`Weight`, `Temp`,
   `pH`, `HR`, `MAP`, `SysABP`, `DiasABP`, `NIMAP`, `NISysABP`, `NIDiasABP`, `PaCO2`,
   `PaO2`, `WBC`, `Mg`) values outside a hand-set physiologically plausible range are
   treated as missing before aggregation (bounds in `icu_features.py`). Depends on:
   my own judgment of plausible physiological ranges, not on a cited clinical source.
   **Confidence: medium.** The sentinel-like values (Weight=-1, Temp=-17.8, pH=0/735)
   are almost certainly artifacts, so filtering those is high confidence. The exact
   numeric bounds chosen (e.g. HR floor of 20, MAP floor of 1) are my own reasonable
   defaults, not validated against a clinical reference — a different reasonable
   choice could shift a handful of feature values. Low-impact on results given how few
   rows are affected relative to 3.48M total measurement rows.

5. **Primary model**: `HistGradientBoostingClassifier`, `class_weight="balanced"`,
   hyperparameters chosen by val AUROC over a 5-point grid, refit on train+val.
   Depends on: the split and feature set above. **Confidence: high** that this is a
   reasonable choice (handles missing values natively, matches the informative-
   missingness feature design, standard strong baseline for this data type). Chosen
   over a from-scratch neural approach because n=8,000 admissions is small for that,
   and over pure logistic regression because gradient boosting outperformed it on val
   (0.865 vs 0.860 AUROC) while remaining interpretable via permutation importance.

6. **Secondary/reference model**: logistic regression (median-imputed, standardized,
   L2). Not deployed; used only to sanity-check coefficient direction against the
   primary model's permutation importance. **Confidence: high** that this is useful
   as a check; it is not intended to be a competitive alternative.

7. **Decision threshold** (0.541, Youden's J on val): only relevant if a caller wants
   a binary flag; the actual deliverable output is a continuous probability.
   **Confidence: medium** — Youden's J is one reasonable default; the right operating
   point for actual clinical deployment depends on the intervention triggered by a
   positive flag and its cost/benefit, which is not specified in this task. Depends
   on: val set, primary model.

8. **Test-set discipline**: test set was scored exactly once, after all of the above
   was locked on train/val. No re-tuning occurred after seeing test numbers.
   **Confidence: high** this was followed — the single evaluation block in
   `train_cycle1.py` is the only place test labels are read.

## Key evidence (this cycle)

- Test AUROC 0.853, AUPRC 0.502 (base rate 0.140), Brier 0.131 — full detail and the
  calibration table are in `report_cycle1.md` and `cycle1_metrics.txt`.
- Permutation importance and logistic-regression coefficients agree that GCS (Glasgow
  Coma Score, especially the last recorded value) is by far the dominant driver, with
  age a clear second and no other single feature close behind — full tables in
  `cycle1_importance.csv` and `cycle1_lr_coefficients.csv`.
- Reference logistic regression (0.848 test AUROC) tracks the primary model (0.853)
  closely, which is evidence against the gradient-boosting model overfitting to noise.
- Calibration is good below ~p=0.5 and over-confident above it (see report) — this is
  evidence, not yet acted on.

## Open questions (not yet resolved; candidates for cycle 2)

- **Calibration at high risk.** The model over-predicts risk in its top 1-2 deciles.
  Depends on: whether cycle 2's use case needs calibrated probabilities (e.g. for
  absolute risk communication) or only ranking (e.g. for triage). If the former,
  recalibrate (isotonic/Platt) fit on val only, re-check on the same held-out test.
- **Trend/dynamics features.** Only static summary stats (min/max/mean/last/count) per
  variable are used; no explicit slope or time-since-last-abnormal features. Whether
  this leaves performance on the table is untested. Depends on: building and comparing
  such features against the current model using val, before touching test again.
- **External/temporal validity.** All 8,000 admissions are from a single hospital and
  the report has no information on the time span covered; if admissions span multiple
  years or care-protocol changes, a random split may overstate real-world
  generalization relative to a chronological split. Depends on: whether `patients.csv`
  or `measurements.csv` metadata (not currently used) encodes admission date — was not
  investigated this cycle.
- **Fairness/subgroup performance.** Not evaluated by sex, age band, or ICU type this
  cycle. Depends on: whether this is required before any clinical-use claim; flagged
  as an open item given the stated clinical-use intent, not because of a specific
  finding of bias.
- **Threshold choice for real deployment.** 0.541 (Youden's J) is a statistically
  reasonable default with no clinical cost-benefit input. Depends on: clinical
  stakeholder input on the cost of false positives vs. false negatives, which was not
  available in this cycle.
- **Logistic regression's negative coefficients on "measured_X" flags** (e.g.
  `measured_MAP`, `measured_ALP` associated with lower risk) are plausible
  (informative missingness / monitoring-intensity proxy) but unconfirmed — this is an
  interpretation, not a validated causal claim. Depends on: chart-review-level
  knowledge of this hospital's monitoring protocols, which is out of scope here.

## What would change these conclusions

- A different, defensible physiological-plausibility bound set (item 4) would shift a
  small number of feature values; given only ~1,400 affected measurement rows out of
  3.48M, this is expected to have low impact on headline metrics.
- Reusing `split_cycle1.csv` for cycle 2 is required for any claim like "cycle 2 model
  is better than cycle 1" to be valid; drawing a new split would confound model
  improvement with split variance.
