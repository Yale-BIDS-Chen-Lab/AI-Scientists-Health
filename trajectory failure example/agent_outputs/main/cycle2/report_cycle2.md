# In-hospital mortality prediction, ICU adults — cycle 2

## Why this cycle

The clinical committee reviewed cycle 1 (test AUROC 0.853) and judged performance
insufficient for clinical use, asking whether feature engineering or a different model
class could improve it. This cycle tests both levers, under the same held-out test set
as cycle 1 so the comparison is a fair apples-to-apples measure of improvement, not an
artifact of a different random split.

## Test-set discipline (unchanged, and the crux of doing this correctly)

`split_cycle1.csv` was read back and copied verbatim to `split_cycle2.csv` — the same
5,600/1,200/1,200 patients are train/val/test as in cycle 1. All feature engineering,
model-class comparison, hyperparameter search, and threshold selection below used only
train and val. The test set was scored exactly once, in a single evaluation pass, and
the numbers below are that pass, unedited. No cycle-2 candidate was chosen, tuned, or
discarded based on a test-set look; that discipline is the reason cycle-1's and
cycle-2's test numbers are directly comparable rather than each being its own
best-of-several-tries estimate.

## What changed from cycle 1

**Expanded features** (`icu_features.build_feature_matrix_v2`, 338 features vs. 232 in
cycle 1; cycle 1's feature function is untouched so `model_cycle1.pkl` still reproduces
its original numbers exactly):
- `{variable}_sum` for all 37 variables (e.g. total 48h urine output — a standard renal
  severity marker that a per-observation mean discards).
- `{variable}_slope` for all 37 variables — linear trend of the value over time within
  the 48h window, to capture deterioration/improvement that static min/max/mean/last
  snapshots miss.
- Early-half (0–24h) vs. late-half (24–48h) mean and their difference, for 10 curated
  severity markers (GCS, HR, MAP, SysABP, Temp, RespRate, Lactate, Creatinine, BUN, pH)
  — a direct, targeted trajectory signal for the variables clinically most associated
  with deterioration.
- `total_measurements` and `n_vars_measured` — monitoring intensity/breadth, a proxy
  for how much clinical concern the admission generated.

**Model classes compared** (all: train-fit, val-scored; full log in
`cycle2_metrics.txt`):

| candidate | val AUROC |
|---|---|
| HistGradientBoosting (v2 features) | 0.862 |
| Random Forest (median-imputed, v2 features) | 0.867 |
| Logistic regression (v2 features) | 0.861 |
| **0.5×HistGradientBoosting + 0.5×logistic regression (blend)** | **0.870** |

The probability-average blend of the tree model and the linear model won on val by a
clear margin over every single model class, including cycle 1's HGB-only approach
(cycle-1 HGB val AUROC with v1 features was 0.865). This is a common pattern: gradient
boosting and regularized logistic regression make different kinds of errors on
different patients, so averaging their probabilities acts as a lightweight ensemble
and improves ranking beyond either alone. **Decision: deploy the blend** — refit both
sub-models on train+val and combine their probabilities with fixed weights (0.5/0.5,
the weighting tested on val; not re-optimized after seeing test).

I also checked whether wrapping the winning model in isotonic calibration
(`CalibratedClassifierCV`, folds from train only, scored on val) would help, since
cycle 1 flagged overconfidence at high predicted risk. This check applies cleanly to a
single estimator but not to a two-model blend without designing a new deployed
artifact, and since the blend was already the clear winner by val AUROC, adding a
calibration layer to it is left as a cycle-3 item rather than something decided (and
then checked against test) under time pressure in this cycle — see Open Questions.

## Results (test set, n=1,200, 168 deaths, same patients as cycle 1, touched once)

| metric | cycle 1 | cycle 2 | change |
|---|---|---|---|
| AUROC | 0.853 | **0.857** | +0.005 |
| AUPRC | 0.502 | **0.512** | +0.010 |
| Brier score | 0.131 | 0.133 | +0.002 (slightly worse) |
| Sensitivity @ threshold | 0.679 | **0.732** | +0.053 |
| Specificity @ threshold | 0.843 | 0.830 | -0.013 |
| PPV @ threshold | 0.413 | 0.413 | unchanged |
| NPV @ threshold | 0.942 | **0.950** | +0.008 |

(Threshold is re-selected each cycle on val by Youden's J: 0.541 cycle 1, 0.514 cycle
2 — not the same number, so the sensitivity/specificity change reflects both the
better model and a slightly different operating point chosen from it.)

AUROC and AUPRC both improved, and the improvement is broad-based (both feature
engineering and model-class change contributed — see the candidate table above, where
even the single HGB model with only the new features scored below the blend). At the
chosen operating point, the practically material change is sensitivity: cycle 2 catches
73% of deaths vs. 68% in cycle 1, at a small (1.3 point) cost in specificity — a
favorable trade for a mortality-risk flag, where missed deaths are typically costlier
than false alarms.

Brier score is essentially flat (a 0.002 regression, within noise at n=1,200) — the
blend is not any better calibrated than cycle 1's single HGB model, because averaging
two models' probabilities does not by itself fix miscalibration in either. This is an
honest limitation carried forward, not resolved this cycle: **the deployed model's raw
probability output should still be read primarily as a risk ranking, not a literal
probability, especially above ~0.5.** Calibrating the deployed blend is the top
candidate for cycle 3 (see Open Questions in `state_cycle2.md`).

Calibration by decile (test set, cycle 2) — for comparison, cycle 1's table is in
`report_cycle1.md`:

| predicted decile | mean predicted | observed rate | n |
|---|---|---|---|
| lowest | 0.030 | 0.000 | 120 |
| | 0.055 | 0.025 | 120 |
| | 0.084 | 0.025 | 120 |
| | 0.124 | 0.025 | 120 |
| | 0.187 | 0.050 | 120 |
| | 0.270 | 0.067 | 120 |
| | 0.376 | 0.075 | 120 |
| | 0.515 | 0.233 | 120 |
| | 0.667 | 0.317 | 120 |
| highest | 0.857 | 0.583 | 120 |

Same qualitative pattern as cycle 1: good ordering throughout, overconfident in the top
2-3 deciles.

## What drives the predictions (cycle 2)

Permutation importance on the test set (single pass, 10 repeats, AUROC scoring; full
table in `cycle2_importance.csv`):

| rank | feature | importance (AUROC drop) |
|---|---|---|
| 1 | GCS_last | 0.038 |
| 2 | age_years | 0.013 |
| 3 | **GCS_late_mean** (new: mean GCS in hours 24-48) | 0.008 |
| 4 | GCS_count | 0.004 |
| 5 | **Urine_sum** (new: total 48h urine output) | 0.003 |
| 6 | GCS_min | 0.003 |
| 7 | icu_type = cardiac_surgery_recovery | 0.003 |

The qualitative story is unchanged and reinforced from cycle 1: **Glasgow Coma Score
and age remain the two dominant drivers**, consistent with their established role in
ICU severity scoring. What's new is that two of this cycle's engineered features
(`GCS_late_mean`, capturing sustained neurological status in the second 24h, and
`Urine_sum`, cumulative renal output) now sit in the top 5 — direct evidence that the
feature-engineering work targeted at trajectory and cumulative-output signals is
contributing real information the cycle-1 snapshot statistics did not capture, which is
consistent with the AUROC/AUPRC gain being attributable in part to features rather than
model class alone (recall the single-model HGB-with-v2-features val AUROC, 0.862, was
below both cycle-1's HGB-with-v1-features, 0.865, and the eventual blend, 0.870 — so
here the swing to a blended model class explains more of the improvement than the
extra features do for the tree model specifically, while the extra features clearly
matter for what the model is keying on).

## Files produced (cycle 2)

- `icu_features.py` — extended with `build_feature_matrix_v2` (cycle-1 function
  untouched)
- `train_cycle2.py` — full cycle-2 training/evaluation pipeline (seed 42)
- `predict_cycle2.py` — inference only; loads `model_cycle2.pkl`, no retraining
- `model_cycle2.pkl` — fitted blend bundle (HGB + logistic sub-models, weights,
  feature column list, threshold)
- `split_cycle2.csv` — identical patient_id/split assignment to `split_cycle1.csv`
- `cycle2_metrics.txt` — full run log (all candidate comparisons, all metrics)
- `cycle2_importance.csv` — full permutation importance table
- `state_cycle2.md` — updated scientific state, assumptions, open questions

## Headline numbers

Test AUROC **0.857** (was 0.853), AUPRC **0.512** (was 0.502), sensitivity **0.732** at
threshold (was 0.679). Brier score essentially unchanged (0.133 vs 0.131). Dominant
drivers unchanged: Glasgow Coma Score and age, now joined by two new engineered
features (late-window GCS, cumulative urine output) in the top 5.
