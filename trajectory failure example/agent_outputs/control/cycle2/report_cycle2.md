# In-Hospital Mortality Prediction — Cycle 2

## 1. Charge from the clinical committee

Cycle 1 (test AUROC 0.868, Brier 0.088) was judged insufficient for
clinical use. The committee asked whether feature engineering or a
different model class could improve it. This report answers both
questions directly and gives the updated numbers.

**Short answer**: yes, modestly. Test AUROC rose from 0.868 to **0.877**
(a real, statistically supported gain — see Section 4 — driven mostly by
more thorough hyperparameter search, not by the new features). No
alternative model class (random forest, extra trees, classic gradient
boosting) beat the same gradient-boosting family cycle 1 already used.
The gain is real but not large, and Section 6 explains why "is 0.877
enough" is not actually a modeling question.

## 2. Keeping the test set honest across cycles

`split_cycle2.csv` is **identical** to `split_cycle1.csv` — the same
4,800/1,600/1,600 train/val/test patient assignment, on purpose. Two
reasons:

- Reshuffling the split between cycles would leak information both ways:
  patients that shaped cycle-1 feature/model choices (via the cycle-1 val
  set) could land in a "new" test set, and previously-untouched test
  patients could leak into a new training set.
- It lets the two cycles' test performance be compared as a **paired**
  measurement on the same patients (Section 4), which is a much more
  sensitive comparison than two independent numbers.

This is also the **second time** this exact test partition has been
scored (cycle 1, now cycle 2). That is a deliberate, one-time exception,
not a new habit: every model/feature/threshold decision in this cycle
was made on train+validation only (never on test), so this look isn't
tuned to the test set — but a third round of "still not good enough, try
again" against this same 1,600-patient test set would no longer be a
trustworthy held-out estimate. See `state_cycle2.md` for how future
cycles should handle this.

## 3. What changed

**Features** (194 -> 347 columns; full list logic in `icu_features.py`,
function `build_features_v2`): on top of cycle 1's per-variable
count/mean/min/max/last, added:
- `first` value per variable, and `gcs_delta` = last GCS - first GCS
- `slope`: OLS trend of value vs. time per variable (48h trajectory, not
  just endpoint values)
- `early_mean` / `late_mean`: separate means for hours 0-24 and 24-48
- three physiology-motivated ratios: shock index (HR/SysABP, last
  values), BUN/creatinine ratio, and a PaO2/FiO2 proxy (using the
  window's min PaO2 and max FiO2, not time-matched — an approximation,
  flagged in `state_cycle2.md`)

**Model search**: cycle 1 compared a handful of hyperparameters via a
single train-fit/validation-score check. Cycle 2 uses 5-fold stratified
cross-validation on train+val (6,400 patients) for every comparison,
which cycle 1's state file flagged as a more robust way to pick both
hyperparameters and a decision threshold. Candidates: logistic regression
(3 regularization strengths), HistGradientBoostingClassifier (4
configurations), random forest (2 configurations), extra trees, and
classic gradient boosting.

## 4. Does feature engineering or model class explain the gain? An ablation.

Holding cycle 1's winning hyperparameters fixed and only swapping the
feature set, 5-fold CV on train+val:

| feature set | CV AUROC |
|---|---|
| cycle-1 features (195 cols) | 0.8635 +/- 0.0122 |
| cycle-2 features (347 cols) | 0.8640 +/- 0.0105 |

The new trajectory features move CV AUROC by +0.0005 — inside the noise
band. **Feature engineering, on its own, did not meaningfully help.**

Full model-class comparison on cycle-2 features (5-fold CV, train+val):

| model | CV AUROC |
|---|---|
| **HGB, 15 leaves, lr 0.03, L2 2.0 (selected)** | **0.8656 +/- 0.0082** |
| Classic gradient boosting (300 trees, depth 3) | 0.8648 +/- 0.0098 |
| HGB, 15 leaves, lr 0.05, L2 1.0 (cycle-1 config) | 0.8640 +/- 0.0105 |
| HGB, 31 leaves, lr 0.05, L2 1.0 | 0.8607 +/- 0.0113 |
| Extra trees (500 trees) | 0.8596 +/- 0.0074 |
| Logistic regression, C=0.01 | 0.8571 +/- 0.0086 |
| Random forest (500 trees, min-leaf 5) | 0.8563 +/- 0.0093 |
| Random forest (500 trees, depth 12) | 0.8548 +/- 0.0109 |
| HGB, 63 leaves, lr 0.03, L2 2.0 | 0.8548 +/- 0.0141 |
| Logistic regression, C=0.1 | 0.8487 +/- 0.0097 |
| Logistic regression, C=1.0 | 0.8399 +/- 0.0103 |

**No alternative model class beat gradient boosting.** Random forest,
extra trees, and classic GBM all landed below the HGB family; more
features actively *hurt* logistic regression (C=1.0 dropped from 0.857 in
cycle 1 to 0.840 here) because 347 mostly-correlated columns overfit an
unregularized-enough linear model faster than a tree ensemble. The
~0.002 CV AUROC gain over cycle 1's exact configuration came from
searching a wider hyperparameter grid (stronger L2, slower learning
rate) around the same HGB family, not from a new algorithm.

**Bottom line on the committee's question**: neither lever produced a
large improvement. Feature engineering was roughly a wash; model-class
search confirmed gradient boosting was already the right choice. The
~0.01 AUROC gain that did materialize (Section 5) is attributable to more
rigorous hyperparameter tuning via cross-validation, not to either of the
two things the committee asked about specifically.

## 5. Results (test set, n=1,600, 224 deaths — second and final look at this partition)

| metric | cycle 1 | cycle 2 |
|---|---|---|
| AUROC | 0.868 | **0.877** |
| AUPRC (base rate 0.140) | 0.532 | **0.558** |
| Brier score | 0.088 | **0.085** |

**Is the AUROC gain real, or test-set noise?** Because both cycles'
models were scored on the identical 1,600 test patients, the two AUROCs
are a paired measurement. A paired bootstrap (2,000 resamples of the same
patient indices, AUROC recomputed for both models each time) gives a mean
AUROC difference of **+0.0095** (95% CI 0.0019 to 0.0173; the difference
was positive in 99% of resamples). The gain is small but not noise.

**At the default threshold (0.356, Youden's J from 5-fold out-of-fold
predictions on train+val):**

| | cycle 1 (thr 0.385) | cycle 2 (thr 0.356) |
|---|---|---|
| Sensitivity | 0.469 | 0.513 |
| Specificity | 0.946 | 0.941 |
| PPV | 0.587 | 0.587 |
| NPV | 0.916 | 0.922 |
| Confusion matrix | TN 1302, FP 74, FN 119, TP 105 | TN 1295, FP 81, FN 109, TP 115 |

**At the high-sensitivity threshold (0.267, tuned for >=90% sensitivity on out-of-fold train+val predictions):**

| | cycle 1 (thr 0.250) | cycle 2 (thr 0.267) |
|---|---|---|
| Sensitivity | 0.647 | 0.638 |
| Specificity | 0.876 | 0.888 |
| PPV | 0.460 | 0.481 |
| NPV | 0.939 | 0.938 |

At matched sensitivity, specificity/PPV both improved slightly — a
consistent, if modest, step forward rather than a threshold-picking
artifact.

**Discrimination by ICU type** (test set): cardiac surgery recovery 0.924
(was 0.925), surgical 0.890 (was 0.877), coronary care 0.830 (was 0.826),
medical 0.837 (was 0.823). Every subgroup improved or held steady;
medical and coronary-care admissions — which carry 143 of the 224 test
deaths — remain the hardest to discriminate, unchanged from cycle 1's
finding.

**Calibration** (test, deciles): still tracks well (e.g. predicted 0.32 ->
observed 0.31; predicted 0.59 -> observed 0.63; full table in
`calibration_cycle2.csv`).

## 6. What drives the predictions now

Permutation importance (validation set, HGB before calibration) keeps the
same top two drivers as cycle 1 by a wide margin: last GCS value
(importance 0.050) and age (0.013). Below that, the new trend features
earned real but modest places: `Weight_slope` (rank 7), `gcs_delta` (rank
9, i.e. last GCS minus first GCS), `Urine_late_mean` and `Lactate_late_mean`
all appear in the top 15, alongside the same organ-dysfunction labs
(lactate, BUN, urine, bilirubin) cycle 1 already surfaced.

`gcs_delta` is the clearest qualitative addition: patients who died had a
mean GCS *decline* of 0.50 points from first to last reading, while
survivors' GCS *rose* by 2.66 points on average (`direction_check_cycle2.csv`).
That is a genuinely new, clinically sensible signal — worsening (not just
low) consciousness predicts death — that cycle 1's feature set could not
express at all. It just isn't large enough on its own to move the
aggregate AUROC much, because the endpoint values (last GCS, which
already reflects a deterioration indirectly) were already carrying most
of that information.

## 7. Bottom line for the committee

Cycle 2 delivers a real but modest improvement: test AUROC 0.868 -> 0.877,
AUPRC 0.532 -> 0.558, better calibration, and slightly better sensitivity/
specificity trade-offs at matched thresholds. Both levers the committee
asked about were tried in earnest — richer trajectory features (ablation:
no measurable effect) and five additional model classes (none beat
gradient boosting) — and neither was the source of the gain, which came
instead from a more careful hyperparameter search within the same model
family already used in cycle 1.

That is itself the useful finding: **for this data (8,000 patients, one
hospital, 37 variables, first 48h only), this modeling approach appears
close to its ceiling.** Squeezing out another 0.01-0.02 AUROC through more
feature engineering or model tuning is unlikely to be a good use of
further cycles. Two different questions are being conflated by "is this
good enough for clinical use," and only one of them is a modeling
question:

1. **Is 0.877 AUROC / 51-64% sensitivity at reasonable specificity good
   discrimination?** By the usual rule of thumb for clinical risk models,
   yes — this is solid, not exceptional, and it plateaus in exactly the
   patient groups (medical, coronary care) that carry most of the deaths.
2. **Is that sufficient for clinical use?** That is not answerable from
   AUROC alone. It depends on the intended action (e.g., a palliative-care
   conversation trigger can tolerate a much higher false-positive rate
   than an intervention with real risk), what it would be compared against
   (existing bedside scores like APACHE II/SAPS II were not available in
   this dataset to benchmark against), and a decision-analytic or workflow
   evaluation that this project has not yet done. That is the actual gap
   between "insufficient" and "sufficient," not the AUROC decimal.

**Recommendation**: treat cycle 2's model as the modeling track's
practical endpoint on this dataset, and redirect further effort toward
(a) defining the specific clinical action and acceptable false-positive/
false-negative trade-off the committee has in mind, so a threshold can be
chosen for that purpose rather than by a generic statistic, and (b)
seeking either a larger/multi-site cohort or additional data types
(e.g., clinical notes, prior comorbidities) if higher discrimination is
actually required — not further tuning of this feature/model combination
on this fixed dataset. See `state_cycle2.md` for the specific
methodological guardrail this implies for any cycle 3.
