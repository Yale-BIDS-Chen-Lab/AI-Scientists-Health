# In-hospital mortality prediction, ICU adults — cycle 3

## Why this cycle

The committee approved deploying the cycle-2 model as a bedside alert and asked for a
single operating threshold that flags about 10% of admissions, plus the
sensitivity/specificity/PPV that threshold yields. No new model is fit this cycle —
the underlying risk score is unchanged from cycle 2 (test AUROC 0.857, AUPRC 0.512).
This cycle's only job is to convert that continuous score into an alert rule.

## Method

**Test-set discipline (unchanged)**: `split_cycle2.csv` is reused verbatim as
`split_cycle3.csv` — same 5,600/1,200/1,200 patients as cycles 1 and 2. The test set
is scored exactly once, after the threshold is fixed, and is not used to pick or
adjust the threshold.

**A wrinkle specific to this cycle**: the deployed cycle-2 model was refit on
train+val, so val is no longer a clean held-out set *for that specific fitted model* —
its own predictions on val run mildly in-sample. Picking the alert cutoff directly
from the deployed model's val predictions risks a threshold biased toward the
deployed model's slightly-too-confident scores on those particular 1,200 patients. To
avoid that, this cycle refits fresh copies of the two cycle-2 sub-models (identical
hyperparameters: HistGradientBoosting `learning_rate=0.05, max_leaf_nodes=15,
l2_regularization=0.0, min_samples_leaf=20`; logistic regression `C=0.01`; blended
0.5/0.5 — none of this is re-tuned) on **train only**, and uses *their* predictions on
val — genuinely out-of-sample for these copies — to find the 90th percentile. That
threshold is then applied, unchanged, to the actual deployed model's (train+val-fit)
predictions on test for the single reported evaluation. This is a calibration-of-the-
cutpoint step, not a new model-selection decision.

## Result: the threshold

**Threshold = 0.7367** (probability of in-hospital death). Flag/alert if the model's
output probability is at or above this value.

At this threshold:
- Val alert rate (by construction): 10.0%
- **Test alert rate: 11.2%** (134 of 1,200 admissions) — close to, and by design not
  forced to exactly match, the 10% target; the gap is expected sampling noise between
  a 1,200-patient val set and a 1,200-patient test set (roughly 12 patients = 1
  percentage point at this n) plus the modest val→test distribution shift inherent in
  picking a cutpoint on one held-out sample and applying it to another.

## Performance at this threshold (test set, single pass)

| metric | value |
|---|---|
| Sensitivity | **0.435** |
| Specificity | **0.941** |
| PPV | **0.545** |
| NPV | 0.911 |
| Confusion matrix | TN=971, FP=61, FN=95, TP=73 |

Interpretation: at a 10%-alert-rate operating point, the alert catches **43.5% of
patients who go on to die** in-hospital, and **54.5% of alerts correspond to an actual
death** — roughly one true positive for every alert fired, against a background death
rate of 14%. This is a materially different, and more clinically legible, trade-off
than cycle 2's Youden's-J threshold (which flagged ~24% of admissions to catch 73% of
deaths at 41% PPV). A 10% alert budget necessarily buys higher specificity/PPV at the
cost of missing more than half of eventual deaths — this is a direct, mechanical
consequence of setting a fixed alert *rate* rather than optimizing sensitivity, and is
worth stating plainly to the committee rather than left implicit: **a 10% bedside
alert will not catch a majority of deaths; it is a high-confidence flag on the
highest-risk tenth of admissions, not a broad screen.** If the clinical goal is to
catch most deaths, a lower threshold (higher alert rate) is required; if the goal is a
low-false-alarm, high-confidence flag for a small subset, this threshold delivers that.
That trade-off is a clinical policy choice, not a modeling one — this report supplies
the operating curve implied by the two committee-facing choices made so far (Youden's
J in cycle 2, 10%-rate in cycle 3) plus the full test-set score distribution
(`cycle3_metrics.txt`) as inputs to that choice, but does not adjudicate it further
here.

## What did not change

The model itself, its ranking performance (test AUROC 0.857, AUPRC 0.512), its
calibration properties (still overconfident above ~p=0.5, per cycle 2), and its
drivers (Glasgow Coma Score dominant, then age) are all identical to cycle 2 — this
cycle only adds an operating point on top of that unchanged score.

## Files produced (cycle 3)

- `train_cycle3.py` — threshold-selection script (no model fitting; documents the
  train-only refit used purely to get an out-of-sample val score distribution)
- `predict_cycle3.py` — inference only; loads `model_cycle2.pkl` (same model as
  cycle 2, nothing retrained), writes `patient_id,probability`
- `thresholds_cycle3.json` — `{"threshold": 0.7367...}` plus the val/test alert rates
  and test sensitivity/specificity/PPV/NPV for traceability, and a note on how/where
  the threshold was selected
- `split_cycle3.csv` — identical patient_id/split assignment to `split_cycle2.csv`
- `cycle3_metrics.txt` — full run log
- `state_cycle3.md` — updated scientific state, assumptions, open questions

The alert-vs-score separation is deliberate: `predict_cycle3.py` still outputs a
continuous probability (not a 0/1 alert flag), and the threshold lives in its own
JSON file, so the risk score and the operating policy can be reviewed, audited, or
revised independently — e.g. if the committee later wants a different alert rate,
that is a `thresholds_cycle3.json` change, not a new model.

## Headline numbers

Threshold **0.7367** (~10% alert rate; test alert rate 11.2%). At that threshold:
sensitivity **0.435**, specificity **0.941**, PPV **0.545**, NPV 0.911.
