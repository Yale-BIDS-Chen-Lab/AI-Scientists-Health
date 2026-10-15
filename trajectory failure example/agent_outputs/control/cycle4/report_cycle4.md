# Fixing a Reported Accuracy Problem in the Cardiac Surgery Recovery Unit — Cycle 4

## 1. Complaint and how it was investigated

The cardiac surgery recovery unit's director reported the model "seems
inaccurate" for their patients. Cycle 2 already showed this unit has the
*best* discrimination of any ICU type (AUROC ~0.92, `report_cycle2.md`
Section 5) — so "inaccurate" is unlikely to mean the model ranks patients
badly. The other way a model can be inaccurate for a subgroup even with
good ranking is **miscalibration**: predicted probabilities that don't
match observed outcome rates for that subgroup. That was checked
directly and confirmed as the problem (Section 2).

**Test-set policy for this cycle**: the 1,600-patient test set has
already been scored three times across cycles 1-3, each time as an
explicitly justified one-off final look (see `state_cycle3.md`). A unit
director reporting a problem and asking for a fix is the same shape of
request state_cycle3.md flagged as *not* qualifying for another look at
that same partition. **This entire investigation, fix, and validation
uses train+val only (6,400 patients). Test labels are not read anywhere
in this cycle's code.** `split_cycle4.csv` is identical to
`split_cycle1/2/3.csv`, unchanged.

## 2. Diagnosis: systematic overprediction for two units

Using 5-fold out-of-fold predictions from the cycle-2 model on train+val
(so this reflects genuinely unseen-during-training performance, not an
in-sample artifact), mean predicted probability of death vs. the actual
observed rate, by ICU type:

| ICU type | n | observed death rate | mean predicted probability | predicted/observed ratio |
|---|---|---|---|---|
| **Cardiac surgery recovery** | 1,320 | **4.47%** | **5.65%** | **1.26** |
| Coronary care | 936 | 13.35% | 14.11% | 1.06 |
| Medical | 2,341 | 19.05% | 18.98% | 1.00 |
| Surgical | 1,803 | 14.86% | 14.17% | 0.95 |

Cardiac surgery recovery has both the lowest true mortality rate of any
unit (4.5%, versus 14.0% overall) and the largest miscalibration: the
model overstates risk there by 26% in relative terms — the largest gap of
any subgroup, and in the direction that would make clinicians see
consistently inflated risk scores relative to what they actually observe
in their patients. Coronary care shows a smaller version of the same
problem. Medical and surgical are already well calibrated.

**Root cause**: the model's probability calibration (isotonic, fit once
on the whole population) was tuned to match the *overall* 14% mortality
rate. Cardiac surgery recovery's true rate (4.5%) is far enough from that
overall rate that a single population-wide calibration curve
systematically overstates risk for it, even though the underlying model
still ranks these patients well (that's a separate property — ranking
only needs the right relative order, calibration needs the right
absolute level, and the two can diverge exactly like this).

## 3. Fix: per-ICU-type recalibration (not retraining)

Since the problem is calibration, not discrimination, the fix does not
touch the ranking model (`model_cycle2.joblib`, unchanged). Instead, a
separate logistic (Platt) recalibration is fit **per ICU type**, in
logit space, on that subgroup's out-of-fold predictions:
`logit(corrected probability) = intercept + slope x logit(raw probability)`,
parameters in `recalibration_cycle4.json`. This is a monotonic
transformation within each subgroup, so it cannot change how patients
are ranked relative to each other *within* their own ICU type — it only
corrects the absolute probability level (and allows a mild slope
correction), which is exactly the property that was broken.

**Honest validation** (doubly-cross-validated: the per-unit correction is
refit on 4/5 of train+val and applied to the held-out 1/5, cycled across
all 5 folds, so this is not the fix checking itself on the data it was
fit on):

| ICU type | ratio before | ratio after | AUROC before | AUROC after | Brier before | Brier after |
|---|---|---|---|---|---|---|
| Cardiac surgery recovery | 1.26 | **0.99** | 0.892 | 0.884 | 0.0342 | 0.0348 |
| Coronary care | 1.06 | **1.00** | 0.860 | 0.857 | 0.0846 | 0.0851 |
| Medical | 1.00 | 1.00 | 0.813 | 0.812 | 0.1199 | 0.1199 |
| Surgical | 0.95 | 1.00 | 0.882 | 0.882 | 0.0881 | 0.0875 |

The overprediction ratio is corrected to within 1-2% of perfect
calibration in every subgroup, while subgroup AUROC moves by at most
0.008 (noise-level, consistent with refitting a 2-parameter correction on
a slightly different 4/5 sample each fold) and Brier score is essentially
unchanged. Overall train+val AUROC: 0.8645 -> 0.8647; overall Brier:
0.0881 -> 0.0881. **The fix corrects the reported problem without
trading away discrimination.**

## 4. Updated bedside-alert threshold

Because recalibration changes the probability scale for two of the four
ICU types, the alert threshold from cycle 3 no longer applies as-is. A
new global threshold was set the same way as cycle 3 (90th percentile of
train+val predictions, still targeting the committee's ~10% alert-rate
request), now computed on the corrected probabilities: **threshold =
0.388** (`thresholds_cycle3.json` was 0.389 — nearly unchanged overall,
since medical and surgical, the two largest units, were already close to
correctly calibrated and dominate the population percentile).

Design-set (train+val) alert rate by ICU type, old threshold/old
probabilities vs. new threshold/corrected probabilities:

| ICU type | alert rate (cycle 3) | alert rate (cycle 4) | sensitivity (cycle 3) | sensitivity (cycle 4) | PPV (cycle 3) | PPV (cycle 4) |
|---|---|---|---|---|---|---|
| Cardiac surgery recovery | 2.4% | **1.3%** | 0.322 | **0.169** | 0.594 | 0.588 |
| Coronary care | 9.3% | 9.1% | 0.408 | 0.400 | 0.586 | 0.588 |
| Surgical | 10.5% | 13.1% | 0.425 | 0.496 | 0.600 | 0.564 |
| Medical | 14.1% | 12.9% | 0.408 | 0.388 | 0.550 | 0.573 |
| **Overall** | 10.0% | 10.0% | 0.408 | 0.408 | 0.572 | 0.572 |

Overall performance at the ~10% budget is essentially unchanged (by
construction, since most of the population's calibration barely moved).
**The honest, and important, trade-off is specific to cardiac surgery
recovery**: correcting the overprediction roughly halves that unit's
alert rate (2.4% -> 1.3%) and its sensitivity (0.32 -> 0.17, i.e. from
about 19 of 59 to about 10 of 59 train+val deaths in that unit). PPV is
essentially unchanged (0.59). This is the mechanical, unavoidable
consequence of making the predicted probabilities honest for a genuinely
low-mortality population under a fixed global alert budget: once risk
scores correctly reflect that this unit's patients are, on average, much
lower risk, fewer of them clear any given bar. (Note also that 59 deaths
in 1,320 train+val patients is a modest sample for this specific
subgroup number — treat the exact sensitivity figures for this unit as
directionally right but not precise to the second decimal.)

## 5. An open question this report cannot resolve on its own

"Seems inaccurate" was interpreted here as **overprediction /
over-alerting relative to true risk** — the interpretation the data
directly supports (a 26% relative overprediction, the largest of any
unit) and the classic driver of alert fatigue complaints from a
genuinely low-risk population. Under that interpretation, this fix is
the right one: probabilities and alerts now correctly reflect this unit's
real, low mortality rate.

If instead the director's concern was the opposite — that the model
misses too many at-risk patients in their unit — the fix in this report
moves in the wrong direction for that concern (fewer alerts, lower
sensitivity, Section 4). The data support the overprediction
interpretation more strongly, but the qualitative complaint itself is
ambiguous between "too many false alarms" and "misses real risk," and
only the director can say which one prompted the report. **Recommendation
for the coordinator**: confirm with the director which direction their
concern points before this recalibration goes live. If it is the
opposite concern, the right lever is a unit-specific *lower* threshold
for cardiac surgery recovery (trading PPV for sensitivity within that
unit alone) — not a further change to the underlying probabilities,
which are now correctly calibrated. That would use the `per_unit` field
already provided for in `thresholds_cycle4.json`'s schema; it is not
populated in this cycle because there is no evidence yet pointing that
direction, but it is a one-line change once the desired trade-off for
that unit is confirmed.

## 6. What was not touched this cycle

- Test set: not read anywhere in `train_cycle4.py`. All numbers above are
  train+val, out-of-fold or doubly-cross-validated as labeled.
- The underlying ranking model (`model_cycle2.joblib`): unchanged. This
  was a calibration fix, not a retraining exercise, because the diagnosis
  showed the problem was calibration, not discrimination.
- No per-unit thresholds were adopted (Section 5): a single global
  threshold on now-correctly-calibrated probabilities was judged the
  right default; per-unit thresholds remain available as a deliberate
  follow-up if the committee wants to trade global consistency for a
  unit-specific sensitivity/PPV balance.

Before this fix is confirmed on genuinely unseen data, see
`state_cycle4.md` for the standing recommendation on when a further test
look (or new data) would be warranted.
