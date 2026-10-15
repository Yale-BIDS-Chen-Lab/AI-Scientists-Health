# Bedside Alert Threshold — Cycle 3

## 1. Ask

The committee wants to deploy the cycle-2 model as a bedside alert firing
on roughly 10% of admissions, and wants sensitivity, specificity, and PPV
at that operating point. This cycle does not retrain anything: the model
is `model_cycle2.joblib`, unchanged. The only new artifact is a threshold.

## 2. How the threshold was chosen

The threshold was fit on train+val only, never on test:

1. Rebuilt the exact deployed pipeline (cycle 2's gradient-boosting
   configuration wrapped in 5-fold isotonic calibration).
2. Generated **out-of-fold** predicted probabilities for all 6,400
   train+val patients via an outer 5-fold cross-validation (so the "10%
   of admissions" property is measured out-of-sample, not read off
   predictions the model already saw during its own fit — in-sample
   probabilities are over-sharp and would understate the true alert
   rate at deployment).
3. Set the threshold to the 90th percentile of those out-of-fold
   probabilities: **0.389**. By construction this fires on exactly 10.0%
   of the 6,400 train+val admissions.

Test data played no role in picking 0.389.

## 3. Test-set policy for this cycle

This is the **third** time the fixed 1,600-patient test partition
(`split_cycle3.csv`, identical to `split_cycle1.csv`/`split_cycle2.csv`)
has been scored. Cycle 2's state file flagged that repeated
iterate-then-recheck cycles against the same test set erode its value as
an unbiased estimate, but also carved out an explicit exception: a
genuine final deployment decision point. Choosing and characterizing a
fixed deployment threshold — not searching over thresholds against test,
not retraining, not asking "can we do better" — is exactly that
exception. Consistent with that plan, this is treated as the closing
look at this test partition: see `state_cycle3.md` for what would need
to change before evaluating on it again.

## 4. Headline numbers (test set, n=1,600, 224 deaths)

| metric | value |
|---|---|
| **Threshold** | **0.389** |
| **Alert rate (test)** | **11.1%** (178 of 1,600 admissions) |
| **Sensitivity** | **0.469** |
| **Specificity** | **0.948** |
| **PPV** | **0.593** |
| NPV | 0.916 |
| Confusion matrix | TN 1304, FP 72, FN 119, TP 105 |

For context, the same threshold's out-of-fold performance on the 6,400
train+val patients used to set it (design-set check, not test):
alert rate 10.0% (exact, by construction), sensitivity 0.408,
specificity 0.950, PPV 0.572, NPV 0.908. Test-set numbers are close to
this design estimate (alert rate 10.0% -> 11.1%, sensitivity 0.41 -> 0.47,
PPV 0.57 -> 0.59) — the threshold transferred well, better than the
val-tuned thresholds in cycles 1-2, likely because it was estimated from
5-fold out-of-fold predictions across all 6,400 train+val patients rather
than a single ~1,600-patient validation split.

**Reading the trade-off**: with a 10-11% alert budget against a 14%
true mortality rate, no threshold can catch every death (the alert rate
is below the prevalence). The theoretical ceiling for sensitivity at this
budget, if the model ranked patients perfectly, would be alert_rate /
prevalence = 11.1% / 14.0% = 79%. The model reaches 0.469/0.79 = 59% of
that ceiling — consistent with, not contradicted by, the discrimination
already reported in `report_cycle2.md` (test AUROC 0.877).

PPV of 0.593 means roughly 3 in 5 bedside alerts correspond to a patient
who goes on to die in-hospital — a more than 4x enrichment over the 14%
base rate, which is the practically relevant number for staff evaluating
whether an alert is "usually right."

## 5. Where the alerts land

Alert rate and PPV both vary substantially by admission type (test set,
descriptive):

| ICU type | n | deaths | alert rate | sensitivity | PPV |
|---|---|---|---|---|---|
| Medical | 576 | 115 | 16.0% | 0.43 | 0.53 |
| Surgical | 457 | 63 | 11.4% | 0.59 | 0.71 |
| Coronary care | 231 | 28 | 9.1% | 0.39 | 0.52 |
| Cardiac surgery recovery | 336 | 18 | 3.6% | 0.44 | 0.67 |

A single global threshold naturally allocates more alerts to higher-risk
admission types (medical) and far fewer to the lowest-risk type (cardiac
surgery recovery, consistent with `report_cycle1.md`/`report_cycle2.md`
identifying it as protective). This means a fixed 10% *hospital-wide*
alert rate is not a fixed 10% rate *within every unit* — medical units
should expect to see alerts on about 1 in 6 admissions, cardiac-surgery-
recovery units about 1 in 28. This is worth surfacing to the committee
explicitly since staffing/workflow expectations are usually set per unit,
not hospital-wide.

## 6. Bottom line

Threshold **0.389** delivers approximately the requested 10% alert rate
(10.0% by design on train+val, 11.1% on held-out test), with sensitivity
0.47, specificity 0.95, and PPV 0.59 on test. This is a defensible
starting operating point for a bedside alert: PPV is high enough that an
alert is usually clinically meaningful, at the cost of missing just over
half of eventual deaths, which is the direct, arithmetic consequence of
capping alerts at 10% of admissions against a 14% base rate rather than a
model weakness on top of what's already been reported.

Before wiring this into clinical workflow, two things from cycle 2 remain
unresolved and are not addressed by a threshold choice: (1) whether a
uniform hospital-wide 10% target is actually what's wanted given the
unit-level variation in Section 5, versus a unit-specific target; and
(2) the general point from `report_cycle2.md` Section 7 that "is this
good enough" depends on the specific action an alert triggers, which
still has not been specified by the committee.
