# In-hospital mortality prediction, ICU adults — cycle 4

## Why this cycle

The director of the cardiac surgery recovery unit reports the bedside alert (cycle 3,
single global threshold 0.7367) seems inaccurate for their patients. This cycle
diagnoses why using train/val only, fixes it, and checks the fix against test exactly
once. No new model is fit — the underlying risk score is unchanged from cycle 2.

## Diagnosis (train/val only, test not touched for this step)

Baseline mortality by ICU type is highly unequal (train+val, 8,000-patient dataset):

| icu_type | n (train+val) | mortality rate |
|---|---|---|
| cardiac_surgery_recovery | 1,424 | **4.7%** |
| coronary_care | 983 | 13.8% |
| surgical | 1,906 | 14.4% |
| medical | 2,487 | 19.2% |

Cardiac surgery recovery patients die in-hospital roughly 3-4x less often than every
other unit. Using out-of-sample validation scores (a train-only refit of the cycle-2
sub-models, scored on val — the same method used to set the cycle-3 threshold, chosen
again here so the diagnosis isn't distorted by evaluating a model on data it was
partly fit on), broken down by unit:

| icu_type | n (val) | AUROC | mean predicted risk | alert rate at the cycle-3 global threshold (0.7367) |
|---|---|---|---|---|
| cardiac_surgery_recovery | 260 | **0.884** (highest) | 0.132 | **1.2%** |
| coronary_care | 187 | 0.848 | 0.317 | 7.5% |
| surgical | 314 | 0.886 | 0.319 | 9.2% |
| medical | 439 | 0.852 | 0.411 | 16.9% |

**Root cause identified**: discrimination is not the problem — AUROC for cardiac
surgery recovery is the *best* of the four units, so the model ranks that unit's
patients by relative risk about as well as it ranks anyone else's. The problem is a
**single global threshold applied to units with very different baseline risk**. Because
cardiac surgery recovery patients' absolute predicted probabilities sit systematically
far lower (mean 0.132 vs. 0.317-0.411 elsewhere) — a direct, expected consequence of
their much lower true event rate — a cutoff set from the whole population's top 10%
almost never reaches into this unit's score range. The alert fires for roughly 1 in
80 of their admissions, against the intended 1 in 10. From the unit's point of view,
the alert is effectively silent, which is a completely reasonable basis for "this seems
inaccurate for our patients," even though the score underneath it is sound.

## Fix: per-unit thresholds

Rather than one global cutoff, each icu_type gets its own threshold, set to the 90th
percentile of *that unit's own* out-of-sample score distribution (same val-based
method as above, train-only refit). This reframes the alert as "flag the highest-risk
tenth of admissions within this unit" rather than "flag anyone above a fixed absolute
probability" — a unit-relative risk stratification, consistent with how severity
scores are typically interpreted per care setting.

| icu_type | threshold | val alert rate at this threshold |
|---|---|---|
| cardiac_surgery_recovery | **0.377** | 10.0% |
| coronary_care | 0.705 | 10.2% |
| surgical | 0.718 | 10.2% |
| medical | 0.810 | 10.0% |

## Checking the fix against test (single pass, both policies evaluated together)

| | OLD (single global 0.7367) | NEW (per-unit) |
|---|---|---|
| **cardiac_surgery_recovery** alert rate | 6.0% | 13.8% |
| **cardiac_surgery_recovery** sensitivity | 0.50 | **0.60** |
| **cardiac_surgery_recovery** specificity | 0.96 | 0.88 |
| **cardiac_surgery_recovery** PPV | 0.357 | **0.188** |
| Overall alert rate | 11.2% | 10.8% |
| Overall sensitivity | 0.435 | 0.387 |
| Overall PPV | 0.545 | 0.500 |

Full by-unit and overall breakdown for both policies: `thresholds_cycle4.json`,
`cycle4_metrics.txt`.

**Honest reading of this check, not smoothed over**: on this particular 1,200-patient
test draw, the cardiac surgery recovery subgroup has only 10 deaths total — any
subgroup metric here is estimated from a small number of events and moves a lot per
patient (each patient is worth ~10 points of sensitivity or PPV in that unit). The old
policy's test-set alert rate for this unit (6.0%) is already closer to its 10% target
than the val diagnosis predicted (1.2%), which is itself evidence of how much sampling
noise is in play at this subgroup size. Taking the numbers at face value: the new
policy raises this unit's sensitivity (0.50→0.60) and pushes its alert rate toward
the intended 10% band, but at a real, not-explained-away cost in precision (PPV
0.357→0.188 — most of this unit's alerts under the new policy are false positives).
Overall population metrics also move slightly in the wrong direction (sensitivity
0.435→0.387, PPV 0.545→0.500), mostly because the medical unit's alert budget — which
the old global threshold had over-allocated to 16.9% instead of 10% — gets pulled back
to its fair 10% share, at some sensitivity cost for that (higher-volume, higher-base-
rate) unit.

## Decision

**Deploy per-unit thresholds.** The reasoning: the diagnosed mechanism — a global
cutoff structurally under-serving a unit with a much lower base rate, despite the
model discriminating that unit's risk as well as any other — is well evidenced (a
5-fold gap in baseline mortality, the best AUROC of any unit, and a 14x gap in alert
rate relative to target) and will recur regardless of any one test draw's sampling
luck. A policy where one unit's alert is effectively non-functional (1 in 80 fired)
is a worse and more clearly wrong failure mode than a policy where alerts are
appropriately frequent but noisier in precision for a low-base-rate, low-volume unit.
The test-set PPV cost for cardiac surgery recovery is real and should not be
minimized — it is flagged explicitly below as the top open item for the next cycle,
to be revisited with a more robust (variance-reducing) estimate of that unit's
threshold once more data are available, rather than re-touching this same test set
again in the meantime.

## Files produced (cycle 4)

- `train_cycle4.py` — diagnosis (train/val only) + per-unit threshold fitting + the
  single final test check, run together
- `predict_cycle4.py` — inference only; loads `model_cycle2.pkl` (same model as
  cycles 2-3, nothing retrained), writes `patient_id,probability`
- `thresholds_cycle4.json` — `{"threshold": 0.7367..., "per_unit": {icu_type:
  threshold, ...}}` plus both policies' full test breakdown for traceability
- `split_cycle4.csv` — identical patient_id/split assignment to `split_cycle3.csv`
- `cycle4_metrics.txt` — full run log
- `state_cycle4.md` — updated scientific state, assumptions, open questions

## Headline numbers

Root cause: cardiac surgery recovery's baseline mortality (4.7%) is 3-4x lower than
other units, so cycle 3's single global threshold gave it an alert rate of ~1% instead
of the intended 10% — despite the model's discrimination there being the *best* of any
unit (AUROC 0.884). Fix: per-unit thresholds (cardiac_surgery_recovery: **0.377**,
vs. 0.70-0.81 elsewhere), each calibrated to ~10% alert rate within its own unit. On
the held-out test set (single check): cardiac surgery recovery sensitivity improves
0.50→0.60 and its alert rate moves from 6.0% toward the 10% target (13.8%, still noisy
at only 10 events), at a real precision cost (PPV 0.357→0.188); overall population
sensitivity/PPV move slightly down (0.435→0.387 / 0.545→0.500) as the medical unit's
previously over-sized alert share is corrected back to its fair 10%.
