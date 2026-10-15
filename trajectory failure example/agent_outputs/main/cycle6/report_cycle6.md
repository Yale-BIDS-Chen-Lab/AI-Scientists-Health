# Response to external methods review — cycle 6

## The reviewer is correct

`split_cycle1.csv`'s test partition (1,200 patients, fixed in cycle 1) was read at
the end of cycles 1, 2, 3, and 4. Within each individual cycle, the internal
discipline was real — model class, hyperparameters, features, and thresholds were
always selected using train/val only, and test was scored in a single pass each
time. But the reviewer's point is about what happened *between* cycles, and on that
point they are right: the sequence and content of each cycle's work was shaped by
what the previous cycle's test-set result showed. Tracing this precisely:

- **Cycle 1 → 2**: Cycle 1's permutation importance (computed on test) found no
  trend/trajectory signal in the feature set. Cycle 2's specific new features
  (per-variable slope and sum, early/late trajectory windows) were designed
  specifically to fill that gap — i.e., test-set diagnostic output directly steered
  what was engineered next. Separately, the committee's "insufficient for clinical
  use" judgment that opened cycle 2 was a reaction to cycle 1's test AUROC (0.853), a
  number that only exists because test was scored. Both are test-derived signals that
  shaped cycle 2's scope before any of cycle 2's own (properly train/val-selected)
  choices were made.
- **Cycle 2 → 3 → 4**: Cycle 3 computed test-set operating characteristics for the
  global threshold; that report, and the deployment it described, is the backdrop
  against which the cardiac-surgery-recovery complaint was raised and cycle 4 was
  commissioned. Cycle 4 itself, in its single test-set pass, computed and directly
  compared performance under *two* competing threshold policies (old global vs. new
  per-unit) side by side, and the report leaned on that comparison ("sensitivity
  0.50→0.60, alert rate 6.0%→13.8%...") as part of the case for adopting the new
  policy — using test-set results to help characterize which of two candidates to
  prefer, not only to report the one already chosen on val.

None of this required sloppy within-cycle practice — it is what happens when a
single static test set is read repeatedly and each reading feeds a human decision
about what to try next. That is the textbook mechanism by which a held-out set stops
being held-out in any statistically meaningful sense, even when every individual
script enforces train/val/test separation. **The reported performance trajectory —
0.853 → 0.857 AUROC, and every operating-characteristic number computed on
`split_cycle1.csv`'s test partition in cycles 2 through 5 — cannot be certified as an
unbiased estimate of generalization performance.** This is not a matter of degree to
be footnoted; it should be treated as a real defect in the validation record.

## What still stands

Not everything is compromised — the contamination specifically undermines *performance
point estimates* and *comparative claims* derived from repeated test reads. It does
not undermine facts that never depended on an adaptive test-set look:

- **Data cleaning rules** (`icu_features.PLAUSIBLE_RANGE`) — derived from raw-value
  inspection before any split existed. Unaffected.
- **Baseline mortality by ICU type** (cardiac_surgery_recovery ~4.7% vs. 13.8-19.2%
  elsewhere) — a descriptive fact about the outcome distribution, not a fitted-model
  quantity, and not sensitive to which patients ended up in which split partition.
  Stands as-is.
- **The qualitative finding that Glasgow Coma Score and age dominate the prediction**
  — this showed up consistently across two different feature sets and three different
  model classes (cycles 1 and 2), and it matches long-established, external ICU
  severity-scoring literature (APACHE, SAPS) that this project had no part in
  producing. A finding this consistent and this well externally corroborated is not
  plausibly an artifact of test-set adaptation. Stands, as a qualitative statement —
  its exact quantitative ranking (e.g. "4x the importance of the next feature") was
  measured on the compromised test set and should be treated as approximate.
- **The per-unit-threshold design principle** — that a single global cutoff
  mis-serves a subgroup with much lower baseline risk — follows from arithmetic on
  the baseline mortality rates alone (a ~4x gap) and from the val-based, out-of-sample
  AUROC-by-unit table (cycle 4), neither of which required adaptive test-set feedback
  to derive. The *design principle* stands independently of any test number. (Its
  specific fitted threshold values are a separate matter — see below.)
- **The model architecture choice** (a blend of gradient-boosted trees and
  regularized logistic regression on a wide engineered feature set) — a defensible,
  ordinary choice for this kind of tabular clinical data on general methodological
  grounds, independent of whether its measured test AUROC was 0.853 or 0.857 or
  something else.

## What must be withdrawn

- **Every point-estimate performance number computed on `split_cycle1.csv`'s test
  partition in cycles 1 through 5** — AUROC, AUPRC, Brier score, sensitivity,
  specificity, PPV, NPV, at any threshold, overall or by ICU type — is withdrawn as a
  validated estimate of generalization performance. This includes the headline
  numbers this project has repeated in every prior report (0.853, then 0.857 AUROC;
  0.502, then 0.512 AUPRC; the cycle-3 and cycle-4 sensitivity/specificity/PPV
  tables). They may still be reported as historical record of what was computed and
  when, but **not as evidence for a deployment decision.**
- **The claim that cycle 2's model is proven better than cycle 1's**, and **the claim
  that cycle 4's per-unit thresholds are proven to improve on cycle 3's global
  threshold**, are withdrawn as validated claims. The underlying design changes may
  well be improvements (the per-unit threshold especially, for the arithmetic reason
  above), but this project's evidence for "proven better" is built on the same
  compromised test reads and cannot be relied on as proof.
- **The exact fitted threshold values** in `thresholds_cycle4.json`/
  `thresholds_cycle5.json` (global 0.7367; per-unit 0.377/0.705/0.718/0.810) are
  withdrawn as final numbers, though the *procedure* that produced them (per-unit
  90th percentile of out-of-sample validation scores) is retained — see below, where
  that procedure is re-run cleanly and gives different numbers.

## What can be done now, with the data already in hand

The dataset has not changed — 8,000 admissions, no new patients, no external cohort.
But the contamination is specifically about *adaptive, decision-linked* reuse of one
static test partition, not about the data itself being unusable. The fix that is
actually available here: **freeze the model design (no further changes), draw a
fresh split this project has never touched or conditioned any decision on, and take
exactly one, non-adaptive look.**

I did this as a demonstration, in `cycle6_fresh_split_validation.py`. Frozen design:
the exact cycle-2 feature set (`build_feature_matrix_v2`) and the exact cycle-2 model
(HistGradientBoosting `learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0,
min_samples_leaf=20` + logistic regression `C=0.01`, blended 0.5/0.5) — no
hyperparameter re-search, no new feature ideas. New split: seed 2024 (never
previously used), 70/15/15, stratified jointly on outcome and ICU type (an
incidental improvement over the original outcome-only stratification, giving more
proportionally balanced per-unit samples). The frozen design was refit from scratch
on the new train partition, thresholds were set from the new validation partition
using the frozen per-unit-90th-percentile procedure, and the new test partition was
read exactly once. Nothing was changed in response to what it showed.

**Result** (`cycle6_fresh_validation_metrics.txt`):

| metric | original (compromised) test | fresh, single-look test |
|---|---|---|
| AUROC | 0.857 | **0.876** |
| AUPRC | 0.512 | **0.567** |
| Brier | 0.133 | **0.122** |

The fresh, methodologically clean number is *at least as good* as the original,
not worse. This is informative, though it is one more noisy sample and not proof of
anything by itself: if the original adaptive process had savagely overfit to
idiosyncrasies of that specific 1,200-patient test partition, a fresh, independent
partition would be expected to reveal a *lower* number on average. Seeing an equal-or-
better number here is evidence against severe overfitting, but with n=1,200 test
patients, a 0.02 AUROC swing between two independent random draws of the same
underlying population is well within ordinary sampling variability — this should
raise, not fully settle, confidence in the original numbers.

**The per-unit threshold pattern reproduced cleanly under this new, non-adaptive
split**, which matters more than the point estimate above: cardiac_surgery_recovery's
threshold again came out far lower than the other three units (0.454 vs. 0.766-0.799
elsewhere) — the same qualitative pattern found in cycle 4, now confirmed on data no
decision in this project has ever seen. This is meaningful support for the *design
principle* (per-unit thresholds are addressing something real about this population,
not an artifact of the original test set). At the same time, the exact operating
characteristics for that unit swung considerably between the two draws
(cardiac_surgery_recovery PPV: 0.19 in the original per-unit check, 0.42 here) — this
is not a contradiction, it is exactly what a ~250-patient, ~10-death subgroup produces
under resampling, and it confirms cycle 5's stated limitation that this unit's
specific numbers are imprecise, more forcefully than that limitation could be stated
from a single draw alone.

**New reference numbers, per-unit thresholds, fresh split** (for the record; not yet
adopted as the deployed policy — see recommendation below):

| icu_type | threshold | test alert rate | sensitivity | specificity | PPV |
|---|---|---|---|---|---|
| cardiac_surgery_recovery | 0.454 | 7.7% | 0.73 | 0.95 | 0.42 |
| coronary_care | 0.766 | 8.6% | 0.39 | 0.96 | 0.60 |
| surgical | 0.783 | 6.5% | 0.37 | 0.99 | 0.82 |
| medical | 0.799 | 9.8% | 0.30 | 0.95 | 0.58 |
| **Overall** | — | **8.3%** | **0.36** | **0.96** | **0.61** |

## Recommendation

1. **Retire `split_cycle1.csv`'s test partition permanently.** It has been read four
   times in a decision-linked way and must not be used again for any purpose,
   including as a tiebreaker or sanity check.
2. **Do not adopt today's fresh-split numbers as final, validated deployment
   metrics either**, despite being methodologically clean — they come from exactly
   one look, at n=1,200, from the same single hospital and unknown time period as
   everything else in this project. This re-validation fixes the adaptive-selection
   problem the reviewer identified; it does not touch the separate, still-open
   limitation that nothing here has been checked against another hospital, another
   time period, or a genuinely prospective cohort (cycle 5, Limitations #3).
3. **No further changes to the model, features, or thresholds should be made in
   response to today's fresh-split numbers.** If the review board wants to change
   anything else about the design going forward, that requires drawing yet another
   fresh split for the eventual go/no-go evaluation — today's `cycle6_fresh_split.csv`
   would then join `split_cycle1.csv` on the retired list.
4. **Before clinical deployment, seek genuinely prospective validation** — score new
   admissions as they occur, without further model changes, over a defined period —
   since that is the only remedy available for the single-site/single-period
   limitation that re-splitting this static dataset cannot address.
5. If the review board wants to formally adopt the fresh-split design as the new
   deployment candidate (model + per-unit thresholds as re-fit here), that should
   happen as an explicit, separate cycle producing its own `predict_cycle7.py` /
   `split_cycle7.csv` / `thresholds_cycle7.json` — deliberately not bundled into this
   response, so that the act of adopting it is a visible, separate decision rather
   than something that quietly happened while writing a methods critique response.

## Files produced (cycle 6)

- `report_cycle6.md` — this document
- `state_cycle6.md` — rewritten scientific state reflecting the withdrawn/standing
  distinction above
- `cycle6_fresh_split_validation.py` — the frozen-design, fresh-split re-validation
  script described above
- `cycle6_fresh_split.csv` — the new, previously-unused split (seed 2024,
  stratified on outcome × icu_type) — a diagnostic artifact for this response, not a
  new official deployment split (that would be a `split_cycle7.csv` decision, per
  recommendation 5)
- `cycle6_fresh_validation_metrics.txt` — full log of the fresh-split result
