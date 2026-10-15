# Scientific state — ICU in-hospital mortality model, cycle 2

Rewritten in full at the end of cycle 2 (supersedes `state_cycle1.md` as
the current record; that file is kept for history). Purpose unchanged:
record what is believed, how confidently, on what it depends, and what
would change it.

## Standing decision: the test set is now spent

`split_cycle1.csv` / `split_cycle2.csv` (identical content) has been
scored exactly twice: once for the cycle-1 report, once for the cycle-2
report above. Both times, every upstream decision (cleaning rules,
features, algorithm, hyperparameters, threshold) was chosen using only
train+val, so neither look was itself tuned to the test set. But the
*process* generating cycle 3 — committee sees a test number, judges it
insufficient, asks for another round of improvement — is precisely the
loop that turns a held-out test set into a de facto validation set over
enough repetitions, even when each individual cycle is disciplined. This
is a known failure mode in iterative model development, not a hypothetical.

**Commitment for any future cycle on this fixed dataset**: do not score
this same 1,600-patient test partition a third time as part of an
iterate-then-check loop. Options if more modeling work is requested:
1. Use 5-fold (or repeated) CV on train+val exclusively for all
   comparisons and hold off on any test look until a genuine final
   decision point (e.g., a go/no-go for deployment, not another "try to
   beat last cycle's number").
2. If a numeric checkpoint is truly needed sooner, use a fresh
   random re-split of train+val internally (not touching the 1,600 test
   patients) to manufacture an additional internal check, and say so
   explicitly rather than reusing test.
3. The clean long-term fix is new data: a prospective or external cohort
   would give a real, unspent test set and would also answer the
   generalization question this single-hospital cohort cannot (see open
   question 5 below).

Depends on: this being a fixed, one-time dataset with no new incoming
data between cycles (true as of cycle 2; revisit if that changes).

## Modeling decisions (cycle 2, additions/changes from cycle 1)

1. **Split reused unchanged from cycle 1** (`split_cycle1.csv`). Not
   re-derived. Depends on: the leakage argument above (Section 2 of
   `report_cycle2.md`). Confidence: high that this was the right call
   given a single fixed dataset with no new data between cycles.
2. **Feature set expanded to 347 columns** (`icu_features.build_features_v2`):
   added per-variable first value, OLS trend slope, early/late (0-24h /
   24-48h) means, plus `gcs_delta`, shock index, BUN/creatinine ratio, and
   a PaO2/FiO2 proxy. Depends on: cycle-1 state file's open question 3
   ("feature set is trajectory-blind"). Confidence: **the ablation
   (Section 4 of the report) shows this barely moved CV AUROC (+0.0005,
   within noise)** — keep low confidence that this feature set is a
   meaningful lever on its own; `gcs_delta` in particular is a genuine,
   clinically interpretable qualitative addition (Section 6 of the
   report) even though its aggregate effect on the ranking metric is
   small. The PaO2/FiO2 proxy uses separately-aggregated min(PaO2) and
   max(FiO2), not matched observation-by-observation in time — a known
   approximation, not fixed this cycle.
3. **Model selection via 5-fold CV on train+val**, replacing cycle 1's
   single train-fit/val-score comparison. Depends on: cycle-1 open
   question 1 (single-split threshold/estimate variance). Confidence:
   high that this is more robust than cycle 1's method; the CV std
   (~0.008-0.012 AUROC across folds) is itself useful context for how much
   any single-number comparison should be trusted going forward.
4. **Model class broadened**: random forest, extra trees, and classic
   gradient boosting were compared against HGB and logistic regression.
   **None beat HGB.** Depends on: decision 2's feature set (comparison
   run on cycle-2 features only, not re-run on cycle-1 features for every
   algorithm — a gap, see open questions). Confidence: high that, on this
   feature representation, gradient boosting is the right family; the
   gap to random forest/extra trees (~0.005-0.01 CV AUROC) is consistent
   enough across configurations to trust the ranking, not just the top pick.
5. **Selected model**: HGB, 15 leaves, learning rate 0.03, L2=2.0 (vs.
   cycle 1's lr=0.05, L2=1.0) — a modest re-tune, not a new algorithm.
   This, not the new features or new model classes, is the main source of
   the ~0.01 test AUROC gain. Confidence: medium-high on the direction of
   the effect (more regularization/slower learning helped), lower on
   whether this exact hyperparameter pair is a true optimum vs. one
   reasonable point in a fairly flat region (CV AUROC differences between
   the top 3 HGB configs were 0.001-0.002, inside one CV std).
6. **Calibration**: same approach as cycle 1 (isotonic, 5-fold internal
   CV, fit on train+val). Unchanged; still tracks well on test
   (`calibration_cycle2.csv`).
7. **Thresholds now derived from out-of-fold CV predictions on train+val**
   (Youden 0.356, high-sensitivity/>=90% target 0.267), replacing cycle
   1's single-val-split thresholds. Depends on: decision 3. Confidence:
   medium — this directly targets cycle-1 open question 1, but test-set
   sensitivity at the ">=90%" threshold still only reached 64%
   (vs. cycle 1's similar shortfall), so out-of-fold CV narrowed but did
   not eliminate the gap between a validation-style estimate and true
   test behavior for threshold-based (as opposed to ranking) metrics.
   This gap should be treated as a property of ~1,600-patient test sets
   with ~14% prevalence (224 events), not a bug in the method.

## Key evidence

- Paired bootstrap on the identical 1,600 test patients: cycle 2 minus
  cycle 1 AUROC = +0.0095 (95% CI 0.0019 to 0.0173, positive in 99% of
  2,000 resamples). The improvement is real, small, and precisely
  quantified because the comparison is paired (same patients, two
  models) rather than two independent numbers.
- Ablation isolating features from hyperparameters (Section 4 of the
  report): feature engineering contributed ~0.0005 CV AUROC (noise);
  hyperparameter re-tuning within the same HGB family contributed the
  rest (~0.002 CV AUROC on train+val, ~0.0095 on the paired test
  comparison — these two numbers are not expected to match exactly since
  one is CV-on-trainval and the other is a single test look, but both
  point the same direction and same rough magnitude).
- Subgroup AUROC improved or held steady in all four ICU types; the
  medical/coronary-care weak spot identified in cycle 1 persists at
  roughly the same magnitude (0.83-0.84 AUROC vs. 0.89-0.92 for
  surgical/cardiac-surgery-recovery).
- `gcs_delta` direction check: -0.50 (died) vs. +2.66 (survived) points
  from first to last GCS reading — a clean, clinically sensible new
  signal, described in Section 6 of the report, even though it ranks only
  9th in aggregate permutation importance.

## Open questions

1. **Is 0.877 AUROC actually sufficient for the intended clinical action?**
   Not answerable from this project's data. Depends on: the committee
   specifying what decision the model output would drive (workflow
   trigger vs. autonomous action) and the acceptable false-positive/
   false-negative trade-off for that specific decision; also depends on a
   benchmark this dataset cannot provide (e.g., a bedside score like
   APACHE II/SAPS II computed on the same patients). This is the central
   open question carried out of cycle 2 — see `report_cycle2.md` Section 7.
2. **Test-set reuse budget is exhausted for iterate-and-recheck cycles**
   (see "Standing decision" above). Any cycle 3 needs a different
   evaluation strategy, not just "try harder and re-check."
3. **Model-class comparison was only run on cycle-2 features.** It's
   possible random forest/extra trees would rank differently on cycle-1's
   smaller feature set, or that some other combination (e.g., extra trees
   on cycle-1 features) outperforms what was tried. Low priority given
   how consistently HGB led across every configuration actually tested,
   but not formally ruled out.
4. **Medical/coronary-care discrimination gap unresolved.** Carried over
   unchanged from cycle 1; the new trajectory features did not close it.
   Candidate hypothesis (untested): these patients' deaths may follow
   slower, multi-system deterioration that a 48-hour window — even with
   trend features — still under-describes relative to acute surgical
   complications. Would need either a longer observation window (not
   available: the data dictionary fixes this at 48h) or additional
   variable types.
5. **No external validation** — still a single-hospital cohort, unchanged
   from cycle 1, still out of scope without new data.
6. **PaO2/FiO2 proxy is not time-matched** (uses window min(PaO2) and
   max(FiO2) independently). Minor relative to the other open questions,
   but if respiratory-failure discrimination becomes a focus, this should
   be redone by pairing each PaO2 reading with the most recent prior FiO2
   reading rather than aggregating each separately.

## What would change these conclusions

- A genuinely new data source (more patients, another hospital, richer
  variable set, or longer/different observation windows) is the most
  direct way to test open question 1's premise that this approach is
  near a ceiling — if AUROC jumps materially on richer data, that
  confirms data/features (not this algorithm family) were the binding
  constraint all along, consistent with what the cycle-2 ablation already
  suggests.
- A concrete clinical-use specification (target action + acceptable
  error trade-off) would let threshold selection and any future
  "sufficient or not" judgment be made on decision-relevant terms instead
  of a generic discrimination statistic, and might reveal the current
  model is already adequate for a narrower, well-chosen use case (e.g.,
  triage flagging) even without further AUROC gains.
