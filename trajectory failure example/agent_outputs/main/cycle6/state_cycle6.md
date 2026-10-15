# State — ICU in-hospital mortality model, cycle 6

Supersedes `state_cycle5.md` in full. This cycle is a methodological correction, not
an extension — an external reviewer identified a real defect in how this project used
its test set across cycles 1-4, and this file rewrites the project's confidence
levels accordingly rather than simply appending to them. Where a cycle-5 item's
confidence has changed, that change is stated explicitly, not silently dropped.

## The defect, precisely

`split_cycle1.csv`'s test partition was read at the end of cycles 1, 2, 3, and 4.
Each read was internally disciplined (train/val used for all selection, test scored
once per cycle) but each read's result also shaped what happened next: cycle 1's
test-set permutation importance directed cycle 2's specific new features; cycle 1's
test-set AUROC drove the "insufficient" judgment that opened cycle 2; cycle 4's single
test pass computed and compared two competing threshold policies rather than only
confirming the one already chosen on val. This is decision-linked, repeated reuse of
one static held-out set — sufficient to invalidate the reported performance
trajectory as an unbiased estimate, regardless of each cycle's internal discipline.
**Confidence in this diagnosis: high** — the mechanism is traceable step by step
through the project's own reports (cited in `report_cycle6.md`), not a suspicion.

## Status of prior claims (this section replaces cycle 5's "decisions in effect")

**Withdrawn as validated performance claims** (may still be cited as historical
record of what was computed, never as evidence for a decision):
- All AUROC/AUPRC/Brier/sensitivity/specificity/PPV/NPV numbers computed on
  `split_cycle1.csv`'s test partition in cycles 1-5, overall or by ICU type.
- "Cycle 2's model is proven better than cycle 1's."
- "Cycle 4's per-unit thresholds are proven to outperform cycle 3's global
  threshold."
- The specific threshold values 0.7367 (global) and 0.377/0.705/0.718/0.810
  (per-unit) as final numbers — the *procedure* that produced them is not withdrawn
  (see below), only those particular fitted values, which came from a val partition
  belonging to the now-retired split.

**Still stands** (did not depend on adaptive test-set reuse):
- Data cleaning rules (`icu_features.PLAUSIBLE_RANGE`) — set from raw-value
  inspection before any split existed.
- Baseline mortality by ICU type (cardiac_surgery_recovery ~4.7% vs. 13.8-19.2%
  elsewhere) — a descriptive fact about outcomes, not a fitted quantity.
- Qualitative finding: GCS and age dominate the prediction — consistent across two
  feature sets and three model classes, and externally corroborated by established
  ICU severity-scoring literature this project did not produce. **Confidence:
  high** as a qualitative statement; **confidence: medium** on the exact
  quantitative ranking, which was measured on the compromised test set.
- The per-unit-threshold *design principle* (a single global cutoff mis-serves a
  low-base-rate unit) — follows from the baseline-mortality arithmetic alone plus a
  val-only AUROC-by-unit table, neither of which required adaptive test feedback.
- The model architecture choice (HGB + logistic regression blend on an engineered
  feature set) as a reasonable design on general grounds, independent of its
  measured test score.

## What this cycle added: one clean, non-adaptive re-validation

`cycle6_fresh_split_validation.py` froze the cycle-2 design (feature set, model
architecture, exact hyperparameters — no re-search) and the cycle-3/4 threshold
*procedure* (per-icu_type 90th percentile of out-of-sample validation scores), drew a
fresh split (seed 2024, stratified on outcome × icu_type, never previously used for
any decision in this project), refit from scratch on the new train partition, set
thresholds from the new val partition, and read the new test partition exactly once.

**New reference numbers** (`cycle6_fresh_validation_metrics.txt`): test AUROC
**0.876**, AUPRC **0.567**, Brier **0.122** — at or above the original (withdrawn)
numbers, which is evidence against severe overfitting to the original test
partition's idiosyncrasies, though a single fresh draw at n=1,200 cannot rule out
ordinary sampling variability as the explanation (a ~0.02 AUROC swing between
independent draws of this size is unremarkable). **Confidence: medium-high** that
this number is a methodologically valid single estimate; **confidence: low** that it
precisely represents "true" generalization performance, both because it is one draw
and because — like every number in this project — it comes from one hospital and an
unknown time period.

**Per-unit threshold pattern reproduced**: cardiac_surgery_recovery's fresh threshold
(0.454) again came out far below the other three units (0.766-0.799), the same
qualitative pattern as cycle 4, now on data no prior decision touched. This raises
confidence in the per-unit design principle specifically (not just asserted from
arithmetic, but now observed twice, independently). It also re-confirmed, more
forcefully than before, that this subgroup's exact operating characteristics are
unstable at this sample size: PPV for that unit was 0.19 in the original per-unit
check and 0.42 in this fresh one — a large swing from resampling alone, on ~250
patients and ~10-20 deaths. **Confidence: high** that per-unit thresholds are the
right design; **confidence: low** that any specific number reported for
cardiac_surgery_recovery to date (either draw) is precise.

## Open questions (revised)

1. **No further changes should be made in response to the cycle-6 fresh-split
   numbers.** They exist to demonstrate the remedy and to sanity-check the prior
   numbers, not to be optimized against. If the design changes further, a new split
   is needed for its final evaluation (see recommendation 3 in `report_cycle6.md`).
2. **`split_cycle1.csv`'s test partition and `cycle6_fresh_split.csv`'s test
   partition are both now retired** for any adaptive purpose. A genuine go/no-go
   deployment decision should use a third, not-yet-drawn split, evaluated exactly
   once, with no changes made afterward regardless of outcome.
3. **Single-site/single-period generalization remains completely open** — re-
   splitting the same 8,000-patient pool, however cleanly done, cannot substitute
   for external or prospective validation. This was true in cycle 5 and is unchanged
   by this cycle's work. Prospective validation (scoring new admissions as they
   occur, without further model changes) is the recommended path before deployment.
4. **Carried forward, unresolved, and unaffected by this cycle's findings**:
   calibration above ~p=0.5 was never corrected in any cycle; only ICU type has been
   examined as a subgroup (not sex/age); "equal alert rate per unit" vs. "equal
   sensitivity per unit" as the fairness target was never adjudicated by the
   committee; the 0.5/0.5 blend weight was never tuned; no threshold in this project
   has incorporated an explicit clinical cost-benefit input.
5. **If the review board wants to adopt the cycle-6 frozen design as the new
   deployment candidate**, that should be a deliberate, separate act (a `cycle 7`
   producing its own `predict_cycle7.py` / `split_cycle7.csv` /
   `thresholds_cycle7.json`), not something inferred from this response.

## What would change these conclusions

- If the review board is satisfied that within-cycle discipline is sufficient and
  disagrees that cross-cycle adaptive reuse invalidates the reported numbers, that
  would restore standing to the withdrawn claims — but this project's own account of
  events (Section "The defect, precisely" in `report_cycle6.md`) supports the
  reviewer's reading, and I am not aware of a basis to argue otherwise in good faith.
- A second and third fresh-split re-validation, each showing similarly strong
  performance and a similarly-shaped per-unit threshold pattern, would substantially
  raise confidence that the original numbers were not inflated by adaptive reuse
  (though it would still not address single-site generalization). One fresh draw is
  suggestive, not conclusive.
