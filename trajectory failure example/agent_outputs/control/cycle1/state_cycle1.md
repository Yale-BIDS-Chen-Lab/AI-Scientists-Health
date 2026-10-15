# Scientific state — ICU in-hospital mortality model, cycle 1

Purpose: record what is currently believed, how confidently, and on what it
depends, so cycle 2 can extend rather than re-derive this. Test-set numbers
here are the single, final look taken at the end of cycle 1 (see
`split_cycle1.csv` for the partition, fixed before any modeling).

## Modeling decisions (committed, cycle 1)

1. **Split**: 60/20/20 train/val/test, stratified on outcome, fixed seed 42,
   written to `split_cycle1.csv` before any model was touched. Depends on:
   nothing upstream; this is the root decision. Future cycles should keep
   this exact split (reuse `split_cycle1.csv`) rather than re-splitting, so
   the test set stays untouched and comparable across cycles.
2. **Feature set**: per-patient count/mean/min/max/last of all 37
   measurement variables over the full 48h + demographics + ICU type +
   `Urine_sum`. Depends on: the data dictionary's variable list; no
   assumption about which variables matter (left to the model/regularization).
3. **Data cleaning rules**: fixed numeric bounds nulling clear sentinel/
   entry-error values (Weight=-1, height outside [100,230] cm, weight
   >400 kg, Temp outside [25,45] C, pH outside [6.5,8.0], eight
   hemodynamic/respiratory variables <=0). Depends on: manual inspection of
   raw value distributions (documented in this cycle's exploration, not
   re-derived from a formal outlier test). Confidence: high for the
   sentinel/impossible cases (Weight=-1, pH>14, height <20cm or >300cm);
   medium for the exact boundary values chosen (e.g. why 230 cm and not
   220 or 250) — these were picked to be clearly conservative, not tuned.
4. **Algorithm**: `HistGradientBoostingClassifier` (chosen over logistic
   regression on validation AUROC, ~0.865 vs ~0.857). Depends on: the
   feature set above; native NaN handling was the deciding qualitative
   reason (labs are missing informatively — ordered more for sicker
   patients — which imputation would blur). Confidence: high that HGB
   beats plain median-imputed LR on this feature set; not yet tested
   against LR with missingness *indicators* added explicitly, which could
   close some of the gap (open question below).
5. **Calibration**: isotonic, via 5-fold `CalibratedClassifierCV`, fit on
   train+val. Depends on: decision 4 (which base estimator to calibrate).
   Confidence: medium-high — the resulting test-set calibration curve
   tracks closely (see `calibration_cycle1.csv`), but isotonic calibration
   with only ~1,600 val-equivalent events split across folds is a modest
   sample for the extreme deciles.
6. **Decision thresholds**: 0.385 (Youden's J on val) as default, 0.250
   (val-tuned for >=90% sensitivity) as a high-sensitivity alternative.
   Depends on: decision 5's calibrated probabilities. Confidence: **low**
   on exact transferability — see open question 1 below, this is the
   weakest link in cycle 1.
7. **Model shipped for inference**: the calibrated HGB refit on train+val,
   serialized to `model_cycle1.joblib`, with the exact training feature
   column order pinned in `feature_columns_cycle1.json` and reproduced by
   `icu_features.align_features` so a later cycle's differently-shaped data
   can't silently misalign columns.

## Key evidence

- Test AUROC 0.868, AUPRC 0.532 (vs. 0.140 base rate), Brier 0.088 — all
  from the single end-of-cycle test evaluation, n=1,600, 224 events.
- Validation AUROC (0.865) and test AUROC (0.868) are essentially
  identical — no evidence of overfitting in model selection, gives
  confidence the AUROC number itself will hold up.
- Top permutation-importance features (GCS, age, lactate, urine output,
  glucose, BUN, ICU type, bilirubin, HCO3, HR) all have directions matching
  standard ICU clinical knowledge, and a logistic-regression coefficient
  check agrees in sign on all ten. This cross-check is the main basis for
  trusting the "what drives predictions" section, not just the permutation
  importance ranking on its own.
- Calibration curve (test, 10 bins) tracks observed rate closely at every
  decile (`calibration_cycle1.csv`) — supports using raw probabilities, not
  only rank, downstream.

## Open questions (not blocking, carried to cycle 2)

1. **Threshold transfer**: the val-tuned "90% sensitivity" threshold
   (0.250) achieved only 65% sensitivity on test; the Youden threshold
   (0.385) achieved 47% on val-implied ~82% but 47% on test. Both moved by
   more than sampling noise alone would suggest is comfortable at n=1,600/
   224 events, though it's within plausible binomial variation for this n.
   Depends on: a single val/test split with modest event counts. Next
   step: estimate thresholds via repeated stratified CV on train+val rather
   than one val split, before trusting an operating point for deployment.
2. **Subgroup discrimination**: AUROC is 0.92/0.88 for
   cardiac-surgery-recovery/surgical but only 0.82-0.83 for medical/
   coronary-care, which together hold 143 of 224 test deaths. Depends on:
   test_subgroup_auroc_by_icu_type in `metrics_cycle1.json` (n=231-576 per
   group, so individual subgroup AUROCs are themselves noisy). Not yet
   investigated *why* medical/coronary-care patients are harder to rank —
   candidate hypothesis for cycle 2: their deaths may be driven more by
   slower, multi-system trajectories that whole-window min/max/last stats
   don't capture as well as they capture single acute events (e.g. a
   cardiac-surgery bleed shows up clearly in one lab trend).
3. **Feature set is trajectory-blind**: count/mean/min/max/last per
   variable ignores *when* the worst value occurred and how fast a
   variable is trending across the 48h. This is a deliberate cycle-1
   simplification (keep the first model interpretable and fast to
   validate), not an oversight — but it is the most likely lever for
   raising AUROC above 0.868 in cycle 2.
4. **LR-with-missingness-indicators not tried**: decision 4 compared HGB
   against plain median-imputed LR; LR augmented with explicit "was this
   variable measured" binary flags per variable was not tested and might
   narrow the gap while keeping a fully linear, more auditable model. Low
   priority unless interpretability requirements tighten.
5. **No external validation**: this is a single-hospital cohort; nothing
   here speaks to how the model would perform at another site or under
   case-mix shift. Out of scope for cycle 1 (no other data available in
   this project), flagged for whoever considers deployment.

## What would change these conclusions

- A different random split (different seed) is the most direct sanity
  check on open question 1 (threshold transfer) and on how tight the 0.868
  AUROC estimate really is — cycle 2 could refit under 3-5 seeds and report
  a range instead of a point estimate, without touching the held-out test
  set from this cycle more than once.
- If cycle 2 adds trajectory features (open question 3) and AUROC rises
  materially, that would confirm the current feature set (not the
  algorithm) is the binding constraint, redirecting further effort there
  rather than into further algorithm tuning.
