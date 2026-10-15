# State — ICU in-hospital mortality model, cycle 4

Supersedes `state_cycle3.md`. Items unchanged are marked "carried forward"; new items
are marked "cycle 4".

## Modeling decisions locked in this cycle

1. **No new model was fit.** `predict_cycle4.py` loads the same `model_cycle2.pkl` as
   cycles 2-3; only the alert-threshold policy changed. **Confidence: high** — by
   construction.

2. **Test set: reused verbatim again.** `split_cycle4.csv` is byte-identical to
   `split_cycle3.csv`/`split_cycle2.csv`/`split_cycle1.csv`. **Confidence: high**, and
   must continue in any future cycle for comparability.

3. **Diagnosis: the cardiac-surgery-recovery complaint is a threshold-policy problem,
   not a model-quality problem.** Evidence: AUROC for that unit (0.884, out-of-sample
   val) is the *highest* of the four units, so the model's within-unit ranking is not
   deficient there. The unit's baseline mortality (4.7%, train+val) is 3-4x lower than
   every other unit (13.8-19.2%), so its predicted probabilities sit systematically
   lower on the same absolute scale, and cycle 3's single global threshold (90th
   percentile of the whole population's scores) fired for only ~1.2% of this unit's
   admissions instead of the intended 10%. **Confidence: high** in this mechanism —
   it is a direct, arithmetic consequence of a shared threshold plus a large base-rate
   gap, not a subtle or contested inference, and is corroborated by three independent
   signals (base-rate table, AUROC-by-unit table, alert-rate-by-unit table) rather
   than one.

4. **Fix chosen: per-icu_type thresholds**, each the 90th percentile of that unit's
   own out-of-sample val score distribution (train-only refit of the cycle-2
   sub-models — same methodology as cycle 3, applied per subgroup instead of
   population-wide). Saved in `thresholds_cycle4.json` under `per_unit`, with the
   cycle-3 global value retained under `threshold` as a fallback for an
   unrecognized/missing icu_type. **Confidence: medium-high** that this is the right
   category of fix (it directly removes the diagnosed mechanism and was anticipated
   as an acceptable approach in the coordinator's own request). **Confidence: lower**
   that the specific per-unit numbers, especially cardiac_surgery_recovery's 0.377,
   are precisely right — see the test-check caveat below, which is the main open risk
   carried out of this cycle.

5. **Test check (single pass) showed a real trade-off, not an unambiguous win, and I
   deployed the fix anyway.** On test: cardiac_surgery_recovery sensitivity improved
   (0.50→0.60) and its alert rate moved toward the 10% target (6.0%→13.8%), but its
   PPV dropped sharply (0.357→0.188), and overall population sensitivity/PPV both
   moved slightly down (0.435→0.387 / 0.545→0.500, driven mainly by medical's
   previously-oversized 16.9% alert share being corrected to 10%). **Confidence:
   medium** that deploying anyway was the right call — the reasoning (in
   `report_cycle4.md`) is that the diagnosed structural mechanism will recur
   regardless of this one test draw's sampling luck, and a near-silent alert for an
   entire unit is a worse failure mode than a noisier-but-functioning one. This is a
   judgment call, not a metric-forced conclusion, and is flagged as such.

## Key evidence (cycle 4)

- Baseline mortality by icu_type (train+val, n=8,000 total minus test): cardiac
  surgery recovery 4.7% vs. 13.8-19.2% elsewhere. Source: `cycle4_metrics.txt`.
- AUROC by icu_type (val, out-of-sample): cardiac_surgery_recovery 0.884 (best),
  surgical 0.886, medical 0.852, coronary_care 0.848 — no unit is poorly discriminated.
- Alert rate at the cycle-3 global threshold, by unit (val, out-of-sample): cardiac
  surgery recovery 1.2%, coronary_care 7.5%, surgical 9.2%, medical 16.9% — a roughly
  14x spread that a single global cutoff cannot avoid given the base-rate gap.
- Per-unit thresholds bring each unit's val alert rate to 10.0-10.2%, by construction.
- Test-set check (n=1,200, single pass, both policies): full numbers in
  `thresholds_cycle4.json` (`test_by_unit_old_global_threshold_policy` and
  `test_by_unit_new_per_unit_policy` keys) and `cycle4_metrics.txt`.
- **Subgroup sample sizes are small**: cardiac_surgery_recovery has only 18 deaths in
  val and 10 deaths in test. Every subgroup metric reported for this unit (sensitivity,
  PPV, alert rate) should be read with that in mind — a difference of 1-2 patients
  moves these numbers by several percentage points.

## Open questions (carried forward + new)

- **Top priority, new this cycle**: the per-unit threshold for cardiac_surgery_recovery
  (0.377) was set from a small subgroup (val n=260, 18 events) and the test check
  showed a real PPV cost there (0.188). A more robust estimate — e.g. shrinking the
  unit-specific quantile toward the global quantile in proportion to subgroup size
  (partial pooling / empirical-Bayes-style), rather than using the raw within-unit
  90th percentile — is a well-scoped, not-yet-attempted improvement. Depends on:
  doing this via val only (train-only refit, same as this cycle) and reserving a
  single new test look for cycle 5, rather than revisiting this cycle's test read.
- **Carried forward, still unresolved from cycle 2**: calibration of the deployed
  blend, untuned blend weight, external/temporal validity, and the decision threshold
  still has no clinical cost-benefit input (now made more concrete by this cycle: the
  cost-benefit trade-off is now visibly different by unit too, not just in aggregate).
- **Fairness framing**: this cycle treated "equal ~10% alert rate per unit" as the
  fairness target because it directly matches the coordinator's original ask (10%
  alert rate) applied per subgroup. Whether equal alert *rate* is the right equity
  criterion (versus, say, equal sensitivity per unit, which would require different
  per-unit thresholds and produce different, likely higher, per-unit alert rates for
  low-prevalence units) was not adjudicated here and is a policy question for the
  committee, not a modeling one.
- **Only icu_type was examined as a subgroup.** Sex, age band, and other potential
  subgroups (carried forward from cycle 2's open questions) remain unexamined.

## What would change these conclusions

- If a future cycle obtains more cardiac-surgery-recovery admissions (more data), the
  per-unit threshold estimate for that unit should be redone — 0.377 is based on a
  quantile of ~260 out-of-sample scores with only 18 events, among the least reliable
  numbers in this whole project.
- If the committee decides equal sensitivity, not equal alert rate, is the correct
  cross-unit equity target, every per-unit threshold in `thresholds_cycle4.json` would
  need to be recomputed against that different objective.
