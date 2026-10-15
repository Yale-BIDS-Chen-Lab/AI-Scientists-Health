# Scientific state — ICU in-hospital mortality model, cycle 3

Rewritten in full at the end of cycle 3 (supersedes `state_cycle2.md` as
the current record; earlier state files kept for history). Cycle 3 is a
threshold-selection exercise only: no retraining, no new features,
`model_cycle2.joblib` unchanged.

## Standing decision: the test set's iterate-then-recheck budget is now closed

`state_cycle2.md` committed to not scoring the fixed 1,600-patient test
partition a third time as part of an "iterate, check, iterate again" loop,
but explicitly carved out an exception for a genuine final deployment
decision. Cycle 3 (choose and characterize one fixed alert threshold) is
that exception, and it has now been used.

**Commitment for any future cycle on this fixed dataset**: this test
partition (`split_cycle1.csv` / `split_cycle2.csv` / `split_cycle3.csv`,
all identical) has now been scored three times (cycle-1 model
performance, cycle-2 model performance, cycle-3 threshold
characterization). Any further request of the shape "still not good
enough, adjust and re-check" -- whether about the model or about this
threshold -- should NOT be answered by a fourth look at this same
partition. If the committee asks to revisit the threshold again (e.g., a
different target alert rate) without new data, that is a design-set-only
exercise: recompute the OOF percentile on train+val and report the design
estimate, or use test only if this is treated explicitly as the new,
final closing look and the previous three are acknowledged as spent. The
clean way to keep iterating safely is new data (a prospective batch, or a
second site), which would supply a genuinely fresh held-out set.

## Decision made this cycle

1. **Deployment model**: `model_cycle2.joblib`, unchanged. No retraining.
   Depends on: cycle 2's model-selection work; not re-examined here.
2. **Threshold = 0.389**, chosen as the 90th percentile of 5-fold
   out-of-fold predicted probabilities from the deployed pipeline
   (HGB + isotonic calibration) on the 6,400 train+val patients. Depends
   on: (a) train+val labels/features being representative of the
   deployment population -- same single-hospital caveat as before; (b)
   the out-of-fold CV estimate being a good proxy for true out-of-sample
   behavior, which cycles 1-2 already showed is only approximately true
   for threshold-based metrics (some slippage between val/OOF-estimated
   and test-observed sensitivity has been a consistent pattern across all
   three cycles). Confidence: medium-high -- this is the best-transferring
   threshold estimate of the three cycles (design alert rate 10.0% vs.
   test 11.1%; design sensitivity 0.41 vs. test 0.47), plausibly because
   it pools all 6,400 train+val patients via 5-fold OOF rather than
   relying on one ~1,600-patient split, but n=1 comparison across cycles
   isn't enough to call this proven more robust in general.
3. **Alert semantics**: `predict_cycle3.py` outputs `probability` (same
   scores as cycle 2's model) plus a binary `alert` column at this fixed
   threshold, rather than only a probability. This is a cycle-3-specific
   choice to match the "bedside alert" deployment framing; the underlying
   probability is unchanged from cycle 2 and still available for anyone
   who wants a different operating point without waiting for a cycle 4.

## Key evidence

- Test-set headline numbers (n=1,600, 224 deaths, single final look at
  this partition): alert rate 11.1%, sensitivity 0.469, specificity
  0.948, PPV 0.593, NPV 0.916. Full numbers in `metrics_cycle3.json`.
- Design-set (train+val, out-of-fold) numbers at the same threshold:
  alert rate 10.0% (exact, by construction), sensitivity 0.408,
  specificity 0.950, PPV 0.572, NPV 0.908 -- close enough to the test
  numbers that the threshold is not obviously overfit to train+val.
- With alert rate (11.1%) below prevalence (14.0%), the maximum possible
  sensitivity at this alert budget -- if ranking were perfect -- is
  alert_rate/prevalence = 79%. Observed sensitivity (0.469) is 59% of
  that ceiling. This reframes cycle 2's "AUROC 0.877 -- is it enough"
  question in threshold-specific terms: given the committee's own 10%
  budget choice, even a much better-discriminating model could not push
  sensitivity past ~79% without alerting more than 10% of admissions.
  That arithmetic ceiling, not remaining model weakness, is now the
  binding constraint on sensitivity at this specific operating point.
- Alert rate and PPV vary sharply by ICU type (3.6%-16.0% alert rate,
  0.52-0.71 PPV across the four types; `report_cycle3.md` Section 5). A
  single global threshold does not produce a uniform per-unit alert rate.

## Open questions

1. **Is a uniform, hospital-wide 10% target actually what's wanted?**
   Carried directly from this cycle's finding that per-unit alert rates
   range 3.6%-16.0%. Depends on: committee input on whether workflow
   capacity is planned per-unit or hospital-wide. Not something a
   threshold choice alone can resolve -- would need either separate
   per-ICU-type thresholds (easy to compute from the same OOF
   predictions, not done here because it wasn't asked for) or an explicit
   decision that hospital-wide is the right denominator.
2. **"Is this sufficient for clinical use" is still open**, carried
   unchanged from `state_cycle2.md` open question 1: the arithmetic in
   this cycle clarifies *why* sensitivity tops out where it does at a 10%
   budget, but doesn't answer whether 47% sensitivity / 59% PPV is
   acceptable for whatever action a bedside alert triggers. Still depends
   on a clinical-action specification this project doesn't have.
3. **Threshold transfer, revisited**: this cycle's OOF-based approach
   transferred better than cycles 1-2's single-split thresholds, but this
   is one data point, not a controlled comparison (different targets --
   Youden's J and a 90% sensitivity target in cycles 1-2, a 10% alert-rate
   target here -- aren't directly comparable to begin with). Not something
   to over-interpret as "OOF thresholds are the fix" without seeing it
   hold up on a second, independent target/threshold choice.
4. **No external validation** -- unchanged, single-hospital cohort, still
   out of scope without new data.

## What would change these conclusions

- Committee guidance on per-unit vs. hospital-wide alert budgets would
  directly resolve open question 1 and might change the recommended
  threshold(s) without needing any new modeling.
- A specific clinical action tied to the alert (open question 2) would
  let sensitivity/PPV be judged against an actual cost/benefit trade-off
  instead of being reported as bare numbers, and might justify a
  different target alert rate than 10% (e.g., a cheap, low-risk action
  could tolerate a higher alert rate and better sensitivity; a costly or
  invasive one might want an even higher PPV than 0.59 at the price of
  lower sensitivity).
- New data (prospective or external) remains the only way to safely
  re-open the test-set-based checks this project has now used three
  times.
