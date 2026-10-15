# State — ICU in-hospital mortality model, cycle 3

Supersedes `state_cycle2.md`. Items unchanged are marked "carried forward"; new items
are marked "cycle 3".

## Modeling decisions locked in this cycle

1. **No new model was fit this cycle.** The deployed risk score is unchanged from
   cycle 2 (`model_cycle2.pkl`: 0.5/0.5 blend of HistGradientBoosting and logistic
   regression on the 338-feature v2 representation). `predict_cycle3.py` loads that
   same file; nothing is retrained. **Confidence: high** — this is by construction,
   not an empirical finding.

2. **Test set: reused verbatim again.** `split_cycle3.csv` is byte-identical to
   `split_cycle2.csv`/`split_cycle1.csv`. **Confidence: high**, and this must continue:
   any future cycle that wants to claim a specific alert rate or sensitivity/PPV
   number needs to be evaluated on this same held-out set to be comparable to what's
   reported here.

3. **Threshold-selection method**: rather than reading the ~10th-percentile cutoff
   directly off the deployed (train+val-fit) model's own val predictions, this cycle
   refit train-only copies of the two cycle-2 sub-models (same hyperparameters) to get
   val predictions that are genuinely out-of-sample for those copies, and took the
   90th percentile of *those* as the threshold. Depends on: the reasoning that a
   train+val-fit model's predictions on val are mildly in-sample and could bias a
   quantile cutoff. **Confidence: medium.** This is the methodologically more careful
   choice and is the one used to produce `thresholds_cycle3.json`, but it was not
   validated against the simpler alternative (reading the quantile off the deployed
   model's own val predictions) in a way that would require a second test-set look —
   so I cannot report which of the two would have landed closer to exactly 10% on
   test without spending a second test-set touch, which I chose not to spend. Both
   approaches are defensible; I picked the one with the cleaner methodological
   justification (avoiding in-sample quantile estimation) and committed to it before
   evaluating test.

4. **Threshold value: 0.7367.** Test alert rate at this threshold: 11.2% (target
   10%). Depends on: item 3 above, and on val/test both being ~1,200 patients (roughly
   12-patient, i.e. 1-percentage-point, granularity — the 1.2-point gap between target
   and realized test alert rate is within that noise band). **Confidence: high** that
   0.7367 is a reasonable, honestly-derived threshold; **confidence: medium** that a
   fresh 1,200-patient draw would reproduce exactly 10-11% rather than, say, 8-13%.

5. **Design choice: probability and threshold are kept as separate artifacts.**
   `predict_cycle3.py` outputs a continuous probability (same interface as cycles 1-2);
   `thresholds_cycle3.json` stores the cutoff separately, and no script bakes the
   alert decision into the model's output. Depends on: this being the preferred
   deployment shape for auditability/revisability, not a technical constraint.
   **Confidence: high** this is a reasonable choice; it was not specified explicitly
   by the committee, so flagging it here as an interpretation of their request.

## Key evidence (cycle 3)

- At threshold 0.7367 on test: sensitivity 0.435, specificity 0.941, PPV 0.545, NPV
  0.911, confusion matrix TN=971/FP=61/FN=95/TP=73. Source: `cycle3_metrics.txt`,
  `thresholds_cycle3.json`.
- The underlying model's ranking performance (test AUROC 0.857, AUPRC 0.512) and
  calibration issues (overconfident above ~p=0.5, carried forward from cycle 2,
  unresolved) are unchanged, since no model was refit.
- A 10%-alert-rate operating point trades a large drop in sensitivity (0.435 here vs.
  0.732 at cycle 2's Youden's-J threshold, which flagged ~24% of admissions) for a
  large gain in specificity/PPV (0.941/0.545 here vs. 0.830/0.413 at Youden's J). This
  is a mechanical consequence of the alert-rate constraint, not a new model property.

## Open questions (carried forward + new)

- **Cycle 2's unresolved items are all still open and untouched by this cycle**:
  calibration of the deployed blend, untuned blend weight, external/temporal
  validity, subgroup/fairness performance. None of these were in scope for a
  threshold-only cycle.
- **Is a 10% alert rate actually the right clinical target?** This cycle answered the
  literal question asked (what threshold gives ~10% alerts, and what are its
  operating characteristics) but did not evaluate whether 10% is well-matched to the
  clinical workflow the alert feeds into (e.g. rapid-response team capacity). Depends
  on: committee input on alert-fatigue tolerance and downstream capacity, which is
  outside this report's scope.
- **Sensitivity of 0.435 means the alert misses more than half of in-hospital
  deaths.** This is stated plainly in `report_cycle3.md` rather than left as an
  implicit caveat: if the committee's actual goal is to catch most deaths, a 10%
  alert rate is very likely too conservative, and this should be surfaced back to them
  as a decision point, not quietly absorbed.
- **The train-only-refit method for threshold calibration (item 3 above) adds
  complexity that a future cycle should reconsider** if this project moves toward
  routinely deploying models fit on train-only (rather than train+val) — in that case
  the deployed model's own val predictions would be genuinely out-of-sample and this
  extra refit step would be unnecessary. Depends on: whether cycle 4+ changes the
  train/val/test usage convention established in cycle 1.

## What would change these conclusions

- If the committee specifies a different target alert rate (e.g. 20% instead of 10%),
  the threshold, and all four reported operating-characteristic numbers, must be
  recomputed — they are specific to the 10% constraint, not general properties of the
  model.
- If cycle 2's calibration work (still open) changes the deployed model's probability
  outputs, this cycle's threshold (a raw score cutoff) would need to be redone against
  the recalibrated scores, since 0.7367 is a property of the current, uncalibrated
  score distribution.
