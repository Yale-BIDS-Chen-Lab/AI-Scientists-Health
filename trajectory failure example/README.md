# Mortality prediction example

This folder contains the execution records for the mortality prediction example in the Supplementary Information. An AI Scientist agent developed a model that predicts in-hospital death from the first 48 hours of ICU data. It worked through six cycles of requests. A five-cycle control run used the same data and prompts, plus one added instruction in cycle 1.

## Setup

- **Agent:** Claude Sonnet 5 (`claude-sonnet-5`), run through the Agent tool in Claude Code 2.1.270 with effort set to `high`. Within each run, the same agent was continued from cycle to cycle. The agent wrote and ran all of the code in `agent_outputs/` itself.
- **Prompts:** the prompts were fixed in advance and are given in full in [`prompts.md`](prompts.md). We did not edit the code, results or reports the agent produced.
- **Data seen by the agent:** 8,000 admissions (PhysioNet/CinC Challenge 2012, sets A and B). Every admission received a new random identifier (`P00001`–`P12000`), so the agent could not tell which source set a record came from.
- **Withheld data:** another 4,000 admissions (set C), which the agent never saw and was never told about. After each cycle, we ran that cycle's model on these admissions. File names call them `setc`.

## Data

This folder contains no PhysioNet data. To rebuild the data, download these files from the [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) page (Open Data Commons Attribution License v1.0):

```
set-a.tar.gz  set-b.tar.gz  set-c.tar.gz  Outcomes-a.txt  Outcomes-b.txt  Outcomes-c.txt
```

Put them in one directory, then run:

```
python3 scripts/prepare_workspace.py --source <download dir> --workspace <ws> --stage <stage>
```

This writes the agent's working data to `<ws>/data/` and `<ws>/docs/`. The withheld set C and the outcomes go to `<stage>/truth/`. The identifier assignment is fixed (seed 2012), so the `patient_id` values match those in every file here.

## Contents

| Path | Contents |
|---|---|
| `prompts.md` | The full prompt for each cycle, for both runs |
| `agent_outputs/main/cycle1–6/` | Main run: everything the agent produced in each cycle |
| `agent_outputs/control/cycle1–5/` | Control run: the same, per cycle |
| `evaluation/predictions/<run>/cycleN/` | Per-admission predicted probabilities from that cycle's model: `setc_pred.csv` (4,000 withheld admissions) and `dev_pred.csv` (the 8,000 admissions the agent used) |
| `evaluation/results_original/` | Metrics computed from those predictions after each cycle |
| `scripts/prepare_workspace.py` | Builds the agent's data and the withheld set from the PhysioNet files |
| `scripts/evaluate_setc.py` | The script we ran after each cycle to produce the predictions and metrics |
| `scripts/recompute_from_predictions.py` | Recomputes all metrics from the saved predictions and checks them against `results_original/` |

### Inside each cycle folder

- `report_cycleN.md`: the agent's research report for that cycle.
- `state_cycleN.md`: the agent's scientific state (decisions, assumptions with confidence, evidence, open questions).
- `split_cycleN.csv`: how the agent used each admission (`train` / `validation` / `test`, or its own labels).
- `predict_cycleN.py`, `thresholds_cycleN.json`: the delivered prediction script and alert thresholds.
- `code/`: the agent's training code, feature code and run logs (`*_metrics.txt`, `metrics_*.json`), plus intermediate tables.

Some cycles have no `code/` folder because the agent did not write new code in them.

In the agent's working directory, all of these files sat in a single directory. We moved the supporting files into `code/` when archiving, so some files appear in both places.

### Execution logs

In this folder, "execution logs" means the per-cycle records the agent wrote: the run logs in `code/`, the reports and the state files. Full conversation transcripts are not included.

## Checking the numbers

The fitted model files were not retained after the runs, so `predict_cycleN.py` and `evaluate_setc.py` cannot be re-executed from this folder. The predictions those scripts produced at the time are in `evaluation/predictions/`. Every metric can be recomputed from them:

```
python3 scripts/recompute_from_predictions.py --truth <stage>/truth
```

The script uses only the Python standard library. For each run and cycle it prints the AUROC on the agent's own test split, the AUROC on the 4,000 withheld admissions, and whether every value matches `results_original/`.

Fields in `results_original/eval_<run>_cycleN.json`:

- `setc`: the 4,000 withheld admissions. Includes AUROC and, from cycle 3 onwards, alert rate, sensitivity, specificity and PPV at the agent's threshold. Where per-unit thresholds were used, the fields ending in `_per_unit_thresholds` apply them.
- `setc_by_unit`: the same metrics for each ICU type.
- `dev_by_split`: the 8,000 development admissions, grouped by the agent's own `split_cycleN.csv`. The `test` entry reproduces the test-set AUROC the agent reported.
- `dev_all`: all 8,000 development admissions together.

Cycle 6 of the main run has no evaluation entry. In that cycle the agent re-split the development data and re-evaluated a frozen design (`agent_outputs/main/cycle6/code/`) but delivered no new prediction script.

## Notes

- Each run is a single execution. Re-running the prompts will not reproduce the same trajectory.
- The agent's code uses Python 3 with numpy, pandas and scikit-learn. Package versions were not recorded.

## Citation for the data

Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality of ICU patients: the PhysioNet/Computing in Cardiology Challenge 2012. *Computing in Cardiology* 2012;39:245–248.
