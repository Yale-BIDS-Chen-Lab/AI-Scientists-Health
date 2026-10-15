#!/usr/bin/env python3
"""Recompute every number in evaluation/results_original/ from the saved
per-patient predictions, and check that they match.

The fitted model files were not retained, so predict_cycleN.py cannot be re-run.
This script needs only the predictions in evaluation/predictions/, the agent's
split and threshold files in agent_outputs/, and the outcomes written by
prepare_workspace.py. Python standard library only.

Usage: python3 recompute_from_predictions.py --truth <stage>/truth
"""
import argparse, csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path, key, val):
    with open(path) as f:
        return {r[key]: r[val] for r in csv.DictReader(f)}


def auroc(y, p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(y); neg = len(y) - pos
    return (sum(r for r, t in zip(ranks, y) if t) - pos * (pos + 1) / 2) / (pos * neg)


def at_threshold(y, alarm):
    tp = sum(a and t for a, t in zip(alarm, y)); fp = sum(a and not t for a, t in zip(alarm, y))
    fn = sum(t and not a for a, t in zip(alarm, y)); tn = len(y) - tp - fp - fn
    return {"alarm_rate": sum(alarm) / len(y), "sensitivity": tp / max(tp + fn, 1),
            "ppv": tp / max(tp + fp, 1), "specificity": tn / max(tn + fp, 1)}


def evaluate(arm, n, truth):
    pred_dir = ROOT / "evaluation" / "predictions" / arm / f"cycle{n}"
    agent_dir = ROOT / "agent_outputs" / arm / f"cycle{n}"
    out = {}

    # withheld set-c (4,000 admissions)
    pred = {k: float(v) for k, v in read(pred_dir / "setc_pred.csv", "patient_id", "probability").items()}
    death = {k: int(v) for k, v in read(truth / "setc_outcomes.csv", "patient_id", "in_hospital_death").items()}
    unit = read(truth / "setc" / "patients.csv", "patient_id", "icu_type")
    ids = sorted(pred); y = [death[i] for i in ids]; p = [pred[i] for i in ids]
    out["setc_auroc"] = auroc(y, p)
    tf = agent_dir / f"thresholds_cycle{n}.json"
    if tf.exists():
        t = json.loads(tf.read_text())
        for k, v in at_threshold(y, [x >= t["threshold"] for x in p]).items():
            out[f"setc_{k}"] = v
        per_unit = t.get("per_unit") or {}
        if per_unit:
            alarm = [pred[i] >= per_unit.get(unit[i], t["threshold"]) for i in ids]
            r = at_threshold(y, alarm)
            out["setc_alarm_rate_per_unit_thresholds"] = r["alarm_rate"]
            out["setc_ppv_per_unit_thresholds"] = r["ppv"]

    # development data (8,000 admissions), split as the agent split it
    pred = {k: float(v) for k, v in read(pred_dir / "dev_pred.csv", "patient_id", "probability").items()}
    death = {k: int(v) for k, v in read(truth / "dev_outcomes_full.csv", "patient_id", "in_hospital_death").items()}
    split = read(agent_dir / f"split_cycle{n}.csv", "patient_id", "split")
    for s in sorted(set(split.values())):
        ids = [i for i in pred if split.get(i) == s]
        out[f"dev_{s}_auroc"] = auroc([death[i] for i in ids], [pred[i] for i in ids])
    return out


def original(arm, n):
    d = json.loads((ROOT / "evaluation" / "results_original" / f"eval_{arm}_cycle{n}.json").read_text())
    out = {f"setc_{k}": v for k, v in d["setc"].items() if k not in ("n", "deaths", "threshold")}
    out.update({f"dev_{s}_auroc": m["auroc"] for s, m in d["dev_by_split"].items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="<stage>/truth written by prepare_workspace.py")
    truth = Path(ap.parse_args().truth)
    ok = True
    print(f"{'arm':8} {'cycle':>5} {'reported test AUROC':>20} {'set-c AUROC':>12}  matches original")
    for arm in ("main", "control"):
        for n in range(1, 6):
            new, old = evaluate(arm, n, truth), original(arm, n)
            same = old.keys() == new.keys() and all(abs(new[k] - old[k]) < 1e-9 for k in old)
            ok &= same
            print(f"{arm:8} {n:>5} {new['dev_test_auroc']:>20.3f} {new['setc_auroc']:>12.3f}  {'yes' if same else 'NO'}")
    print("All values match evaluation/results_original/." if ok else "Some values differ.")


if __name__ == "__main__":
    main()
