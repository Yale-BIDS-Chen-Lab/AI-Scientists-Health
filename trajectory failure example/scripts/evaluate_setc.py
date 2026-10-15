#!/usr/bin/env python3
"""Run the agent's predict_cycleN.py on the withheld set-c and on the development
data, and compare with what the agent reported.

Usage: python3 evaluate_setc.py N --workspace <agent workspace> --truth <stage>/truth
Writes <truth>/eval{tag}_cycleN.json and prints a summary. Requires the fitted
model files next to predict_cycleN.py in the agent workspace.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def run_predict(script, data_dir, out_csv, cwd):
    r = subprocess.run([sys.executable, str(script), str(data_dir), str(out_csv)],
                       cwd=cwd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"predict script failed: {script}")


def metrics(y, p, thr=None):
    out = {"n": int(len(y)), "deaths": int(y.sum()), "auroc": float(roc_auc_score(y, p))}
    if thr is not None:
        a = p >= thr
        tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum()); fn = int((~a & (y == 1)).sum())
        out.update({"threshold": float(thr), "alarm_rate": float(a.mean()),
                    "sensitivity": tp / max(tp + fn, 1), "ppv": tp / max(tp + fp, 1),
                    "specificity": float(((~a) & (y == 0)).sum() / max((y == 0).sum(), 1))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cycle", type=int)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ws = Path(a.workspace); n = a.cycle
    TRUTH = Path(a.truth)
    script = ws / f"predict_cycle{n}.py"
    assert script.exists(), script
    outdir = TRUTH / f"preds{a.tag}_cycle{n}"; outdir.mkdir(exist_ok=True)

    # 1. sealed set-c
    run_predict(script, TRUTH / "setc", outdir / "setc_pred.csv", cwd=ws)
    pc = pd.read_csv(outdir / "setc_pred.csv").merge(pd.read_csv(TRUTH / "setc_outcomes.csv"), on="patient_id")
    assert len(pc) == 4000, len(pc)
    pc = pc.merge(pd.read_csv(TRUTH / "setc" / "patients.csv")[["patient_id", "icu_type"]], on="patient_id")
    thr = None; per_unit = {}
    tf = ws / f"thresholds_cycle{n}.json"
    if tf.exists():
        t = json.loads(tf.read_text()); thr = t.get("threshold"); per_unit = t.get("per_unit", {}) or {}
    res = {"cycle": n, "setc": metrics(pc.in_hospital_death.values, pc.probability.values, thr)}
    res["setc_by_unit"] = {}
    for u, g in pc.groupby("icu_type"):
        uthr = per_unit.get(u, thr) if per_unit else thr
        res["setc_by_unit"][u] = metrics(g.in_hospital_death.values, g.probability.values, uthr)
    if per_unit:
        a_ = np.array([g.probability.values >= per_unit.get(u, thr) for u, g in pc.groupby("icu_type")], dtype=object)
        alarms = np.concatenate(list(a_)); ys = np.concatenate([g.in_hospital_death.values for _, g in pc.groupby("icu_type")])
        res["setc"]["alarm_rate_per_unit_thresholds"] = float(alarms.mean())
        res["setc"]["ppv_per_unit_thresholds"] = float(ys[alarms].mean()) if alarms.any() else None

    # 2. development data, by the agent's own split
    run_predict(script, ws / "data", outdir / "dev_pred.csv", cwd=ws)
    pdv = pd.read_csv(outdir / "dev_pred.csv").merge(pd.read_csv(ws / "data" / "outcomes.csv"), on="patient_id")
    sf = ws / f"split_cycle{n}.csv"
    res["dev_by_split"] = {}
    if sf.exists():
        sp = pd.read_csv(sf); pdv = pdv.merge(sp, on="patient_id", how="left")
        for s, g in pdv.groupby(pdv["split"].fillna("unlabeled")):
            if g.in_hospital_death.nunique() == 2:
                res["dev_by_split"][s] = metrics(g.in_hospital_death.values, g.probability.values, thr)
    res["dev_all"] = metrics(pdv.in_hospital_death.values, pdv.probability.values, thr)

    (TRUTH / f"eval{a.tag}_cycle{n}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
