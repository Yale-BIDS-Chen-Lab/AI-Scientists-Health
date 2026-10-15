#!/usr/bin/env python3
"""
Cycle-5 inference script (deployment-review-board package).

Usage:
    python3 predict_cycle5.py <data_dir> <out_csv>

No new model was fit in cycle 5 -- this cycle only consolidates the
documentation for the deployment review board. The deployed risk model is
unchanged since cycle 2 (a fixed-weight probability blend of a
gradient-boosted-tree model and a logistic-regression model, both already
fitted -- nothing is refit here); the alert-threshold policy is unchanged
since cycle 4 (per-icu_type thresholds). This script loads model_cycle2.pkl
and writes the same patient_id,probability output as predict_cycle2.py /
predict_cycle3.py / predict_cycle4.py.

To turn a probability into a bedside alert under the current policy, look up
the patient's icu_type in thresholds_cycle5.json's "per_unit" map and alert
if probability >= that unit's threshold (fall back to the top-level
"threshold" only if icu_type is missing/unrecognized). That comparison is
kept out of this script so the risk score and the alert policy remain
independently reviewable/updatable artifacts.
"""

import os
import sys
import pickle

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icu_features import build_feature_matrix_v2, align_features  # noqa: E402


def predict_proba_from_bundle(model, feats):
    if isinstance(model, dict) and model.get("kind") == "blend":
        w_hgb, w_lr = model["weights"]
        p = (w_hgb * model["hgb"].predict_proba(feats)[:, 1]
             + w_lr * model["lr"].predict_proba(feats)[:, 1])
        return p
    return model.predict_proba(feats)[:, 1]


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 predict_cycle5.py <data_dir> <out_csv>")

    data_dir, out_csv = sys.argv[1], sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    measurements = pd.read_csv(os.path.join(data_dir, "measurements.csv"))

    # Same fitted model as cycles 2-4 -- cycle 5 changed no model and no
    # threshold; it only compiles the deployment-review documentation.
    with open(os.path.join(script_dir, "model_cycle2.pkl"), "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    feats = build_feature_matrix_v2(patients, measurements)
    feats = align_features(feats, feature_columns)

    proba = predict_proba_from_bundle(model, feats)

    out = pd.DataFrame({
        "patient_id": feats.index,
        "probability": proba,
    })
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} predictions to {out_csv}")


if __name__ == "__main__":
    main()
