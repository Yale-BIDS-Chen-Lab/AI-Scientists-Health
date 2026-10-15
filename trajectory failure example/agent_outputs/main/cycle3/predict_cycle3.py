#!/usr/bin/env python3
"""
Cycle-3 inference script.

Usage:
    python3 predict_cycle3.py <data_dir> <out_csv>

No new model was fit in cycle 3 -- only a bedside-alert operating threshold
was chosen (see thresholds_cycle3.json). This script therefore loads the
same fitted model as cycle 2 (model_cycle2.pkl, a fixed-weight probability
blend of a gradient-boosted-tree model and a logistic-regression model,
both already fitted -- nothing is refit here) and writes the same
patient_id,probability output as predict_cycle2.py.

To turn a probability into a bedside alert, compare it to the threshold in
thresholds_cycle3.json (alert if probability >= threshold); that comparison
is deliberately kept out of this script so the underlying risk score and the
operating threshold remain two independently reviewable/updatable artifacts.
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
        sys.exit("Usage: python3 predict_cycle3.py <data_dir> <out_csv>")

    data_dir, out_csv = sys.argv[1], sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    measurements = pd.read_csv(os.path.join(data_dir, "measurements.csv"))

    # Same fitted model as cycle 2 -- cycle 3 changed only the operating
    # threshold, stored separately in thresholds_cycle3.json.
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
