#!/usr/bin/env python3
"""
Cycle-2 inference script.

Usage:
    python3 predict_cycle2.py <data_dir> <out_csv>

<data_dir> must contain patients.csv and measurements.csv in the same format
described in docs/data_dictionary.md. Loads the model fitted in cycle 2
(model_cycle2.pkl) -- no retraining happens here -- builds the cycle-2
feature set (icu_features.build_feature_matrix_v2), and writes <out_csv>
with columns: patient_id,probability

probability is the model's predicted probability of in_hospital_death.

The cycle-2 model is a fixed-weight probability average of two fitted
sub-models (a gradient-boosted-tree model and a logistic-regression model);
both sub-models and their blend weights are loaded from model_cycle2.pkl, so
this script performs inference only, never any fitting.
"""

import os
import sys
import pickle

import numpy as np
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
        sys.exit("Usage: python3 predict_cycle2.py <data_dir> <out_csv>")

    data_dir, out_csv = sys.argv[1], sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    measurements = pd.read_csv(os.path.join(data_dir, "measurements.csv"))

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
