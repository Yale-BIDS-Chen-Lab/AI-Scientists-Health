#!/usr/bin/env python3
"""
Cycle-1 inference script.

Usage:
    python3 predict_cycle1.py <data_dir> <out_csv>

<data_dir> must contain patients.csv and measurements.csv in the same format
described in docs/data_dictionary.md (see icu_features.py for the exact
columns expected). Loads the model fitted in cycle 1 (model_cycle1.pkl) --
no retraining happens here -- builds the same features used at training
time, and writes <out_csv> with columns: patient_id,probability

probability is the model's predicted probability of in_hospital_death.
"""

import os
import sys
import pickle

import pandas as pd

# Make sure icu_features.py (in the same directory as this script) is
# importable regardless of the caller's current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icu_features import build_feature_matrix, align_features  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 predict_cycle1.py <data_dir> <out_csv>")

    data_dir, out_csv = sys.argv[1], sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    measurements = pd.read_csv(os.path.join(data_dir, "measurements.csv"))

    with open(os.path.join(script_dir, "model_cycle1.pkl"), "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    feats = build_feature_matrix(patients, measurements)
    feats = align_features(feats, feature_columns)

    proba = model.predict_proba(feats)[:, 1]

    out = pd.DataFrame({
        "patient_id": feats.index,
        "probability": proba,
    })
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} predictions to {out_csv}")


if __name__ == "__main__":
    main()
