#!/usr/bin/env python3
"""
Score new ICU admissions with the cycle-1 in-hospital mortality model.

Usage:
    python3 predict_cycle1.py <data_dir> <out_csv>

<data_dir> must contain patients.csv and measurements.csv in the same
format as docs/data_dictionary.md (outcomes.csv is not required and is
ignored if present -- this script never trains or refits anything, it
only loads model_cycle1.joblib fit during cycle 1 and applies it).

Writes <out_csv> with columns: patient_id,probability
(probability = predicted probability of in_hospital_death).
"""
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icu_features import build_features, align_features  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    data_dir, out_csv = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))

    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    measurements = pd.read_csv(os.path.join(data_dir, "measurements.csv"))
    patients["patient_id"] = patients["patient_id"].astype(str)

    model = joblib.load(os.path.join(here, "model_cycle1.joblib"))
    with open(os.path.join(here, "feature_columns_cycle1.json")) as f:
        feature_columns = json.load(f)

    feat = build_features(patients, measurements)
    feat = align_features(feat, feature_columns)

    proba = model.predict_proba(feat)[:, 1]

    out = pd.DataFrame({"patient_id": feat.index, "probability": proba})
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} predictions to {out_csv}")


if __name__ == "__main__":
    main()
