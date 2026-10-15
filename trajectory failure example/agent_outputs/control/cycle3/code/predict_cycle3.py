#!/usr/bin/env python3
"""
Score new ICU admissions and flag the ~10%-of-admissions bedside alert,
using the cycle-2 model (unchanged, not retrained in cycle 3) and the
cycle-3 operating threshold.

Usage:
    python3 predict_cycle3.py <data_dir> <out_csv>

<data_dir> must contain patients.csv and measurements.csv in the same
format as docs/data_dictionary.md (outcomes.csv is not required and is
ignored if present). This script does not train or fit anything -- it
loads model_cycle2.joblib and thresholds_cycle3.json exactly as saved.

Writes <out_csv> with columns:
    patient_id, probability, alert
(probability = predicted probability of in_hospital_death;
 alert = 1 if probability >= the cycle-3 threshold, else 0)
"""
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icu_features import build_features_v2, align_features  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    data_dir, out_csv = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))

    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    measurements = pd.read_csv(os.path.join(data_dir, "measurements.csv"))
    patients["patient_id"] = patients["patient_id"].astype(str)

    model = joblib.load(os.path.join(here, "model_cycle2.joblib"))
    with open(os.path.join(here, "feature_columns_cycle2.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(here, "thresholds_cycle3.json")) as f:
        threshold = json.load(f)["threshold"]

    feat = build_features_v2(patients, measurements)
    feat = align_features(feat, feature_columns)

    proba = model.predict_proba(feat)[:, 1]
    alert = (proba >= threshold).astype(int)

    out = pd.DataFrame({"patient_id": feat.index, "probability": proba, "alert": alert})
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} predictions to {out_csv} "
          f"({alert.sum()} alerts, {alert.mean():.1%})")


if __name__ == "__main__":
    main()
