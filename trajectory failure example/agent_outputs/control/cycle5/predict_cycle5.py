#!/usr/bin/env python3
"""
Final deployment scoring script for the in-hospital mortality model
(deployment review board version). Identical pipeline to predict_cycle4.py
-- cycle 5 changed no model, feature, or threshold logic; it only ran the
final confirmatory validation reported in report_cycle5.md and re-saved
the frozen artifacts under cycle-5 names for the review record.

Usage:
    python3 predict_cycle5.py <data_dir> <out_csv>

<data_dir> must contain patients.csv and measurements.csv in the same
format as docs/data_dictionary.md (outcomes.csv is not required and is
ignored if present). Nothing is trained or fit here -- this script loads
model_cycle2.joblib (the gradient-boosting classifier fit in cycle 2),
recalibration_cycle4.json (the per-icu_type calibration correction fit in
cycle 4), and thresholds_cycle5.json (the operating threshold, unchanged
from cycle 4) exactly as saved.

Writes <out_csv> with columns:
    patient_id, probability, alert
(probability = final, per-icu_type-recalibrated predicted probability of
 in_hospital_death; alert = 1 if probability >= the deployment threshold,
 else 0)

A patient whose icu_type is missing or not one of the four known
categories is scored with the uncorrected cycle-2 probability (no
per-unit correction is defined for them).
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icu_features import build_features_v2, align_features  # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    with open(os.path.join(here, "recalibration_cycle4.json")) as f:
        recal_params = json.load(f)["params"]
    with open(os.path.join(here, "thresholds_cycle5.json")) as f:
        threshold = json.load(f)["threshold"]

    feat = build_features_v2(patients, measurements)
    feat = align_features(feat, feature_columns)

    raw_p = model.predict_proba(feat)[:, 1]
    icu_type = patients.set_index("patient_id").reindex(feat.index)["icu_type"]

    recal_p = raw_p.copy()
    for cat, params in recal_params.items():
        mask = (icu_type == cat).to_numpy()
        if mask.any():
            recal_p[mask] = sigmoid(params["intercept"] + params["slope"] * logit(raw_p[mask]))

    alert = (recal_p >= threshold).astype(int)

    out = pd.DataFrame({"patient_id": feat.index, "probability": recal_p, "alert": alert})
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} predictions to {out_csv} "
          f"({alert.sum()} alerts, {alert.mean():.1%})")


if __name__ == "__main__":
    main()
