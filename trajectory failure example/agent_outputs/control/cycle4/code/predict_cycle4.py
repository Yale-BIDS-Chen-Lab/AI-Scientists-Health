#!/usr/bin/env python3
"""
Score new ICU admissions with the cycle-2 model, per-icu_type recalibration
(cycle 4, fixing systematic overprediction for cardiac_surgery_recovery /
coronary_care patients), and the cycle-4 bedside-alert threshold.

Usage:
    python3 predict_cycle4.py <data_dir> <out_csv>

<data_dir> must contain patients.csv and measurements.csv in the same
format as docs/data_dictionary.md (outcomes.csv is not required and is
ignored if present). Nothing is trained or fit here -- this script loads
model_cycle2.joblib (unchanged since cycle 2), recalibration_cycle4.json,
and thresholds_cycle4.json exactly as saved.

Writes <out_csv> with columns:
    patient_id, probability, alert
(probability = recalibrated predicted probability of in_hospital_death,
 i.e. the cycle-2 model's raw output corrected per icu_type; alert = 1 if
 probability >= the cycle-4 threshold, else 0)

A patient whose icu_type is missing or not one of the four known
categories cannot be recalibrated (no per-unit correction applies to
them) and is scored with the cycle-2 model's raw, uncorrected
probability instead, since that is the best available estimate.
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
    with open(os.path.join(here, "thresholds_cycle4.json")) as f:
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
