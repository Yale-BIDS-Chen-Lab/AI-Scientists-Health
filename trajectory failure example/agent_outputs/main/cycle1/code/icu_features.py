"""
Shared feature-engineering for the ICU in-hospital mortality model (cycle 1).

Used by both train_cycle1.py (to build the training matrix) and predict_cycle1.py
(to build the matrix for new patients). Keeping this logic in one place guarantees
that inference-time features are constructed identically to training-time features.

All summary statistics are computed only from measurements with
minutes_since_admission in [0, 2880], i.e. the first 48 hours, consistent with the
data dictionary.
"""

import numpy as np
import pandas as pd

ALL_VARIABLES = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR",
    "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
    "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight",
]

# Physiologically implausible values that are almost certainly sensor/entry
# artifacts (documented sentinel codes and out-of-range readings observed in
# this dataset, e.g. Weight == -1, Temp == -17.8, pH == 0 or 735). Values
# outside these [low, high] bounds are set to missing before aggregation.
# Variables not listed here are left unfiltered (their observed ranges look
# physiologically plausible).
PLAUSIBLE_RANGE = {
    "Weight": (20, 300),
    "Temp": (25, 45),
    "pH": (6.5, 8.0),
    "HR": (20, 300),
    "MAP": (1, 300),
    "SysABP": (1, 300),
    "DiasABP": (1, 300),
    "NIMAP": (1, 300),
    "NISysABP": (1, 300),
    "NIDiasABP": (1, 300),
    "PaCO2": (1, 200),
    "PaO2": (1, 700),
    "WBC": (0.01, 500),
    "Mg": (0.1, 15),
}

ICU_TYPES = ["coronary_care", "cardiac_surgery_recovery", "medical", "surgical"]

STAT_NAMES = ["count", "mean", "min", "max", "last"]


def _clean_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    m = measurements.copy()
    m = m[(m["minutes_since_admission"] >= 0) & (m["minutes_since_admission"] <= 2880)]
    for var, (lo, hi) in PLAUSIBLE_RANGE.items():
        mask = m["variable"] == var
        bad = mask & ((m["value"] < lo) | (m["value"] > hi))
        m.loc[bad, "value"] = np.nan
    m = m.dropna(subset=["value"])
    return m


def _measurement_stats(measurements: pd.DataFrame, patient_ids: pd.Index) -> pd.DataFrame:
    """Per-patient, per-variable count/mean/min/max/last-value, wide format."""
    m = _clean_measurements(measurements)

    grp = m.groupby(["patient_id", "variable"])["value"]
    basic = grp.agg(["count", "mean", "min", "max"])

    last = (
        m.sort_values("minutes_since_admission")
        .groupby(["patient_id", "variable"])["value"]
        .last()
        .rename("last")
    )
    stats = basic.join(last)

    wide = stats.unstack("variable")  # columns: MultiIndex (stat, variable)
    wide.columns = [f"{var}_{stat}" for stat, var in wide.columns]
    wide = wide.reindex(patient_ids)

    # Build the full, fixed column set so every variable/stat combo exists
    # even if absent in this particular dataset.
    full_cols = [f"{var}_{stat}" for var in ALL_VARIABLES for stat in STAT_NAMES]
    wide = wide.reindex(columns=full_cols)

    for var in ALL_VARIABLES:
        count_col = f"{var}_count"
        wide[count_col] = wide[count_col].fillna(0)
        wide[f"measured_{var}"] = (wide[count_col] > 0).astype(int)

    return wide


def build_feature_matrix(patients: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature matrix, indexed by patient_id.

    patients: columns patient_id, age_years, sex, height_cm, weight_kg, icu_type
    measurements: columns patient_id, minutes_since_admission, variable, value
    """
    p = patients.set_index("patient_id")
    stats = _measurement_stats(measurements, p.index)

    feats = pd.DataFrame(index=p.index)
    feats["age_years"] = p["age_years"]
    feats["height_cm"] = p["height_cm"]
    feats["weight_kg_admission"] = p["weight_kg"]
    bmi = p["weight_kg"] / (p["height_cm"] / 100.0) ** 2
    feats["bmi"] = bmi.where((bmi > 10) & (bmi < 100))

    feats["is_male"] = (p["sex"] == "male").astype(int)
    feats["is_female"] = (p["sex"] == "female").astype(int)

    for t in ICU_TYPES:
        feats[f"icu_type_{t}"] = (p["icu_type"] == t).astype(int)

    feats = feats.join(stats)

    return feats


def get_feature_columns(feats: pd.DataFrame) -> list:
    return list(feats.columns)


def align_features(feats: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Reindex a freshly-built feature matrix to the training-time column
    order/set, so the model always sees the same columns in the same order.
    Missing dummy/indicator columns are filled with 0; missing numeric stat
    columns are left as NaN (true missingness)."""
    out = feats.reindex(columns=feature_columns)
    zero_fill_prefixes = ("is_male", "is_female", "icu_type_", "measured_")
    for col in feature_columns:
        if col.startswith(zero_fill_prefixes) or col.endswith("_count"):
            out[col] = out[col].fillna(0)
    return out
