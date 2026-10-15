"""
Shared feature engineering for the ICU in-hospital mortality model (cycle 1).

Used by both the training script and predict_cycle1.py, so that inference
reproduces exactly the transformation the model was fit on. This module has
no side effects at import time and does not read any files itself.
"""
import numpy as np
import pandas as pd

# All 37 long-format variables, per docs/data_dictionary.md
MEASURE_VARS = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR",
    "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
    "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight",
]

# Variables where a reading of exactly 0 (or below) is physiologically
# impossible in a living, monitored patient and is treated as a sensor/entry
# artifact (arterial-line flush, lead-off, etc.) rather than a real value.
_ZERO_CLIP_VARS = [
    "HR", "SysABP", "DiasABP", "MAP", "NISysABP", "NIDiasABP", "NIMAP",
    "RespRate",
]

STAT_NAMES = ["count", "mean", "min", "max", "last"]


def clean_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    """Null out a small number of physiologically implausible values.

    These rules were chosen by inspecting the raw distributions (see
    state_cycle1.md): Weight uses -1 as an explicit missing-value sentinel;
    a few hundred Temp readings are far outside a survivable range; a
    handful of pH readings are off by a missing decimal point (e.g. 735
    instead of 7.35); and a small fraction of arterial/BP/HR/RespRate
    readings of <=0 are monitor artifacts. Together these affect well
    under 1% of rows for any single variable and are set to NaN (not
    imputed to a guessed value).
    """
    m = measurements.copy()
    v = m["variable"]
    val = m["value"]

    m.loc[(v == "Weight") & (val == -1), "value"] = np.nan
    m.loc[(v == "Temp") & ((val < 25) | (val > 45)), "value"] = np.nan
    m.loc[(v == "pH") & ((val < 6.5) | (val > 8.0)), "value"] = np.nan
    m.loc[v.isin(_ZERO_CLIP_VARS) & (val <= 0), "value"] = np.nan

    return m.dropna(subset=["value"])


def clean_patients(patients: pd.DataFrame) -> pd.DataFrame:
    """Null out a small number of implausible height/weight entries.

    height_cm has 20 values (of 4176 non-missing) outside [100, 230] cm,
    ranging from 1.8 cm to 462.3 cm -- clearly unit/decimal errors, not
    real adult heights. weight_kg has one value of 472 kg, also treated as
    an entry error (a few hundred kg still occur in real morbidly-obese
    ICU patients and are kept). These are set to NaN, not corrected by
    guessing the intended value, since without knowing the input error
    that produced them a guess is not defensible; the model handles the
    resulting missingness like any other missing value.
    """
    p = patients.copy()
    bad_height = (p["height_cm"] < 100) | (p["height_cm"] > 230)
    p.loc[bad_height, "height_cm"] = np.nan
    bad_weight = p["weight_kg"] > 400
    p.loc[bad_weight, "weight_kg"] = np.nan
    return p


def build_features(patients: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    """Build one row of features per patient_id.

    Feature groups:
      - demographics: age, sex, height, weight, BMI, icu_type one-hot
      - per measurement variable (37 vars x 5 stats = 185 cols):
        count / mean / min / max / last (last = value at the latest
        minutes_since_admission for that patient-variable)
      - Urine_sum: total recorded urine output over the 48h window
        (clinically meaningful as a total, unlike mean/min/max of a
        cumulative-interval quantity)

    Missing variable-patient combinations: *_count is filled with 0
    (the variable was never measured); all other *_mean/min/max/last stay
    NaN and are left for the model to handle (the chosen model natively
    supports NaN; see state_cycle1.md).
    """
    patients = clean_patients(patients)
    patients["patient_id"] = patients["patient_id"].astype(str)
    all_ids = patients["patient_id"]

    m = clean_measurements(measurements)
    m = m.copy()
    m["patient_id"] = m["patient_id"].astype(str)

    if len(m) > 0:
        stats = (
            m.groupby(["patient_id", "variable"])["value"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
        )
        last = (
            m.sort_values("minutes_since_admission")
            .groupby(["patient_id", "variable"])["value"]
            .last()
            .reset_index(name="last")
        )
        stats = stats.merge(last, on=["patient_id", "variable"], how="outer")
        wide = stats.pivot(index="patient_id", columns="variable", values=STAT_NAMES)
        wide.columns = [f"{var}_{stat}" for stat, var in wide.columns]
    else:
        wide = pd.DataFrame(index=pd.Index([], name="patient_id"))

    # Ensure every variable/stat combo exists as a column even if absent
    # from this particular dataset (keeps column set stable across cycles).
    expected_cols = [f"{var}_{stat}" for var in MEASURE_VARS for stat in STAT_NAMES]
    wide = wide.reindex(columns=expected_cols)
    wide = wide.reindex(all_ids.values)

    for var in MEASURE_VARS:
        col = f"{var}_count"
        wide[col] = wide[col].fillna(0)

    if len(m) > 0:
        urine_sum = (
            m[m["variable"] == "Urine"].groupby("patient_id")["value"].sum().rename("Urine_sum")
        )
    else:
        urine_sum = pd.Series(dtype=float, name="Urine_sum")
    urine_sum = urine_sum.reindex(all_ids.values)
    wide["Urine_sum"] = urine_sum

    demo = patients.set_index("patient_id")
    demo = demo.reindex(all_ids.values)
    demo_feat = pd.DataFrame(index=demo.index)
    demo_feat["age_years"] = demo["age_years"]
    demo_feat["is_male"] = demo["sex"].map({"male": 1, "female": 0})
    demo_feat["height_cm"] = demo["height_cm"]
    demo_feat["weight_kg"] = demo["weight_kg"]
    with np.errstate(divide="ignore", invalid="ignore"):
        demo_feat["bmi"] = demo["weight_kg"] / (demo["height_cm"] / 100.0) ** 2

    icu_categories = ["coronary_care", "cardiac_surgery_recovery", "medical", "surgical"]
    for cat in icu_categories:
        demo_feat[f"icu_{cat}"] = (demo["icu_type"] == cat).astype(int)

    feat = pd.concat([demo_feat, wide], axis=1)
    feat.index.name = "patient_id"
    return feat


def get_feature_columns(feat: pd.DataFrame) -> list:
    return list(feat.columns)


def align_features(feat: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Reindex a freshly built feature frame to the exact training column
    order/set, so a model fit in one cycle can score data built in another
    run without silently shifting columns. Any column the training set had
    but this data doesn't get NaN (or 0 for *_count); any extra column in
    this data that training never saw is dropped.
    """
    out = feat.reindex(columns=feature_columns)
    for col in feature_columns:
        if col.endswith("_count"):
            out[col] = out[col].fillna(0)
    return out
