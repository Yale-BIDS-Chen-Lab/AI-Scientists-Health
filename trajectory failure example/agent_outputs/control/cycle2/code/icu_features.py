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


STAT_NAMES_V2 = ["count", "mean", "min", "max", "first", "last"]

# A handful of physiology-motivated composite ratios, computed from the
# per-variable aggregate columns after the wide frame is built (not
# time-matched observation by observation -- an approximation flagged in
# state_cycle2.md).
_INTERACTION_SPECS = [
    ("shock_index_last", "HR_last", "SysABP_last"),
    ("bun_creat_ratio_last", "BUN_last", "Creatinine_last"),
]


def build_features_v2(patients: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    """Cycle-2 feature set: cycle-1's per-variable count/mean/min/max/last,
    plus signal cycle 1 flagged as missing (see state_cycle1.md open
    question 3, "feature set is trajectory-blind"):

      - first: first recorded value (so last-first captures a trend)
      - slope: ordinary-least-squares slope of value vs. time (per hour)
        across all readings of that variable for that patient -- direction
        and speed of change, not just endpoint/extreme values
      - {var}_early_mean / {var}_late_mean: mean value in the first 24h vs.
        second 24h of the window -- a cheap, robust trajectory summary that
        doesn't need >=2 points spread over time the way slope does
      - a few composite clinical ratios (shock index, BUN/creatinine
        ratio) and a GCS first-to-last delta, motivated by standard ICU
        severity scoring rather than mined from validation performance

    Same missingness convention as build_features: *_count is 0 when a
    variable was never measured; every other derived column is NaN when it
    cannot be computed (e.g. slope needs >=2 timestamped readings).
    """
    patients = clean_patients(patients)
    patients["patient_id"] = patients["patient_id"].astype(str)
    all_ids = patients["patient_id"]

    m = clean_measurements(measurements)
    m = m.copy()
    m["patient_id"] = m["patient_id"].astype(str)
    m["t_hours"] = m["minutes_since_admission"] / 60.0

    if len(m) > 0:
        stats = (
            m.groupby(["patient_id", "variable"])["value"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
        )
        m_sorted = m.sort_values("minutes_since_admission")
        first = (
            m_sorted.groupby(["patient_id", "variable"])["value"].first()
            .reset_index(name="first")
        )
        last = (
            m_sorted.groupby(["patient_id", "variable"])["value"].last()
            .reset_index(name="last")
        )
        stats = stats.merge(first, on=["patient_id", "variable"], how="outer")
        stats = stats.merge(last, on=["patient_id", "variable"], how="outer")
        wide = stats.pivot(index="patient_id", columns="variable", values=STAT_NAMES_V2)
        wide.columns = [f"{var}_{stat}" for stat, var in wide.columns]

        # Slope (per hour) via the closed-form OLS formula, fully vectorized:
        # slope = (n*sum_ty - sum_t*sum_y) / (n*sum_tt - sum_t^2), NaN if
        # n<2 or the time values are degenerate (all readings same minute).
        m["_ty"] = m["t_hours"] * m["value"]
        m["_tt"] = m["t_hours"] * m["t_hours"]
        g = m.groupby(["patient_id", "variable"])
        sums = g[["t_hours", "value", "_ty", "_tt"]].sum()
        n = g.size().rename("n")
        sums = sums.join(n)
        denom = sums["n"] * sums["_tt"] - sums["t_hours"] ** 2
        slope = (sums["n"] * sums["_ty"] - sums["t_hours"] * sums["value"]) / denom
        slope = slope.where((sums["n"] >= 2) & (denom.abs() > 1e-9))
        slope_wide = slope.reset_index(name="value").pivot(
            index="patient_id", columns="variable", values="value"
        )
        slope_wide.columns = [f"{c}_slope" for c in slope_wide.columns]

        # Early (0-24h) / late (24-48h) means.
        m["half"] = np.where(m["minutes_since_admission"] < 1440, "early", "late")
        half_mean = (
            m.groupby(["patient_id", "variable", "half"])["value"].mean().reset_index()
        )
        half_wide = half_mean.pivot(
            index="patient_id", columns=["variable", "half"], values="value"
        )
        half_wide.columns = [f"{var}_{half}_mean" for var, half in half_wide.columns]
    else:
        wide = pd.DataFrame(index=pd.Index([], name="patient_id"))
        slope_wide = pd.DataFrame(index=pd.Index([], name="patient_id"))
        half_wide = pd.DataFrame(index=pd.Index([], name="patient_id"))

    expected_cols = [f"{var}_{stat}" for var in MEASURE_VARS for stat in STAT_NAMES_V2]
    wide = wide.reindex(columns=expected_cols).reindex(all_ids.values)
    for var in MEASURE_VARS:
        wide[f"{var}_count"] = wide[f"{var}_count"].fillna(0)

    slope_cols = [f"{var}_slope" for var in MEASURE_VARS]
    slope_wide = slope_wide.reindex(columns=slope_cols).reindex(all_ids.values)

    half_cols = [f"{var}_{half}_mean" for var in MEASURE_VARS for half in ("early", "late")]
    half_wide = half_wide.reindex(columns=half_cols).reindex(all_ids.values)

    if len(m) > 0:
        urine_sum = (
            m[m["variable"] == "Urine"].groupby("patient_id")["value"].sum().rename("Urine_sum")
        )
    else:
        urine_sum = pd.Series(dtype=float, name="Urine_sum")
    urine_sum = urine_sum.reindex(all_ids.values)

    demo = patients.set_index("patient_id").reindex(all_ids.values)
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

    feat = pd.concat([demo_feat, wide, slope_wide, half_wide], axis=1)
    feat["Urine_sum"] = urine_sum

    for name, num_col, den_col in _INTERACTION_SPECS:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = feat[num_col] / feat[den_col]
        feat[name] = ratio.replace([np.inf, -np.inf], np.nan)

    # FiO2 is stored as a fraction (0-1) per the data dictionary, so the
    # standard PaO2/FiO2 ratio needs no percentage conversion.
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = feat["PaO2_min"] / feat["FiO2_max"]
    feat["pf_ratio_proxy"] = pf.replace([np.inf, -np.inf], np.nan)
    feat["gcs_delta"] = feat["GCS_last"] - feat["GCS_first"]

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
