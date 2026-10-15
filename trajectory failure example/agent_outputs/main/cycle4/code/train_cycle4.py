"""
Cycle 4: investigate and fix the cardiac-surgery-recovery-unit complaint.

The director of the cardiac surgery recovery unit reports the bedside alert
(cycle 3, single global threshold 0.7367) seems inaccurate for their
patients. This script diagnoses the cause using TRAIN/VAL only, then fixes
it with per-icu_type thresholds, and finally checks the fix on TEST exactly
once.

No new model is fit: the underlying risk score is unchanged from cycle 2
(model_cycle2.pkl). This cycle only changes how the score is turned into an
alert, the same scope as cycle 3.

Test-set discipline (unchanged from cycles 1-3): split_cycle3.csv is reused
verbatim as split_cycle4.csv. All diagnosis and threshold-fitting below uses
train/val only (via a train-only refit of the cycle-2 sub-models, exactly as
in cycle 3, to get genuinely out-of-sample val scores for a model that was
NOT fit on val). Test is scored exactly once, at the end, broken down by
icu_type, and that is the only test-set read in this script.
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from icu_features import build_feature_matrix_v2, align_features

RANDOM_STATE = 42
TARGET_ALERT_RATE = 0.10
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


HGB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0,
                   min_samples_leaf=20)
LR_C = 0.01
BLEND_WEIGHTS = (0.5, 0.5)

# ---------------------------------------------------------------------------
# 1. Load data; reuse the split verbatim (never redrawn)
# ---------------------------------------------------------------------------
patients = pd.read_csv("data/patients.csv")
measurements = pd.read_csv("data/measurements.csv")
outcomes = pd.read_csv("data/outcomes.csv").set_index("patient_id")["in_hospital_death"]
icu_type_all = patients.set_index("patient_id")["icu_type"]

split = pd.read_csv("split_cycle3.csv").set_index("patient_id")["split"]
split.rename("split").rename_axis("patient_id").reset_index().to_csv(
    "split_cycle4.csv", index=False
)
log("Reused split_cycle3.csv assignment verbatim as split_cycle4.csv "
    "(same train/val/test patients as cycles 1-3).")

feats = build_feature_matrix_v2(patients, measurements)
y = outcomes.reindex(feats.index)

train_ids = split[split == "train"].index
val_ids = split[split == "val"].index
test_ids = split[split == "test"].index

X_train, y_train = feats.loc[train_ids], y.loc[train_ids]
X_val, y_val = feats.loc[val_ids], y.loc[val_ids]
X_test, y_test = feats.loc[test_ids], y.loc[test_ids]

log(f"\nOverall mortality rate by icu_type, train+val only (test withheld):")
tv_ids = train_ids.union(val_ids)
log(outcomes.reindex(tv_ids).groupby(icu_type_all.reindex(tv_ids)).agg(["size", "mean"]).to_string())

# ---------------------------------------------------------------------------
# 2. Diagnosis: train-only refit of the cycle-2 sub-models (same
#    hyperparameters, unchanged), scored on VAL (genuinely out-of-sample for
#    these copies) -- exactly the cycle-3 methodology, reused here for
#    diagnosis instead of a single global threshold.
# ---------------------------------------------------------------------------
hgb_trainonly = HistGradientBoostingClassifier(
    random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15,
    class_weight="balanced", **HGB_PARAMS,
)
hgb_trainonly.fit(X_train, y_train)
lr_trainonly = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(C=LR_C, max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)),
])
lr_trainonly.fit(X_train, y_train)

w_hgb, w_lr = BLEND_WEIGHTS
p_val_oos = (w_hgb * hgb_trainonly.predict_proba(X_val)[:, 1]
             + w_lr * lr_trainonly.predict_proba(X_val)[:, 1])

diag = pd.DataFrame({
    "y": y_val.to_numpy(),
    "p": p_val_oos,
    "icu_type": icu_type_all.reindex(val_ids).to_numpy(),
})

global_threshold = float(np.quantile(diag["p"], 1 - TARGET_ALERT_RATE))
log(f"\nGlobal threshold (as in cycle 3, recomputed here for consistency): "
    f"{global_threshold:.4f}")

log("\n--- Diagnosis on VAL (out-of-sample): discrimination/calibration by icu_type ---")
for unit, g in diag.groupby("icu_type"):
    auc = roc_auc_score(g["y"], g["p"])
    auprc = average_precision_score(g["y"], g["p"])
    brier = brier_score_loss(g["y"], g["p"])
    alert_rate_global = (g["p"] >= global_threshold).mean()
    log(f"  {unit:28s} n={len(g):4d} death_rate={g['y'].mean():.3f} "
        f"mean_pred={g['p'].mean():.3f} AUROC={auc:.3f} AUPRC={auprc:.3f} "
        f"Brier={brier:.4f} alert_rate_at_GLOBAL_threshold={alert_rate_global:.3f}")

log("\nDiagnosis: AUROC is good-to-excellent in every unit, including cardiac_surgery_"
    "recovery (highest of the four) -- the model discriminates within-unit risk well "
    "everywhere. The problem is the GLOBAL threshold: cardiac_surgery_recovery's "
    "baseline mortality (~4.7% train+val) is roughly 3-4x lower than the other units "
    "(14-19%), so its predicted probabilities sit systematically lower on the same "
    "absolute scale, and a single population-wide 90th-percentile cutoff almost never "
    "reaches into that unit's score range. Result: the alert fires for barely over 1% "
    "of cardiac_surgery_recovery admissions (vs. its 10% population-wide target), "
    "catching very few of that unit's deaths, which plausibly reads to the unit "
    "director as \"inaccurate\" / effectively non-functional, even though the "
    "underlying risk score ranks that unit's patients about as well as any other.")

# ---------------------------------------------------------------------------
# 3. Fix: per-icu_type thresholds, each calibrated to ~10% alert rate WITHIN
#    that unit, using the same out-of-sample VAL scores.
# ---------------------------------------------------------------------------
per_unit_threshold = {}
log("\n--- Fix: per-unit thresholds for ~10% within-unit alert rate (VAL, out-of-sample) ---")
for unit, g in diag.groupby("icu_type"):
    thr = float(np.quantile(g["p"], 1 - TARGET_ALERT_RATE))
    per_unit_threshold[unit] = thr
    alert = (g["p"] >= thr).astype(int)
    tp = int(((alert == 1) & (g["y"] == 1)).sum())
    fp = int(((alert == 1) & (g["y"] == 0)).sum())
    fn = int(((alert == 0) & (g["y"] == 1)).sum())
    tn = int(((alert == 0) & (g["y"] == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    log(f"  {unit:28s} thr={thr:.4f} alert_rate={alert.mean():.3f} "
        f"sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f}")

# ---------------------------------------------------------------------------
# 4. Apply the DEPLOYED model (model_cycle2.pkl, train+val-fit, unchanged)
#    to TEST, exactly once, both under the old single global threshold and
#    the new per-unit thresholds, broken down by icu_type.
# ---------------------------------------------------------------------------
with open("model_cycle2.pkl", "rb") as f:
    bundle = pickle.load(f)
deployed_model = bundle["model"]
feature_columns = bundle["feature_columns"]


def predict_proba_deployed(X):
    if isinstance(deployed_model, dict) and deployed_model.get("kind") == "blend":
        wh, wl = deployed_model["weights"]
        return (wh * deployed_model["hgb"].predict_proba(X)[:, 1]
                + wl * deployed_model["lr"].predict_proba(X)[:, 1])
    return deployed_model.predict_proba(X)[:, 1]


X_test_aligned = align_features(X_test, feature_columns)
log("\n=== FINAL TEST EVALUATION, CYCLE 4 (single pass, deployed model, by icu_type) ===")
p_test = predict_proba_deployed(X_test_aligned)
test_icu_type = icu_type_all.reindex(test_ids)
test_df = pd.DataFrame({"y": y_test.to_numpy(), "p": p_test,
                         "icu_type": test_icu_type.to_numpy()})


def summarize(g, thr):
    alert = (g["p"] >= thr).astype(int)
    tp = int(((alert == 1) & (g["y"] == 1)).sum())
    fp = int(((alert == 1) & (g["y"] == 0)).sum())
    fn = int(((alert == 0) & (g["y"] == 1)).sum())
    tn = int(((alert == 0) & (g["y"] == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return dict(n=len(g), alert_rate=float(alert.mean()), tp=tp, fp=fp, fn=fn, tn=tn,
                sensitivity=sens, specificity=spec, ppv=ppv, npv=npv)


log("\n-- OLD policy: single global threshold (%.4f), by icu_type --" % global_threshold)
old_by_unit = {}
for unit, g in test_df.groupby("icu_type"):
    s = summarize(g, global_threshold)
    old_by_unit[unit] = s
    log(f"  {unit:28s} {s}")
s_overall_old = summarize(test_df, global_threshold)
log(f"  {'OVERALL':28s} {s_overall_old}")

log("\n-- NEW policy: per-unit thresholds, by icu_type --")
new_by_unit = {}
for unit, g in test_df.groupby("icu_type"):
    thr = per_unit_threshold[unit]
    s = summarize(g, thr)
    new_by_unit[unit] = s
    log(f"  {unit:28s} thr={thr:.4f} {s}")
test_df["thr"] = test_df["icu_type"].map(per_unit_threshold)
alert_new_overall = (test_df["p"] >= test_df["thr"]).astype(int)
tp = int(((alert_new_overall == 1) & (test_df["y"] == 1)).sum())
fp = int(((alert_new_overall == 1) & (test_df["y"] == 0)).sum())
fn = int(((alert_new_overall == 0) & (test_df["y"] == 1)).sum())
tn = int(((alert_new_overall == 0) & (test_df["y"] == 0)).sum())
sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / (tp + fp); npv = tn / (tn + fn)
s_overall_new = dict(n=len(test_df), alert_rate=float(alert_new_overall.mean()),
                      tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens, specificity=spec,
                      ppv=ppv, npv=npv)
log(f"  {'OVERALL':28s} {s_overall_new}")

test_auroc = roc_auc_score(test_df["y"], test_df["p"])
test_auprc = average_precision_score(test_df["y"], test_df["p"])
log(f"\n(For reference, unchanged from cycle 2: test AUROC={test_auroc:.4f}, "
    f"test AUPRC={test_auprc:.4f} -- the underlying model is identical; only the "
    f"alert policy changed.)")

# ---------------------------------------------------------------------------
# 5. Persist thresholds + metrics
# ---------------------------------------------------------------------------
threshold_record = {
    "threshold": global_threshold,
    "per_unit": per_unit_threshold,
    "target_alert_rate": TARGET_ALERT_RATE,
    "selected_on": "val (out-of-sample scores from train-only refit of cycle-2 sub-models), "
                   "same method as cycle 3, computed overall (threshold) and per icu_type (per_unit)",
    "applied_to": "model_cycle2.pkl (deployed, train+val-fit) predictions",
    "model_source": "model_cycle2.pkl",
    "usage": "for a given patient, look up per_unit[patient's icu_type]; fall back to "
             "the top-level threshold only if icu_type is missing/unrecognized",
    "test_overall_old_global_threshold_policy": s_overall_old,
    "test_overall_new_per_unit_policy": s_overall_new,
    "test_by_unit_old_global_threshold_policy": old_by_unit,
    "test_by_unit_new_per_unit_policy": new_by_unit,
}
with open("thresholds_cycle4.json", "w") as f:
    json.dump(threshold_record, f, indent=2)

with open("cycle4_metrics.txt", "w") as f:
    f.write("\n".join(log_lines))

log("\nSaved thresholds_cycle4.json, split_cycle4.csv, cycle4_metrics.txt")
