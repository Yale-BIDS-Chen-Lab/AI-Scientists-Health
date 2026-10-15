"""
Cycle 6: one clean, non-adaptive re-validation of the FROZEN model design.

Purpose: the external methods reviewer correctly noted that split_cycle1.csv's
test partition was read at the end of cycles 1, 2, 3, and 4, and that the
findings of each look shaped what was tried next (cycle 1's test-set
permutation importance motivated cycle 2's new trend/sum features; cycle 1's
"insufficient" verdict -- itself a reaction to a test-set number -- licensed
cycle 2's broader search; cycle 4 computed and compared test-set performance
for two competing threshold policies in reaching its decision). That
adaptive, human-in-the-loop feedback loop means the reported performance
trajectory (0.853 -> 0.857 AUROC, and everything downstream of it) cannot be
treated as an unbiased estimate of generalization performance, regardless of
how carefully any single cycle's internal train/val split was respected.

This script does the one thing that can actually fix that within the
existing 8,000-patient dataset: it FREEZES the model design decided by
cycle 2 (feature set v2, HGB+LR blend, exact hyperparameters) and the
threshold-setting PROCEDURE decided by cycles 3-4 (per-icu_type 90th
percentile of out-of-sample validation scores) -- no new hyperparameter
search, no new feature ideas, no new candidate models -- draws a FRESH
random split of all 8,000 patients that no decision in this project has
ever been conditioned on, fits the frozen design on the new train partition
only, sets thresholds from the new validation partition only, and reads the
new test partition exactly once, with no further changes made in response.

This corrects for adaptive-selection bias (repeated, decision-linked looks
at one static test set). It does NOT address the separate, still-open
limitation that all 8,000 patients come from one hospital over one
(unknown) time period -- that requires genuinely external or prospective
data, which is not available in this project's data/ directory.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from icu_features import build_feature_matrix_v2

FRESH_SEED = 2024  # deliberately different from seed 42 used in cycles 1-5
TARGET_ALERT_RATE = 0.10
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# Frozen from cycle 2 -- not re-tuned here.
HGB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0,
                   min_samples_leaf=20)
LR_C = 0.01
BLEND_WEIGHTS = (0.5, 0.5)

patients = pd.read_csv("data/patients.csv")
measurements = pd.read_csv("data/measurements.csv")
outcomes = pd.read_csv("data/outcomes.csv").set_index("patient_id")["in_hospital_death"]
icu_type_all = patients.set_index("patient_id")["icu_type"]

feats = build_feature_matrix_v2(patients, measurements)
y = outcomes.reindex(feats.index)
ids = feats.index.to_numpy()

# Stratify on outcome x icu_type jointly this time (an incidental
# improvement noted as an open item in cycle 5 -- larger, better-balanced
# per-unit test samples than a stratify-on-outcome-only split gives), using
# a seed no prior decision in this project has ever seen.
strata = (y.astype(str) + "_" + icu_type_all.reindex(feats.index).astype(str)).loc[ids]

train_ids, temp_ids = train_test_split(
    ids, test_size=0.30, stratify=strata, random_state=FRESH_SEED,
)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.50, stratify=strata.loc[temp_ids], random_state=FRESH_SEED,
)

split_map = pd.Series("train", index=ids)
split_map.loc[val_ids] = "val"
split_map.loc[test_ids] = "test"
split_map.rename("split").rename_axis("patient_id").reset_index().to_csv(
    "cycle6_fresh_split.csv", index=False
)

log(f"Fresh split (seed={FRESH_SEED}, stratified on outcome x icu_type, "
    f"never previously used for any decision in this project):")
log(f"  train n={len(train_ids)}, val n={len(val_ids)}, test n={len(test_ids)}")

X_train, y_train = feats.loc[train_ids], y.loc[train_ids]
X_val, y_val = feats.loc[val_ids], y.loc[val_ids]
X_test, y_test = feats.loc[test_ids], y.loc[test_ids]

# ---------------------------------------------------------------------------
# Fit the FROZEN design on the new train partition only. No search.
# ---------------------------------------------------------------------------
hgb = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.15,
    class_weight="balanced", **HGB_PARAMS,
)
hgb.fit(X_train, y_train)
lr = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(C=LR_C, max_iter=2000, class_weight="balanced",
                               random_state=42)),
])
lr.fit(X_train, y_train)

w_hgb, w_lr = BLEND_WEIGHTS


def blend_proba(X):
    return w_hgb * hgb.predict_proba(X)[:, 1] + w_lr * lr.predict_proba(X)[:, 1]


p_val = blend_proba(X_val)

# ---------------------------------------------------------------------------
# Set thresholds from the new val partition only, using the frozen
# per-icu_type-90th-percentile PROCEDURE from cycles 3-4.
# ---------------------------------------------------------------------------
val_icu = icu_type_all.reindex(val_ids)
global_threshold = float(np.quantile(p_val, 1 - TARGET_ALERT_RATE))
per_unit_threshold = {}
for unit in val_icu.unique():
    mask = (val_icu == unit).to_numpy()
    per_unit_threshold[unit] = float(np.quantile(p_val[mask], 1 - TARGET_ALERT_RATE))

log(f"\nFresh global threshold: {global_threshold:.4f}")
log(f"Fresh per-unit thresholds: {per_unit_threshold}")

# ---------------------------------------------------------------------------
# SINGLE look at the new test partition. Whatever comes out is reported;
# no further changes are made in response to it.
# ---------------------------------------------------------------------------
p_test = blend_proba(X_test)
test_auroc = roc_auc_score(y_test, p_test)
test_auprc = average_precision_score(y_test, p_test)
test_brier = brier_score_loss(y_test, p_test)
log(f"\n=== FRESH-SPLIT TEST RESULT (single look, frozen design, never adapted to) ===")
log(f"Test AUROC:  {test_auroc:.4f}   (originally reported, cycle 2, old split: 0.8572)")
log(f"Test AUPRC:  {test_auprc:.4f}   (originally reported, cycle 2, old split: 0.5116)")
log(f"Test Brier:  {test_brier:.4f}   (originally reported, cycle 2, old split: 0.1328)")

test_icu = icu_type_all.reindex(test_ids)
test_df = pd.DataFrame({"y": y_test.to_numpy(), "p": p_test, "icu_type": test_icu.to_numpy(),
                         "thr": test_icu.map(per_unit_threshold).to_numpy()})
log("\n-- Per-unit thresholds applied to fresh test partition --")
overall_alert = (test_df["p"] >= test_df["thr"]).astype(int)
for unit, g in test_df.groupby("icu_type"):
    alert = (g["p"] >= g["thr"]).astype(int)
    tp = int(((alert == 1) & (g["y"] == 1)).sum())
    fp = int(((alert == 1) & (g["y"] == 0)).sum())
    fn = int(((alert == 0) & (g["y"] == 1)).sum())
    tn = int(((alert == 0) & (g["y"] == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    log(f"  {unit:28s} n={len(g):4d} deaths={int(g['y'].sum()):3d} alert_rate={alert.mean():.3f} "
        f"sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f}")
tp = int(((overall_alert == 1) & (test_df["y"] == 1)).sum())
fp = int(((overall_alert == 1) & (test_df["y"] == 0)).sum())
fn = int(((overall_alert == 0) & (test_df["y"] == 1)).sum())
tn = int(((overall_alert == 0) & (test_df["y"] == 0)).sum())
sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / (tp + fp)
log(f"  {'OVERALL':28s} n={len(test_df):4d} deaths={int(test_df['y'].sum()):3d} "
    f"alert_rate={overall_alert.mean():.3f} sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f}")

with open("cycle6_fresh_validation_metrics.txt", "w") as f:
    f.write("\n".join(log_lines))
log("\nSaved cycle6_fresh_split.csv, cycle6_fresh_validation_metrics.txt")
