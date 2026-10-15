"""
Cycle-1 training script for the ICU in-hospital mortality model.

Discipline enforced here (test-set integrity):
  1. patient_id is split ONCE into train/val/test (stratified on outcome, fixed seed).
  2. All feature-engineering choices, hyperparameter search, and threshold
     selection use ONLY train and val.
  3. The test set is loaded but its labels are not looked at until the single
     final evaluation block at the bottom of this script, which runs once and
     is treated as final -- no re-tuning after seeing test numbers.
  4. split_cycle1.csv records the assignment so future cycles can reuse the
     exact same held-out test set instead of drawing a new one.

Outputs written to this directory:
  - model_cycle1.pkl      (fitted primary model + feature column list)
  - split_cycle1.csv       (patient_id,split)
  - cycle1_metrics.txt     (val model-selection log + final test metrics)
  - cycle1_importance.csv  (permutation importance on the test set)
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve,
)
from sklearn.inspection import permutation_importance

from icu_features import build_feature_matrix, get_feature_columns

RANDOM_STATE = 42
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
patients = pd.read_csv("data/patients.csv")
measurements = pd.read_csv("data/measurements.csv")
outcomes = pd.read_csv("data/outcomes.csv").set_index("patient_id")["in_hospital_death"]

feats = build_feature_matrix(patients, measurements)
feature_columns = get_feature_columns(feats)
y = outcomes.reindex(feats.index)
assert y.isna().sum() == 0, "every admission must have an outcome label"

log(f"Feature matrix: {feats.shape[0]} patients x {feats.shape[1]} features")
log(f"Overall mortality rate: {y.mean():.4f}")

# ---------------------------------------------------------------------------
# 2. Single train / val / test split (70/15/15), stratified on outcome
# ---------------------------------------------------------------------------
ids = feats.index.to_numpy()
train_ids, temp_ids, y_train_ids, y_temp_ids = train_test_split(
    ids, y.loc[ids].to_numpy(), test_size=0.30, stratify=y.loc[ids],
    random_state=RANDOM_STATE,
)
val_ids, test_ids, y_val_ids, y_test_ids = train_test_split(
    temp_ids, y_temp_ids, test_size=0.50, stratify=y_temp_ids,
    random_state=RANDOM_STATE,
)

split_map = pd.Series("train", index=ids)
split_map.loc[val_ids] = "val"
split_map.loc[test_ids] = "test"
split_df = split_map.rename("split").rename_axis("patient_id").reset_index()
split_df.to_csv("split_cycle1.csv", index=False)

log(f"train n={len(train_ids)} (deaths={y.loc[train_ids].sum()}), "
    f"val n={len(val_ids)} (deaths={y.loc[val_ids].sum()}), "
    f"test n={len(test_ids)} (deaths={y.loc[test_ids].sum()})")

X_train, y_train = feats.loc[train_ids], y.loc[train_ids]
X_val, y_val = feats.loc[val_ids], y.loc[val_ids]
X_test, y_test = feats.loc[test_ids], y.loc[test_ids]

# ---------------------------------------------------------------------------
# 3. Model selection on train/val ONLY -- primary model: HistGradientBoosting
#    (handles missing values natively; no imputation needed)
# ---------------------------------------------------------------------------
log("\n--- Hyperparameter search (train fit, val AUROC) ---")
param_grid = [
    dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0),
    dict(learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0),
    dict(learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0),
    dict(learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0),
    dict(learning_rate=0.1, max_leaf_nodes=63, l2_regularization=1.0),
]
best_auc, best_params = -1, None
for params in param_grid:
    clf = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, early_stopping=True,
        validation_fraction=0.15, class_weight="balanced", **params,
    )
    clf.fit(X_train, y_train)
    p_val = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val)
    log(f"  {params} -> val AUROC={auc:.4f} (n_iter={clf.n_iter_})")
    if auc > best_auc:
        best_auc, best_params = auc, params

log(f"Best HGB params: {best_params} (val AUROC={best_auc:.4f})")

# Refit on train+val with the chosen hyperparameters (more data for the
# deployed model); early_stopping still uses an internal held-out slice of
# train+val to pick the number of boosting iterations, not the test set.
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_model = HistGradientBoostingClassifier(
    random_state=RANDOM_STATE, early_stopping=True,
    validation_fraction=0.15, class_weight="balanced", **best_params,
)
final_model.fit(X_trainval, y_trainval)
log(f"Final HGB refit on train+val (n={len(X_trainval)}), n_iter={final_model.n_iter_}")

# ---------------------------------------------------------------------------
# 3b. Secondary interpretable reference model: logistic regression
#     (for sanity-checking direction/sign of effects, not for deployment)
# ---------------------------------------------------------------------------
log("\n--- Logistic regression reference model (train fit, val AUROC) ---")
best_c, best_c_auc = None, -1
for C in [0.003, 0.01, 0.03, 0.1, 0.3]:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    p_val = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val)
    log(f"  C={C} -> val AUROC={auc:.4f}")
    if auc > best_c_auc:
        best_c_auc, best_c = auc, C

lr_final = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)),
])
lr_final.fit(X_trainval, y_trainval)
log(f"Logistic regression: best C={best_c} (val AUROC={best_c_auc:.4f}), refit on train+val")

# ---------------------------------------------------------------------------
# 4. Decision threshold chosen on VAL only (Youden's J), before touching test
# ---------------------------------------------------------------------------
p_val_final = final_model.predict_proba(X_val)[:, 1]
fpr, tpr, thr = roc_curve(y_val, p_val_final)
youden = tpr - fpr
chosen_threshold = thr[np.argmax(youden)]
log(f"\nDecision threshold chosen on val (Youden's J): {chosen_threshold:.4f}")

# ---------------------------------------------------------------------------
# 5. SINGLE final evaluation on the held-out test set (touched exactly once)
# ---------------------------------------------------------------------------
log("\n=== FINAL TEST EVALUATION (single pass, not used for any tuning) ===")
p_test = final_model.predict_proba(X_test)[:, 1]
test_auroc = roc_auc_score(y_test, p_test)
test_auprc = average_precision_score(y_test, p_test)
test_brier = brier_score_loss(y_test, p_test)
pred_label = (p_test >= chosen_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_label).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) else float("nan")
npv = tn / (tn + fn) if (tn + fn) else float("nan")

log(f"Test AUROC:  {test_auroc:.4f}")
log(f"Test AUPRC:  {test_auprc:.4f} (base rate {y_test.mean():.4f})")
log(f"Test Brier:  {test_brier:.4f}")
log(f"At threshold {chosen_threshold:.4f}: sensitivity={sens:.3f}, "
    f"specificity={spec:.3f}, PPV={ppv:.3f}, NPV={npv:.3f}")
log(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

# calibration by decile
cal = pd.DataFrame({"y": y_test.to_numpy(), "p": p_test})
cal["bin"] = pd.qcut(cal["p"], 10, duplicates="drop")
cal_table = cal.groupby("bin", observed=True).agg(mean_pred=("p", "mean"),
                                                    obs_rate=("y", "mean"),
                                                    n=("y", "size"))
log("\nCalibration by predicted-risk decile (test set):")
log(cal_table.to_string())

# Logistic reference model's test AUROC, for comparison only
p_test_lr = lr_final.predict_proba(X_test)[:, 1]
log(f"\n[Reference] Logistic regression test AUROC: {roc_auc_score(y_test, p_test_lr):.4f}")

# ---------------------------------------------------------------------------
# 6. Interpretability: permutation importance on the test set (single pass)
# ---------------------------------------------------------------------------
log("\n--- Permutation importance (primary model, test set, single pass) ---")
X_test_filled = X_test  # HGB handles NaN natively
perm = permutation_importance(
    final_model, X_test_filled, y_test, n_repeats=10,
    random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=1,
)
imp_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
imp_df.to_csv("cycle1_importance.csv", index=False)
log(imp_df.head(25).to_string(index=False))

# native gain-based importance too, cheap cross-check
native_imp = pd.Series(final_model.feature_importances_, index=X_test.columns) \
    if hasattr(final_model, "feature_importances_") else None
if native_imp is not None:
    log("\nNative (gain-based) importance not exposed by HistGradientBoostingClassifier; "
        "permutation importance above is the primary evidence.")

# Logistic regression coefficients, for direction sanity-check
lr_coefs = pd.Series(
    lr_final.named_steps["lr"].coef_[0], index=X_train.columns
).sort_values(key=np.abs, ascending=False)
lr_coefs.to_csv("cycle1_lr_coefficients.csv", header=["standardized_coef"])
log("\nTop 15 logistic-regression standardized coefficients (sanity check on direction):")
log(lr_coefs.head(15).to_string())

# ---------------------------------------------------------------------------
# 7. Persist model bundle for predict_cycle1.py
# ---------------------------------------------------------------------------
bundle = {
    "model": final_model,
    "feature_columns": feature_columns,
    "threshold": float(chosen_threshold),
    "model_type": "HistGradientBoostingClassifier",
    "trained_on": "train+val (cycle1)",
}
with open("model_cycle1.pkl", "wb") as f:
    pickle.dump(bundle, f)

with open("cycle1_metrics.txt", "w") as f:
    f.write("\n".join(log_lines))

log("\nSaved model_cycle1.pkl, split_cycle1.csv, cycle1_metrics.txt, "
    "cycle1_importance.csv, cycle1_lr_coefficients.csv")
