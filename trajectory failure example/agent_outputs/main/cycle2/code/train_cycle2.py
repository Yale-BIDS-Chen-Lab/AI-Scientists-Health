"""
Cycle-2 training script for the ICU in-hospital mortality model.

Responds to the clinical committee's feedback that cycle-1 performance
(test AUROC 0.853) is insufficient, by trying (a) an expanded feature set and
(b) additional model classes, selected via train/val only.

Test-set discipline (unchanged from cycle 1, and this is the point of this
cycle given the project's name): the train/val/test PATIENT ASSIGNMENT is
read from split_cycle1.csv and reused byte-for-byte (re-saved as
split_cycle2.csv) rather than redrawn. All feature engineering, model-class
comparison, hyperparameter search, calibration choice, and threshold
selection below use ONLY train and val. The test set is scored exactly once,
in the single evaluation block near the bottom, and is not looked at again
afterwards to pick among cycle-2 candidates.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve,
)
from sklearn.inspection import permutation_importance

from icu_features import build_feature_matrix_v2, get_feature_columns, align_features

RANDOM_STATE = 42
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# 1. Load data and reuse the cycle-1 split exactly (no redraw)
# ---------------------------------------------------------------------------
patients = pd.read_csv("data/patients.csv")
measurements = pd.read_csv("data/measurements.csv")
outcomes = pd.read_csv("data/outcomes.csv").set_index("patient_id")["in_hospital_death"]

split_prev = pd.read_csv("split_cycle1.csv").set_index("patient_id")["split"]
split_prev.rename("split").rename_axis("patient_id").reset_index().to_csv(
    "split_cycle2.csv", index=False
)
log("Reused split_cycle1.csv assignment verbatim as split_cycle2.csv "
    "(same train/val/test patients as cycle 1 -- test set is not redrawn).")

feats = build_feature_matrix_v2(patients, measurements)
feature_columns = get_feature_columns(feats)
y = outcomes.reindex(feats.index)
assert y.isna().sum() == 0

log(f"Feature matrix v2: {feats.shape[0]} patients x {feats.shape[1]} features "
    f"(cycle 1 had 232)")

train_ids = split_prev[split_prev == "train"].index
val_ids = split_prev[split_prev == "val"].index
test_ids = split_prev[split_prev == "test"].index

X_train, y_train = feats.loc[train_ids], y.loc[train_ids]
X_val, y_val = feats.loc[val_ids], y.loc[val_ids]
X_test, y_test = feats.loc[test_ids], y.loc[test_ids]
log(f"train n={len(train_ids)}, val n={len(val_ids)}, test n={len(test_ids)} "
    "(identical patients to cycle 1)")

# ---------------------------------------------------------------------------
# 2. Candidate model classes, all selected on train-fit / val-scored basis
# ---------------------------------------------------------------------------
candidates = {}  # name -> (val_auroc, fitted_on_train_model_or_pipeline, kind)

log("\n--- Candidate A: HistGradientBoostingClassifier, expanded features ---")
hgb_grid = [
    dict(learning_rate=0.03, max_leaf_nodes=15, l2_regularization=0.0, min_samples_leaf=20),
    dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0, min_samples_leaf=20),
    dict(learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=20),
    dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=40),
    dict(learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20),
    dict(learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40),
]
best_hgb_auc, best_hgb_params, best_hgb_model = -1, None, None
for params in hgb_grid:
    clf = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, early_stopping=True,
        validation_fraction=0.15, class_weight="balanced", **params,
    )
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    log(f"  {params} -> val AUROC={auc:.4f} (n_iter={clf.n_iter_})")
    if auc > best_hgb_auc:
        best_hgb_auc, best_hgb_params, best_hgb_model = auc, params, clf
log(f"Best HGB (v2 features): {best_hgb_params} -> val AUROC={best_hgb_auc:.4f} "
    f"(cycle-1 HGB v1-features val AUROC was 0.8653)")
candidates["HGB_v2"] = (best_hgb_auc, best_hgb_params, "hgb")

log("\n--- Candidate B: RandomForestClassifier (median-imputed), expanded features ---")
rf_grid = [
    dict(n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt"),
    dict(n_estimators=600, max_depth=12, min_samples_leaf=3, max_features="sqrt"),
    dict(n_estimators=600, max_depth=None, min_samples_leaf=5, max_features=0.3),
]
best_rf_auc, best_rf_params = -1, None
for params in rf_grid:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                       class_weight="balanced_subsample", **params)),
    ])
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    log(f"  {params} -> val AUROC={auc:.4f}")
    if auc > best_rf_auc:
        best_rf_auc, best_rf_params = auc, params
log(f"Best RandomForest (v2 features): {best_rf_params} -> val AUROC={best_rf_auc:.4f}")
candidates["RF_v2"] = (best_rf_auc, best_rf_params, "rf")

log("\n--- Candidate C: logistic regression (median-imputed, standardized), expanded features ---")
best_lr_auc, best_lr_c = -1, None
for C in [0.003, 0.01, 0.03, 0.1, 0.3]:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    log(f"  C={C} -> val AUROC={auc:.4f}")
    if auc > best_lr_auc:
        best_lr_auc, best_lr_c = auc, C
log(f"Best logistic regression (v2 features): C={best_lr_c} -> val AUROC={best_lr_auc:.4f} "
    f"(cycle-1 LR v1-features val AUROC was 0.8595)")
candidates["LR_v2"] = (best_lr_auc, best_lr_c, "lr")

log("\n--- Candidate D: probability-average blend of best HGB_v2 and best LR_v2 ---")
lr_pipe_train = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(C=best_lr_c, max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)),
])
lr_pipe_train.fit(X_train, y_train)
p_val_hgb = best_hgb_model.predict_proba(X_val)[:, 1]
p_val_lr = lr_pipe_train.predict_proba(X_val)[:, 1]
p_val_blend = 0.5 * p_val_hgb + 0.5 * p_val_lr
blend_auc = roc_auc_score(y_val, p_val_blend)
log(f"  0.5*HGB_v2 + 0.5*LR_v2 -> val AUROC={blend_auc:.4f}")
candidates["Blend_v2"] = (blend_auc, {"hgb": best_hgb_params, "lr_C": best_lr_c}, "blend")

log("\n--- Candidate summary (val AUROC, train-only fits) ---")
for name, (auc, _, _) in sorted(candidates.items(), key=lambda kv: -kv[1][0]):
    log(f"  {name}: {auc:.4f}")
winner_name = max(candidates, key=lambda k: candidates[k][0])
log(f"Winner (highest val AUROC): {winner_name}")

# ---------------------------------------------------------------------------
# 3. Calibration check for the winning model family (train-fit, val-scored)
#    CalibratedClassifierCV's internal 5-fold CV uses TRAIN only here, so VAL
#    remains an independent check of whether calibration helps.
# ---------------------------------------------------------------------------
log("\n--- Calibration check (winning family, cv folds from train only, scored on val) ---")


def make_base_estimator(name):
    if candidates[name][2] == "hgb":
        p = candidates[name][1]
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True,
                                               validation_fraction=0.15,
                                               class_weight="balanced", **p)
    if candidates[name][2] == "rf":
        p = candidates[name][1]
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                           class_weight="balanced_subsample", **p)),
        ])
    if candidates[name][2] == "lr":
        C = candidates[name][1]
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ])
    raise ValueError("Blend is not calibrated as a single estimator in this cycle.")


use_calibration = False
if candidates[winner_name][2] in ("hgb", "rf", "lr"):
    base = make_base_estimator(winner_name)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
    cal.fit(X_train, y_train)
    p_val_uncal = candidates[winner_name][0]  # already computed AUROC above (train-only fit)
    p_val_cal = cal.predict_proba(X_val)[:, 1]
    auc_cal = roc_auc_score(y_val, p_val_cal)
    brier_cal = brier_score_loss(y_val, p_val_cal)
    if winner_name == "HGB_v2":
        brier_uncal = brier_score_loss(y_val, p_val_hgb)
    elif winner_name == "LR_v2":
        brier_uncal = brier_score_loss(y_val, p_val_lr)
    else:
        pipe_check = make_base_estimator(winner_name)
        pipe_check.fit(X_train, y_train)
        brier_uncal = brier_score_loss(y_val, pipe_check.predict_proba(X_val)[:, 1])
    log(f"  Uncalibrated {winner_name}: val AUROC={p_val_uncal:.4f}, val Brier={brier_uncal:.4f}")
    log(f"  Isotonic-calibrated {winner_name}: val AUROC={auc_cal:.4f}, val Brier={brier_cal:.4f}")
    # Calibration should preserve (not hurt) discrimination while improving Brier.
    if brier_cal < brier_uncal - 1e-4 and auc_cal > p_val_uncal - 0.005:
        use_calibration = True
        log("  -> Calibration improves Brier without meaningfully hurting AUROC; "
            "will deploy the calibrated version.")
    else:
        log("  -> Calibration did not clearly help on val; deploying the uncalibrated model.")
else:
    log("  Winner is the blend, which is not wrapped in CalibratedClassifierCV this cycle.")

# ---------------------------------------------------------------------------
# 4. Refit the winning configuration on train+val for deployment
# ---------------------------------------------------------------------------
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

if winner_name == "Blend_v2":
    final_hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True,
                                                validation_fraction=0.15,
                                                class_weight="balanced", **best_hgb_params)
    final_hgb.fit(X_trainval, y_trainval)
    final_lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=best_lr_c, max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    final_lr.fit(X_trainval, y_trainval)
    final_model = {"kind": "blend", "hgb": final_hgb, "lr": final_lr, "weights": (0.5, 0.5)}
    model_type_str = "Blend(HGB_v2, LR_v2)"
elif use_calibration:
    base = make_base_estimator(winner_name)
    final_model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    final_model.fit(X_trainval, y_trainval)
    model_type_str = f"CalibratedClassifierCV(isotonic, cv=5) wrapping {winner_name}"
else:
    final_model = make_base_estimator(winner_name)
    final_model.fit(X_trainval, y_trainval)
    model_type_str = winner_name

log(f"\nFinal deployed model: {model_type_str}, refit on train+val (n={len(X_trainval)})")


def predict_proba_final(X):
    if isinstance(final_model, dict) and final_model.get("kind") == "blend":
        w_hgb, w_lr = final_model["weights"]
        return (w_hgb * final_model["hgb"].predict_proba(X)[:, 1]
                + w_lr * final_model["lr"].predict_proba(X)[:, 1])
    return final_model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 5. Decision threshold chosen on VAL only, before touching test
# ---------------------------------------------------------------------------
p_val_final = predict_proba_final(X_val)
fpr, tpr, thr = roc_curve(y_val, p_val_final)
chosen_threshold = thr[np.argmax(tpr - fpr)]
log(f"Decision threshold chosen on val (Youden's J): {chosen_threshold:.4f}")

# ---------------------------------------------------------------------------
# 6. SINGLE final evaluation on the held-out test set (touched exactly once)
# ---------------------------------------------------------------------------
log("\n=== FINAL TEST EVALUATION, CYCLE 2 (single pass) ===")
p_test = predict_proba_final(X_test)
test_auroc = roc_auc_score(y_test, p_test)
test_auprc = average_precision_score(y_test, p_test)
test_brier = brier_score_loss(y_test, p_test)
pred_label = (p_test >= chosen_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_label).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) else float("nan")
npv = tn / (tn + fn) if (tn + fn) else float("nan")

log(f"Test AUROC:  {test_auroc:.4f}  (cycle 1 was 0.8527)")
log(f"Test AUPRC:  {test_auprc:.4f}  (cycle 1 was 0.5019; base rate {y_test.mean():.4f})")
log(f"Test Brier:  {test_brier:.4f}  (cycle 1 was 0.1312)")
log(f"At threshold {chosen_threshold:.4f}: sensitivity={sens:.3f}, "
    f"specificity={spec:.3f}, PPV={ppv:.3f}, NPV={npv:.3f}")
log(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

cal_df = pd.DataFrame({"y": y_test.to_numpy(), "p": p_test})
cal_df["bin"] = pd.qcut(cal_df["p"], 10, duplicates="drop")
cal_table = cal_df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"),
                                                       obs_rate=("y", "mean"),
                                                       n=("y", "size"))
log("\nCalibration by predicted-risk decile (test set, cycle 2):")
log(cal_table.to_string())

# ---------------------------------------------------------------------------
# 7. Interpretability: permutation importance on test set (single pass)
# ---------------------------------------------------------------------------
log("\n--- Permutation importance (final model, test set, single pass) ---")


class _FinalModelWrapper:
    """Thin sklearn-estimator-like wrapper so permutation_importance can
    call predict_proba uniformly whether the final model is a blend dict or
    a plain fitted estimator/pipeline."""

    _estimator_type = "classifier"
    classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = predict_proba_final(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (predict_proba_final(X) >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {}

    def __sklearn_tags__(self):
        from sklearn.utils import Tags, ClassifierTags, TargetTags, InputTags
        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True),
            classifier_tags=ClassifierTags(),
            input_tags=InputTags(allow_nan=True),
        )


wrapper = _FinalModelWrapper()
perm = permutation_importance(
    wrapper, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
    scoring="roc_auc", n_jobs=1,
)
imp_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
imp_df.to_csv("cycle2_importance.csv", index=False)
log(imp_df.head(25).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Persist model bundle for predict_cycle2.py
# ---------------------------------------------------------------------------
bundle = {
    "model": final_model,
    "feature_columns": feature_columns,
    "threshold": float(chosen_threshold),
    "model_type": model_type_str,
    "trained_on": "train+val (cycle2, same patients as cycle1)",
    "feature_builder": "build_feature_matrix_v2",
}
with open("model_cycle2.pkl", "wb") as f:
    pickle.dump(bundle, f)

with open("cycle2_metrics.txt", "w") as f:
    f.write("\n".join(log_lines))

log("\nSaved model_cycle2.pkl, split_cycle2.csv, cycle2_metrics.txt, cycle2_importance.csv")
