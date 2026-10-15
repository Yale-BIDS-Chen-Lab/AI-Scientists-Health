"""
Cycle-3: operating-threshold selection for bedside-alert deployment.

No new model is fit this cycle. The committee asked for a single operating
threshold on the cycle-2 model's probability such that roughly 10% of
admissions trigger an alert, plus sensitivity/specificity/PPV at that
threshold. This script only chooses and checks that threshold.

Test-set discipline (unchanged from cycles 1-2): split_cycle2.csv is reused
verbatim as split_cycle3.csv -- the test set is never redrawn. The threshold
is chosen entirely from VAL and is scored against TEST exactly once.

Subtlety specific to this cycle: the deployed cycle-2 model (in
model_cycle2.pkl) was refit on train+val, so val is no longer a clean
held-out set FOR THAT MODEL -- its own predictions on val would be mildly
optimistic/overconfident (in-sample) relative to how it scores genuinely new
patients, which would bias a quantile-based threshold picked directly from
those predictions. To avoid that bias, this script refits fresh copies of
the two cycle-2 sub-models (same hyperparameters, same 0.5/0.5 blend) on
TRAIN ONLY, and uses THEIR predictions on VAL (genuinely out-of-sample for
these copies) to locate the 90th percentile. That threshold is then applied,
unchanged, to the actual deployed model's (train+val-fit) predictions on
TEST for the single, final reported evaluation.
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
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from icu_features import build_feature_matrix_v2, align_features

RANDOM_STATE = 42
TARGET_ALERT_RATE = 0.10
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# Hyperparameters selected in cycle 2 (see cycle2_metrics.txt) -- unchanged
# here; cycle 3 does not re-tune anything, only picks a threshold.
HGB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.0,
                   min_samples_leaf=20)
LR_C = 0.01
BLEND_WEIGHTS = (0.5, 0.5)

# ---------------------------------------------------------------------------
# 1. Load data; reuse the cycle-1/cycle-2 split verbatim (never redrawn)
# ---------------------------------------------------------------------------
patients = pd.read_csv("data/patients.csv")
measurements = pd.read_csv("data/measurements.csv")
outcomes = pd.read_csv("data/outcomes.csv").set_index("patient_id")["in_hospital_death"]

split = pd.read_csv("split_cycle2.csv").set_index("patient_id")["split"]
split.rename("split").rename_axis("patient_id").reset_index().to_csv(
    "split_cycle3.csv", index=False
)
log("Reused split_cycle2.csv assignment verbatim as split_cycle3.csv "
    "(same train/val/test patients as cycles 1-2).")

feats = build_feature_matrix_v2(patients, measurements)
y = outcomes.reindex(feats.index)

train_ids = split[split == "train"].index
val_ids = split[split == "val"].index
test_ids = split[split == "test"].index

X_train, y_train = feats.loc[train_ids], y.loc[train_ids]
X_val, y_val = feats.loc[val_ids], y.loc[val_ids]
X_test, y_test = feats.loc[test_ids], y.loc[test_ids]
log(f"train n={len(train_ids)}, val n={len(val_ids)}, test n={len(test_ids)}")

# ---------------------------------------------------------------------------
# 2. Refit train-only copies of the cycle-2 sub-models, purely to get a
#    genuinely out-of-sample VAL score distribution for threshold-setting.
#    (No model-selection decision is made here -- hyperparameters are fixed
#    from cycle 2.)
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

# ---------------------------------------------------------------------------
# 3. Threshold = the value above which ~10% of VAL admissions fall, using
#    the out-of-sample VAL scores above.
# ---------------------------------------------------------------------------
threshold = float(np.quantile(p_val_oos, 1 - TARGET_ALERT_RATE))
val_alert_rate = float((p_val_oos >= threshold).mean())
log(f"\nThreshold chosen from out-of-sample VAL scores (90th percentile): {threshold:.4f}")
log(f"Resulting VAL alert rate at this threshold: {val_alert_rate:.4f} "
    f"(target {TARGET_ALERT_RATE:.2f})")

# ---------------------------------------------------------------------------
# 4. Apply this threshold to the ACTUAL DEPLOYED model (train+val-fit, from
#    model_cycle2.pkl -- unchanged, not refit here) on TEST. Single pass.
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
log("\n=== FINAL TEST EVALUATION, CYCLE 3 (single pass, deployed model, chosen threshold) ===")
p_test = predict_proba_deployed(X_test_aligned)
test_alert_rate = float((p_test >= threshold).mean())
pred_label = (p_test >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_label).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) else float("nan")
npv = tn / (tn + fn) if (tn + fn) else float("nan")
test_auroc = roc_auc_score(y_test, p_test)
test_auprc = average_precision_score(y_test, p_test)

log(f"Threshold: {threshold:.4f}")
log(f"Test alert rate: {test_alert_rate:.4f} (target ~0.10; n alerted = {tp + fp} / {len(y_test)})")
log(f"Sensitivity: {sens:.3f}")
log(f"Specificity: {spec:.3f}")
log(f"PPV:         {ppv:.3f}")
log(f"NPV:         {npv:.3f}")
log(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
log(f"(For reference, unchanged from cycle 2: test AUROC={test_auroc:.4f}, "
    f"test AUPRC={test_auprc:.4f} -- the underlying model and its ranking "
    f"performance are identical to cycle 2; only the operating threshold is new.)")

# ---------------------------------------------------------------------------
# 5. Persist threshold + a copy of the metrics
# ---------------------------------------------------------------------------
threshold_record = {
    "threshold": threshold,
    "target_alert_rate": TARGET_ALERT_RATE,
    "val_alert_rate_at_threshold": val_alert_rate,
    "test_alert_rate_at_threshold": test_alert_rate,
    "test_sensitivity": sens,
    "test_specificity": spec,
    "test_ppv": ppv,
    "test_npv": npv,
    "selected_on": "val (out-of-sample scores from train-only refit of cycle-2 sub-models)",
    "applied_to": "model_cycle2.pkl (deployed, train+val-fit) predictions",
    "model_source": "model_cycle2.pkl",
}
with open("thresholds_cycle3.json", "w") as f:
    json.dump(threshold_record, f, indent=2)

with open("cycle3_metrics.txt", "w") as f:
    f.write("\n".join(log_lines))

log("\nSaved thresholds_cycle3.json, split_cycle3.csv, cycle3_metrics.txt")
