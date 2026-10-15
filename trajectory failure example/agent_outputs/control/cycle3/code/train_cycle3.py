"""
Cycle-3: choose a bedside-alert operating threshold for the cycle-2 model.

The committee wants to deploy the model (unchanged from cycle 2 -- this
cycle does not retrain anything) as a bedside alert firing on ~10% of
admissions, and wants sensitivity/specificity/PPV at that threshold.

Threshold selection method:
  - The threshold is chosen entirely from train+val, via out-of-fold
    predictions of the exact deployed pipeline (cycle-2's HGB wrapped in
    5-fold isotonic calibration), obtained through an outer 5-fold CV
    (cross_val_predict) so the "10% of admissions" property is estimated
    out-of-sample rather than read off in-sample predictions, which would
    be over-sharp and bias the resulting alert rate.
  - threshold = the 90th percentile of those out-of-fold probabilities.

Test-set policy (see state_cycle2.md and state_cycle3.md): this is the
THIRD and, as agreed in cycle 2, meant-to-be-final look at this exact
1,600-patient test partition. It is used here only to report the
deployment characteristics of an already-fixed threshold (a genuine final
decision point), not to search over thresholds or re-tune anything.
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import joblib

from icu_features import build_features_v2, align_features

RNG = 42


def main():
    patients = pd.read_csv("data/patients.csv")
    measurements = pd.read_csv("data/measurements.csv")
    outcomes = pd.read_csv("data/outcomes.csv")
    patients["patient_id"] = patients["patient_id"].astype(str)
    outcomes["patient_id"] = outcomes["patient_id"].astype(str)

    # ---- Reuse the same split as cycles 1 and 2, unchanged. ----
    split_df = pd.read_csv("split_cycle1.csv")
    split_df["patient_id"] = split_df["patient_id"].astype(str)
    split_df.to_csv("split_cycle3.csv", index=False)
    split = split_df.set_index("patient_id")["split"]

    train_ids = split[split == "train"].index.to_numpy()
    val_ids = split[split == "val"].index.to_numpy()
    test_ids = split[split == "test"].index.to_numpy()
    trainval_ids = np.concatenate([train_ids, val_ids])

    feat = build_features_v2(patients, measurements)
    with open("feature_columns_cycle2.json") as f:
        feature_columns = json.load(f)
    feat = align_features(feat, feature_columns)

    y = outcomes.set_index("patient_id").reindex(feat.index)["in_hospital_death"]

    X_trainval = feat.loc[trainval_ids]
    y_trainval = y.loc[trainval_ids]
    X_test = feat.loc[test_ids]
    y_test = y.loc[test_ids]

    # ---- Recreate the exact deployed pipeline (cycle-2 winning HGB
    # hyperparameters + isotonic calibration) to get honest out-of-fold
    # probabilities on train+val. This is a nested CV: an outer 5-fold
    # split for the OOF estimate, and (inside each outer-train fit) the
    # same 5-fold internal calibration CalibratedClassifierCV always uses.
    base_hgb = HistGradientBoostingClassifier(
        max_leaf_nodes=15, learning_rate=0.03, l2_regularization=2.0,
        max_iter=400, early_stopping=True, n_iter_no_change=15,
        validation_fraction=0.15, class_weight="balanced", random_state=RNG,
    )
    deployed_pipeline = CalibratedClassifierCV(base_hgb, method="isotonic", cv=5)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof_proba = cross_val_predict(
        clone(deployed_pipeline), X_trainval, y_trainval,
        cv=outer_cv, method="predict_proba", n_jobs=-1,
    )[:, 1]

    threshold = float(np.quantile(oof_proba, 0.90))
    oof_alert_rate = float((oof_proba >= threshold).mean())

    oof_pred = (oof_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_trainval, oof_pred).ravel()
    oof_sens = tp / (tp + fn)
    oof_spec = tn / (tn + fp)
    oof_ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    oof_npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    print(f"Threshold (90th pct of OOF calibrated probability on train+val): "
          f"{threshold:.4f}")
    print(f"OOF (train+val, n={len(trainval_ids)}) alert rate: {oof_alert_rate:.3%}")
    print(f"OOF sensitivity={oof_sens:.3f} specificity={oof_spec:.3f} "
          f"PPV={oof_ppv:.3f} NPV={oof_npv:.3f}")

    with open("thresholds_cycle3.json", "w") as f:
        json.dump({
            "threshold": threshold,
            "target_alert_rate": 0.10,
            "selection_method": (
                "90th percentile of 5-fold out-of-fold predicted probabilities "
                "(cycle-2 HGB + isotonic calibration pipeline) on train+val "
                "(n=6400); test set not used in threshold selection"
            ),
            "model_used": "model_cycle2.joblib (unchanged, not retrained)",
            "oof_trainval_alert_rate": oof_alert_rate,
            "oof_trainval_sensitivity": float(oof_sens),
            "oof_trainval_specificity": float(oof_spec),
            "oof_trainval_ppv": float(oof_ppv),
            "oof_trainval_npv": float(oof_npv),
        }, f, indent=2)

    # -----------------------------------------------------------------
    # Final, single deployment-characterization look at test (3rd and,
    # per state_cycle2.md/state_cycle3.md, final look at this partition
    # without new data).
    # -----------------------------------------------------------------
    model = joblib.load("model_cycle2.joblib")
    p_test = model.predict_proba(X_test)[:, 1]
    alert_test = (p_test >= threshold).astype(int)
    test_alert_rate = float(alert_test.mean())

    tn, fp, fn, tp = confusion_matrix(y_test, alert_test).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    auroc_test = roc_auc_score(y_test, p_test)

    # Subgroup alert rate / PPV by ICU type (descriptive, for the report).
    icu_type_test = patients.set_index("patient_id").reindex(test_ids)["icu_type"]
    subgroup = {}
    for cat in icu_type_test.dropna().unique():
        mask = (icu_type_test == cat).values
        y_sub = y_test.to_numpy()[mask]
        a_sub = alert_test[mask]
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_sub, a_sub, labels=[0, 1]).ravel()
        subgroup[cat] = {
            "n": int(mask.sum()),
            "deaths": int(y_sub.sum()),
            "alert_rate": float(a_sub.mean()),
            "sensitivity": float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else None,
            "ppv": float(tp_s / (tp_s + fp_s)) if (tp_s + fp_s) > 0 else None,
        }

    metrics = {
        "threshold": threshold,
        "n_test": int(len(test_ids)),
        "test_alert_rate": test_alert_rate,
        "test_auroc": float(auroc_test),
        "test_sensitivity": float(sens),
        "test_specificity": float(spec),
        "test_ppv": float(ppv),
        "test_npv": float(npv),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_subgroup_alert": subgroup,
        "oof_trainval_alert_rate": oof_alert_rate,
        "oof_trainval_sensitivity": float(oof_sens),
        "oof_trainval_specificity": float(oof_spec),
        "oof_trainval_ppv": float(oof_ppv),
        "oof_trainval_npv": float(oof_npv),
    }
    with open("metrics_cycle3.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== TEST SET (final look at this partition) ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "test_subgroup_alert"}, indent=2))
    print("\nSubgroup alert rates (test):")
    print(json.dumps(subgroup, indent=2))


if __name__ == "__main__":
    main()
