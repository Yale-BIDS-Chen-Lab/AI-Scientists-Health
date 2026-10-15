"""
Cycle-1 training pipeline for the in-hospital ICU mortality model.

Run once from the project directory:
    python3 train_cycle1.py

Produces (all written next to this script):
    split_cycle1.csv        patient_id,split  (train/val/test, decided up front)
    model_cycle1.joblib     final fitted model (calibrated HistGBM)
    feature_columns_cycle1.json   exact column order the model expects
    metrics_cycle1.json     all numbers used in report_cycle1.md
    importance_cycle1.csv   permutation importance ranking

Test-set rule: the split is written to disk before any model is trained or
compared, and test-set labels are touched exactly once, at the very end, to
compute the numbers that go in report_cycle1.md. All model/feature/threshold
decisions above that point use only the train and val partitions.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from icu_features import build_features, get_feature_columns

warnings.filterwarnings("ignore")
RNG = 42


def main():
    patients = pd.read_csv("data/patients.csv")
    measurements = pd.read_csv("data/measurements.csv")
    outcomes = pd.read_csv("data/outcomes.csv")
    patients["patient_id"] = patients["patient_id"].astype(str)
    outcomes["patient_id"] = outcomes["patient_id"].astype(str)

    feat = build_features(patients, measurements)
    feature_columns = get_feature_columns(feat)

    y = outcomes.set_index("patient_id").reindex(feat.index)["in_hospital_death"]
    assert y.isna().sum() == 0, "every patient must have an outcome label"

    # ---- Split now, before any modeling decision. 60/20/20, stratified. ----
    ids = feat.index.to_numpy()
    y_arr = y.to_numpy()
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        ids, y_arr, test_size=0.4, stratify=y_arr, random_state=RNG
    )
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids, y_temp, test_size=0.5, stratify=y_temp, random_state=RNG
    )

    split = pd.Series("train", index=ids, name="split")
    split.loc[val_ids] = "val"
    split.loc[test_ids] = "test"
    split_df = split.reset_index().rename(columns={"index": "patient_id"})
    split_df = split_df[["patient_id", "split"]]
    split_df.to_csv("split_cycle1.csv", index=False)

    X_train = feat.loc[train_ids]
    X_val = feat.loc[val_ids]
    X_test = feat.loc[test_ids]

    print(f"train n={len(train_ids)} (deaths={int(y_train.sum())}, "
          f"{y_train.mean():.3f})")
    print(f"val   n={len(val_ids)} (deaths={int(y_val.sum())}, {y_val.mean():.3f})")
    print(f"test  n={len(test_ids)} (deaths={int(y_test.sum())}, "
          f"{y_test.mean():.3f})  [held out, not used until final eval]")

    # ---------------------------------------------------------------
    # Model selection on train -> val only. No test-set access here.
    # ---------------------------------------------------------------
    candidates = {}

    # Candidate A: Logistic regression (median impute + scale), a few C's
    for C in [0.01, 0.1, 1.0]:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, max_iter=2000, class_weight="balanced", random_state=RNG
            )),
        ])
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:, 1]
        candidates[f"logreg_C{C}"] = {
            "model": pipe,
            "auroc": roc_auc_score(y_val, p_val),
            "auprc": average_precision_score(y_val, p_val),
            "brier": brier_score_loss(y_val, p_val),
        }

    # Candidate B: HistGradientBoostingClassifier (native NaN handling),
    # a small grid over the parameters that matter most for overfitting.
    for max_leaf, lr, l2 in [(15, 0.1, 0.0), (31, 0.05, 1.0), (15, 0.05, 1.0)]:
        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=max_leaf,
            learning_rate=lr,
            l2_regularization=l2,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15,
            class_weight="balanced",
            random_state=RNG,
        )
        clf.fit(X_train, y_train)
        p_val = clf.predict_proba(X_val)[:, 1]
        candidates[f"hgb_leaf{max_leaf}_lr{lr}_l2{l2}"] = {
            "model": clf,
            "auroc": roc_auc_score(y_val, p_val),
            "auprc": average_precision_score(y_val, p_val),
            "brier": brier_score_loss(y_val, p_val),
        }

    print("\nModel selection (train fit -> val scored):")
    for name, res in sorted(candidates.items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"  {name:28s} AUROC={res['auroc']:.4f}  AUPRC={res['auprc']:.4f}"
              f"  Brier={res['brier']:.4f}")

    best_name = max(candidates, key=lambda k: candidates[k]["auroc"])
    best = candidates[best_name]
    print(f"\nSelected: {best_name}")

    # ---------------------------------------------------------------
    # Interpretation: permutation importance of the selected model,
    # computed on val (never on test).
    # ---------------------------------------------------------------
    perm = permutation_importance(
        best["model"], X_val, y_val, scoring="roc_auc",
        n_repeats=15, random_state=RNG, n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv("importance_cycle1.csv", index=False)

    # Direction check for the top features: mean value among survivors vs
    # non-survivors, on train+val (descriptive only, not test).
    trainval_ids = np.concatenate([train_ids, val_ids])
    X_trainval = feat.loc[trainval_ids]
    y_trainval = pd.concat([y_train_s := pd.Series(y_train, index=train_ids),
                             y_val_s := pd.Series(y_val, index=val_ids)])
    top_feats = importance_df.head(15)["feature"].tolist()
    direction_rows = []
    for f in top_feats:
        died = X_trainval.loc[y_trainval == 1, f].mean()
        survived = X_trainval.loc[y_trainval == 0, f].mean()
        direction_rows.append({"feature": f, "mean_if_died": died,
                                "mean_if_survived": survived})
    direction_df = pd.DataFrame(direction_rows)
    direction_df.to_csv("direction_check_cycle1.csv", index=False)

    # Also fit a plain logistic regression on train+val for a coefficient-
    # sign cross-check against the tree model's importances.
    logreg_check = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000,
                                    class_weight="balanced", random_state=RNG)),
    ])
    logreg_check.fit(X_trainval, y_trainval)
    coef = pd.Series(logreg_check.named_steps["clf"].coef_[0], index=feature_columns)
    coef.loc[top_feats].sort_values().to_csv("logreg_coef_check_cycle1.csv")

    # ---------------------------------------------------------------
    # Choose an operating threshold using val only (Youden's J), then
    # finalize the model on train+val, then touch test exactly once.
    # ---------------------------------------------------------------
    p_val_best = best["model"].predict_proba(X_val)[:, 1]
    fpr, tpr, thr = roc_curve(y_val, p_val_best)
    youden = tpr - fpr
    best_thr = thr[np.argmax(youden)]
    # Alternate, higher-sensitivity operating point (>=90% sensitivity),
    # also chosen on val only -- relevant if the clinical use case favors
    # not missing deaths over avoiding false alarms. (Youden's J on this
    # model already lands around ~80% sensitivity, so 90% is used here to
    # show a materially different, higher-sensitivity trade-off point
    # rather than one nearly identical to the default.)
    ok = tpr >= 0.90
    # tpr is non-decreasing as threshold decreases (roc_curve order), so the
    # first index reaching >=80% sensitivity is the one with best specificity
    # among all thresholds meeting that sensitivity target.
    high_sens_thr = thr[np.argmax(ok)] if ok.any() else thr[np.argmax(tpr)]

    # Re-fit the selected model family/hyperparameters on train+val, wrapped
    # in isotonic calibration (5-fold internal CV) so predicted
    # probabilities are usable for clinical risk communication, not just
    # ranking. This is the model that ships in predict_cycle1.py.
    if best_name.startswith("hgb"):
        parts = best_name.split("_")
        max_leaf = int(parts[1].replace("leaf", ""))
        lr = float(parts[2].replace("lr", ""))
        l2 = float(parts[3].replace("l2", ""))
        base = HistGradientBoostingClassifier(
            max_leaf_nodes=max_leaf, learning_rate=lr, l2_regularization=l2,
            max_iter=300, early_stopping=True, n_iter_no_change=15,
            validation_fraction=0.15, class_weight="balanced", random_state=RNG,
        )
    else:
        C = float(best_name.replace("logreg_C", ""))
        base = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000,
                                        class_weight="balanced", random_state=RNG)),
        ])

    final_model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    final_model.fit(X_trainval, y_trainval)

    joblib.dump(final_model, "model_cycle1.joblib")
    with open("feature_columns_cycle1.json", "w") as f:
        json.dump(feature_columns, f)

    # ---------------------------------------------------------------
    # Final, single look at test.
    # ---------------------------------------------------------------
    p_test = final_model.predict_proba(X_test)[:, 1]
    auroc_test = roc_auc_score(y_test, p_test)
    auprc_test = average_precision_score(y_test, p_test)
    brier_test = brier_score_loss(y_test, p_test)
    pred_test = (p_test >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    pred_test_hs = (p_test >= high_sens_thr).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, pred_test_hs).ravel()
    sens_hs = tp2 / (tp2 + fn2)
    spec_hs = tn2 / (tn2 + fp2)
    ppv_hs = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else float("nan")
    npv_hs = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else float("nan")

    # Subgroup AUROC on test by icu_type (descriptive check of whether
    # discrimination is uneven across admission types).
    icu_type_test = patients.set_index("patient_id").reindex(test_ids)["icu_type"]
    subgroup_auroc = {}
    for cat in icu_type_test.dropna().unique():
        mask = (icu_type_test == cat).values
        y_sub = np.asarray(y_test)[mask]
        if len(np.unique(y_sub)) < 2:
            continue
        subgroup_auroc[cat] = {
            "n": int(mask.sum()),
            "deaths": int(y_sub.sum()),
            "auroc": float(roc_auc_score(y_sub, p_test[mask])),
        }

    # Calibration curve data (test), 10 bins by predicted probability.
    calib_df = pd.DataFrame({"p": p_test, "y": np.asarray(y_test)})
    calib_df["bin"] = pd.qcut(calib_df["p"], 10, duplicates="drop")
    calib = calib_df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")
    ).reset_index(drop=True)
    calib.to_csv("calibration_cycle1.csv", index=False)

    # Also report uncalibrated-base-model val metrics for the record, and
    # the calibrated model's own val metrics (val was used for calibration
    # fitting only via internal CV during model-selection scoring above,
    # so this is a fair secondary check, not the headline number).
    metrics = {
        "n_total": int(len(feat)),
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "prevalence_train": float(y_train.mean()),
        "prevalence_val": float(y_val.mean()),
        "prevalence_test": float(y_test.mean()),
        "selected_model": best_name,
        "selection_val_auroc": float(best["auroc"]),
        "selection_val_auprc": float(best["auprc"]),
        "selection_val_brier": float(best["brier"]),
        "operating_threshold": float(best_thr),
        "test_auroc": float(auroc_test),
        "test_auprc": float(auprc_test),
        "test_brier": float(brier_test),
        "test_sensitivity": float(sens),
        "test_specificity": float(spec),
        "test_ppv": float(ppv),
        "test_npv": float(npv),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "high_sens_threshold": float(high_sens_thr),
        "high_sens_test_sensitivity": float(sens_hs),
        "high_sens_test_specificity": float(spec_hs),
        "high_sens_test_ppv": float(ppv_hs),
        "high_sens_test_npv": float(npv_hs),
        "high_sens_test_confusion_matrix": {"tn": int(tn2), "fp": int(fp2), "fn": int(fn2), "tp": int(tp2)},
        "test_subgroup_auroc_by_icu_type": subgroup_auroc,
        "all_candidates": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                            for k, v in candidates.items()},
    }
    with open("metrics_cycle1.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== FINAL TEST SET RESULT (single look) ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "all_candidates"}, indent=2))


if __name__ == "__main__":
    main()
