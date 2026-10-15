"""
Cycle-2 training pipeline for the in-hospital ICU mortality model.

The clinical committee found cycle 1's performance (test AUROC 0.868)
insufficient and asked whether feature engineering or a different model
class could improve it. This script:

  1. Reuses the EXACT patient-level split from cycle 1 (split_cycle1.csv)
     instead of re-splitting. Reshuffling patients between train/val/test
     between cycles would leak cycle-1 modeling decisions (which were
     validated against the cycle-1 val set) into what would become a
     "new" test set, and vice versa -- the only leak-free way to reuse a
     fixed, unrepeated dataset across cycles is to keep the partition
     fixed and only extend what happens within train/val.
  2. Adds trajectory-aware features (first value, per-variable trend
     slope, early/late-window means, a few composite clinical ratios) on
     top of cycle 1's count/mean/min/max/last features.
  3. Compares more model classes (random forest, extra trees, gradient
     boosting, HGB, logistic regression) using 5-fold stratified
     cross-validation on train+val (not a single val split, which cycle 1
     flagged as noisy for threshold selection) -- test is not touched.
  4. Runs an ablation isolating the feature-engineering contribution from
     the model-class contribution: cycle-1 features vs. cycle-2 features,
     both scored with cycle-1's winning HGB hyperparameters, via the same
     CV.
  5. Finalizes the winning configuration (calibrated, refit on train+val)
     and evaluates it on the SAME test set used in cycle 1 -- this is
     that test set's second and, as documented in state_cycle2.md, meant
     to be final look for this modeling track.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from icu_features import build_features, build_features_v2, get_feature_columns, align_features

warnings.filterwarnings("ignore")
RNG = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)


def main():
    patients = pd.read_csv("data/patients.csv")
    measurements = pd.read_csv("data/measurements.csv")
    outcomes = pd.read_csv("data/outcomes.csv")
    patients["patient_id"] = patients["patient_id"].astype(str)
    outcomes["patient_id"] = outcomes["patient_id"].astype(str)

    # ---- Reuse cycle 1's split exactly. ----
    split1_df = pd.read_csv("split_cycle1.csv")
    split1_df["patient_id"] = split1_df["patient_id"].astype(str)
    split1_df.to_csv("split_cycle2.csv", index=False)  # identical, on purpose
    split1 = split1_df.set_index("patient_id")["split"]

    feat_v2 = build_features_v2(patients, measurements)
    feat_v1 = build_features(patients, measurements)
    feature_columns_v2 = get_feature_columns(feat_v2)

    y = outcomes.set_index("patient_id").reindex(feat_v2.index)["in_hospital_death"]
    assert y.isna().sum() == 0

    train_ids = split1[split1 == "train"].index.to_numpy()
    val_ids = split1[split1 == "val"].index.to_numpy()
    test_ids = split1[split1 == "test"].index.to_numpy()
    trainval_ids = np.concatenate([train_ids, val_ids])

    X_trainval_v2 = feat_v2.loc[trainval_ids]
    y_trainval = y.loc[trainval_ids]
    X_test_v2 = feat_v2.loc[test_ids]
    y_test = y.loc[test_ids]

    print(f"train+val n={len(trainval_ids)} (deaths={int(y_trainval.sum())}, "
          f"{y_trainval.mean():.3f})")
    print(f"test      n={len(test_ids)} (deaths={int(y_test.sum())}, "
          f"{y_test.mean():.3f})  [same partition as cycle 1, touched once below]")
    print(f"cycle-2 feature count: {feat_v2.shape[1]} (cycle-1 was {feat_v1.shape[1]})")

    # -----------------------------------------------------------------
    # Ablation: cycle-1's winning HGB hyperparameters, cycle-1 features
    # vs. cycle-2 features, both via 5-fold CV on train+val.
    # -----------------------------------------------------------------
    cycle1_hgb = HistGradientBoostingClassifier(
        max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0,
        max_iter=300, early_stopping=True, n_iter_no_change=15,
        validation_fraction=0.15, class_weight="balanced", random_state=RNG,
    )
    X_trainval_v1 = feat_v1.loc[trainval_ids]
    auroc_v1_feats = cross_val_score(cycle1_hgb, X_trainval_v1, y_trainval,
                                      cv=CV, scoring="roc_auc", n_jobs=-1)
    auroc_v2_feats = cross_val_score(cycle1_hgb, X_trainval_v2, y_trainval,
                                      cv=CV, scoring="roc_auc", n_jobs=-1)
    print("\nAblation (same HGB hyperparams, 5-fold CV on train+val):")
    print(f"  cycle-1 features (n={feat_v1.shape[1]}): "
          f"AUROC {auroc_v1_feats.mean():.4f} +/- {auroc_v1_feats.std():.4f}")
    print(f"  cycle-2 features (n={feat_v2.shape[1]}): "
          f"AUROC {auroc_v2_feats.mean():.4f} +/- {auroc_v2_feats.std():.4f}")

    # -----------------------------------------------------------------
    # Model-class comparison on cycle-2 features, 5-fold CV, train+val only.
    # -----------------------------------------------------------------
    candidates = {}

    for C in [0.01, 0.1, 1.0]:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000,
                                        class_weight="balanced", random_state=RNG)),
        ])
        candidates[f"logreg_C{C}"] = pipe

    for max_leaf, lr, l2 in [(15, 0.05, 1.0), (31, 0.05, 1.0), (15, 0.03, 2.0),
                              (63, 0.03, 2.0)]:
        candidates[f"hgb_leaf{max_leaf}_lr{lr}_l2{l2}"] = HistGradientBoostingClassifier(
            max_leaf_nodes=max_leaf, learning_rate=lr, l2_regularization=l2,
            max_iter=400, early_stopping=True, n_iter_no_change=15,
            validation_fraction=0.15, class_weight="balanced", random_state=RNG,
        )

    for n_est, max_depth, msl in [(500, None, 5), (500, 12, 2)]:
        candidates[f"rf_n{n_est}_d{max_depth}_msl{msl}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth, min_samples_leaf=msl,
                class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
            )),
        ])

    candidates["extratrees_n500"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=RNG, n_jobs=-1,
        )),
    ])

    candidates["gbrt_lr0.05_d3"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=RNG,
        )),
    ])

    results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_trainval_v2, y_trainval, cv=CV,
                                  scoring="roc_auc", n_jobs=-1)
        results[name] = {"cv_auroc_mean": float(scores.mean()),
                          "cv_auroc_std": float(scores.std())}
        print(f"  {name:24s} CV AUROC = {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_auroc_mean"])
    best_model_template = candidates[best_name]
    print(f"\nSelected: {best_name} (CV AUROC {results[best_name]['cv_auroc_mean']:.4f})")

    # -----------------------------------------------------------------
    # CV-averaged operating thresholds (Youden's J and a >=90%-sensitivity
    # target), computed from out-of-fold predictions on train+val so the
    # threshold estimate isn't tied to one arbitrary split (cycle-1 open
    # question 1).
    # -----------------------------------------------------------------
    oof_proba = cross_val_predict(best_model_template, X_trainval_v2, y_trainval,
                                   cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, thr = roc_curve(y_trainval, oof_proba)
    youden_thr = float(thr[np.argmax(tpr - fpr)])
    ok = tpr >= 0.90
    high_sens_thr = float(thr[np.argmax(ok)] if ok.any() else thr[np.argmax(tpr)])
    print(f"OOF-derived thresholds: youden={youden_thr:.3f}, "
          f"high_sens(>=90%)={high_sens_thr:.3f}")

    # -----------------------------------------------------------------
    # Interpretation: permutation importance from out-of-fold-style fit --
    # fit on train only (first 80% of train+val via one CV fold's train
    # side is not representative of the whole set), so instead fit on
    # train+val minus nothing is not possible without touching test; we
    # fit on the train partition alone (same convention as cycle 1) and
    # score importance on val, both still test-free.
    # -----------------------------------------------------------------
    X_train_v2 = feat_v2.loc[train_ids]
    y_train = y.loc[train_ids]
    X_val_v2 = feat_v2.loc[val_ids]
    y_val = y.loc[val_ids]
    interp_model = clone_or_copy(best_model_template)
    interp_model.fit(X_train_v2, y_train)
    perm = permutation_importance(interp_model, X_val_v2, y_val, scoring="roc_auc",
                                   n_repeats=15, random_state=RNG, n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": feature_columns_v2,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv("importance_cycle2.csv", index=False)

    top_feats = importance_df.head(15)["feature"].tolist()
    direction_rows = []
    for f in top_feats:
        died = X_trainval_v2.loc[y_trainval == 1, f].mean()
        survived = X_trainval_v2.loc[y_trainval == 0, f].mean()
        direction_rows.append({"feature": f, "mean_if_died": died,
                                "mean_if_survived": survived})
    pd.DataFrame(direction_rows).to_csv("direction_check_cycle2.csv", index=False)

    # -----------------------------------------------------------------
    # Finalize: refit selected model+features on the full train+val,
    # wrapped in isotonic calibration (5-fold internal CV, same as cycle 1).
    # -----------------------------------------------------------------
    final_model = CalibratedClassifierCV(clone_or_copy(best_model_template),
                                          method="isotonic", cv=5)
    final_model.fit(X_trainval_v2, y_trainval)

    joblib.dump(final_model, "model_cycle2.joblib")
    with open("feature_columns_cycle2.json", "w") as f:
        json.dump(feature_columns_v2, f)

    # -----------------------------------------------------------------
    # Single, final look at test (second time this exact partition has
    # been evaluated -- see state_cycle2.md).
    # -----------------------------------------------------------------
    p_test = final_model.predict_proba(X_test_v2)[:, 1]
    auroc_test = roc_auc_score(y_test, p_test)
    auprc_test = average_precision_score(y_test, p_test)
    brier_test = brier_score_loss(y_test, p_test)

    pred_test = (p_test >= youden_thr).astype(int)
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

    icu_type_test = patients.set_index("patient_id").reindex(test_ids)["icu_type"]
    subgroup_auroc = {}
    for cat in icu_type_test.dropna().unique():
        mask = (icu_type_test == cat).values
        y_sub = y_test.to_numpy()[mask]
        if len(np.unique(y_sub)) < 2:
            continue
        subgroup_auroc[cat] = {
            "n": int(mask.sum()), "deaths": int(y_sub.sum()),
            "auroc": float(roc_auc_score(y_sub, p_test[mask])),
        }

    calib_df = pd.DataFrame({"p": p_test, "y": y_test.to_numpy()})
    calib_df["bin"] = pd.qcut(calib_df["p"], 10, duplicates="drop")
    calib = calib_df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")
    ).reset_index(drop=True)
    calib.to_csv("calibration_cycle2.csv", index=False)

    metrics = {
        "n_total": int(len(feat_v2)),
        "n_train": int(len(train_ids)), "n_val": int(len(val_ids)), "n_test": int(len(test_ids)),
        "reused_split_from": "split_cycle1.csv",
        "n_features_cycle1": int(feat_v1.shape[1]),
        "n_features_cycle2": int(feat_v2.shape[1]),
        "ablation_cv_auroc_cycle1_features": {
            "mean": float(auroc_v1_feats.mean()), "std": float(auroc_v1_feats.std())
        },
        "ablation_cv_auroc_cycle2_features": {
            "mean": float(auroc_v2_feats.mean()), "std": float(auroc_v2_feats.std())
        },
        "model_selection_cv_results": results,
        "selected_model": best_name,
        "selected_model_cv_auroc": results[best_name],
        "operating_threshold_youden": youden_thr,
        "operating_threshold_high_sens": high_sens_thr,
        "test_auroc": float(auroc_test),
        "test_auprc": float(auprc_test),
        "test_brier": float(brier_test),
        "test_sensitivity": float(sens), "test_specificity": float(spec),
        "test_ppv": float(ppv), "test_npv": float(npv),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "high_sens_test_sensitivity": float(sens_hs), "high_sens_test_specificity": float(spec_hs),
        "high_sens_test_ppv": float(ppv_hs), "high_sens_test_npv": float(npv_hs),
        "high_sens_test_confusion_matrix": {"tn": int(tn2), "fp": int(fp2), "fn": int(fn2), "tp": int(tp2)},
        "test_subgroup_auroc_by_icu_type": subgroup_auroc,
        "cycle1_test_auroc_for_reference": 0.8678623338870433,
        "cycle1_test_auprc_for_reference": 0.5319709915511441,
        "cycle1_test_brier_for_reference": 0.08753171075681854,
    }
    with open("metrics_cycle2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== FINAL TEST SET RESULT (single look, same partition as cycle 1) ===")
    print(json.dumps({k: v for k, v in metrics.items()
                       if k not in ("model_selection_cv_results",)}, indent=2))


def clone_or_copy(model):
    from sklearn.base import clone
    return clone(model)


if __name__ == "__main__":
    main()
