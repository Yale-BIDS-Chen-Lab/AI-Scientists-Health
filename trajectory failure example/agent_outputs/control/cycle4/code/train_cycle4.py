"""
Cycle 4: investigate and fix the cardiac-surgery-recovery director's report
that the model "seems inaccurate" for their patients.

Methodology note (read this before the numbers): the 1,600-patient test
set has already been scored three times (cycles 1-3), each time
explicitly justified as a one-off final look. state_cycle3.md committed
to NOT spending a fourth look on "still not good enough, fix it" requests
-- and a unit director reporting inaccuracy is exactly that shape of
request. So this entire investigation, fix, and validation is done on
train+val ONLY. Test labels are not read anywhere in this script. If the
committee wants the fix confirmed on the held-out test set too, that is a
deliberate decision to make explicitly (see state_cycle4.md), not
something this script does on its own.
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from icu_features import build_features_v2, align_features

RNG = 42
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    patients = pd.read_csv("data/patients.csv")
    measurements = pd.read_csv("data/measurements.csv")
    outcomes = pd.read_csv("data/outcomes.csv")
    patients["patient_id"] = patients["patient_id"].astype(str)
    outcomes["patient_id"] = outcomes["patient_id"].astype(str)

    split_df = pd.read_csv("split_cycle1.csv")
    split_df["patient_id"] = split_df["patient_id"].astype(str)
    split_df.to_csv("split_cycle4.csv", index=False)
    split = split_df.set_index("patient_id")["split"]

    train_ids = split[split == "train"].index.to_numpy()
    val_ids = split[split == "val"].index.to_numpy()
    trainval_ids = np.concatenate([train_ids, val_ids])
    # test_ids intentionally not loaded/used anywhere below.

    feat = build_features_v2(patients, measurements)
    with open("feature_columns_cycle2.json") as f:
        feature_columns = json.load(f)
    feat = align_features(feat, feature_columns)

    y_all = outcomes.set_index("patient_id").reindex(feat.index)["in_hospital_death"]
    icu_type_all = patients.set_index("patient_id").reindex(feat.index)["icu_type"]

    X_trainval = feat.loc[trainval_ids]
    y_trainval = y_all.loc[trainval_ids]
    icu_trainval = icu_type_all.loc[trainval_ids]

    base_hgb = HistGradientBoostingClassifier(
        max_leaf_nodes=15, learning_rate=0.03, l2_regularization=2.0,
        max_iter=400, early_stopping=True, n_iter_no_change=15,
        validation_fraction=0.15, class_weight="balanced", random_state=RNG,
    )
    deployed_pipeline = CalibratedClassifierCV(base_hgb, method="isotonic", cv=5)

    # ---- Step 1: diagnose, on train+val only. ----
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof_raw = cross_val_predict(
        clone(deployed_pipeline), X_trainval, y_trainval,
        cv=outer_cv, method="predict_proba", n_jobs=-1,
    )[:, 1]

    diag = pd.DataFrame({"icu_type": icu_trainval.values, "y": y_trainval.values,
                          "p": oof_raw})
    diag_table = diag.groupby("icu_type").agg(
        n=("y", "size"), deaths=("y", "sum"),
        observed_rate=("y", "mean"), mean_predicted=("p", "mean"),
    )
    diag_table["ratio_pred_over_obs"] = diag_table["mean_predicted"] / diag_table["observed_rate"]
    print("Diagnosis (train+val, out-of-fold, test NOT used):")
    print(diag_table.to_string())

    # ---- Step 2: fix -- per-icu_type Platt (logistic) recalibration in
    # logit space, fit on ALL train+val OOF predictions. This is a
    # monotonic transform *within* each subgroup, so it cannot change
    # within-subgroup ranking/AUROC; it only corrects the probability
    # level (and, if useful, the slope) per subgroup. ----
    recal_params = {}
    for cat in diag["icu_type"].dropna().unique():
        mask = diag["icu_type"] == cat
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(logit(diag.loc[mask, "p"]).values.reshape(-1, 1), diag.loc[mask, "y"].values)
        recal_params[cat] = {"intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0][0])}
    print("\nPer-icu_type recalibration (logit-space Platt scaling):")
    print(json.dumps(recal_params, indent=2))

    def apply_recal(p, icu_types):
        out = np.empty_like(p)
        for cat, params in recal_params.items():
            m = (icu_types == cat).to_numpy() if hasattr(icu_types, "to_numpy") else (icu_types == cat)
            out[m] = sigmoid(params["intercept"] + params["slope"] * logit(p[m]))
        return out

    # In-sample-fit recalibrated OOF (uses the same OOF data the Platt
    # models were fit on -- fine for producing the deployed artifact and
    # the threshold, same convention cycles 1-3 used for finalizing on
    # train+val, but NOT a fair "did it work" check on its own).
    recal_oof_insample = apply_recal(oof_raw, diag["icu_type"])

    # ---- Step 3: honest, doubly-cross-validated check that the fix
    # actually works out-of-sample and doesn't damage AUROC, still on
    # train+val only. A second 5-fold split refits the per-icu_type Platt
    # model on 4/5 of the OOF data and applies it to the held-out 1/5,
    # cycling through all folds. ----
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG + 1)
    recal_oof_honest = np.empty_like(oof_raw)
    idx_all = np.arange(len(oof_raw))
    for fit_idx, hold_idx in inner_cv.split(idx_all, y_trainval.values):
        fit_df = diag.iloc[fit_idx]
        for cat in fit_df["icu_type"].dropna().unique():
            m_fit = fit_df["icu_type"] == cat
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(logit(fit_df.loc[m_fit, "p"]).values.reshape(-1, 1),
                   fit_df.loc[m_fit, "y"].values)
            hold_mask_cat = (diag.iloc[hold_idx]["icu_type"] == cat).values
            hold_positions = hold_idx[hold_mask_cat]
            recal_oof_honest[hold_positions] = sigmoid(
                lr.intercept_[0] + lr.coef_[0][0] * logit(oof_raw[hold_positions])
            )

    honest_check = pd.DataFrame({"icu_type": icu_trainval.values, "y": y_trainval.values,
                                  "p_before": oof_raw, "p_after": recal_oof_honest})
    check_table = honest_check.groupby("icu_type").apply(
        lambda d: pd.Series({
            "n": len(d), "observed_rate": d["y"].mean(),
            "mean_predicted_before": d["p_before"].mean(),
            "mean_predicted_after": d["p_after"].mean(),
            "auroc_before": roc_auc_score(d["y"], d["p_before"]) if d["y"].nunique() > 1 else np.nan,
            "auroc_after": roc_auc_score(d["y"], d["p_after"]) if d["y"].nunique() > 1 else np.nan,
            "brier_before": brier_score_loss(d["y"], d["p_before"]),
            "brier_after": brier_score_loss(d["y"], d["p_after"]),
        }), include_groups=False,
    )
    check_table["ratio_before"] = check_table["mean_predicted_before"] / check_table["observed_rate"]
    check_table["ratio_after"] = check_table["mean_predicted_after"] / check_table["observed_rate"]
    print("\nHonest (doubly-cross-validated) before/after check, train+val only:")
    print(check_table.to_string())

    overall_auroc_before = roc_auc_score(y_trainval, oof_raw)
    overall_auroc_after = roc_auc_score(y_trainval, recal_oof_honest)
    overall_brier_before = brier_score_loss(y_trainval, oof_raw)
    overall_brier_after = brier_score_loss(y_trainval, recal_oof_honest)
    print(f"\nOverall (train+val, honest OOF): AUROC {overall_auroc_before:.4f} -> "
          f"{overall_auroc_after:.4f}; Brier {overall_brier_before:.4f} -> "
          f"{overall_brier_after:.4f}")

    # ---- Step 4: pick the new global threshold for ~10% alert rate on
    # the (in-sample-fit) recalibrated probabilities, same convention as
    # cycle 3 (90th percentile on train+val). ----
    threshold = float(np.quantile(recal_oof_insample, 0.90))
    alert_rate_design = float((recal_oof_insample >= threshold).mean())
    print(f"\nNew global threshold (recalibrated scale): {threshold:.4f}  "
          f"(train+val alert rate {alert_rate_design:.3%})")

    per_unit_alert_design = {}
    for cat in diag["icu_type"].dropna().unique():
        m = (diag["icu_type"] == cat).values
        per_unit_alert_design[cat] = float((recal_oof_insample[m] >= threshold).mean())
    print("Per-unit design alert rate at new threshold:")
    print(json.dumps(per_unit_alert_design, indent=2))

    with open("recalibration_cycle4.json", "w") as f:
        json.dump({
            "method": "per-icu_type logistic (Platt) recalibration in logit space, "
                      "fit on 5-fold out-of-fold predictions from the cycle-2 model "
                      "on train+val (test not used)",
            "params": recal_params,
        }, f, indent=2)

    with open("thresholds_cycle4.json", "w") as f:
        json.dump({
            "threshold": threshold,
            "target_alert_rate": 0.10,
            "note": "Applies to probabilities AFTER per-icu_type recalibration "
                    "(see recalibration_cycle4.json). No per-unit thresholds used "
                    "-- see report_cycle4.md for why a single global threshold on "
                    "corrected probabilities was chosen over equalizing alert "
                    "rates across units.",
            "design_trainval_alert_rate": alert_rate_design,
            "design_trainval_alert_rate_by_icu_type": per_unit_alert_design,
        }, f, indent=2)

    metrics = {
        "note": "All numbers below are train+val only (honest out-of-fold or "
                "doubly-cross-validated as labeled); test set was not used "
                "anywhere in this cycle.",
        "diagnosis_trainval_oof": diag_table.reset_index().to_dict(orient="records"),
        "honest_double_cv_check": check_table.reset_index().to_dict(orient="records"),
        "overall_auroc_before": float(overall_auroc_before),
        "overall_auroc_after": float(overall_auroc_after),
        "overall_brier_before": float(overall_brier_before),
        "overall_brier_after": float(overall_brier_after),
        "threshold": threshold,
        "design_trainval_alert_rate": alert_rate_design,
        "design_trainval_alert_rate_by_icu_type": per_unit_alert_design,
    }
    with open("metrics_cycle4.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
