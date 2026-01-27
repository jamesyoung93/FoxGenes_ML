#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data_prep import DatasetSpec, get_labeled_subset, load_dataset
from src.feature_sets import (
    BASELINE_FULL,
    PRINCIPLED_EXPRESSION,
    PRINCIPLED_EXPRESSION_NO_POSITION,
    PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION,
    PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION,
    select_feature_columns,
)
from src.nested_cv import NestedCVConfig, build_model_search_spaces, precision_at_k


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to prepared_for_modeling*.csv (may include Unknown rows)")
    ap.add_argument("--out", required=True, help="Output directory for final models + predictions")

    ap.add_argument(
        "--feature_set",
        default=PRINCIPLED_EXPRESSION_NO_POSITION.name,
        choices=[
            BASELINE_FULL.name,
            PRINCIPLED_EXPRESSION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION.name,
        ],
        help="Which feature set to use"
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["rf"],
        choices=["logreg", "rf", "xgb"],
        help="Which final models to train"
    )

    # Final tuning CV (not nested)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=60, help="RandomizedSearch iterations for final tuning")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)

    ap.add_argument("--save_all_predictions", action="store_true",
                    help="If set, also write predictions for all rows (FOX/NotFOX/Unknown).")

    ap.add_argument(
        "--make_ensemble",
        action="store_true",
        help=(
            "If set, also write an ensemble prediction that averages predicted probabilities "
            "across the trained base models (requires >=2 models)."
        ),
    )

    return ap.parse_args()


def main() -> None:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = DatasetSpec()  # id_col='gene', label_col='label', unk_label='Unknown', etc.

    # Load full dataset (including Unknown)
    df = load_dataset(a.data, spec=spec, recompute_fold_changes=True)

    # Labeled subset for training
    labeled_df, y = get_labeled_subset(df, spec=spec)
    feat_cols = select_feature_columns(labeled_df, feature_set=a.feature_set)
    X = labeled_df[feat_cols].copy()

    # Unknown rows for prediction
    unk_mask = df[spec.label_col].astype(str) == spec.unk_label
    unknown_df = df.loc[unk_mask].copy()
    X_unk = unknown_df[feat_cols].copy() if len(unknown_df) else None

    # Compute scale_pos_weight for XGB based on full labeled set
    n_pos = max(1, int(np.sum(y == 1)))
    n_neg = int(np.sum(y == 0))
    spw = float(n_neg / n_pos)

    # Build pipelines + param spaces (same as nested CV)
    spaces = build_model_search_spaces(
        scale_pos_weight=spw,
        random_state=a.random_state,
        n_jobs=a.n_jobs,
    )

    cv = StratifiedKFold(n_splits=a.cv_folds, shuffle=True, random_state=a.random_state)

    summary_rows = []

    # Collect probabilities for optional ensemble outputs
    unk_probs_by_model = {}
    all_probs_by_model = {}

    for model_name in a.models:
        estimator, param_dist = spaces[model_name]

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=a.n_iter,
            scoring="average_precision",
            n_jobs=a.n_jobs,
            cv=cv,
            refit=True,
            random_state=a.random_state + 10_000,
            verbose=0,
        )
        search.fit(X, y)

        best = search.best_estimator_
        best_params = search.best_params_
        best_cv_ap = float(search.best_score_)

        # Fit-time (in-sample) metrics for sanity ONLY (not unbiased)
        y_hat = best.predict_proba(X)[:, 1]
        in_roc = float(roc_auc_score(y, y_hat))
        in_ap = float(average_precision_score(y, y_hat))

        row = {
            "model": model_name,
            "feature_set": a.feature_set,
            "n_labeled": int(len(y)),
            "n_unknown": int(len(unknown_df)),
            "scale_pos_weight_used_for_xgb": spw if model_name == "xgb" else "",
            "tuning_cv_avg_precision": best_cv_ap,
            "train_roc_auc": in_roc,
            "train_avg_precision": in_ap,
        }
        for k in (20, 50, 100):
            row[f"train_precision_at_{k}"] = precision_at_k(y, y_hat, k)
        summary_rows.append(row)

        # Save model + params
        joblib.dump(best, out_dir / f"final_model_{model_name}.joblib")
        with open(out_dir / f"final_params_{model_name}.json", "w") as f:
            json.dump(best_params, f, indent=2)

        # Predict unknowns
        if X_unk is not None and len(unknown_df) > 0:
            p_unk = best.predict_proba(X_unk)[:, 1]
            unk_probs_by_model[model_name] = p_unk
            pred_unk = pd.DataFrame({
                spec.id_col: unknown_df[spec.id_col].astype(str).values,
                "prob_FOX": p_unk,
            })
            # include helpful identifiers if present
            for extra in ["protein_id", "old_locus_tag", "locus_tag", "gene_symbol", "product"]:
                if extra in unknown_df.columns:
                    pred_unk[extra] = unknown_df[extra].astype(str).values

            pred_unk = pred_unk.sort_values("prob_FOX", ascending=False)
            pred_unk["rank"] = np.arange(1, len(pred_unk) + 1)
            pred_unk.to_csv(out_dir / f"predictions_unknown_{model_name}.csv", index=False)

        # Optionally save predictions for all rows
        if a.save_all_predictions:
            X_all = df[feat_cols].copy()
            p_all = best.predict_proba(X_all)[:, 1]
            all_probs_by_model[model_name] = p_all
            pred_all = pd.DataFrame({
                spec.id_col: df[spec.id_col].astype(str).values,
                spec.label_col: df[spec.label_col].astype(str).values,
                "prob_FOX": p_all,
            })
            for extra in ["protein_id", "old_locus_tag", "locus_tag", "gene_symbol", "product"]:
                if extra in df.columns:
                    pred_all[extra] = df[extra].astype(str).values
            pred_all.to_csv(out_dir / f"predictions_all_{model_name}.csv", index=False)

        print(f"[DONE] trained {model_name}; saved model + predictions to {out_dir}")

    # ------------------------------------------------------------------
    # Optional ensemble predictions (mean probability across trained models)
    # ------------------------------------------------------------------
    if a.make_ensemble and len(a.models) >= 2:
        # Unknowns
        if len(unk_probs_by_model) >= 2 and X_unk is not None and len(unknown_df) > 0:
            ens_unk = np.mean(np.column_stack([unk_probs_by_model[m] for m in a.models if m in unk_probs_by_model]), axis=1)
            pred_unk = pd.DataFrame({
                spec.id_col: unknown_df[spec.id_col].astype(str).values,
                "prob_FOX": ens_unk,
            })
            for extra in ["protein_id", "old_locus_tag", "locus_tag", "gene_symbol", "product"]:
                if extra in unknown_df.columns:
                    pred_unk[extra] = unknown_df[extra].astype(str).values
            pred_unk = pred_unk.sort_values("prob_FOX", ascending=False)
            pred_unk["rank"] = np.arange(1, len(pred_unk) + 1)
            pred_unk.to_csv(out_dir / "predictions_unknown_ensemble.csv", index=False)

        # All rows (optional)
        if a.save_all_predictions and len(all_probs_by_model) >= 2:
            ens_all = np.mean(np.column_stack([all_probs_by_model[m] for m in a.models if m in all_probs_by_model]), axis=1)
            pred_all = pd.DataFrame({
                spec.id_col: df[spec.id_col].astype(str).values,
                spec.label_col: df[spec.label_col].astype(str).values,
                "prob_FOX": ens_all,
            })
            for extra in ["protein_id", "old_locus_tag", "locus_tag", "gene_symbol", "product"]:
                if extra in df.columns:
                    pred_all[extra] = df[extra].astype(str).values
            pred_all.to_csv(out_dir / "predictions_all_ensemble.csv", index=False)

        # Add a lightweight summary row for the ensemble (train-set sanity metrics)
        try:
            y_hat_by_model = []
            for m in a.models:
                if m in unk_probs_by_model:
                    # not train predictions
                    pass
            # compute train probs for each trained estimator
            train_probs = []
            for m in a.models:
                model_path = out_dir / f"final_model_{m}.joblib"
                if model_path.exists():
                    est = joblib.load(model_path)
                    train_probs.append(est.predict_proba(X)[:, 1])
            if len(train_probs) >= 2:
                ens_train = np.mean(np.column_stack(train_probs), axis=1)
                row = {
                    "model": "ensemble",
                    "feature_set": a.feature_set,
                    "n_labeled": int(len(y)),
                    "n_unknown": int(len(unknown_df)),
                    "scale_pos_weight_used_for_xgb": "",
                    "tuning_cv_avg_precision": "",
                    "train_roc_auc": float(roc_auc_score(y, ens_train)),
                    "train_avg_precision": float(average_precision_score(y, ens_train)),
                }
                for k in (20, 50, 100):
                    row[f"train_precision_at_{k}"] = precision_at_k(y, ens_train, k)
                summary_rows.append(row)
        except Exception:
            # Ensemble summary is convenience-only; ignore failures.
            pass

    pd.DataFrame(summary_rows).to_csv(out_dir / "final_fit_summary.csv", index=False)
    print("[DONE] wrote", out_dir / "final_fit_summary.csv")
    print("Note: use nested CV metrics for unbiased performance; train metrics here are sanity checks.")

if __name__ == "__main__":
    main()
