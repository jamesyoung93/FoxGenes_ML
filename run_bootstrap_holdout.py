#!/usr/bin/env python3
"""CLI: bootstrap-resampled stratified holdout evaluation.

This is the reviewer-facing protocol used to generate Table 1-style metrics:
mean ± SD across repeated stratified train/test splits.

Example
-------
python ml/run_bootstrap_holdout.py \
  --data ml/data/prepared_for_modeling.csv \
  --feature_set principled_expression_no_position_plus_conservation \
  --out ml/outputs/bootstrap_principled_no_pos_plus_cons \
  --n_splits 20 \
  --train_fracs 0.7 0.8 0.9 \
  --models logreg rf xgb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_prep import DatasetSpec, get_labeled_subset, load_dataset
from src.feature_sets import (
    BASELINE_FULL,
    PRINCIPLED_EXPRESSION,
    PRINCIPLED_EXPRESSION_NO_POSITION,
    PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION,
    PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION,
    select_feature_columns,
)
from src.bootstrap_holdout import BootstrapHoldoutConfig, run_bootstrap_holdout, save_bootstrap_holdout_results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to prepared_for_modeling*.csv")
    ap.add_argument("--out", required=True, help="Output directory")

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
    )

    ap.add_argument("--train_fracs", nargs="+", type=float, default=[0.7, 0.8, 0.9])
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--no_tune", action="store_true", help="Disable RandomizedSearchCV; use default params.")

    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf", "xgb"],
        choices=["logreg", "rf", "xgb"],
        help="Which models to run",
    )
    ap.add_argument("--no_ensemble", action="store_true", help="Disable ensemble scoring.")
    ap.add_argument("--no_predictions", action="store_true", help="Do not save per-split predictions.")
    ap.add_argument("--save_best_params", action="store_true", help="Save best params per split (JSONL).")

    return ap.parse_args()


def main() -> None:
    a = parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(a.data)
    labeled_df, y = get_labeled_subset(df, spec=DatasetSpec())

    feat_cols = select_feature_columns(labeled_df, feature_set=a.feature_set)
    X = labeled_df[feat_cols].copy()
    ids = labeled_df["gene"] if "gene" in labeled_df.columns else pd.Series(range(len(y)))

    cfg = BootstrapHoldoutConfig(
        train_fracs=tuple(a.train_fracs),
        n_splits=a.n_splits,
        inner_folds=a.inner_folds,
        n_iter=a.n_iter,
        random_state=a.random_state,
        n_jobs=a.n_jobs,
        tune=not a.no_tune,
    )

    res = run_bootstrap_holdout(
        X,
        y,
        ids=ids,
        config=cfg,
        models=tuple(a.models),
        include_ensemble=not a.no_ensemble,
        store_predictions=not a.no_predictions,
        store_best_params=a.save_best_params,
    )

    save_bootstrap_holdout_results(res, out_dir, feature_set=a.feature_set)

    print("✓ Bootstrap holdout evaluation complete")
    print("Outputs written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
