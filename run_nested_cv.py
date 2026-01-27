#!/usr/bin/env python3
"""
CLI entrypoint: nested cross-validation with inner-loop hyperparameter tuning.

Examples
--------
# Baseline (includes RNA count proxies)
python ml/run_nested_cv.py --data ml/data/prepared_for_modeling.csv --feature_set baseline_full --out ml/outputs/nested_cv_baseline

# Principled expression (drops redundant count proxies)
python ml/run_nested_cv.py --data ml/data/prepared_for_modeling.csv --feature_set principled_expression --out ml/outputs/nested_cv_principled

# Principled expression + no positional features (leakage check)
python ml/run_nested_cv.py --data ml/data/prepared_for_modeling.csv --feature_set principled_expression_no_position --out ml/outputs/nested_cv_no_position
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
from src.nested_cv import NestedCVConfig, run_nested_cv, save_results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to prepared_for_modeling.csv")
    ap.add_argument(
        "--feature_set",
        default=PRINCIPLED_EXPRESSION.name,
        choices=[
            BASELINE_FULL.name,
            PRINCIPLED_EXPRESSION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION.name,
        ],
        help="Which feature set to evaluate",
    )
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--n_iter", type=int, default=30, help="RandomizedSearch iterations")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for inner search (use 1 for portability)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf", "xgb"],
        choices=["logreg", "rf", "xgb"],
        help="Which models to run",
    )
    ap.add_argument(
        "--keep_unknown",
        action="store_true",
        help="If set, do not drop Unknown rows (NOTE: evaluation still uses labeled only).",
    )
    return ap.parse_args()


def main() -> None:
    a = parse_args()

    df = load_dataset(a.data)
    labeled_df, y = get_labeled_subset(df, spec=DatasetSpec())

    feat_cols = select_feature_columns(labeled_df, feature_set=a.feature_set)
    X = labeled_df[feat_cols].copy()

    ids = labeled_df["gene"] if "gene" in labeled_df.columns else pd.Series(range(len(y)))

    cfg = NestedCVConfig(
        outer_folds=a.outer_folds,
        inner_folds=a.inner_folds,
        n_iter=a.n_iter,
        random_state=a.random_state,
        n_jobs=a.n_jobs,
    )

    results = run_nested_cv(X, y, ids=ids, config=cfg, models=tuple(a.models))
    save_results(results, a.out, feature_set=a.feature_set, config=cfg)

    print("✓ Nested CV complete")
    print("Outputs written to:", Path(a.out).resolve())


if __name__ == "__main__":
    main()
