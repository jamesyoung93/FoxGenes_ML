#!/usr/bin/env python3
"""CLI: leave-one-area-out (blocked) cross-validation.

This script implements the reviewer-requested positional leakage check by
holding out contiguous genomic regions defined by chromosome coordinates.

It is designed to be used with feature sets that *exclude* positional
coordinates as model inputs (e.g. principled_expression_no_position...), while
still using coordinates to define the holdout blocks.

Example
-------
python ml/run_blocked_area_cv.py \
  --data ml/data/prepared_for_modeling.csv \
  --feature_set principled_expression_no_position_plus_conservation \
  --out ml/outputs/blocked_cv_no_pos_plus_cons \
  --n_groups 5 \
  --n_partitions 4
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
from src.blocked_area_cv import BlockedAreaCVConfig, run_blocked_area_cv, save_blocked_area_cv_results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Prepared feature matrix CSV")
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

    ap.add_argument("--coord_col", default="chromosome_region_start", help="Column used to define genomic order")

    ap.add_argument("--n_groups", type=int, default=5)
    ap.add_argument(
        "--n_partitions",
        type=int,
        default=1,
        help=(
            "Number of shifted block partitions. Total evaluated splits = n_groups * n_partitions. "
            "E.g., n_groups=5 and n_partitions=4 yields 20 holdout splits."
        ),
    )
    ap.add_argument("--offset_strategy", choices=["linspace", "random"], default="linspace")

    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--no_tune", action="store_true")

    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf", "xgb"],
        choices=["logreg", "rf", "xgb"],
    )
    ap.add_argument("--no_ensemble", action="store_true")
    ap.add_argument("--no_predictions", action="store_true")
    ap.add_argument("--save_best_params", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(a.data)
    labeled_df, y = get_labeled_subset(df, spec=DatasetSpec())

    if a.coord_col not in labeled_df.columns:
        raise SystemExit(
            f"Coordinate column '{a.coord_col}' not found in labeled data. "
            "Provide a CSV with chromosome coordinates or set --coord_col accordingly."
        )

    coords = pd.to_numeric(labeled_df[a.coord_col], errors="coerce").to_numpy()
    if pd.isna(coords).any():
        raise SystemExit(
            f"Coordinate column '{a.coord_col}' contains NaN after numeric conversion. "
            "Fix upstream data or choose a different --coord_col."
        )

    feat_cols = select_feature_columns(labeled_df, feature_set=a.feature_set)
    X = labeled_df[feat_cols].copy()
    ids = labeled_df["gene"] if "gene" in labeled_df.columns else pd.Series(range(len(y)))

    cfg = BlockedAreaCVConfig(
        n_groups=a.n_groups,
        n_partitions=a.n_partitions,
        offset_strategy=a.offset_strategy,
        inner_folds=a.inner_folds,
        n_iter=a.n_iter,
        random_state=a.random_state,
        n_jobs=a.n_jobs,
        tune=not a.no_tune,
    )

    res = run_blocked_area_cv(
        X,
        y,
        coords,
        ids=ids,
        config=cfg,
        models=tuple(a.models),
        include_ensemble=not a.no_ensemble,
        store_predictions=not a.no_predictions,
        store_best_params=a.save_best_params,
    )

    save_blocked_area_cv_results(res, out_dir, feature_set=a.feature_set)

    print("✓ Blocked area CV complete")
    print("Outputs written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
