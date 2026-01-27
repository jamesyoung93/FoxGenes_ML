#!/usr/bin/env python3
"""Run a reviewer-facing bootstrap/holdout ablation suite.

This script runs the repeated stratified holdout evaluation across multiple
feature sets (ablation conditions) and writes both per-condition outputs and a
single combined comparison table.

Default suite (matches the common reviewer remediation sequence)
--------------------------------------------------------------
1) baseline_full
2) principled_expression
3) principled_expression_no_position
4) principled_expression_no_position_no_conservation
5) principled_expression_no_position_plus_conservation

Example
-------
python ml/run_bootstrap_ablation_suite.py \
  --data ml/data/prepared_for_modeling_augmented.csv \
  --out ml/outputs/bootstrap_ablation_suite \
  --n_splits 20 --train_fracs 0.7 0.8 0.9
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


DEFAULT_FEATURE_SETS = [
    BASELINE_FULL.name,
    PRINCIPLED_EXPRESSION.name,
    PRINCIPLED_EXPRESSION_NO_POSITION.name,
    PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION.name,
    PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION.name,
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Prepared feature matrix CSV")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--feature_sets", nargs="+", default=DEFAULT_FEATURE_SETS)

    ap.add_argument("--train_fracs", nargs="+", type=float, default=[0.7, 0.8, 0.9])
    ap.add_argument("--n_splits", type=int, default=20)
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

    return ap.parse_args()


def main() -> None:
    a = parse_args()
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_dataset(a.data)
    labeled_df, y = get_labeled_subset(df, spec=DatasetSpec())
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

    summary_rows = []

    for fs in a.feature_sets:
        try:
            feat_cols = select_feature_columns(labeled_df, feature_set=fs)
        except ValueError as e:
            # Common case: requesting "plus_conservation" before augmentation.
            print(f"[WARN] Skipping feature_set='{fs}': {e}")
            continue

        X = labeled_df[feat_cols].copy()

        fs_out = out_root / fs
        fs_out.mkdir(parents=True, exist_ok=True)

        res = run_bootstrap_holdout(
            X,
            y,
            ids=ids,
            config=cfg,
            models=tuple(a.models),
            include_ensemble=not a.no_ensemble,
            store_predictions=not a.no_predictions,
        )
        save_bootstrap_holdout_results(res, fs_out, feature_set=fs)

        # Collect summary for combined comparison
        try:
            summ = pd.read_csv(fs_out / "summary.csv")
            summary_rows.append(summ)
        except Exception:
            pass

    if summary_rows:
        comp = pd.concat(summary_rows, ignore_index=True)
        comp.to_csv(out_root / "ablation_suite_summary.csv", index=False)

    print("✓ Bootstrap ablation suite complete")
    print("Outputs written to:", out_root.resolve())


if __name__ == "__main__":
    main()
