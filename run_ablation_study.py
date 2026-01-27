#!/usr/bin/env python3
"""
Ablation study runner for reviewer-facing robustness:

Compares:
1) baseline_full  (includes RNA count proxies)
2) principled_expression (drops redundant RNA count proxies; retains RPKM + RNA FC + protein FC)

This directly supports reviewer concerns about:
- Feature redundancy / technical confounding (multiple correlated expression proxies).
- Evaluation clarity (reports ROC-AUC, Average Precision, Precision@k under nested CV).

Example
-------
python ml/run_ablation_study.py --data ml/data/prepared_for_modeling.csv --out ml/outputs/ablation --n_iter 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_prep import DatasetSpec, get_labeled_subset, load_dataset
from src.feature_sets import BASELINE_FULL, PRINCIPLED_EXPRESSION, select_feature_columns
from src.nested_cv import NestedCVConfig, run_nested_cv, save_results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to prepared_for_modeling.csv")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for inner search (use 1 for portability)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf", "xgb"],
        choices=["logreg", "rf", "xgb"],
    )
    return ap.parse_args()


def main() -> None:
    a = parse_args()

    df = load_dataset(a.data)
    labeled_df, y = get_labeled_subset(df, spec=DatasetSpec())
    ids = labeled_df["gene"] if "gene" in labeled_df.columns else pd.Series(range(len(y)))

    cfg = NestedCVConfig(
        outer_folds=a.outer_folds,
        inner_folds=a.inner_folds,
        n_iter=a.n_iter,
        random_state=a.random_state,
        n_jobs=a.n_jobs,
    )

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    comparisons = []
    for fs in [BASELINE_FULL.name, PRINCIPLED_EXPRESSION.name]:
        feat_cols = select_feature_columns(labeled_df, feature_set=fs)
        X = labeled_df[feat_cols].copy()

        res = run_nested_cv(X, y, ids=ids, config=cfg, models=tuple(a.models))
        out_dir = out_root / fs
        save_results(res, str(out_dir), feature_set=fs, config=cfg)

        for model_name, r in res.items():
            row = {"feature_set": fs, "model": model_name, **r["summary"]}
            comparisons.append(row)

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(out_root / "ablation_comparison_summary.csv", index=False)

    print("✓ Ablation study complete")
    print("Comparison written to:", (out_root / "ablation_comparison_summary.csv").resolve())


if __name__ == "__main__":
    main()
