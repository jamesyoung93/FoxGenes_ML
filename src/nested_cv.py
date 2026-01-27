"""
Nested cross-validation runner with inner-loop hyperparameter tuning.

This is the primary implementation for the "full nested CV" reviewer remediation.

Key design choices:
- Outer loop: unbiased model evaluation (generalization estimate).
- Inner loop: hyperparameter tuning using RandomizedSearchCV.
- Scoring for tuning: average precision (AP), which is informative under class imbalance.
- Reported metrics: ROC-AUC, AP, and Precision@k.

References:
- scikit-learn nested CV example: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- average_precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint, uniform

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


@dataclass(frozen=True)
class NestedCVConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    n_iter: int = 30
    random_state: int = 42
    n_jobs: int = 1  # set >1 on HPC; keep 1 for maximal portability
    tune_scoring: str = "average_precision"  # robust under imbalance
    precision_ks: Tuple[int, ...] = (20, 50, 100)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute Precision@k from scores (higher is better)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    n = len(y_true)
    if n == 0:
        return float("nan")
    k_eff = min(k, n)
    order = np.argsort(-y_score)[:k_eff]
    return float(np.mean(y_true[order] == 1))


def _common_steps(scale: bool) -> List[Tuple[str, BaseEstimator]]:
    steps: List[Tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def build_model_search_spaces(
    *,
    scale_pos_weight: float,
    random_state: int,
    n_jobs: int,
) -> Dict[str, Tuple[Pipeline, Dict]]:
    """
    Return model pipelines and parameter distributions for inner-loop tuning.

    Notes:
    - We intentionally keep hyperparameter spaces moderate to avoid a "behemoth."
    - For XGBoost, scale_pos_weight is set from the *outer training fold* imbalance ratio.
    """
    # Logistic regression: allow L1 or L2 penalty, tuned via C.
    logreg = Pipeline(
        _common_steps(scale=True)
        + [
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            )
        ]
    )
    logreg_params = {
        "clf__penalty": ["l1", "l2"],
        "clf__C": loguniform(1e-3, 1e3),
    }

    rf = Pipeline(
        _common_steps(scale=False)
        + [
            (
                "clf",
                RandomForestClassifier(
                    n_jobs=n_jobs,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            )
        ]
    )
    rf_params = {
        "clf__n_estimators": randint(200, 900),
        "clf__max_depth": [None, 3, 5, 8, 12, 16],
        "clf__min_samples_split": randint(2, 15),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", 0.3, 0.5, 0.7],
    }

    xgb = Pipeline(
        _common_steps(scale=False)
        + [
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=max(1, n_jobs),
                    random_state=random_state,
                    scale_pos_weight=scale_pos_weight,
                    tree_method="hist",
                ),
            )
        ]
    )
    xgb_params = {
        "clf__n_estimators": randint(150, 900),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__max_depth": randint(2, 9),
        "clf__min_child_weight": randint(1, 12),
        "clf__subsample": uniform(0.6, 0.4),  # 0.6..1.0
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__gamma": uniform(0.0, 5.0),
        "clf__reg_lambda": loguniform(1e-3, 50.0),
    }

    return {
        "logreg": (logreg, logreg_params),
        "rf": (rf, rf_params),
        "xgb": (xgb, xgb_params),
    }


def os_cpu_count() -> Optional[int]:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    ids: Optional[pd.Series] = None,
    config: NestedCVConfig = NestedCVConfig(),
    models: Tuple[str, ...] = ("logreg", "rf", "xgb"),
) -> Dict[str, Dict]:
    """
    Run nested CV for the requested models and return a results dictionary.
    """
    if ids is None:
        ids = pd.Series(np.arange(len(y)), name="id")

    outer = StratifiedKFold(
        n_splits=config.outer_folds, shuffle=True, random_state=config.random_state
    )

    results: Dict[str, Dict] = {}

    for model_name in models:
        results[model_name] = {
            "outer_fold_metrics": [],
            "best_params": [],
            "y_true": [],
            "y_score": [],
            "ids": [],
        }

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        ids_te = ids.iloc[test_idx].tolist()

        # Scale_pos_weight for XGBoost: negatives/positives in training fold
        n_pos = max(1, int(np.sum(y_tr == 1)))
        n_neg = int(np.sum(y_tr == 0))
        spw = float(n_neg / n_pos)

        search_spaces = build_model_search_spaces(
            scale_pos_weight=spw,
            random_state=config.random_state + fold_idx,
            n_jobs=config.n_jobs,
        )

        inner = StratifiedKFold(
            n_splits=config.inner_folds,
            shuffle=True,
            random_state=config.random_state + 1000 + fold_idx,
        )

        for model_name in models:
            estimator, param_dist = search_spaces[model_name]

            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_iter=config.n_iter,
                scoring=config.tune_scoring,
                n_jobs=config.n_jobs,
                cv=inner,
                refit=True,
                random_state=config.random_state + 10_000 + fold_idx,
                verbose=0,
            )

            search.fit(X_tr, y_tr)

            best = search.best_estimator_
            # score = P(FOX)
            y_score = best.predict_proba(X_te)[:, 1]

            fold_roc = roc_auc_score(y_te, y_score)
            fold_ap = average_precision_score(y_te, y_score)

            results[model_name]["outer_fold_metrics"].append(
                {"fold": fold_idx, "roc_auc": float(fold_roc), "avg_precision": float(fold_ap)}
            )
            results[model_name]["best_params"].append(search.best_params_)
            results[model_name]["y_true"].extend(y_te.tolist())
            results[model_name]["y_score"].extend(y_score.tolist())
            results[model_name]["ids"].extend(ids_te)

    # Aggregate-level metrics
    for model_name in models:
        y_true = np.array(results[model_name]["y_true"], dtype=int)
        y_score = np.array(results[model_name]["y_score"], dtype=float)

        summary = {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "avg_precision": float(average_precision_score(y_true, y_score)),
        }
        for k in config.precision_ks:
            summary[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)

        # fold mean/std
        fold_df = pd.DataFrame(results[model_name]["outer_fold_metrics"])
        summary["roc_auc_mean"] = float(fold_df["roc_auc"].mean())
        summary["roc_auc_std"] = float(fold_df["roc_auc"].std(ddof=1))
        summary["avg_precision_mean"] = float(fold_df["avg_precision"].mean())
        summary["avg_precision_std"] = float(fold_df["avg_precision"].std(ddof=1))

        results[model_name]["summary"] = summary

    return results


def save_results(
    results: Dict[str, Dict],
    out_dir: str,
    *,
    feature_set: str,
    config: NestedCVConfig,
) -> None:
    """
    Persist results to disk as:
    - summary.csv (one row per model)
    - fold_metrics.csv
    - predictions_{model}.csv
    - best_params_{model}.json
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    # Summary table
    rows = []
    for model_name, r in results.items():
        row = {
            "feature_set": feature_set,
            "model": model_name,
            "outer_folds": config.outer_folds,
            "inner_folds": config.inner_folds,
            "n_iter": config.n_iter,
            **r["summary"],
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{out_dir}/summary.csv", index=False)

    # Fold metrics
    fold_rows = []
    for model_name, r in results.items():
        for m in r["outer_fold_metrics"]:
            fold_rows.append({"feature_set": feature_set, "model": model_name, **m})
    pd.DataFrame(fold_rows).to_csv(f"{out_dir}/fold_metrics.csv", index=False)

    # Predictions + params
    for model_name, r in results.items():
        pred_df = pd.DataFrame(
            {"id": r["ids"], "y_true": r["y_true"], "y_score": r["y_score"]}
        )
        pred_df.to_csv(f"{out_dir}/predictions_{model_name}.csv", index=False)

        with open(f"{out_dir}/best_params_{model_name}.json", "w") as fh:
            json.dump(r["best_params"], fh, indent=2)
