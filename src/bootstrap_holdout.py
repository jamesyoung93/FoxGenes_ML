"""ml/src/bootstrap_holdout.py

Bootstrap-resampled stratified holdout evaluation.

This module implements the *primary* evaluation protocol used in the revised
manuscript's Table 1: repeated, stratified train/test splits at fixed train
fractions (e.g., 0.7/0.8/0.9) with performance summarized as mean ± SD.

Why this exists
---------------
Reviewer 2 requested that we reconcile evaluation language ("nested CV" vs
"repeated splits") and report metrics more informative under class imbalance.

We keep this protocol intentionally simple and transparent:
  - outer resampling: StratifiedShuffleSplit
  - optional inner tuning: RandomizedSearchCV on the training split
  - metrics: ROC-AUC, Average Precision (PR-AUC summary), Precision@k

Outputs are saved in a long/tidy format so downstream plotting (layered ROC,
enrichment curves) is straightforward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit

from .nested_cv import build_model_search_spaces, precision_at_k


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC is undefined when only one class is present."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


@dataclass(frozen=True)
class BootstrapHoldoutConfig:
    train_fracs: Tuple[float, ...] = (0.7, 0.8, 0.9)
    n_splits: int = 20
    inner_folds: int = 3
    n_iter: int = 30
    random_state: int = 42
    n_jobs: int = 1
    tune: bool = True
    tune_scoring: str = "average_precision"
    precision_ks: Tuple[int, ...] = (20, 50, 100)


def run_bootstrap_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    ids: Optional[pd.Series] = None,
    config: BootstrapHoldoutConfig = BootstrapHoldoutConfig(),
    models: Tuple[str, ...] = ("logreg", "rf", "xgb"),
    include_ensemble: bool = True,
    store_predictions: bool = True,
    store_best_params: bool = False,
) -> Dict[str, object]:
    """Run repeated stratified holdout splits and return metrics/predictions.

    Parameters
    ----------
    X, y:
        Labeled feature matrix and binary labels.
    ids:
        Identifier for each row (gene). If None, uses integer row index.
    config:
        Resampling and tuning configuration.
    models:
        Which base models to evaluate (must be keys of build_model_search_spaces).
    include_ensemble:
        If True, also evaluate a simple ensemble = mean(probabilities across models).
    store_predictions:
        If True, return a tidy long dataframe of all test-set predictions.
    store_best_params:
        If True, return tuned hyperparameters per split (can be large).
    """
    if ids is None:
        ids = pd.Series(np.arange(len(y)), name="id")
    y = np.asarray(y, dtype=int)

    # Prepare collectors
    metric_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    params_rows: List[Dict[str, object]] = []

    # Optional progress bar
    try:
        from tqdm import tqdm  # type: ignore

        _iter = tqdm
    except Exception:
        _iter = lambda x, **_: x  # noqa: E731

    for train_frac in config.train_fracs:
        if not (0.0 < float(train_frac) < 1.0):
            raise ValueError(f"train_frac must be in (0,1); got {train_frac}")

        splitter = StratifiedShuffleSplit(
            n_splits=config.n_splits,
            train_size=float(train_frac),
            random_state=config.random_state,
        )

        for split_idx, (tr_idx, te_idx) in _iter(
            enumerate(splitter.split(X, y), start=1),
            total=config.n_splits,
            desc=f"holdout(frac={train_frac})",
        ):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            ids_te = ids.iloc[te_idx].astype(str).tolist()

            # scale_pos_weight for XGBoost from this training split
            n_pos = max(1, int(np.sum(y_tr == 1)))
            n_neg = int(np.sum(y_tr == 0))
            spw = float(n_neg / n_pos)

            search_spaces = build_model_search_spaces(
                scale_pos_weight=spw,
                random_state=config.random_state + split_idx,
                n_jobs=config.n_jobs,
            )

            # Inner CV for hyperparameter tuning (optional)
            inner = StratifiedKFold(
                n_splits=config.inner_folds,
                shuffle=True,
                random_state=config.random_state + 1000 + split_idx,
            )

            # Collect per-model test probabilities for ensemble
            model_test_scores: Dict[str, np.ndarray] = {}

            for model_name in models:
                if model_name not in search_spaces:
                    raise KeyError(f"Unknown model '{model_name}'. Valid: {list(search_spaces)}")
                estimator, param_dist = search_spaces[model_name]

                if config.tune:
                    search = RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions=param_dist,
                        n_iter=config.n_iter,
                        scoring=config.tune_scoring,
                        n_jobs=config.n_jobs,
                        cv=inner,
                        refit=True,
                        random_state=config.random_state + 10_000 + split_idx,
                        verbose=0,
                    )
                    search.fit(X_tr, y_tr)
                    best = search.best_estimator_
                    best_params = search.best_params_
                    best_cv_score = float(search.best_score_)
                else:
                    best = estimator.fit(X_tr, y_tr)
                    best_params = {}
                    best_cv_score = float("nan")

                y_score = best.predict_proba(X_te)[:, 1].astype(float)
                model_test_scores[model_name] = y_score

                roc = _safe_roc_auc(y_te, y_score)
                ap = float(average_precision_score(y_te, y_score))

                row = {
                    "train_frac": float(train_frac),
                    "split": int(split_idx),
                    "n_train": int(len(tr_idx)),
                    "n_test": int(len(te_idx)),
                    "model": model_name,
                    "roc_auc": roc,
                    "avg_precision": ap,
                    "tuning_score": best_cv_score,
                    "tuning_scoring": config.tune_scoring if config.tune else "",
                }
                for k in config.precision_ks:
                    row[f"precision_at_{k}"] = precision_at_k(y_te, y_score, int(k))
                metric_rows.append(row)

                if store_predictions:
                    for gid, yt, ys in zip(ids_te, y_te.tolist(), y_score.tolist()):
                        pred_rows.append(
                            {
                                "train_frac": float(train_frac),
                                "split": int(split_idx),
                                "model": model_name,
                                "id": gid,
                                "y_true": int(yt),
                                "y_score": float(ys),
                            }
                        )

                if store_best_params and config.tune:
                    params_rows.append(
                        {
                            "train_frac": float(train_frac),
                            "split": int(split_idx),
                            "model": model_name,
                            "best_params": best_params,
                        }
                    )

            # Ensemble (mean of predicted probabilities)
            if include_ensemble and len(model_test_scores) >= 2:
                ens = np.mean(np.column_stack(list(model_test_scores.values())), axis=1)
                roc = _safe_roc_auc(y_te, ens)
                ap = float(average_precision_score(y_te, ens))
                row = {
                    "train_frac": float(train_frac),
                    "split": int(split_idx),
                    "n_train": int(len(tr_idx)),
                    "n_test": int(len(te_idx)),
                    "model": "ensemble",
                    "roc_auc": roc,
                    "avg_precision": ap,
                    "tuning_score": float("nan"),
                    "tuning_scoring": "",
                }
                for k in config.precision_ks:
                    row[f"precision_at_{k}"] = precision_at_k(y_te, ens, int(k))
                metric_rows.append(row)

                if store_predictions:
                    for gid, yt, ys in zip(ids_te, y_te.tolist(), ens.tolist()):
                        pred_rows.append(
                            {
                                "train_frac": float(train_frac),
                                "split": int(split_idx),
                                "model": "ensemble",
                                "id": gid,
                                "y_true": int(yt),
                                "y_score": float(ys),
                            }
                        )

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows) if store_predictions else pd.DataFrame()

    out: Dict[str, object] = {
        "metrics": metrics_df,
        "predictions": preds_df,
        "config": config,
    }
    if store_best_params:
        out["best_params"] = params_rows
    return out


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std summary grouped by train_frac and model."""
    if metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [c for c in metrics_df.columns if c in {"roc_auc", "avg_precision"} or c.startswith("precision_at_")]

    g = metrics_df.groupby(["train_frac", "model"], dropna=False)
    out_rows: List[Dict[str, object]] = []
    for (train_frac, model), df_g in g:
        row: Dict[str, object] = {
            "train_frac": float(train_frac),
            "model": str(model),
            "n_splits": int(df_g["split"].nunique()),
        }
        for c in metric_cols:
            row[f"{c}_mean"] = float(df_g[c].mean())
            row[f"{c}_std"] = float(df_g[c].std(ddof=1))
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["train_frac", "model"]).reset_index(drop=True)


def save_bootstrap_holdout_results(
    results: Dict[str, object],
    out_dir: str | Path,
    *,
    feature_set: str,
) -> None:
    """Write results to disk (metrics, summary, predictions, config)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_df: pd.DataFrame = results["metrics"]  # type: ignore[assignment]
    preds_df: pd.DataFrame = results.get("predictions", pd.DataFrame())  # type: ignore[assignment]
    cfg: BootstrapHoldoutConfig = results["config"]  # type: ignore[assignment]

    # Attach feature set
    if not metrics_df.empty:
        metrics_df = metrics_df.copy()
        metrics_df.insert(0, "feature_set", feature_set)
        metrics_df.to_csv(out_path / "split_metrics.csv", index=False)

    summary_df = summarize_metrics(metrics_df)
    if not summary_df.empty:
        summary_df.insert(0, "feature_set", feature_set)
        summary_df.to_csv(out_path / "summary.csv", index=False)

    if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
        preds_df = preds_df.copy()
        preds_df.insert(0, "feature_set", feature_set)
        preds_df.to_csv(out_path / "predictions.csv", index=False)

    with open(out_path / "config.json", "w") as fh:
        json.dump({"feature_set": feature_set, **asdict(cfg)}, fh, indent=2)

    # Best params (optional)
    if "best_params" in results:
        with open(out_path / "best_params.jsonl", "w") as fh:
            for row in results["best_params"]:  # type: ignore[index]
                fh.write(json.dumps(row) + "\n")
