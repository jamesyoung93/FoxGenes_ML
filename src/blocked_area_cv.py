"""ml/src/blocked_area_cv.py

Blocked / contiguous-genome cross-validation.

Reviewer 2 raised a plausible risk of positional information leakage: if genes in
the same genomic island appear in both train and test folds, a model can exploit
positional proxies or island-specific correlated features.

When curated "genomic island" labels are unavailable, a pragmatic approximation
is to hold out *contiguous* regions along the chromosome. This module implements
"leave-one-area-out" evaluation by partitioning genes into *k* contiguous blocks
along chromosome coordinates and using each block as a holdout test set.

To obtain more than k test splits (e.g., 20 "bootstraps" with k=5), the block
boundaries can be shifted by a small offset across multiple partitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .bootstrap_holdout import _safe_roc_auc
from .nested_cv import build_model_search_spaces, precision_at_k


def assign_contiguous_groups(
    coords: np.ndarray,
    *,
    n_groups: int,
    offset: int = 0,
) -> np.ndarray:
    """Assign each sample to a contiguous group along the coordinate axis.

    Parameters
    ----------
    coords:
        1D array of genomic coordinates (e.g., chromosome_region_start).
    n_groups:
        Number of contiguous blocks.
    offset:
        Shift applied in *rank space* (0..block_size-1). This slides block
        boundaries without wrapping, keeping all blocks contiguous.

    Returns
    -------
    groups:
        Integer array in {0..n_groups-1} of length len(coords).
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n < n_groups:
        raise ValueError(f"Need at least n_groups samples (n={n}, n_groups={n_groups})")

    order = np.argsort(coords)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)

    block = max(1, n // n_groups)
    off = int(offset)
    if off < 0:
        off = 0
    if off >= block:
        off = off % block

    groups = (ranks + off) // block
    groups = groups.astype(int)
    groups[groups >= n_groups] = n_groups - 1
    return groups


def _offset_schedule(block_size: int, n_partitions: int, random_state: int, strategy: str) -> List[int]:
    if n_partitions < 1:
        raise ValueError("n_partitions must be >= 1")
    if block_size <= 1:
        return [0] * n_partitions

    if strategy == "linspace":
        if n_partitions == 1:
            return [0]
        # Spread offsets across [0, block_size-1]
        vals = np.linspace(0, block_size - 1, n_partitions)
        return [int(round(v)) for v in vals]

    if strategy == "random":
        rng = np.random.default_rng(int(random_state))
        return [int(rng.integers(0, block_size)) for _ in range(n_partitions)]

    raise ValueError("strategy must be one of: linspace, random")


@dataclass(frozen=True)
class BlockedAreaCVConfig:
    n_groups: int = 5
    n_partitions: int = 1
    offset_strategy: str = "linspace"  # or "random"
    inner_folds: int = 3
    n_iter: int = 30
    random_state: int = 42
    n_jobs: int = 1
    tune: bool = True
    tune_scoring: str = "average_precision"
    precision_ks: Tuple[int, ...] = (20, 50, 100)


def run_blocked_area_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    coords: np.ndarray,
    *,
    ids: Optional[pd.Series] = None,
    config: BlockedAreaCVConfig = BlockedAreaCVConfig(),
    models: Tuple[str, ...] = ("logreg", "rf", "xgb"),
    include_ensemble: bool = True,
    store_predictions: bool = True,
    store_best_params: bool = False,
) -> Dict[str, object]:
    """Run leave-one-area-out (contiguous block) CV.

    The total number of evaluated splits equals n_partitions * n_groups.
    """
    if ids is None:
        ids = pd.Series(np.arange(len(y)), name="id")
    y = np.asarray(y, dtype=int)

    coords = np.asarray(coords)
    if len(coords) != len(y):
        raise ValueError("coords must be same length as y")

    n = len(y)
    block = max(1, n // config.n_groups)
    offsets = _offset_schedule(block, config.n_partitions, config.random_state, config.offset_strategy)

    metric_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    params_rows: List[Dict[str, object]] = []

    # Optional progress bar
    try:
        from tqdm import tqdm  # type: ignore

        _iter = tqdm
    except Exception:
        _iter = lambda x, **_: x  # noqa: E731

    for part_idx, off in _iter(
        list(enumerate(offsets, start=1)),
        total=len(offsets),
        desc=f"blocked_cv(partitions={config.n_partitions},groups={config.n_groups})",
    ):
        groups = assign_contiguous_groups(coords, n_groups=config.n_groups, offset=int(off))

        for fold_group in range(config.n_groups):
            te_mask = groups == fold_group
            te_idx = np.where(te_mask)[0]
            tr_idx = np.where(~te_mask)[0]

            if len(te_idx) == 0 or len(tr_idx) == 0:
                continue

            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            ids_te = ids.iloc[te_idx].astype(str).tolist()

            # scale_pos_weight for XGBoost from this training split
            n_pos = max(1, int(np.sum(y_tr == 1)))
            n_neg = int(np.sum(y_tr == 0))
            spw = float(n_neg / n_pos)

            search_spaces = build_model_search_spaces(
                scale_pos_weight=spw,
                random_state=config.random_state + part_idx * 100 + fold_group,
                n_jobs=config.n_jobs,
            )

            inner = StratifiedKFold(
                n_splits=config.inner_folds,
                shuffle=True,
                random_state=config.random_state + 2000 + part_idx * 100 + fold_group,
            )

            model_test_scores: Dict[str, np.ndarray] = {}

            for model_name in models:
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
                        random_state=config.random_state + 30_000 + part_idx * 100 + fold_group,
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
                    "partition": int(part_idx),
                    "offset": int(off),
                    "fold_group": int(fold_group),
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
                                "partition": int(part_idx),
                                "offset": int(off),
                                "fold_group": int(fold_group),
                                "model": model_name,
                                "id": gid,
                                "y_true": int(yt),
                                "y_score": float(ys),
                            }
                        )

                if store_best_params and config.tune:
                    params_rows.append(
                        {
                            "partition": int(part_idx),
                            "offset": int(off),
                            "fold_group": int(fold_group),
                            "model": model_name,
                            "best_params": best_params,
                        }
                    )

            if include_ensemble and len(model_test_scores) >= 2:
                ens = np.mean(np.column_stack(list(model_test_scores.values())), axis=1)
                roc = _safe_roc_auc(y_te, ens)
                ap = float(average_precision_score(y_te, ens))
                row = {
                    "partition": int(part_idx),
                    "offset": int(off),
                    "fold_group": int(fold_group),
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
                                "partition": int(part_idx),
                                "offset": int(off),
                                "fold_group": int(fold_group),
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


def summarize_blocked_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [c for c in metrics_df.columns if c in {"roc_auc", "avg_precision"} or c.startswith("precision_at_")]

    out_rows: List[Dict[str, object]] = []
    for model, df_m in metrics_df.groupby("model", dropna=False):
        row: Dict[str, object] = {
            "model": str(model),
            "n_splits": int(len(df_m)),
            "n_partitions": int(df_m["partition"].nunique()) if "partition" in df_m else 1,
            "n_groups": int(df_m["fold_group"].nunique()) if "fold_group" in df_m else 0,
        }
        for c in metric_cols:
            row[f"{c}_mean"] = float(df_m[c].mean())
            row[f"{c}_std"] = float(df_m[c].std(ddof=1))
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(["model"]).reset_index(drop=True)


def save_blocked_area_cv_results(
    results: Dict[str, object],
    out_dir: str | Path,
    *,
    feature_set: str,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_df: pd.DataFrame = results["metrics"]  # type: ignore[assignment]
    preds_df: pd.DataFrame = results.get("predictions", pd.DataFrame())  # type: ignore[assignment]
    cfg: BlockedAreaCVConfig = results["config"]  # type: ignore[assignment]

    if not metrics_df.empty:
        metrics_df = metrics_df.copy()
        metrics_df.insert(0, "feature_set", feature_set)
        metrics_df.to_csv(out_path / "fold_metrics.csv", index=False)

    summary_df = summarize_blocked_metrics(metrics_df)
    if not summary_df.empty:
        summary_df.insert(0, "feature_set", feature_set)
        summary_df.to_csv(out_path / "summary.csv", index=False)

    if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
        preds_df = preds_df.copy()
        preds_df.insert(0, "feature_set", feature_set)
        preds_df.to_csv(out_path / "predictions.csv", index=False)

    with open(out_path / "config.json", "w") as fh:
        json.dump({"feature_set": feature_set, **asdict(cfg)}, fh, indent=2)

    if "best_params" in results:
        with open(out_path / "best_params.jsonl", "w") as fh:
            for row in results["best_params"]:  # type: ignore[index]
                fh.write(json.dumps(row) + "\n")
