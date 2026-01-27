#!/usr/bin/env python3
"""Figure 5 SHAP plots with labeled narrative examples (Reviewer-facing).

Key design points
-----------------
1) Produces high-resolution SHAP panels suitable for assembling Figure 5:
   - Global bar (mean |SHAP|)
   - Global beeswarm
   - Decision plot with *named example genes*
   - Waterfall plots for each example gene

2) Avoids a known SHAP<->XGBoost compatibility bug where
   `shap.TreeExplainer(XGBClassifier)` fails parsing `base_score` when XGBoost
   serializes it as a list (e.g., "[5E-1]"). Instead, we compute contributions
   using XGBoost's native TreeSHAP via `Booster.predict(pred_contribs=True)`.

3) SHAP values computed this way are in *raw margin* space (log-odds for
   binary logistic XGBoost). The decision plot is rendered with link='logit'
   so the x-axis is probability, while retaining additive log-odds decomposition.

Example
-------
python ml/fig5_shap_examples.py \
  --data ml/data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression_no_position \
  --out outputs/fig5_shap \
  --include_unlabeled_examples \
  --dpi 600

You can pass explicit narrative genes (recommended for Reviewer 1):

python ml/fig5_shap_examples.py \
  --data ml/data/prepared_for_modeling.plus_conservation_v2.csv \
  --feature_set principled_expression_no_position \
  --out outputs/fig5_shap \
  --fox_examples all1457 alr1407 \
  --notfox_examples all9999 alr1234 \
  --dpi 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import xgboost as xgb
from xgboost import XGBClassifier


def _ensure_ml_dir_on_path() -> None:
    here = Path(__file__).resolve().parent
    if (here / "src").exists() and str(here) not in sys.path:
        sys.path.insert(0, str(here))


_ensure_ml_dir_on_path()

try:
    import shap  # noqa: E402
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "SHAP is not installed. Install it (pip install shap) and re-run. "
        f"Original error: {e}"
    )

from src.data_prep import DatasetSpec, get_labeled_subset, load_dataset  # noqa: E402
from src.feature_sets import select_feature_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Prepared feature matrix CSV")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--feature_set",
        default="principled_expression_no_position",
        help="Feature set name (baseline_full, principled_expression, principled_expression_no_position, ...)"
    )

    ap.add_argument("--id_col", default="gene")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--pos_label", default="FOX")
    ap.add_argument("--neg_label", default="NotFOX")
    ap.add_argument("--unk_label", default="Unknown")

    ap.add_argument("--fox_examples", nargs="*", default=None)
    ap.add_argument("--notfox_examples", nargs="*", default=None)
    ap.add_argument(
        "--include_unlabeled_examples",
        action="store_true",
        help="Also include top/bottom predicted unlabeled genes as examples",
    )
    ap.add_argument(
        "--n_auto",
        type=int,
        default=2,
        help="If no explicit examples provided, number to auto-pick per class",
    )

    ap.add_argument("--max_display", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument(
        "--max_shap_samples",
        type=int,
        default=0,
        help="If >0, subsample labeled rows for global SHAP summary",
    )

    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)

    return ap.parse_args()


def _parse_gene_list(vals: Optional[List[str]]) -> List[str]:
    if not vals:
        return []
    out: List[str] = []
    for v in vals:
        v = str(v).strip()
        if v:
            out.append(v)
    return out


def fit_xgb_legacy(X: pd.DataFrame, y: np.ndarray, *, random_state: int, n_jobs: int) -> XGBClassifier:
    """Fit XGBoost with fixed hyperparameters (legacy-aligned defaults).

    We keep this stable to make figure interpretation reproducible.
    """
    y = np.asarray(y, dtype=int)
    n_pos = max(1, int(np.sum(y == 1)))
    n_neg = int(np.sum(y == 0))
    scale_pos_weight = float(n_neg / n_pos)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=max(1, int(n_jobs)),
        random_state=int(random_state),
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )
    model.fit(X, y)
    return model


def xgb_pred_contribs(
    booster: xgb.Booster,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """Return (shap_values, base_value) using XGBoost's built-in TreeSHAP.

    `pred_contribs=True` returns an array of shape (n, n_features+1), where the
    last column is the bias term. The row sum equals the *raw margin*.
    """
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    contrib = booster.predict(dm, pred_contribs=True)
    contrib = np.asarray(contrib, dtype=float)

    if contrib.ndim != 2 or contrib.shape[1] != (X.shape[1] + 1):
        raise RuntimeError(
            f"Unexpected pred_contribs shape {contrib.shape}; expected (n, {X.shape[1] + 1})."
        )

    shap_vals = contrib[:, :-1]
    bias = contrib[:, -1]
    base_value = float(np.mean(bias))
    return shap_vals, base_value


def auto_pick_examples(
    ids: pd.Series,
    y: np.ndarray,
    prob: np.ndarray,
    *,
    n_auto: int,
) -> Tuple[List[str], List[str]]:
    df = pd.DataFrame({"id": ids.astype(str).tolist(), "y": y.astype(int).tolist(), "prob": prob.astype(float).tolist()})
    fox = df.loc[df["y"] == 1].sort_values("prob", ascending=False)["id"].tolist()
    notfox = df.loc[df["y"] == 0].sort_values("prob", ascending=True)["id"].tolist()
    return fox[:n_auto], notfox[:n_auto]


def pick_unlabeled_examples(
    df: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    unk_label: str,
    prob: np.ndarray,
    n_auto: int,
) -> Tuple[List[str], List[str]]:
    mask = df[label_col].astype(str) == str(unk_label)
    if not mask.any():
        return [], []
    tmp = df.loc[mask, [id_col]].copy()
    tmp["prob"] = prob[mask]
    top = tmp.sort_values("prob", ascending=False)[id_col].astype(str).head(n_auto).tolist()
    bot = tmp.sort_values("prob", ascending=True)[id_col].astype(str).head(n_auto).tolist()
    return top, bot


def _save_fig(path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main() -> None:
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = DatasetSpec(
        id_col=a.id_col,
        label_col=a.label_col,
        pos_label=a.pos_label,
        neg_label=a.neg_label,
        unk_label=a.unk_label,
    )

    df = load_dataset(a.data, spec=spec)

    labeled_df, y = get_labeled_subset(df, spec=spec)
    ids_lab = labeled_df[spec.id_col].astype(str)

    feat_cols = select_feature_columns(labeled_df, feature_set=a.feature_set)
    X_lab = labeled_df[feat_cols].copy()

    model = fit_xgb_legacy(X_lab, y, random_state=a.random_state, n_jobs=a.n_jobs)
    booster = model.get_booster()

    prob_lab = model.predict_proba(X_lab)[:, 1].astype(float)

    # Full-dataset probability (for unlabeled example selection)
    prob_all = model.predict_proba(df[feat_cols])[:, 1].astype(float)

    fox_examples = _parse_gene_list(a.fox_examples)
    notfox_examples = _parse_gene_list(a.notfox_examples)

    if not fox_examples or not notfox_examples:
        fox_auto, notfox_auto = auto_pick_examples(ids_lab, y, prob_lab, n_auto=int(a.n_auto))
        if not fox_examples:
            fox_examples = fox_auto
        if not notfox_examples:
            notfox_examples = notfox_auto

    pred_pos, pred_neg = [], []
    if a.include_unlabeled_examples:
        pred_pos, pred_neg = pick_unlabeled_examples(
            df,
            id_col=spec.id_col,
            label_col=spec.label_col,
            unk_label=spec.unk_label,
            prob=prob_all,
            n_auto=int(a.n_auto),
        )

    example_ids = list(dict.fromkeys(fox_examples + notfox_examples + pred_pos + pred_neg))

    id_to_row = df.set_index(spec.id_col, drop=False)
    missing = [gid for gid in example_ids if gid not in id_to_row.index]
    if missing:
        raise SystemExit("Example gene IDs not found in dataset: " + ", ".join(missing))

    ex_df = id_to_row.loc[example_ids].copy()
    X_ex = ex_df[feat_cols].copy()
    prob_ex = model.predict_proba(X_ex)[:, 1].astype(float)

    def _example_class(i: int) -> str:
        lab = str(ex_df.iloc[i][spec.label_col])
        if lab == spec.pos_label:
            return "FOX (labeled)"
        if lab == spec.neg_label:
            return "NotFOX (labeled)"
        if lab == spec.unk_label:
            return "Pred FOX" if prob_ex[i] >= 0.5 else "Pred NotFOX"
        return lab

    ex_labels = [_example_class(i) for i in range(len(example_ids))]

    ex_table = pd.DataFrame(
        {
            "gene": example_ids,
            "label": [str(ex_df.iloc[i][spec.label_col]) for i in range(len(example_ids))],
            "example_class": ex_labels,
            "p_fox": prob_ex,
        }
    )
    ex_table.to_csv(out_dir / "fig5_examples_table.csv", index=False)

    # ---- Global SHAP (subsample if requested) ----
    X_sum = X_lab
    if a.max_shap_samples and int(a.max_shap_samples) > 0 and len(X_lab) > int(a.max_shap_samples):
        X_sum = X_lab.sample(int(a.max_shap_samples), random_state=a.random_state)

    shap_vals_sum, base_value = xgb_pred_contribs(booster, X_sum)
    shap_vals_ex, _ = xgb_pred_contribs(booster, X_ex)

    # Global bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals_sum, X_sum, plot_type="bar", max_display=int(a.max_display), show=False)
    plt.title(f"Global feature importance (mean |SHAP|) — feature_set={a.feature_set}")
    _save_fig(out_dir / "fig5_shap_bar.png", dpi=a.dpi)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals_sum, X_sum, plot_type="bar", max_display=int(a.max_display), show=False)
    plt.title(f"Global feature importance (mean |SHAP|) — feature_set={a.feature_set}")
    _save_fig(out_dir / "fig5_shap_bar.pdf", dpi=a.dpi)

    # Global beeswarm
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals_sum, X_sum, max_display=int(a.max_display), show=False)
    plt.title(f"Global SHAP distribution (raw margin/log-odds) — feature_set={a.feature_set}")
    _save_fig(out_dir / "fig5_shap_beeswarm.png", dpi=a.dpi)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals_sum, X_sum, max_display=int(a.max_display), show=False)
    plt.title(f"Global SHAP distribution (raw margin/log-odds) — feature_set={a.feature_set}")
    _save_fig(out_dir / "fig5_shap_beeswarm.pdf", dpi=a.dpi)

    # Decision plot for named examples
    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        base_value,
        shap_vals_ex,
        X_ex,
        feature_names=X_ex.columns.tolist(),
        link="logit",
        show=False,
    )
    ax = plt.gca()
    lines = ax.get_lines()
    for i, ln in enumerate(lines[: len(example_ids)]):
        ln.set_label(f"{example_ids[i]} ({ex_labels[i]}) p={prob_ex[i]:.2f}")
    ax.legend(loc="best", fontsize=7, frameon=False)
    plt.title("SHAP decision plot (examples; x-axis is probability via logit link)")
    _save_fig(out_dir / "fig5_shap_decision_examples.png", dpi=a.dpi)

    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        base_value,
        shap_vals_ex,
        X_ex,
        feature_names=X_ex.columns.tolist(),
        link="logit",
        show=False,
    )
    ax = plt.gca()
    lines = ax.get_lines()
    for i, ln in enumerate(lines[: len(example_ids)]):
        ln.set_label(f"{example_ids[i]} ({ex_labels[i]}) p={prob_ex[i]:.2f}")
    ax.legend(loc="best", fontsize=7, frameon=False)
    plt.title("SHAP decision plot (examples; x-axis is probability via logit link)")
    _save_fig(out_dir / "fig5_shap_decision_examples.pdf", dpi=a.dpi)

    # Waterfall plots per example
    for i, gid in enumerate(example_ids):
        vals = shap_vals_ex[i]
        data_row = X_ex.iloc[i]
        try:
            exp = shap.Explanation(
                values=vals,
                base_values=base_value,
                data=data_row.to_numpy(),
                feature_names=X_ex.columns.tolist(),
            )
            plt.figure(figsize=(8, 5))
            shap.plots.waterfall(exp, max_display=int(a.max_display), show=False)
        except Exception:
            plt.figure(figsize=(8, 5))
            shap.waterfall_plot(base_value, vals, data_row, max_display=int(a.max_display), show=False)

        plt.title(f"{gid} ({ex_labels[i]}) — p(FOX)={prob_ex[i]:.3f} — SHAP in log-odds", fontsize=10)
        _save_fig(out_dir / f"fig5_waterfall_{gid}.png", dpi=a.dpi)

        # PDF
        try:
            exp = shap.Explanation(
                values=vals,
                base_values=base_value,
                data=data_row.to_numpy(),
                feature_names=X_ex.columns.tolist(),
            )
            plt.figure(figsize=(8, 5))
            shap.plots.waterfall(exp, max_display=int(a.max_display), show=False)
        except Exception:
            plt.figure(figsize=(8, 5))
            shap.waterfall_plot(base_value, vals, data_row, max_display=int(a.max_display), show=False)

        plt.title(f"{gid} ({ex_labels[i]}) — p(FOX)={prob_ex[i]:.3f} — SHAP in log-odds", fontsize=10)
        _save_fig(out_dir / f"fig5_waterfall_{gid}.pdf", dpi=a.dpi)

    note = (
        "SHAP values for Fig 5 were computed using XGBoost native TreeSHAP via "
        "Booster.predict(pred_contribs=True). The output is in raw margin space "
        "(log-odds for binary logistic XGBoost); the last column returned by pred_contribs "
        "is the bias term, and the row-sum equals the raw (untransformed) margin. "
        "Decision plots were rendered with link='logit' so the x-axis is probability.\n"
    )
    (out_dir / "fig5_shap_output_note.txt").write_text(note)

    print("✓ Figure 5 SHAP outputs written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
