"""
Data loading and light preprocessing for the FOX genes ML framework.

Design goals:
- Keep preprocessing transparent and easy to describe in the manuscript.
- Fix known CSV issues where fold-change columns may be non-numeric ("#NAME?").
- Provide a single point of truth for how training labels are derived.

Important:
- This module does NOT attempt to "fix" biological labels; it only maps FOX/NotFOX
  to {1,0} and drops Unknown for training/evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    id_col: str = "gene"
    label_col: str = "label"
    pos_label: str = "FOX"
    neg_label: str = "NotFOX"
    unk_label: str = "Unknown"


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, coercing errors to NaN."""
    return pd.to_numeric(series.replace("#NAME?", np.nan), errors="coerce")


def recompute_rna_fold_changes(
    df: pd.DataFrame,
    *,
    baseline_timepoint: int = 0,
    timepoints: Tuple[int, ...] = (6, 12, 21),
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """
    Recompute RNA fold-change features from RPKM columns to ensure numeric consistency.

    We compute a ratio-based fold change:
        FC(0->t) = (RPKM_t + pseudocount) / (RPKM_0 + pseudocount)

    Rationale:
    - Reviewer concerns include feature redundancy and technical confounding.
    - The provided CSV fold-change columns can contain non-numeric placeholders.
    - Computing FC from the RPKM columns is transparent and reproducible.

    Returns a copy of df with updated/created fold-change columns named:
        fold_change_in_rpkm_0_to_{t}_hours
    """
    out = df.copy()

    base_col = f"{baseline_timepoint}_hour_rpkm"
    if base_col not in out.columns:
        raise KeyError(f"Expected baseline RPKM column '{base_col}' not found")

    r0 = _safe_numeric(out[base_col])

    for t in timepoints:
        rt_col = f"{t}_hour_rpkm"
        if rt_col not in out.columns:
            raise KeyError(f"Expected RPKM column '{rt_col}' not found")

        rt = _safe_numeric(out[rt_col])
        fc = (rt + pseudocount) / (r0 + pseudocount)

        fc_col = f"fold_change_in_rpkm_0_to_{t}_hours"
        out[fc_col] = fc.astype(float)

    return out


def load_dataset(
    path: str,
    *,
    spec: DatasetSpec = DatasetSpec(),
    recompute_fold_changes: bool = True,
) -> pd.DataFrame:
    """
    Load the dataset CSV and apply light cleanup.

    - Recompute RNA fold-change columns from RPKM values (optional but recommended).
    - Coerce any object columns that should be numeric into numeric where possible.
    """
    df = pd.read_csv(path)

    if recompute_fold_changes:
        df = recompute_rna_fold_changes(df)

    # Ensure protein FC columns are numeric (they should already be, but be safe)
    for c in ["FC_0_to_3_protein", "FC_0_to_12_protein", "FC_0_to_24_protein"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c]).astype(float)

    # Coerce core numeric columns that sometimes show up as object
    for c in df.columns:
        if c.startswith("fold_change_in_rpkm_"):
            df[c] = _safe_numeric(df[c]).astype(float)

    return df


def get_labeled_subset(
    df: pd.DataFrame,
    *,
    spec: DatasetSpec = DatasetSpec(),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Return the labeled subset and binary labels y in {0,1}.
    """
    mask = df[spec.label_col].isin([spec.pos_label, spec.neg_label])
    labeled = df.loc[mask].copy()

    y = (labeled[spec.label_col] == spec.pos_label).astype(int).to_numpy()
    return labeled, y
