"""
Feature-set definitions for the FOX genes ML framework.

Goal (review-driven):
- Keep the code *opinionated and simple* (avoid overly-automated feature engineering).
- Reduce redundant/technically-confounded transcriptomics proxies by selecting a single,
  biologically-meaningful representation per timepoint (RPKM here), while retaining
  protein-level representation (FC_*) as a distinct modality.

This module intentionally avoids complex automated feature selection. The main
"selection" performed here is:
  1) Removing redundant RNA-seq count proxies (unique/total reads) when using the
     'principled' feature set.
  2) Optional removal of positional features for leakage checks.

If you need additional pruning, prefer:
- Regularization (LogisticRegression with L1/L2),
- Or a very light, unsupervised VarianceThreshold in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    """A named feature set and its rationale (for reporting)."""
    name: str
    description: str


BASELINE_FULL = FeatureSet(
    name="baseline_full",
    description=(
        "All numeric features (as provided) including RNA-seq read-count proxies "
        "(unique_gene_reads, total_gene_reads), RPKM, derived RNA fold-changes, "
        "protein FC features, GO one-hot terms, gene length, and positional features."
    ),
)

PRINCIPLED_EXPRESSION = FeatureSet(
    name="principled_expression",
    description=(
        "Opinionated removal of redundant RNA-seq count proxies; retain only RPKM at "
        "each timepoint plus derived RNA fold-changes, while keeping protein FC features "
        "as a separate modality. Keeps positional features."
    ),
)

PRINCIPLED_EXPRESSION_NO_POSITION = FeatureSet(
    name="principled_expression_no_position",
    description=(
        "Same as principled_expression, but removes chromosome_region_start/end to help "
        "assess positional information leakage."
    ),
)

# ---------------------------------------------------------------------------
# Optional bioinformatics / orthology summary features
#
# Reviewer-facing ablations sometimes require explicitly *removing* conservation
# features (to show the incremental contribution of comparative bioinformatics)
# and then adding them back in.
#
# These columns are produced by the BLAST/RBH workflow + augmentation scripts.
# They are intentionally kept simple summaries (counts, mean % identity, etc.).
# ---------------------------------------------------------------------------

PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION = FeatureSet(
    name="principled_expression_no_position_no_conservation",
    description=(
        "Same as principled_expression_no_position, but explicitly removes conservation/"
        "orthology summary features (e.g., RBH hit counts, mean % identity) so their "
        "incremental impact can be quantified in ablation experiments."
    ),
)

PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION = FeatureSet(
    name="principled_expression_no_position_plus_conservation",
    description=(
        "Same as principled_expression_no_position, and additionally requires that one or "
        "more conservation/orthology summary features are present (e.g., RBH hit counts and "
        "mean % identity across diazotroph proteomes)."
    ),
)


CONSERVATION_PREFIXES = (
    "diazo_",
    "nondiazo_",
)

CONSERVATION_EXACT = {
    "has_diazo_hit",
    "has_nondiazo_hit",
}


def _is_conservation_feature(col: str) -> bool:
    c = str(col)
    if c in CONSERVATION_EXACT:
        return True
    return any(c.startswith(p) for p in CONSERVATION_PREFIXES)


def _drop_if_present(cols: List[str], drop_cols: List[str]) -> List[str]:
    drop = set(drop_cols)
    return [c for c in cols if c not in drop]


def select_feature_columns(df: pd.DataFrame, *, feature_set: str) -> List[str]:
    """
    Return the feature columns to use for a given feature set.

    Notes:
    - Assumes fold-change columns have already been cleaned / recomputed as numeric.
    - Only numeric columns are used as model inputs (categoricals like go_id are excluded).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Safety: never use label or id columns even if they were numeric.
    for banned in ["label", "gene"]:
        if banned in numeric_cols:
            numeric_cols.remove(banned)

    if feature_set == BASELINE_FULL.name:
        return numeric_cols

    if feature_set in {
        PRINCIPLED_EXPRESSION.name,
        PRINCIPLED_EXPRESSION_NO_POSITION.name,
        PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION.name,
        PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION.name,
    }:
        cols = numeric_cols.copy()

        # Opinionated removal: count-based RNA-seq proxies can carry technical signal.
        # Keep RPKM and fold-change features instead.
        cols = [c for c in cols if "unique_gene_reads" not in c and "total_gene_reads" not in c]

        # DROP_FOLD_CHANGE_RPKM_FOR_PRINCIPLED
        # Principled sets should NOT include RNA fold-change features (e.g., fold_change_in_rpkm_0_to_*_hours).
        cols = [c for c in cols if "fold_change_in_rpkm" not in c]

        if feature_set in {
            PRINCIPLED_EXPRESSION_NO_POSITION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION.name,
            PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION.name,
        }:
            cols = _drop_if_present(cols, ["chromosome_region_start", "chromosome_region_end"])

        if feature_set == PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION.name:
            cols = [c for c in cols if not _is_conservation_feature(c)]

        if feature_set == PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION.name:
            cons_cols = [c for c in cols if _is_conservation_feature(c)]
            if len(cons_cols) == 0:
                raise ValueError(
                    "Feature set 'principled_expression_no_position_plus_conservation' "
                    "requires conservation/orthology features (e.g., diazo_hit_count, "
                    "diazo_mean_pident). None were found among numeric columns. "
                    "If you have not yet augmented the feature matrix, run the "
                    "augmentation script (ml/scripts/augment_model_with_crosswalk_and_conservation.py) "
                    "or provide a prepared CSV that includes these columns."
                )

        return cols

    raise ValueError(
        f"Unknown feature_set='{feature_set}'. Valid options: "
        f"{BASELINE_FULL.name}, {PRINCIPLED_EXPRESSION.name}, {PRINCIPLED_EXPRESSION_NO_POSITION.name}, "
        f"{PRINCIPLED_EXPRESSION_NO_POSITION_NO_CONSERVATION.name}, "
        f"{PRINCIPLED_EXPRESSION_NO_POSITION_PLUS_CONSERVATION.name}"
    )
