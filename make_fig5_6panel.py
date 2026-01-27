# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
make_fig5_6panel.py

Publication-grade 6-panel TreeSHAP figure for Figure 5.

Panels a-d: Main SHAP dependence plots (colored by class)
Panel e: 21h RPKM colored by interaction with non-diazo hit
Panel f: Divergent promoter colored by interaction with 21h RPKM
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

try:
    import numpy.core as npcore
    import numpy.core._multiarray_umath as _mu
    sys.modules.setdefault("numpy._core", npcore)
    sys.modules.setdefault("numpy._core._multiarray_umath", _mu)
except Exception:
    pass


def normalize_label(x):
    s = str(x).strip().lower()
    if s in {"fox", "positive", "pos", "1", "true"}:
        return "FOX"
    if s in {"notfox", "nonfox", "negative", "neg", "0", "false", "non-fox", "not_fox"}:
        return "NotFOX"
    return "Unknown"


def auto_find_feature_matrix(path_hint):
    p = Path(path_hint)
    if p.exists():
        return str(p)
    candidates = glob.glob("**/prepared_for_modeling.plus_conservation_v2.csv", recursive=True)
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(f"Feature matrix not found: '{path_hint}'.")


def compute_xgb_contribs(pipeline, X_df):
    from xgboost import DMatrix
    imputer = pipeline.named_steps["impute"]
    var = pipeline.named_steps["var"]
    clf = pipeline.named_steps["clf"]
    X_imp = imputer.transform(X_df)
    support = var.get_support()
    feat_names = np.array(X_df.columns)[support]
    X_var = var.transform(X_imp)
    booster = clf.get_booster()
    dm = DMatrix(X_var, feature_names=feat_names.tolist())
    contribs = booster.predict(dm, pred_contribs=True)
    shap_vals = contribs[:, :-1]
    shap_df = pd.DataFrame(shap_vals, index=X_df.index, columns=feat_names)
    X_used = pd.DataFrame(X_var, index=X_df.index, columns=feat_names)
    return shap_df, X_used, feat_names


def compute_interactions(pipeline, X_df):
    from xgboost import DMatrix
    imputer = pipeline.named_steps["impute"]
    var = pipeline.named_steps["var"]
    clf = pipeline.named_steps["clf"]
    X_imp = imputer.transform(X_df)
    support = var.get_support()
    feat_names = np.array(X_df.columns)[support]
    X_var = var.transform(X_imp)
    booster = clf.get_booster()
    dm = DMatrix(X_var, feature_names=feat_names.tolist())
    print("[INFO] Computing SHAP interaction values...")
    interaction_values = booster.predict(dm, pred_interactions=True)
    interaction_values = interaction_values[:, :-1, :-1]
    X_transformed = pd.DataFrame(X_var, index=X_df.index, columns=feat_names)
    return interaction_values, X_transformed, feat_names


FEATURE_EXAMPLES = {
    "21_hour_rpkm": {
        "FOX": ["all1455", "alr3546"],
        "NotFOX": ["alr0022", "all5306"],
        "Unknown": ["alr1407", "all5342", "all1457", "alr2839"],
    },
    "chromosome_region_end": {
        "FOX": ["all1455", "alr3546"],
        "NotFOX": ["alr0022", "all5306"],
        "Unknown": ["all1457", "all5342", "alr2839", "alr1407"],
    },
    "divergent_promoter_distance": {
        "FOX": ["all1455", "alr3546"],
        "NotFOX": ["all5306"],
        "Unknown": ["all5342", "alr1407", "all1457"],
    },
    "has_nondiazo_hit": {
        "FOX": ["all1455", "alr3546"],
        "NotFOX": ["alr0022", "all5306"],
        "Unknown": ["all1457", "alr1407", "all5342"],
    },
}


def get_examples(feature, meta):
    if feature not in FEATURE_EXAMPLES:
        return []
    examples = []
    for label, genes in FEATURE_EXAMPLES[feature].items():
        for g in genes:
            if g in meta.index and g not in examples:
                examples.append(g)
    return examples


def plot_main_panel(ax, feature, shap_df, X_used, meta, example_genes,
                    seed=1, divergent_sentinel=9999, show_legend=False, legend_loc="lower right"):
    """Standard SHAP dependence panel (a-d)."""
    df = pd.DataFrame(
        {"x": X_used[feature], "shap": shap_df[feature]},
        index=X_used.index,
    ).join(meta[["label"]], how="left")

    COLORS = {"Unknown": "#888888", "NotFOX": "#0072B2", "FOX": "#E69F00"}
    
    if feature in {"chromosome_region_end", "21_hour_rpkm"}:
        ALPHA = {"Unknown": 0.08, "NotFOX": 0.35, "FOX": 0.65}
    else:
        ALPHA = {"Unknown": 0.06, "NotFOX": 0.30, "FOX": 0.55}
    SIZE = {"Unknown": 8, "NotFOX": 10, "FOX": 12}

    rng = np.random.default_rng(seed)
    df["x_plot"] = df["x"].astype(float)
    
    nice_labels = {
        "21_hour_rpkm": "21 h RPKM",
        "chromosome_region_end": "Chromosome region end",
        "divergent_promoter_distance": "Divergent promoter distance",
        "has_nondiazo_hit": "Has non-diazotroph hit (0/1)",
    }
    x_label = nice_labels.get(feature, feature)

    if feature == "has_nondiazo_hit":
        df["x_plot"] = df["x_plot"] + rng.uniform(-0.07, 0.07, size=len(df))
    if feature == "chromosome_region_end":
        df["x_plot"] = df["x_plot"] / 1e6
        x_label = "Chromosome region end (Mb)"
    
    NONE_X = -0.4
    if feature == "divergent_promoter_distance":
        x = df["x"].astype(float)
        none_mask = x >= divergent_sentinel
        df.loc[~none_mask, "x_plot"] = x[~none_mask] / 1000.0
        df.loc[none_mask, "x_plot"] = NONE_X + rng.uniform(-0.06, 0.06, size=none_mask.sum())
        x_label = "Divergent promoter distance (kb)"

    for lab in ["Unknown", "NotFOX", "FOX"]:
        sub = df[df["label"] == lab]
        if sub.empty:
            continue
        ax.scatter(sub["x_plot"], sub["shap"], s=SIZE[lab], c=COLORS[lab],
                  alpha=ALPHA[lab], marker="o", edgecolors="none", rasterized=True,
                  zorder=1 if lab == "Unknown" else 2)

    ax.axhline(0, color="black", lw=0.8, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel("TreeSHAP contribution\n(log-odds)")
    ax.set_title(nice_labels.get(feature, feature), fontweight='bold')

    if feature == "21_hour_rpkm":
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_xlabel("21 h RPKM (symlog)")
    if feature == "divergent_promoter_distance":
        ax.set_xlim(-0.7, 2.2)
        ax.set_xticks([NONE_X, 0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xticklabels(["None", "0", "0.5", "1", "1.5", "2"])
    if feature == "has_nondiazo_hit":
        ax.set_xticks([0, 1])
        ax.set_xlim(-0.25, 1.25)

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin - 0.02 * y_range, ymax + 0.04 * y_range)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w", label="Unknown",
                   markerfacecolor=COLORS["Unknown"], markersize=7, alpha=0.85),
            Line2D([0], [0], marker="o", color="w", label="NotFOX",
                   markerfacecolor=COLORS["NotFOX"], markersize=7, alpha=0.95),
            Line2D([0], [0], marker="o", color="w", label="FOX",
                   markerfacecolor=COLORS["FOX"], markersize=7, alpha=0.95),
        ]
        ax.legend(handles=handles, frameon=False, loc=legend_loc,
                 handletextpad=0.4, labelspacing=0.35, fontsize=10)

    chosen = [g for g in example_genes if g in df.index]
    for g in chosen:
        row = df.loc[g]
        lab = row["label"]
        ax.scatter([row["x_plot"]], [row["shap"]], s=130,
                  c=COLORS.get(lab, "black"), alpha=1.0,
                  edgecolors="black", linewidths=1.1, zorder=10)


def plot_interaction_panel(ax, feature_x, feature_color, X_transformed, 
                           interaction_values, feat_names, meta, example_genes,
                           color_label_short, seed=1, divergent_sentinel=9999):
    """Interaction panel with highlighted genes and clean colorbar."""
    rng = np.random.default_rng(seed)
    
    idx_x = np.where(feat_names == feature_x)[0][0]
    idx_c = np.where(feat_names == feature_color)[0][0]
    
    x_vals = X_transformed[feature_x].values.copy()
    interaction_vals = interaction_values[:, idx_x, idx_c]
    total_effect = interaction_values[:, idx_x, :].sum(axis=1)
    
    df = pd.DataFrame({
        'x_raw': x_vals,
        'interaction': interaction_vals,
        'total_effect': total_effect,
    }, index=X_transformed.index)
    
    x_plot = x_vals.copy()
    
    nice_x = {
        "21_hour_rpkm": "21 h RPKM",
        "chromosome_region_end": "Chr. position",
        "divergent_promoter_distance": "Divergent promoter distance",
        "has_nondiazo_hit": "Non-diazo hit",
    }
    
    x_label = nice_x.get(feature_x, feature_x)
    
    NONE_X = -0.4
    if feature_x == "chromosome_region_end":
        x_plot = x_plot / 1e6
        x_label = "Chromosome position (Mb)"
    elif feature_x == "divergent_promoter_distance":
        none_mask = x_vals >= divergent_sentinel
        x_plot[~none_mask] = x_vals[~none_mask] / 1000.0
        x_plot[none_mask] = NONE_X + rng.uniform(-0.06, 0.06, size=none_mask.sum())
        x_label = "Divergent promoter distance (kb)"
    elif feature_x == "has_nondiazo_hit":
        x_plot = x_plot + rng.uniform(-0.07, 0.07, size=len(x_plot))
        x_label = "Has non-diazo hit"
    elif "rpkm" in feature_x.lower():
        x_label = "21 h RPKM (symlog)"
    
    df['x_plot'] = x_plot
    
    # Symmetric color scale
    vmin, vmax = np.percentile(interaction_vals, [2, 98])
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax
    
    scatter = ax.scatter(x_plot, total_effect, c=interaction_vals,
                        cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        s=8, alpha=0.6, rasterized=True)
    
    ax.axhline(0, color='black', lw=0.8, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel("TreeSHAP contribution\n(log-odds)")
    
    if "rpkm" in feature_x.lower():
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    if feature_x == "divergent_promoter_distance":
        ax.set_xlim(-0.7, 2.2)
        ax.set_xticks([NONE_X, 0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xticklabels(["None", "0", "0.5", "1", "1.5", "2"])
    if feature_x == "has_nondiazo_hit":
        ax.set_xticks([0, 1])
        ax.set_xlim(-0.25, 1.25)
    
    # Colorbar - horizontal at top to avoid overlap
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       shrink=0.6, pad=0.02, aspect=25,
                       location='top')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(color_label_short, fontsize=9, labelpad=2)
    
    ax.set_title(f'{nice_x.get(feature_x, feature_x)}', fontweight='bold', pad=35)
    
    # Highlight example genes
    chosen = [g for g in example_genes if g in df.index]
    for g in chosen:
        row = df.loc[g]
        int_val = row['interaction']
        norm_val = (int_val - vmin) / (vmax - vmin)
        norm_val = np.clip(norm_val, 0, 1)
        cmap = plt.cm.RdBu_r
        gene_color = cmap(norm_val)
        
        ax.scatter([row['x_plot']], [row['total_effect']], s=130,
                  c=[gene_color], alpha=1.0,
                  edgecolors="black", linewidths=1.3, zorder=10)
    
    print(f"\n  Highlighted genes in {feature_x} x {feature_color} panel:")
    for g in chosen:
        row = df.loc[g]
        lab = meta.loc[g, 'label'] if g in meta.index else 'Unknown'
        print(f"    {g}: x_plot={row['x_plot']:.3f}, y={row['total_effect']:.3f}, "
              f"interaction={row['interaction']:.4f}, label={lab}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_joblib", required=True)
    ap.add_argument("--feature_matrix", required=True)
    ap.add_argument("--predictions_all", required=True)
    ap.add_argument("--outdir", default="fig5_6panel_output")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--divergent_sentinel", type=int, default=9999)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pipeline = joblib.load(args.model_joblib)
    args.feature_matrix = auto_find_feature_matrix(args.feature_matrix)

    df_feat = pd.read_csv(args.feature_matrix)
    df_pred = pd.read_csv(args.predictions_all)
    df_pred["label"] = df_pred["label"].apply(normalize_label)

    meta = df_pred[["gene", "label"]].set_index("gene")

    feature_list = list(pipeline.feature_names_in_)
    df_feat_idx = df_feat.set_index("gene")
    X = df_feat_idx[feature_list]
    
    shap_df, X_used, feat_names = compute_xgb_contribs(pipeline, X)
    interaction_values, X_transformed, feat_names = compute_interactions(pipeline, X)
    
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.30)

    # Row 1: (a) 21h RPKM, (b) Chromosome position
    ax_a = fig.add_subplot(gs[0, 0])
    plot_main_panel(ax_a, "21_hour_rpkm", shap_df, X_used, meta,
                   get_examples("21_hour_rpkm", meta), show_legend=True, legend_loc="lower right")
    ax_a.text(-0.12, 1.05, 'a', transform=ax_a.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

    ax_b = fig.add_subplot(gs[0, 1])
    plot_main_panel(ax_b, "chromosome_region_end", shap_df, X_used, meta,
                   get_examples("chromosome_region_end", meta), show_legend=False)
    ax_b.text(-0.12, 1.05, 'b', transform=ax_b.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

    # Row 2: (c) Divergent promoter, (d) Non-diazo hit
    ax_c = fig.add_subplot(gs[1, 0])
    plot_main_panel(ax_c, "divergent_promoter_distance", shap_df, X_used, meta,
                   get_examples("divergent_promoter_distance", meta), show_legend=False)
    ax_c.text(-0.12, 1.05, 'c', transform=ax_c.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

    ax_d = fig.add_subplot(gs[1, 1])
    plot_main_panel(ax_d, "has_nondiazo_hit", shap_df, X_used, meta,
                   get_examples("has_nondiazo_hit", meta), show_legend=False, legend_loc="lower left")
    ax_d.text(-0.12, 1.05, 'd', transform=ax_d.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

    # Row 3: Interaction panels
    ax_e = fig.add_subplot(gs[2, 0])
    print("\nPanel e: 21h RPKM effect, colored by interaction with non-diazo hit")
    plot_interaction_panel(ax_e, "21_hour_rpkm", "has_nondiazo_hit",
                          X_transformed, interaction_values, feat_names, meta,
                          get_examples("21_hour_rpkm", meta),
                          color_label_short="Interaction w/ non-diazo hit")
    ax_e.text(-0.12, 1.12, 'e', transform=ax_e.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

    ax_f = fig.add_subplot(gs[2, 1])
    print("\nPanel f: Divergent promoter effect, colored by interaction with 21h RPKM")
    plot_interaction_panel(ax_f, "divergent_promoter_distance", "21_hour_rpkm",
                          X_transformed, interaction_values, feat_names, meta,
                          get_examples("divergent_promoter_distance", meta),
                          color_label_short="Interaction w/ 21h RPKM")
    ax_f.text(-0.12, 1.12, 'f', transform=ax_f.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

    out_base = os.path.join(args.outdir, "fig5_6panel")
    fig.savefig(out_base + ".png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"\n[DONE] Saved 6-panel figure to: {out_base}.png and .pdf")


if __name__ == "__main__":
    main()