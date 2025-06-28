#
# Requirements
# ------------
#   pip install pandas numpy shap matplotlib scikit-learn joblib xgboost \
#               statsmodels tqdm seaborn

import argparse, warnings, joblib
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams.update({"figure.autolayout": True})

# --------------------------------------------------------------------------- #
#                              Helper functions                               #
# --------------------------------------------------------------------------- #
def _ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()

def _plot_flag_distribution(flag_series, shap_values, out_path):
    """Box + swarm plot for a binary flag feature."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(4, 4))
    order = [0, 1]
    sns.boxplot(x=flag_series, y=shap_values, order=order, ax=ax)
    sns.stripplot(
        x=flag_series,
        y=shap_values,
        order=order,
        color="black",
        size=3,
        jitter=0.25,
        alpha=0.25,
        ax=ax,
    )
    ax.set_xlabel("divergent_promoter_flag (0 = no, 1 = yes)")
    ax.set_ylabel("SHAP value")
    ax.set_title("Effect of divergent_promoter_flag")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _add_smooth_trend(ax, x_vals, y_vals, frac):
    """LOWESS moving-average trend line (red)."""
    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
    if mask.sum() < 10:
        return
    smooth = lowess(y_vals[mask], x_vals[mask], frac=frac, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color="red", linewidth=2)

# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Generate SHAP dependence plots with red LOWESS trend lines "
        "for selected variables from a saved XGBoost model."
    )
    ap.add_argument("--data", required=True, help="CSV containing feature matrix")
    ap.add_argument("--model", required=True, help="Path to XGBoost .pkl (`joblib.dump`)")
    ap.add_argument("--out", default="effect_plots", help="Directory for PNG outputs")
    ap.add_argument(
        "--variables",
        nargs="+",
        default=[
            "21_hour_rpkm",
            "chromosome_region_start",
            "divergent_promoter_flag",
            "divergent_promoter_distance",
        ],
        help="Variable names to visualise",
    )
    ap.add_argument(
        "--smooth_frac",
        default=0.15,
        type=float,
        help="LOWESS fraction (0 < frac ≤ 1)",
    )
    args = ap.parse_args()

    out_dir = _ensure_out_dir(Path(args.out))
    print(f"[INFO] Output plots → {out_dir}")

    # --------------------------------------------------------------------- #
    #   1. Load model and data                                              #
    # --------------------------------------------------------------------- #
    print("[INFO] Loading model …")
    model = joblib.load(args.model)
    feature_names = model.get_booster().feature_names
    if feature_names is None:
        raise RuntimeError(
            "Model lacks feature names – re-train with correct feature_names."
        )

    print("[INFO] Reading data …")
    df = pd.read_csv(args.data)

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(
            "Missing feature columns in the data file: "
            f"{', '.join(sorted(missing))}"
        )
    X = df[feature_names]

    # --------------------------------------------------------------------- #
    #   2. SHAP value computation                                           #
    # --------------------------------------------------------------------- #
    print("[INFO] Computing SHAP values …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --------------------------------------------------------------------- #
    #   3. Generate plots                                                   #
    # --------------------------------------------------------------------- #
    for var in args.variables:
        if var not in feature_names:
            warnings.warn(f"Variable {var} not found in model – skipped.")
            continue

        print(f"[INFO] Plotting {var} …")

        # a) SHAP dependence plot
        fig_out = out_dir / f"{var}_shap_dependence.png"
        shap.dependence_plot(
            var,
            shap_values,
            X,
            interaction_index=None,
            show=False,
            alpha=0.35,        # lighter points
            dot_size=6,        # smaller dots
        )
        ax = plt.gca()
        ax.set_title(f"SHAP dependence – {var}")

        # Overlay red LOWESS trend line for numeric vars
        if np.issubdtype(X[var].dtype, np.number) and var != "divergent_promoter_flag":
            _add_smooth_trend(
                ax,
                X[var].values,
                shap_values[:, feature_names.index(var)],
                frac=args.smooth_frac,
            )

        plt.gcf().set_size_inches(6, 4)
        plt.savefig(fig_out, dpi=300)
        plt.close()

        # b) Special treatment for binary flag
        if var == "divergent_promoter_flag":
            box_out = out_dir / "divergent_promoter_flag_distribution.png"
            _plot_flag_distribution(
                X[var], shap_values[:, feature_names.index(var)], box_out
            )

    print("✓ All plots generated.")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
