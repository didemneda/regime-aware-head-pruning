"""Publication figures from runs/analysis/ outputs.

Categorical palette (fixed assignment, validated defaults):
  baseline gray, static blue, dynamic orange, random aqua,
  magnitude yellow, scratch magenta.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RUNS_DIR, SWEEP_KEEPS, TOTAL_HEADS

ANALYSIS_DIR = RUNS_DIR / "analysis"
FIG_DIR = ANALYSIS_DIR / "figures"

COLORS = {
    "baseline": "#5f5e59",
    "static": "#2a78d6",
    "dynamic_joint": "#eb6834",
    "random": "#1baf7a",
    "magnitude": "#eda100",
    "scratch_static": "#e87ba4",
    "scratch_regime": "#4a3aa7",
}
LABELS = {
    "baseline": "Unpruned baseline",
    "static": "Static (global importance)",
    "dynamic_joint": "Dynamic joint (regime-aware)",
    "random": "Random (mean of 3)",
    "magnitude": "Magnitude",
    "static_greedy": "Static greedy (control)",
}
COLORS["static_greedy"] = "#4a3aa7"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def fig_importance_heatmap():
    """Regime x head importance, seed-averaged, one panel per dataset@96."""
    datasets = sorted({p.parts[-5] for p in RUNS_DIR.glob(
        "*/pred96/seed_*/pruning/head_importance.csv")})
    fig, axes = plt.subplots(len(datasets), 3, figsize=(10, 2.6 * len(datasets)),
                             squeeze=False)
    for row, dataset in enumerate(datasets):
        frames = [pd.read_csv(p) for p in RUNS_DIR.glob(
            f"{dataset}/pred96/seed_*/pruning/head_importance.csv")]
        mean = sum(f[["trend_importance", "seasonal_importance",
                      "residual_importance"]].to_numpy() for f in frames
                   ) / len(frames)
        vmax = np.abs(mean).max()
        for col, regime in enumerate(["trend", "seasonal", "residual"]):
            ax = axes[row][col]
            grid = mean[:, col].reshape(3, 8)
            im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(8))
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"L{i}" for i in range(3)])
            ax.grid(False)
            if row == 0:
                ax.set_title(f"{regime}-dominant", fontsize=9)
            if col == 0:
                ax.set_ylabel(dataset)
        fig.colorbar(im, ax=axes[row], shrink=0.8, pad=0.01,
                     label="Δ val MSE when head removed")
    fig.suptitle("Head importance is regime-dependent "
                 "(mean over 5 seeds, pred_len=96)", y=1.0)
    fig.savefig(FIG_DIR / "fig1_importance_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig1_importance_heatmap.png")
    plt.close(fig)


def fig_sparsity_sweep():
    agg = pd.read_csv(ANALYSIS_DIR / "sweep_summary.csv")
    test = agg[(agg.split == "test") & (agg.pred_len == 96)]
    datasets = sorted(test.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(3.2 * len(datasets), 3.0),
                             squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = test[test.dataset == dataset]
        base = sub[sub.method_group == "baseline"].mse_mean.iloc[0]
        ax.axhline(base, color=COLORS["baseline"], linewidth=1.5,
                   linestyle="--", label=LABELS["baseline"])
        for method in ["static", "dynamic_joint", "static_greedy",
                       "random", "magnitude"]:
            rows = (sub[sub.method_group == method]
                    .sort_values("keep_heads", ascending=False))
            if not len(rows):
                continue
            pruned_pct = (1 - rows.keep_heads / TOTAL_HEADS) * 100
            ax.plot(pruned_pct, rows.mse_mean, marker="o", markersize=3.5,
                    linewidth=2, color=COLORS[method], label=LABELS[method])
            ax.fill_between(pruned_pct, rows.mse_mean - rows.mse_std,
                            rows.mse_mean + rows.mse_std,
                            color=COLORS[method], alpha=0.15, linewidth=0)
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("Heads pruned (%)")
    axes[0][0].set_ylabel("Test MSE (mean ± std, 5 seeds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.savefig(FIG_DIR / "fig2_sparsity_sweep.pdf")
    fig.savefig(FIG_DIR / "fig2_sparsity_sweep.png")
    plt.close(fig)


def fig_sweep_all_horizons():
    agg = pd.read_csv(ANALYSIS_DIR / "sweep_summary.csv")
    test = agg[agg.split == "test"]
    cells = sorted(test.groupby(["dataset", "pred_len"]).groups)
    ncols = 4
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 2.6 * nrows),
                             squeeze=False)
    for i, (dataset, pred_len) in enumerate(cells):
        ax = axes[i // ncols][i % ncols]
        sub = test[(test.dataset == dataset) & (test.pred_len == pred_len)]
        base = sub[sub.method_group == "baseline"].mse_mean.iloc[0]
        ax.axhline(base, color=COLORS["baseline"], linewidth=1.2,
                   linestyle="--")
        for method in ["static", "dynamic_joint", "static_greedy",
                       "random", "magnitude"]:
            rows = (sub[sub.method_group == method]
                    .sort_values("keep_heads", ascending=False))
            if not len(rows):
                continue
            pruned_pct = (1 - rows.keep_heads / TOTAL_HEADS) * 100
            ax.plot(pruned_pct, rows.mse_mean, marker="o", markersize=2.5,
                    linewidth=1.6, color=COLORS[method])
        ax.set_title(f"{dataset} / {pred_len}", fontsize=9)
    for j in range(len(cells), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.supxlabel("Heads pruned (%)")
    fig.supylabel("Test MSE")
    fig.savefig(FIG_DIR / "fig3_sweep_all_cells.pdf")
    fig.savefig(FIG_DIR / "fig3_sweep_all_cells.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_importance_heatmap()
    fig_sparsity_sweep()
    fig_sweep_all_horizons()
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
