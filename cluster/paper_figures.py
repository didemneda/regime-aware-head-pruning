"""All publication figures, from both result trees (PatchTST + iTransformer).

Reads only the analysis CSVs / result files; no hard-coded numbers.
Writes PDF + PNG pairs to ~/recahs/paper_figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / "recahs"
PT = ROOT / "runs" / "analysis"            # PatchTST tree
ITR = ROOT / "runs_iTransformer" / "analysis"
OUT = ROOT / "paper_figures"
TOTAL_HEADS = 24

COLORS = {
    "baseline": "#5f5e59", "static": "#2a78d6", "dynamic_joint": "#eb6834",
    "static_greedy": "#4a3aa7", "random": "#1baf7a", "magnitude": "#eda100",
    "ensemble": "#eb6834", "router": "#8a93a5", "diag": "#c9cdd6",
}
LABELS = {
    "baseline": "Unpruned", "static": "Static (global importance)",
    "dynamic_joint": "Dynamic joint (regime-routed)",
    "static_greedy": "Static greedy (pooled control)",
    "random": "Random (mean of 3)", "magnitude": "Magnitude",
}
METHOD_ORDER = ["static", "dynamic_joint", "static_greedy", "random",
                "magnitude"]

plt.rcParams.update({
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.titlesize": 9.5, "legend.fontsize": 8,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def sweep_panels(analysis_dir, name, title):
    agg = pd.read_csv(analysis_dir / "sweep_summary.csv")
    test = agg[(agg.split == "test") & (agg.pred_len == 96)]
    datasets = sorted(test.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(2.9 * len(datasets), 2.7),
                             squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = test[test.dataset == dataset]
        base = sub[sub.method_group == "baseline"].mse_mean.iloc[0]
        ax.axhline(base, color=COLORS["baseline"], linewidth=1.4,
                   linestyle="--", label=LABELS["baseline"])
        for method in METHOD_ORDER:
            rows = (sub[sub.method_group == method]
                    .sort_values("keep_heads", ascending=False))
            if not len(rows):
                continue
            x = (1 - rows.keep_heads / TOTAL_HEADS) * 100
            ax.plot(x, rows.mse_mean, marker="o", markersize=3,
                    linewidth=1.8, color=COLORS[method],
                    label=LABELS[method])
            ax.fill_between(x, rows.mse_mean - rows.mse_std,
                            rows.mse_mean + rows.mse_std,
                            color=COLORS[method], alpha=0.13, linewidth=0)
        ax.set_title(dataset)
        ax.set_xlabel("Heads pruned (%)")
    axes[0][0].set_ylabel("Test MSE (mean ± std, 5 seeds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.14))
    fig.suptitle(title, y=1.22, fontsize=10)
    save(fig, name)


def fig_dose_response():
    """Wins of dynamic vs static and vs static-greedy, per sparsity."""
    pt = pd.read_csv(PT / "paired_tests.csv")
    keeps = sorted(pt.keep_heads.unique(), reverse=True)
    x = np.arange(len(keeps))
    wins_static, wins_sg = [], []
    for keep in keeps:
        g = pt[pt.keep_heads == keep]
        wins_static.append((g.mean_diff_dyn_minus_static < 0).sum())
        wins_sg.append((g.mean_diff_dyn_minus_static_greedy < 0).sum())
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    width = 0.38
    ax.bar(x - width / 2, wins_static, width, color=COLORS["static"],
           label="vs static (global importance)")
    ax.bar(x + width / 2, wins_sg, width, color=COLORS["static_greedy"],
           label="vs static greedy (control)")
    ax.axhline(8, color=COLORS["baseline"], linewidth=1, linestyle=":")
    ax.text(len(keeps) - 0.4, 8.25, "8/16 = tie", fontsize=7.5,
            color=COLORS["baseline"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{(1-k/TOTAL_HEADS)*100:.0f}%" for k in keeps])
    ax.set_xlabel("Heads pruned")
    ax.set_ylabel("Cells won by dynamic (of 16)")
    ax.set_ylim(0, 16.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Attribution: the criterion, not the routing (PatchTST)")
    save(fig, "fig_dose_response")


def fig_router_recovery():
    """Oracle-gain recovery of every router / ensemble variant."""
    entries = []
    lab = pd.read_csv(PT / "router_lab.csv").groupby(
        ["dataset", "pred_len"]).mean(numeric_only=True)
    for key, label in [("argmin_clf", "Argmin classifier"),
                       ("loss_reg", "Loss regression"),
                       ("delta_reg", "Regret regression"),
                       ("temporal", "Delayed feedback"),
                       ("DIAG_insample", "In-sample fit (diagnostic)")]:
        entries.append((label, lab[f"{key}_recovery_pct"].mean(),
                        "diag" if key.startswith("DIAG") else "router"))
    extras = pd.read_csv(PT / "router_extras.csv").groupby(
        ["dataset", "pred_len"]).mean(numeric_only=True)
    entries.append(("Bandit (block-96)",
                    extras["block_96_recovery_pct"].mean(), "router"))
    entries.append(("Regret-gated (q95)",
                    extras["gate_q95_recovery_pct"].mean(), "router"))
    reps = pd.read_csv(PT / "router_reps.csv").groupby(
        "dataset").mean(numeric_only=True)
    entries.append(("Encoder representations",
                    reps["rep_delta_recovery_pct"].mean(), "router"))
    soft = pd.read_csv(PT / "soft_ensemble.csv").groupby(
        "dataset").mean(numeric_only=True)
    entries.append(("Hard STL routing",
                    soft["stl_a100_recovery_pct"].mean(), "router"))
    entries.append(("Soft STL ensemble (α=0.4)",
                    soft["stl_a40_recovery_pct"].mean(), "ensemble"))
    ens = pd.read_csv(PT / "ensemble_eval.csv").groupby(
        "dataset").mean(numeric_only=True)
    for key, label in [("mean4", "Uniform ensemble"),
                       ("top2", "Top-2 ensemble"),
                       ("invmse", "Val-weighted ensemble")]:
        entries.append((label, ens[f"{key}_recovery_pct"].mean(),
                        "ensemble"))
    entries.sort(key=lambda e: e[1])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y = np.arange(len(entries))
    for i, (label, value, kind) in enumerate(entries):
        color = {"router": COLORS["router"], "ensemble": COLORS["ensemble"],
                 "diag": COLORS["diag"]}[kind]
        hatch = "//" if kind == "diag" else None
        ax.barh(i, value, color=color, hatch=hatch,
                edgecolor="white", linewidth=0.5)
        ax.text(value + (1.5 if value >= 0 else -1.5), i,
                f"{value:+.0f}%", va="center",
                ha="left" if value >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color="#5f5e59", linewidth=1)
    ax.axvline(100, color="#5f5e59", linewidth=1, linestyle=":")
    ax.text(100, len(entries) - 0.2, "oracle", fontsize=7.5, ha="center")
    low = min(value for _, value, _ in entries)
    ax.set_xlim(low - 16, 112)
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries], fontsize=8)
    ax.set_xlabel("Oracle-gain recovery vs pooled static-greedy (%)")
    ax.set_title("Selection fails; combination works")
    ax.grid(axis="y", alpha=0)
    save(fig, "fig_router_recovery")


def fig_learning_curve():
    curve = pd.read_csv(PT / "router_curve.csv")
    agg = curve.groupby("setting").mean(numeric_only=True)
    # train_plus_val excluded: its val regret is in-sample by construction
    order = ["train_10", "train_25", "train_50", "train_100"]
    agg = agg.reindex(order)
    x = agg["n_router_train"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(x, agg["train_regret"], marker="o", markersize=3.5,
            linewidth=1.8, color=COLORS["static"], label="Router train set")
    ax.plot(x, agg["val_regret"], marker="s", markersize=3.5,
            linewidth=1.8, color=COLORS["dynamic_joint"],
            label="Held-out validation")
    ax.plot(x, agg["test_regret"], marker="^", markersize=3.5,
            linewidth=1.8, color=COLORS["static_greedy"], label="Test")
    ax.axhline(agg["sg_test_regret"].iloc[0], color=COLORS["baseline"],
               linewidth=1.3, linestyle="--",
               label="Static-greedy (no routing)")
    ax.set_xscale("log")
    ax.set_xlabel("Router training windows")
    ax.set_ylabel("Mean regret vs oracle")
    ax.set_title("Routers memorize but never generalize")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7.5, loc="center left",
              bbox_to_anchor=(0.02, 0.42))
    save(fig, "fig_learning_curve")


def fig_alpha_sweep():
    soft = pd.read_csv(PT / "soft_ensemble.csv")
    ens = pd.read_csv(PT / "ensemble_eval.csv")[
        ["dataset", "seed", "invmse_recovery_pct"]]
    merged = soft.merge(ens, on=["dataset", "seed"])
    alphas = [0.4, 0.6, 0.8, 1.0]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for dataset, g in merged.groupby("dataset"):
        values = [g[f"stl_a{int(a*100)}_recovery_pct"].mean()
                  for a in alphas]
        ax.plot(alphas, values, linewidth=1, color="#9aa3b2", alpha=0.7)
        ax.annotate(dataset, (alphas[-1], values[-1]), fontsize=6.5,
                    color="#767e8c", xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center")
    ax.set_xlim(0.36, 1.12)
    mean_values = [merged[f"stl_a{int(a*100)}_recovery_pct"].mean()
                   for a in alphas]
    ax.plot(alphas, mean_values, marker="o", markersize=4, linewidth=2.2,
            color=COLORS["dynamic_joint"], label="Mean over datasets")
    ax.axhline(merged["invmse_recovery_pct"].mean(),
               color=COLORS["static_greedy"], linestyle="--", linewidth=1.4,
               label="Regime-blind weighted ensemble")
    ax.axhline(0, color=COLORS["baseline"], linewidth=1)
    ax.set_xticks(alphas)
    ax.set_xlabel("α (weight on the detected regime's mask)")
    ax.set_ylabel("Oracle-gain recovery (%)")
    ax.set_title("Trusting the regime label less helps more")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    save(fig, "fig_alpha_sweep")


def fig_arch_comparison():
    """Degradation at 25% / 75% pruning per method, PatchTST vs iTransformer."""
    frames = {}
    for name, path in [("PatchTST", PT), ("iTransformer", ITR)]:
        agg = pd.read_csv(path / "sweep_summary.csv")
        test = agg[(agg.split == "test") & (agg.pred_len == 96)]
        frames[name] = test
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, keep in zip(axes, [18, 6]):
        datasets = sorted(frames["PatchTST"].dataset.unique())
        x = np.arange(len(datasets))
        width = 0.2
        offsets = {"static": -1.5, "static_greedy": -0.5,
                   "dynamic_joint": 0.5, "random": 1.5}
        for arch, hatch in [("PatchTST", None), ("iTransformer", "//")]:
            sub = frames[arch]
            for method, off in offsets.items():
                vals = []
                for dataset in datasets:
                    g = sub[(sub.dataset == dataset)
                            & (sub.keep_heads == keep)
                            & (sub.method_group == method)]
                    base = sub[(sub.dataset == dataset)
                               & (sub.method_group == "baseline")
                               ].mse_mean.iloc[0]
                    vals.append((g.mse_mean.iloc[0] / base - 1) * 100
                                if len(g) else np.nan)
                ax.bar(x + off * width / 2
                       + (0 if arch == "PatchTST" else width / 4.5),
                       vals, width / 2.2, color=COLORS[method], hatch=hatch,
                       edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="#5f5e59", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(f"{(1-keep/TOTAL_HEADS)*100:.0f}% heads pruned")
    axes[0].set_ylabel("Test MSE vs baseline (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m])
               for m in offsets]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                 edgecolor="#5f5e59", hatch="//"))
    fig.legend(handles,
               [LABELS[m] for m in offsets] + ["iTransformer (hatched)"],
               loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.12), fontsize=7.5)
    save(fig, "fig_arch_comparison")


def fig_importance_heatmap():
    datasets = sorted({p.parts[-5] for p in (ROOT / "runs").glob(
        "*/pred96/seed_*/pruning/head_importance.csv")})
    fig, axes = plt.subplots(len(datasets), 3,
                             figsize=(8.2, 1.9 * len(datasets)),
                             squeeze=False)
    for row, dataset in enumerate(datasets):
        frames = [pd.read_csv(p) for p in (ROOT / "runs").glob(
            f"{dataset}/pred96/seed_*/pruning/head_importance.csv")]
        mean = sum(f[["trend_importance", "seasonal_importance",
                      "residual_importance"]].to_numpy()
                   for f in frames) / len(frames)
        vmax = np.abs(mean).max()
        for col, regime in enumerate(["trend", "seasonal", "residual"]):
            ax = axes[row][col]
            im = ax.imshow(mean[:, col].reshape(3, 8), cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(8))
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"L{i}" for i in range(3)])
            ax.grid(False)
            if row == 0:
                ax.set_title(f"{regime}-dominant")
            if col == 0:
                ax.set_ylabel(dataset)
        fig.colorbar(im, ax=axes[row], shrink=0.85, pad=0.01)
    save(fig, "fig_importance_heatmap")


def fig_all_cells():
    agg = pd.read_csv(PT / "sweep_summary.csv")
    test = agg[agg.split == "test"]
    cells = sorted(test.groupby(["dataset", "pred_len"]).groups)
    fig, axes = plt.subplots(4, 4, figsize=(11, 9.4), squeeze=False)
    for i, (dataset, pred_len) in enumerate(cells):
        ax = axes[i // 4][i % 4]
        sub = test[(test.dataset == dataset) & (test.pred_len == pred_len)]
        base = sub[sub.method_group == "baseline"].mse_mean.iloc[0]
        ax.axhline(base, color=COLORS["baseline"], linewidth=1,
                   linestyle="--")
        for method in METHOD_ORDER:
            rows = (sub[sub.method_group == method]
                    .sort_values("keep_heads", ascending=False))
            if not len(rows):
                continue
            x = (1 - rows.keep_heads / TOTAL_HEADS) * 100
            ax.plot(x, rows.mse_mean, marker="o", markersize=2.2,
                    linewidth=1.4, color=COLORS[method])
        ax.set_title(f"{dataset} / {pred_len}", fontsize=8.5)
    fig.supxlabel("Heads pruned (%)")
    fig.supylabel("Test MSE")
    fig.subplots_adjust(hspace=0.55, wspace=0.38)
    save(fig, "fig_all_cells")


def fig_bootstrap_forest():
    """Block-bootstrap CIs at 25% pruning, per cell, three comparisons."""
    boot = pd.read_csv(PT / "bootstrap_keep18.csv")
    comparisons = [("dynamic_joint", "static", "dynamic − static"),
                   ("dynamic_joint", "baseline", "dynamic − unpruned"),
                   ("static", "baseline", "static − unpruned")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharey=True)
    cells = sorted(set(zip(boot.dataset, boot.pred_len)))
    ylabels = [f"{d}@{p}" for d, p in cells]
    for ax, (left, right, title) in zip(axes, comparisons):
        sub = boot[(boot.left == left) & (boot.right == right)]
        for i, (d, p) in enumerate(cells):
            row = sub[(sub.dataset == d) & (sub.pred_len == p)].iloc[0]
            sig = row.ci_low > 0 or row.ci_high < 0
            color = COLORS["dynamic_joint"] if sig else COLORS["router"]
            ax.plot([row.ci_low, row.ci_high], [i, i], color=color,
                    linewidth=1.6)
            ax.plot(row.observed_diff, i, "o", color=color, markersize=3.5)
        ax.axvline(0, color="#5f5e59", linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Test MSE difference")
    axes[0].set_yticks(range(len(cells)))
    axes[0].set_yticklabels(ylabels, fontsize=7)
    fig.suptitle("Hierarchical block bootstrap, 95% CI, 25% pruning "
                 "(colored = CI excludes zero)", y=1.02, fontsize=10)
    save(fig, "fig_bootstrap_forest")


def fig_scratch():
    """Zero-shot masks vs masked-from-scratch training vs baseline."""
    per_seed = pd.read_csv(PT / "sweep_per_seed.csv")
    scratch = pd.read_csv(PT / "scratch_summary.csv")
    cells = [("ETTh1", 96), ("ETTm1", 96)]
    method_specs = [
        ("baseline", 24, "sweep", "Unpruned", COLORS["baseline"]),
        ("static", 18, "sweep", "Static zero-shot", COLORS["static"]),
        ("dynamic_joint", 18, "sweep", "Dynamic zero-shot",
         COLORS["dynamic_joint"]),
        ("scratch_static", 18, "scratch", "Static from scratch",
         "#e87ba4"),
        ("scratch_regime", 18, "scratch", "Regime-cond. from scratch",
         "#008300"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, (dataset, pred) in zip(axes, cells):
        base = per_seed[(per_seed.dataset == dataset)
                        & (per_seed.pred_len == pred)
                        & (per_seed.split == "test")
                        & (per_seed.method_group == "baseline")].mse.mean()
        ax.axhline(base, color=COLORS["baseline"], linewidth=1,
                   linestyle="--")
        for i, (method, keep, source, label, color) in enumerate(
                method_specs):
            if source == "sweep":
                vals = per_seed[(per_seed.dataset == dataset)
                                & (per_seed.pred_len == pred)
                                & (per_seed.split == "test")
                                & (per_seed.method_group == method)
                                & (per_seed.keep_heads == keep)].mse
            else:
                vals = scratch[(scratch.dataset == dataset)
                               & (scratch.pred_len == pred)
                               & (scratch.method == method)].test_mse
            ax.errorbar(i, vals.mean(), yerr=vals.std(), color=color,
                        marker="o", markersize=6, capsize=3, linewidth=1.4)
        ax.set_xticks(range(len(method_specs)))
        ax.set_xticklabels([m[3] for m in method_specs], rotation=28,
                           ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(method_specs) - 0.4)
        ax.set_title(f"{dataset} @ {pred}")
    axes[0].set_ylabel("Test MSE (mean ± std, 5 seeds)")
    fig.suptitle("Training with the mask makes 25% pruning free", y=1.04,
                 fontsize=10)
    save(fig, "fig_scratch")


def fig_diversity_and_headroom():
    """Regime vs random specialist pools: realized ensembles and oracles."""
    div = pd.concat([
        pd.read_csv(PT / "diversity_ablation_ETTh1_ETTh2.csv"),
        pd.read_csv(PT / "diversity_ablation_ETTm1_weather.csv")],
        ignore_index=True).groupby("dataset").mean(numeric_only=True)
    ens = pd.read_csv(PT / "ensemble_eval.csv").groupby(
        "dataset").mean(numeric_only=True)
    datasets = sorted(div.index)
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    width = 0.19
    bars = [
        ("Static greedy", ens.static_greedy, COLORS["static_greedy"], None),
        ("Regime-pool ensemble", ens.invmse, COLORS["dynamic_joint"], None),
        ("Random-pool ensemble", div.rnd_invmse, COLORS["dynamic_joint"],
         "//"),
    ]
    for i, (label, series, color, hatch) in enumerate(bars):
        ax.bar(x + (i - 1) * width, [series[d] for d in datasets], width,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.5,
               label=label)
    for d_i, dataset in enumerate(datasets):
        ax.plot(d_i, ens.baseline[dataset], marker="D", markersize=5,
                color=COLORS["baseline"], linestyle="none",
                label="Unpruned" if d_i == 0 else None)
        ax.plot(d_i - width, ens.oracle4[dataset], marker="v",
                markersize=5, color="#1a2029", linestyle="none",
                label="Oracle (regime pool)" if d_i == 0 else None)
        ax.plot(d_i + width, div.oracle_random_pool[dataset], marker="v",
                markersize=5, markerfacecolor="white", color="#1a2029",
                linestyle="none",
                label="Oracle (random pool)" if d_i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test MSE (mean over 5 seeds)")
    ax.set_title("Ensemble gains are diversity gains; regime pools have "
                 "the deepest oracles")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_ylim(min(ens.oracle4.min(), div.oracle_random_pool.min()) * 0.9,
                None)
    save(fig, "fig_diversity_headroom")


def fig_fallback_appendix():
    fb = pd.read_csv(PT / "confidence_fallback.csv")
    fb = fb[(fb.pred_len == 96) & (fb.fallback == "static")]
    datasets = sorted(fb.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(2.5 * len(datasets), 2.4))
    for ax, dataset in zip(axes, datasets):
        g = fb[fb.dataset == dataset].groupby(
            "threshold").mean(numeric_only=True)
        ax.plot(g.index, g.test_mse, marker="o", markersize=3,
                color=COLORS["dynamic_joint"], linewidth=1.6,
                label="STL routing + fallback")
        ax.axhline(g.pure_dynamic_mse.iloc[0], color=COLORS["static"],
                   linewidth=1.2, linestyle=":", label="Pure STL routing")
        ax.axhline(g.static_mse.iloc[0], color=COLORS["static_greedy"],
                   linewidth=1.2, linestyle="--", label="Static (25%)")
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Test MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.14), fontsize=8)
    save(fig, "fig_fallback_appendix")


def fig_matched_advantage():
    """Matched-mask advantage per cell (recomputed from stored losses)."""
    import csv as csv_mod
    regimes = ["trend", "seasonal", "residual"]
    rows = []
    for npz_path in sorted((ROOT / "runs").glob(
            "*/pred*/seed_*/pruning/window_losses.npz")):
        parts = npz_path.parts
        dataset, pred = parts[-5], int(parts[-4][4:])
        z = np.load(npz_path)
        try:
            per = {r: z[f"test/regimemask_{r}/keep18/mse"]
                   for r in regimes}
        except KeyError:
            continue
        labels_csv = (ROOT / "runs" / dataset / f"pred{pred}"
                      / "regime_labels" / "test_regimes.csv")
        labels = np.array([r["regime"] for r in csv_mod.DictReader(
            open(labels_csv))])
        stack = np.stack([per[r] for r in regimes])
        idx = {r: i for i, r in enumerate(regimes)}
        n = len(labels)
        matched = stack[[idx[r] for r in labels], np.arange(n)]
        mismatched = (stack.sum(0) - matched) / 2.0
        rows.append({"dataset": dataset, "pred_len": pred,
                     "adv": ((mismatched - matched).mean()
                             / matched.mean()) * 100})
    frame = pd.DataFrame(rows).groupby(
        ["dataset", "pred_len"]).mean(numeric_only=True).reset_index()
    frame = frame.sort_values(["dataset", "pred_len"])
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    colors = {"ETTh1": COLORS["static"], "ETTh2": COLORS["static_greedy"],
              "ETTm1": COLORS["dynamic_joint"],
              "weather": COLORS["random"]}
    x = np.arange(len(frame))
    ax.bar(x, frame.adv, color=[colors[d] for d in frame.dataset],
           width=0.7)
    ax.axhline(0, color="#5f5e59", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(frame.pred_len, fontsize=7)
    ax.set_xlabel("Prediction horizon")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[d])
               for d in sorted(colors)]
    ax.legend(handles, sorted(colors), frameon=False, fontsize=7, ncol=4,
              loc="upper left")
    ax.set_ylabel("Matched-mask advantage (%)")
    ax.set_title("Each regime's mask is (mostly) best on its own regime's "
                 "windows")
    save(fig, "fig_matched_advantage")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sweep_panels(PT, "fig_sweep_patchtst",
                 "PatchTST, horizon 96")
    sweep_panels(ITR, "fig_sweep_itransformer",
                 "iTransformer, horizon 96")
    fig_dose_response()
    fig_router_recovery()
    fig_learning_curve()
    fig_alpha_sweep()
    fig_arch_comparison()
    fig_importance_heatmap()
    fig_all_cells()
    fig_bootstrap_forest()
    fig_scratch()
    fig_diversity_and_headroom()
    fig_fallback_appendix()
    fig_matched_advantage()
    print("all figures done ->", OUT)


if __name__ == "__main__":
    main()
