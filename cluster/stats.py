"""Consolidated statistics over all completed cells.

Outputs (under runs/analysis/):
  hypothesis_tests.csv    -- per (cell, seed): circular-rotation permutation
                             test that head importance depends on regime,
                             plus rank correlations within/across regimes.
  sweep_summary.csv       -- mean+-std per (cell, method, keep) + per-seed.
  paired_tests.csv        -- per cell+keep: paired (per-seed) dynamic vs
                             static differences; sign test across cells.
  bootstrap_keep18.csv    -- hierarchical circular block bootstrap per cell
                             at keep=18 (seeds x temporal blocks).
  oracle_detector.csv     -- STL / predicted-label / oracle routing results
                             at keep=18 from stored per-window losses.
  scratch_summary.csv     -- masked-from-scratch training results.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RUNS_DIR, SEEDS, SWEEP_KEEPS, TOTAL_HEADS, load_labels

ANALYSIS_DIR = RUNS_DIR / "analysis"
REGIMES = ["trend", "seasonal", "residual"]
N_PERMUTATIONS = 2000
BLOCK_LEN = 96
N_BOOT = 10000


def discover_cells():
    cells = []
    for meta in sorted(RUNS_DIR.glob("*/pred*/seed_*/pruning/DONE")):
        seed_dir = meta.parent.parent
        pred_dir = seed_dir.parent
        cells.append((pred_dir.parent.name, int(pred_dir.name[4:]),
                      int(seed_dir.name.split("_")[1])))
    return cells


def kendall(x, y):
    from scipy.stats import kendalltau
    tau, _ = kendalltau(x, y)
    return float(tau)


# ---------------------------------------------------------------- hypothesis
def hypothesis_tests(cells):
    rows = []
    by_cell = {}
    for dataset, pred_len, seed in cells:
        by_cell.setdefault((dataset, pred_len), []).append(seed)
    for (dataset, pred_len), seeds in by_cell.items():
        val_labels = load_labels(dataset, pred_len, "val")
        regimes = val_labels["regime"].to_numpy()
        regime_sets = {r: regimes == r for r in REGIMES
                       if (regimes == r).sum() >= 20}
        per_seed_rankings = {r: {} for r in regime_sets}
        for seed in seeds:
            prune_dir = (RUNS_DIR / dataset / f"pred{pred_len}"
                         / f"seed_{seed}" / "pruning")
            data = np.load(prune_dir / "loo_val_window_mse.npz")
            delta = data["loo"] - data["baseline"][None, :]  # (24, n_val)

            def regime_stat(labels_arr):
                """Sum over heads of variance of per-regime mean importance."""
                means = np.stack([
                    delta[:, labels_arr == r].mean(axis=1)
                    for r in regime_sets])
                return float(means.var(axis=0).sum())

            observed = regime_stat(regimes)
            rng = np.random.RandomState(seed)
            n_val = len(regimes)
            null = np.empty(N_PERMUTATIONS)
            for i in range(N_PERMUTATIONS):
                shift = rng.randint(1, n_val)
                null[i] = regime_stat(np.roll(regimes, shift))
            p_value = float((null >= observed).mean())

            imp = pd.read_csv(prune_dir / "head_importance.csv")
            taus = {}
            for a, b in itertools.combinations(regime_sets, 2):
                taus[f"tau_{a}_{b}"] = kendall(
                    imp[f"{a}_importance"], imp[f"{b}_importance"])
            for r in regime_sets:
                per_seed_rankings[r][seed] = imp[f"{r}_importance"].to_numpy()
            matrix = imp[[f"{r}_importance" for r in regime_sets]].to_numpy()
            flips = int(((matrix.max(1) > 0) & (matrix.min(1) < 0)).sum())
            rows.append({
                "dataset": dataset, "pred_len": pred_len, "seed": seed,
                "observed_stat": observed,
                "null_mean": float(null.mean()),
                "p_value": p_value, "opposite_sign_heads": flips,
                **taus,
            })
        # cross-seed stability of within-regime rankings
        for r, rankings in per_seed_rankings.items():
            pairs = list(itertools.combinations(sorted(rankings), 2))
            if pairs:
                taus = [kendall(rankings[a], rankings[b]) for a, b in pairs]
                rows.append({
                    "dataset": dataset, "pred_len": pred_len, "seed": -1,
                    "stability_regime": r,
                    "cross_seed_tau_mean": float(np.mean(taus)),
                    "cross_seed_tau_min": float(np.min(taus)),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- sweep
def load_all_summaries(cells):
    frames = []
    for dataset, pred_len, seed in cells:
        path = (RUNS_DIR / dataset / f"pred{pred_len}" / f"seed_{seed}"
                / "pruning" / "summary.csv")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def sweep_summary(summaries):
    frame = summaries.copy()
    # average the random draws into one method per seed first
    frame["method_group"] = frame["method"].str.replace(
        r"random\d+", "random", regex=True)
    per_seed = (frame.groupby(
        ["dataset", "pred_len", "seed", "split", "method_group",
         "keep_heads"], as_index=False)[["mse", "mae"]].mean())
    agg = (per_seed.groupby(
        ["dataset", "pred_len", "split", "method_group", "keep_heads"])
        .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"),
             mae_mean=("mae", "mean"), mae_std=("mae", "std"),
             n_seeds=("seed", "nunique")).reset_index())
    base = agg[agg.method_group == "baseline"][
        ["dataset", "pred_len", "split", "mse_mean"]].rename(
        columns={"mse_mean": "baseline_mse"})
    agg = agg.merge(base, on=["dataset", "pred_len", "split"])
    agg["mse_vs_baseline_pct"] = (
        (agg["mse_mean"] - agg["baseline_mse"]) / agg["baseline_mse"] * 100)
    return per_seed, agg


def paired_tests(per_seed):
    """Per-seed paired dynamic-static differences and cross-cell sign test."""
    rows = []
    test = per_seed[per_seed.split == "test"]
    for (dataset, pred_len, keep), group in test.groupby(
            ["dataset", "pred_len", "keep_heads"]):
        if keep == TOTAL_HEADS:
            continue
        pivot = group.pivot_table(
            index="seed", columns="method_group", values="mse")
        if "dynamic_joint" not in pivot or "static" not in pivot:
            continue
        record = {
            "dataset": dataset, "pred_len": pred_len, "keep_heads": keep,
        }
        diff = (pivot["dynamic_joint"] - pivot["static"]).dropna()
        record.update({
            "n_seeds": len(diff),
            "mean_diff_dyn_minus_static": float(diff.mean()),
            "seeds_dynamic_better": int((diff < 0).sum()),
            "wilcoxon_p": _wilcoxon_p(diff.to_numpy()),
        })
        if "static_greedy" in pivot:
            diff_sg = (pivot["dynamic_joint"]
                       - pivot["static_greedy"]).dropna()
            record.update({
                "mean_diff_dyn_minus_static_greedy": float(diff_sg.mean()),
                "seeds_dynamic_better_vs_greedy": int((diff_sg < 0).sum()),
                "wilcoxon_p_vs_greedy": _wilcoxon_p(diff_sg.to_numpy()),
            })
        rows.append(record)
    cellwise = pd.DataFrame(rows)
    # sign test across cells at each keep
    sign_rows = []
    from scipy.stats import binomtest
    for keep, group in cellwise.groupby("keep_heads"):
        wins = int((group["mean_diff_dyn_minus_static"] < 0).sum())
        n = len(group)
        sign_rows.append({
            "keep_heads": keep, "cells": n, "dynamic_wins": wins,
            "sign_test_p": float(binomtest(wins, n).pvalue) if n else np.nan,
        })
    return cellwise, pd.DataFrame(sign_rows)


def _wilcoxon_p(diff):
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(diff).pvalue)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------- bootstrap
def hierarchical_bootstrap(cells, keep=18):
    """Seeds x circular temporal blocks, per cell, dynamic vs static and
    each vs baseline, on test MSE at the given keep count."""
    rows = []
    by_cell = {}
    for dataset, pred_len, seed in cells:
        by_cell.setdefault((dataset, pred_len), []).append(seed)
    comparisons = [("dynamic_joint", "static"),
                   ("dynamic_joint", "baseline"), ("static", "baseline")]
    for (dataset, pred_len), seeds in sorted(by_cell.items()):
        losses = {}
        for seed in seeds:
            data = np.load(RUNS_DIR / dataset / f"pred{pred_len}"
                           / f"seed_{seed}" / "pruning"
                           / "window_losses.npz")
            losses[seed] = {
                "static": data[f"test/static/keep{keep}/mse"],
                "dynamic_joint": data[f"test/dynamic_joint/keep{keep}/mse"],
                "baseline": data[f"test/baseline/keep{TOTAL_HEADS}/mse"],
            }
        n_windows = len(next(iter(losses.values()))["static"])
        n_blocks = int(np.ceil(n_windows / BLOCK_LEN))
        rng = np.random.RandomState(20260816)
        for left, right in comparisons:
            per_seed_diff = {
                s: losses[s][left] - losses[s][right] for s in seeds}
            observed = float(np.mean(
                [d.mean() for d in per_seed_diff.values()]))
            boots = np.empty(N_BOOT)
            seed_list = list(seeds)
            for b in range(N_BOOT):
                chosen = rng.choice(len(seed_list), len(seed_list))
                starts = rng.randint(0, n_windows, n_blocks)
                idx = (starts[:, None]
                       + np.arange(BLOCK_LEN)[None, :]).ravel() % n_windows
                idx = idx[:n_windows]
                boots[b] = np.mean(
                    [per_seed_diff[seed_list[c]][idx].mean()
                     for c in chosen])
            low, high = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "dataset": dataset, "pred_len": pred_len, "keep_heads": keep,
                "left": left, "right": right,
                "observed_diff": observed,
                "ci_low": float(low), "ci_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "conclusion": (
                    "left_better" if high < 0 else
                    "right_better" if low > 0 else "inconclusive"),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------- oracle/detector
def oracle_detector(cells, keep=18):
    rows = []
    for dataset, pred_len, seed in cells:
        cell = RUNS_DIR / dataset / f"pred{pred_len}"
        data = np.load(cell / f"seed_{seed}" / "pruning"
                       / "window_losses.npz")
        test_labels = load_labels(dataset, pred_len, "test")
        regimes = test_labels["regime"].to_numpy()
        per_mask = {r: data[f"test/regimemask_{r}/keep{keep}/mse"]
                    for r in REGIMES}
        stacked = np.stack([per_mask[r] for r in REGIMES])
        idx_of = {r: i for i, r in enumerate(REGIMES)}
        stl_route = stacked[[idx_of[r] for r in regimes],
                            np.arange(len(regimes))]
        oracle = stacked.min(axis=0)
        record = {
            "dataset": dataset, "pred_len": pred_len, "seed": seed,
            "baseline": float(
                data[f"test/baseline/keep{TOTAL_HEADS}/mse"].mean()),
            "static": float(data[f"test/static/keep{keep}/mse"].mean()),
            "dynamic_stl": float(stl_route.mean()),
            "dynamic_oracle": float(oracle.mean()),
            "oracle_route_match_stl_pct": float(
                (stacked.argmin(axis=0)
                 == np.array([idx_of[r] for r in regimes])).mean() * 100),
        }
        for name in ("logreg", "hgb"):
            pred_path = cell / "detector" / f"test_pred_{name}.npy"
            if pred_path.exists():
                pred = np.load(pred_path, allow_pickle=True)
                route = stacked[[idx_of[r] for r in pred],
                                np.arange(len(pred))]
                record[f"dynamic_{name}"] = float(route.mean())
        rows.append(record)
    return pd.DataFrame(rows)


# ------------------------------------------------- confidence-aware fallback
FALLBACK_THRESHOLDS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2]


def confidence_fallback(cells, keep=18):
    """Route ambiguous windows (confidence_margin < t) to a fallback.

    fallback=static : ambiguous windows use the static keep-18 mask
                      (same compute budget as dynamic).
    fallback=full   : ambiguous windows use the unpruned model
                      (adaptive-compute variant).
    Evaluated offline from stored per-window losses; t=0 recovers pure
    dynamic routing.
    """
    rows = []
    for dataset, pred_len, seed in cells:
        prune_dir = (RUNS_DIR / dataset / f"pred{pred_len}"
                     / f"seed_{seed}" / "pruning")
        data = np.load(prune_dir / "window_losses.npz")
        test_labels = load_labels(dataset, pred_len, "test")
        regimes = test_labels["regime"].to_numpy()
        margins = test_labels["confidence_margin"].to_numpy()
        per_mask = {r: data[f"test/regimemask_{r}/keep{keep}/mse"]
                    for r in REGIMES}
        stl_route = np.stack(
            [per_mask[r] for r in REGIMES])[
            [REGIMES.index(r) for r in regimes],
            np.arange(len(regimes))]
        static = data[f"test/static/keep{keep}/mse"]
        baseline = data[f"test/baseline/keep{TOTAL_HEADS}/mse"]
        for t in FALLBACK_THRESHOLDS:
            ambiguous = margins < t
            for name, fallback in (("static", static), ("full", baseline)):
                routed = np.where(ambiguous, fallback, stl_route)
                rows.append({
                    "dataset": dataset, "pred_len": pred_len, "seed": seed,
                    "threshold": t, "fallback": name,
                    "ambiguous_pct": float(ambiguous.mean() * 100),
                    "test_mse": float(routed.mean()),
                    "pure_dynamic_mse": float(stl_route.mean()),
                    "static_mse": float(static.mean()),
                    "baseline_mse": float(baseline.mean()),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- scratch
def scratch_summary():
    frames = []
    for path in sorted(RUNS_DIR.glob("*/pred*/seed_*/scratch_*/summary.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cells = discover_cells()
    print(f"{len(cells)} completed (dataset, pred, seed) cells")
    if not cells:
        return

    hyp = hypothesis_tests(cells)
    hyp.to_csv(ANALYSIS_DIR / "hypothesis_tests.csv", index=False)
    print("hypothesis tests written")

    summaries = load_all_summaries(cells)
    per_seed, agg = sweep_summary(summaries)
    per_seed.to_csv(ANALYSIS_DIR / "sweep_per_seed.csv", index=False)
    agg.to_csv(ANALYSIS_DIR / "sweep_summary.csv", index=False)
    cellwise, sign = paired_tests(per_seed)
    cellwise.to_csv(ANALYSIS_DIR / "paired_tests.csv", index=False)
    sign.to_csv(ANALYSIS_DIR / "sign_tests.csv", index=False)
    print("sweep + paired tests written")

    boot = hierarchical_bootstrap(cells)
    boot.to_csv(ANALYSIS_DIR / "bootstrap_keep18.csv", index=False)
    print("bootstrap written")

    od = oracle_detector(cells)
    od.to_csv(ANALYSIS_DIR / "oracle_detector.csv", index=False)
    fallback = confidence_fallback(cells)
    fallback.to_csv(ANALYSIS_DIR / "confidence_fallback.csv", index=False)
    scratch = scratch_summary()
    if len(scratch):
        scratch.to_csv(ANALYSIS_DIR / "scratch_summary.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
