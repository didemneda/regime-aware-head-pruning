"""Slow-timescale routing + regret-gated deviation, from stored losses.

All methods are inference-time legal (feedback is used only after the
pred_len observation delay) and are compared against pooled static-greedy.

  block_K      -- re-decide every K windows: pick the mask with lowest mean
                  observed regret over the last W observable windows.
  swucb_K      -- same decision points, but sliding-window UCB1 (adds an
                  exploration bonus so under-sampled masks get re-checked).
  gate_qXX     -- delta-regression router (trained on val) that deviates
                  from static_greedy only on the XX% of windows with the
                  largest predicted advantage.

Writes runs/analysis/router_extras.csv over all 16 cells x 5 seeds.
"""
import numpy as np
import pandas as pd

from common import RUNS_DIR
from oracle_router import cell_features
from router_lab import fit_predict_delta, load_cell

WINDOW = 336            # feedback window (observable windows)
BLOCKS = [24, 96, 336]
GATE_QUANTILES = [0.95, 0.90, 0.75]
UCB_C = 0.5


def observed(losses, t, pred_len):
    """Indices of windows whose ground truth is available at window t."""
    return t - pred_len


def block_router(L_test, pred_len, block, ucb=False):
    n = L_test.shape[1]
    regret = L_test - L_test.min(axis=0, keepdims=True)
    route = np.full(n, 3, dtype=int)          # warmup -> static_greedy
    counts = np.ones(4)
    for start in range(0, n, block):
        hi = start - pred_len                  # last observable window
        if hi <= 0:
            continue
        lo = max(0, hi - WINDOW)
        means = regret[:, lo:hi].mean(axis=1)
        if ucb:
            bonus = UCB_C * np.sqrt(np.log(max(start, 2)) / counts)
            score = means - bonus * regret[:, lo:hi].std(axis=1)
        else:
            score = means
        arm = int(np.argmin(score))
        counts[arm] += 1
        route[start:start + block] = arm
    return route


def gated_delta(L_val, L_test, X_val, X_test, quantile):
    pred = fit_predict_delta(X_val, L_val, X_test)      # (4, n) centered
    advantage = pred[3] - pred.min(axis=0)              # predicted gain
    threshold = np.quantile(advantage, quantile)
    route = np.where(advantage >= threshold, pred.argmin(axis=0), 3)
    return route, float((route != 3).mean() * 100)


def main():
    rows = []
    for npz_path in sorted(RUNS_DIR.glob(
            "*/pred*/seed_*/pruning/window_losses.npz")):
        try:
            dataset, pred_len, seed, losses = load_cell(str(npz_path))
        except (KeyError, FileNotFoundError):
            continue
        L_val, L_test = losses["val"], losses["test"]
        n = L_test.shape[1]
        idx = np.arange(n)
        sg = float(L_test[3].mean())
        oracle = float(L_test.min(axis=0).mean())
        record = {"dataset": dataset, "pred_len": pred_len, "seed": seed,
                  "static_greedy": sg, "oracle4": oracle}

        def add(name, route, extra=None):
            mse = float(L_test[route, idx].mean())
            record[name] = mse
            record[f"{name}_recovery_pct"] = (
                100.0 * (sg - mse) / (sg - oracle)
                if sg - oracle > 1e-12 else np.nan)
            if extra is not None:
                record[f"{name}_deviate_pct"] = extra

        for block in BLOCKS:
            add(f"block_{block}", block_router(L_test, pred_len, block))
            add(f"swucb_{block}",
                block_router(L_test, pred_len, block, ucb=True))
        X_val = cell_features(dataset, pred_len, "val", L_val.shape[1])
        X_test = cell_features(dataset, pred_len, "test", n)
        for q in GATE_QUANTILES:
            route, dev = gated_delta(L_val, L_test, X_val, X_test, q)
            add(f"gate_q{int(q*100)}", route, extra=dev)
        rows.append(record)
        print(f"{dataset}@{pred_len} s{seed}: sg={sg:.4f} "
              f"block96={record['block_96']:.4f} "
              f"gate95={record['gate_q95']:.4f} "
              f"oracle={oracle:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    out = RUNS_DIR / "analysis" / "router_extras.csv"
    frame.to_csv(out, index=False)
    cell = frame.groupby(["dataset", "pred_len"]).mean(numeric_only=True)
    methods = ([f"block_{b}" for b in BLOCKS]
               + [f"swucb_{b}" for b in BLOCKS]
               + [f"gate_q{int(q*100)}" for q in GATE_QUANTILES])
    print(f"\n=== wins vs static_greedy over {len(cell)} cells ===")
    for m in methods:
        wins = int((cell[m] < cell["static_greedy"]).sum())
        rec = cell[f"{m}_recovery_pct"].mean()
        print(f"{m:10s} wins {wins:2d}/{len(cell)}  recovery {rec:+.1f}%")
    print("written:", out)


if __name__ == "__main__":
    main()
