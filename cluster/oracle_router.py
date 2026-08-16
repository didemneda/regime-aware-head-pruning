"""Learned oracle router: predict the best mask per window directly.

For each (dataset, pred_len, seed) cell at keep=18:
  candidates = {trend, seasonal, residual, static_greedy} masks.
  Train labels: argmin per-window VAL loss over the candidates
                (val targets only -- no test information).
  Features:     the cheap window features from detector.py.
  Router:       HistGradientBoosting classifier, sample-weighted by the
                margin between best and worst candidate (windows where the
                choice matters count more).
  Evaluation:   route each TEST window by predicted mask id and look up its
                stored per-window loss. Compare vs static_greedy, STL
                routing, and the true oracle over the same 4 candidates.

CPU-only; uses only stored losses + raw series. Writes
runs/analysis/oracle_router.csv.
"""
import glob
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import DATA_DIR, DATASETS, RUNS_DIR, SEQ_LEN, split_borders
from detector import window_features

REGIMES = ["trend", "seasonal", "residual"]
CANDIDATES = REGIMES + ["static_greedy"]
KEEP = 18


def cell_features(dataset, pred_len, split, n_windows, cache={}):
    key = (dataset, pred_len, split)
    if key not in cache:
        cfg = DATASETS[dataset]
        raw = pd.read_csv(DATA_DIR / cfg["csv"])
        border1, border2 = split_borders(dataset, split, len(raw))
        values = raw.iloc[border1:border2]["OT"].to_numpy(dtype=np.float64)
        cache[key] = np.stack([
            window_features(values[i:i + SEQ_LEN], cfg["stl_period"])
            for i in range(n_windows)])
    return cache[key]


def run_cell(npz_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    match = re.match(r".*/(.+)/pred(\d+)/seed_(\d+)/", npz_path)
    dataset, pred_len, seed = (match.group(1), int(match.group(2)),
                               int(match.group(3)))
    z = np.load(npz_path)
    sg = np.load(npz_path.replace("window_losses", "static_greedy_losses"))

    losses = {}
    for split in ("val", "test"):
        stack = [z[f"{split}/regimemask_{r}/keep{KEEP}/mse"]
                 for r in REGIMES]
        stack.append(sg[f"{split}/static_greedy/keep{KEEP}/mse"])
        losses[split] = np.stack(stack)          # (4, n_windows)

    labels_dir = RUNS_DIR / dataset / f"pred{pred_len}" / "regime_labels"
    test_regimes = pd.read_csv(labels_dir / "test_regimes.csv")[
        "regime"].to_numpy()

    n_val = losses["val"].shape[1]
    n_test = losses["test"].shape[1]
    X_val = cell_features(dataset, pred_len, "val", n_val)
    X_test = cell_features(dataset, pred_len, "test", n_test)

    y_val = losses["val"].argmin(axis=0)
    margin = losses["val"].max(axis=0) - losses["val"].min(axis=0)
    weight = margin / (margin.mean() + 1e-12)

    router = HistGradientBoostingClassifier(random_state=0)
    router.fit(X_val, y_val, sample_weight=weight)
    pred = router.predict(X_test)

    n = np.arange(n_test)
    routed = losses["test"][pred, n]
    idx = {r: i for i, r in enumerate(REGIMES)}
    stl = losses["test"][[idx[r] for r in test_regimes], n]
    oracle = losses["test"].min(axis=0)
    return {
        "dataset": dataset, "pred_len": pred_len, "seed": seed,
        "baseline": float(z["test/baseline/keep24/mse"].mean()),
        "static_greedy": float(losses["test"][3].mean()),
        "dynamic_stl": float(stl.mean()),
        "learned_router": float(routed.mean()),
        "oracle4": float(oracle.mean()),
        "router_matches_oracle_pct": float(
            (pred == losses["test"].argmin(axis=0)).mean() * 100),
        "router_pick_distribution": dict(Counter(
            [CANDIDATES[i] for i in pred])),
    }


def main():
    rows = []
    for npz_path in sorted(RUNS_DIR.glob(
            "*/pred*/seed_*/pruning/window_losses.npz")):
        try:
            rows.append(run_cell(str(npz_path)))
            r = rows[-1]
            print(f"{r['dataset']}@{r['pred_len']} s{r['seed']}: "
                  f"sg={r['static_greedy']:.4f} stl={r['dynamic_stl']:.4f} "
                  f"router={r['learned_router']:.4f} "
                  f"oracle={r['oracle4']:.4f}", flush=True)
        except (KeyError, FileNotFoundError) as error:
            print(f"skip {npz_path}: {error}", flush=True)
    frame = pd.DataFrame(rows)
    out = RUNS_DIR / "analysis" / "oracle_router.csv"
    frame.to_csv(out, index=False)
    print("written:", out)


if __name__ == "__main__" and __import__("sys").argv[-1] != "--gated":
    main()


def run_cell_gated(npz_path, thresholds=(0.5, 0.6, 0.7, 0.8)):
    """v2: confidence-gated router — deviate from static_greedy only when
    the router's predicted probability for its pick exceeds a threshold."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    match = re.match(r".*/(.+)/pred(\d+)/seed_(\d+)/", npz_path)
    dataset, pred_len, seed = (match.group(1), int(match.group(2)),
                               int(match.group(3)))
    z = np.load(npz_path)
    sg = np.load(npz_path.replace("window_losses", "static_greedy_losses"))
    losses = {}
    for split in ("val", "test"):
        stack = [z[f"{split}/regimemask_{r}/keep{KEEP}/mse"]
                 for r in REGIMES]
        stack.append(sg[f"{split}/static_greedy/keep{KEEP}/mse"])
        losses[split] = np.stack(stack)
    n_val = losses["val"].shape[1]
    n_test = losses["test"].shape[1]
    X_val = cell_features(dataset, pred_len, "val", n_val)
    X_test = cell_features(dataset, pred_len, "test", n_test)
    y_val = losses["val"].argmin(axis=0)
    margin = losses["val"].max(axis=0) - losses["val"].min(axis=0)
    router = HistGradientBoostingClassifier(random_state=0)
    router.fit(X_val, y_val, sample_weight=margin / (margin.mean() + 1e-12))
    proba = router.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    classes = router.classes_
    n = np.arange(n_test)
    record = {"dataset": dataset, "pred_len": pred_len, "seed": seed,
              "static_greedy": float(losses["test"][3].mean())}
    for t in thresholds:
        routed_idx = np.where(conf >= t, classes[pred], 3)
        routed = losses["test"][routed_idx, n]
        record[f"gated_{t}"] = float(routed.mean())
        record[f"gated_{t}_deviate_pct"] = float((routed_idx != 3).mean()
                                                 * 100)
    return record


def main_gated():
    rows = []
    for npz_path in sorted(RUNS_DIR.glob(
            "*/pred*/seed_*/pruning/window_losses.npz")):
        try:
            rows.append(run_cell_gated(str(npz_path)))
        except (KeyError, FileNotFoundError):
            continue
    frame = pd.DataFrame(rows)
    frame.to_csv(RUNS_DIR / "analysis" / "oracle_router_gated.csv",
                 index=False)
    print("gated written")


if __name__ == "__main__" and __import__("sys").argv[-1] == "--gated":
    main_gated()
