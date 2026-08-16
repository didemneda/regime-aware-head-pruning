"""Router lab: can ANY inference-time-legal router close the oracle gap?

All methods run on the stored per-window mask losses (no model re-runs)
and are compared against the pooled static-greedy mask at keep=18.

Methods (all trained on validation only, except the labeled diagnostic):
  argmin_clf     -- v1 baseline: HGB classifier on argmin labels.
  loss_reg       -- HGB regressor per mask on raw per-window loss;
                    route to lowest predicted loss.
  delta_reg      -- HGB regressor per mask on centered loss
                    (loss_m - mean_m loss): removes window-difficulty
                    variance, keeps only the discriminative signal.
  delta_reg_mw   -- delta_reg with margin sample-weights (windows where
                    masks differ most count most).
  temporal       -- deployable delayed-feedback router: at test window w,
                    route by rolling mean observed regret over windows
                    <= w - pred_len (ground truth already observable);
                    warmup windows fall back to static_greedy. No learning.
  temporal_feat  -- delta_reg with rolling observed-regret features
                    appended (delayed-feedback + input features).
  DIAG_insample  -- diagnostic ONLY (fits on test): upper bound on what
                    these features could ever predict. Not deployable.

Outputs runs/analysis/router_lab.csv with per-cell test MSE, oracle
regret recovery, and win counts vs static_greedy.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from common import RUNS_DIR
from oracle_router import cell_features, KEEP, REGIMES

CANDIDATES = REGIMES + ["static_greedy"]
ROLL_K = 96          # rolling window (in windows) for delayed feedback


def load_cell(npz_path):
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
        losses[split] = np.stack(stack).astype(np.float64)
    return dataset, pred_len, seed, losses


def fit_predict_delta(X_tr, L_tr, X_te, margin_weight=False):
    """Per-mask regression of centered loss; returns predicted (4, n_te)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    centered = L_tr - L_tr.mean(axis=0, keepdims=True)
    weight = None
    if margin_weight:
        gap = np.sort(L_tr, axis=0)[1] - L_tr.min(axis=0)
        weight = gap / (gap.mean() + 1e-12)
    preds = []
    for m in range(L_tr.shape[0]):
        reg = HistGradientBoostingRegressor(random_state=0)
        reg.fit(X_tr, centered[m], sample_weight=weight)
        preds.append(reg.predict(X_te))
    return np.stack(preds)


def fit_predict_raw(X_tr, L_tr, X_te):
    from sklearn.ensemble import HistGradientBoostingRegressor
    preds = []
    for m in range(L_tr.shape[0]):
        reg = HistGradientBoostingRegressor(random_state=0)
        reg.fit(X_tr, L_tr[m])
        preds.append(reg.predict(X_te))
    return np.stack(preds)


def fit_predict_clf(X_tr, L_tr, X_te):
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = L_tr.argmin(axis=0)
    margin = L_tr.max(axis=0) - L_tr.min(axis=0)
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X_tr, y, sample_weight=margin / (margin.mean() + 1e-12))
    pred = clf.predict(X_te)
    scores = np.full((L_tr.shape[0], len(pred)), 0.0)
    for i, p in enumerate(pred):
        scores[:, i] = 1.0
        scores[p, i] = 0.0          # chosen mask gets lowest "loss"
    return scores


def delayed_regret_features(L_test, pred_len, roll=ROLL_K):
    """(4, n) rolling mean regret over observable windows (delay=pred_len)."""
    n = L_test.shape[1]
    regret = L_test - L_test.min(axis=0, keepdims=True)
    cum = np.cumsum(regret, axis=1)
    feats = np.zeros_like(regret)
    for w in range(n):
        hi = w - pred_len            # last observable window index
        if hi < 0:
            feats[:, w] = 0.0
            continue
        lo = max(0, hi - roll)
        feats[:, w] = (cum[:, hi] - (cum[:, lo - 1] if lo > 0 else 0)) \
            / (hi - lo + 1)
    return feats, (np.arange(n) - pred_len >= 0)


def evaluate_route(L_test, route_idx):
    return float(L_test[route_idx, np.arange(L_test.shape[1])].mean())


def run_cell(npz_path):
    dataset, pred_len, seed, losses = load_cell(npz_path)
    L_val, L_test = losses["val"], losses["test"]
    n_val, n_test = L_val.shape[1], L_test.shape[1]
    X_val = cell_features(dataset, pred_len, "val", n_val)
    X_test = cell_features(dataset, pred_len, "test", n_test)

    sg_mse = float(L_test[3].mean())
    oracle_mse = float(L_test.min(axis=0).mean())
    record = {"dataset": dataset, "pred_len": pred_len, "seed": seed,
              "static_greedy": sg_mse, "oracle4": oracle_mse}

    def add(name, route_idx):
        mse = evaluate_route(L_test, route_idx)
        record[name] = mse
        gap = sg_mse - oracle_mse
        record[f"{name}_recovery_pct"] = (
            100.0 * (sg_mse - mse) / gap if gap > 1e-12 else np.nan)

    # 1) argmin classification (v1 baseline)
    add("argmin_clf", fit_predict_clf(X_val, L_val, X_test).argmin(axis=0))
    # 2) raw loss regression
    add("loss_reg", fit_predict_raw(X_val, L_val, X_test).argmin(axis=0))
    # 3) centered (soft) regression
    add("delta_reg", fit_predict_delta(X_val, L_val, X_test).argmin(axis=0))
    # 4) + margin weights
    add("delta_reg_mw", fit_predict_delta(
        X_val, L_val, X_test, margin_weight=True).argmin(axis=0))
    # 5) delayed-feedback rolling router (deployable, no learning)
    roll, observable = delayed_regret_features(L_test, pred_len)
    route = np.where(observable, roll.argmin(axis=0), 3)
    add("temporal", route)
    # 6) delta_reg + delayed-feedback features
    roll_val, obs_val = delayed_regret_features(L_val, pred_len)
    Xv = np.hstack([X_val, roll_val.T])
    Xt = np.hstack([X_test, roll.T])
    pred = fit_predict_delta(Xv, L_val, Xt).argmin(axis=0)
    add("temporal_feat", np.where(observable, pred, 3))
    # 7) DIAGNOSTIC: in-sample fit on test (upper bound of feature signal)
    add("DIAG_insample",
        fit_predict_delta(X_test, L_test, X_test).argmin(axis=0))
    return record


def main():
    rows = []
    for npz_path in sorted(RUNS_DIR.glob(
            "*/pred*/seed_*/pruning/window_losses.npz")):
        try:
            rows.append(run_cell(str(npz_path)))
            r = rows[-1]
            print(f"{r['dataset']}@{r['pred_len']} s{r['seed']}: "
                  f"sg={r['static_greedy']:.4f} "
                  f"delta={r['delta_reg']:.4f} "
                  f"temporal={r['temporal']:.4f} "
                  f"diag={r['DIAG_insample']:.4f} "
                  f"oracle={r['oracle4']:.4f}", flush=True)
        except (KeyError, FileNotFoundError) as error:
            print(f"skip {npz_path}: {error}", flush=True)
    frame = pd.DataFrame(rows)
    out = RUNS_DIR / "analysis" / "router_lab.csv"
    frame.to_csv(out, index=False)

    methods = ["argmin_clf", "loss_reg", "delta_reg", "delta_reg_mw",
               "temporal", "temporal_feat", "DIAG_insample"]
    cell = frame.groupby(["dataset", "pred_len"]).mean(numeric_only=True)
    print("\n=== wins vs static_greedy over cells / mean oracle recovery ===")
    for m in methods:
        wins = int((cell[m] < cell["static_greedy"]).sum())
        rec = cell[f"{m}_recovery_pct"].mean()
        print(f"{m:14s} wins {wins:2d}/{len(cell)}  recovery {rec:+.1f}%")
    print("written:", out)


if __name__ == "__main__":
    main()
