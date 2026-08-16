"""Cheap learned regime detector to replace per-window STL at serving time.

Trains on train-split STL labels, evaluates label accuracy on val/test, and
measures per-window latency of feature extraction + prediction. End-to-end
forecasting impact is computed offline in stats.py from the stored
per-regime-mask window losses (no model re-runs needed).
"""
import argparse
import time

import numpy as np
import pandas as pd

from common import (DATA_DIR, DATASETS, SEQ_LEN, cell_dir, load_labels,
                    split_borders)


def window_features(values, period):
    """~12 cheap features for a length-336 window (no STL)."""
    x = np.arange(len(values), dtype=np.float64)
    mean = values.mean()
    std = values.std() + 1e-12
    z = (values - mean) / std
    # linear trend fit
    slope, intercept = np.polyfit(x, z, 1)
    resid_lin = z - (slope * x + intercept)
    r2_lin = 1.0 - resid_lin.var() / (z.var() + 1e-12)
    # seasonal energy: FFT power at the seasonal frequency and harmonics
    spec = np.abs(np.fft.rfft(z)) ** 2
    total_power = spec[1:].sum() + 1e-12
    k = len(values) / period  # fundamental seasonal bin
    seasonal_power = 0.0
    for harmonic in (1, 2, 3):
        bin_center = harmonic * k
        low = int(np.floor(bin_center - 1))
        high = int(np.ceil(bin_center + 1))
        seasonal_power += spec[max(low, 1):min(high + 1, len(spec))].sum()
    # low-frequency (trend) power: bins below half the fundamental
    trend_power = spec[1:max(int(k / 2), 2)].sum()
    diff = np.diff(z)
    lag1 = np.corrcoef(z[:-1], z[1:])[0, 1]
    lagp = (np.corrcoef(z[:-period], z[period:])[0, 1]
            if len(z) > period else 0.0)
    return np.array([
        slope, r2_lin, seasonal_power / total_power,
        trend_power / total_power, diff.std(), diff.mean(),
        lag1, lagp, z.min(), z.max(),
        np.abs(np.diff(np.sign(diff))).mean(),  # turning-point rate
        std,
    ])


def build_features(dataset, pred_len, split):
    cfg = DATASETS[dataset]
    raw = pd.read_csv(DATA_DIR / cfg["csv"])
    border1, border2 = split_borders(dataset, split, len(raw))
    values = raw.iloc[border1:border2]["OT"].to_numpy(dtype=np.float64)
    labels = load_labels(dataset, pred_len, split)
    features = np.stack([
        window_features(values[i:i + SEQ_LEN], cfg["stl_period"])
        for i in range(len(labels))])
    return features, labels["regime"].to_numpy()


def main(dataset, pred_len):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out_dir = cell_dir(dataset, pred_len) / "detector"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = build_features(dataset, pred_len, "train")
    X_val, y_val = build_features(dataset, pred_len, "val")
    X_test, y_test = build_features(dataset, pred_len, "test")

    models = {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(random_state=0),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_acc = float((model.predict(X_val) == y_val).mean())
        test_pred = model.predict(X_test)
        test_acc = float((test_pred == y_test).mean())
        # latency: features + predict, per window, CPU
        cfg = DATASETS[dataset]
        raw = pd.read_csv(DATA_DIR / cfg["csv"])
        border1, _ = split_borders(dataset, "test", len(raw))
        sample = raw.iloc[border1:border1 + SEQ_LEN + 200]["OT"].to_numpy(
            dtype=np.float64)
        t0 = time.perf_counter()
        n_timing = 200
        for i in range(n_timing):
            feats = window_features(sample[i:i + SEQ_LEN], cfg["stl_period"])
            model.predict(feats.reshape(1, -1))
        latency_ms = (time.perf_counter() - t0) / n_timing * 1000
        rows.append({"dataset": dataset, "pred_len": pred_len,
                     "model": name, "val_acc": val_acc, "test_acc": test_acc,
                     "latency_ms_per_window": latency_ms})
        np.save(out_dir / f"test_pred_{name}.npy", test_pred)
        np.save(out_dir / f"val_pred_{name}.npy", model.predict(X_val))
        print(f"{name}: val_acc={val_acc:.3f} test_acc={test_acc:.3f} "
              f"latency={latency_ms:.3f}ms")

    # STL latency reference on the same windows
    from common import stl_label_window
    cfg = DATASETS[dataset]
    t0 = time.perf_counter()
    for i in range(20):
        stl_label_window(sample[i:i + SEQ_LEN], cfg["stl_period"])
    stl_ms = (time.perf_counter() - t0) / 20 * 1000
    rows.append({"dataset": dataset, "pred_len": pred_len, "model": "stl",
                 "val_acc": 1.0, "test_acc": 1.0,
                 "latency_ms_per_window": stl_ms})
    print(f"stl reference latency: {stl_ms:.1f}ms")
    pd.DataFrame(rows).to_csv(out_dir / "detector_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    cli = parser.parse_args()
    main(cli.dataset, cli.pred_len)
