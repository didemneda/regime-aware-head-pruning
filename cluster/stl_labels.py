"""Compute STL regime labels for one (dataset, pred_len) cell, all splits.

Parallelized over CPU cores. Labels are aligned with TSLib loader windows
(border arithmetic mirrored from the loaders and asserted against loader
lengths in prune_eval.py).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# one BLAS thread per STL worker; parallelism comes from the process pool
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import pandas as pd

from common import (DATA_DIR, DATASETS, SEQ_LEN, CONFIDENCE_THRESHOLD,
                    labels_path, split_borders, stl_label_window)


def _label_one(window_values, period):
    return stl_label_window(window_values, period)


def label_split(dataset, pred_len, split, workers):
    cfg = DATASETS[dataset]
    out_path = labels_path(dataset, pred_len, split)
    raw = pd.read_csv(DATA_DIR / cfg["csv"])
    border1, border2 = split_borders(dataset, split, len(raw))
    values = raw.iloc[border1:border2]["OT"].to_numpy(dtype=np.float64)
    count = len(values) - SEQ_LEN - pred_len + 1
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if len(existing) == count:
            print(f"{dataset} pred{pred_len} {split}: cached ({count})")
            return
    windows = [values[i:i + SEQ_LEN] for i in range(count)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(
            partial(_label_one, period=cfg["stl_period"]),
            windows, chunksize=64))
    labels = pd.DataFrame(records)
    labels.insert(0, "window_id", np.arange(count))
    labels["is_confident"] = (
        labels["confidence_margin"] >= CONFIDENCE_THRESHOLD)
    labels.to_csv(out_path, index=False)
    print(f"{dataset} pred{pred_len} {split}: {count} windows -> {out_path}")
    print(labels["regime"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=30)
    cli = parser.parse_args()
    for split in cli.splits:
        label_split(cli.dataset, cli.pred_len, split, cli.workers)
