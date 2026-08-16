"""Train one B4 PatchTST baseline via TSLib run.py (seed-patched).

Mirrors notebook 14's training command exactly (same hyperparameters).
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from common import (TSLIB_DIR, TSLIB_COMMIT, DATA_DIR, DATASETS, cell_dir,
                    sha256_file, build_args)


def train(dataset, pred_len, seed):
    cfg = DATASETS[dataset]
    args = build_args(dataset, pred_len)
    seed_dir = cell_dir(dataset, pred_len) / f"seed_{seed}"
    checkpoint_root = seed_dir / "checkpoints"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    meta_path = seed_dir / "run_metadata.json"
    if meta_path.exists() and list(checkpoint_root.glob("**/checkpoint.pth")):
        print(f"{dataset} pred{pred_len} seed{seed}: already trained")
        return json.loads(meta_path.read_text())

    model_id = f"{dataset}_{args.seq_len}_{pred_len}_dm128_h8_seed{seed}"
    command = [
        sys.executable, "-u", "run.py",
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", str(DATA_DIR) + "/", "--data_path", cfg["csv"],
        "--model_id", model_id, "--model", "PatchTST",
        "--data", cfg["data_flag"], "--features", "M",
        "--seq_len", str(args.seq_len), "--label_len", str(args.label_len),
        "--pred_len", str(pred_len),
        "--enc_in", str(args.enc_in), "--dec_in", str(args.dec_in),
        "--c_out", str(args.c_out),
        "--e_layers", "3", "--d_layers", "1", "--factor", "3",
        "--d_model", "128", "--d_ff", "256", "--n_heads", "8",
        "--batch_size", "32", "--train_epochs", "10", "--patience", "3",
        "--learning_rate", "0.0001", "--num_workers", "0",
        "--freq", cfg["freq"],
        "--checkpoints", str(checkpoint_root),
        "--des", f"baseline_b4_seed{seed}", "--itr", "1",
    ]
    env = os.environ.copy()
    env["RECAHS_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    log_path = seed_dir / "training.log"
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=str(TSLIB_DIR), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in process.stdout:
            log_file.write(line)
        returncode = process.wait()
    finished = datetime.now(timezone.utc)
    if returncode != 0:
        raise RuntimeError(
            f"training failed for {dataset}/pred{pred_len}/seed{seed}; "
            f"see {log_path}")

    metric_candidates = sorted(
        (TSLIB_DIR / "results").glob(f"*{model_id}*/metrics.npy"),
        key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_candidates = sorted(checkpoint_root.glob("**/checkpoint.pth"))
    if not metric_candidates or len(checkpoint_candidates) != 1:
        raise RuntimeError(
            f"{dataset}/pred{pred_len}/seed{seed}: metrics="
            f"{len(metric_candidates)} ckpt={len(checkpoint_candidates)}")

    metrics = np.load(metric_candidates[0])
    record = {
        "dataset": dataset, "pred_len": pred_len, "seed": seed,
        "method": "unpruned_baseline",
        "test_mae": float(metrics[0]), "test_mse": float(metrics[1]),
        "test_rmse": float(metrics[2]),
        "dataset_sha256": sha256_file(DATA_DIR / cfg["csv"]),
        "tslib_commit": TSLIB_COMMIT,
        "checkpoint": str(checkpoint_candidates[0]),
        "checkpoint_sha256": sha256_file(checkpoint_candidates[0]),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{dataset} pred{pred_len} seed{seed}: "
          f"test_mse={record['test_mse']:.4f} "
          f"({record['duration_seconds']:.0f}s)")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    cli = parser.parse_args()
    train(cli.dataset, cli.pred_len, cli.seed)
