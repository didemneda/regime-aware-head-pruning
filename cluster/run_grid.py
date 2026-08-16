"""Grid orchestrator: distributes cell jobs across the two GPUs.

Usage:
  python run_grid.py --stage labels
  python run_grid.py --stage train
  python run_grid.py --stage prune
  python run_grid.py --stage scratch
"""
import argparse
import itertools
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

from common import RUNS_DIR, SEEDS, cell_dir

GRID_DATASETS = os.environ.get(
    "RECAHS_DATASETS", "ETTh1,ETTh2,ETTm1").split(",")
GRID_PREDS = [int(x) for x in os.environ.get(
    "RECAHS_PREDS", "96,192,336,720").split(",")]
SCRATCH_CELLS = [("ETTh1", 96), ("ETTm1", 96)]
GPUS = [0, 1]
WORKERS_PER_GPU = int(os.environ.get("RECAHS_WORKERS_PER_GPU", "1"))
LOG_DIR = RUNS_DIR / "logs"


def tasks_for(stage):
    tasks = []
    if stage == "labels":
        for ds, p in itertools.product(GRID_DATASETS, GRID_PREDS):
            splits = ["val", "test"]
            if (ds, p) in SCRATCH_CELLS:
                splits.append("train")
            tasks.append((f"labels_{ds}_p{p}", [
                sys.executable, "-u", "stl_labels.py", "--dataset", ds,
                "--pred_len", str(p), "--splits", *splits], None))
    elif stage == "train":
        for p, ds, s in itertools.product(GRID_PREDS, GRID_DATASETS, SEEDS):
            if (cell_dir(ds, p) / f"seed_{s}" / "run_metadata.json").exists():
                continue
            tasks.append((f"train_{ds}_p{p}_s{s}", [
                sys.executable, "-u", "train_baselines.py", "--dataset", ds,
                "--pred_len", str(p), "--seed", str(s)], "gpu"))
    elif stage == "prune":
        for p, ds, s in itertools.product(GRID_PREDS, GRID_DATASETS, SEEDS):
            if (cell_dir(ds, p) / f"seed_{s}" / "pruning" / "DONE").exists():
                continue
            tasks.append((f"prune_{ds}_p{p}_s{s}", [
                sys.executable, "-u", "prune_eval.py", "--dataset", ds,
                "--pred_len", str(p), "--seed", str(s)], "gpu"))
    elif stage == "static_greedy":
        for p, ds, s in itertools.product(GRID_PREDS, GRID_DATASETS, SEEDS):
            pr = cell_dir(ds, p) / f"seed_{s}" / "pruning"
            if not (pr / "DONE").exists() or (pr / "GREEDY_DONE").exists():
                continue
            tasks.append((f"sgreedy_{ds}_p{p}_s{s}", [
                sys.executable, "-u", "static_greedy.py", "--dataset", ds,
                "--pred_len", str(p), "--seed", str(s)], "gpu"))
    elif stage == "scratch":
        for (ds, p), s, m in itertools.product(
                SCRATCH_CELLS, SEEDS, ["static", "regime"]):
            if (cell_dir(ds, p) / f"seed_{s}" / f"scratch_{m}"
                    / "summary.csv").exists():
                continue
            tasks.append((f"scratch_{ds}_p{p}_s{s}_{m}", [
                sys.executable, "-u", "masked_scratch.py", "--dataset", ds,
                "--pred_len", str(p), "--seed", str(s), "--method", m],
                "gpu"))
    else:
        raise ValueError(stage)
    return tasks


def worker(queue, gpu, failures):
    while True:
        item = queue.get()
        if item is None:
            return
        name, command, needs_gpu = item
        env = os.environ.copy()
        if needs_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["OMP_NUM_THREADS"] = str(
                max(2, 28 // (len(GPUS) * WORKERS_PER_GPU)))
        log_path = LOG_DIR / f"{name}.log"
        t0 = time.time()
        with log_path.open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=log, env=env,
                                   cwd=str(Path(__file__).parent))
        status = "ok" if code == 0 else f"FAIL({code})"
        print(f"[gpu{gpu}] {name}: {status} {time.time()-t0:.0f}s",
              flush=True)
        if code != 0:
            failures.append(name)


def main(stage):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = tasks_for(stage)
    print(f"stage={stage}: {len(tasks)} tasks", flush=True)
    if not tasks:
        return
    queue = Queue()
    for task in tasks:
        queue.put(task)
    n_workers = (len(GPUS) * WORKERS_PER_GPU
                 if tasks[0][2] == "gpu" else 1)
    failures = []
    threads = []
    for i in range(n_workers):
        queue.put(None)
        thread = threading.Thread(
            target=worker, args=(queue, GPUS[i % len(GPUS)], failures))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    print(f"stage={stage} complete, failures={failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["labels", "train", "prune",
                                 "static_greedy", "scratch"])
    main(parser.parse_args().stage)
