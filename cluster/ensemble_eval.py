"""Prediction ensembling over the 4 candidate masks (@96 cells, keep=18).

Instead of routing (selecting one mask per window), run all four masked
forward passes and AVERAGE the predictions. Schemes:
  mean4     -- uniform average of trend/seasonal/residual/static_greedy
  mean3     -- uniform average of the three regime masks
  top2      -- uniform average of the two masks with lowest val MSE
  invmse    -- weights proportional to 1/mean val MSE (val-derived only)

Per-window test MSE of each scheme is computed on the fly (predictions are
never stored), cross-checked against stored single-mask losses, and written
to pruning/ensemble_losses.npz + runs/analysis/ensemble_eval.csv.
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (EVAL_BATCH, RUNS_DIR, HeadMaskController, build_model,
                    cell_dir, find_checkpoint, get_split_dataset,
                    mask_from_removal_prefix)
from router_lab import KEEP, REGIMES, load_cell

PRED_LEN = 96
DATASETS_96 = ["ETTh1", "ETTh2", "ETTm1", "weather"]
SEEDS = [7, 42, 1234, 2026, 3407]
MASK_NAMES = REGIMES + ["static_greedy"]


def build_masks(prune_dir):
    orders = json.loads((prune_dir / "removal_orders.json").read_text())
    sg_order = json.loads(
        (prune_dir / "static_greedy_order.json").read_text())
    static_mask = mask_from_removal_prefix(
        [tuple(x) for x in orders["static"]], KEEP)
    masks = {}
    for regime in REGIMES:
        if regime in orders["greedy"]:
            masks[regime] = mask_from_removal_prefix(
                [tuple(x) for x in orders["greedy"][regime]], KEEP)
        else:
            masks[regime] = static_mask
    masks["static_greedy"] = mask_from_removal_prefix(
        [tuple(x) for x in sg_order["pooled"]], KEEP)
    return masks


def run_cell(dataset, seed, device):
    prune_dir = (cell_dir(dataset, PRED_LEN) / f"seed_{seed}" / "pruning")
    npz = str(prune_dir / "window_losses.npz")
    _, _, _, losses = load_cell(npz)
    L_val, L_test = losses["val"], losses["test"]

    # val-derived weights
    val_means = L_val.mean(axis=1)
    inv = 1.0 / val_means
    schemes = {
        "mean4": np.ones(4) / 4.0,
        "mean3": np.array([1 / 3, 1 / 3, 1 / 3, 0.0]),
        "invmse": inv / inv.sum(),
    }
    top2 = np.argsort(val_means)[:2]
    w = np.zeros(4)
    w[top2] = 0.5
    schemes["top2"] = w

    masks = build_masks(prune_dir)
    model, args = build_model(
        dataset, PRED_LEN, find_checkpoint(dataset, PRED_LEN, seed), device)
    controller = HeadMaskController(model)
    controller.install()
    loader = DataLoader(get_split_dataset(dataset, PRED_LEN, "test"),
                        batch_size=EVAL_BATCH, shuffle=False, num_workers=0)

    per_scheme = {k: [] for k in schemes}
    per_mask_check = {m: [] for m in MASK_NAMES}
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            true = batch_y[:, -PRED_LEN:, :]
            preds = []
            for name in MASK_NAMES:
                controller.set_mask(masks[name])
                out = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                preds.append(out)
                per_mask_check[name].append(
                    ((out - true) ** 2).mean(dim=(1, 2)).cpu().numpy())
            stack = torch.stack(preds)              # (4, B, pred, C)
            for key, weight in schemes.items():
                wt = torch.tensor(weight, dtype=stack.dtype,
                                  device=stack.device).view(4, 1, 1, 1)
                combo = (stack * wt).sum(dim=0)
                per_scheme[key].append(
                    ((combo - true) ** 2).mean(dim=(1, 2)).cpu().numpy())

    record = {"dataset": dataset, "seed": seed,
              "baseline": float(np.load(npz)["test/baseline/keep24/mse"]
                                .mean()),
              "static_greedy": float(L_test[3].mean()),
              "oracle4": float(L_test.min(axis=0).mean())}
    # alignment cross-check
    for i, name in enumerate(MASK_NAMES):
        fresh = np.concatenate(per_mask_check[name])
        drift = abs(fresh.mean() - L_test[i].mean())
        if drift > 1e-4:
            raise RuntimeError(f"mask loss mismatch {name}: {drift}")
    store = {}
    sg = record["static_greedy"]
    oracle = record["oracle4"]
    for key in schemes:
        vec = np.concatenate(per_scheme[key])
        store[f"test/{key}/mse"] = vec.astype(np.float32)
        record[key] = float(vec.mean())
        record[f"{key}_recovery_pct"] = (
            100.0 * (sg - record[key]) / (sg - oracle)
            if sg - oracle > 1e-12 else np.nan)
    np.savez_compressed(prune_dir / "ensemble_losses.npz", **store)
    del model
    torch.cuda.empty_cache()
    return record


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in DATASETS_96:
        for seed in SEEDS:
            rows.append(run_cell(dataset, seed, device))
            r = rows[-1]
            print(f"{dataset} s{seed}: base={r['baseline']:.4f} "
                  f"sg={r['static_greedy']:.4f} mean4={r['mean4']:.4f} "
                  f"invmse={r['invmse']:.4f} oracle={r['oracle4']:.4f} "
                  f"rec(mean4)={r['mean4_recovery_pct']:+.0f}%", flush=True)
    frame = pd.DataFrame(rows)
    out = RUNS_DIR / "analysis" / "ensemble_eval.csv"
    frame.to_csv(out, index=False)
    agg = frame.groupby("dataset").mean(numeric_only=True)
    print("\n=== per-dataset means ===")
    print(agg[["baseline", "static_greedy", "mean4", "mean3", "top2",
               "invmse", "oracle4"]].round(4).to_string())
    for k in ("mean4", "mean3", "top2", "invmse"):
        wins_sg = int((agg[k] < agg["static_greedy"]).sum())
        wins_base = int((agg[k] < agg["baseline"]).sum())
        print(f"{k}: beats static_greedy {wins_sg}/4 datasets, "
              f"beats baseline {wins_base}/4, "
              f"mean recovery {agg[f'{k}_recovery_pct'].mean():+.1f}%")
    print("written:", out)


if __name__ == "__main__":
    main()
