"""Head importance, mask selection, and sparsity sweep for one
(dataset, pred_len, seed) cell.

Produces per-window loss vectors for every evaluated configuration so all
statistics (bootstrap, sign tests, oracle, detector post-processing) can be
computed offline without re-running the model.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from common import (EVAL_BATCH, GREEDY_SELECTION_SEED, GREEDY_SUBSAMPLE_SIZE,
                    MIN_KEEP, N_HEADS, N_LAYERS, SWEEP_KEEPS, TOTAL_HEADS,
                    HeadMaskController, build_model, cell_dir, evaluate,
                    find_checkpoint, get_split_dataset, greedy_removal_order,
                    load_labels, magnitude_removal_order, mask_from_active,
                    mask_from_removal_prefix, set_global_seed, write_json)

REGIMES = ["trend", "seasonal", "residual"]
N_RANDOM_DRAWS = 3


def main(dataset, pred_len, seed):
    t0 = time.time()
    set_global_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_dir = cell_dir(dataset, pred_len) / f"seed_{seed}"
    out_dir = seed_dir / "pruning"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "DONE"
    if done_marker.exists():
        print(f"{dataset} pred{pred_len} seed{seed}: already done")
        return

    val_labels = load_labels(dataset, pred_len, "val")
    test_labels = load_labels(dataset, pred_len, "test")
    val_data = get_split_dataset(dataset, pred_len, "val")
    test_data = get_split_dataset(dataset, pred_len, "test")
    assert len(val_data) == len(val_labels), (len(val_data), len(val_labels))
    assert len(test_data) == len(test_labels)
    val_loader = DataLoader(val_data, batch_size=EVAL_BATCH, shuffle=False,
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=EVAL_BATCH, shuffle=False,
                             num_workers=0, drop_last=False)

    checkpoint = find_checkpoint(dataset, pred_len, seed)
    model, args = build_model(dataset, pred_len, checkpoint, device)
    controller = HeadMaskController(model)
    controller.install()
    all_heads = torch.ones(N_LAYERS, N_HEADS)

    store = {}          # key -> per-window loss vector
    summary_rows = []

    def record(split, method, keep, result):
        store[f"{split}/{method}/keep{keep}/mse"] = result["window_mse"]
        store[f"{split}/{method}/keep{keep}/mae"] = result["window_mae"]
        summary_rows.append({
            "dataset": dataset, "pred_len": pred_len, "seed": seed,
            "split": split, "method": method, "keep_heads": keep,
            "mse": result["mse"], "mae": result["mae"],
        })

    def run_eval(global_mask=None, regime_masks=None, split="val"):
        loader = val_loader if split == "val" else test_loader
        labels = val_labels if split == "val" else test_labels
        return evaluate(model, loader, labels, controller, pred_len, device,
                        global_mask=global_mask, regime_masks=regime_masks)

    # ---- baseline -------------------------------------------------------
    base_val = run_eval(global_mask=all_heads, split="val")
    base_test = run_eval(global_mask=all_heads, split="test")
    record("val", "baseline", TOTAL_HEADS, base_val)
    record("test", "baseline", TOTAL_HEADS, base_test)
    meta = json.loads((seed_dir / "run_metadata.json").read_text())
    drift = abs(base_test["mse"] - meta["test_mse"])
    if drift > 1e-4:
        raise RuntimeError(
            f"baseline mismatch: eval={base_test['mse']:.6f} "
            f"train={meta['test_mse']:.6f}")
    print(f"baseline ok (drift {drift:.2e}) val={base_val['mse']:.4f} "
          f"test={base_test['mse']:.4f}", flush=True)

    # ---- leave-one-out head importance (per-window, val) ----------------
    n_val = len(val_labels)
    loo_matrix = np.zeros((TOTAL_HEADS, n_val), dtype=np.float32)
    importance_rows = []
    val_regimes = val_labels["regime"].to_numpy()
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            mask = all_heads.clone()
            mask[layer, head] = 0
            result = run_eval(global_mask=mask, split="val")
            flat = layer * N_HEADS + head
            loo_matrix[flat] = result["window_mse"]
            row = {"layer": layer, "head": head,
                   "masked_val_mse": result["mse"],
                   "importance": result["mse"] - base_val["mse"]}
            for regime in REGIMES:
                sel = val_regimes == regime
                if sel.any():
                    row[f"{regime}_importance"] = float(
                        result["window_mse"][sel].mean()
                        - base_val["window_mse"][sel].mean())
                else:
                    row[f"{regime}_importance"] = np.nan
            importance_rows.append(row)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(out_dir / "head_importance.csv", index=False)
    np.savez_compressed(out_dir / "loo_val_window_mse.npz",
                        loo=loo_matrix,
                        baseline=base_val["window_mse"].astype(np.float32))
    print(f"importance done ({time.time()-t0:.0f}s)", flush=True)

    # ---- removal orders -------------------------------------------------
    static_order = [
        (int(r.layer), int(r.head))
        for r in importance.sort_values(
            ["importance", "layer", "head"]).itertuples()]
    magnitude_order = magnitude_removal_order(model)
    rng_orders = {}
    for draw in range(N_RANDOM_DRAWS):
        rng = np.random.RandomState(1000 * seed + draw)
        flat = rng.permutation(TOTAL_HEADS)
        rng_orders[draw] = [(int(f) // N_HEADS, int(f) % N_HEADS)
                            for f in flat]

    # greedy joint per regime on a fixed validation subsample
    import random as pyrandom
    greedy_orders, greedy_steps = {}, []
    for regime in REGIMES:
        indices = val_labels.index[val_labels.regime == regime].tolist()
        if not indices:
            greedy_orders[regime] = None
            continue
        rng = pyrandom.Random(GREEDY_SELECTION_SEED)
        if len(indices) > GREEDY_SUBSAMPLE_SIZE:
            indices = sorted(rng.sample(indices, GREEDY_SUBSAMPLE_SIZE))
        loader = DataLoader(Subset(val_data, indices), batch_size=EVAL_BATCH,
                            shuffle=False, num_workers=0)
        batches = [(x.float().to(device), y.float().to(device),
                    xm.float().to(device), ym.float().to(device))
                   for x, y, xm, ym in loader]
        order, steps = greedy_removal_order(
            model, controller, batches, pred_len,
            min_keep=MIN_KEEP, log_prefix=f"{regime}")
        greedy_orders[regime] = order
        steps.insert(0, "regime", regime)
        greedy_steps.append(steps)
    pd.concat(greedy_steps, ignore_index=True).to_csv(
        out_dir / "greedy_removal_steps.csv", index=False)
    write_json(out_dir / "removal_orders.json", {
        "static": static_order, "magnitude": magnitude_order,
        "random": {str(k): v for k, v in rng_orders.items()},
        "greedy": {k: v for k, v in greedy_orders.items() if v is not None},
    })
    print(f"orders done ({time.time()-t0:.0f}s)", flush=True)

    # ---- sparsity sweep -------------------------------------------------
    for keep in SWEEP_KEEPS:
        configs = {
            "static": dict(global_mask=mask_from_removal_prefix(
                static_order, keep)),
            "magnitude": dict(global_mask=mask_from_removal_prefix(
                magnitude_order, keep)),
        }
        for draw, order in rng_orders.items():
            configs[f"random{draw}"] = dict(
                global_mask=mask_from_removal_prefix(order, keep))
        regime_masks = {}
        for regime in REGIMES:
            order = greedy_orders[regime]
            regime_masks[regime] = (
                mask_from_removal_prefix(order, keep) if order is not None
                else mask_from_removal_prefix(static_order, keep))
        configs["dynamic_joint"] = dict(regime_masks=regime_masks)
        for method, kwargs in configs.items():
            for split in ("val", "test"):
                record(split, method, keep,
                       run_eval(split=split, **kwargs))
        print(f"sweep keep={keep} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-regime-mask cross evaluation at keep=18 (oracle/detector) --
    for regime in REGIMES:
        order = greedy_orders[regime]
        mask = (mask_from_removal_prefix(order, 18) if order is not None
                else mask_from_removal_prefix(static_order, 18))
        for split in ("val", "test"):
            record(split, f"regimemask_{regime}", 18,
                   run_eval(global_mask=mask, split=split))

    np.savez_compressed(out_dir / "window_losses.npz", **store)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    done_marker.write_text("ok")
    print(f"{dataset} pred{pred_len} seed{seed} COMPLETE "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    cli = parser.parse_args()
    main(cli.dataset, cli.pred_len, cli.seed)
