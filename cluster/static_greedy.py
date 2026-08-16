"""Static-greedy control: greedy backward elimination on a pooled
(regime-agnostic) validation subsample, evaluated as a single static mask.

Isolates the selection criterion from regime conditioning: comparing
  dynamic_joint (greedy, per-regime)  vs  static_greedy (greedy, pooled)
holds the greedy criterion constant, so any remaining gap is attributable
to regime-awareness itself.

Appends rows (method='static_greedy') to the cell's pruning/summary.csv and
stores per-window losses in static_greedy_losses.npz. Idempotent via the
GREEDY_DONE marker.
"""
import argparse
import random as pyrandom

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from common import (EVAL_BATCH, GREEDY_SELECTION_SEED, GREEDY_SUBSAMPLE_SIZE,
                    MIN_KEEP, SWEEP_KEEPS, HeadMaskController, build_model,
                    cell_dir, evaluate, find_checkpoint, get_split_dataset,
                    greedy_removal_order, load_labels,
                    mask_from_removal_prefix, set_global_seed, write_json)


def main(dataset, pred_len, seed):
    set_global_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = cell_dir(dataset, pred_len) / f"seed_{seed}" / "pruning"
    marker = out_dir / "GREEDY_DONE"
    if marker.exists():
        print("already done")
        return
    if not (out_dir / "DONE").exists():
        raise RuntimeError("prune_eval must finish first")

    val_labels = load_labels(dataset, pred_len, "val")
    test_labels = load_labels(dataset, pred_len, "test")
    val_data = get_split_dataset(dataset, pred_len, "val")
    test_data = get_split_dataset(dataset, pred_len, "test")
    val_loader = DataLoader(val_data, batch_size=EVAL_BATCH, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_data, batch_size=EVAL_BATCH, shuffle=False,
                             num_workers=0)

    checkpoint = find_checkpoint(dataset, pred_len, seed)
    model, args = build_model(dataset, pred_len, checkpoint, device)
    controller = HeadMaskController(model)
    controller.install()

    # pooled subsample: same size and rng protocol as the per-regime greedy
    indices = list(range(len(val_data)))
    rng = pyrandom.Random(GREEDY_SELECTION_SEED)
    if len(indices) > GREEDY_SUBSAMPLE_SIZE:
        indices = sorted(rng.sample(indices, GREEDY_SUBSAMPLE_SIZE))
    loader = DataLoader(Subset(val_data, indices), batch_size=EVAL_BATCH,
                        shuffle=False, num_workers=0)
    batches = [(x.float().to(device), y.float().to(device),
                xm.float().to(device), ym.float().to(device))
               for x, y, xm, ym in loader]
    order, steps = greedy_removal_order(
        model, controller, batches, pred_len, min_keep=MIN_KEEP,
        log_prefix="pooled")
    steps.insert(0, "regime", "pooled")
    steps.to_csv(out_dir / "static_greedy_removal_steps.csv", index=False)
    write_json(out_dir / "static_greedy_order.json", {"pooled": order})

    store, rows = {}, []
    for keep in SWEEP_KEEPS:
        mask = mask_from_removal_prefix(order, keep)
        for split, loader_, labels in (("val", val_loader, val_labels),
                                       ("test", test_loader, test_labels)):
            result = evaluate(model, loader_, labels, controller, pred_len,
                              device, global_mask=mask)
            store[f"{split}/static_greedy/keep{keep}/mse"] = (
                result["window_mse"])
            store[f"{split}/static_greedy/keep{keep}/mae"] = (
                result["window_mae"])
            rows.append({
                "dataset": dataset, "pred_len": pred_len, "seed": seed,
                "split": split, "method": "static_greedy",
                "keep_heads": keep,
                "mse": result["mse"], "mae": result["mae"],
            })
        print(f"keep={keep} done", flush=True)

    np.savez_compressed(out_dir / "static_greedy_losses.npz", **store)
    summary = pd.read_csv(out_dir / "summary.csv")
    summary = summary[summary.method != "static_greedy"]
    summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    marker.write_text("ok")
    print(f"static_greedy {dataset} pred{pred_len} seed{seed} COMPLETE",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    cli = parser.parse_args()
    main(cli.dataset, cli.pred_len, cli.seed)
