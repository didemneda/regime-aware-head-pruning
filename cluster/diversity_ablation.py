"""Diversity-source ablation: is REGIME-based mask diversity special?

For each (@96 dataset, seed) cell, build an alternative specialist pool of
the same size and selection criterion but WITHOUT regime information:
3 greedy masks selected on random 256-window validation subsamples
(regime-blind) + the pooled static-greedy mask. Evaluate the same ensemble
schemes over this pool and compare against the regime pool's ensembles.

If random-subsample pools ensemble as well as regime pools, regime
information is not needed even offline; if they don't, regime labels earn
their place as the diversity source. Writes
runs/analysis/diversity_ablation.csv.
"""
import argparse
import random as pyrandom

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from common import (EVAL_BATCH, GREEDY_SUBSAMPLE_SIZE, RUNS_DIR,
                    HeadMaskController, build_model, cell_dir, evaluate,
                    find_checkpoint, get_split_dataset,
                    greedy_removal_order, load_labels)
from ensemble_eval import build_masks
from router_lab import KEEP, load_cell

PRED_LEN = 96
SEEDS = [7, 42, 1234, 2026, 3407]
N_RANDOM_MASKS = 3


def random_subsample_masks(model, controller, val_data, seed, device):
    masks = []
    for draw in range(N_RANDOM_MASKS):
        rng = pyrandom.Random(9000 + 100 * seed + draw)
        indices = sorted(rng.sample(range(len(val_data)),
                                    min(GREEDY_SUBSAMPLE_SIZE,
                                        len(val_data))))
        loader = DataLoader(Subset(val_data, indices),
                            batch_size=EVAL_BATCH, shuffle=False,
                            num_workers=0)
        batches = [(x.float().to(device), y.float().to(device),
                    xm.float().to(device), ym.float().to(device))
                   for x, y, xm, ym in loader]
        order, _ = greedy_removal_order(
            model, controller, batches, PRED_LEN, min_keep=KEEP)
        from common import mask_from_removal_prefix
        masks.append(mask_from_removal_prefix(order, KEEP))
    return masks


def run_cell(dataset, seed, device):
    prune_dir = cell_dir(dataset, PRED_LEN) / f"seed_{seed}" / "pruning"
    npz = str(prune_dir / "window_losses.npz")
    _, _, _, losses = load_cell(npz)
    L_val_regime, L_test_regime = losses["val"], losses["test"]

    model, _ = build_model(
        dataset, PRED_LEN, find_checkpoint(dataset, PRED_LEN, seed), device)
    controller = HeadMaskController(model)
    controller.install()
    val_data = get_split_dataset(dataset, PRED_LEN, "val")
    test_data = get_split_dataset(dataset, PRED_LEN, "test")
    val_loader = DataLoader(val_data, batch_size=EVAL_BATCH, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_data, batch_size=EVAL_BATCH,
                             shuffle=False, num_workers=0)
    val_labels = load_labels(dataset, PRED_LEN, "val")
    test_labels = load_labels(dataset, PRED_LEN, "test")

    rand_masks = random_subsample_masks(
        model, controller, val_data, seed, device)
    sg_mask = build_masks(prune_dir)["static_greedy"]
    pool = rand_masks + [sg_mask]

    # per-mask val losses (for weights) and test predictions ensembling
    val_means = []
    for mask in pool:
        result = evaluate(model, val_loader, val_labels, controller,
                          PRED_LEN, device, global_mask=mask)
        val_means.append(result["mse"])
    val_means = np.array(val_means)
    inv = 1.0 / val_means
    schemes = {"rnd_mean4": np.ones(4) / 4.0, "rnd_invmse": inv / inv.sum()}
    top2 = np.argsort(val_means)[:2]
    w = np.zeros(4)
    w[top2] = 0.5
    schemes["rnd_top2"] = w

    per_scheme = {k: [] for k in schemes}
    per_mask = [[] for _ in pool]
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            true = batch_y[:, -PRED_LEN:, :]
            preds = []
            for i, mask in enumerate(pool):
                controller.set_mask(mask)
                out = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                preds.append(out)
                per_mask[i].append(
                    ((out - true) ** 2).mean(dim=(1, 2)).cpu().numpy())
            stack = torch.stack(preds)
            for key, weight in schemes.items():
                wt = torch.tensor(weight, dtype=stack.dtype,
                                  device=stack.device).view(4, 1, 1, 1)
                combo = (stack * wt).sum(dim=0)
                per_scheme[key].append(
                    ((combo - true) ** 2).mean(dim=(1, 2)).cpu().numpy())

    mask_losses = np.stack([np.concatenate(v) for v in per_mask])
    sg = float(L_test_regime[3].mean())
    oracle_regime = float(L_test_regime.min(axis=0).mean())
    record = {
        "dataset": dataset, "seed": seed, "static_greedy": sg,
        "oracle_regime_pool": oracle_regime,
        "oracle_random_pool": float(mask_losses.min(axis=0).mean()),
    }
    for key in schemes:
        vec = np.concatenate(per_scheme[key])
        record[key] = float(vec.mean())
        gap = sg - oracle_regime
        record[f"{key}_recovery_pct"] = (
            100.0 * (sg - record[key]) / gap if gap > 1e-12 else np.nan)
    del model
    torch.cuda.empty_cache()
    return record


def main(datasets):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in datasets:
        for seed in SEEDS:
            rows.append(run_cell(dataset, seed, device))
            r = rows[-1]
            print(f"{dataset} s{seed}: sg={r['static_greedy']:.4f} "
                  f"rnd_invmse={r['rnd_invmse']:.4f} "
                  f"rnd_top2={r['rnd_top2']:.4f} "
                  f"orc_rnd={r['oracle_random_pool']:.4f} "
                  f"orc_reg={r['oracle_regime_pool']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    tag = "_".join(datasets)
    out = RUNS_DIR / "analysis" / f"diversity_ablation_{tag}.csv"
    frame.to_csv(out, index=False)
    print("written:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    cli = parser.parse_args()
    main(cli.datasets)
