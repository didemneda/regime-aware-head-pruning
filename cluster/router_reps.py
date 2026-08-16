"""Representation routing (@96 cells): route from PatchTST's own encoder
representations instead of handcrafted input features.

For each (dataset, 96, seed): one forward pass of the UNPRUNED model over
val+test extracts mean-pooled last-layer encoder representations
(d_model=128). A delta-regression router (identical to router_lab's
delta_reg) is trained on: (a) representations alone, (b) representations
+ handcrafted features. Also adds the cleaner DIAG_halftest diagnostic:
fit on the first half of TEST, evaluate on the second half -- separates
distribution shift from absence of signal (both halves in-distribution).

Writes runs/analysis/router_reps.csv.
"""
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (EVAL_BATCH, RUNS_DIR, HeadMaskController, build_model,
                    find_checkpoint, get_split_dataset)
from oracle_router import cell_features
from router_lab import KEEP, REGIMES, fit_predict_delta, load_cell

PRED_LEN = 96


def extract_reps(model, loader, device):
    captured = []

    def hook(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        captured.append(x.detach())

    handle = model.encoder.register_forward_hook(hook)
    reps = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            captured.clear()
            B = batch_x.shape[0]
            model(batch_x.float().to(device), batch_x_mark.float().to(device),
                  batch_y.float().to(device), batch_y_mark.float().to(device))
            x = captured[-1]                    # [B*nvars, patch_num, d]
            x = x.reshape(B, -1, x.shape[-2], x.shape[-1])
            reps.append(x.mean(dim=(1, 2)).cpu().numpy())
    handle.remove()
    return np.concatenate(reps)


def run_cell(dataset, seed):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    npz_path = str(RUNS_DIR / dataset / f"pred{PRED_LEN}" / f"seed_{seed}"
                   / "pruning" / "window_losses.npz")
    _, _, _, losses = load_cell(npz_path)
    L_val, L_test = losses["val"], losses["test"]

    rep_cache = (RUNS_DIR / dataset / f"pred{PRED_LEN}" / f"seed_{seed}"
                 / "pruning" / "encoder_reps.npz")
    if rep_cache.exists():
        data = np.load(rep_cache)
        R_val, R_test = data["val"], data["test"]
    else:
        checkpoint = find_checkpoint(dataset, PRED_LEN, seed)
        model, _ = build_model(dataset, PRED_LEN, checkpoint, device)
        controller = HeadMaskController(model)
        controller.install()
        controller.set_mask(None)
        loaders = {
            split: DataLoader(get_split_dataset(dataset, PRED_LEN, split),
                              batch_size=EVAL_BATCH, shuffle=False,
                              num_workers=0)
            for split in ("val", "test")}
        R_val = extract_reps(model, loaders["val"], device)
        R_test = extract_reps(model, loaders["test"], device)
        np.savez_compressed(rep_cache, val=R_val, test=R_test)
        del model
        torch.cuda.empty_cache()

    X_val = cell_features(dataset, PRED_LEN, "val", L_val.shape[1])
    X_test = cell_features(dataset, PRED_LEN, "test", L_test.shape[1])

    sg = float(L_test[3].mean())
    oracle = float(L_test.min(axis=0).mean())
    record = {"dataset": dataset, "pred_len": PRED_LEN, "seed": seed,
              "static_greedy": sg, "oracle4": oracle}

    def add(name, route_idx):
        mse = float(L_test[route_idx, np.arange(L_test.shape[1])].mean())
        record[name] = mse
        gap = sg - oracle
        record[f"{name}_recovery_pct"] = (
            100.0 * (sg - mse) / gap if gap > 1e-12 else np.nan)

    add("rep_delta", fit_predict_delta(R_val, L_val, R_test).argmin(axis=0))
    add("rep_plus_feat", fit_predict_delta(
        np.hstack([R_val, X_val]), L_val,
        np.hstack([R_test, X_test])).argmin(axis=0))
    # cleaner diagnostic: first half of test -> second half (no val shift)
    n = L_test.shape[1]
    half = n // 2
    pred = fit_predict_delta(
        np.hstack([R_test, X_test])[:half], L_test[:, :half],
        np.hstack([R_test, X_test])[half:]).argmin(axis=0)
    second_half = L_test[:, half:]
    sg_h = float(second_half[3].mean())
    or_h = float(second_half.min(axis=0).mean())
    mse_h = float(second_half[pred, np.arange(n - half)].mean())
    record["DIAG_halftest"] = mse_h
    record["DIAG_halftest_sg"] = sg_h
    record["DIAG_halftest_recovery_pct"] = (
        100.0 * (sg_h - mse_h) / (sg_h - or_h) if sg_h - or_h > 1e-12
        else np.nan)
    return record


def main():
    rows = []
    for dataset in ("ETTh1", "ETTh2", "ETTm1", "weather"):
        for seed in (7, 42, 1234, 2026, 3407):
            try:
                rows.append(run_cell(dataset, seed))
                r = rows[-1]
                print(f"{dataset} s{seed}: sg={r['static_greedy']:.4f} "
                      f"rep={r['rep_delta']:.4f} "
                      f"rep+f={r['rep_plus_feat']:.4f} "
                      f"diag_half={r['DIAG_halftest']:.4f} "
                      f"(sg_h={r['DIAG_halftest_sg']:.4f}) "
                      f"oracle={r['oracle4']:.4f}", flush=True)
            except (KeyError, FileNotFoundError, RuntimeError) as error:
                print(f"skip {dataset} s{seed}: {error}", flush=True)
    frame = pd.DataFrame(rows)
    out = RUNS_DIR / "analysis" / "router_reps.csv"
    frame.to_csv(out, index=False)
    cell = frame.groupby("dataset").mean(numeric_only=True)
    for m in ("rep_delta", "rep_plus_feat", "DIAG_halftest"):
        base = ("static_greedy" if m != "DIAG_halftest"
                else "DIAG_halftest_sg")
        wins = int((cell[m] < cell[base]).sum())
        print(f"{m:14s} wins {wins}/{len(cell)} datasets, "
              f"mean recovery {cell[f'{m}_recovery_pct'].mean():+.1f}%")
    print("written:", out)


if __name__ == "__main__":
    main()
