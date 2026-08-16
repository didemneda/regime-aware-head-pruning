"""Learning-curve test: is the oracle-router data-limited or saturated?

Stage A (--stage losses, GPU): for each (@96 dataset, seed) cell, evaluate
the 4 candidate masks (3 regime greedy + pooled greedy, keep=18) on the
TRAIN split and store per-window losses. Test data is never touched.

Stage B (--stage curve, CPU): delta-regression router trained on
increasing fractions of the train split (10/25/50/100%) plus a final
train+val point. For each size: in-sample (train-subset) regret, held-out
val regret, test regret, test MSE, oracle-gain recovery vs pooled greedy.

Saturating val/test regret despite 4x more data => not data-limited.
Writes runs/analysis/router_curve.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import (EVAL_BATCH, RUNS_DIR, HeadMaskController, build_model,
                    cell_dir, evaluate, find_checkpoint, get_split_dataset,
                    mask_from_removal_prefix)
from oracle_router import cell_features
from router_lab import KEEP, REGIMES, fit_predict_delta, load_cell

PRED_LEN = 96
DATASETS_96 = ["ETTh1", "ETTh2", "ETTm1", "weather"]
SEEDS = [7, 42, 1234, 2026, 3407]
FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def train_loss_path(dataset, seed):
    return (cell_dir(dataset, PRED_LEN) / f"seed_{seed}" / "pruning"
            / "train_mask_losses.npz")


def stage_losses():
    import torch
    from torch.utils.data import DataLoader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for dataset in DATASETS_96:
        train_data = get_split_dataset(dataset, PRED_LEN, "train")
        loader = DataLoader(train_data, batch_size=EVAL_BATCH, shuffle=False,
                            num_workers=0)
        for seed in SEEDS:
            out = train_loss_path(dataset, seed)
            if out.exists():
                print(f"{dataset} s{seed}: cached", flush=True)
                continue
            prune_dir = out.parent
            orders = json.loads(
                (prune_dir / "removal_orders.json").read_text())
            sg_order = json.loads(
                (prune_dir / "static_greedy_order.json").read_text())
            masks = {}
            static_mask = mask_from_removal_prefix(
                [tuple(x) for x in orders["static"]], KEEP)
            for regime in REGIMES:
                if regime in orders["greedy"]:
                    masks[regime] = mask_from_removal_prefix(
                        [tuple(x) for x in orders["greedy"][regime]], KEEP)
                else:
                    masks[regime] = static_mask
            masks["static_greedy"] = mask_from_removal_prefix(
                [tuple(x) for x in sg_order["pooled"]], KEEP)
            model, _ = build_model(
                dataset, PRED_LEN, find_checkpoint(dataset, PRED_LEN, seed),
                device)
            controller = HeadMaskController(model)
            controller.install()
            store = {}
            for name in REGIMES + ["static_greedy"]:
                result = evaluate(model, loader, None, controller, PRED_LEN,
                                  device, global_mask=masks[name])
                store[name] = result["window_mse"].astype(np.float32)
                print(f"{dataset} s{seed} {name}: {result['mse']:.4f}",
                      flush=True)
            np.savez_compressed(out, **store)
            del model
            torch.cuda.empty_cache()


def regret(L, route_idx):
    n = np.arange(L.shape[1])
    return float((L[route_idx, n] - L.min(axis=0)).mean())


def stage_curve():
    rows = []
    for dataset in DATASETS_96:
        for seed in SEEDS:
            path = train_loss_path(dataset, seed)
            if not path.exists():
                print(f"missing train losses: {dataset} s{seed}")
                continue
            data = np.load(path)
            L_train = np.stack([data[r] for r in REGIMES]
                               + [data["static_greedy"]]).astype(np.float64)
            npz = str(RUNS_DIR / dataset / f"pred{PRED_LEN}"
                      / f"seed_{seed}" / "pruning" / "window_losses.npz")
            _, _, _, losses = load_cell(npz)
            L_val, L_test = losses["val"], losses["test"]
            n_train, n_val = L_train.shape[1], L_val.shape[1]
            n_test = L_test.shape[1]
            X_train = cell_features(dataset, PRED_LEN, "train", n_train)
            X_val = cell_features(dataset, PRED_LEN, "val", n_val)
            X_test = cell_features(dataset, PRED_LEN, "test", n_test)

            sg = float(L_test[3].mean())
            oracle = float(L_test.min(axis=0).mean())
            rng = np.random.RandomState(seed)
            order = rng.permutation(n_train)

            settings = [(f"train_{int(f*100)}", None, f) for f in FRACTIONS]
            settings.append(("train_plus_val", "val", 1.0))
            for name, extra, fraction in settings:
                take = order[: max(200, int(n_train * fraction))]
                X_fit = X_train[take]
                L_fit = L_train[:, take]
                if extra == "val":
                    X_fit = np.vstack([X_fit, X_val])
                    L_fit = np.hstack([L_fit, L_val])
                pred_tr = fit_predict_delta(X_fit, L_fit, X_fit)
                pred_val = fit_predict_delta(X_fit, L_fit, X_val)
                pred_te = fit_predict_delta(X_fit, L_fit, X_test)
                test_route = pred_te.argmin(axis=0)
                n = np.arange(n_test)
                test_mse = float(L_test[test_route, n].mean())
                rows.append({
                    "dataset": dataset, "seed": seed, "setting": name,
                    "n_router_train": L_fit.shape[1],
                    "train_regret": regret(L_fit, pred_tr.argmin(axis=0)),
                    "val_regret": regret(L_val, pred_val.argmin(axis=0)),
                    "test_regret": regret(L_test, test_route),
                    "sg_test_regret": regret(
                        L_test, np.full(n_test, 3, dtype=int)),
                    "test_mse": test_mse,
                    "static_greedy": sg, "oracle4": oracle,
                    "recovery_pct": (100.0 * (sg - test_mse)
                                     / (sg - oracle)
                                     if sg - oracle > 1e-12 else np.nan),
                })
                print(f"{dataset} s{seed} {name} "
                      f"(n={L_fit.shape[1]}): "
                      f"val_regret={rows[-1]['val_regret']:.5f} "
                      f"test_mse={test_mse:.4f} "
                      f"rec={rows[-1]['recovery_pct']:+.0f}%", flush=True)
    frame = pd.DataFrame(rows)
    out = RUNS_DIR / "analysis" / "router_curve.csv"
    frame.to_csv(out, index=False)
    print("\n=== learning curve (mean over cells) ===")
    agg = frame.groupby("setting").mean(numeric_only=True)
    agg = agg.reindex([f"train_{int(f*100)}" for f in FRACTIONS]
                      + ["train_plus_val"])
    print(agg[["n_router_train", "train_regret", "val_regret",
               "test_regret", "sg_test_regret", "recovery_pct"]]
          .round(5).to_string())
    print("written:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["losses", "curve"],
                        required=True)
    cli = parser.parse_args()
    if cli.stage == "losses":
        stage_losses()
    else:
        stage_curve()
