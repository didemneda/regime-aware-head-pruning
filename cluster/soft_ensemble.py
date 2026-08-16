"""Dynamic regime-conditioned SOFT ensemble (@96 cells, keep=18).

Instead of hard-routing to the detected regime's mask, combine all four
masked predictions with regime-dependent weights:

  stl_aXX    -- the STL-detected regime's matching mask gets alpha=0.XX,
                the other three masks share (1-alpha) equally.
                alpha=1.0 reproduces hard STL routing (sanity check).
  reg_invmse -- data-driven variant: for a window labeled regime r, mask m
                gets weight proportional to 1 / (mask m's mean VAL MSE on
                regime-r validation windows). Val-derived only.

Per-window test MSE computed on the fly; compared against pooled
static-greedy, hard routing, global ensembles (from ensemble_eval), the
unpruned baseline and the oracle. Writes runs/analysis/soft_ensemble.csv.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (EVAL_BATCH, RUNS_DIR, HeadMaskController, build_model,
                    cell_dir, find_checkpoint, get_split_dataset,
                    load_labels)
from ensemble_eval import MASK_NAMES, build_masks
from router_lab import REGIMES, load_cell

PRED_LEN = 96
DATASETS_96 = ["ETTh1", "ETTh2", "ETTm1", "weather"]
SEEDS = [7, 42, 1234, 2026, 3407]
ALPHAS = [0.4, 0.6, 0.8, 1.0]


def scheme_weights(dataset, seed, L_val):
    """Per-regime weight vectors for each scheme: {scheme: {regime: w(4)}}."""
    val_labels = load_labels(dataset, PRED_LEN, "val")
    val_regimes = val_labels["regime"].to_numpy()
    schemes = {}
    for alpha in ALPHAS:
        weights = {}
        for r_idx, regime in enumerate(REGIMES):
            w = np.full(4, (1.0 - alpha) / 3.0)
            w[r_idx] = alpha
            weights[regime] = w
        schemes[f"stl_a{int(alpha*100)}"] = weights
    weights = {}
    for regime in REGIMES:
        sel = val_regimes == regime
        if sel.sum() >= 20:
            means = L_val[:, sel].mean(axis=1)
        else:
            means = L_val.mean(axis=1)
        inv = 1.0 / means
        weights[regime] = inv / inv.sum()
    schemes["reg_invmse"] = weights
    return schemes


def run_cell(dataset, seed, device):
    prune_dir = cell_dir(dataset, PRED_LEN) / f"seed_{seed}" / "pruning"
    npz = str(prune_dir / "window_losses.npz")
    _, _, _, losses = load_cell(npz)
    L_val, L_test = losses["val"], losses["test"]
    test_regimes = load_labels(dataset, PRED_LEN, "test")[
        "regime"].to_numpy()
    schemes = scheme_weights(dataset, seed, L_val)

    masks = build_masks(prune_dir)
    model, _ = build_model(
        dataset, PRED_LEN, find_checkpoint(dataset, PRED_LEN, seed), device)
    controller = HeadMaskController(model)
    controller.install()
    loader = DataLoader(get_split_dataset(dataset, PRED_LEN, "test"),
                        batch_size=EVAL_BATCH, shuffle=False, num_workers=0)

    per_scheme = {k: [] for k in schemes}
    offset = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            true = batch_y[:, -PRED_LEN:, :]
            B = batch_x.shape[0]
            preds = []
            for name in MASK_NAMES:
                controller.set_mask(masks[name])
                preds.append(model(batch_x, batch_x_mark, batch_y,
                                   batch_y_mark))
            stack = torch.stack(preds)            # (4, B, pred, C)
            batch_regimes = test_regimes[offset:offset + B]
            for key, weights in schemes.items():
                w = np.stack([weights[r] for r in batch_regimes])  # (B, 4)
                wt = torch.tensor(w.T, dtype=stack.dtype,
                                  device=stack.device).view(4, B, 1, 1)
                combo = (stack * wt).sum(dim=0)
                per_scheme[key].append(
                    ((combo - true) ** 2).mean(dim=(1, 2)).cpu().numpy())
            offset += B

    z = np.load(npz)
    sg = float(L_test[3].mean())
    oracle = float(L_test.min(axis=0).mean())
    record = {"dataset": dataset, "seed": seed,
              "baseline": float(z["test/baseline/keep24/mse"].mean()),
              "static_greedy": sg, "oracle4": oracle}
    # hard routing reference from stored losses
    idx = {r: i for i, r in enumerate(REGIMES)}
    hard = L_test[[idx[r] for r in test_regimes],
                  np.arange(L_test.shape[1])]
    record["hard_stl"] = float(hard.mean())
    store = {}
    for key in schemes:
        vec = np.concatenate(per_scheme[key])
        store[f"test/{key}/mse"] = vec.astype(np.float32)
        record[key] = float(vec.mean())
        record[f"{key}_recovery_pct"] = (
            100.0 * (sg - record[key]) / (sg - oracle)
            if sg - oracle > 1e-12 else np.nan)
    drift = abs(record["stl_a100"] - record["hard_stl"])
    if drift > 1e-4:
        raise RuntimeError(f"alpha=1 sanity check failed: {drift}")
    np.savez_compressed(prune_dir / "soft_ensemble_losses.npz", **store)
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
            print(f"{dataset} s{seed}: sg={r['static_greedy']:.4f} "
                  f"hard={r['hard_stl']:.4f} a60={r['stl_a60']:.4f} "
                  f"a40={r['stl_a40']:.4f} "
                  f"reg_invmse={r['reg_invmse']:.4f} "
                  f"oracle={r['oracle4']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    # merge in the global (regime-blind) ensembles for comparison
    prior = pd.read_csv(RUNS_DIR / "analysis" / "ensemble_eval.csv")[
        ["dataset", "seed", "mean4", "top2", "invmse"]]
    frame = frame.merge(prior, on=["dataset", "seed"])
    out = RUNS_DIR / "analysis" / "soft_ensemble.csv"
    frame.to_csv(out, index=False)
    agg = frame.groupby("dataset").mean(numeric_only=True)
    cols = (["baseline", "static_greedy", "hard_stl"]
            + [f"stl_a{int(a*100)}" for a in ALPHAS[:-1]]
            + ["reg_invmse", "top2", "invmse", "oracle4"])
    print("\n=== per-dataset means ===")
    print(agg[cols].round(4).to_string())
    print()
    for m in ([f"stl_a{int(a*100)}" for a in ALPHAS[:-1]]
              + ["reg_invmse"]):
        wins_sg = int((agg[m] < agg["static_greedy"]).sum())
        wins_inv = int((agg[m] < agg["invmse"]).sum())
        wins_base = int((agg[m] < agg["baseline"]).sum())
        print(f"{m}: beats static_greedy {wins_sg}/4, "
              f"beats global invmse {wins_inv}/4, "
              f"beats unpruned {wins_base}/4, "
              f"recovery {agg[f'{m}_recovery_pct'].mean():+.1f}%")
    print("written:", out)


if __name__ == "__main__":
    main()
