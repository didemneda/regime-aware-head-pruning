"""Train B4 from scratch WITH a pruning mask active (per (dataset,pred,seed)).

Two methods:
  static  -- the seed's static keep-18 mask applied to every window.
  regime  -- regime-conditional: each training window uses its STL regime's
             greedy keep-18 mask (masks fixed, selected from the pretrained
             model of the same seed). A novel "regime-conditional training"
             variant of Policy B.

Replicates TSLib's training recipe: Adam lr=1e-4, MSE loss, 10 epochs,
early stopping patience 3 on (masked) val loss, lr halved each epoch.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (EVAL_BATCH, N_HEADS, N_LAYERS, HeadMaskController,
                    build_args, cell_dir, evaluate, get_split_dataset,
                    load_labels, mask_from_removal_prefix, set_global_seed)

KEEP = 18


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return (*self.base[idx], idx)


def main(dataset, pred_len, seed, method):
    set_global_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_dir = cell_dir(dataset, pred_len) / f"seed_{seed}"
    out_dir = seed_dir / f"scratch_{method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.csv").exists():
        print(f"{dataset} pred{pred_len} seed{seed} {method}: done")
        return

    orders = json.loads(
        (seed_dir / "pruning" / "removal_orders.json").read_text())
    static_mask = mask_from_removal_prefix(
        [tuple(x) for x in orders["static"]], KEEP)
    regime_masks = {}
    for regime in ("trend", "seasonal", "residual"):
        if regime in orders["greedy"]:
            regime_masks[regime] = mask_from_removal_prefix(
                [tuple(x) for x in orders["greedy"][regime]], KEEP)
        else:
            regime_masks[regime] = static_mask

    train_data = get_split_dataset(dataset, pred_len, "train")
    val_data = get_split_dataset(dataset, pred_len, "val")
    test_data = get_split_dataset(dataset, pred_len, "test")
    val_labels = load_labels(dataset, pred_len, "val")
    test_labels = load_labels(dataset, pred_len, "test")
    train_regimes = None
    if method == "regime":
        train_labels = load_labels(dataset, pred_len, "train")
        assert len(train_labels) == len(train_data), (
            len(train_labels), len(train_data))
        train_regimes = train_labels["regime"].to_numpy()

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        IndexedDataset(train_data), batch_size=32, shuffle=True,
        num_workers=0, drop_last=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=EVAL_BATCH, shuffle=False,
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=EVAL_BATCH, shuffle=False,
                             num_workers=0, drop_last=False)

    from models.PatchTST import Model as PatchTSTModel
    args = build_args(dataset, pred_len)
    model = PatchTSTModel(args).to(device)
    controller = HeadMaskController(model)
    controller.install()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    def masked_val_mse():
        if method == "regime":
            result = evaluate(model, val_loader, val_labels, controller,
                              pred_len, device, regime_masks=regime_masks)
        else:
            result = evaluate(model, val_loader, val_labels, controller,
                              pred_len, device, global_mask=static_mask)
        return result["mse"]

    best_state, best_val, patience_left = None, float("inf"), 3
    history = []
    for epoch in range(10):
        model.train()
        epoch_losses = []
        t0 = time.time()
        for batch in train_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, idx = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            if method == "regime":
                masks = torch.stack(
                    [regime_masks[train_regimes[int(i)]] for i in idx])
                controller.set_mask(masks)
            else:
                controller.set_mask(static_mask)
            optimizer.zero_grad()
            output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            loss = criterion(output, batch_y[:, -pred_len:, :])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        val_mse = masked_val_mse()
        history.append({"epoch": epoch + 1,
                        "train_mse": float(np.mean(epoch_losses)),
                        "masked_val_mse": val_mse,
                        "seconds": time.time() - t0})
        print(f"epoch {epoch+1}: train={np.mean(epoch_losses):.4f} "
              f"val={val_mse:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val_mse < best_val - 1e-7:
            best_val, patience_left = val_mse, 3
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print("early stop", flush=True)
                break
        # TSLib lradj type1: halve lr each epoch
        for group in optimizer.param_groups:
            group["lr"] = 1e-4 * (0.5 ** (epoch + 1))

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "checkpoint.pth")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    kwargs = (dict(regime_masks=regime_masks) if method == "regime"
              else dict(global_mask=static_mask))
    final_val = evaluate(model, val_loader, val_labels, controller,
                         pred_len, device, **kwargs)
    final_test = evaluate(model, test_loader, test_labels, controller,
                          pred_len, device, **kwargs)
    np.savez_compressed(
        out_dir / "window_losses.npz",
        **{"val/mse": final_val["window_mse"],
           "val/mae": final_val["window_mae"],
           "test/mse": final_test["window_mse"],
           "test/mae": final_test["window_mae"]})
    pd.DataFrame([{
        "dataset": dataset, "pred_len": pred_len, "seed": seed,
        "method": f"scratch_{method}", "keep_heads": KEEP,
        "val_mse": final_val["mse"], "val_mae": final_val["mae"],
        "test_mse": final_test["mse"], "test_mae": final_test["mae"],
    }]).to_csv(out_dir / "summary.csv", index=False)
    print(f"scratch_{method} {dataset} pred{pred_len} seed{seed}: "
          f"val={final_val['mse']:.4f} test={final_test['mse']:.4f}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=["static", "regime"],
                        required=True)
    cli = parser.parse_args()
    main(cli.dataset, cli.pred_len, cli.seed, cli.method)
