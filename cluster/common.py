"""Shared infrastructure for the ReCAHS cluster experiments.

Faithfully ports the logic of notebooks 14/15 (HeadMaskController, STL
labeling, greedy joint selection, evaluation with per-window losses) into
importable form, generalized over datasets and prediction lengths.
"""
import hashlib
import json
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

RECAHS_ROOT = Path(os.environ.get("RECAHS_ROOT", Path.home() / "recahs"))
TSLIB_DIR = RECAHS_ROOT / "tslib"
DATA_DIR = RECAHS_ROOT / "data"
# Model under study; a separate runs tree per model keeps results isolated.
MODEL_NAME = os.environ.get("RECAHS_MODEL", "PatchTST")
RUNS_DIR = RECAHS_ROOT / os.environ.get("RECAHS_RUNS_SUBDIR", "runs")
TSLIB_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"

SEEDS = [7, 42, 1234, 2026, 3407]
SEQ_LEN = 336
LABEL_LEN = 48
N_LAYERS = 3
N_HEADS = 8
TOTAL_HEADS = N_LAYERS * N_HEADS
EVAL_BATCH = 128
GREEDY_SUBSAMPLE_SIZE = 256
GREEDY_SELECTION_SEED = 42
CONFIDENCE_THRESHOLD = 0.05
# Sweep over keep counts: 24 heads total -> pruned ratio 12.5% ... 75%
SWEEP_KEEPS = [21, 18, 15, 12, 9, 6]
MIN_KEEP = min(SWEEP_KEEPS)

if str(TSLIB_DIR) not in sys.path:
    sys.path.insert(0, str(TSLIB_DIR))

# Per-dataset configuration. Border logic mirrors the TSLib data loaders so
# STL window labels stay aligned with loader windows (verified by count
# asserts at load time).
DATASETS = {
    "ETTh1": dict(
        csv="ETTh1.csv", loader="Dataset_ETT_hour", data_flag="ETTh1",
        enc_in=7, freq="h", stl_period=24,
        train_size=12 * 30 * 24, val_size=4 * 30 * 24, test_size=4 * 30 * 24,
    ),
    "ETTh2": dict(
        csv="ETTh2.csv", loader="Dataset_ETT_hour", data_flag="ETTh2",
        enc_in=7, freq="h", stl_period=24,
        train_size=12 * 30 * 24, val_size=4 * 30 * 24, test_size=4 * 30 * 24,
    ),
    "ETTm1": dict(
        csv="ETTm1.csv", loader="Dataset_ETT_minute", data_flag="ETTm1",
        enc_in=7, freq="t", stl_period=96,
        train_size=12 * 30 * 96, val_size=4 * 30 * 96, test_size=4 * 30 * 96,
    ),
    "weather": dict(
        csv="weather.csv", loader="Dataset_Custom", data_flag="custom",
        enc_in=21, freq="t", stl_period=144,
        split_ratio=(0.7, 0.2),  # train, test; val = remainder (TSLib custom)
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_global_seed(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_args(dataset, pred_len):
    cfg = DATASETS[dataset]
    return Namespace(
        task_name="long_term_forecast", data=cfg["data_flag"],
        root_path=str(DATA_DIR) + "/", data_path=cfg["csv"],
        features="M", target="OT", freq=cfg["freq"], embed="timeF",
        seasonal_patterns="Monthly",
        seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=pred_len,
        enc_in=cfg["enc_in"], dec_in=cfg["enc_in"], c_out=cfg["enc_in"],
        d_model=128, n_heads=N_HEADS, e_layers=N_LAYERS, d_layers=1,
        d_ff=256, factor=3, dropout=0.1, activation="gelu",
        augmentation_ratio=0.0, batch_size=32, num_workers=0,
    )


def loader_class(dataset):
    from data_provider.data_loader import (
        Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute)
    return {
        "Dataset_ETT_hour": Dataset_ETT_hour,
        "Dataset_ETT_minute": Dataset_ETT_minute,
        "Dataset_Custom": Dataset_Custom,
    }[DATASETS[dataset]["loader"]]


def get_split_dataset(dataset, pred_len, flag):
    args = build_args(dataset, pred_len)
    cls = loader_class(dataset)
    return cls(
        args=args, root_path=args.root_path, flag=flag,
        size=[SEQ_LEN, LABEL_LEN, pred_len], features="M",
        data_path=args.data_path, target="OT", timeenc=1, freq=args.freq,
    )


def split_borders(dataset, split, n_rows):
    """Start/end row indices of the raw series feeding `split` windows.

    Mirrors the border arithmetic in TSLib's data loaders.
    """
    cfg = DATASETS[dataset]
    if "split_ratio" in cfg:  # Dataset_Custom
        num_train = int(n_rows * cfg["split_ratio"][0])
        num_test = int(n_rows * cfg["split_ratio"][1])
        num_val = n_rows - num_train - num_test
        border1s = [0, num_train - SEQ_LEN, n_rows - num_test - SEQ_LEN]
        border2s = [num_train, num_train + num_val, n_rows]
    else:
        tr, va, te = cfg["train_size"], cfg["val_size"], cfg["test_size"]
        border1s = [0, tr - SEQ_LEN, tr + va - SEQ_LEN]
        border2s = [tr, tr + va, tr + va + te]
    idx = {"train": 0, "val": 1, "test": 2}[split]
    return border1s[idx], border2s[idx]


def stl_label_window(values, period):
    from statsmodels.tsa.seasonal import STL
    fit = STL(values, period=period, robust=True).fit()
    variances = np.array(
        [np.var(fit.trend), np.var(fit.seasonal), np.var(fit.resid)],
        dtype=float,
    )
    scores = (variances / variances.sum()
              if variances.sum() > 1e-12 else np.zeros(3))
    names = ["trend", "seasonal", "residual"]
    order = np.argsort(scores)[::-1]
    return {
        "regime": names[int(order[0])],
        "trend_score": float(scores[0]),
        "seasonal_score": float(scores[1]),
        "residual_score": float(scores[2]),
        "confidence_margin": float(scores[order[0]] - scores[order[1]]),
    }


def labels_path(dataset, pred_len, split):
    directory = RUNS_DIR / dataset / f"pred{pred_len}" / "regime_labels"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{split}_regimes.csv"


def load_labels(dataset, pred_len, split):
    path = labels_path(dataset, pred_len, split)
    labels = pd.read_csv(path)
    labels["is_confident"] = (
        labels["confidence_margin"] >= CONFIDENCE_THRESHOLD)
    return labels


class HeadMaskController:
    """Zeroes selected attention heads at the attention output.

    Mask shapes: (n_layers, n_heads) global, or
    (batch, n_layers, n_heads) per-sample.
    """

    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        self.current_mask = None

    def install(self):
        for layer_idx, encoder_layer in enumerate(
                self.model.encoder.attn_layers):
            attention_layer = encoder_layer.attention
            self.original_forwards[layer_idx] = attention_layer.forward

            def make_forward(layer_idx, attention_layer):
                def masked_forward(queries, keys, values, attn_mask,
                                   tau=None, delta=None):
                    B, L, _ = queries.shape
                    _, S, _ = keys.shape
                    H = attention_layer.n_heads
                    q = attention_layer.query_projection(queries).view(
                        B, L, H, -1)
                    k = attention_layer.key_projection(keys).view(B, S, H, -1)
                    v = attention_layer.value_projection(values).view(
                        B, S, H, -1)
                    out, attn = attention_layer.inner_attention(
                        q, k, v, attn_mask, tau=tau, delta=delta)
                    if self.current_mask is not None:
                        mask = self.current_mask.to(out.device)
                        if mask.ndim == 2:
                            layer_mask = mask[layer_idx].view(1, 1, H, 1)
                        else:
                            if mask.shape[0] != B:
                                if B % mask.shape[0] != 0:
                                    raise ValueError((mask.shape, B))
                                mask = mask.repeat_interleave(
                                    B // mask.shape[0], dim=0)
                            layer_mask = mask[:, layer_idx, :].view(B, 1, H, 1)
                        out = out * layer_mask
                    return attention_layer.out_projection(
                        out.reshape(B, L, -1)), attn

                return masked_forward

            attention_layer.forward = make_forward(layer_idx, attention_layer)

    def set_mask(self, mask):
        self.current_mask = (
            None if mask is None else mask.detach().clone().float())


def mask_from_active(active):
    mask = torch.zeros(N_LAYERS, N_HEADS)
    for layer, head in active:
        mask[layer, head] = 1
    return mask


def mask_from_removal_prefix(order, keep):
    """Mask keeping `keep` heads: all heads minus first (24-keep) removals."""
    removed = order[: TOTAL_HEADS - keep]
    active = {(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)}
    active -= {tuple(x) for x in removed}
    assert len(active) == keep
    return mask_from_active(active)


_mse_none = nn.MSELoss(reduction="none")
_mae_none = nn.L1Loss(reduction="none")


def evaluate(model, loader, labels, controller, pred_len, device,
             global_mask=None, regime_masks=None):
    """Per-window MSE/MAE under a global or regime-conditional mask."""
    model.eval()
    mse_parts, mae_parts = [], []
    offset = 0
    regimes = labels["regime"].to_numpy() if labels is not None else None
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            B = batch_x.shape[0]
            if regime_masks is None:
                controller.set_mask(global_mask)
            else:
                batch_masks = [
                    regime_masks[regimes[offset + i]] for i in range(B)]
                controller.set_mask(torch.stack(batch_masks))
            output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            true = batch_y[:, -pred_len:, :]
            mse_parts.append(
                _mse_none(output, true).mean(dim=(1, 2)).cpu().numpy())
            mae_parts.append(
                _mae_none(output, true).mean(dim=(1, 2)).cpu().numpy())
            offset += B
    mse = np.concatenate(mse_parts)
    mae = np.concatenate(mae_parts)
    return {"mse": float(mse.mean()), "mae": float(mae.mean()),
            "window_mse": mse, "window_mae": mae}


def eval_batches(model, controller, mask, batches, pred_len):
    controller.set_mask(mask)
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y, xm, ym in batches:
            pred = model(x, xm, y, ym)
            true = y[:, -pred_len:, :]
            values = _mse_none(pred, true).mean(dim=(1, 2))
            total += values.sum().item()
            count += len(values)
    return total / count


def greedy_removal_order(model, controller, batches, pred_len,
                         min_keep=MIN_KEEP, log_prefix=""):
    """Greedy backward elimination from 24 heads down to `min_keep`.

    Returns the removal order (list of (layer, head)) and per-step MSE, so a
    mask at any keep-count >= min_keep is a prefix of this order.
    """
    active = {(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)}
    order, rows = [], []
    while len(active) > min_keep:
        best_head, best_mse = None, None
        for candidate in sorted(active):
            trial = mask_from_active(active - {candidate})
            score = eval_batches(model, controller, trial, batches, pred_len)
            if (best_mse is None or score < best_mse - 1e-12
                    or (abs(score - best_mse) <= 1e-12
                        and candidate < best_head)):
                best_head, best_mse = candidate, score
        active.remove(best_head)
        order.append(best_head)
        rows.append({
            "step": len(order), "layer": best_head[0], "head": best_head[1],
            "resulting_mse": best_mse, "remaining_heads": len(active),
        })
        if log_prefix:
            print(f"  {log_prefix} step {len(order)}: remove {best_head} "
                  f"mse={best_mse:.6f}", flush=True)
    return order, pd.DataFrame(rows)


def magnitude_removal_order(model):
    """Heads ordered by ascending L2 norm of their projection weights."""
    head_dim = None
    norms = []
    for layer_idx, encoder_layer in enumerate(model.encoder.attn_layers):
        att = encoder_layer.attention
        d_model = att.query_projection.weight.shape[1]
        head_dim = d_model // N_HEADS
        for head in range(N_HEADS):
            rows = slice(head * head_dim, (head + 1) * head_dim)
            total = 0.0
            for proj in (att.query_projection, att.key_projection,
                         att.value_projection):
                total += float(proj.weight[rows, :].norm() ** 2)
                if proj.bias is not None:
                    total += float(proj.bias[rows].norm() ** 2)
            total += float(att.out_projection.weight[:, rows].norm() ** 2)
            norms.append(((layer_idx, head), total ** 0.5))
    norms.sort(key=lambda item: (item[1], item[0]))
    return [head for head, _ in norms]


def build_model(dataset, pred_len, checkpoint, device):
    import importlib
    model_cls = importlib.import_module(f"models.{MODEL_NAME}").Model
    args = build_args(dataset, pred_len)
    model = model_cls(args).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, args


def cell_dir(dataset, pred_len):
    directory = RUNS_DIR / dataset / f"pred{pred_len}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_checkpoint(dataset, pred_len, seed):
    root = cell_dir(dataset, pred_len) / f"seed_{seed}" / "checkpoints"
    candidates = list(root.glob("**/checkpoint.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"{dataset}/pred{pred_len}/seed{seed}: "
            f"{len(candidates)} checkpoints found")
    return candidates[0]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
