"""Train the teaching spectral Transformer on Indian Pines (notebooks/07 protocol).

Protocol (kept in sync with notebooks/07):
    - 10% train / 10% val / 80% test, stratified two-step split, seed 42 (protocol C)
    - raw 200 bands, StandardScaler fitted on the training split only
    - NO patch: each of the 200 bands is one token (scalar -> d_model via Linear),
      learned positional embedding, TransformerEncoder(2 layers, 4 heads),
      mean pooling over tokens, FC head
    - Adam(lr=1e-3, weight_decay=1e-4), CE loss, 10 epochs, batch 256

Artifacts land in results/transformer/<dataset>/ (same layout as train_1d_cnn.py)
plus best_model.pth for the attention visualisation.

Usage:
    python scripts/train_transformer.py
    python scripts/train_transformer.py --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, recall_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SpectralTransformerClassifier(nn.Module):
    """Teaching spectral Transformer — kept in sync with notebooks/07."""

    def __init__(self, num_bands: int, num_classes: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.band_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_bands, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)          # (N, 200) -> (N, 200, 1): one token per band
        x = self.band_proj(x)        # (N, 200, d_model)
        x = x + self.pos_embed       # band order/position injected here
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)            # mean pooling over band tokens
        return self.classifier(x)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the teaching spectral Transformer.")
    parser.add_argument("--dataset", default="IP", choices=["IP", "SA", "PU"])
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--train-rate", type=float, default=0.1)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results/transformer")
    return parser


def spectra_from_map(cube: np.ndarray, label_map: np.ndarray):
    rows, cols = np.nonzero(label_map)
    X = cube[rows, cols].astype(np.float32)
    y = (label_map[rows, cols] - 1).astype(np.int64)
    return X, y


def layer_param_table(model: nn.Module, in_shape: tuple[int, ...]):
    rows = []

    def hook(name):
        def fn(mod, inp, out):
            rows.append(
                {
                    "layer": name,
                    "type": mod.__class__.__name__,
                    "out_shape": list(out.shape) if torch.is_tensor(out) else str(type(out)),
                    "params": sum(p.numel() for p in mod.parameters()),
                }
            )
        return fn

    handles = [
        mod.register_forward_hook(hook(name))
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Linear, nn.LayerNorm, nn.TransformerEncoderLayer,
                            nn.TransformerEncoder, nn.Dropout, nn.GELU))
    ]
    with torch.no_grad():
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return rows


def count_macs(model: nn.Module, in_shape: tuple[int, ...]) -> int:
    """Analytic MACs: Linear layers + QKV/attention products."""
    total = {"v": 0}

    def linear_hook(mod, inp, out):
        total["v"] += out.numel() * mod.in_features

    handles = [m.register_forward_hook(linear_hook)
               for m in model.modules() if isinstance(m, nn.Linear)]
    # attention matmuls: per layer per sample, QK^T and attn@V over heads
    n_tokens, d_model, heads, layers = in_shape[1], 64, 4, 2
    total["v"] += layers * n_tokens * n_tokens * (d_model // heads) * heads * 2
    with torch.no_grad():
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return total["v"]


def main() -> int:
    from hsi_learning.data import build_sample_report, load_hsi_dataset, split_ground_truth
    from hsi_learning.engine import TrainingConfig, fit
    from hsi_learning.utils import ensure_dir, resolve_device, save_json, set_seed

    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / args.dataset)

    cube, gt, class_names = load_hsi_dataset(args.dataset, dataset_dir=ROOT / args.dataset_dir)
    train_gt, val_gt, test_gt = split_ground_truth(
        gt, train_rate=args.train_rate, val_rate=args.val_rate, random_state=args.seed
    )
    print(build_sample_report(gt, train_gt, val_gt, test_gt, class_names=class_names))

    X_train, y_train = spectra_from_map(cube, train_gt)
    X_val, y_val = spectra_from_map(cube, val_gt)
    X_test, y_test = spectra_from_map(cube, test_gt)
    X_all = cube.reshape(-1, cube.shape[2]).astype(np.float32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_all_s = scaler.transform(X_all).astype(np.float32)

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers)

    train_loader = to_loader(X_train_s, y_train, shuffle=True)
    val_loader = to_loader(X_val_s, y_val, shuffle=False)
    test_loader = to_loader(X_test_s, y_test, shuffle=False)

    model = SpectralTransformerClassifier(num_bands=X_train.shape[1],
                                          num_classes=len(class_names)).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    fit_result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_interval=1,
            output_dir=output_dir,
        ),
    )

    model.load_state_dict(torch.load(fit_result.best_checkpoint, map_location=device))
    best_epoch = int(np.argmax(fit_result.history["val_acc"])) + 1

    y_true = np.concatenate([y.numpy() for _, y in test_loader])
    y_pred = np.concatenate([
        model(x.to(device)).argmax(dim=1).cpu().numpy()
        for x, _ in test_loader
    ])
    labels = list(range(1, len(class_names) + 1))
    oa = accuracy_score(y_true + 1, y_pred + 1)
    aa = recall_score(y_true + 1, y_pred + 1, labels=labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true + 1, y_pred + 1, labels=labels)
    report = classification_report(y_true + 1, y_pred + 1, labels=labels,
                                   target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true + 1, y_pred + 1, labels=labels)

    preds = []
    with torch.no_grad():
        for start in range(0, len(X_all_s), 1024):
            batch = torch.from_numpy(X_all_s[start:start + 1024]).to(device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy() + 1)
    pred_map = np.concatenate(preds).reshape(gt.shape)
    np.save(output_dir / "prediction_map.npy", pred_map)
    np.save(output_dir / "confusion_matrix.npy", cm)
    torch.save(model.state_dict(), output_dir / "best_model.pth")

    (output_dir / "classification_report.txt").write_text(
        f"OA: {oa:.6f}\nAA: {aa:.6f}\nKappa: {kappa:.6f}\n\n{report}", encoding="utf-8"
    )
    save_json(
        {
            "dataset": args.dataset,
            "protocol": {
                "train_rate": args.train_rate,
                "val_rate": args.val_rate,
                "test_rate": round(1 - args.train_rate - args.val_rate, 4),
                "random_state": args.seed,
                "split": "stratified two-step (split_ground_truth)",
                "preprocessing": "StandardScaler fit on the training split only",
                "input": f"(batch, {X_train.shape[1]}) raw standardized bands, "
                         "each band one token",
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "device": str(device),
            },
            "model": {
                "name": "SpectralTransformerClassifier",
                "total_params": total_params,
                "macs_per_sample": count_macs(model, (1, X_train.shape[1])),
                "layers": layer_param_table(model, (1, X_train.shape[1])),
            },
            "split_counts": {
                "train": int(len(y_train)),
                "val": int(len(y_val)),
                "test": int(len(y_test)),
            },
        },
        output_dir / "run_config.json",
    )
    save_json(
        {
            "oa": oa, "aa": aa, "kappa": kappa,
            "best_val_acc": fit_result.best_val_acc, "best_epoch": best_epoch,
            "total_params": total_params,
        },
        output_dir / "metrics.json",
    )

    print(f"\nbest checkpoint: {fit_result.best_checkpoint} (epoch {best_epoch}, "
          f"val_acc={fit_result.best_val_acc:.4f})")
    print(f"OA: {oa:.4f}  AA: {aa:.4f}  Kappa: {kappa:.4f}  params: {total_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
