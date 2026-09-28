"""Train the teaching 1D CNN on Indian Pines spectra (notebooks/04 protocol).

Protocol (kept in sync with notebooks/04):
    - 10% train / 10% val / 80% test, stratified two-step split, seed 42
    - StandardScaler fitted on the training split ONLY (anti-leakage)
    - spectrum passed to Conv1d as (batch, 1, bands)
    - SpectralCNN1D, Adam(lr=1e-3, weight_decay=1e-4), CE loss, 15 epochs

Artifacts land in results/1d_cnn/<dataset>/:
    run_config.json, training_history.json, training_curves.png,
    best checkpoint (.pth), metrics.json, classification_report.txt,
    confusion_matrix.npy, prediction_map.npy

Usage:
    python scripts/train_1d_cnn.py
    python scripts/train_1d_cnn.py --epochs 30 --lr 3e-4
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


class SpectralCNN1D(nn.Module):
    """Teaching 1D CNN — kept in sync with notebooks/04 (SpectralCNN1D)."""

    def __init__(self, num_bands: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the teaching 1D CNN on spectra.")
    parser.add_argument("--dataset", default="IP", choices=["IP", "SA", "PU"])
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--train-rate", type=float, default=0.1, help="Training split ratio.")
    parser.add_argument("--val-rate", type=float, default=0.1, help="Validation ratio of the total.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, auto, or an integer index.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results/1d_cnn")
    return parser


def spectra_from_map(cube: np.ndarray, label_map: np.ndarray):
    rows, cols = np.nonzero(label_map)
    X = cube[rows, cols].astype(np.float32)
    y = (label_map[rows, cols] - 1).astype(np.int64)  # 1..16 -> 0..15
    return X, y


def layer_param_table(model: nn.Module, in_shape: tuple[int, ...]):
    """Per-layer output shape and parameter count via a dummy forward pass."""
    rows = []

    def hook(name):
        def fn(mod, inp, out):
            rows.append(
                {
                    "layer": name,
                    "type": mod.__class__.__name__,
                    "out_shape": list(out.shape),
                    "params": sum(p.numel() for p in mod.parameters()),
                }
            )
        return fn

    handles = [
        mod.register_forward_hook(hook(name))
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d,
                            nn.Linear, nn.Flatten, nn.Dropout))
    ]
    with torch.no_grad():
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return rows


@torch.no_grad()
def predict_all(model: nn.Module, features: np.ndarray, device: torch.device,
                batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).unsqueeze(1).to(device)
        preds.append(model(batch).argmax(dim=1).cpu().numpy() + 1)  # 0..15 -> 1..16
    return np.concatenate(preds)


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

    # StandardScaler on the training split only (anti-leakage; notebooks/04 convention).
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_all_s = scaler.transform(X_all).astype(np.float32)

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers)

    train_loader = to_loader(X_train_s, y_train, shuffle=True)
    val_loader = to_loader(X_val_s, y_val, shuffle=False)
    test_loader = to_loader(X_test_s, y_test, shuffle=False)

    model = SpectralCNN1D(num_bands=X_train.shape[1], num_classes=len(class_names)).to(device)
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

    # test metrics (labels reported on the 1..16 scale)
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

    # full-image prediction map
    pred_map = predict_all(model, X_all_s, device).reshape(gt.shape)
    np.save(output_dir / "prediction_map.npy", pred_map)
    np.save(output_dir / "confusion_matrix.npy", cm)

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
                "input": "(batch, 1, bands) raw standardized bands",
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "device": str(device),
            },
            "model": {
                "name": "SpectralCNN1D",
                "total_params": total_params,
                "layers": layer_param_table(model, (1, 1, X_train.shape[1])),
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
            "oa": oa,
            "aa": aa,
            "kappa": kappa,
            "best_val_acc": fit_result.best_val_acc,
            "best_epoch": best_epoch,
            "best_checkpoint": str(fit_result.best_checkpoint),
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
