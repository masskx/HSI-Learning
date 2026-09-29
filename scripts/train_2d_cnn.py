"""Train the teaching 2D CNN on Indian Pines patches (notebooks/05 protocol).

Protocol (kept in sync with notebooks/05):
    - 10% train / 10% val / 80% test, stratified two-step split, seed 42
    - PCA(12, whiten=True) fitted on the WHOLE cube (notebook convention)
    - StandardScaler fitted on the reduced TRAIN pixels only (anti-leakage)
    - 9x9 patches around each labeled pixel, (C, H, W) tensors
    - SpectralSpatialCNN2D, Adam(lr=1e-3, weight_decay=1e-4), CE loss, 10 epochs

Artifacts land in results/2d_cnn/<dataset>/ (same layout as train_1d_cnn.py).

Usage:
    python scripts/train_2d_cnn.py
    python scripts/train_2d_cnn.py --patch-size 25 --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, recall_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SpectralSpatialCNN2D(nn.Module):
    """Teaching 2D CNN — kept in sync with notebooks/05 (SpectralSpatialCNN2D)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PatchDataset2D(Dataset):
    """(C, H, W) patches around given pixel positions — kept in sync with notebooks/05."""

    def __init__(self, cube: np.ndarray, positions: np.ndarray, labels: np.ndarray | None,
                 patch_size: int):
        self.patch_size = patch_size
        self.labels = labels
        self.positions = np.asarray(positions)
        self.pad = patch_size // 2
        self.cube = np.pad(cube, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                           mode="constant")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        r, c = self.positions[idx]
        rp, cp = r + self.pad, c + self.pad
        p = self.patch_size // 2
        patch = self.cube[rp - p: rp + p + 1, cp - p: cp + p + 1]
        patch = torch.from_numpy(patch.transpose(2, 0, 1).astype(np.float32))
        if self.labels is None:
            return patch
        return patch, torch.tensor(int(self.labels[idx]), dtype=torch.long)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., ICCV 2017): down-weights easy examples.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t); gamma=0 reduces to plain CE.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


def build_criterion(loss_name: str, y_train: np.ndarray, num_classes: int,
                    focal_gamma: float, device: torch.device) -> nn.Module:
    """Loss factory for the class-imbalance study (Chapter 15)."""
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "weighted":
        counts = np.bincount(y_train, minlength=num_classes)
        weights = len(y_train) / (num_classes * np.maximum(counts, 1))  # sklearn "balanced"
        print("class weights (balanced):", np.round(weights, 3))
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma)
    raise ValueError(f"unknown loss {loss_name!r} (ce | weighted | focal)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the teaching 2D CNN on patches.")
    parser.add_argument("--dataset", default="IP", choices=["IP", "SA", "PU"])
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--train-rate", type=float, default=0.1)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", default="ce", choices=["ce", "weighted", "focal"],
                        help="class-imbalance loss (Chapter 15): plain CE, "
                             "class-weighted CE, or focal loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results/2d_cnn")
    return parser


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
        if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                            nn.Linear, nn.Flatten, nn.Dropout))
    ]
    with torch.no_grad():
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return rows


@torch.no_grad()
def predict_all(model: nn.Module, dataset: Dataset, device: torch.device,
                batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for batch in loader:
        preds.append(model(batch.to(device)).argmax(dim=1).cpu().numpy() + 1)  # 0..15 -> 1..16
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

    # PCA on the whole cube (notebook-05 convention, mirrors apply_pca_cube).
    flat = cube.reshape(-1, cube.shape[2]).astype(np.float32)
    pca = PCA(n_components=args.pca_components, whiten=True)
    reduced = pca.fit_transform(flat).astype(np.float32).reshape(cube.shape[0], cube.shape[1], -1)
    print(f"PCA-{args.pca_components} cumulative EVR: "
          f"{float(pca.explained_variance_ratio_.sum()) * 100:.2f}%")

    positions = np.argwhere(gt > 0)
    y_all = (gt[gt > 0] - 1).astype(np.int64)  # 0..15
    train_mask, val_mask, test_mask = (train_gt > 0), (val_gt > 0), (test_gt > 0)
    pos_train = positions[train_mask[gt > 0]]
    pos_val = positions[val_mask[gt > 0]]
    pos_test = positions[test_mask[gt > 0]]
    y_train, y_val, y_test = y_all[train_mask[gt > 0]], y_all[val_mask[gt > 0]], y_all[test_mask[gt > 0]]

    # StandardScaler on reduced features, fitted on train pixels only.
    scaler = StandardScaler()
    scaler.fit(reduced[pos_train[:, 0], pos_train[:, 1]])
    reduced_scaled = scaler.transform(reduced.reshape(-1, reduced.shape[2])).astype(np.float32)
    reduced_scaled = reduced_scaled.reshape(reduced.shape)

    train_ds = PatchDataset2D(reduced_scaled, pos_train, y_train, args.patch_size)
    val_ds = PatchDataset2D(reduced_scaled, pos_val, y_val, args.patch_size)
    test_ds = PatchDataset2D(reduced_scaled, pos_test, y_test, args.patch_size)
    all_ds = PatchDataset2D(reduced_scaled, np.argwhere(np.ones_like(gt, dtype=bool)),
                            labels=None, patch_size=args.patch_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = SpectralSpatialCNN2D(in_channels=args.pca_components,
                                 num_classes=len(class_names)).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    criterion = build_criterion(args.loss, y_train, len(class_names),
                                args.focal_gamma, device)

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
        criterion=criterion,
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

    pred_map = predict_all(model, all_ds, device).reshape(gt.shape)
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
                "preprocessing": (f"PCA-{args.pca_components} whiten on the whole cube, "
                                  "then StandardScaler fitted on reduced train pixels"),
                "input": f"(batch, {args.pca_components}, {args.patch_size}, {args.patch_size})",
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "loss": args.loss,
                "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
                "device": str(device),
            },
            "model": {
                "name": "SpectralSpatialCNN2D",
                "total_params": total_params,
                "layers": layer_param_table(model, (1, args.pca_components,
                                                    args.patch_size, args.patch_size)),
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
