"""Train the teaching SSRN on Indian Pines patches (notebooks/08 protocol).

Protocol (kept in sync with notebooks/08, mode="notebook"):
    - per-class split: 20% train / 10% val / 70% test (rng seed 1334, min 1 per class)
    - NO PCA: raw 200 bands as the Conv3d depth axis; StandardScaler fitted on
      the WHOLE cube (notebook-08 convention)
    - 7x7 patches, (batch, 1, 7, 7, 200)
    - SSRN (spectral res-block -> transition -> spatial res-block),
      RMSprop(lr=3e-4, weight_decay=1e-4), CE loss, 10 epochs, batch 16

mode="sklearn" swaps in the course-standard stratified two-step split
(10/10/80, seed 42) so SSRN can join the protocol-C scoreboard.

Beyond the notebook: --lambda-sir adds the paper's spectral invariance
regularisation (Zhong et al. 2018) — the spectral-residual-block output is
penalised for varying across spatial positions of the patch. The notebook
does not implement it; the ablation is this chapter's hands-on contribution.

Artifacts land in results/ssrn/<dataset>/ (same layout as train_1d_cnn.py).

Usage:
    python scripts/train_ssrn.py                                  # notebook parity
    python scripts/train_ssrn.py --epochs 40
    python scripts/train_ssrn.py --epochs 40 --lambda-sir 0.1     # + SIR
    python scripts/train_ssrn.py --mode sklearn --epochs 40       # protocol C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, recall_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SpectralResBlock(nn.Module):
    """Spectral residual block — kept in sync with notebooks/08 (1,1,7 kernels).

    ``use_shortcut=False`` turns the block into plain sequential convolutions
    (the Chapter-12 "remove residual" ablation).
    """

    def __init__(self, channels: int, use_shortcut: bool = True):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, (1, 1, 7), padding=(0, 0, 3))
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, (1, 1, 7), padding=(0, 0, 3))
        self.use_shortcut = use_shortcut
        if use_shortcut:
            self.shortcut = nn.Conv3d(channels, channels, (1, 1, 1))

    def forward(self, x):
        identity = self.shortcut(x) if self.use_shortcut else 0
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return F.relu(identity + out)


class SpatialResBlock(nn.Module):
    """Spatial residual block — kept in sync with notebooks/08 (3,3,1 kernels).

    ``use_shortcut=False`` turns the block into plain sequential convolutions
    (the Chapter-12 "remove residual" ablation).
    """

    def __init__(self, channels: int, use_shortcut: bool = True):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, (3, 3, 1), padding=(1, 1, 0))
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, (3, 3, 1), padding=(1, 1, 0))
        self.use_shortcut = use_shortcut
        if use_shortcut:
            self.shortcut = nn.Conv3d(channels, channels, (1, 1, 1))

    def forward(self, x):
        identity = self.shortcut(x) if self.use_shortcut else 0
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return F.relu(identity + out)


class SSRN(nn.Module):
    """Teaching SSRN — kept in sync with notebooks/08.

    Forward stores the spectral-residual-block output in ``self.last_spectral``
    (before the transition conv) so the training loop can add the paper's
    spectral invariance regularisation without touching the graph elsewhere.
    """

    def __init__(self, num_classes: int, patch_size: int, input_bands: int,
                 use_shortcut: bool = True):
        super().__init__()

        self.spectral_conv1 = nn.Sequential(
            nn.Conv3d(1, 24, (1, 1, 7), stride=(1, 1, 2)),
            nn.BatchNorm3d(24),
            nn.ReLU(True),
        )
        self.spectral_res = SpectralResBlock(24, use_shortcut=use_shortcut)
        self.spectral_bn = nn.BatchNorm3d(24)

        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, patch_size, patch_size, input_bands)
            dummy = self.spectral_conv1(dummy)
            dummy = self.spectral_res(dummy)
            spectral_dim = dummy.shape[-1]

        self.transition_conv = nn.Conv3d(24, 128, (1, 1, spectral_dim))

        self.spatial_conv1 = nn.Sequential(
            nn.Conv3d(1, 24, (3, 3, 128)),
            nn.BatchNorm3d(24),
            nn.ReLU(True),
        )
        self.spatial_res = SpatialResBlock(24, use_shortcut=use_shortcut)
        self.spatial_bn = nn.BatchNorm3d(24)

        with torch.no_grad():
            dummy2 = torch.zeros(1, 1, patch_size, patch_size, 128)
            dummy2 = self.spatial_conv1(dummy2)
            dummy2 = self.spatial_res(dummy2)
            pool_size = (dummy2.shape[2], dummy2.shape[3], dummy2.shape[4])

        self.avg_pool = nn.AvgPool3d(pool_size)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(24, num_classes)
        self.last_spectral: torch.Tensor | None = None

    def forward(self, x):
        x = self.spectral_conv1(x)
        x = self.spectral_res(x)
        x = F.relu(self.spectral_bn(x))
        self.last_spectral = x

        x = self.transition_conv(x)
        x = x.permute(0, 4, 2, 3, 1)

        x = self.spatial_conv1(x)
        x = self.spatial_res(x)
        x = F.relu(self.spatial_bn(x))

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class HSIPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray | None):
        self.patches = torch.from_numpy(patches).unsqueeze(1).float()
        self.labels = None if labels is None else torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.patches[idx]
        return self.patches[idx], self.labels[idx]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the teaching SSRN on patches.")
    parser.add_argument("--dataset", default="IP", choices=["IP", "SA", "PU"])
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--mode", default="notebook", choices=["notebook", "sklearn"],
                        help="notebook = per-class 20/10/70 (seed 1334); "
                             "sklearn = stratified two-step 10/10/80 (seed 42, protocol C)")
    parser.add_argument("--train-rate", type=float, default=0.2, help="notebook mode only.")
    parser.add_argument("--val-rate", type=float, default=0.1, help="notebook mode only.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-sir", type=float, default=0.0,
                        help="weight of the spectral invariance regularisation "
                             "(0 = off, mirroring the notebook).")
    parser.add_argument("--no-residual", action="store_true",
                        help="ablation: drop the residual shortcuts (sequential blocks).")
    parser.add_argument("--seed", type=int, default=None,
                        help="defaults to 1334 (notebook) / 42 (sklearn mode).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results/ssrn")
    return parser


def per_class_split(gt: np.ndarray, train_rate: float, val_rate: float, seed: int):
    """Notebook-08 split: shuffle per class, then 20/10/70 with min 1 per class."""
    class_indices = {c: [] for c in range(1, int(gt.max()) + 1)}
    for i, j in np.argwhere(gt > 0):
        class_indices[gt[i, j]].append((i, j))
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in sorted(class_indices):
        indices = class_indices[c].copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(train_rate * n))
        n_val = max(1, int(val_rate * n))
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])
    return (np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx))


def layer_param_table(model: nn.Module, in_shape: tuple[int, ...]):
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
        if isinstance(mod, (nn.Conv3d, nn.BatchNorm3d, nn.AvgPool3d, nn.Linear, nn.Dropout))
    ]
    with torch.no_grad():
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return rows


def count_macs(model: nn.Module, in_shape: tuple[int, ...]) -> int:
    total = {"v": 0}

    def hook(mod, inp, out):
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kv = 1
            for v in mod.kernel_size:
                kv *= v
            total["v"] += out.numel() * mod.in_channels * kv
        elif isinstance(mod, nn.Linear):
            total["v"] += out.numel() * mod.in_features

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
    ]
    with torch.no_grad():
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return total["v"]


def main() -> int:
    from hsi_learning.data import load_hsi_dataset, split_ground_truth
    from hsi_learning.utils import ensure_dir, resolve_device, save_json, set_seed

    args = build_parser().parse_args()
    seed = args.seed if args.seed is not None else (1334 if args.mode == "notebook" else 42)
    set_seed(seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / args.dataset)

    cube, gt, class_names = load_hsi_dataset(args.dataset, dataset_dir=ROOT / args.dataset_dir)
    h, w, bands = cube.shape

    # preprocessing: no PCA; StandardScaler on the whole cube (notebook-08 convention)
    scaler = StandardScaler()
    norm_cube = scaler.fit_transform(cube.reshape(-1, bands)).astype(np.float32).reshape(h, w, bands)

    if args.mode == "notebook":
        pos_train, pos_val, pos_test = per_class_split(gt, args.train_rate, args.val_rate, seed)
        protocol = {
            "name": "protocol-D (notebook-08)",
            "split": f"per-class {args.train_rate}/{args.val_rate}/"
                     f"{round(1 - args.train_rate - args.val_rate, 2)}, min 1 per class",
            "random_state": seed,
        }
    else:
        train_gt, val_gt, test_gt = split_ground_truth(gt, 0.1, 0.1, random_state=seed)
        pos_train = np.argwhere(train_gt > 0)
        pos_val = np.argwhere(val_gt > 0)
        pos_test = np.argwhere(test_gt > 0)
        protocol = {
            "name": "protocol-C",
            "split": "stratified two-step 10/10/80",
            "random_state": seed,
        }

    def extract(index_list: np.ndarray):
        pad = 7 // 2
        padded = np.pad(norm_cube, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        patches = np.zeros((len(index_list), 7, 7, bands), dtype=np.float32)
        labels = np.zeros(len(index_list), dtype=np.int64)
        for k, (i, j) in enumerate(index_list):
            patches[k] = padded[i: i + 7, j: j + 7]
            labels[k] = gt[i, j] - 1
        return patches, labels

    train_patches, y_train = extract(pos_train)
    val_patches, y_val = extract(pos_val)
    test_patches, y_test = extract(pos_test)
    print(f"train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    train_loader = DataLoader(HSIPatchDataset(train_patches, y_train),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(HSIPatchDataset(val_patches, y_val),
                            batch_size=256, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(HSIPatchDataset(test_patches, y_test),
                             batch_size=256, shuffle=False, num_workers=args.num_workers)
    all_patches, _ = extract(np.argwhere(np.ones_like(gt, dtype=bool)))
    all_loader = DataLoader(HSIPatchDataset(all_patches, None),
                            batch_size=256, shuffle=False, num_workers=args.num_workers)

    model = SSRN(num_classes=len(class_names), patch_size=7, input_bands=bands,
                 use_shortcut=not args.no_residual).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "sir": []}
    best_val_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = running_correct = running_total = 0
        running_sir = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            sir_value = 0.0
            if args.lambda_sir > 0 and model.last_spectral is not None:
                spec = model.last_spectral
                # spectral invariance: penalise variance of the spectral-block
                # output across the patch's spatial positions (H, W)
                sir = ((spec - spec.mean(dim=(2, 3), keepdim=True)) ** 2).mean()
                loss = loss + args.lambda_sir * sir
                sir_value = float(sir.detach())
            loss.backward()
            optimizer.step()
            bs = yb.size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(dim=1) == yb).sum().item()
            running_total += bs
            running_sir += sir_value * bs
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for vb, yb in val_loader:
                correct += (model(vb.to(device)).argmax(dim=1) == yb.to(device)).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        history["train_loss"].append(running_loss / running_total)
        history["train_acc"].append(running_correct / running_total)
        history["val_acc"].append(val_acc)
        history["sir"].append(running_sir / running_total)
        if val_acc >= best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch}/{args.epochs} loss={history['train_loss'][-1]:.4f} "
              f"train_acc={history['train_acc'][-1]:.4f} val_acc={val_acc:.4f}")

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")
    torch.save(best_state, output_dir / "best_model.pth")
    model.load_state_dict(best_state)

    y_true = np.concatenate([y.numpy() for _, y in test_loader])
    y_pred = np.concatenate([
        model(x.to(device)).argmax(dim=1).cpu().numpy() for x, _ in test_loader
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
        for batch in all_loader:
            preds.append(model(batch.to(device)).argmax(dim=1).cpu().numpy() + 1)
    pred_map = np.concatenate(preds).reshape(gt.shape)
    np.save(output_dir / "prediction_map.npy", pred_map)
    np.save(output_dir / "confusion_matrix.npy", cm)

    (output_dir / "classification_report.txt").write_text(
        f"OA: {oa:.6f}\nAA: {aa:.6f}\nKappa: {kappa:.6f}\n\n{report}", encoding="utf-8"
    )
    save_json(
        {
            "dataset": args.dataset,
            "protocol": {
                **protocol,
                "preprocessing": "no PCA; StandardScaler fitted on the whole cube",
                "input": f"(batch, 1, 7, 7, {bands})",
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "optimizer": "RMSprop",
                "lambda_sir": args.lambda_sir,
                "no_residual": args.no_residual,
                "device": str(device),
            },
            "model": {
                "name": "SSRN (teaching variant)",
                "total_params": total_params,
                "macs_per_sample": count_macs(model, (1, 1, 7, 7, bands)),
                "layers": layer_param_table(model, (1, 1, 7, 7, bands)),
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
            "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "total_params": total_params, "lambda_sir": args.lambda_sir,
            "mode": args.mode, "seed": seed,
        },
        output_dir / "metrics.json",
    )
    save_json(history, output_dir / "training_history.json")

    print(f"\nbest epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
    print(f"OA: {oa:.4f}  AA: {aa:.4f}  Kappa: {kappa:.4f}  params: {total_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
