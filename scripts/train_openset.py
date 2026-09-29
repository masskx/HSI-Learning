"""Open-set HSI classification evaluation (Chapter 17).

Protocol:
    Train 2D CNN on KNOWN classes only (12 of 16 for IP).
    Test on ALL classes (known + unknown).
    The model must classify known samples correctly AND detect unknown ones.

Methods:
    1. MSP (Maximum Softmax Probability) — the simplest open-set baseline.
    2. Distance-threshold — distance to nearest class prototype in embedding space.

Metrics:
    AUROC for known vs unknown discrimination (higher = better).

Usage:
    python scripts/train_openset.py
    python scripts/train_openset.py --n-unknown 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_2d_cnn import SpectralSpatialCNN2D  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-set HSI classification (ch17).")
    parser.add_argument("--dataset", default="IP", choices=["IP"])
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--train-rate", type=float, default=0.1)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/openset")
    return parser


def extract_patches(reduced_scaled: np.ndarray, positions: np.ndarray, patch_size: int):
    pad = patch_size // 2
    padded = np.pad(reduced_scaled, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    p = patch_size // 2
    patches = np.zeros((len(positions), patch_size, patch_size, reduced_scaled.shape[2]), dtype=np.float32)
    for k, (r, c) in enumerate(positions):
        patches[k] = padded[r: r + patch_size, c: c + patch_size]
    return patches


@torch.no_grad()
def extract_embeddings(model, patches: np.ndarray, device: torch.device, batch_size: int = 512):
    embeddings = []
    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[start:start + batch_size]).permute(0, 3, 1, 2).to(device)
        emb = model.features(batch).flatten(1)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings)


@torch.no_grad()
def predict_logits(model, patches: np.ndarray, device: torch.device, batch_size: int = 512):
    logits = []
    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[start:start + batch_size]).permute(0, 3, 1, 2).to(device)
        logits.append(model(batch).cpu().numpy())
    return np.concatenate(logits)


def main() -> int:
    from hsi_learning.data import load_hsi_dataset, split_ground_truth
    from hsi_learning.utils import ensure_dir, resolve_device, save_json, set_seed

    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / args.dataset)

    cube, gt, class_names = load_hsi_dataset(args.dataset, dataset_dir=ROOT / args.dataset_dir)
    n_classes = len(class_names)

    # known/unknown class split: use class frequency, take the n_unknown smallest as unknown
    class_counts = {c: int((gt == c).sum()) for c in range(1, n_classes + 1)}
    sorted_by_freq = sorted(class_counts, key=class_counts.get)
    n_unknown = 4
    unknown_classes = set(sorted_by_freq[:n_unknown])
    known_classes = set(range(1, n_classes + 1)) - unknown_classes
    print(f"known classes ({len(known_classes)}): {sorted(known_classes)}")
    print(f"unknown classes ({len(unknown_classes)}): {sorted(unknown_classes)}")
    print(f"  unknown names: {[class_names[c - 1] for c in sorted(unknown_classes)]}")

    # PCA + scaler (same pipeline as train_2d_cnn.py)
    flat = cube.reshape(-1, cube.shape[2]).astype(np.float32)
    pca = PCA(n_components=args.pca_components, whiten=True)
    reduced = pca.fit_transform(flat).astype(np.float32).reshape(cube.shape[0], cube.shape[1], -1)

    train_gt, val_gt, test_gt = split_ground_truth(
        gt, train_rate=args.train_rate, val_rate=args.val_rate, random_state=args.seed
    )

    # filter training to known classes only
    train_mask = (train_gt > 0) & np.isin(train_gt, list(known_classes))
    pos_train = np.argwhere(train_mask)
    y_train = (train_gt[train_mask] - 1).astype(np.int64)
    # remap known class ids to 0..len(known)-1
    known_sorted = sorted(known_classes)
    class_map = {c: i for i, c in enumerate(known_sorted)}
    y_train = np.array([class_map[c + 1] for c in y_train])

    pos_test = np.argwhere(test_gt > 0)
    y_test_original = (test_gt[test_gt > 0] - 1).astype(np.int64)  # 0..15
    is_known_test = np.isin(y_test_original + 1, list(known_classes))  # True = known, False = unknown

    scaler = StandardScaler()
    scaler.fit(reduced[pos_train[:, 0], pos_train[:, 1]])
    reduced_scaled = scaler.transform(reduced.reshape(-1, reduced.shape[2])).astype(np.float32)
    reduced_scaled = reduced_scaled.reshape(reduced.shape)

    train_patches = extract_patches(reduced_scaled, pos_train, args.patch_size)
    test_patches = extract_patches(reduced_scaled, pos_test, args.patch_size)

    # train 2D CNN on known classes only
    n_known = len(known_classes)
    model = SpectralSpatialCNN2D(in_channels=args.pca_components, num_classes=n_known).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.from_numpy(train_patches).permute(0, 3, 1, 2), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print(f"training on {len(y_train)} known-class samples for {args.epochs} epochs...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch}/{args.epochs}  loss={total_loss / total:.4f}  train_acc={correct / total:.4f}")

    # ---- evaluation -----------------------------------------------------------
    model.eval()

    # 1. MSP (max softmax probability)
    logits = predict_logits(model, test_patches, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    msp_scores = probs.max(axis=1)  # higher = more confident = more likely known

    # 2. distance to nearest class prototype (in embedding space)
    embeddings = extract_embeddings(model, test_patches, device)
    train_embeddings = extract_embeddings(model, train_patches, device)
    prototypes = np.stack([train_embeddings[y_train == c].mean(axis=0) for c in range(n_known)])
    # distance to nearest prototype (lower = more likely known)
    distances = np.sqrt(((embeddings[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2))
    min_distances = distances.min(axis=1)  # lower = more likely known

    # AUROC: known samples should have HIGHER MSP and LOWER distance
    # for AUROC, "positive" = unknown → negate MSP, use distance directly
    auroc_msp = roc_auc_score(~is_known_test, -msp_scores)  # negate: unknown has low MSP
    auroc_dist = roc_auc_score(~is_known_test, min_distances)  # unknown has high distance

    # closed-set accuracy on known test samples
    known_mask = is_known_test
    known_preds = logits[known_mask].argmax(axis=1)
    known_true = np.array([class_map[c + 1] for c in y_test_original[known_mask]])
    closed_acc = accuracy_score(known_true, known_preds)

    print(f"\nclosed-set accuracy (known classes only): {closed_acc:.4f}")
    print(f"AUROC (MSP):        {auroc_msp:.4f}")
    print(f"AUROC (distance):   {auroc_dist:.4f}")

    # save per-sample scores for figure generation
    np.save(output_dir / "msp_scores.npy", msp_scores)
    np.save(output_dir / "min_distances.npy", min_distances)
    np.save(output_dir / "is_known_test.npy", is_known_test)
    save_json(
        {
            "known_classes": sorted(known_classes),
            "unknown_classes": sorted(unknown_classes),
            "unknown_names": [class_names[c - 1] for c in sorted(unknown_classes)],
            "n_train_known": int(len(y_train)),
            "n_test_known": int(is_known_test.sum()),
            "n_test_unknown": int((~is_known_test).sum()),
            "closed_set_accuracy": float(closed_acc),
            "auroc_msp": float(auroc_msp),
            "auroc_distance": float(auroc_dist),
            "seed": args.seed,
        },
        output_dir / "openset_results.json",
    )
    print(f"saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
