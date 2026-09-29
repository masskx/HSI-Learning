"""Train Prototypical Networks for few-shot HSI classification (Chapter 16).

Core idea (Snell et al., NeurIPS 2017):
    Learn an embedding space where class prototypes (mean of support
    embeddings) classify query samples by Euclidean distance.

Episodic training:
    Each episode samples N classes, K support + Q query samples per class.
    The model learns to generalize from K labeled examples to Q queries.

Uses the same 2D CNN backbone (SpectralSpatialCNN2D.features) as ch06 —
without the classifier head; the prototype distance replaces it.

Usage:
    python scripts/train_protonet.py                          # 5-way 5-shot
    python scripts/train_protonet.py --k-shot 1               # 1-shot
    python scripts/train_protonet.py --k-shot 10              # 10-shot
    python scripts/train_protonet.py --n-way 3                # 3-way
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_2d_cnn import SpectralSpatialCNN2D  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Prototypical Networks on HSI patches.")
    parser.add_argument("--dataset", default="IP", choices=["IP", "SA", "PU"])
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--train-rate", type=float, default=0.1)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--n-way", type=int, default=5, help="Classes per episode.")
    parser.add_argument("--k-shot", type=int, default=5, help="Support samples per class.")
    parser.add_argument("--q-query", type=int, default=15, help="Query samples per class.")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Test episodes.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/protonet")
    return parser


class EpisodicSampler:
    """Samples N-way K-shot + Q-query episodes from labeled patches."""

    def __init__(self, patches: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
        self.patches = patches
        self.labels = labels
        self.rng = rng
        self.class_indices = {
            c: np.where(labels == c)[0] for c in range(labels.max() + 1)
        }

    def sample_episode(self, n_way: int, k_shot: int, q_query: int):
        """Returns (support_patches, support_labels, query_patches, query_labels)."""
        available = [c for c in self.class_indices if len(self.class_indices[c]) >= k_shot + q_query]
        classes = self.rng.choice(available, size=min(n_way, len(available)), replace=False)

        support_x, support_y, query_x, query_y = [], [], [], []
        for new_label, c in enumerate(classes):
            idx = self.rng.choice(self.class_indices[c], size=k_shot + q_query, replace=False)
            support_x.append(self.patches[idx[:k_shot]])
            support_y.append(np.full(k_shot, new_label))
            query_x.append(self.patches[idx[k_shot:]])
            query_y.append(np.full(q_query, new_label))

        return (
            np.concatenate(support_x),
            np.concatenate(support_y),
            np.concatenate(query_x),
            np.concatenate(query_y),
        )


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N, D) prototypes; b: (M, D) queries → (M, N) distances."""
    return torch.cdist(b, a)  # (M, N)


def prototypical_loss(support_emb: torch.Tensor, support_y: torch.Tensor,
                      query_emb: torch.Tensor, query_y: torch.Tensor,
                      n_way: int) -> torch.Tensor:
    """Compute prototypes from support set, classify queries by distance."""
    prototypes = torch.stack([
        support_emb[support_y == c].mean(dim=0) for c in range(n_way)
    ])  # (n_way, D)
    distances = euclidean_distance(prototypes, query_emb)  # (M, n_way)
    log_p_y = F.log_softmax(-distances, dim=1)  # closer → higher log-prob
    return F.nll_loss(log_p_y, query_y)


@torch.no_grad()
def evaluate_episodes(model, sampler: EpisodicSampler, n_episodes: int,
                      n_way: int, k_shot: int, q_query: int,
                      device: torch.device) -> float:
    """Average accuracy over n_episodes test episodes."""
    correct = total = 0
    for _ in range(n_episodes):
        sx, sy, qx, qy = sampler.sample_episode(n_way, k_shot, q_query)
        sx = torch.from_numpy(sx).permute(0, 3, 1, 2).to(device)
        qx = torch.from_numpy(qx).permute(0, 3, 1, 2).to(device)
        sy = torch.from_numpy(sy).to(device)
        qy = torch.from_numpy(qy).to(device)

        support_emb = model.features(sx).flatten(1)
        query_emb = model.features(qx).flatten(1)
        prototypes = torch.stack([support_emb[sy == c].mean(dim=0) for c in range(n_way)])
        distances = euclidean_distance(prototypes, query_emb)
        preds = distances.argmin(dim=1)
        correct += (preds == qy).sum().item()
        total += len(qy)
    return correct / total if total else 0.0


def main() -> int:
    from hsi_learning.data import load_hsi_dataset, split_ground_truth
    from hsi_learning.utils import ensure_dir, resolve_device, save_json, set_seed

    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / args.dataset)

    cube, gt, class_names = load_hsi_dataset(args.dataset, dataset_dir=ROOT / args.dataset_dir)

    # PCA + scaler (same pipeline as train_2d_cnn.py)
    flat = cube.reshape(-1, cube.shape[2]).astype(np.float32)
    pca = PCA(n_components=args.pca_components, whiten=True)
    reduced = pca.fit_transform(flat).astype(np.float32).reshape(cube.shape[0], cube.shape[1], -1)

    train_gt, val_gt, test_gt = split_ground_truth(
        gt, train_rate=args.train_rate, val_rate=args.val_rate, random_state=args.seed
    )
    scaler = StandardScaler()
    train_pos = np.argwhere(train_gt > 0)
    scaler.fit(reduced[train_pos[:, 0], train_pos[:, 1]])
    reduced_scaled = scaler.transform(reduced.reshape(-1, reduced.shape[2])).astype(np.float32)
    reduced_scaled = reduced_scaled.reshape(reduced.shape)

    def extract(index_list):
        pad = args.patch_size // 2
        padded = np.pad(reduced_scaled, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        patches = np.zeros((len(index_list), args.patch_size, args.patch_size, args.pca_components), dtype=np.float32)
        labels = np.zeros(len(index_list), dtype=np.int64)
        for k, (i, j) in enumerate(index_list):
            patches[k] = padded[i: i + args.patch_size, j: j + args.patch_size]
            labels[k] = gt[i, j] - 1
        return patches, labels

    # train patches (for episodic sampling) — use ALL labeled train pixels
    pos_train = np.argwhere(train_gt > 0)
    y_train = (train_gt[train_gt > 0] - 1).astype(np.int64)
    train_patches, _ = extract(pos_train)
    pos_test = np.argwhere(test_gt > 0)
    y_test = (test_gt[test_gt > 0] - 1).astype(np.int64)
    test_patches, _ = extract(pos_test)

    print(f"train: {len(y_train)}, test: {len(y_test)}, classes: {len(np.unique(y_train))}")
    print(f"episodic: {args.n_way}-way {args.k_shot}-shot, q_query={args.q_query}")

    rng = np.random.default_rng(args.seed)
    train_sampler = EpisodicSampler(train_patches, y_train, rng)
    test_sampler = EpisodicSampler(test_patches, y_test, rng)

    # encoder = 2D CNN features (no classifier head)
    encoder = SpectralSpatialCNN2D(in_channels=args.pca_components,
                                   num_classes=len(np.unique(y_train))).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    history = {"loss": [], "val_acc": []}
    for ep in range(1, args.episodes + 1):
        encoder.train()
        sx, sy, qx, qy = train_sampler.sample_episode(args.n_way, args.k_shot, args.q_query)
        sx = torch.from_numpy(sx).permute(0, 3, 1, 2).to(device)
        qx = torch.from_numpy(qx).permute(0, 3, 1, 2).to(device)
        sy_t = torch.from_numpy(sy).to(device)
        qy_t = torch.from_numpy(qy).to(device)

        optimizer.zero_grad()
        support_emb = encoder.features(sx).flatten(1)
        query_emb = encoder.features(qx).flatten(1)
        loss = prototypical_loss(support_emb, sy_t, query_emb, qy_t, args.n_way)
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())

        if ep % 50 == 0 or ep == 1:
            encoder.eval()
            val_acc = evaluate_episodes(encoder, test_sampler, 20, args.n_way, args.k_shot, args.q_query, device)
            history["val_acc"].append(val_acc)
            print(f"episode {ep}/{args.episodes}  loss={loss.item():.4f}  test_acc({args.n_way}-way {args.k_shot}-shot)={val_acc:.4f}")

    # final evaluation
    encoder.eval()
    final_acc = evaluate_episodes(encoder, test_sampler, args.eval_episodes,
                                  args.n_way, args.k_shot, args.q_query, device)
    print(f"\nfinal {args.n_way}-way {args.k_shot}-shot accuracy over {args.eval_episodes} episodes: {final_acc:.4f}")

    save_json(
        {
            "n_way": args.n_way, "k_shot": args.k_shot, "q_query": args.q_query,
            "episodes": args.episodes, "eval_episodes": args.eval_episodes,
            "final_accuracy": final_acc,
            "total_params": sum(p.numel() for p in encoder.parameters()),
            "seed": args.seed, "dataset": args.dataset,
            "history_loss_last10": history["loss"][-10:],
        },
        output_dir / f"protonet_{args.n_way}way_{args.k_shot}shot.json",
    )
    torch.save(encoder.state_dict(), output_dir / "encoder.pth")
    print(f"saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
