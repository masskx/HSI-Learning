"""Generate Chapter 15 analysis figures: t-SNE, Grad-CAM, spatial bucketing.

Reads the trained 2D CNN checkpoints from results/ch15/{ce,weighted}_seed42/IP/
and produces three figures:

    1. ch15-tsne.png          — t-SNE of penultimate features, CE vs weighted.
    2. ch15-gradcam.png       — Grad-CAM spatial heatmaps on sample patches.
    3. ch15-boundary-error.png — accuracy per patch-purity bucket (CE vs weighted).

Requires the ch15 ablation checkpoints (results/ch15/{ce,weighted}_seed42/IP/).

Usage:
    python scripts/generate_ch15_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _chfigure_utils import (  # noqa: E402
    ASSETS,
    CLASS_NAMES,
    GRID,
    INK,
    apply_style,
    load_indian_pines,
    stretch,
)
from hsi_learning.data import split_ground_truth  # noqa: E402
from train_2d_cnn import SpectralSpatialCNN2D  # noqa: E402

RESULTS = ROOT / "results"
FREQ = [20, 28, 46, 93, 205, 237, 386, 478, 483, 593, 730, 830, 972, 1265, 1428, 2455]

# 4-head color cycle for t-SNE classes (rare classes highlighted)
RARE = {1, 7, 9}  # Alfalfa, Grass-pasture-mowed, Oats (1-based)
PALETTE = ["#898781", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7",
           "#e34948", "#008300", "#e87ba4", "#52514e", "#86b6ef", "#f6a97a",
           "#9085e9", "#0d366b", "#d03b3b", "#cde2fb"]


def load_model_and_data(loss_tag: str, seed: int = 42):
    """Load trained 2D CNN + test data, return (model, features, labels, patches, gt)."""
    cube, gt = load_indian_pines()
    train_gt, val_gt, test_gt = split_ground_truth(gt, 0.1, 0.1, random_state=42)

    # PCA + scaler (same as train_2d_cnn.py)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    flat = cube.reshape(-1, cube.shape[2]).astype(np.float32)
    pca = PCA(n_components=12, whiten=True)
    reduced = pca.fit_transform(flat).astype(np.float32).reshape(cube.shape[0], cube.shape[1], -1)

    pos_test = np.argwhere(test_gt > 0)
    y_test = (test_gt[test_gt > 0] - 1).astype(np.int64)

    scaler = StandardScaler()
    train_pixels = reduced[np.argwhere(train_gt > 0)[:, 0], np.argwhere(train_gt > 0)[:, 1]]
    scaler.fit(train_pixels)
    reduced_scaled = scaler.transform(reduced.reshape(-1, reduced.shape[2])).astype(np.float32)
    reduced_scaled = reduced_scaled.reshape(reduced.shape)

    # extract patches (9x9, zero-padded)
    pad = 4
    padded = np.pad(reduced_scaled, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    patches = np.zeros((len(pos_test), 9, 9, 12), dtype=np.float32)
    for k, (r, c) in enumerate(pos_test):
        patches[k] = padded[r: r + 9, c: c + 9]

    # load model (engine.fit saves epoch_XXX_valacc_YYYY.pth, keep the best)
    ckpts = sorted((RESULTS / "ch15" / f"{loss_tag}_seed{seed}" / "IP").glob("epoch_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {RESULTS / 'ch15' / f'{loss_tag}_seed{seed}'}")
    ckpt = ckpts[-1]  # engine.fit keeps only the best; take the last remaining
    model = SpectralSpatialCNN2D(in_channels=12, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # extract penultimate features (128-d after avg_pool, before fc)
    feats = []
    with torch.no_grad():
        for start in range(0, len(patches), 512):
            batch = torch.from_numpy(patches[start:start + 512]).permute(0, 3, 1, 2)
            f = model.features(batch)          # (B, 128, 1, 1)
            feats.append(f.flatten(1).numpy())
    features = np.concatenate(feats)

    return model, features, y_test, patches, gt, test_gt, reduced_scaled


def fig_tsne(model, features: np.ndarray, y_test: np.ndarray) -> None:
    """t-SNE of penultimate features, colored by class, rare classes starred."""
    sample = np.random.RandomState(42).choice(len(features), 1500, replace=False)
    sub_feat = features[sample]
    sub_y = y_test[sample]

    emb = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(sub_feat)

    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
    for c in range(1, 17):
        mask = sub_y == c - 1
        if not mask.any():
            continue
        is_rare = c in RARE
        ax.scatter(emb[mask, 0], emb[mask, 1], s=14 if is_rare else 8,
                   c=PALETTE[(c - 1) % 16], marker="*" if is_rare else "o",
                   label=CLASS_NAMES[c - 1], alpha=0.85, edgecolors="none")
    ax.legend(fontsize=6.5, ncol=2, frameon=False, markerscale=1.5, loc="best")
    ax.set_title("t-SNE of penultimate features (1,500 test samples, ★ = rare classes)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(ASSETS / "ch15-tsne.png", dpi=200)
    plt.close(fig)


def fig_gradcam(model, patches: np.ndarray, y_test: np.ndarray, gt: np.ndarray, test_gt: np.ndarray) -> None:
    """Grad-CAM on the last conv layer (24, 5, 5, 1), upsampled to 9x9."""
    pos_test = np.argwhere(test_gt > 0)
    # pick 3 samples: one rare-class correct, one common-class correct, one boundary
    rare_cls = 9 - 1  # Oats (0-based)
    rare_idx = np.where(y_test == rare_cls)[0]
    common_cls = 11 - 1  # Soybean-mintill
    common_idx = np.where(y_test == common_cls)[0]

    picks = []
    if len(rare_idx):
        picks.append(("Oats\n(rare)", rare_idx[0]))
    picks.append(("Soybean-mintill\n(common)", common_idx[0]))
    # boundary sample: low purity from ch04 purity calc
    from scipy.ndimage import uniform_filter
    purity = np.zeros_like(gt, dtype=np.float64)
    for c in range(1, 17):
        mask = (gt == c).astype(np.float64)
        purity += mask * uniform_filter(mask, size=25, mode="constant", cval=0.0)
    purity[gt == 0] = 1.0
    test_purity = purity[pos_test[:, 0], pos_test[:, 1]]
    boundary_idx = int(np.argmin(test_purity))
    picks.append(("Boundary\n(lowest purity)", boundary_idx))

    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(4 * (len(picks) + 1), 4.4),
                             constrained_layout=True)
    # first panel: the patch RGB for reference
    cube, gt_full = load_indian_pines()

    for ax_i, (label, idx) in enumerate(picks):
        ax = axes[ax_i]
        r, c = pos_test[idx]
        patch = patches[idx]  # (9, 9, 12)

        # Grad-CAM: gradient of predicted class logit w.r.t. last conv output
        x = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)  # (1, 12, 9, 9)
        from hsi_learning.teaching import grad_cam
        # Select the actual last convolution (index 7), not the preceding ReLU.
        cam_tensor, logits = grad_cam(model, model.features[7], x)
        pred_cls = int(logits.argmax(dim=1).item())
        cam_up = cam_tensor[0].cpu().numpy()

        # display: patch PCA-0 grayscale + CAM overlay
        disp = patch[:, :, 0]
        lo, hi = np.percentile(disp, [2, 98])
        disp = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(disp, cmap="gray", interpolation="nearest")
        ax.imshow(cam_up, cmap="jet", alpha=0.45, interpolation="bilinear")
        true_name = CLASS_NAMES[y_test[idx]]
        pred_name = CLASS_NAMES[pred_cls]
        ax.set_title(f"{label}\ntrue: {true_name}\npred: {pred_name}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # last panel: purity map for context
    ax = axes[-1]
    purity = np.zeros_like(gt, dtype=np.float64)
    from scipy.ndimage import uniform_filter
    for c in range(1, 17):
        mask = (gt == c).astype(np.float64)
        purity += mask * uniform_filter(mask, size=25, mode="constant", cval=0.0)
    purity[gt == 0] = np.nan
    ax.imshow(purity, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Patch purity\n(第 4 章图 4-2 的空间分布)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Grad-CAM：模型在看 patch 的哪个位置？（jet = 高关注）", fontsize=10)
    fig.savefig(ASSETS / "ch15-gradcam.png", dpi=200)
    plt.close(fig)


def fig_boundary_error(model_ce, model_w, patches: np.ndarray, y_test: np.ndarray,
                       gt: np.ndarray, test_gt: np.ndarray) -> None:
    """Accuracy per patch-purity bucket: CE vs weighted CE."""
    pos_test = np.argwhere(test_gt > 0)
    from scipy.ndimage import uniform_filter
    purity = np.zeros_like(gt, dtype=np.float64)
    for c in range(1, 17):
        mask = (gt == c).astype(np.float64)
        purity += mask * uniform_filter(mask, size=25, mode="constant", cval=0.0)
    test_purity = purity[pos_test[:, 0], pos_test[:, 1]]

    buckets = [(0.0, 0.3, "边界\n(<0.3)"), (0.3, 0.8, "过渡\n(0.3–0.8)"), (0.8, 1.01, "内部\n(≥0.8)")]
    preds = {}
    for name, mdl in [("CE", model_ce), ("加权 CE", model_w)]:
        with torch.no_grad():
            logits = []
            for start in range(0, len(patches), 512):
                batch = torch.from_numpy(patches[start:start + 512]).permute(0, 3, 1, 2)
                logits.append(mdl(batch).argmax(dim=1).numpy())
        preds[name] = np.concatenate(logits)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(3)
    w = 0.38
    colors = {"CE": "#898781", "加权 CE": "#2a78d6"}
    for j, (name, pred) in enumerate(preds.items()):
        accs = []
        for lo, hi, _ in buckets:
            mask = (test_purity >= lo) & (test_purity < hi)
            acc = (pred[mask] == y_test[mask]).mean()
            accs.append(acc)
        ax.bar(x + (j - 0.5) * w, np.array(accs) * 100, w, color=colors[name], label=name)
        for xi, a in zip(x + (j - 0.5) * w, accs):
            ax.text(xi, a * 100 + 0.8, f"{a * 100:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [b[2] for b in buckets], fontsize=10)
    ax.set_ylabel("精度（%）", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("空间分桶误差：边界样本的精度损失 = 空间上下文的价值\n（协议 C，2D CNN，patch 9）", fontsize=11)
    fig.savefig(ASSETS / "ch15-boundary-error.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()

    model_ce, feats_ce, y_test, patches, gt, test_gt, _ = load_model_and_data("ce")
    model_w, _, _, _, _, _, _ = load_model_and_data("weighted")

    fig_tsne(model_ce, feats_ce, y_test)
    fig_gradcam(model_ce, patches, y_test, gt, test_gt)
    fig_boundary_error(model_ce, model_w, patches, y_test, gt, test_gt)
    print("analysis figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
