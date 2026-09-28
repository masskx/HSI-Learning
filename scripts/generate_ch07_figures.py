"""Generate Chapter 7 figures: 3D CNN on Indian Pines patches.

Reads artifacts from results/3d_cnn/IP (10-epoch default),
results/3d_cnn_ip_epochs40 (extended), results/2d_cnn_ip_epochs40/IP and
results/1d_cnn (for cross-model comparisons), and produces:

    1. ch07-conv3d-footprint.png  — one (7,3,3) Conv3d kernel's footprint over
                                    real band slices of a 9x9 patch.
    2. ch07-cost-comparison.png   — parameters and measured CPU forward time
                                    of the 1D / 2D / 3D teaching models.
    3. ch07-val-curves.png        — default-config val_acc curves of the three
                                    models under protocol C.
    4. ch07-confusion-matrix.png  — 3D CNN (40 epochs) test confusion matrix.

Run scripts/train_3d_cnn.py (+ the 40-epoch control) first.

Usage:
    python scripts/generate_ch07_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _chfigure_utils import (  # noqa: E402
    ASSETS,
    CLASS_NAMES,
    GRID,
    apply_style,
    fig_confusion_matrix,
    load_indian_pines,
)
from train_1d_cnn import SpectralCNN1D  # noqa: E402
from train_2d_cnn import SpectralSpatialCNN2D  # noqa: E402
from train_3d_cnn import SpectralSpatialCNN3D  # noqa: E402

RESULTS = ROOT / "results"
RUN_DEFAULT = RESULTS / "3d_cnn" / "IP"
RUN_40 = RESULTS / "3d_cnn_ip_epochs40" / "IP"
RUN_2D_40 = RESULTS / "2d_cnn_ip_epochs40" / "IP"
RUN_1D = RESULTS / "1d_cnn" / "IP"
RUN_2D = RESULTS / "2d_cnn" / "IP"
ACCENT = ["#2a78d6", "#eb6834", "#1baf7a"]


def fig_conv3d_footprint(cube: np.ndarray, gt: np.ndarray) -> None:
    """Real 9x9 patch band slices with one (7,3,3) kernel's footprint shaded."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=15, whiten=True)
    reduced = pca.fit_transform(cube.reshape(-1, cube.shape[2]).astype(np.float32))
    reduced_cube = reduced.reshape(145, 145, 15)

    # boundary pixel (ch04's B): smallest same-class 25x25 neighborhood
    purity = np.zeros_like(gt, dtype=np.float64)
    for c in range(1, 17):
        mask = (gt == c).astype(np.float64)
        purity += mask * uniform_filter(mask, size=25, mode="constant", cval=0.0)
    purity[gt == 0] = 1.0
    r, c = np.unravel_index(np.argmin(purity), gt.shape)

    pad = 12
    padded = np.pad(reduced_cube, ((pad, pad), (pad, pad), (0, 0)))
    patch = padded[r: r + 25, c: c + 25][8:17, 8:17]  # (9, 9, 15), pipeline-identical

    fig, axes = plt.subplots(3, 5, figsize=(13.5, 8.0), constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        lo, hi = np.percentile(patch[:, :, k], [2, 98])
        ax.imshow(patch[:, :, k], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        if k < 7:  # footprint of a (7,3,3) kernel: bands 0..6, center 3x3
            ax.add_patch(Rectangle((3, 3), 3, 3, fill=True, facecolor="#eb6834",
                                   alpha=0.45, edgecolor="#eb6834", linewidth=1.4))
        ax.set_title(f"band {k}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("一个 Conv3d(7,3,3) 核的单次采样：跨 7 个波段、每次 3×3 空间（橙色）"
                 "——63 个原始值聚合为 1 个特征", fontsize=11)
    fig.savefig(ASSETS / "ch07-conv3d-footprint.png", dpi=200)
    plt.close(fig)


def count_macs(model: torch.nn.Module, in_shape: tuple[int, ...]) -> int:
    """Multiply-accumulate ops per sample, counted analytically via hooks.

    Deterministic (unlike wall-clock timing on a busy machine): a Conv layer
    contributes out_numel x C_in x prod(kernel_size) MACs, a Linear layer
    out_numel x in_features.
    """
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
        model(torch.zeros(1, *in_shape))
    for handle in handles:
        handle.remove()
    return total["v"]


def fig_cost_comparison() -> None:
    models = {
        "1D CNN": (SpectralCNN1D(num_bands=200, num_classes=16), (1, 200)),
        "2D CNN": (SpectralSpatialCNN2D(in_channels=12, num_classes=16), (12, 9, 9)),
        "3D CNN": (SpectralSpatialCNN3D(num_classes=16), (1, 15, 9, 9)),
    }
    names, params, macs = [], [], []
    for name, (model, in_shape) in models.items():
        names.append(name)
        params.append(sum(p.numel() for p in model.parameters()))
        macs.append(count_macs(model, in_shape))

    fig, (ax_p, ax_m) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    bars = ax_p.bar(names, params, color="#2a78d6", width=0.55)
    ax_p.set_ylabel("参数量", fontsize=10)
    ax_p.set_title("参数量：3D 核在通道维聚合，参数反而最少", fontsize=10)
    for b, p in zip(bars, params):
        ax_p.text(b.get_x() + b.get_width() / 2, p, f"{p / 1000:.1f}k", ha="center",
                  va="bottom", fontsize=9)

    bars = ax_m.bar(names, macs, color="#eb6834", width=0.55)
    ax_m.set_ylabel("每样本乘加次数（MACs，解析计算）", fontsize=10)
    ax_m.set_title("计算量：3D 的卷积跨三个维度", fontsize=10)
    for b, m in zip(bars, macs):
        ax_m.text(b.get_x() + b.get_width() / 2, m, f"{m / 1e6:.2f}M", ha="center",
                  va="bottom", fontsize=9)

    for ax in (ax_p, ax_m):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(ASSETS / "ch07-cost-comparison.png", dpi=200)
    plt.close(fig)
    print("MACs per sample (analytic):",
          {n: f"{m / 1e6:.2f}M" for n, m in zip(names, macs)})


def fig_val_curves() -> None:
    runs = [
        ("1D CNN（15 epochs）", RUN_1D / "training_history.json", ACCENT[0]),
        ("2D CNN（10 epochs）", RUN_2D / "training_history.json", ACCENT[1]),
        ("3D CNN（10 epochs）", RUN_DEFAULT / "training_history.json", ACCENT[2]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, path, color in runs:
        hist = json.loads(Path(path).read_text(encoding="utf-8"))
        epochs = np.arange(1, len(hist["val_acc"]) + 1)
        ax.plot(epochs, hist["val_acc"], color=color, linewidth=2.0, marker="o",
                markersize=3.5, label=name)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("验证集准确率", fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(ASSETS / "ch07-val-curves.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()

    metrics_10 = json.loads((RUN_DEFAULT / "metrics.json").read_text(encoding="utf-8"))
    metrics_40 = json.loads((RUN_40 / "metrics.json").read_text(encoding="utf-8"))
    cm_40 = np.load(RUN_40 / "confusion_matrix.npy")
    cm_2d = np.load(RUN_2D_40 / "confusion_matrix.npy")

    fig_conv3d_footprint(cube, gt)
    fig_cost_comparison()
    fig_val_curves()
    fig_confusion_matrix(cm_40, ASSETS / "ch07-confusion-matrix.png")

    print(f"3D CNN 10 epochs: OA={metrics_10['oa'] * 100:.2f}  AA={metrics_10['aa'] * 100:.2f}  "
          f"Kappa={metrics_10['kappa']:.4f}  best_epoch={metrics_10['best_epoch']}")
    print(f"3D CNN 40 epochs: OA={metrics_40['oa'] * 100:.2f}  AA={metrics_40['aa'] * 100:.2f}  "
          f"Kappa={metrics_40['kappa']:.4f}  best_epoch={metrics_40['best_epoch']}")

    recall_2d = np.diag(cm_2d) / cm_2d.sum(axis=1)
    recall_3d = np.diag(cm_40) / cm_40.sum(axis=1)
    print("\nper-class recall (2D 40ep vs 3D 40ep):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i + 1:>2}  {name:<32}{recall_2d[i]:.3f}  ->  {recall_3d[i]:.3f}")
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
