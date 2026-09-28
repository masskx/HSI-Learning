"""Generate Chapter 6 figures: 2D CNN on Indian Pines patches.

Reads artifacts from
    results/2d_cnn/IP          (notebook-05 default run, 10 epochs)
    results/2d_cnn_ip_epochs40 (extended run, 40 epochs)
    results/1d_cnn_ip_epochs60 (ch05 extended 1D run, for the map comparison)
and produces:

    1. ch06-patch-and-receptive-field.png — the 9x9 model input vs the 25x25
       patch, plus the receptive-field growth 3 -> 7 -> 11.
    2. ch06-training-curves.png           — curves of the default 10-epoch run.
    3. ch06-confusion-matrix.png          — 40-epoch run (properly trained).
    4. ch06-1d-vs-2d-maps.png             — GT vs 1D CNN vs 2D CNN full maps.

Run scripts/train_2d_cnn.py (and the 40-epoch control) first.

Usage:
    python scripts/generate_ch06_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter

from _chfigure_utils import (
    ASSETS,
    CLASS_NAMES,
    GRID,
    INK,
    apply_style,
    fig_confusion_matrix,
    load_indian_pines,
    stretch,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUN_DEFAULT = RESULTS / "2d_cnn" / "IP"
RUN_40 = RESULTS / "2d_cnn_ip_epochs40" / "IP"
RUN_1D_60 = RESULTS / "1d_cnn_ip_epochs60" / "IP"
ACCENT = ["#2a78d6", "#eb6834", "#1baf7a"]  # conv1 / conv2 / conv3 RF colors


def select_interior_pixel(gt: np.ndarray) -> tuple[int, int]:
    """Largest same-class 25x25 neighborhood inside class 11 (same as ch04)."""
    purity = np.zeros_like(gt, dtype=np.float64)
    mask = (gt == 11).astype(np.float64)
    purity = mask * uniform_filter(mask, size=25, mode="constant", cval=0.0)
    return np.unravel_index(np.argmax(purity), gt.shape)


def select_boundary_pixel(gt: np.ndarray) -> tuple[int, int]:
    """Smallest same-class 25x25 neighborhood among labeled pixels (ch04's B)."""
    purity = np.zeros_like(gt, dtype=np.float64)
    for c in range(1, 17):
        mask = (gt == c).astype(np.float64)
        purity += mask * uniform_filter(mask, size=25, mode="constant", cval=0.0)
    purity[gt == 0] = 1.0
    return np.unravel_index(np.argmin(purity), gt.shape)


def fig_patch_and_rf(cube: np.ndarray, gt: np.ndarray) -> None:
    # Actual PCA channel 0 (whiten, whole-cube fit — the training pipeline's own
    # preprocessing), so the displayed patch is exactly what the model sees.
    from sklearn.decomposition import PCA

    pca = PCA(n_components=12, whiten=True)
    reduced = pca.fit_transform(cube.reshape(-1, cube.shape[2]).astype(np.float32))
    pca0 = stretch(reduced[:, 0].reshape(cube.shape[0], cube.shape[1]))
    r, c = select_boundary_pixel(gt)
    pad = 12
    padded = np.pad(pca0, pad)
    patch25 = padded[r: r + 25, c: c + 25]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)

    ax = axes[0]
    p9 = patch25[8:17, 8:17]  # centered 9x9
    ax.imshow(p9, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("模型实际输入：9×9 patch\n（边界像元，PCA 通道 0）", fontsize=10)
    ax.set_xticks(np.arange(-0.5, 9.5, 1))
    ax.set_yticks(np.arange(-0.5, 9.5, 1))
    ax.grid(color=GRID, linewidth=0.5)
    ax.tick_params(labelbottom=False, labelleft=False)

    ax = axes[1]
    ax.imshow(patch25, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.add_patch(Rectangle((8, 8), 9, 9, fill=False, edgecolor="#eb6834", linewidth=1.8))
    ax.set_title("同一像元的 25×25 patch（橙框 = 9×9）", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    ax.set_facecolor("#f7f9fd")
    ax.add_patch(Rectangle((0, 0), 9, 9, facecolor="#b9b8b1", edgecolor="none"))
    for size, color, ls in [(3, ACCENT[0], "-"), (7, ACCENT[1], "-"), (11, ACCENT[2], "--")]:
        half = size // 2
        ax.add_patch(Rectangle((4 - half, 4 - half), size, size, fill=False,
                               edgecolor=color, linewidth=2.2, linestyle=ls))
    ax.plot(4, 4, marker="+", markersize=12, markeredgewidth=1.6, color=INK, linestyle="none")
    ax.set_title("感受野增长：3×3 → 7×7 → 11×11\n（虚线 = 第三层已超出 9×9 patch）", fontsize=9.5)
    ax.set_xlim(-1.8, 9.8)
    ax.set_ylim(9.8, -1.8)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.grid(color="#ffffff", linewidth=0.6)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_aspect("equal")

    fig.savefig(ASSETS / "ch06-patch-and-receptive-field.png", dpi=200)
    plt.close(fig)


def fig_training_curves(history: dict, best_epoch: int) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    ax_loss.plot(epochs, history["train_loss"], color="#2a78d6", linewidth=2.0,
                 marker="o", markersize=4)
    ax_loss.set_title("训练损失（10 epochs 默认配置）", fontsize=10)
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("CrossEntropy Loss", fontsize=9)

    ax_acc.plot(epochs, history["train_acc"], color="#2a78d6", linewidth=2.0,
                marker="o", markersize=4, label="train_acc")
    ax_acc.plot(epochs, history["val_acc"], color="#eb6834", linewidth=2.0,
                marker="s", markersize=4, label="val_acc")
    ax_acc.axvline(best_epoch, color="#898781", linewidth=1.0, linestyle="--")
    ax_acc.annotate(f"best epoch = {best_epoch}/10\n（仍在上升 = 训练不足）",
                    xy=(best_epoch, history["val_acc"][best_epoch - 1]),
                    xytext=(best_epoch - 4.6, 0.45), fontsize=8.5,
                    arrowprops={"arrowstyle": "->", "color": "#898781", "lw": 0.9})
    ax_acc.set_title("准确率曲线", fontsize=10)
    ax_acc.set_xlabel("Epoch", fontsize=9)
    ax_acc.set_ylabel("Accuracy", fontsize=9)
    ax_acc.legend(fontsize=9, frameon=False, loc="upper left")

    for ax in (ax_loss, ax_acc):
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(ASSETS / "ch06-training-curves.png", dpi=200)
    plt.close(fig)


def fig_maps(gt: np.ndarray, map_1d: np.ndarray, map_2d: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), constrained_layout=True)
    panels = [
        ("Ground Truth", gt),
        ("1D CNN（60 epochs，无空间信息）", map_1d),
        ("2D CNN（40 epochs，9×9 patch）", map_2d),
    ]
    for ax, (title, m) in zip(axes, panels):
        ax.imshow(m * (gt != 0), cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(ASSETS / "ch06-1d-vs-2d-maps.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()

    hist_default = json.loads((RUN_DEFAULT / "training_history.json").read_text(encoding="utf-8"))
    metrics_default = json.loads((RUN_DEFAULT / "metrics.json").read_text(encoding="utf-8"))
    metrics_40 = json.loads((RUN_40 / "metrics.json").read_text(encoding="utf-8"))
    cm_40 = np.load(RUN_40 / "confusion_matrix.npy")
    map_1d = np.load(RUN_1D_60 / "prediction_map.npy")
    map_2d = np.load(RUN_40 / "prediction_map.npy")

    fig_patch_and_rf(cube, gt)
    fig_training_curves(hist_default, int(metrics_default["best_epoch"]))
    fig_confusion_matrix(cm_40, ASSETS / "ch06-confusion-matrix.png")
    fig_maps(gt, map_1d, map_2d)

    print(f"2D CNN 10 epochs (notebook default): OA={metrics_default['oa'] * 100:.2f}  "
          f"AA={metrics_default['aa'] * 100:.2f}  Kappa={metrics_default['kappa']:.4f}  "
          f"best_epoch={metrics_default['best_epoch']}")
    print(f"2D CNN 40 epochs: OA={metrics_40['oa'] * 100:.2f}  AA={metrics_40['aa'] * 100:.2f}  "
          f"Kappa={metrics_40['kappa']:.4f}  best_epoch={metrics_40['best_epoch']}")

    recall_1d = np.diag(np.load(RUN_1D_60 / "confusion_matrix.npy")) / np.load(
        RUN_1D_60 / "confusion_matrix.npy").sum(axis=1)
    recall_2d = np.diag(cm_40) / cm_40.sum(axis=1)
    print("\nper-class recall (1D 60ep vs 2D 40ep):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i + 1:>2}  {name:<32}{recall_1d[i]:.3f}  ->  {recall_2d[i]:.3f}")
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
