"""Generate Chapter 4 figures: the patch paradigm on Indian Pines.

    1. ch04-patch-neighborhood.png — what a patch is: an interior pixel vs a
       class-boundary pixel, each with its GT neighborhood and 25x25 window.
    2. ch04-patch-purity.png — distribution of same-class fraction inside a
       patch, for patch sizes 11 / 25 / 49 (zero padding mirrored).
    3. ch04-patch-tensor.png — one actual training sample: 15 PCA-whitened
       25x25 slices (the (C, H, W) tensor a model consumes).

Also prints purity statistics quoted in the chapter.

Usage:
    python scripts/generate_ch04_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA

from _chfigure_utils import (
    ASSETS,
    CLASS_NAMES,
    GRID,
    INK,
    apply_style,
    load_indian_pines,
    stretch,
)

WINDOW = 41  # zoom window size around the selected pixels
PAD = WINDOW // 2


def patch_purity(gt: np.ndarray, patch_size: int) -> np.ndarray:
    """Same-class fraction inside a zero-padded patch around every pixel."""
    purity = np.zeros_like(gt, dtype=np.float64)
    for c in range(1, 17):
        mask = (gt == c).astype(np.float64)
        frac = uniform_filter(mask, size=patch_size, mode="constant", cval=0.0)
        purity += mask * frac  # only labeled pixels keep their own class's fraction
    return purity


def fig_neighborhood(gt: np.ndarray, purity25: np.ndarray) -> None:
    interior = np.unravel_index(np.argmax(purity25 * (gt == 11)), gt.shape)
    boundary = np.unravel_index(np.argmin(purity25 + (gt == 0) * 10), gt.shape)

    gt_pad = np.pad(gt, PAD)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.imshow(gt, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
    ax.plot(interior[1], interior[0], marker="o", markersize=7, markerfacecolor="#ffffff",
            markeredgecolor=INK, markeredgewidth=1.2, linestyle="none", label="A：田块内部")
    ax.plot(boundary[1], boundary[0], marker="^", markersize=8, markerfacecolor="#eda100",
            markeredgecolor=INK, markeredgewidth=1.2, linestyle="none", label="B：类别交界")
    ax.set_title("Ground Truth（分析像元位置）", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95, edgecolor=GRID)
    ax.set_xticks([])
    ax.set_yticks([])

    for ax, (r, c), name in [
        (axes[1], interior, f"A：同类田块内部（25×25 patch 纯度 {purity25[interior]:.2f}）"),
        (axes[2], boundary, f"B：类别交界（25×25 patch 纯度 {purity25[boundary]:.2f}）"),
    ]:
        win = gt_pad[r : r + WINDOW, c : c + WINDOW]
        ax.imshow(win, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
        ax.add_patch(Rectangle((PAD - 12, PAD - 12), 25, 25, fill=False, edgecolor="#ffffff",
                               linewidth=1.8))
        ax.plot(PAD, PAD, marker="+", markersize=10, markeredgewidth=1.4,
                color="#ffffff", linestyle="none")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(ASSETS / "ch04-patch-neighborhood.png", dpi=200)
    plt.close(fig)
    return interior


def fig_purity_hist(gt: np.ndarray) -> None:
    sizes = [11, 25, 49]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(8.6, 5), constrained_layout=True)
    bins = np.linspace(0, 1, 21)
    for ps, color in zip(sizes, colors):
        purity = patch_purity(gt, ps)[gt > 0]
        ax.hist(purity, bins=bins, histtype="step", linewidth=2.0, color=color,
                label=f"patch {ps}×{ps}（均值 {purity.mean():.2f}）")
        print(f"patch {ps:>2}: mean purity {purity.mean():.4f}, "
              f">=0.9 share {(purity >= 0.9).mean() * 100:.1f}%, "
              f"<0.5 share {(purity < 0.5).mean() * 100:.1f}%")
    ax.set_xlabel("patch 内与中心像元同类的比例", fontsize=10)
    ax.set_ylabel("标注像元数", fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(ASSETS / "ch04-patch-purity.png", dpi=200)
    plt.close(fig)


def fig_patch_tensor(cube: np.ndarray, interior: tuple[int, int]) -> None:
    n_bands = cube.shape[2]
    pca = PCA(n_components=15, whiten=True).fit(cube.reshape(-1, n_bands))
    reduced = pca.transform(cube.reshape(-1, n_bands)).reshape(cube.shape[0], cube.shape[1], 15)
    pad = 12
    reduced_pad = np.pad(reduced, ((pad, pad), (pad, pad), (0, 0)))
    r, c = interior
    patch = reduced_pad[r : r + 25, c : c + 25]            # (25, 25, 15)
    patch_chw = patch.transpose(2, 0, 1)                    # (15, 25, 25)

    fig, axes = plt.subplots(3, 5, figsize=(12.5, 7.6), constrained_layout=True)
    for ax, k in zip(axes.ravel(), range(15)):
        ax.imshow(patch_chw[k], cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
        ax.set_title(f"通道 {k}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("一个真实训练样本：(15, 25, 25) 张量 —— PCA-whiten 后的 15 个通道切片（RdBu，±3σ）",
                 fontsize=11)
    fig.savefig(ASSETS / "ch04-patch-tensor.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()
    print(f"class names: {CLASS_NAMES[10]} (class 11) used for the interior example")

    purity25 = patch_purity(gt, 25)
    interior = fig_neighborhood(gt, purity25)
    fig_purity_hist(gt)
    fig_patch_tensor(cube, interior)
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
