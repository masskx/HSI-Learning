"""Generate Chapter 1 figures from the Indian Pines dataset.

Deliberately minimal dependencies (numpy / scipy / matplotlib only — no torch),
so figures can be regenerated in a lightweight environment.

Outputs four figures into chapters/assets/ and prints class statistics
used in the chapter text.

Usage:
    python scripts/generate_ch01_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _chfigure_utils import (
    ASSETS,
    BASELINE,
    CLASS_NAMES,
    GRAY_SERIES,
    GRID,
    INK,
    MUTED,
    apply_style,
    load_indian_pines,
    stretch,
)

# Highlight palette (dataviz-validated, fixed slot order): blue, orange, aqua, yellow, magenta
HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# Highlighted classes (1-based ids): Corn-notill, Grass-trees, Hay-windrowed, Woods, Stone-Steel-Towers
HIGHLIGHT_CLASSES = [2, 6, 8, 14, 16]


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig_band_slices(cube: np.ndarray) -> None:
    bands = [0, 50, 100, 150]
    stretched = stretch(cube)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    for ax, b in zip(axes, bands):
        ax.imshow(stretched[:, :, b], cmap="gray")
        ax.set_title(f"波段 {b}（约 {400 + b * 10} nm）", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(ASSETS / "ch01-cube-band-slices.png", dpi=200)
    plt.close(fig)


def fig_spectral_curves(cube: np.ndarray, gt: np.ndarray, class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)

    ax.axvspan(25, 35, color="#f0efec", zorder=0)

    for class_id in range(1, len(class_names) + 1):
        spectra = cube[gt == class_id]
        mean = spectra.mean(axis=0)
        if class_id in HIGHLIGHT_CLASSES:
            color = HIGHLIGHT_COLORS[HIGHLIGHT_CLASSES.index(class_id)]
            ax.plot(mean, color=color, linewidth=2.2, label=class_names[class_id - 1], zorder=3)
        else:
            ax.plot(mean, color=GRAY_SERIES, linewidth=0.7, alpha=0.6, zorder=1)

    ax.text(30, 1.02, "红边区（约 650–750 nm）", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=9, color=MUTED)

    style_axes(ax)
    ax.set_xlabel("波段序号（间距约 10 nm，0–199 对应约 400–2400 nm）")
    ax.set_ylabel("平均光谱响应（DN，未定标）")
    ax.set_xlim(0, 199)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.95, edgecolor=GRID)
    fig.savefig(ASSETS / "ch01-spectral-curves.png", dpi=200)
    plt.close(fig)


def fig_falsecolor_and_gt(cube: np.ndarray, gt: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)

    rgb = np.stack(
        [stretch(cube[:, :, 60], 2, 98), stretch(cube[:, :, 30], 2, 98), stretch(cube[:, :, 10], 2, 98)],
        axis=2,
    )
    ax1.imshow(rgb)
    ax1.set_title("伪彩色合成（R=波段 60，G=波段 30，B=波段 10）", fontsize=11)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(gt, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
    ax2.set_title("Ground Truth 标签图（nipy_spectral）", fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar = fig.colorbar(ax2.get_images()[0], ax=ax2, fraction=0.046, pad=0.02, ticks=range(0, 17))
    cbar.set_label("类别编号（0 = 无标签背景）", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(ASSETS / "ch01-falsecolor-gt.png", dpi=200)
    plt.close(fig)


def fig_class_distribution(gt: np.ndarray, class_names: list[str]) -> None:
    counts = {class_names[c - 1]: int((gt == c).sum()) for c in range(1, len(class_names) + 1)}
    items = sorted(counts.items(), key=lambda kv: kv[1])  # smallest first -> ends bottom-up
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    ax.barh(names, values, color="#2a78d6", height=0.62)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("标注像素数")
    ax.tick_params(axis="y", labelsize=9)
    fig.savefig(ASSETS / "ch01-class-distribution.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()
    class_names = CLASS_NAMES

    fig_band_slices(cube)
    fig_spectral_curves(cube, gt, class_names)
    fig_falsecolor_and_gt(cube, gt)
    fig_class_distribution(gt, class_names)

    print(f"cube: {cube.shape}, dtype float32, DN range [{cube.min():.0f}, {cube.max():.0f}]")
    print(f"labeled pixels: {int(np.count_nonzero(gt))} / {gt.size}")
    print("\nclass counts (id, name, pixels):")
    for c in range(1, len(class_names) + 1):
        print(f"  {c:>2}  {class_names[c - 1]:<32}{int((gt == c).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
