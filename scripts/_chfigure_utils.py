"""Shared helpers for chapter figure scripts.

Deliberately minimal dependencies (numpy / scipy / matplotlib / scikit-learn
only — no torch), so chapter figures can be regenerated in a lightweight
environment. Dataset constants are kept in sync with
``src/hsi_learning/data.py`` (DATASET_SPECS) and split semantics mirror
``sample_ground_truth`` / ``split_ground_truth``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "chapters" / "assets"

# Kept in sync with DATASET_SPECS["IP"] in src/hsi_learning/data.py.
CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]

# Chart ink/grid chrome (dataviz reference palette).
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GRAY_SERIES = "#d0cfc8"
SURFACE = "#ffffff"

# Sequential single-hue ramp (light -> dark blue) for magnitude encodings.
SEQ_BLUES = [
    "#f7f9fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6",
    "#256abf", "#1c5cab", "#104281", "#0d366b",
]


def apply_style() -> None:
    """Apply the shared matplotlib style (CJK-capable font, light surface).

    The font list falls back across Windows / macOS / Linux CJK fonts so the
    Chinese figure labels render everywhere; the first available match wins,
    so Windows rendering is unchanged.
    """
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",     # Windows
                "SimHei",              # Windows (older)
                "PingFang SC",         # macOS
                "Hiragino Sans GB",    # macOS (older)
                "Noto Sans CJK SC",    # Linux
                "WenQuanYi Zen Hei",   # Linux (older)
                "Arial",               # last-resort Latin fallback
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )


def load_indian_pines() -> tuple[np.ndarray, np.ndarray]:
    """Load the Indian Pines cube (float32) and ground truth (int64)."""
    cube = loadmat(ROOT / "dataset" / "Indian_pines_corrected.mat")["indian_pines_corrected"]
    gt = loadmat(ROOT / "dataset" / "Indian_pines_gt.mat")["indian_pines_gt"]
    return np.asarray(cube, np.float32), np.asarray(gt, np.int64)


def stretch(img: np.ndarray, low: float = 2.0, high: float = 99.5) -> np.ndarray:
    """Percentile stretch for display."""
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _stratified_two_way(source_gt: np.ndarray, rate: float, random_state: int):
    """Stratified split of labeled pixels into two label maps (mirror of
    ``sample_ground_truth`` in src/hsi_learning/data.py)."""
    rows, cols = np.nonzero(source_gt)
    labels = source_gt[rows, cols]
    part_a, part_b = train_test_split(
        np.arange(len(rows)),
        train_size=rate,
        stratify=labels,
        random_state=random_state,
    )
    gt_a = np.zeros_like(source_gt)
    gt_b = np.zeros_like(source_gt)
    gt_a[rows[part_a], cols[part_a]] = labels[part_a]
    gt_b[rows[part_b], cols[part_b]] = labels[part_b]
    return gt_a, gt_b


def two_step_split(gt: np.ndarray, train_rate: float, val_rate: float, random_state: int = 100):
    """Mirror of ``split_ground_truth``: stratified train first, then split the
    remainder into val/test so that val holds ``val_rate`` of the total."""
    train_gt, rest_gt = _stratified_two_way(gt, train_rate, random_state)
    if val_rate == 0:
        return train_gt, np.zeros_like(gt), rest_gt
    val_gt, test_gt = _stratified_two_way(rest_gt, val_rate / (1 - train_rate), random_state)
    return train_gt, val_gt, test_gt


def fig_confusion_matrix(cm: np.ndarray, out_path: str | Path) -> None:
    """Row-normalized confusion matrix (recall view) with raw-count annotations."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    n = cm.shape[0]
    row_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9.6, 8.4), constrained_layout=True)
    im = ax.imshow(row_norm, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ids = np.arange(1, n + 1)
    ax.set_xticks(ids - 1, labels=ids, fontsize=8)
    ax.set_yticks(ids - 1, labels=ids, fontsize=8)
    ax.set_xlabel("预测类别编号", fontsize=10)
    ax.set_ylabel("真实类别编号", fontsize=10)

    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=6, color="#ffffff" if row_norm[i, j] > 0.5 else INK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("行归一化比例（召回率视角）", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
