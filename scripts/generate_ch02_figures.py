"""Generate Chapter 2 figures from the Indian Pines dataset.

Figures:
    1. Stratified two-step split visualization (30% train / 10% val / 60% test,
       seed 100 — the course-standard protocol of src/hsi_learning/data.py).
    2. Confusion matrix of an RBF-SVM under the notebook-02 protocol
       (80/20 stratified split, random_state=11, C=100, gamma='scale').

Also prints the per-class split statistics and the SVM metrics quoted in the
chapter text.

Usage:
    python scripts/generate_ch02_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from _chfigure_utils import (
    ASSETS,
    CLASS_NAMES,
    apply_style,
    fig_confusion_matrix,
    load_indian_pines,
    two_step_split,
)


def fig_split_maps(train_gt: np.ndarray, val_gt: np.ndarray, test_gt: np.ndarray) -> None:
    panels = [
        ("训练集（30%）", train_gt),
        ("验证集（10%）", val_gt),
        ("测试集（60%）", test_gt),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), constrained_layout=True)
    for ax, (title, gt_map) in zip(axes, panels):
        ax.imshow(gt_map, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(f"{title}　{int(np.count_nonzero(gt_map))} 像元", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(axes[0].get_images()[0], ax=axes, fraction=0.02, pad=0.015, ticks=range(0, 17))
    cbar.set_label("类别编号（0 = 无标签背景）", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(ASSETS / "ch02-split-visualization.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()
    class_names = CLASS_NAMES

    # ---- Figure 2-1: course-standard three-way split ------------------------
    train_gt, val_gt, test_gt = two_step_split(gt, train_rate=0.3, val_rate=0.1, random_state=100)
    fig_split_maps(train_gt, val_gt, test_gt)

    print("split report (train 30% / val 10% / test 60%, random_state=100):")
    print(f"{'id':>3}{'name':<32}{'train':>7}{'val':>7}{'test':>7}{'total':>7}")
    for c in range(1, 17):
        print(
            f"{c:>3}{class_names[c - 1]:<32}"
            f"{int((train_gt == c).sum()):>7}"
            f"{int((val_gt == c).sum()):>7}"
            f"{int((test_gt == c).sum()):>7}"
            f"{int((gt == c).sum()):>7}"
        )
    print(
        f"{'all':>35}"
        f"{int(np.count_nonzero(train_gt)):>7}"
        f"{int(np.count_nonzero(val_gt)):>7}"
        f"{int(np.count_nonzero(test_gt)):>7}"
        f"{int(np.count_nonzero(gt)):>7}"
    )

    # ---- Figure 2-2: SVM confusion matrix (notebook 02 protocol) ------------
    n_bands = cube.shape[2]
    X = cube.reshape(-1, n_bands)
    y = gt.ravel()
    labeled = y > 0

    X_train, X_test, y_train, y_test = train_test_split(
        X[labeled], y[labeled], test_size=0.20, random_state=11, stratify=y[labeled]
    )
    print(f"\nSVM protocol (notebook 02): 80/20 stratified split, random_state=11, "
          f"C=100, kernel='rbf', gamma='scale'")
    print(f"train samples: {len(y_train)}, test samples: {len(y_test)}")

    svm = SVC(C=100, kernel="rbf", cache_size=2048)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    oa = accuracy_score(y_test, y_pred)
    aa = recall_score(y_test, y_pred, labels=range(1, 17), average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred, labels=range(1, 17))
    print(f"OA: {oa:.4f}  AA: {aa:.4f}  Kappa: {kappa:.4f}")

    per_class = recall_score(y_test, y_pred, labels=range(1, 17), average=None, zero_division=0)
    print("per-class recall:")
    for c, r in enumerate(per_class, start=1):
        print(f"  {c:>2}  {class_names[c - 1]:<32}{r:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=range(1, 17))
    fig_confusion_matrix(cm, ASSETS / "ch02-confusion-matrix.png")
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
