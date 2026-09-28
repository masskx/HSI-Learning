"""Generate Chapter 5 figures: 1D CNN on Indian Pines spectra.

Reads the artifacts produced by scripts/train_1d_cnn.py (results/1d_cnn/IP)
and adds two real-data teaching figures:

    1. ch05-conv-as-filter.png    — convolution as a sliding filter over real
                                    class spectra (smoothing + difference kernel).
    2. ch05-training-curves.png   — train loss / accuracy curves of the
                                    notebook-04 run (15 epochs).
    3. ch05-confusion-matrix.png  — test-set confusion matrix of the same run.
    4. ch05-prediction-map.png    — full-image prediction vs Ground Truth.

Also prints the protocol-C SVM anchor (same split/scaler) for the benchmark
scoreboard.

Run scripts/train_1d_cnn.py first.

Usage:
    python scripts/generate_ch05_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from _chfigure_utils import (
    ASSETS,
    GRID,
    apply_style,
    fig_confusion_matrix,
    load_indian_pines,
    two_step_split,
)

HIGHLIGHT_TRIPLE = ["#2a78d6", "#eb6834", "#1baf7a"]

RESULTS = Path(__file__).resolve().parents[1] / "results" / "1d_cnn" / "IP"
HIGHLIGHT_CLASSES = [2, 6, 16]  # Corn-notill, Grass-trees, Stone-Steel-Towers
CLASS_LABELS = {2: "Corn-notill", 6: "Grass-trees", 16: "Stone-Steel-Towers"}


def fig_conv_as_filter(cube: np.ndarray, gt: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    spectra = {}
    for k, class_id in enumerate(HIGHLIGHT_CLASSES):
        spectra[class_id] = cube[gt == class_id].mean(axis=0)

    ax = axes[0]
    for k, class_id in enumerate(HIGHLIGHT_CLASSES):
        ax.plot(spectra[class_id], color=HIGHLIGHT_TRIPLE[k], linewidth=1.6,
                label=CLASS_LABELS[class_id])
    ax.set_title("原始平均光谱（DN）", fontsize=10)
    ax.set_xlabel("波段序号", fontsize=9)

    ax = axes[1]
    smooth_kernel = np.ones(9) / 9.0
    for k, class_id in enumerate(HIGHLIGHT_CLASSES):
        ax.plot(np.convolve(spectra[class_id], smooth_kernel, mode="same"),
                color=HIGHLIGHT_TRIPLE[k], linewidth=1.6)
    ax.set_title("平滑核响应（k=9 均值滤波）", fontsize=10)
    ax.set_xlabel("波段序号", fontsize=9)

    ax = axes[2]
    diff_kernel = np.array([-1.0, 0.0, 1.0])
    for k, class_id in enumerate(HIGHLIGHT_CLASSES):
        ax.plot(np.convolve(spectra[class_id], diff_kernel, mode="same"),
                color=HIGHLIGHT_TRIPLE[k], linewidth=1.4)
    ax.axvspan(25, 35, color="#f0efec", zorder=0)
    ax.text(30, ax.get_ylim()[1] * 0.9, "红边区", ha="center", fontsize=8.5, color="#898781")
    ax.set_title("差分核响应（[-1, 0, 1]，即一阶差分）", fontsize=10)
    ax.set_xlabel("波段序号", fontsize=9)

    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(ASSETS / "ch05-conv-as-filter.png", dpi=200)
    plt.close(fig)


def fig_training_curves(history: dict, best_epoch: int) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    ax_loss.plot(epochs, history["train_loss"], color="#2a78d6", linewidth=2.0, marker="o", markersize=3.5)
    ax_loss.set_title("训练损失", fontsize=10)
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("CrossEntropy Loss", fontsize=9)

    ax_acc.plot(epochs, history["train_acc"], color="#2a78d6", linewidth=2.0,
                marker="o", markersize=3.5, label="train_acc")
    ax_acc.plot(epochs, history["val_acc"], color="#eb6834", linewidth=2.0,
                marker="s", markersize=3.5, label="val_acc")
    ax_acc.axvline(best_epoch, color="#898781", linewidth=1.0, linestyle="--")
    ax_acc.annotate(f"best epoch = {best_epoch}\n（压在最后一个 epoch\n= 训练不足的信号）",
                    xy=(best_epoch, history["val_acc"][best_epoch - 1]),
                    xytext=(best_epoch - 6.5, 0.30), fontsize=8.5,
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
    fig.savefig(ASSETS / "ch05-training-curves.png", dpi=200)
    plt.close(fig)


def fig_prediction_map(gt: np.ndarray, pred_map: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
    ax1.imshow(gt, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
    ax1.set_title("Ground Truth", fontsize=10)
    ax2.imshow(pred_map * (gt != 0), cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
    ax2.set_title("1D CNN 整图预测（背景掩除）", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(ASSETS / "ch05-prediction-map.png", dpi=200)
    plt.close(fig)


def svm_protocol_c(cube: np.ndarray, gt: np.ndarray) -> None:
    """SVM anchor under the same protocol-C split and scaler (scoreboard row)."""
    train_gt, val_gt, test_gt = two_step_split(gt, 0.1, 0.1, random_state=42)
    X = cube.reshape(-1, cube.shape[2])
    y = gt.ravel()
    labeled = y > 0

    rows_tr = np.nonzero(train_gt.ravel())[0]
    rows_te = np.nonzero(test_gt.ravel())[0]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[rows_tr])
    X_te = scaler.transform(X[rows_te])

    svm = SVC(C=100, kernel="rbf", cache_size=2048).fit(X_tr, y[rows_tr])
    y_pred = svm.predict(X_te)
    labels = list(range(1, 17))
    oa = accuracy_score(y[rows_te], y_pred)
    aa = recall_score(y[rows_te], y_pred, labels=labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y[rows_te], y_pred, labels=labels)
    print(f"\nprotocol-C SVM anchor (10/10/80, seed 42, StandardScaler, C=100 rbf):")
    print(f"train: {len(rows_tr)}, val: {int(np.count_nonzero(val_gt))}, test: {len(rows_te)}")
    print(f"OA: {oa:.4f}  AA: {aa:.4f}  Kappa: {kappa:.4f}")


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()

    history = json.loads((RESULTS / "training_history.json").read_text(encoding="utf-8"))
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    cm = np.load(RESULTS / "confusion_matrix.npy")
    pred_map = np.load(RESULTS / "prediction_map.npy")
    best_epoch = int(metrics["best_epoch"])

    fig_conv_as_filter(cube, gt)
    fig_training_curves(history, best_epoch)
    fig_confusion_matrix(cm, ASSETS / "ch05-confusion-matrix.png")
    fig_prediction_map(gt, pred_map)

    print(f"headline run: epochs={len(history['train_loss'])}, best_epoch={best_epoch}, "
          f"OA={metrics['oa'] * 100:.2f}%, AA={metrics['aa'] * 100:.2f}%, "
          f"Kappa={metrics['kappa']:.4f}, params={metrics['total_params']}")
    svm_protocol_c(cube, gt)
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
