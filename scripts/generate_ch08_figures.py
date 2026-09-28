"""Generate Chapter 8 figures: HybridSN (notebook-03 / train_hybridsn.py).

Reads artifacts from results/hybridsn/IP (protocol-B default run of
scripts/train_hybridsn.py) and produces:

    1. ch08-shape-flow.png     — the HybridSN data flow with real shapes,
                                 highlighting the 3D->2D reshape moment.
    2. ch08-training-curves.png — protocol-B training curves (val every 5).
    3. ch08-confusion-matrix.png — test-set confusion matrix, reconstructed
                                   from the saved full-image prediction map.
    4. ch08-prediction-map.png   — full-image prediction vs Ground Truth.

Also prints the per-layer parameter table and the analytic MACs of HybridSN.

Run scripts/train_hybridsn.py --dataset IP --epochs 20 first.

Usage:
    python scripts/generate_ch08_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _chfigure_utils import (  # noqa: E402
    ASSETS,
    GRID,
    INK,
    apply_style,
    fig_confusion_matrix,
    load_indian_pines,
)
from hsi_learning.data import split_ground_truth  # noqa: E402
from hsi_learning.models import HybridSN  # noqa: E402

RESULTS = ROOT / "results"
RUN = RESULTS / "hybridsn" / "IP"
BOX = "#2a78d6"
BOX2 = "#eb6834"
HIGHLIGHT = "#eda100"


def fig_shape_flow() -> None:
    """Data-flow diagram with real shapes; the 3D->2D reshape is highlighted."""
    stages_top = [
        ("输入\n(1, 15, 25, 25)", BOX),
        ("Conv3d 8×(7,3,3)\n+ReLU", BOX),
        ("(8, 9, 23, 23)", None),
        ("Conv3d 16×(5,3,3)\n+ReLU", BOX),
        ("(16, 5, 21, 21)", None),
        ("Conv3d 32×(3,3,3)\n+ReLU", BOX),
        ("(32, 3, 19, 19)", None),
    ]
    stages_bottom = [
        ("view → (96, 19, 19)\n把 (C, D) 合并为通道", HIGHLIGHT),
        ("Conv2d 64×(3,3)\n+ReLU", BOX2),
        ("(64, 17, 17)", None),
        ("Flatten\n18,496", None),
        ("FC 256\nDropout 0.4", BOX2),
        ("FC 128\nDropout 0.4", BOX2),
        ("FC 16\n(类别)", BOX2),
    ]

    fig, ax = plt.subplots(figsize=(15.5, 4.6), constrained_layout=True)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def draw_row(stages, y, row_label):
        width, height, gap = 1.9, 1.05, 0.35
        centers = []
        x = 0.25
        for text, color in stages:
            face = color if color else "#f0efec"
            tcolor = "#ffffff" if color in (BOX, BOX2) else INK
            box = FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=face, edgecolor="#c3c2b7", linewidth=1.0)
            ax.add_patch(box)
            ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                    fontsize=8.2, color=tcolor)
            centers.append(x + width / 2)
            x += width + gap
        for i in range(len(stages) - 1):
            x0 = centers[i] + width / 2 + 0.03
            x1 = centers[i + 1] - width / 2 - 0.03
            ax.add_patch(FancyArrowPatch((x0, y + height / 2), (x1, y + height / 2),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#898781", linewidth=1.2))
        ax.text(0.02, y + height + 0.18, row_label, fontsize=9.5, color=INK,
                fontweight="bold")

    draw_row(stages_top, 2.55, "3D 阶段：谱空联合（第 7 章的设计，三层后光谱维耗尽）")
    draw_row(stages_bottom, 0.55, "2D 阶段：把 D′ 并入通道，用标准图像卷积精炼空间")

    # connector between rows
    ax.add_patch(FancyArrowPatch((14.2, 2.5), (1.2, 1.75),
                                 connectionstyle="arc3,rad=0.25",
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=HIGHLIGHT, linewidth=1.8, linestyle="--"))
    fig.savefig(ASSETS / "ch08-shape-flow.png", dpi=200)
    plt.close(fig)


def layer_param_table(model: nn.Module, in_shape: tuple[int, ...]):
    rows = []

    def hook(name):
        def fn(mod, inp, out):
            rows.append(
                {
                    "layer": name,
                    "type": mod.__class__.__name__,
                    "out_shape": list(out.shape),
                    "params": sum(p.numel() for p in mod.parameters()),
                }
            )
        return fn

    handles = [
        mod.register_forward_hook(hook(name))
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Conv3d, nn.Linear, nn.Flatten, nn.Dropout))
    ]
    with torch.no_grad():
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return rows


def count_macs(model: nn.Module, in_shape: tuple[int, ...]) -> int:
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
        model(torch.zeros(in_shape))
    for handle in handles:
        handle.remove()
    return total["v"]


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()

    history = json.loads((RUN / "training_history.json").read_text(encoding="utf-8"))
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    pred_map = np.load(RUN / "prediction.npy")

    # ---- 图 8-1 shape flow --------------------------------------------------
    fig_shape_flow()

    # ---- 表 8-1 layer table + MACs ------------------------------------------
    model = HybridSN(input_channels=15, patch_size=25, num_classes=16)
    layers = layer_param_table(model, (1, 15, 25, 25))
    macs = count_macs(model, (1, 15, 25, 25))
    total = sum(p.numel() for p in model.parameters())
    print("layer table (HybridSN, input (1,1,15,25,25)):")
    for row in layers:
        print(f"  {row['layer']:<12}{row['type']:<10}{str(tuple(row['out_shape'])):<22}{row['params']}")
    print(f"total params: {total:,}   MACs/sample: {macs / 1e6:.2f}M")

    # ---- 图 8-2 training curves ---------------------------------------------
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    val_epochs = history["val_epoch"]
    val_accs = history["val_acc"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax_loss.plot(epochs, history["train_loss"], color="#2a78d6", linewidth=2.0,
                 marker="o", markersize=4)
    ax_loss.set_title("训练损失（协议 B：30/10/60，seed 63466）", fontsize=10)
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("CrossEntropy Loss", fontsize=9)

    ax_acc.plot(epochs, history["train_acc"], color="#2a78d6", linewidth=2.0,
                marker="o", markersize=4, label="train_acc")
    ax_acc.plot(val_epochs, val_accs, color="#eb6834", linewidth=2.0,
                marker="s", markersize=5, label="val_acc（每 5 epochs）")
    best = val_epochs[int(np.argmax(val_accs))]
    ax_acc.axvline(best, color="#898781", linewidth=1.0, linestyle="--")
    ax_acc.annotate(f"best val_acc = {max(val_accs):.4f}\n(epoch {best})", xy=(best, max(val_accs)),
                    xytext=(best - 7.5, 0.80), fontsize=8.5,
                    arrowprops={"arrowstyle": "->", "color": "#898781", "lw": 0.9})
    ax_acc.set_title("准确率曲线", fontsize=10)
    ax_acc.set_xlabel("Epoch", fontsize=9)
    ax_acc.set_ylabel("Accuracy", fontsize=9)
    ax_acc.legend(fontsize=9, frameon=False, loc="lower right")
    for ax in (ax_loss, ax_acc):
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(ASSETS / "ch08-training-curves.png", dpi=200)
    plt.close(fig)

    # ---- 图 8-3 confusion matrix (reconstructed from prediction map) ---------
    train_gt, val_gt, test_gt = split_ground_truth(gt, 0.3, 0.1, random_state=63466)
    test_true = test_gt[test_gt != 0]
    test_pred = pred_map[test_gt != 0]
    labels = list(range(1, 17))
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(test_true, test_pred, labels=labels)
    fig_confusion_matrix(cm, ASSETS / "ch08-confusion-matrix.png")
    print(f"protocol-B reconstructed test samples: {len(test_true)}")

    # ---- 图 8-4 prediction map ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
    ax1.imshow(gt, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
    ax1.set_title("Ground Truth", fontsize=10)
    ax2.imshow(pred_map * (gt != 0), cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
    ax2.set_title(f"HybridSN 整图预测（协议 B，OA={metrics['oa'] * 100:.2f}%）", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(ASSETS / "ch08-prediction-map.png", dpi=200)
    plt.close(fig)

    print(f"protocol-B metrics: OA={metrics['oa'] * 100:.2f}  AA={metrics['aa'] * 100:.2f}  "
          f"Kappa={metrics['kappa']:.4f}  best_val_acc={metrics['best_val_acc']:.4f}")
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
