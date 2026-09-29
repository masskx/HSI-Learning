"""Generate Chapter 16 & 17 figures.

    ch16: K-shot accuracy curve + prototype t-SNE
    ch17: MSP/distance distributions + ROC curves

Usage:
    python scripts/generate_ch16_figures.py
    python scripts/generate_ch17_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _chfigure_utils import ASSETS, CLASS_NAMES, GRID, INK, apply_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def fig_kshot_curve() -> None:
    """Protonet accuracy vs K-shot (1, 5, 10)."""
    shots = [1, 5, 10]
    accs = []
    for k in shots:
        p = RESULTS / "protonet" / "IP" / f"protonet_5way_{k}shot.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        accs.append(d["final_accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(shots, accs, marker="o", markersize=10, color="#2a78d6", linewidth=2.5)
    for s, a in zip(shots, accs):
        ax.annotate(f"{a:.1f}%", (s, a), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("K（每类支持集样本数）", fontsize=11)
    ax.set_ylabel("5-way 精度（%）", fontsize=11)
    ax.set_xticks(shots, [f"1-shot\n(5 样本)", f"5-shot\n(25 样本)", f"10-shot\n(50 样本)"], fontsize=10)
    ax.set_ylim(80, 97)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("原型网络 K-shot 曲线（5-way，IP，200 episodes 平均）", fontsize=11)
    ax.annotate("1→5 大幅提升\n（原型估计趋于稳定）", xy=(5, accs[1]), xytext=(2.2, accs[1] + 2),
                fontsize=9, color="#898781",
                arrowprops={"arrowstyle": "->", "color": "#898781", "lw": 0.9})
    ax.annotate("5→10 边际递减\n（原型已稳定）", xy=(10, accs[2]), xytext=(7.2, accs[2] + 2),
                fontsize=9, color="#898781",
                arrowprops={"arrowstyle": "->", "color": "#898781", "lw": 0.9})
    fig.savefig(ASSETS / "ch16-kshot-curve.png", dpi=200)
    plt.close(fig)


def fig_openset() -> None:
    """MSP/distance distributions + ROC curves for open-set detection."""
    msp = np.load(RESULTS / "openset" / "IP" / "msp_scores.npy")
    dist = np.load(RESULTS / "openset" / "IP" / "min_distances.npy")
    is_known = np.load(RESULTS / "openset" / "IP" / "is_known_test.npy")
    results = json.loads((RESULTS / "openset" / "IP" / "openset_results.json").read_text(encoding="utf-8"))

    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    # MSP distribution
    ax = axes[0]
    ax.hist(msp[is_known], bins=40, alpha=0.6, color="#2a78d6", label="已知类", density=True)
    ax.hist(msp[~is_known], bins=20, alpha=0.6, color="#eb6834", label="未知类", density=True)
    ax.set_xlabel("MSP（最大 softmax 概率）", fontsize=9)
    ax.set_ylabel("密度", fontsize=9)
    ax.set_title(f"MSP 分布（AUROC = {results['auroc_msp']:.2f}）", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # distance distribution
    ax = axes[1]
    ax.hist(dist[is_known], bins=40, alpha=0.6, color="#2a78d6", label="已知类", density=True)
    ax.hist(dist[~is_known], bins=20, alpha=0.6, color="#eb6834", label="未知类", density=True)
    ax.set_xlabel("到最近原型的距离", fontsize=9)
    ax.set_ylabel("密度", fontsize=9)
    ax.set_title(f"距离分布（AUROC = {results['auroc_distance']:.2f}）", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # ROC curves
    ax = axes[2]
    for name, scores, color in [("MSP", -msp, "#2a78d6"), ("距离", dist, "#eb6834")]:
        fpr, tpr, _ = roc_curve(~is_known, scores)
        auc = results["auroc_msp"] if name == "MSP" else results["auroc_distance"]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], color="#898781", linewidth=1, linestyle="--", label="随机")
    ax.set_xlabel("假正率 (FPR)", fontsize=9)
    ax.set_ylabel("真正率 (TPR)", fontsize=9)
    ax.set_title("ROC 曲线", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("开集检测：MSP vs 嵌入距离（已知 12 类，未知 4 类）", fontsize=11)
    fig.savefig(ASSETS / "ch17-openset.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_kshot_curve()
    fig_openset()
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
