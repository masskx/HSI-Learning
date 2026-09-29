"""Generate Chapter 15 figures: class-imbalance weapons ablation.

    1. ch15-loss-tradeoff.png  — OA vs AA bar chart (3 losses, error bars).
    2. ch15-perclass-recall.png — per-class recall: CE vs weighted vs focal,
       sorted by class frequency (rare classes at the left).

Reads results/ch15_summary.json (produced by the inline aggregation in the
session) and the per-run confusion matrices in results/ch15/.

Usage:
    python scripts/generate_ch15_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _chfigure_utils import ASSETS, CLASS_NAMES, GRID, INK, apply_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CONFIGS = [
    ("CE（基线）", "#898781"),
    ("加权 CE", "#2a78d6"),
    ("Focal γ=2", "#eb6834"),
]
# IP class frequencies (ascending) for the per-class panel
FREQ = [20, 28, 46, 93, 205, 237, 386, 478, 483, 593, 730, 830, 972, 1265, 1428, 2455]


def load_cm(tag: str, seed: int) -> np.ndarray:
    p = RESULTS / "ch15" / f"{tag}_seed{seed}" / "IP" / "confusion_matrix.npy"
    return np.load(p)


def fig_tradeoff(summary: dict) -> None:
    labels = [label for label, _ in CONFIGS]
    keys = ["CE（基线）", "加权 CE", "Focal γ=2"]
    oa_m = [summary[k]["oa_mean"] for k in keys]
    oa_s = [summary[k]["oa_std"] for k in keys]
    aa_m = [summary[k]["aa_mean"] for k in keys]
    aa_s = [summary[k]["aa_std"] for k in keys]
    colors = [color for _, color in CONFIGS]

    x = np.arange(3)
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    ax.bar(x - w / 2, oa_m, w, yerr=oa_s, capsize=4, color="#2a78d6", label="OA")
    ax.bar(x + w / 2, aa_m, w, yerr=aa_s, capsize=4, color="#eb6834", label="AA")
    for xi, m, s in zip(x - w / 2, oa_m, oa_s):
        ax.text(xi, m + s + 0.3, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, m, s in zip(x + w / 2, aa_m, aa_s):
        ax.text(xi, m + s + 0.3, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    # annotate the AA delta
    ax.annotate("", xy=(x[1] + w / 2, aa_m[1]), xytext=(x[0] + w / 2, aa_m[0]),
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.4,
                            "connectionstyle": "arc3,rad=-0.3"})
    ax.text(x[0] + 0.65, 92.5, "AA +8.3", fontsize=10, color=INK, fontweight="bold")
    ax.text(x[0] + 0.65, 91.0, "OA −2.0", fontsize=9, color="#898781")
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(60, 102)
    ax.set_ylabel("准确率（3 种子均值 ± 标准差）", fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("类别不均衡武器消融（协议 C，IP，2D CNN，40 epochs）", fontsize=11)
    fig.savefig(ASSETS / "ch15-loss-tradeoff.png", dpi=200)
    plt.close(fig)


def fig_perclass_recall() -> None:
    tags = ["ce", "weighted", "focal"]
    seeds = [42, 1334, 1335]
    colors = ["#898781", "#2a78d6", "#eb6834"]
    labels = ["CE", "加权 CE", "Focal γ=2"]

    # average per-class recall across seeds, ordered by class frequency (ascending)
    recalls = {}
    for tag in tags:
        cms = [load_cm(tag, s) for s in seeds]
        cm = np.mean(cms, axis=0)
        rec = np.diag(cm) / cm.sum(axis=1)
        order = np.argsort(FREQ)  # rare classes first
        recalls[tag] = rec[order]

    sorted_names = [CLASS_NAMES[i] for i in np.argsort(FREQ)]
    sorted_freq = np.sort(FREQ)
    x = np.arange(16)

    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    for tag, color, label in zip(tags, colors, labels):
        ax.plot(x, recalls[tag] * 100, marker="o", markersize=5, linewidth=1.8,
                color=color, label=label)
    ax.set_xticks(x, [f"{n}\n({f})" for n, f in zip(sorted_names, sorted_freq)],
                  fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("逐类召回率（%，3 种子平均）", fontsize=10)
    ax.set_xlabel("类别（按标注样本数升序，稀有类在左）", fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.axhline(100, color=GRID, linewidth=0.6, linestyle=":")
    ax.set_title("逐类召回率 vs 类别频率：加权 CE 对稀有类的提升一目了然", fontsize=11)
    fig.savefig(ASSETS / "ch15-perclass-recall.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "ch15_summary.json").read_text(encoding="utf-8"))
    fig_tradeoff(summary)
    fig_perclass_recall()
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
