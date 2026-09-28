"""Generate Chapter 14 figure: the capstone six-stage pipeline.

    1. ch14-capstone-pipeline.png — the six stages of the capstone with each
       stage's deliverable.

Usage:
    python scripts/generate_ch14_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _chfigure_utils import ASSETS, GRID, INK, apply_style

STAGES = [
    ("① 选题开题", "gap → 假设\n立项书", "#2a78d6"),
    ("② 基线复现", "协议对齐的\n基线数字", "#2a78d6"),
    ("③ 改进实现", "模块 + 开关\n(可消融)", "#1baf7a"),
    ("④ 实验消融", "3 数据集 ×\n3 种子 × 消融", "#1baf7a"),
    ("⑤ 论文撰写", "IEEEtran\n骨架填满", "#eb6834"),
    ("⑥ 模拟投稿", "自查 + 互评\n+ rebuttal 演练", "#eb6834"),
]


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.6), constrained_layout=True)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    width, height, gap = 2.15, 1.5, 0.35
    x = 0.2
    centers = []
    for title, deliverable, color in STAGES:
        ax.add_patch(FancyBboxPatch((x, 2.3), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=color, edgecolor="#c3c2b7", linewidth=1.0))
        ax.text(x + width / 2, 2.3 + height * 0.68, title, ha="center", va="center",
                fontsize=11, color="#ffffff", fontweight="bold")
        ax.text(x + width / 2, 2.3 + height * 0.28, deliverable, ha="center", va="center",
                fontsize=8.2, color="#ffffff")
        centers.append(x + width / 2)
        x += width + gap
    for i in range(len(STAGES) - 1):
        x0 = centers[i] + width / 2 + 0.03
        x1 = centers[i + 1] - width / 2 - 0.03
        ax.add_patch(FancyArrowPatch((x0, 2.3 + height / 2), (x1, 2.3 + height / 2),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color="#898781", linewidth=1.6))

    # iteration loop: ⑥ back to ③ (review findings loop back to implementation)
    ax.add_patch(FancyArrowPatch((centers[-1], 2.3 - 0.12), (centers[2], 2.3 - 0.12),
                                 connectionstyle="arc3,rad=0.25", arrowstyle="-|>",
                                 mutation_scale=13, color="#898781", linewidth=1.4,
                                 linestyle="--"))
    ax.text((centers[2] + centers[-1]) / 2, 0.85, "返修循环：互评/模拟审稿意见回到实现与实验",
            fontsize=9, color="#898781", ha="center")

    ax.text(0.2, 4.35, "每阶段的完成标准 = 该阶段交付物可被他人独立检验", fontsize=10, color=INK)
    fig.savefig(ASSETS / "ch14-capstone-pipeline.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
