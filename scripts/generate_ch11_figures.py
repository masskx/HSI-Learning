"""Generate Chapter 11 figures: the course's literature map.

    1. ch11-literature-timeline.png — timeline of the key references used
       across Chapters 1-10, colored by method family. This doubles as the
       "field map" of Section 11.2.

Usage:
    python scripts/generate_ch11_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _chfigure_utils import ASSETS, GRID, INK, apply_style

# (year, short label, family, venue)
PAPERS = [
    (1960, "Cohen\nKappa", "评估", "EPM"),
    (1968, "Hughes\n现象", "传统", "TIT"),
    (1985, "Goetz\n成像光谱", "传统", "Science"),
    (1991, "Congalton\n精度评估", "评估", "RSE"),
    (2002, "Landgrebe\n高光谱综述", "传统", "IEEE SPM"),
    (2004, "SVM\nMelgani & Bruzzone", "传统", "TGRS"),
    (2015, "1D CNN\nHu et al.", "卷积", "JSTARS"),
    (2015, "2D CNN\nMakantasis", "卷积", "IGARSS"),
    (2016, "3D CNN\nChen et al.", "卷积", "TGRS"),
    (2016, "ResNet\nHe et al.", "残差", "CVPR"),
    (2017, "Transformer\nVaswani", "注意力", "NeurIPS"),
    (2018, "SSRN\nZhong et al.", "残差", "TGRS"),
    (2019, "DL 综述\nLi et al.", "综述", "TGRS"),
    (2020, "HybridSN\nRoy et al.", "混合", "GRSL"),
    (2021, "ViT\nDosovitskiy", "注意力", "ICLR"),
    (2022, "SpectralFormer\nHong et al.", "注意力", "TGRS"),
    (2024, "SS-Mamba\n等线性 SSM", "前沿", "arXiv/期刊"),
]

FAMILIES = {
    "传统": "#898781",
    "评估": "#4a3aa7",
    "综述": "#e87ba4",
    "卷积": "#2a78d6",
    "混合": "#eda100",
    "残差": "#008300",
    "注意力": "#eb6834",
    "前沿": "#1baf7a",
}

LANES = ["传统", "评估", "综述", "卷积", "混合", "残差", "注意力", "前沿"]
LANE_Y = {name: i for i, name in enumerate(LANES)}


def fig_timeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2), constrained_layout=True)
    for name, y in LANE_Y.items():
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)

    # stagger labels within each lane (alternate above/below) to avoid collisions
    lane_counter: dict[str, int] = {name: 0 for name in LANE_Y}
    # manual x-offsets for the few remaining near-collisions
    dx_overrides = {
        "1D CNN\nHu et al.": -14,
        "2D CNN\nMakantasis": 16,
        "SSRN\nZhong et al.": -12,
        "HybridSN\nRoy et al.": 18,
    }
    for year, label, family, venue in PAPERS:
        y = LANE_Y[family]
        ax.plot(year, y, marker="o", markersize=9, color=FAMILIES[family],
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        idx = lane_counter[family]
        lane_counter[family] += 1
        if idx % 2 == 0:
            dy, va = 11, "bottom"
        else:
            dy, va = -13, "top"
        dx = dx_overrides.get(label, 0)
        ax.annotate(f"{label}\n{venue}", (year, y), xytext=(dx, dy),
                    textcoords="offset points", ha="center", va=va, fontsize=7.2,
                    color=INK, zorder=4)

    ax.set_yticks(range(len(LANES)), LANES, fontsize=9)
    ax.set_xlim(1956, 2028)
    ax.set_ylim(-1.15, len(LANES) - 0.35)
    ax.set_xlabel("发表年份", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("本课程参考文献地图（第 1–10 章引用的核心文献，按方法家族分道）", fontsize=11)
    fig.savefig(ASSETS / "ch11-literature-timeline.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_timeline()
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
