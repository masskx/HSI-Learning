"""Generate Chapter 12 figures: the multi-seed ablation study.

Reads results/ch12_ablation_summary.json (produced by
scripts/aggregate_ch12_ablation.py) and the per-run training histories in
results/ch12/, and produces:

    1. ch12-ablation-bars.png  — OA/AA of the three configs with std error
                                 bars (the mean ± std of 3 seeds).
    2. ch12-seed-variance.png  — per-seed validation curves, showing the
                                 run-to-run variance that single-seed reports
                                 hide.

Run the 9 background trainings + scripts/aggregate_ch12_ablation.py first.

Usage:
    python scripts/generate_ch12_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _chfigure_utils import ASSETS, GRID, apply_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = [1334, 1335, 1336]
CONFIGS = [
    ("sir0", "SIR λ=0（基线）", "#2a78d6"),
    ("sir01", "SIR λ=0.1", "#eb6834"),
    ("nores", "去残差（顺序堆叠）", "#eda100"),
]


def load_histories(tag: str) -> list[dict]:
    hists = []
    for seed in SEEDS:
        path = RESULTS / "ch12" / f"{tag}_seed{seed}" / "IP" / "training_history.json"
        hists.append(json.loads(path.read_text(encoding="utf-8")))
    return hists


def fig_ablation_bars(summary: dict) -> None:
    labels = [label for _, label, _ in CONFIGS]
    oa_m = [summary[l]["oa_mean"] for l in labels]
    oa_s = [summary[l]["oa_std"] for l in labels]
    aa_m = [summary[l]["aa_mean"] for l in labels]
    aa_s = [summary[l]["aa_std"] for l in labels]
    colors = [color for _, _, color in CONFIGS]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    ax.bar(x - w / 2, oa_m, w, yerr=oa_s, capsize=4, color="#2a78d6", label="OA")
    ax.bar(x + w / 2, aa_m, w, yerr=aa_s, capsize=4, color="#eb6834", label="AA")
    for xi, m, s in zip(x - w / 2, oa_m, oa_s):
        ax.text(xi, m + s + 0.005, f"{m * 100:.2f}", ha="center", va="bottom", fontsize=8.5)
    for xi, m, s in zip(x + w / 2, aa_m, aa_s):
        ax.text(xi, m + s + 0.005, f"{m * 100:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("准确率（3 种子均值 ± 标准差）", fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(ASSETS / "ch12-ablation-bars.png", dpi=200)
    plt.close(fig)


def fig_seed_variance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True,
                             sharey=True)
    panels = [("SIR 消融：λ=0（实线）vs λ=0.1（虚线）", "sir0", "sir01"),
              ("残差消融：完整（实线）vs 去残差（虚线）", "sir0", "nores")]
    seed_shades = ["#1c5cab", "#2a78d6", "#86b6ef"]
    seed_shades_warm = ["#a53f12", "#eb6834", "#f6a97a"]
    for ax, (title, tag_solid, tag_dash) in zip(axes, panels):
        for i, hist in enumerate(load_histories(tag_solid)):
            epochs = np.arange(1, len(hist["val_acc"]) + 1)
            ax.plot(epochs, hist["val_acc"], color=seed_shades[i], linewidth=1.8,
                    label=f"seed {SEEDS[i]}")
        for i, hist in enumerate(load_histories(tag_dash)):
            epochs = np.arange(1, len(hist["val_acc"]) + 1)
            ax.plot(epochs, hist["val_acc"], color=seed_shades_warm[i], linewidth=1.8,
                    linestyle="--", label=f"seed {SEEDS[i]}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("验证集准确率", fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
    fig.savefig(ASSETS / "ch12-seed-variance.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    summary = json.loads(
        (RESULTS / "ch12_ablation_summary.json").read_text(encoding="utf-8")
    )
    fig_ablation_bars(summary)
    fig_seed_variance()
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
