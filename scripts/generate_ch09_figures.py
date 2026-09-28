"""Generate Chapter 9 figures: teaching SSRN on Indian Pines.

Reads artifacts from
    results/ssrn/IP                (notebook-08 default run, 10 epochs)
    results/ssrn_ip_40ep/IP        (40-epoch control, no SIR)
    results/ssrn_ip_40ep_sir/IP    (40-epoch + spectral invariance regularisation)
    results/ssrn_ip_protocolC/IP   (protocol-C run for the scoreboard)
and produces:

    1. ch09-ssrn-flow.png   — the two-branch structure with real shapes and
                              the residual shortcuts.
    2. ch09-sir-ablation.png — validation curves + test metrics of the SIR
                               ablation (lambda = 0 vs 0.1, 40 epochs).
    3. ch09-confusion-matrix.png — confusion matrix of the 40-epoch control.

Run scripts/train_ssrn.py (+ the three control runs) first.

Usage:
    python scripts/generate_ch09_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _chfigure_utils import ASSETS, GRID, INK, apply_style, fig_confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN_DEFAULT = RESULTS / "ssrn" / "IP"
RUN_40 = RESULTS / "ssrn_ip_40ep" / "IP"
RUN_SIR = RESULTS / "ssrn_ip_40ep_sir" / "IP"
RUN_C = RESULTS / "ssrn_ip_protocolC" / "IP"
SPECTRAL = "#2a78d6"
SPATIAL = "#eb6834"
HIGHLIGHT = "#eda100"


def fig_ssrn_flow(shapes: dict) -> None:
    """Two-branch structure diagram with real shapes and residual shortcuts."""
    top = [
        ("输入\n(1, 7, 7, 200)", SPECTRAL),
        ("Conv3d 24×(1,1,7)\nstride 2 + BN + ReLU", SPECTRAL),
        ("(24, 7, 7, 98)", None),
        ("SpectralResBlock\n2×Conv3d(1,1,7)\n+ shortcut", SPECTRAL),
        ("(24, 7, 7, 98)", None),
        ("Transition\nConv3d 128×(1,1,98)", HIGHLIGHT),
        ("(128, 7, 7, 1)", None),
    ]
    bottom = [
        ("permute →\n(1, 7, 7, 128)", HIGHLIGHT),
        ("Conv3d 24×(3,3,128)\n+ BN + ReLU", SPATIAL),
        ("(24, 5, 5, 1)", None),
        ("SpatialResBlock\n2×Conv3d(3,3,1)\n+ shortcut", SPATIAL),
        ("(24, 5, 5, 1)", None),
        ("AvgPool → 24\nDropout 0.5 → FC 16", SPATIAL),
    ]

    fig, ax = plt.subplots(figsize=(15.5, 5.2), constrained_layout=True)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def draw_row(stages, y, row_label):
        width, height, gap = 1.9, 1.15, 0.28
        x = 0.25
        centers = []
        for text, color in stages:
            face = color if color else "#f0efec"
            tcolor = "#ffffff" if color in (SPECTRAL, SPATIAL) else INK
            ax.add_patch(FancyBboxPatch((x, y), width, height,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=face, edgecolor="#c3c2b7", linewidth=1.0))
            ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                    fontsize=7.6, color=tcolor)
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
        return centers, width, height

    centers_top, w, h = draw_row(top, 3.5, "谱分支：1×1×7 核沿光谱维滑动（残差跨过两层卷积）")
    centers_bot, _, _ = draw_row(bottom, 0.6, "空分支：3×3 核沿空间维滑动（残差跨过两层卷积）")

    # residual skip arcs over each res block
    ax.add_patch(FancyArrowPatch((centers_top[3] - w / 2, 3.5 + h + 0.06),
                                 (centers_top[3] + w / 2, 3.5 + h + 0.06),
                                 connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=11, color="#898781", linewidth=1.3))
    ax.add_patch(FancyArrowPatch((centers_bot[3] - w / 2, 0.6 + h + 0.06),
                                 (centers_bot[3] + w / 2, 0.6 + h + 0.06),
                                 connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=11, color="#898781", linewidth=1.3))
    ax.text((centers_top[3] + centers_top[4] + w) / 2 + 0.55, 4.98,
            "虚线弧 = shortcut（恒等族，此处为 1×1×1 Conv）", fontsize=8,
            color="#898781", ha="center")

    # branch connector routed through the empty band between the two rows
    y_mid = 2.95
    x_right = centers_top[-1] + w / 2
    x_left = centers_bot[0]
    for p0, p1, style in [
        ((x_right, 3.5), (x_right, y_mid), "-"),
        ((x_right, y_mid), (x_left, y_mid), "-"),
        ((x_left, y_mid), (x_left, 1.78), "-|>"),
    ]:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                     color=HIGHLIGHT, linewidth=1.8, linestyle="--"))
    fig.savefig(ASSETS / "ch09-ssrn-flow.png", dpi=200)
    plt.close(fig)


def fig_sir_ablation(hist_none: dict, hist_sir: dict, met_none: dict, met_sir: dict) -> None:
    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.4, 1]})
    epochs = np.arange(1, len(hist_none["val_acc"]) + 1)
    ax_c.plot(epochs, hist_none["val_acc"], color="#2a78d6", linewidth=2.0,
              marker="o", markersize=3.5, label="λ_SIR = 0（无正则）")
    epochs_s = np.arange(1, len(hist_sir["val_acc"]) + 1)
    ax_c.plot(epochs_s, hist_sir["val_acc"], color="#eb6834", linewidth=2.0,
              marker="s", markersize=3.5, label="λ_SIR = 0.1")
    ax_c.set_title("验证曲线对比（40 epochs，协议 D）", fontsize=10)
    ax_c.set_xlabel("Epoch", fontsize=9)
    ax_c.set_ylabel("验证集准确率", fontsize=9)
    ax_c.grid(color=GRID, linewidth=0.8)
    ax_c.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_c.spines[spine].set_visible(False)
    ax_c.legend(fontsize=9, frameon=False)

    metrics = ["OA", "AA", "Kappa"]
    v_none = [met_none["oa"], met_none["aa"], met_none["kappa"]]
    v_sir = [met_sir["oa"], met_sir["aa"], met_sir["kappa"]]
    x = np.arange(3)
    w = 0.36
    ax_m.bar(x - w / 2, v_none, w, color="#2a78d6", label="λ_SIR = 0")
    ax_m.bar(x + w / 2, v_sir, w, color="#eb6834", label="λ_SIR = 0.1")
    for xi, v in zip(x - w / 2, v_none):
        ax_m.text(xi, v, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, v_sir):
        ax_m.text(xi, v, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=8)
    ax_m.set_xticks(x, metrics)
    ax_m.set_ylim(0, 1.08)
    ax_m.set_title("测试集指标", fontsize=10)
    ax_m.grid(axis="y", color=GRID, linewidth=0.8)
    ax_m.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_m.spines[spine].set_visible(False)
    ax_m.legend(fontsize=9, frameon=False)
    fig.savefig(ASSETS / "ch09-sir-ablation.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((RUN_DEFAULT / "run_config.json").read_text(encoding="utf-8"))
    met_none = json.loads((RUN_40 / "metrics.json").read_text(encoding="utf-8"))
    met_sir = json.loads((RUN_SIR / "metrics.json").read_text(encoding="utf-8"))
    met_c = json.loads((RUN_C / "metrics.json").read_text(encoding="utf-8"))
    hist_none = json.loads((RUN_40 / "training_history.json").read_text(encoding="utf-8"))
    hist_sir = json.loads((RUN_SIR / "training_history.json").read_text(encoding="utf-8"))
    cm = np.load(RUN_40 / "confusion_matrix.npy")

    fig_ssrn_flow({})
    fig_sir_ablation(hist_none, hist_sir, met_none, met_sir)
    fig_confusion_matrix(cm, ASSETS / "ch09-confusion-matrix.png")

    print("layer table (SSRN):")
    for row in cfg["model"]["layers"]:
        if row["params"] or row["type"] in ("AvgPool3d",):
            print(f"  {row['layer']:<24}{row['type']:<14}"
                  f"{str(tuple(row['out_shape'])):<24}{row['params']}")
    print(f"total params: {cfg['model']['total_params']:,}   "
          f"MACs/sample: {cfg['model']['macs_per_sample'] / 1e6:.2f}M")
    print(f"SSRN 10ep (protocol D default): OA={met_none['oa'] * 100:.2f} "
          f"(default run) best={json.loads((RUN_DEFAULT / 'metrics.json').read_text(encoding='utf-8'))['best_epoch']}/10")
    print(f"SSRN 40ep (protocol D, no SIR): OA={met_none['oa'] * 100:.2f}  "
          f"AA={met_none['aa'] * 100:.2f}  Kappa={met_none['kappa']:.4f}  best={met_none['best_epoch']}")
    print(f"SSRN 40ep (protocol D, SIR 0.1): OA={met_sir['oa'] * 100:.2f}  "
          f"AA={met_sir['aa'] * 100:.2f}  Kappa={met_sir['kappa']:.4f}  best={met_sir['best_epoch']}")
    print(f"SSRN 40ep (protocol C, sklearn): OA={met_c['oa'] * 100:.2f}  "
          f"AA={met_c['aa'] * 100:.2f}  Kappa={met_c['kappa']:.4f}  best={met_c['best_epoch']}")
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
