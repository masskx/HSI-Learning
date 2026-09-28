"""Generate Chapter 10 figures: spectral Transformer on Indian Pines.

Reads artifacts from results/transformer/IP (10-epoch default),
results/transformer_ip_epochs40 and results/transformer_ip_epochs120
(extended runs), plus the trained checkpoint for the attention
visualisation, and produces:

    1. ch10-attention.png      — a real pixel's standardized spectrum, the
                                 first-layer multi-head-averaged attention
                                 matrix (200x200 bands), and per-query
                                 attention profiles.
    2. ch10-training-curve.png — the 120-epoch validation curve with the SVM
                                 anchor and the 10/40-epoch checkpoints marked.
    3. ch10-confusion-matrix.png — confusion matrix of the best available run.
    4. ch10-model-summary.png  — Part II capstone: OA and AA of every model
                                 under protocol C (SVM anchor included).

Run scripts/train_transformer.py (+ extended runs) first.

Usage:
    python scripts/generate_ch10_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import LinearSegmentedColormap

from _chfigure_utils import (
    ASSETS,
    CLASS_NAMES,
    GRID,
    INK,
    SEQ_BLUES,
    apply_style,
    fig_confusion_matrix,
    load_indian_pines,
    two_step_split,
)
from train_transformer import SpectralTransformerClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN_DEFAULT = RESULTS / "transformer" / "IP"
RUN_40 = RESULTS / "transformer_ip_epochs40" / "IP"
RUN_120 = RESULTS / "transformer_ip_epochs120" / "IP"


def pick_run() -> Path:
    for candidate in (RUN_120, RUN_40, RUN_DEFAULT):
        if (candidate / "metrics.json").exists():
            return candidate
    raise FileNotFoundError("train the Transformer first (scripts/train_transformer.py)")


def attention_from_model(model, spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-layer multi-head-averaged attention matrix and token embeddings."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(spectrum).unsqueeze(0)          # (1, 200)
        tokens = model.band_proj(x.unsqueeze(-1)) + model.pos_embed
        layer = model.encoder.layers[0]
        attn = layer.self_attn
        qkv = tokens @ attn.in_proj_weight.T + attn.in_proj_bias
        q, k, _ = qkv.chunk(3, dim=-1)
        n_heads = attn.num_heads
        d_head = attn.head_dim
        q = q.view(1, -1, n_heads, d_head).transpose(1, 2)   # (1, h, 200, d_h)
        k = k.view(1, -1, n_heads, d_head).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(d_head), dim=-1)
    return weights.mean(dim=1)[0].numpy(), tokens[0].numpy()


def fig_attention(model, scaler, cube: np.ndarray, gt: np.ndarray) -> None:
    sample_idx = np.argwhere(gt == 8)[0]                 # a Hay-windrowed pixel
    spectrum = scaler.transform(cube[sample_idx[0], sample_idx[1]].reshape(1, -1))[0]
    attn, _ = attention_from_model(model, spectrum)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.1, 1, 1.1]})

    ax = axes[0]
    ax.plot(spectrum, color="#2a78d6", linewidth=1.4)
    ax.set_title("一个 Hay-windrowed 像元的光谱（标准化）", fontsize=10)
    ax.set_xlabel("波段（token 序号）", fontsize=9)
    ax.set_ylabel("标准化响应", fontsize=9)

    ax = axes[1]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=attn.max(), interpolation="nearest")
    ax.set_title("第 1 层注意力矩阵（4 头平均，200×200）", fontsize=10)
    ax.set_xlabel("被关注的波段 token", fontsize=9)
    ax.set_ylabel("查询波段 token", fontsize=9)
    ax.set_xticks([0, 199], ["0", "199"], fontsize=8)
    ax.set_yticks([0, 199], ["0", "199"], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = axes[2]
    for query, color in [(30, "#2a78d6"), (100, "#eb6834"), (170, "#1baf7a")]:
        ax.plot(attn[query], color=color, linewidth=1.5, label=f"query = 波段 {query}")
    ax.set_title("三个查询波段的注意力分布", fontsize=10)
    ax.set_xlabel("被关注的波段 token", fontsize=9)
    ax.set_ylabel("注意力权重", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False)

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(ASSETS / "ch10-attention.png", dpi=200)
    plt.close(fig)


def fig_training_curve(run: Path) -> None:
    hist = json.loads((run / "training_history.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    epochs = np.arange(1, len(hist["val_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    ax.plot(epochs, hist["val_acc"], color="#2a78d6", linewidth=2.0, marker="o", markersize=3)
    ax.axhline(0.8070, color="#898781", linewidth=1.2, linestyle="--")
    ax.text(epochs[-1] * 0.72, 0.8070 + 0.008, "SVM 锚点 80.70%", fontsize=9, color="#898781")
    for marker_epoch, label in [(10, "10 epochs"), (40, "40 epochs")]:
        if marker_epoch <= len(epochs):
            v = hist["val_acc"][marker_epoch - 1]
            ax.plot(marker_epoch, v, marker="*", markersize=14, color="#eda100",
                    markeredgecolor=INK, linestyle="none")
            ax.annotate(f"{label}\nval_acc {v * 100:.1f}%", xy=(marker_epoch, v),
                        xytext=(marker_epoch + 4, v - 0.06), fontsize=8.5,
                        arrowprops={"arrowstyle": "->", "color": "#898781", "lw": 0.9})
    ax.set_title(f"谱 Transformer 验证曲线（协议 C，最终 OA {metrics['oa'] * 100:.2f}%）", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("验证集准确率", fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(ASSETS / "ch10-training-curve.png", dpi=200)
    plt.close(fig)


def fig_model_summary(metrics_by_model: dict) -> None:
    names = list(metrics_by_model)
    oa = [metrics_by_model[n]["oa"] for n in names]
    aa = [metrics_by_model[n]["aa"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 5), constrained_layout=True)
    ax.bar(x - w / 2, oa, w, color="#2a78d6", label="OA")
    ax.bar(x + w / 2, aa, w, color="#eb6834", label="AA")
    for xi, v in zip(x - w / 2, oa):
        ax.text(xi, v, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, aa):
        ax.text(xi, v, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("模型篇总结：协议 C（10/10/80，seed 42）下各模型的 OA 与 AA", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.savefig(ASSETS / "ch10-model-summary.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()
    run = pick_run()

    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    cm = np.load(run / "confusion_matrix.npy")

    # rebuild the identical split/scaler, load the trained model
    train_gt, _, test_gt = two_step_split(gt, 0.1, 0.1, random_state=42)
    rows_tr = np.nonzero(train_gt)
    scaler = StandardScaler().fit(cube[rows_tr])
    model = SpectralTransformerClassifier(num_bands=cube.shape[2], num_classes=16)
    model.load_state_dict(torch.load(run / "best_model.pth", map_location="cpu"))

    fig_attention(model, scaler, cube, gt)
    fig_training_curve(run)
    fig_confusion_matrix(cm, ASSETS / "ch10-confusion-matrix.png")

    # Part II capstone: every model under protocol C (deterministic re-runs)
    summary = {"SVM 锚点": {"oa": 0.8070, "aa": 0.7773}}   # ch05 protocol-C anchor
    for name, path in [
        ("1D CNN", RESULTS / "1d_cnn_ip_epochs60" / "IP"),
        ("3D CNN", RESULTS / "3d_cnn_ip_epochs40" / "IP"),
        ("2D CNN", RESULTS / "2d_cnn_ip_epochs40" / "IP"),
        ("HybridSN", RESULTS / "hybridsn_ip_protocolC" / "IP"),
        ("SSRN", RUN_C := RESULTS / "ssrn_ip_protocolC" / "IP"),
        ("Transformer", run),
    ]:
        m = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        summary[name] = {"oa": m["oa"], "aa": m["aa"]}
    fig_model_summary(summary)
    print("protocol-C summary:", {k: (round(v["oa"] * 100, 2), round(v["aa"] * 100, 2))
                                   for k, v in summary.items()})

    # per-class recall of the best Transformer run
    recall = np.diag(cm) / cm.sum(axis=1)
    print("\nTransformer per-class recall (best run):")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {i + 1:>2}  {n:<32}{recall[i]:.3f}")
    print(f"params: {cfg['model']['total_params']:,}   "
          f"MACs/sample: {cfg['model']['macs_per_sample'] / 1e6:.2f}M")
    print("figures saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
