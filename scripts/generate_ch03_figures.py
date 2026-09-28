"""Generate Chapter 3 figures: classical ML baselines on Indian Pines.

All numbers are real runs under the notebook-02 protocol (80/20 stratified
split, random_state=11) so the chapter text can quote them verbatim:

    1. Full-image prediction maps of 4 classical models (vs GT).
    2. SVM (C, gamma) validation-accuracy grid.
    3. PCA: explained-variance curve + PC-1/PC-2/PC-5 maps.

Also prints the per-model metrics, the 'scale' gamma value, the grid best
cell and PCA statistics quoted in the chapter.

Usage:
    python scripts/generate_ch03_figures.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from _chfigure_utils import (
    ASSETS,
    GRID,
    INK,
    MUTED,
    SEQ_BLUES,
    apply_style,
    load_indian_pines,
    stretch,
)

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)


def metrics(y_true, y_pred):
    labels = list(range(1, 17))
    return (
        accuracy_score(y_true, y_pred),
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        cohen_kappa_score(y_true, y_pred, labels=labels),
    )


def build_models():
    return {
        "SVM (RBF)": SVC(C=100, kernel="rbf", cache_size=2048),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=11, n_jobs=-1),
        "KNN (k=10)": KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000),
    }


def fig_model_maps(gt: np.ndarray, pred_maps: dict[str, np.ndarray], oas: dict[str, float]) -> None:
    panels = [("Ground Truth", gt)] + [(name, pm) for name, pm in pred_maps.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 4.2), constrained_layout=True)
    for ax, (name, pm) in zip(axes, panels):
        ax.imshow(pm, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
        title = name if name == "Ground Truth" else f"{name}\nOA={oas[name] * 100:.2f}%"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(ASSETS / "ch03-model-maps.png", dpi=200)
    plt.close(fig)


def fig_svm_grid(grid_acc: np.ndarray, c_values, gamma_values, gamma_scale: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    im = ax.imshow(grid_acc, cmap=SEQ_CMAP, vmin=min(grid_acc.min(), 0.5) - 0.02, vmax=1.0)

    for i, c in enumerate(c_values):
        for j, g in enumerate(gamma_values):
            ax.text(j, i, f"{grid_acc[i, j] * 100:.1f}", ha="center", va="center",
                    fontsize=10, color="#ffffff" if grid_acc[i, j] > 0.75 else INK)

    best = np.unravel_index(np.argmax(grid_acc), grid_acc.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                               edgecolor=INK, linewidth=2.5))

    ax.set_xticks(range(len(gamma_values)), [f"{g:g}" for g in gamma_values], fontsize=9)
    ax.set_yticks(range(len(c_values)), [f"C={c:g}" for c in c_values], fontsize=9)
    ax.set_xlabel("gamma（RBF 核宽度，越小越平滑）", fontsize=10)
    ax.set_ylabel("惩罚系数 C（越大越贴合训练集）", fontsize=10)
    ax.set_title(f"验证集准确率（%）· 黑框 = 最优组合 · 'scale'≈{gamma_scale:.2e}", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(ASSETS / "ch03-svm-grid.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cube, gt = load_indian_pines()
    n_bands = cube.shape[2]
    X_all = cube.reshape(-1, n_bands)
    y_all = gt.ravel()
    labeled = y_all > 0

    X_train, X_test, y_train, y_test = train_test_split(
        X_all[labeled], y_all[labeled], test_size=0.20, random_state=11, stratify=y_all[labeled]
    )
    print(f"protocol: 80/20 stratified, random_state=11 (notebook 02)")
    print(f"train: {len(y_train)}, test: {len(y_test)}")

    # ---- 4 classical baselines ---------------------------------------------
    results, pred_maps = {}, {}
    for name, model in build_models().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        t_pred = time.perf_counter() - t0
        oa, aa, kappa = metrics(y_test, y_pred)
        results[name] = {"oa": oa, "aa": aa, "kappa": kappa, "t_fit": t_fit, "t_pred": t_pred}
        full_pred = model.predict(X_all).reshape(gt.shape)
        pred_maps[name] = full_pred * (gt != 0)
        print(f"{name:<22} OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}  "
              f"fit={t_fit:.1f}s  predict_test={t_pred:.1f}s")

    oas = {name: r["oa"] for name, r in results.items()}
    fig_model_maps(gt, pred_maps, oas)

    # ---- small-sample protocol (10% train) for the protocol-dependence point
    X_tr10, X_te90, y_tr10, y_te90 = train_test_split(
        X_all[labeled], y_all[labeled], test_size=0.90, random_state=11, stratify=y_all[labeled]
    )
    print(f"\nsmall-sample protocol: 10% train ({len(y_tr10)}), 90% test ({len(y_te90)})")
    for name, model in build_models().items():
        model.fit(X_tr10, y_tr10)
        oa, aa, kappa = metrics(y_te90, model.predict(X_te90))
        results[name]["oa10"] = oa
        results[name]["aa10"] = aa
        results[name]["kappa10"] = kappa
        print(f"{name:<22} OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}")

    # ---- SVM (C, gamma) grid on train2/val ----------------------------------
    # NOTE: features are raw DN (~1e3 magnitude), so the meaningful gamma range
    # is far below the textbook [1e-4, 1e-1] grid that suits standardized
    # features — with the textbook grid every cell collapses to the majority
    # class (24%). gamma='scale' = 2e-9 sits inside the grid below.
    gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
    print(f"\ngamma='scale' = 1/(n_features * Var(X)) = {gamma_scale:.4e}")

    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=11, stratify=y_train
    )
    c_values = [0.1, 1, 10, 100]
    gamma_values = [1e-10, 1e-9, 1e-8, 1e-7]
    grid_acc = np.zeros((len(c_values), len(gamma_values)))
    print("SVM grid search (validation accuracy on 25% of the training split):")
    for i, c in enumerate(c_values):
        for j, g in enumerate(gamma_values):
            svm = SVC(C=c, kernel="rbf", gamma=g, cache_size=1024)
            svm.fit(X_tr2, y_tr2)
            grid_acc[i, j] = accuracy_score(y_val, svm.predict(X_val))
        print(f"  C={c:<6} " + "  ".join(f"g={g:g}:{grid_acc[i, j] * 100:5.1f}%" for j, g in enumerate(gamma_values)))
    best = np.unravel_index(np.argmax(grid_acc), grid_acc.shape)
    print(f"  best: C={c_values[best[0]]}, gamma={gamma_values[best[1]]}, "
          f"val_acc={grid_acc[best] * 100:.2f}%")
    fig_svm_grid(grid_acc, c_values, gamma_values, gamma_scale)

    # ---- PCA ----------------------------------------------------------------
    pca_full = PCA(n_components=20, whiten=True).fit(X_all)   # mirrors apply_pca_cube (fits on the whole cube)
    evr = pca_full.explained_variance_ratio_
    print(f"\nPCA cumulative EVR: k=5 {np.cumsum(evr)[4] * 100:.2f}%, "
          f"k=10 {np.cumsum(evr)[9] * 100:.2f}%, k=15 {np.cumsum(evr)[14] * 100:.2f}%, "
          f"k=20 {np.cumsum(evr)[19] * 100:.2f}%")
    print(f"first 5 individual EVR: {np.round(evr[:5] * 100, 2)}")

    pcs = pca_full.transform(X_all).reshape(gt.shape[0], gt.shape[1], -1)

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    ax = axes[0]
    ax.bar(range(1, 21), evr * 100, color="#2a78d6", label="单个主成分", width=0.7)
    ax.plot(range(1, 21), np.cumsum(evr) * 100, color="#eb6834", linewidth=2.0,
            marker="o", markersize=3.5, label="累计")
    ax.axvline(15, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(f"k=15\n累计 {np.cumsum(evr)[14] * 100:.1f}%", xy=(15, np.cumsum(evr)[14] * 100),
                xytext=(10.5, np.cumsum(evr)[14] * 100 - 30), fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9})
    ax.set_xlabel("主成分序号", fontsize=9)
    ax.set_ylabel("解释方差比例（%）", fontsize=9)
    ax.set_xlim(0, 21)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    for ax, k in zip(axes[1:], [1, 2, 5]):
        ax.imshow(pcs[:, :, k - 1], cmap="nipy_spectral", interpolation="nearest")
        ax.set_title(f"PC-{k}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(ASSETS / "ch03-pca.png", dpi=200)
    plt.close(fig)

    # ---- SVM on PCA-15 ------------------------------------------------------
    X_all_pca = pcs.reshape(-1, pcs.shape[2])[:, :15]
    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_all_pca[labeled], y_all[labeled], test_size=0.20, random_state=11, stratify=y_all[labeled]
    )
    svm_pca = SVC(C=100, kernel="rbf", cache_size=2048).fit(X_tr_p, y_tr_p)
    oa, aa, kappa = metrics(y_te_p, svm_pca.predict(X_te_p))
    print(f"SVM on PCA-15 (whiten=True): OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
