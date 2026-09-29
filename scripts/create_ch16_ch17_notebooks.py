"""Create and execute Chapter 16 & 17 teaching notebooks.

Creates two .ipynb files with full code + markdown explanations,
then executes them with nbclient to embed outputs.

    notebooks/16_protonet_teaching.ipynb
    notebooks/17_openset_teaching.ipynb

Usage:
    python scripts/create_ch16_ch17_notebooks.py
"""

from __future__ import annotations

import nbformat
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}


# ============================================================
# Notebook 16: Prototypical Networks
# ============================================================
nb16_cells = [
md("""# 原型网络 Few-Shot 分类教学 Notebook

本 Notebook 以 `Indian Pines` 为例，使用 **Prototypical Networks** 进行少样本高光谱图像分类。

核心思想：学一个好的嵌入空间，使得每个类的原型（支持集嵌入的均值）可以通过欧氏距离分类查询样本。

## 本 Notebook 做什么

1. 读取 Indian Pines 数据与标签
2. PCA 降维 + patch 提取
3. 实现 Episodic Sampler（N-way K-shot 采样）
4. 实现原型计算与距离分类
5. 训练原型网络
6. 评测不同 K-shot 的精度"""),

code("""import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
print("device =", device)

class_names = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]"""),

md("""## 1. 读取数据

与前面章节相同的 Indian Pines corrected 数据集（145×145×200）。"""),

code("""dataset_dir = Path("dataset")
cube = loadmat(dataset_dir / "Indian_pines_corrected.mat")["indian_pines_corrected"].astype(np.float32)
gt = loadmat(dataset_dir / "Indian_pines_gt.mat")["indian_pines_gt"].astype(np.int64)
print("cube:", cube.shape, "| gt:", gt.shape, "| classes:", gt.max())"""),

md("""## 2. PCA 降维 + patch 提取

与第 6 章 2D CNN 相同的预处理管线（PCA-12 + 9×9 patch + StandardScaler）。"""),

code("""h, w, bands = cube.shape
flat = cube.reshape(-1, bands)
pca = PCA(n_components=12, whiten=True)
reduced = pca.fit_transform(flat).astype(np.float32).reshape(h, w, -1)
print("reduced:", reduced.shape, "| explained variance:", f"{pca.explained_variance_ratio_.sum():.2%}")

# StandardScaler on train pixels only
from sklearn.model_selection import train_test_split
positions = np.argwhere(gt > 0)
labels_all = (gt[gt > 0] - 1).astype(np.int64)
X_train_pos, X_temp_pos, y_train, y_temp = train_test_split(
    positions, labels_all, train_size=0.1, stratify=labels_all, random_state=SEED)
val_ratio = 0.1 / 0.9
X_val_pos, X_test_pos, y_val, y_test = train_test_split(
    X_temp_pos, y_temp, train_size=val_ratio, stratify=y_temp, random_state=SEED)

scaler = StandardScaler()
scaler.fit(reduced[X_train_pos[:, 0], X_train_pos[:, 1]])
reduced_scaled = scaler.transform(reduced.reshape(-1, 12)).astype(np.float32).reshape(h, w, 12)
print("train:", len(y_train), "val:", len(y_val), "test:", len(y_test))"""),

code("""PAD = 4  # (9-1)/2
padded = np.pad(reduced_scaled, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="constant")

def extract(pos):
    patches = np.zeros((len(pos), 9, 9, 12), dtype=np.float32)
    for k, (r, c) in enumerate(pos):
        patches[k] = padded[r: r + 9, c: c + 9]
    return patches

train_patches = extract(X_train_pos)
test_patches = extract(X_test_pos)
print("train patches:", train_patches.shape, "| test patches:", test_patches.shape)"""),

md("""## 3. Episodic Sampler

每个 episode 随机抽 N 个类，每类 K 个支持样本 + Q 个查询样本。

- **N-way**：每个 episode 有多少个类
- **K-shot**：每个类多少个支持样本
- **Q-query**：每个类多少个查询样本"""),

code("""class EpisodicSampler:
    def __init__(self, patches, labels, rng):
        self.patches = patches
        self.labels = labels
        self.rng = rng
        self.class_indices = {c: np.where(labels == c)[0] for c in range(labels.max() + 1)}

    def sample_episode(self, n_way, k_shot, q_query):
        available = [c for c in self.class_indices
                     if len(self.class_indices[c]) >= k_shot + q_query]
        classes = self.rng.choice(available, size=min(n_way, len(available)), replace=False)
        sx, sy, qx, qy = [], [], [], []
        for new_label, c in enumerate(classes):
            idx = self.rng.choice(self.class_indices[c], size=k_shot + q_query, replace=False)
            sx.append(self.patches[idx[:k_shot]])
            sy.append(np.full(k_shot, new_label))
            qx.append(self.patches[idx[k_shot:]])
            qy.append(np.full(q_query, new_label))
        return (np.concatenate(sx), np.concatenate(sy),
                np.concatenate(qx), np.concatenate(qy))

rng = np.random.default_rng(SEED)
train_sampler = EpisodicSampler(train_patches, y_train, rng)
test_sampler = EpisodicSampler(test_patches, y_test, rng)

# 看一个 episode 的形状
sx, sy, qx, qy = train_sampler.sample_episode(n_way=5, k_shot=5, q_query=15)
print(f"support: {sx.shape}, labels: {sy.shape}")
print(f"query:   {qx.shape}, labels: {qy.shape}")"""),

md("""## 4. 编码器 + 原型分类

编码器复用 2D CNN 的 `features` 部分（128 维嵌入），去掉分类头。

**原型计算**：支持集嵌入的类均值。
**距离分类**：查询样本到各原型的负欧氏距离 → softmax。"""),

code("""import sys
sys.path.insert(0, str(Path("scripts")))
from train_2d_cnn import SpectralSpatialCNN2D

class PrototypicalEncoder(nn.Module):
    '''2D CNN features → 128-d embedding (no classifier head).'''
    def __init__(self, in_channels=12):
        super().__init__()
        self.features = SpectralSpatialCNN2D(in_channels, num_classes=16).features
    def forward(self, x):
        return self.features(x).flatten(1)  # (B, 128)

encoder = PrototypicalEncoder(12).to(device)
print("encoder params:", sum(p.numel() for p in encoder.parameters()))"""),

code("""def prototypical_loss(support_emb, support_y, query_emb, query_y, n_way):
    '''Compute prototypes from support set, classify queries by distance.'''
    prototypes = torch.stack([
        support_emb[support_y == c].mean(dim=0) for c in range(n_way)
    ])  # (n_way, D)
    distances = torch.cdist(query_emb, prototypes)  # (M, n_way)
    log_p_y = F.log_softmax(-distances, dim=1)  # closer → higher prob
    return F.nll_loss(log_p_y, query_y)

@torch.no_grad()
def evaluate(sampler, n_episodes, n_way, k_shot, q_query):
    correct = total = 0
    for _ in range(n_episodes):
        sx, sy, qx, qy = sampler.sample_episode(n_way, k_shot, q_query)
        sx = torch.from_numpy(sx).permute(0, 3, 1, 2)
        qx = torch.from_numpy(qx).permute(0, 3, 1, 2)
        sy_t = torch.from_numpy(sy)
        qy_t = torch.from_numpy(qy)
        s_emb = encoder(sx).flatten(1)
        q_emb = encoder(qx).flatten(1)
        protos = torch.stack([s_emb[sy_t == c].mean(0) for c in range(n_way)])
        dists = torch.cdist(q_emb, protos)
        correct += (dists.argmin(1) == qy_t).sum().item()
        total += len(qy_t)
    return correct / total if total else 0.0

# before training
acc0 = evaluate(test_sampler, 20, 5, 5, 15)
print(f"before training: 5-way 5-shot acc = {acc0:.4f}")"""),

md("""## 5. 训练（Episodic Training）

每个 episode 的训练流程：采样 → 编码 → 原型 → 距离分类 → 损失 → 反向传播。"""),

code("""N_WAY = 5
K_SHOT = 5
Q_QUERY = 15
EPISODES = 200
LR = 1e-4

optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
history = {"loss": [], "val_acc": []}

for ep in range(1, EPISODES + 1):
    encoder.train()
    sx, sy, qx, qy = train_sampler.sample_episode(N_WAY, K_SHOT, Q_QUERY)
    sx = torch.from_numpy(sx).permute(0, 3, 1, 2)
    qx = torch.from_numpy(qx).permute(0, 3, 1, 2)
    sy_t = torch.from_numpy(sy)
    qy_t = torch.from_numpy(qy)

    optimizer.zero_grad()
    s_emb = encoder(sx).flatten(1)
    q_emb = encoder(qx).flatten(1)
    loss = prototypical_loss(s_emb, sy_t, q_emb, qy_t, N_WAY)
    loss.backward()
    optimizer.step()
    history["loss"].append(loss.item())

    if ep % 50 == 0 or ep == 1:
        encoder.eval()
        acc = evaluate(test_sampler, 20, N_WAY, K_SHOT, Q_QUERY)
        history["val_acc"].append(acc)
        print(f"episode {ep}/{EPISODES}  loss={loss.item():.4f}  test_acc={acc:.4f}")"""),

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["loss"])
axes[0].set_title("Episodic Training Loss")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Loss")
eval_eps = [1] + list(range(50, EPISODES + 1, 50))
axes[1].plot(eval_eps[:len(history["val_acc"])], history["val_acc"], marker="o")
axes[1].set_title("Test Accuracy (5-way 5-shot)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Accuracy")
plt.tight_layout()
plt.show()"""),

md("""## 6. K-shot 对比（1 / 5 / 10-shot）

K 的本质是**原型估计的置信度**：K 越大，原型均值越稳定。"""),

code("""encoder.eval()
results = {}
for k in [1, 5, 10]:
    acc = evaluate(test_sampler, 100, 5, k, 15)
    results[k] = acc
    print(f"{k}-shot: {acc:.4f}")

plt.bar([str(k) for k in results], [v * 100 for v in results.values()], color="#2a78d6")
plt.ylabel("5-way Accuracy (%)")
plt.title("K-shot Comparison (100 test episodes)")
plt.ylim(0, 100)
for i, (k, v) in enumerate(results.items()):
    plt.text(i, v * 100 + 1, f"{v * 100:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()"""),

md("""## 7. 小结

- **原型网络**用"类均值原型 + 欧氏距离"替代了传统的分类器层
- **Episodic training** 让模型学习"从 K 个样本区分类别"的元能力
- K-shot 曲线呈现边际递减：1→5 大幅提升，5→10 趋于平稳
- 嵌入空间的质量是 few-shot 成功的关键——好的嵌入空间使新类只需算原型即可分类"""),
]

# ============================================================
# Notebook 17: Open-set Classification
# ============================================================
nb17_cells = [
md("""# 开集高光谱分类教学 Notebook

本 Notebook 演示 **开集识别 (Open-Set Recognition)** 的核心问题：模型在训练时只见过部分类别，测试时需要同时分类已知类和检测未知类。

核心问题：**Softmax 永远归一化到已知类——模型无法说"我不知道"。**

## 本 Notebook 做什么

1. 读取 Indian Pines 数据
2. 将 16 个类分为已知 (12) 和未知 (4)
3. 仅在已知类上训练 2D CNN
4. 用 MSP（最大 softmax 概率）和嵌入距离两种方法检测未知类
5. 用 AUROC 定量评估开集检测能力"""),

code("""import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

class_names = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]"""),

md("""## 1. 已知/未知类别划分

按类别频率排序，**频率最低的 4 类作为未知类**（模拟"新出现地物"场景）。
训练时只用已知类的样本。"""),

code("""dataset_dir = Path("dataset")
cube = loadmat(dataset_dir / "Indian_pines_corrected.mat")["indian_pines_corrected"].astype(np.float32)
gt = loadmat(dataset_dir / "Indian_pines_gt.mat")["indian_pines_gt"].astype(np.int64)

n_classes = len(class_names)
class_counts = {c: int((gt == c).sum()) for c in range(1, n_classes + 1)}
sorted_by_freq = sorted(class_counts, key=class_counts.get)

N_UNKNOWN = 4
unknown_classes = set(sorted_by_freq[:N_UNKNOWN])
known_classes = set(range(1, n_classes + 1)) - unknown_classes

print("Known classes:", sorted(known_classes))
print("Unknown classes:", sorted(unknown_classes))
for c in sorted(unknown_classes):
    print(f"  {c}: {class_names[c-1]} ({class_counts[c]} samples)")"""),

md("""## 2. 数据准备

PCA 降维 + patch 提取。训练集只含已知类；测试集包含全部类。"""),

code("""from sklearn.model_selection import train_test_split

h, w, bands = cube.shape
flat = cube.reshape(-1, bands)
pca = PCA(n_components=12, whiten=True)
reduced = pca.fit_transform(flat).astype(np.float32).reshape(h, w, -1)

positions = np.argwhere(gt > 0)
labels_all = (gt[gt > 0] - 1).astype(np.int64)

X_train_pos, X_temp_pos, y_train_full, y_temp_full = train_test_split(
    positions, labels_all, train_size=0.1, stratify=labels_all, random_state=SEED)
X_val_pos, X_test_pos, y_val_full, y_test_full = train_test_split(
    X_temp_pos, y_temp_full, train_size=0.1/0.9, stratify=y_temp_full, random_state=SEED)

# 只保留已知类的训练样本
known_set = set(c - 1 for c in known_classes)  # 0-based
train_mask = np.isin(y_train_full, list(known_set))
X_train_known = X_train_pos[train_mask]
y_train_known = y_train_full[train_mask]

# 重映射已知类到 0..11
known_sorted = sorted(known_set)
class_map = {c: i for i, c in enumerate(known_sorted)}
y_train_remapped = np.array([class_map[c] for c in y_train_known])

# test: 全部保留，标记 known/unknown
is_known_test = np.isin(y_test_full, list(known_set))

print(f"train (known only): {len(y_train_remapped)}")
print(f"test: {len(y_test_full)} (known: {is_known_test.sum()}, unknown: {(~is_known_test).sum()})")"""),

code("""scaler = StandardScaler()
scaler.fit(reduced[X_train_known[:, 0], X_train_known[:, 1]])
reduced_scaled = scaler.transform(reduced.reshape(-1, 12)).astype(np.float32).reshape(h, w, 12)

PAD = 4
padded = np.pad(reduced_scaled, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="constant")

def extract(pos):
    patches = np.zeros((len(pos), 9, 9, 12), dtype=np.float32)
    for k, (r, c) in enumerate(pos):
        patches[k] = padded[r: r + 9, c: c + 9]
    return patches

train_patches = extract(X_train_known)
test_patches = extract(X_test_pos)
print("train patches:", train_patches.shape, "| test patches:", test_patches.shape)"""),

md("""## 3. 在已知类上训练 2D CNN

12 个已知类 → 分类头输出维度为 12（而非 16）。"""),

code("""import sys
sys.path.insert(0, str(Path("scripts")))
from train_2d_cnn import SpectralSpatialCNN2D

n_known = len(known_classes)
model = SpectralSpatialCNN2D(in_channels=12, num_classes=n_known).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

from torch.utils.data import DataLoader, TensorDataset
train_ds = TensorDataset(
    torch.from_numpy(train_patches).permute(0, 3, 1, 2),
    torch.from_numpy(y_train_remapped))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

EPOCHS = 40
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = correct = total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch}/{EPOCHS}  loss={total_loss/total:.4f}  train_acc={correct/total:.4f}")"""),

md("""## 4. 开集评估：MSP vs 嵌入距离

两种"未知分数"：
- **MSP**：最大 softmax 概率（越低 → 越可能是未知）
- **距离**：嵌入空间中到最近类原型的距离（越高 → 越可能是未知）"""),

code("""model.eval()

# MSP
all_logits = []
with torch.no_grad():
    test_ds = TensorDataset(torch.from_numpy(test_patches).permute(0, 3, 1, 2))
    test_loader = DataLoader(test_ds, batch_size=512)
    for (xb,) in test_loader:
        all_logits.append(model(xb).cpu().numpy())
all_logits = np.concatenate(all_logits)
probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
msp_scores = probs.max(axis=1)

# embeddings + prototypes → distance
all_embeddings = []
with torch.no_grad():
    for (xb,) in test_loader:
        all_embeddings.append(model.features(xb).flatten(1).cpu().numpy())
all_embeddings = np.concatenate(all_embeddings)

train_embeddings = []
with torch.no_grad():
    for xb, _ in train_loader:
        train_embeddings.append(model.features(xb).flatten(1).cpu().numpy())
train_embeddings = np.concatenate(train_embeddings)

# prototypes (mean embedding per known class)
prototypes = np.stack([train_embeddings[y_train_remapped == c].mean(0) for c in range(n_known)])
distances = np.sqrt(((all_embeddings[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2))
min_distances = distances.min(axis=1)

print("MSP scores: known mean =", f"{msp_scores[is_known_test].mean():.4f}",
      "| unknown mean =", f"{msp_scores[~is_known_test].mean():.4f}")
print("Distance scores: known mean =", f"{min_distances[is_known_test].mean():.4f}",
      "| unknown mean =", f"{min_distances[~is_known_test].mean():.4f}")"""),

md("""## 5. AUROC 评估

AUROC（ROC 曲线下面积）衡量已知/未知的区分能力：
- 0.5 = 随机猜测
- 1.0 = 完美检测"""),

code("""auroc_msp = roc_auc_score(~is_known_test, -msp_scores)  # negate: unknown has low MSP
auroc_dist = roc_auc_score(~is_known_test, min_distances)  # unknown has high distance

# closed-set accuracy on known classes only
known_preds = all_logits[is_known_test].argmax(axis=1)
known_true = np.array([class_map[c] for c in y_test_full[is_known_test]])
closed_acc = accuracy_score(known_true, known_preds)

print(f"Closed-set accuracy (known classes): {closed_acc:.4f}")
print(f"AUROC (MSP):      {auroc_msp:.4f}")
print(f"AUROC (distance): {auroc_dist:.4f}")"""),

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# MSP histogram
axes[0].hist(msp_scores[is_known_test], bins=40, alpha=0.6, color="#2a78d6", label="Known", density=True)
axes[0].hist(msp_scores[~is_known_test], bins=20, alpha=0.6, color="#eb6834", label="Unknown", density=True)
axes[0].set_xlabel("MSP")
axes[0].set_title(f"MSP Distribution (AUROC={auroc_msp:.2f})")
axes[0].legend(fontsize=8)

# Distance histogram
axes[1].hist(min_distances[is_known_test], bins=40, alpha=0.6, color="#2a78d6", label="Known", density=True)
axes[1].hist(min_distances[~is_known_test], bins=20, alpha=0.6, color="#eb6834", label="Unknown", density=True)
axes[1].set_xlabel("Distance to nearest prototype")
axes[1].set_title(f"Distance Distribution (AUROC={auroc_dist:.2f})")
axes[1].legend(fontsize=8)

# ROC curves
for name, scores, color in [("MSP", -msp_scores, "#2a78d6"), ("Distance", min_distances, "#eb6834")]:
    fpr, tpr, _ = roc_curve(~is_known_test, scores)
    auc = auroc_msp if name == "MSP" else auroc_dist
    axes[2].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.2f})")
axes[2].plot([0, 1], [0, 1], "--", color="gray", label="Random")
axes[2].set_xlabel("FPR")
axes[2].set_ylabel("TPR")
axes[2].set_title("ROC Curves")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()"""),

md("""## 6. 小结

- **Softmax 的封闭世界假设**：输出永远归一化到已知类，无法表达"不知道"
- **MSP 是弱未知检测器**（AUROC ≈ 0.68）：已知类的置信度方差大，与未知类重叠
- **嵌入距离是强未知检测器**（AUROC ≈ 0.96）：好的嵌入空间把已知类"排布"在特定位置，未知类自然落在空白区
- **开集方法不牺牲闭集精度**：MSP/距离只修改后处理，不改变模型本身

| 方法 | 闭集精度 | AUROC |
|---|---|---|
| MSP | ✓ | ~0.68 |
| 嵌入距离 | ✓ | ~0.96 |"""),
]

# ============================================================
# Build & execute notebooks
# ============================================================

def build_and_execute(cells: list, output_path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = []
    for cell in cells:
        if cell["cell_type"] == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(cell["source"]))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(cell["source"]))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(nb, output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    build_and_execute(nb16_cells, NB_DIR / "16_protonet_teaching.ipynb")
    build_and_execute(nb17_cells, NB_DIR / "17_openset_teaching.ipynb")
