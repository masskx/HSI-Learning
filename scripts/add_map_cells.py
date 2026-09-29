"""Add map visualization cells to ch16/ch17 notebooks, then re-execute."""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def md(source):
    return nbformat.v4.new_markdown_cell(source)


def code(source):
    return nbformat.v4.new_code_cell(source)


# ---- ch16: add full-image prototype classification before summary ----
nb16_path = ROOT / "notebooks" / "16_protonet_teaching.ipynb"
nb16 = nbformat.read(nb16_path, as_version=4)

# find the summary cell (last markdown cell)
summary_idx_16 = None
for i, cell in enumerate(nb16.cells):
    if cell.cell_type == "markdown" and "小结" in cell.source:
        summary_idx_16 = i
        break

map_cells_16 = [
    md("## 7. 全图原型分类图\n\n用训练样本计算所有 16 类的原型，然后对整幅图的每个像元进行最近原型分类。"),
    code("""# 计算 16 类原型（用训练集嵌入均值）
encoder.eval()
train_embs = []
with torch.no_grad():
    for start in range(0, len(train_patches), 512):
        batch = torch.from_numpy(train_patches[start:start+512]).permute(0, 3, 1, 2)
        train_embs.append(encoder(batch).flatten(1).cpu().numpy())
train_embs = np.concatenate(train_embs)
all_prototypes = np.stack([train_embs[y_train == c].mean(0) for c in range(16)])
print("prototypes:", all_prototypes.shape)

# 全图像元 embedding + 最近原型分类
all_pos = np.argwhere(np.ones_like(gt, dtype=bool))
all_p = []
PAD = 4
padded_all = np.pad(reduced_scaled, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="constant")
for k, (r, c) in enumerate(all_pos):
    all_p.append(padded[r:r+9, c:c+9])
all_p = np.array(all_p, dtype=np.float32)

all_embs = []
with torch.no_grad():
    for start in range(0, len(all_p), 512):
        batch = torch.from_numpy(all_p[start:start+512]).permute(0, 3, 1, 2)
        all_embs.append(encoder(batch).flatten(1).cpu().numpy())
all_embs = np.concatenate(all_embs)

from scipy.spatial.distance import cdist
dists = cdist(all_embs, all_prototypes)
pred_map = np.array([np.argmin(d) for d in dists]).reshape(145, 145) + 1
print("prediction map:", pred_map.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
axes[0].imshow(gt, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
axes[0].set_title("Ground Truth")
axes[0].set_xticks([]); axes[0].set_yticks([])
im = axes[1].imshow(pred_map * (gt > 0), cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
axes[1].set_title("Prototypical Network Prediction")
axes[1].set_xticks([]); axes[1].set_yticks([])
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
plt.tight_layout()
plt.show()"""),
]

if summary_idx_16 is not None:
    for i, cell in enumerate(map_cells_16):
        nb16.cells.insert(summary_idx_16 + i, cell)

# renumber section headers in remaining markdown cells
for cell in nb16.cells:
    if cell.cell_type == "markdown" and "## 7. 小结" in cell.source:
        cell.source = cell.source.replace("## 7. 小结", "## 8. 小结")

nbformat.write(nb16, nb16_path)
print(f"ch16: added {len(map_cells_16)} cells, re-executing...")

# re-execute ch16
client16 = NotebookClient(nb16, timeout=600, kernel_name="python3",
                          resources={"metadata": {"path": str(ROOT)}})
client16.execute()
nbformat.write(nb16, nb16_path)
print(f"ch16 re-executed and saved")

# ---- ch17: add open-set classification map before summary ----
nb17_path = ROOT / "notebooks" / "17_openset_teaching.ipynb"
nb17 = nbformat.read(nb17_path, as_version=4)

summary_idx_17 = None
for i, cell in enumerate(nb17.cells):
    if cell.cell_type == "markdown" and "小结" in cell.source:
        summary_idx_17 = i
        break

map_cells_17 = [
    md("## 6. 开集分类图\n\n已知类用彩色显示，未知类（低置信度）用黑色标记。"),
    code("""# 全图像元预测
all_pos = np.argwhere(np.ones_like(gt, dtype=bool))
PAD = 4
padded_all = np.pad(reduced_scaled, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="constant")
all_p = np.zeros((len(all_pos), 9, 9, 12), dtype=np.float32)
for k, (r, c) in enumerate(all_pos):
    all_p[k] = padded_all[r:r+9, c:c+9]

all_logits = []
model.eval()
with torch.no_grad():
    for start in range(0, len(all_p), 512):
        batch = torch.from_numpy(all_p[start:start+512]).permute(0, 3, 1, 2)
        all_logits.append(model(batch).cpu().numpy())
all_logits = np.concatenate(all_logits)
all_probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
all_msp = all_probs.max(axis=1)
all_preds = all_logits.argmax(axis=1)

# map back to original class ids for coloring
pred_original = np.array([known_sorted[p] + 1 for p in all_preds])

# MSP threshold
threshold = msp_scores[is_known_test].mean()
print(f"MSP threshold: {threshold:.4f}")

# open-set map
openset_map = np.zeros(h * w, dtype=np.int64)
for i in range(len(all_pos)):
    r, c = all_pos[i]
    if gt[r, c] == 0:
        openset_map[r * w + c] = 0
    elif all_msp[i] < threshold:
        openset_map[r * w + c] = 0  # unknown → black
    else:
        openset_map[r * w + c] = pred_original[i]
openset_map = openset_map.reshape(h, w)

# GT with unknown hidden
gt_vis = gt.copy()
for uc in unknown_classes:
    gt_vis[gt == uc] = 0

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
axes[0].imshow(gt, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
axes[0].set_title("Ground Truth (all 16 classes)")
axes[1].imshow(gt_vis, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
axes[1].set_title("GT (unknown classes hidden)")
axes[2].imshow(openset_map, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
axes[2].set_title("Open-Set Prediction (black = unknown)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()"""),
]

if summary_idx_17 is not None:
    for i, cell in enumerate(map_cells_17):
        nb17.cells.insert(summary_idx_17 + i, cell)

for cell in nb17.cells:
    if cell.cell_type == "markdown" and "## 6. 小结" in cell.source:
        cell.source = cell.source.replace("## 6. 小结", "## 7. 小结")

nbformat.write(nb17, nb17_path)
print(f"ch17: added {len(map_cells_17)} cells, re-executing...")

client17 = NotebookClient(nb17, timeout=600, kernel_name="python3",
                          resources={"metadata": {"path": str(ROOT)}})
client17.execute()
nbformat.write(nb17, nb17_path)
print(f"ch17 re-executed and saved")
