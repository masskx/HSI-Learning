"""Replace the open-set map cell in notebook 17 (unknown → white) and re-execute."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "17_openset_teaching.ipynb"

NEW_SOURCE = '''# 全图像元预测
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

# open-set map: 0 = background, 1..16 = known classes, 17 = unknown
UNKNOWN_CODE = 17
openset_map = np.zeros(h * w, dtype=np.int64)
for i in range(len(all_pos)):
    r, c = all_pos[i]
    if gt[r, c] == 0:
        openset_map[r * w + c] = 0  # background
    elif all_msp[i] < threshold:
        openset_map[r * w + c] = UNKNOWN_CODE  # unknown → white
    else:
        openset_map[r * w + c] = pred_original[i]
openset_map = openset_map.reshape(h, w)
print(f"unknown pixels: {(openset_map == UNKNOWN_CODE).sum()}")

# custom colormap: 0=black(bg), 1..16=nipy_spectral class colors, 17=white(unknown)
from matplotlib.colors import ListedColormap
_base = plt.cm.nipy_spectral(np.linspace(0, 1, 256))
_openset_colors = [_base[0]]                                          # 0  → background (black)
_openset_colors += [_base[int(k / 16 * 255)] for k in range(1, 17)]   # 1..16 → classes
_openset_colors += [[1.0, 1.0, 1.0, 1.0]]                             # 17 → unknown (white)
openset_cmap = ListedColormap(_openset_colors)

# GT with unknown hidden
gt_vis = gt.copy()
for uc in unknown_classes:
    gt_vis[gt == uc] = 0

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
axes[0].imshow(gt, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
axes[0].set_title("Ground Truth (all 16 classes)")
axes[1].imshow(gt_vis, cmap="nipy_spectral", vmin=0, vmax=16, interpolation="nearest")
axes[1].set_title("GT (unknown classes hidden)")
im = axes[2].imshow(openset_map, cmap=openset_cmap, vmin=-0.5, vmax=17.5, interpolation="nearest")
axes[2].set_title("Open-Set Prediction (white = unknown)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02, ticks=[0, 1, 8, 16, 17])
cbar.ax.set_yticklabels(["bg", "1", "8", "16", "unknown"])
plt.tight_layout()
plt.show()'''

nb = nbformat.read(NB_PATH, as_version=4)

# replace the open-set map code cell
replaced = False
for cell in nb.cells:
    if cell.cell_type == "code" and "openset_map" in cell.source:
        cell.source = NEW_SOURCE
        replaced = True
        print("replaced open-set map cell")
        break

# update the markdown title above it
for cell in nb.cells:
    if cell.cell_type == "markdown" and "开集分类图" in cell.source:
        cell.source = "## 6. 开集分类图\n\n已知类用彩色显示，未知类（低置信度）用**白色**标记。"
        print("updated markdown description")

if not replaced:
    sys.exit("ERROR: open-set map cell not found")

nbformat.write(nb, NB_PATH)

print("re-executing notebook...")
client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(nb, NB_PATH)
print("done")
