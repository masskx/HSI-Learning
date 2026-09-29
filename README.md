# HSI-Learning · 高光谱图像分类课程

> A chapter-by-chapter course on hyperspectral image (HSI) classification — from data fundamentals to publishing your first paper.
> 一门分章节讲解的高光谱图像分类课程：从数据基础到独立撰写和发表论文。

本仓库面向高光谱图像分类学习者，按「基础篇 → 模型篇 → 科研篇」三个部分分章节组织：每一章是独立的 markdown 讲解，配套可运行的 notebook 与工程化代码，最终以「毕业实战：完成一篇论文」收尾。

## 课程目录 / Course Outline

状态图例：🚧 未开始 · ✍️ 写作中 · ✅ 已完成

### 第一部分 基础篇 / Part I Foundations

| 章 | 标题 | 配套实操 | 状态 |
|---|---|---|---|
| 1 | [高光谱成像基础 Hyperspectral Imaging Fundamentals](chapters/part1-foundations/ch01-hyperspectral-imaging-fundamentals.md) | [01_data_reading_and_visualization](notebooks/01_data_reading_and_visualization.ipynb) | ✅ |
| 2 | [数据集、划分与评价指标 Datasets, Splits & Metrics](chapters/part1-foundations/ch02-datasets-splits-and-metrics.md) | [01](notebooks/01_data_reading_and_visualization.ipynb) · [02](notebooks/02_svm_baseline.ipynb) | ✅ |
| 3 | [传统机器学习基线 Classical ML Baselines](chapters/part1-foundations/ch03-classical-ml-baselines.md) | [02_svm_baseline](notebooks/02_svm_baseline.ipynb) · [build_indian_pines_csv.py](scripts/build_indian_pines_csv.py) | ✅ |
| 4 | [从像素到 Patch：深度学习数据工程 From Pixels to Patches](chapters/part1-foundations/ch04-from-pixels-to-patches.md) | [03_hybridsn_baseline](notebooks/03_hybridsn_baseline.ipynb) · [src/hsi_learning/data.py](src/hsi_learning/data.py) | ✅ |

### 第二部分 模型篇 / Part II Models

| 章 | 标题 | 配套实操 | 状态 |
|---|---|---|---|
| 5 | [1D CNN：光谱序列建模 1D CNN](chapters/part2-models/ch05-1d-cnn-spectral-modeling.md) | [04_1d_cnn_teaching](notebooks/04_1d_cnn_teaching.ipynb) · [train_1d_cnn.py](scripts/train_1d_cnn.py) | ✅ |
| 6 | [2D CNN：空间建模 2D CNN](chapters/part2-models/ch06-2d-cnn-spatial-modeling.md) | [05_2d_cnn_teaching](notebooks/05_2d_cnn_teaching.ipynb) · [train_2d_cnn.py](scripts/train_2d_cnn.py) | ✅ |
| 7 | [3D CNN：谱空联合建模 3D CNN](chapters/part2-models/ch07-3d-cnn-joint-spatial-spectral-modeling.md) | [06_3d_cnn_teaching](notebooks/06_3d_cnn_teaching.ipynb) · [train_3d_cnn.py](scripts/train_3d_cnn.py) | ✅ |
| 8 | [HybridSN：3D-2D 混合结构 HybridSN](chapters/part2-models/ch08-hybridsn-3d-2d-hybrid.md) | [03](notebooks/03_hybridsn_baseline.ipynb) · [09](notebooks/09_hybridsn_exploring_3d_2d_teaching.ipynb) · [train_hybridsn.py](scripts/train_hybridsn.py) | ✅ |
| 9 | [SSRN：残差谱空网络 SSRN](chapters/part2-models/ch09-ssrn-residual-spectral-spatial-network.md) | [08_ssrn_teaching](notebooks/08_ssrn_teaching.ipynb) · [train_ssrn.py](scripts/train_ssrn.py) | ✅ |
| 10 | [注意力与 Transformer Attention & Transformers](chapters/part2-models/ch10-attention-and-transformers.md) | [07_transformer_teaching](notebooks/07_transformer_teaching.ipynb) · [train_transformer.py](scripts/train_transformer.py) | ✅ |

### 第三部分 科研篇 / Part III Research

| 章 | 标题 | 配套实操 | 状态 |
|---|---|---|---|
| 11 | [文献检索与论文精读 Literature & Paper Reading](chapters/part3-research/ch11-literature-and-paper-reading.md) | [精读笔记范文](chapters/part3-research/notes/hybridsn-roy2020-grsl-reading-notes.md) | ✅ |
| 12 | [科研实验设计与可复现工程 Experimental Design & Reproducibility](chapters/part3-research/ch12-experimental-design-and-reproducibility.md) | [train_hybridsn.py](scripts/train_hybridsn.py) · [aggregate_ch12_ablation.py](scripts/aggregate_ch12_ablation.py) | ✅ |
| 13 | [论文写作与投稿 Paper Writing & Submission](chapters/part3-research/ch13-paper-writing-and-submission.md) | [IEEEtran 论文骨架](chapters/part3-research/latex/paper_skeleton.tex) | ✅ |
| 14 | [毕业实战：完成一篇论文 Capstone: Your First Paper](chapters/part3-research/ch14-capstone-your-first-paper.md) | 全部 | ✅ |
| 15 | [实战补遗：五个拦路虎 Practical Essentials](chapters/part3-research/ch15-practical-essentials.md) | [train_2d_cnn.py](scripts/train_2d_cnn.py) · [generate_ch15_analysis.py](scripts/generate_ch15_analysis.py) | ✅ |

### 第四部分 小样本与开集 / Part IV Few-Shot and Open-Set

| 章 | 标题 | 配套实操 | 状态 |
|---|---|---|---|
| 16 | [小样本高光谱图像分类 Few-Shot Classification](chapters/part4-few-shot-open-set/ch16-few-shot-classification.md) | [train_protonet.py](scripts/train_protonet.py) | ✅ |
| 17 | [开集高光谱图像分类 Open-Set Classification](chapters/part4-few-shot-open-set/ch17-open-set-classification.md) | [train_openset.py](scripts/train_openset.py) | ✅ |

## 学习路径建议 / Suggested Learning Path

1. **基础篇（第 1–4 章）按顺序学**：认识数据 → 建立实验协议 → 跑通传统基线 → 掌握 patch 数据工程。
2. **模型篇（第 5–10 章）建议按序学**：1D → 2D → 3D → HybridSN → SSRN → Transformer 是一条自然的难度与信息融合递进线；也可按兴趣跳读。
3. **科研篇（第 11–14 章）在准备写论文时必读**：精读 → 实验设计 → 写作 → 毕业实战，对应真实科研的完整闭环。

## Quick Start

安装依赖：

```bash
pip install -r requirements.txt
```

当前默认使用 `dataset/` 下的 `Indian_pines_corrected.mat` 和 `Indian_pines_gt.mat`。

如需生成传统机器学习使用的像素表：

```bash
python scripts/build_indian_pines_csv.py --output indian_pines_all.csv
```

如需运行脚本化 `HybridSN` 基线：

```bash
python scripts/train_hybridsn.py --dataset IP --epochs 20
```

## 项目结构 / Project Structure

```text
HSI-Learning/
├─ chapters/                  # 分章课程讲解（本仓库主线）
│  ├─ part1-foundations/      # 第 1–4 章 基础篇
│  ├─ part2-models/           # 第 5–10 章 模型篇
│  ├─ part3-research/         # 第 11–14 章 科研篇
│  └─ assets/                 # 章节插图
├─ dataset/                   # Indian Pines 数据与说明
├─ notebooks/                 # 章节配套的实操 notebook
├─ scripts/                   # 脚本化训练与数据处理入口
├─ src/hsi_learning/          # 可复用代码模块
├─ docs/                      # 使用说明、代码概览与路线图
├─ results/                   # 本地实验输出
└─ requirements.txt
```

## Engineered Code

- [scripts/train_hybridsn.py](scripts/train_hybridsn.py)
- [scripts/build_indian_pines_csv.py](scripts/build_indian_pines_csv.py)
- [src/hsi_learning/](src/hsi_learning)

## Notes

- `Indian Pines` 结果图当前统一使用 `nipy_spectral` 配色。
- `.ipynb_checkpoints` 和 `.jupyter_runtime/` 已忽略，不会提交到远端。
- `results/` 为本地实验输出目录。

## Docs

- [docs/learning-path.md](docs/learning-path.md) — 如何使用本课程
- [docs/codebase-overview.md](docs/codebase-overview.md) — 代码结构总览
- [docs/benchmark.md](docs/benchmark.md) — 课程基准记分板（逐章追加）
- [docs/roadmap.md](docs/roadmap.md) — 课程化路线图
