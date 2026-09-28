# 如何使用本课程

本课程按「基础篇 → 模型篇 → 科研篇」三部分共 14 章组织，完整目录见 [README](../README.md#课程目录--course-outline)。本文档只讲使用方式。

## 学习方式

1. **以 `chapters/` 的 markdown 讲解为主线**。每章包含：本章定位、学习目标、内容规划、配套实操、要点、自测题与延伸阅读。
2. **以 `notebooks/` 作为每章的动手环节**。讲解读完后，跑对应的 notebook，把文中概念落到代码和数据上。
3. **以 `src/` 与 `scripts/` 作为工程参照**。第 4、8、12 章会逐行对照讲解可复用模块与脚本化训练管线。

## 章节与 notebook 对照

| 章 | 主题 | 配套实操 |
|---|---|---|
| 1 | 高光谱成像基础 | `notebooks/01_data_reading_and_visualization.ipynb` |
| 2 | 数据集、划分与评价指标 | `notebooks/01`、`notebooks/02_svm_baseline.ipynb` |
| 3 | 传统机器学习基线 | `notebooks/02_svm_baseline.ipynb` |
| 4 | 从像素到 Patch | `notebooks/03_hybridsn_baseline.ipynb`（前半）、`src/hsi_learning/data.py` |
| 5 | 1D CNN | `notebooks/04_1d_cnn_teaching.ipynb` |
| 6 | 2D CNN | `notebooks/05_2d_cnn_teaching.ipynb` |
| 7 | 3D CNN | `notebooks/06_3d_cnn_teaching.ipynb` |
| 8 | HybridSN | `notebooks/03`、`notebooks/09_hybridsn_exploring_3d_2d_teaching.ipynb`、`scripts/train_hybridsn.py` |
| 9 | SSRN | `notebooks/08_ssrn_teaching.ipynb` |
| 10 | 注意力与 Transformer | `notebooks/07_transformer_teaching.ipynb` |
| 11–14 | 文献精读、实验设计、论文写作、毕业实战 | 讲解 + 工程实践为主 |

## 推荐学习顺序

1. 基础篇（第 1–4 章）顺序学完，打牢数据与实验协议的理解。
2. 模型篇（第 5–10 章）按 1D → 2D → 3D → HybridSN → SSRN → Transformer 的递进线学。
3. 科研篇（第 11–14 章）在准备写论文时开始，最终完成第 14 章的毕业实战。

## 当前状态

课程骨架已搭建完成，各章节正按编号顺序逐章写实。章节内标注了状态：🚧 未开始 / ✍️ 写作中 / ✅ 已完成。写作进度见 [README 课程目录](../README.md#课程目录--course-outline) 与 [roadmap](roadmap.md)。
