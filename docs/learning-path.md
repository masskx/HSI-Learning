# 如何使用本课程

本课程有四部分17章。录制主线为「基础闭集 → 经典模型 → 小样本 → 开集 → 科研实验与写作」，不以新增模型数量或单次分数为目标。第一批11个问题驱动单元及运行入口见 [录课导航](teaching/course-map.md)，首次使用见 [启动指南](getting-started.md)。

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

章节已覆盖基础、经典模型、小样本与开集，但“正文存在”不等于“已完成录制验收”。本轮修正核心协议并准备讲课卡；新测试、新演示资产与Notebook输出待执行验证。各项状态与撤回的旧结论见 [纠错记录](teaching/errata.md) 和 [录制清单](teaching/recording-checklist.md)。历史实验继续留在 [benchmark](benchmark.md)，不得当作修正版代码的运行结果。
