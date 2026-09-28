# 当前代码概览

这个仓库目前由四部分组成：分章课程讲解、教学 notebook、脚本入口、可复用源码模块。

## 1. 分章课程讲解

- [chapters/part1-foundations/](../chapters/part1-foundations) — 第 1–4 章 基础篇（高光谱成像、数据集与指标、传统基线、patch 数据工程）
- [chapters/part2-models/](../chapters/part2-models) — 第 5–10 章 模型篇（1D/2D/3D CNN、HybridSN、SSRN、注意力与 Transformer）
- [chapters/part3-research/](../chapters/part3-research) — 第 11–14 章 科研篇（文献精读、实验设计、论文写作、毕业实战）

课程主线是 markdown 讲解；notebook 是每章的配套实操。完整目录见 [README](../README.md#课程目录--course-outline)。

## 2. 教学 notebook

- [notebooks/01_data_reading_and_visualization.ipynb](../notebooks/01_data_reading_and_visualization.ipynb)
- [notebooks/02_svm_baseline.ipynb](../notebooks/02_svm_baseline.ipynb)
- [notebooks/03_hybridsn_baseline.ipynb](../notebooks/03_hybridsn_baseline.ipynb)
- [notebooks/04_1d_cnn_teaching.ipynb](../notebooks/04_1d_cnn_teaching.ipynb)
- [notebooks/05_2d_cnn_teaching.ipynb](../notebooks/05_2d_cnn_teaching.ipynb)
- [notebooks/06_3d_cnn_teaching.ipynb](../notebooks/06_3d_cnn_teaching.ipynb)
- [notebooks/07_transformer_teaching.ipynb](../notebooks/07_transformer_teaching.ipynb)
- [notebooks/08_ssrn_teaching.ipynb](../notebooks/08_ssrn_teaching.ipynb)
- [notebooks/09_hybridsn_exploring_3d_2d_teaching.ipynb](../notebooks/09_hybridsn_exploring_3d_2d_teaching.ipynb)

这些 notebook 的职责是教学解释、可视化展示和最小可运行示例。

## 3. 脚本入口

- [scripts/build_indian_pines_csv.py](../scripts/build_indian_pines_csv.py)  
  将高光谱立方体展开为传统机器学习可用的样本表。
- [scripts/train_hybridsn.py](../scripts/train_hybridsn.py)  
  提供脚本化的 `HybridSN` 训练、验证、测试与结果导出流程。

## 4. 可复用源码

- [src/hsi_learning/data.py](../src/hsi_learning/data.py)  
  负责数据加载、PCA、patch 提取和数据集封装。
- [src/hsi_learning/models/hybridsn.py](../src/hsi_learning/models/hybridsn.py)  
  定义 `HybridSN` 模型。
- [src/hsi_learning/engine.py](../src/hsi_learning/engine.py)  
  负责训练与验证循环。
- [src/hsi_learning/evaluation.py](../src/hsi_learning/evaluation.py)  
  负责指标计算、整图推理与结果导出。
- [src/hsi_learning/utils.py](../src/hsi_learning/utils.py)  
  负责随机种子、目录管理和基础工具函数。

## 5. 输出约定

- notebook 和脚本的实验结果默认写入 `results/`
- `Indian Pines` 可视化结果统一使用 `nipy_spectral`
- 预测图、训练曲线、分类报告、指标 JSON 尽量保持同一输出风格
