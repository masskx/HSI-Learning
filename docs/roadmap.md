# HSI-Learning 路线图

目标不是只放几个实验文件，而是把仓库逐步打造成一个“可学习、可复现、可扩展”的高光谱学习仓库。

## 阶段 1：把基础打稳

目标：

- 明确仓库定位
- 补齐依赖说明
- 分离源码与实验产物
- 保证现有 notebook 可以顺畅运行

已经完成：

- README 重写
- 代码总览文档
- 路线图文档
- 学习路径文档
- 数据派生脚本
- `HybridSN` 脚本入口
- `src/hsi_learning/` 最小工程化骨架
- 实验产物忽略规则

## 阶段 2：形成入门学习路径

建议内容顺序：

1. 高光谱基础概念
2. 数据集介绍与 `.mat` 结构
3. 光谱立方可视化
4. 标签与评价指标
5. 传统机器学习分类
6. 深度学习入门模型

建议交付物：

- `01_data_reading.ipynb`
- `02_visualization.ipynb`
- `03_metrics_and_splits.ipynb`
- `04_svm_baseline.ipynb`
- `05_hybridsn_baseline.ipynb`

当前状态：

- 内容主线已经清楚，但还没有把现有 notebook 重新整理成课程化命名和章节化结构。

## 阶段 3：把实验做成可复现工程

目标：

- 固定随机种子
- 统一数据划分逻辑
- 统一模型保存路径
- 统一实验结果目录
- 用脚本替代 notebook 里的重复逻辑

当前进度：

- `HybridSN` 已经开始落地到 `src/` 和 `scripts/`
- `results/hybridsn/<dataset>/` 已经形成最小输出约定
- 数据加载、Patch 构造、训练、评估已经具备可复用模块

建议目录形态：

```text
HSI-Learning/
├─ dataset/
├─ notebooks/
├─ scripts/
├─ src/
│  └─ hsi_learning/
│     ├─ models/
│     ├─ baselines/
│     ├─ training/
│     ├─ evaluation/
│     └─ visualization/
├─ docs/
└─ requirements.txt
```

## 阶段 4：丰富模型和任务

传统基线建议补齐：

- SVM
- Random Forest
- KNN
- Logistic Regression

深度学习建议补齐：

- 1D CNN
- 2D CNN
- 3D CNN
- HybridSN
- Transformer 类模型

任务拓展建议：

- 分类
- 波段选择
- 降维
- 目标检测
- 变化检测
- 域适应 / 小样本学习

## 阶段 5：形成“教程 + 代码 + 结果”闭环

理想状态：

- 每个章节有 notebook 讲解
- 每个模型有脚本可复现
- 每个实验有结果记录
- 每个数据集有说明文档
- 每个阶段都能独立学习

## 当前最值得优先做的 5 件事

1. 给现有 3 个 notebook 增加更清晰的教学说明和章节划分。
2. 把传统机器学习训练也补成脚本入口，而不是只有 CSV 生成脚本。
3. 再补一个更轻量的深度学习 baseline，降低第一次复现门槛。
4. 给 `results/` 设计统一输出规范和 benchmark 汇总格式。
5. 逐步把 notebook 目录和命名整理成真正的课程路径。
