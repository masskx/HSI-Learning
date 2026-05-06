# HSI-Learning 路线图

目标不是只放几个实验文件，而是把仓库逐步打造成一个“可学习、可复现、可扩展”的高光谱学习仓库。

## 阶段 1：把基础打稳

目标：

- 明确仓库定位
- 补齐依赖说明
- 分离源码与实验产物
- 保证现有 notebook 可以顺畅运行

已开始：

- README 重写
- 代码总览文档
- 路线图文档
- 数据派生脚本
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

## 阶段 3：把实验做成可复现工程

目标：

- 固定随机种子
- 统一数据划分逻辑
- 统一模型保存路径
- 统一实验结果目录
- 用脚本替代 notebook 里的重复逻辑

建议目录形态：

```text
HSI-Learning/
├─ dataset/
├─ notebooks/
├─ scripts/
├─ src/
│  └─ hsi_learning/
│     ├─ data/
│     ├─ models/
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

1. 先把现有 3 个 notebook 的说明补全。
2. 把传统机器学习 CSV 生成逻辑脚本化。
3. 把 `HybridSN` 的数据、模型、训练拆成独立模块。
4. 统一 `results/` 输出格式，不再混入主仓库。
5. 增加一个最小可复现训练脚本，替代纯手工 notebook 训练。
