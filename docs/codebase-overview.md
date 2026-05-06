# 当前代码总览

这个仓库现在的核心代码基本都集中在 3 个 notebook 中，属于“先跑通实验，再逐步工程化”的状态。

## 1. 数据读取和可视化

文件：

- [数据读取和可视化.ipynb](/D:/JupyterWorkSpace/HSI-Learning/数据读取和可视化.ipynb)

当前内容：

- 读取 `Indian_pines_corrected.mat` 和 `Indian_pines_gt.mat`
- 查看 `.mat` 文件键名和数据形状
- 查看标签类别分布
- 展示 RGB 合成图和 Ground Truth

价值：

- 适合作为高光谱入门第一步
- 帮助理解高光谱数据不是普通 RGB 图像，而是三维光谱立方

当前问题：

- 代码主要是探索式 notebook，复用性较弱
- 还缺少对波段、类别、数据维度含义的系统说明

## 2. 传统机器学习分类

文件：

- [机器学习方法分类.ipynb](/D:/JupyterWorkSpace/HSI-Learning/机器学习方法分类.ipynb)

当前内容：

- 把高光谱立方展开成二维样本矩阵
- 生成 `indian_pines_all.csv`
- 使用 `train_test_split`
- 使用 `SVC(C=100, kernel='rbf')`
- 输出混淆矩阵和 `classification_report`
- 保存分类图到 `results/`

价值：

- 适合放在“传统机器学习基线”章节
- 让读者先建立 baseline 概念，再进入深度学习

当前问题：

- 数据生成和训练逻辑都写在 notebook 里
- CSV 属于派生文件，不应长期手工维护
- 目前只有 SVM，一些常见基线还没补齐

## 3. HybridSN 深度学习流程

文件：

- [HybridSN.ipynb](/D:/JupyterWorkSpace/HSI-Learning/HybridSN.ipynb)

当前内容：

- 支持 `IP / SA / PU` 数据集分支
- PCA 降维
- 训练集 / 验证集 / 测试集划分
- Patch 数据集构造
- `PatchSet(Dataset)` 与 `DataLoader`
- `HybridSN` 3D + 2D 卷积模型
- `Adam` 优化器与 `CrossEntropyLoss`
- 最优模型保存
- 整图推理、分类图导出、`OA / AA / Kappa`

价值：

- 当前仓库中最完整的可训练深度学习基线
- 可以作为后续模块化重构的主参考来源

当前问题：

- 参数、路径、数据集切换仍然偏手工
- 模型、数据、训练、评估没有拆分成独立模块
- notebook 输出较重，不利于版本管理

## 4. 现在仓库的主要短板

- 代码入口分散，主要靠 notebook 手工执行
- 数据派生文件与实验输出和源码混在一起
- 依赖说明和实验说明不足
- 没有统一的实验配置和结果记录结构

## 5. 推荐的后续拆分方向

### 先拆最值得拆的部分

- `scripts/build_indian_pines_csv.py`
- `src/hsi_learning/data/`
- `src/hsi_learning/models/`
- `src/hsi_learning/train/`
- `src/hsi_learning/eval/`

### 保留 notebook 的角色

- notebook 继续作为教程和可视化入口
- 训练与评估逻辑逐步迁移到脚本和模块
- notebook 最终负责“讲解”和“调用”，不再承担全部实现
