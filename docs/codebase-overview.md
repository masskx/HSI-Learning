# 当前代码总览

这个仓库现在正处在“保留 notebook 学习体验，同时逐步工程化”的阶段。核心内容仍然主要来自 3 个 notebook，但已经开始把最关键的深度学习 baseline 拆成脚本和模块，方便后续长期维护。

## 1. 数据读取和可视化

文件：

- [01_data_reading_and_visualization.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/01_data_reading_and_visualization.ipynb)

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

- [02_svm_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/02_svm_baseline.ipynb)
- [scripts/build_indian_pines_csv.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/build_indian_pines_csv.py)

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

- 训练逻辑仍主要写在 notebook 里
- 目前只有 SVM，一些常见基线还没补齐
- 样本划分和结果输出还没完全统一

## 3. HybridSN 深度学习流程

文件：

- [03_hybridsn_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/03_hybridsn_baseline.ipynb)
- [scripts/train_hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/train_hybridsn.py)
- [src/hsi_learning/models/hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/models/hybridsn.py)
- [src/hsi_learning/data.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/data.py)
- [src/hsi_learning/engine.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/engine.py)
- [src/hsi_learning/evaluation.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/evaluation.py)

当前内容：

- 支持 `IP / SA / PU` 数据集分支
- PCA 降维
- 训练集 / 验证集 / 测试集划分
- Patch 数据集构造
- `HybridSN` 3D + 2D 卷积模型
- 最优 checkpoint 选择
- 整图推理、分类图导出、`OA / AA / Kappa`

价值：

- 当前仓库中最完整的深度学习 baseline
- 已经从 notebook 中抽出一个最小可复现的脚本入口
- 为后续引入更多模型提供了统一的代码落点

当前问题：

- 目前只完成了 `HybridSN` 一条线的模块化
- 还没有更轻量的深度学习入门 baseline
- 还缺少更统一的配置体系和结果汇总表

## 4. 新增的工程化骨架

当前已经新增：

- [src/hsi_learning/__init__.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/__init__.py)
- [src/hsi_learning/utils.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/utils.py)
- [src/hsi_learning/data.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/data.py)
- [src/hsi_learning/models/](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/models)
- [src/hsi_learning/engine.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/engine.py)
- [src/hsi_learning/evaluation.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/evaluation.py)

这套结构目前承担的职责：

- `utils.py`
  负责目录创建、随机种子、设备解析和 JSON 持久化。
- `data.py`
  负责数据集加载、PCA、标签划分、PatchDataset 和 DataLoader。
- `models/hybridsn.py`
  负责 `HybridSN` 模型定义。
- `engine.py`
  负责训练循环、验证、checkpoint 和训练曲线保存。
- `evaluation.py`
负责整图推理、指标计算和预测图导出。

## 5. 新增的教学 notebook

当前还补充了三本轻量教学 notebook：

- [04_1d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/04_1d_cnn_teaching.ipynb)
- [05_2d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/05_2d_cnn_teaching.ipynb)
- [06_3d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/06_3d_cnn_teaching.ipynb)
- [07_transformer_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/07_transformer_teaching.ipynb)
- [08_ssrn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/08_ssrn_teaching.ipynb)

这些 notebook 的作用是：

- 降低第一次接触不同深度学习建模方式的门槛
- 保持统一的数据集、统一的结果图配色、统一的结果输出位置
- 让 notebook 成为真正可学习的入口，而不仅是实验草稿

## 6. 现在仓库的主要短板

- notebook 的教学说明还不够完整
- 传统机器学习还没有脚本化训练入口
- 缺少统一的实验配置模板和结果对比机制
- 只覆盖了 `HybridSN`，还没有形成多模型基线矩阵

## 7. 推荐的后续推进方向

### 先继续补最值得补的部分

- `src/hsi_learning/baselines/` 或等价的传统机器学习模块
- 一个更轻量的深度学习 baseline，例如 `1D CNN` 或 `2D CNN`
- 更清晰的 notebook 章节化说明
- 统一的实验汇总表或 benchmark 文档

### notebook 继续保留的角色

- notebook 继续作为教程和可视化入口
- 训练与评估逻辑逐步迁移到脚本和模块
- notebook 最终负责“讲解”和“调用”，不再承担全部实现
