# 当前代码概览

这个仓库目前由三部分组成：教学 notebook、脚本入口、可复用源码模块。

## 1. 教学 notebook

- [01_data_reading_and_visualization.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/01_data_reading_and_visualization.ipynb)
- [02_svm_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/02_svm_baseline.ipynb)
- [03_hybridsn_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/03_hybridsn_baseline.ipynb)
- [04_1d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/04_1d_cnn_teaching.ipynb)
- [05_2d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/05_2d_cnn_teaching.ipynb)
- [06_3d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/06_3d_cnn_teaching.ipynb)
- [07_transformer_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/07_transformer_teaching.ipynb)
- [08_ssrn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/08_ssrn_teaching.ipynb)
- [09_hybridsn_exploring_3d_2d_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/09_hybridsn_exploring_3d_2d_teaching.ipynb)

这些 notebook 的职责是教学解释、可视化展示和最小可运行示例。

## 2. 脚本入口

- [scripts/build_indian_pines_csv.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/build_indian_pines_csv.py)  
  将高光谱立方体展开为传统机器学习可用的样本表。
- [scripts/train_hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/train_hybridsn.py)  
  提供脚本化的 `HybridSN` 训练、验证、测试与结果导出流程。

## 3. 可复用源码

- [src/hsi_learning/data.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/data.py)  
  负责数据加载、PCA、patch 提取和数据集封装。
- [src/hsi_learning/models/hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/models/hybridsn.py)  
  定义 `HybridSN` 模型。
- [src/hsi_learning/engine.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/engine.py)  
  负责训练与验证循环。
- [src/hsi_learning/evaluation.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/evaluation.py)  
  负责指标计算、整图推理与结果导出。
- [src/hsi_learning/utils.py](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning/utils.py)  
  负责随机种子、目录管理和基础工具函数。

## 4. 输出约定

- notebook 和脚本的实验结果默认写入 `results/`
- `Indian Pines` 可视化结果统一使用 `nipy_spectral`
- 预测图、训练曲线、分类报告、指标 JSON 尽量保持同一输出风格
