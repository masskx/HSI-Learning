# HSI-Learning 学习路径

这个仓库当前按“先认识数据，再建立 baseline，再进入深度学习模型”的顺序组织内容。

## 建议学习顺序

1. 从 [01_data_reading_and_visualization.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/01_data_reading_and_visualization.ipynb) 入手。  
   目标是理解高光谱数据立方体、波段、标签图、伪彩色显示和 Ground Truth。
2. 阅读 [02_svm_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/02_svm_baseline.ipynb)。  
   先建立传统机器学习 baseline，理解训练集、测试集、分类报告和结果图。
3. 阅读 [03_hybridsn_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/03_hybridsn_baseline.ipynb) 并对照 [scripts/train_hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/train_hybridsn.py)。  
   重点理解 PCA、patch 提取、3D/2D 卷积组合、训练验证和整图预测。
4. 继续阅读不同教学模型 notebook：  
   `04_1d_cnn_teaching.ipynb`、`05_2d_cnn_teaching.ipynb`、`06_3d_cnn_teaching.ipynb`、`07_transformer_teaching.ipynb`、`08_ssrn_teaching.ipynb`、`09_hybridsn_exploring_3d_2d_teaching.ipynb`。

## 当前 notebook 对应关系

- `notebooks/01_data_reading_and_visualization.ipynb`
  对应“数据入门”。
- `notebooks/02_svm_baseline.ipynb`
  对应“传统机器学习 baseline”。
- `notebooks/03_hybridsn_baseline.ipynb`
  对应“经典深度学习高光谱 baseline”。
- `notebooks/04_1d_cnn_teaching.ipynb`
  对应“光谱序列建模入门”。
- `notebooks/05_2d_cnn_teaching.ipynb`
  对应“空间建模入门”。
- `notebooks/06_3d_cnn_teaching.ipynb`
  对应“谱空联合卷积建模”。
- `notebooks/07_transformer_teaching.ipynb`
  对应“注意力建模入门”。
- `notebooks/08_ssrn_teaching.ipynb`
  对应“经典残差高光谱结构”。
- `notebooks/09_hybridsn_exploring_3d_2d_teaching.ipynb`
  对应“3D-2D 联合卷积结构”。

## 学习时建议重点关注

- `.mat` 数据如何变成可训练样本
- 为什么要做 PCA
- 为什么高光谱分类常用 patch
- OA、AA、Kappa 的意义
- 不同模型分别强调光谱、空间还是谱空联合信息
- 训练结果如何保存成曲线、报告和预测图
