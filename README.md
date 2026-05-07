# HSI-Learning

面向高光谱图像分类学习者的教学型仓库。当前内容围绕 `Indian Pines` 数据集组织，覆盖数据读取、传统机器学习基线、`HybridSN` 工程化实现，以及多种深度学习教学 notebook。

## Project Structure

```text
HSI-Learning/
├─ dataset/                  # Indian Pines 数据与说明
├─ notebooks/                # 教学 notebook
├─ scripts/                  # 脚本化训练与数据处理入口
├─ src/hsi_learning/         # 可复用代码模块
├─ docs/                     # 学习路径与代码结构说明
├─ results/                  # 本地实验输出
└─ requirements.txt
```

## Learning Path

1. [01_data_reading_and_visualization.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/01_data_reading_and_visualization.ipynb)  
   认识 `Indian Pines` 的数据立方体、波段、伪彩色图和标签图。
2. [02_svm_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/02_svm_baseline.ipynb)  
   用传统机器学习建立像素级分类 baseline。
3. [03_hybridsn_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/03_hybridsn_baseline.ipynb)  
   理解 `HybridSN` 的 PCA、patch、训练与整图推理流程。
4. [04_1d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/04_1d_cnn_teaching.ipynb)  
   从光谱序列建模角度理解 1D CNN。
5. [05_2d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/05_2d_cnn_teaching.ipynb)  
   从空间 patch 建模角度理解 2D CNN。
6. [06_3d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/06_3d_cnn_teaching.ipynb)  
   理解联合空间与光谱维度的 3D CNN。
7. [07_transformer_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/07_transformer_teaching.ipynb)  
   理解 Transformer 在高光谱分类中的基本用法。
8. [08_ssrn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/08_ssrn_teaching.ipynb)  
   理解 SSRN 的谱域残差块与空间残差块设计。
9. [09_hybridsn_exploring_3d_2d_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/09_hybridsn_exploring_3d_2d_teaching.ipynb)  
   理解三层 3D 卷积加一层 2D 卷积的 HybridSN Exploring 3D-2D 建模路线。

更多说明见 [docs/learning-path.md](/D:/JupyterWorkSpace/HSI-Learning/docs/learning-path.md)。

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

## Engineered Code

- [scripts/train_hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/train_hybridsn.py)
- [scripts/build_indian_pines_csv.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/build_indian_pines_csv.py)
- [src/hsi_learning/](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning)

## Notes

- `Indian Pines` 结果图当前统一使用 `nipy_spectral` 配色。
- `.ipynb_checkpoints` 和 `.jupyter_runtime/` 已忽略，不会提交到远端。
- `results/` 为本地实验输出目录。

## Docs

- [docs/learning-path.md](/D:/JupyterWorkSpace/HSI-Learning/docs/learning-path.md)
- [docs/codebase-overview.md](/D:/JupyterWorkSpace/HSI-Learning/docs/codebase-overview.md)
- [docs/roadmap.md](/D:/JupyterWorkSpace/HSI-Learning/docs/roadmap.md)
