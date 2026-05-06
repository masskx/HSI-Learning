# HSI-Learning

一个面向高光谱学习者的项目，目标是把 `Indian Pines` 为核心的实验内容整理成一条清晰的学习路径：从数据认识、传统机器学习，到 1D/2D/3D CNN 与 Transformer 的深度学习入门，再逐步过渡到可复现实验脚本和工程化代码。

## Project Structure

```text
HSI-Learning/
├─ dataset/                 # 数据集与说明
├─ notebooks/               # 教学 notebook
├─ scripts/                 # 可复现实验脚本
├─ src/hsi_learning/        # 工程化代码模块
├─ docs/                    # 路线、总览与学习说明
├─ results/                 # 本地实验输出（默认不跟踪）
└─ requirements.txt
```

## Learning Path

当前 notebook 已整理成统一命名的课程式结构：

1. [01_data_reading_and_visualization.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/01_data_reading_and_visualization.ipynb)
   认识 `Indian Pines`、`.mat` 数据结构、伪彩色图与 Ground Truth。
2. [02_svm_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/02_svm_baseline.ipynb)
   传统机器学习基线，帮助建立像素级分类的最短闭环。
3. [03_hybridsn_baseline.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/03_hybridsn_baseline.ipynb)
   `HybridSN` 深度学习流程，包含 PCA、Patch 构造、训练与整图推理。
4. [04_1d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/04_1d_cnn_teaching.ipynb)
   从光谱序列建模角度理解 1D CNN。
5. [05_2d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/05_2d_cnn_teaching.ipynb)
   通过 PCA + 空间 patch 理解 2D CNN。
6. [06_3d_cnn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/06_3d_cnn_teaching.ipynb)
   同时利用空间和光谱维度的 3D CNN 基线。
7. [07_transformer_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/07_transformer_teaching.ipynb)
   用光谱序列建模方式理解 Transformer 在高光谱分类中的用法。
8. [08_ssrn_teaching.ipynb](/D:/JupyterWorkSpace/HSI-Learning/notebooks/08_ssrn_teaching.ipynb)
   通过谱域残差块与空间残差块理解 SSRN 的经典两段式建模思路。

更细的学习说明见 [docs/learning-path.md](/D:/JupyterWorkSpace/HSI-Learning/docs/learning-path.md)。

## Quick Start

### 1. Install Dependencies

建议先根据你的 CPU / CUDA 环境安装 `PyTorch`，再安装其余依赖：

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

当前默认使用 `dataset/` 下的 `Indian Pines` 数据：

- `Indian_pines_corrected.mat`
- `Indian_pines_gt.mat`
- `Indian_pines.mat`

如果后续扩展到 `SA` 或 `PU`，需要将对应 `.mat` 文件补到 `dataset/`。

### 3. Generate Pixel CSV for Classical ML

```bash
python scripts/build_indian_pines_csv.py --output indian_pines_all.csv
```

### 4. Run the Script-Based HybridSN Baseline

```bash
python scripts/train_hybridsn.py --dataset IP --epochs 20
```

默认输出目录为 `results/hybridsn/IP/`，会产出样本划分、训练曲线、分类报告、预测图和最佳 checkpoint。

## Engineered Code

除了 notebook，这个项目已经开始沉淀可复用代码：

- [scripts/train_hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/train_hybridsn.py)
  脱离 notebook 运行 `HybridSN` 的最小脚本入口。
- [src/hsi_learning/](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning)
  目前包含数据读取、PCA、样本划分、训练循环、评估与 `HybridSN` 模型定义。

## Notes

- 分类结果图目前统一采用 `Indian Pines` 常见的 `nipy_spectral` 风格。
- `results/`、`paper/`、`发布图片/` 主要视为本地实验或资料目录，不作为核心源码结构的一部分。
- 仓库正在从“notebook 为主”逐步过渡到“notebook + script + src”的混合结构。

## Docs

- [docs/codebase-overview.md](/D:/JupyterWorkSpace/HSI-Learning/docs/codebase-overview.md)
- [docs/learning-path.md](/D:/JupyterWorkSpace/HSI-Learning/docs/learning-path.md)
- [docs/roadmap.md](/D:/JupyterWorkSpace/HSI-Learning/docs/roadmap.md)
