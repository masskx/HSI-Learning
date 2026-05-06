# HSI-Learning

一个面向高光谱学习者的仓库，目标是把内容逐步整理成“从入门到进阶”的学习路径：先理解数据，再完成传统机器学习基线，最后过渡到深度学习模型复现与扩展。

## 当前仓库里有什么

- [数据读取和可视化.ipynb](/D:/JupyterWorkSpace/HSI-Learning/数据读取和可视化.ipynb)
  负责读取 `Indian Pines` 数据、查看 `.mat` 结构、展示 RGB 合成图与标注图。
- [机器学习方法分类.ipynb](/D:/JupyterWorkSpace/HSI-Learning/机器学习方法分类.ipynb)
  以 `SVM` 为主，完成数据展开、训练测试划分、混淆矩阵、分类图保存。
- [HybridSN.ipynb](/D:/JupyterWorkSpace/HSI-Learning/HybridSN.ipynb)
  当前最完整的深度学习实验流程，包含数据加载、PCA、Patch 构造、`HybridSN` 模型、训练、验证、整图推理与指标计算。
- [scripts/build_indian_pines_csv.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/build_indian_pines_csv.py)
  用脚本生成传统机器学习所需的像素级 CSV，不再手工维护派生文件。
- [scripts/train_hybridsn.py](/D:/JupyterWorkSpace/HSI-Learning/scripts/train_hybridsn.py)
  新增的最小可复现训练入口，可以脱离 notebook 完成 `HybridSN` 训练、验证、整图推理和指标导出。
- [src/hsi_learning/](/D:/JupyterWorkSpace/HSI-Learning/src/hsi_learning)
  新增的工程化代码包，先沉淀 `HybridSN` 相关的数据、模型、训练和评估逻辑。
- [dataset/](/D:/JupyterWorkSpace/HSI-Learning/dataset)
  当前已放入 `Indian Pines` 相关 `.mat` 数据与说明。

更细的代码盘点见 [docs/codebase-overview.md](/D:/JupyterWorkSpace/HSI-Learning/docs/codebase-overview.md)，学习顺序建议见 [docs/learning-path.md](/D:/JupyterWorkSpace/HSI-Learning/docs/learning-path.md)。

## 仓库定位

这个仓库后续会围绕三条线持续推进：

1. 学习线：高光谱数据、可视化、评价指标、经典模型、深度模型。
2. 复现线：把 notebook 里的实验整理成可重复运行的脚本和模块。
3. 工程线：逐步形成清晰目录、依赖说明、数据处理脚本和实验输出规范。

长期安排见 [docs/roadmap.md](/D:/JupyterWorkSpace/HSI-Learning/docs/roadmap.md)。

## 快速开始

### 1. 安装依赖

建议先按你的 CPU/CUDA 环境安装 `PyTorch`，再安装其余依赖：

```bash
pip install -r requirements.txt
```

### 2. 准备数据

当前默认使用 `dataset/` 下的 `Indian Pines` 数据：

- `Indian_pines_corrected.mat`
- `Indian_pines_gt.mat`
- `Indian_pines.mat`

如果要跑 `SA` 或 `PU`，还需要把对应的 `.mat` 数据补到 `dataset/`。

### 3. 生成传统机器学习所需的 CSV

```bash
python scripts/build_indian_pines_csv.py --output indian_pines_all.csv
```

### 4. 运行脚本版 HybridSN 基线

```bash
python scripts/train_hybridsn.py --dataset IP --epochs 20
```

默认输出目录为 `results/hybridsn/IP/`，会产出：

- `sample_report.txt`
- `run_config.json`
- `training_history.json`
- `training_curves.png`
- `classification_report.txt`
- `metrics.json`
- `prediction.jpg`
- `prediction_masked.jpg`
- 最优 checkpoint

如果本地没有安装 `spectral`，脚本仍会保存 `prediction.npy` 和 `prediction_masked.npy`，只是不会生成 `.jpg` 预览图。

## 当前代码能力概览

- 数据读取与可视化：已具备。
- 传统机器学习基线：已具备 `SVM` 流程。
- 深度学习基线：已具备 notebook 版和脚本版 `HybridSN`。
- 可复现实验结构：已完成第一步，`HybridSN` 开始从 notebook 抽离到 `src/` 与 `scripts/`。
- 模块化代码组织：已经建立最小骨架，但还需要继续扩展更多基线和更轻的教学入口。

## 当前最值得继续推进的方向

1. 给现有三个 notebook 增加更清晰的教学说明和章节划分。
2. 在 `src/hsi_learning/` 下继续补传统机器学习 baseline 模块。
3. 统一 `results/` 的输出格式，把数据划分、训练曲线、指标和预测图固定下来。
4. 增加一个更轻量的深度学习 baseline，降低初学者第一次复现门槛。

## 说明

- `results/`、`paper/`、`发布图片/` 这类目录目前视为本地材料或实验产物，不作为主仓库跟踪对象。
- 当前仓库仍然保留 notebook 中心的学习方式，但训练与评估逻辑已经开始渐进式迁移到脚本和模块，不做一次性大搬家。
