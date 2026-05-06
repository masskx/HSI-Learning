# HSI-Learning

一个面向高光谱学习者的仓库，目标是把内容逐步整理成“从入门到进阶”的学习路径：先理解数据，再完成传统机器学习基线，最后过渡到深度学习模型复现与扩展。

## 当前仓库里有什么

- [数据读取和可视化.ipynb](/D:/JupyterWorkSpace/HSI-Learning/数据读取和可视化.ipynb)
  负责读取 `Indian Pines` 数据、查看 `.mat` 结构、展示 RGB 合成图与标注图。
- [机器学习方法分类.ipynb](/D:/JupyterWorkSpace/HSI-Learning/机器学习方法分类.ipynb)
  以 `SVM` 为主，完成数据展开、训练测试划分、混淆矩阵、分类图保存。
- [HybridSN.ipynb](/D:/JupyterWorkSpace/HSI-Learning/HybridSN.ipynb)
  目前最完整的深度学习流程，包含数据加载、PCA、Patch 构造、`HybridSN` 模型、训练、验证、整图推理与评价指标。
- [dataset/](/D:/JupyterWorkSpace/HSI-Learning/dataset)
  当前已放入 `Indian Pines` 相关 `.mat` 数据与说明。

更细的代码盘点见 [docs/codebase-overview.md](/D:/JupyterWorkSpace/HSI-Learning/docs/codebase-overview.md)。

## 仓库定位

这个仓库后续会重点覆盖三条线：

1. 学习线：高光谱数据、可视化、评价指标、经典模型、深度模型。
2. 复现线：把 notebook 里的实验整理成可重复运行的脚本和模块。
3. 工程线：逐步形成清晰目录、依赖说明、数据处理脚本和实验输出规范。

长期路线见 [docs/roadmap.md](/D:/JupyterWorkSpace/HSI-Learning/docs/roadmap.md)。

## 快速开始

### 1. 安装依赖

建议先按你的 CPU/CUDA 环境安装 `PyTorch`，再安装其余依赖：

```bash
pip install -r requirements.txt
```

### 2. 准备数据

当前 notebook 默认使用 `dataset/` 下的 `Indian Pines` 数据：

- `Indian_pines_corrected.mat`
- `Indian_pines_gt.mat`
- `Indian_pines.mat`

### 3. 生成传统机器学习所需的 CSV

`机器学习方法分类.ipynb` 里会把光谱立方展开成 CSV。现在仓库不再直接跟踪这个派生文件，而是改成脚本生成：

```bash
python scripts/build_indian_pines_csv.py --output indian_pines_all.csv
```

## 当前代码能力概览

- 数据读取与可视化：已具备。
- 传统机器学习基线：已具备 `SVM` 流程。
- 深度学习基线：已具备 `HybridSN` 训练与推理。
- 可复现实验结构：还不完善，仍以 notebook 为主。
- 模块化代码组织：还未完成，是下一阶段重点。

## 接下来准备怎么做

### 第一阶段

- 稳定当前 notebook 流程。
- 把数据处理、训练、评估从 notebook 中抽出到脚本或 `src/` 模块。
- 给每个实验补依赖、输入输出说明和评价指标说明。

### 第二阶段

- 增加更多传统基线，例如 `RandomForest`、`KNN`、`1D/2D CNN`。
- 统一数据划分、随机种子、结果保存格式。
- 增加实验对比表和可视化图。

### 第三阶段

- 拓展到更先进的高光谱任务与模型。
- 完成“教程 + 代码 + 实验复现”的闭环。

## 说明

- `results/`、`paper/`、`发布图片/` 这类目录目前视为本地材料或实验产物，不作为主仓库跟踪对象。
- 当前仓库仍然是 notebook 中心结构，后续会渐进式重构，不会一次性大搬家。
