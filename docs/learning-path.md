# HSI-Learning 学习路径

这个仓库的目标不是只放实验文件，而是逐步形成一套适合高光谱学习入门到进阶的内容路径。当前最合理的推进方式，不是推翻现有 notebook，而是在保留学习体验的基础上，把可复现实验和可复用代码逐步沉淀出来。

## 当前建议的学习顺序

1. 先理解高光谱数据长什么样。
   从 `notebooks/01_data_reading_and_visualization.ipynb` 入手，认识数据立方体、波段、标签图、伪彩色可视化和 Ground Truth。
2. 再建立传统机器学习 baseline。
   通过 `notebooks/02_svm_baseline.ipynb` 先完成从像素特征到分类器的最短闭环，理解训练集、测试集、混淆矩阵和分类报告。
3. 然后进入经典深度学习 baseline。
   用 `notebooks/03_hybridsn_baseline.ipynb` 和 `scripts/train_hybridsn.py` 理解 PCA、Patch 构造、3D/2D 卷积组合以及整图推理。
4. 最后补齐不同建模视角。
   继续阅读 `04_1d_cnn_teaching.ipynb`、`05_2d_cnn_teaching.ipynb`、`06_3d_cnn_teaching.ipynb`、`07_transformer_teaching.ipynb` 和 `08_ssrn_teaching.ipynb`，建立不同模型对高光谱数据的理解方式。

## 当前仓库和未来课程形态的对应关系

- `notebooks/01_data_reading_and_visualization.ipynb`
  对应“入门”阶段，重点是认识数据与标签。
- `notebooks/02_svm_baseline.ipynb`
  对应“基础 baseline”阶段，重点是把经典方法跑通。
- `notebooks/03_hybridsn_baseline.ipynb`
  对应“深度学习进阶”阶段，重点是理解空间-光谱联合建模。
- `notebooks/04_1d_cnn_teaching.ipynb`
  对应“光谱序列建模入门”阶段，重点是理解 1D CNN。
- `notebooks/05_2d_cnn_teaching.ipynb`
  对应“空间建模入门”阶段，重点是理解 2D CNN。
- `notebooks/06_3d_cnn_teaching.ipynb`
  对应“联合空间-光谱建模”阶段，重点是理解 3D CNN。
- `notebooks/07_transformer_teaching.ipynb`
  对应“注意力建模入门”阶段，重点是理解光谱 Transformer。
- `notebooks/08_ssrn_teaching.ipynb`
  对应“经典残差结构”阶段，重点是理解 SSRN 的谱域和空间联合残差建模。
- `scripts/train_hybridsn.py`
  对应“工程化复现”阶段，重点是把 notebook 里的实验逻辑沉淀成脚本。
- `src/hsi_learning/`
  对应“长期维护”阶段，重点是逐步形成可扩展的代码库。

## 现在就可以落地的章节规划

### 第一层：入门

- 高光谱图像是什么
- `.mat` 数据结构怎么看
- 波段、像素、类别分别代表什么
- Indian Pines 数据集的基本特征

### 第二层：基础实验

- 训练集、验证集、测试集如何划分
- OA、AA、Kappa 各自说明什么
- 用 SVM 跑一个最小 baseline
- 输出分类图并和 Ground Truth 对比

### 第三层：深度学习

- 为什么要先做 PCA
- 为什么要切 patch
- HybridSN 为什么同时用 3D 和 2D 卷积
- 模型训练、验证、选最好 checkpoint 的基本流程
- 如何做整图推理和生成分类报告

### 第四层：工程化与进阶

- 把 notebook 逻辑拆到 `src/` 和 `scripts/`
- 统一结果输出目录
- 固定随机种子
- 形成可重复运行的 baseline
- 逐步扩展更多模型和更多任务

## 下一步建议

最值得继续推进的是这四件事：

1. 给现有三个 notebook 增加更清晰的教学说明和章节划分。
2. 在 `src/hsi_learning/` 下继续补传统机器学习 baseline 模块。
3. 为 `results/` 约定统一的输出格式，沉淀样本划分、训练曲线、指标和预测图。
4. 再引入一个比 HybridSN 更轻的深度学习 baseline，降低初学者第一次复现的门槛。
