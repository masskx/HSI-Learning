# 第17章 开集高光谱分类 / Open-Set HSI Classification

状态：协议与正文纠错完成；修正版代码、Notebook与输出待执行验收。

## 17.1 已知、未知与背景是三件事

训练只监督已知类，测试出现未知类；模型既需区分已知类，又需拒绝未知。GT=0是未标注背景，不是已知背景类别，更不能自动当作未知类真值。

教学默认留出原始类ID {1,7,9,16} 为未知，其余12类为已知。稀有类留出只是一个例子；不能把“未知”定义成“稀有”，也不能仅此一次实验推广到所有类别组合。

## 17.2 两种分数

分类器在已知类上监督训练，和第16章的ProtoNet encoder**不是同一个训练任务或同一份权重**。

- MSP unknown score：$s(x)=1-\max_k\mathrm{softmax}(z(x))_k$。
- 距离 unknown score：$s(x)=\min_k\|f(x)-c_k\|_2$，$c_k$为已知训练类的特征均值。

两者均越大越未知。普通softmax没有显式unknown输出槽，但置信度仍可用于OOD打分；“softmax不能检测未知”过于绝对。距离也不保证未知落在已知簇之外，必须测量。

## 17.3 先校准，再测试

1. 按已知类构造train/val/known-test，未知类只进入最终测试。
2. PCA和标准化只fit已知训练中心；训练以known val选择checkpoint。
3. 训练类原型的嵌入与标签同批收集；loader shuffle不能导致错配。
4. 在known val分数上取预定接受率（如95%）对应分位数。接受 $s\le\tau$，拒绝 $s>\tau$。
5. 冻结模型、原型、阈值后评估test。测试类标签不能用于选择阈值或方法超参。

95%接受率是校准目标，不保证测试接受率，也不控制未知召回。小验证集应报告阈值不确定性。随机同场景patch可能含未知地物无标签观测；即使中心和预处理只用known train，也不能宣称完全未暴露未知场景。

## 17.4 必须一起报告的指标和地图

| 指标 | 回答的问题 |
|---|---|
| 拒识前已知类分类准确率 | classifier能否区分已知类 |
| AUROC（unknown为正类） | 未知分数的排序能力 |
| 已知误拒率 | 多少可分类样本被拒绝 |
| 已知正确接受率 | 已知样本中最终正确且未拒识的比例 |
| 未知召回/漏检率 | 有多少未知被识别/被强制分成已知 |

拒识会影响已知类可用准确率；不能说“后处理不改变模型，所以不牺牲闭集能力”。AUROC也不是某个固定阈值下的最终分类准确率。

Notebook应并排给：open-set GT、MSP预测、距离预测、已知误拒/未知漏检/已知错分三张错误图。**背景黑色，未知/拒识白色**，已知类保持相同颜色和类ID；GT只能用于显示mask与评估，不能在预测中把真unknown直接涂白。

## 17.5 OpenMax：概念延伸而非已复现算法

OpenMax在训练集**正确分类样本**的激活上计算类均值激活向量（MAV），对到MAV的距离尾部拟合极值分布；测试时重标定高排名已知类激活，将部分质量分配到unknown，再形成含未知的输出。

它不是“对误分类样本拟合Weibull”，也不是这里的最近原型距离阈值。MAV层选择、距离定义、tail size、排名权重都需要与原论文/实现对应。本课只讲原理，未运行OpenMax，不提供伪复现成绩。

## 17.6 历史结果与新版演示

旧脚本结果（MSP AUC .6816、距离 .9561）属于旧的同场景设置；旧Notebook还存在shuffle后原型标签错配与test阈值问题。**旧图不能作为修正版结果，也不能据此说“距离是强、MSP是弱”的普遍结论。**

新版入口：

```bash
python scripts/train_openset.py --epochs 20
python scripts/train_openset.py --unknown-classes 1 7 --output-dir results/openset_two
```

输出权重、预处理数组、split、原型/测试分数、冻结阈值及hash；课堂默认demo读取可信包，不重新fit PCA或用test阈值。运行证据在本轮工具恢复后补验，不预写分数。

## 实践—研究连接

参见 [L09试录脚本](../../docs/teaching/lessons/09-open-set.md)。受控练习：接受率.95→.90，只用validation重校准，看误拒与漏检权衡；未知类组合改变后重新完整评估，不能只挑AUC最高组合。

## 自测与答案

1. 全部拒绝是否完美开集？未知召回可为1，但已知正确接受率为0，系统不可用。
2. 特征提取shuffle后能否按原标签分组？不能；必须保持配对。
3. 距离在所有encoder上都有效吗？不保证，特征尺度、多模态和类重叠影响结果。
4. 白色区域是否都正确？不是，也可能是known false rejection；需看错误图。

## Key Takeaways

开集识别不只是一条ROC曲线：**独立校准、拒识代价、类映射与空间结果同等重要**。

Unknown-score ranking and calibrated decisions answer different questions. Report known-class utility and unknown detection together; white pixels are rejections, not automatically correct detections.

## 来源

[Hendrycks & Gimpel 2017](https://arxiv.org/abs/1610.02136)；[Bendale & Boult 2016](https://arxiv.org/abs/1511.06233)。论文核验状态与教学边界见 [source-audit](../../docs/teaching/source-audit.md)。
