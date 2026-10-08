# 第16章 小样本高光谱分类 / Few-Shot HSI Classification

状态：正文纠错完成；修正版算法、Notebook和实验输出待执行验收。历史输出不可作为新协议成绩。

## 16.1 先说明“少”的是什么

5-way 5-shot 指**一个任务的支持集**有5类、每类5个标签，不是整个encoder仅训练过25个标注样本。meta-training 使用多少标签、是否预训练、测试类是否见过，必须另外报告。

- **same-class 少标注**：训练/验证/测试为同一组类的不同像元，研究同类分类的标注效率。
- **class-disjoint few-shot**：meta-train、meta-val、meta-test 类别不交，验证与测试任务各自提供support，再分类query。
- **同场景局限**：类别不交只保证监督类别不交，空间patch可能包含别类无标签观测；不等于跨场景/空间独立泛化。

本轮主示范使用显式IP类划分：train={2,3,5,6,10,11}，val={1,4,7,8,9}，test={12,13,14,15,16}。它是教学协议，不代表原论文基准。每类须有足够的互不重复support/query。固定候选类集合，不能在K变大时悄悄删掉难类。

## 16.2 原型网络的核心

支持集嵌入的类均值为原型：

$$c_k=\frac{1}{|S_k|}\sum_{(x,y)\in S_k}f_\theta(x).$$

采用平方欧氏距离得到logits与概率：

$$z_k(x)=-\|f_\theta(x)-c_k\|_2^2,\quad p(y=k|x)=\mathrm{softmax}(z(x))_k.$$

二维例：原型(1,0)、(9,0)，query=(4,0)，logits=(-9,-25)。分类不需要额外固定C类Linear头；新类支持集可以构造新原型，但其可泛化性需要未见类测试验证。

“只用标准batch就不可能学可迁移嵌入”是不成立的；监督预训练也是重要few-shot baseline。Episodic training的价值是让训练任务形式接近部署任务，并非唯一途径。

## 16.3 Episodic training 的完整步骤

1. 从meta-training候选类抽N类，每类K+Q个不同像元。
2. 前K个为support，其余为query；把全局类ID映射成episode局部标签0…N−1。
3. encoder产生嵌入，support建原型，query算损失。
4. 反向传播只更新encoder；独立meta-validation任务选择checkpoint。
5. 冻结encoder，在meta-test任务评估；query标签只计算最终指标。

代码：`src/hsi_learning/teaching.py` 的 `EpisodeSampler`、`prototype_logits`；训练入口 `scripts/train_protonet.py`。采样器不足N类时抛错，不静默缩小N-way。三个随机流独立，改变验证频次不能改变训练episode。

BatchNorm采用明确的推理统计策略，不用测试query批量更新统计。PCA/scaler只fit meta-training中心像元。数据预处理、类清单、episode像元位置随模型保存。

## 16.4 K-shot 对照与地图

用**同一冻结encoder**，固定类池与query任务，比较1/5/10-shot。支持集增加通常使均值估计更稳定，但不是每个任务必定提高；相关样本、类内多模态和异常值都可能破坏单调性。

Notebook显示support/query位置、二维手算、真实嵌入分类与任务精度分布。全图演示只使用当前episode的原型，所有像元被强制分配到可用的五类；GT仅用于显示/诊断。它不是16类OA，也不是额外的未见类成绩。

旧表85.12/92.29/92.33来自同类像元划分、分别训练的K设置及反复test监控，**保留历史记录但撤回“25标签达到92%、K=5已稳定”的科研结论**。修正版结果在执行后另存，不预填数字。

## 实操与研究写作

```bash
python scripts/train_protonet.py --episodes 200 --eval-episodes 30
python scripts/train_protonet.py --mode same-class --output-dir results/teaching_sameclass
```

讲课卡：[L08](../../docs/teaching/lessons/08-few-shot.md)。写作必须交代元训练标签预算、类划分、K/Q/N、重复采样依赖、预处理拟合范围、空间重叠、checkpoint选择与seed。

## 自测与答案

1. 5-way 5-shot是否只有25个训练标签？不是，只是当前任务support；encoder训练预算另计。
2. 给query样本加入support会怎样？造成评估泄漏，即使没有梯度更新。
3. K变大为何不一定更好？原型估计受样本代表性影响，独立同分布假设不总成立。
4. 原型网络和闭集精度能直接比较吗？必须相同类集合/测试任务/标签预算，不能比较不同难度任务的裸数。

## Key Takeaways

少样本教学的核心是**任务与标签预算清楚，support/query严格分开，原型计算可手算与可验证**。

Few-shot claims require explicit annotation budgets and class splits. Prototype means and squared distances are simple; trustworthy evaluation is the essential part.

## 来源

[Snell et al., Prototypical Networks, NeurIPS 2017](https://arxiv.org/abs/1703.05175)；[Vinyals et al., Matching Networks](https://arxiv.org/abs/1606.04080)。来源访问/核验状态见 [source-audit](../../docs/teaching/source-audit.md)。
