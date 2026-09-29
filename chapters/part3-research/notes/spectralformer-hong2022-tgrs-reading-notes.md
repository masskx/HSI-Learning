# 精读笔记示例 ②：SpectralFormer

> 第 11 章九节模板的第二次实战——刻意选一篇**比 HybridSN 难一个量级**的前沿论文（Transformer + 预训练 + 密集消融），检验模板在难论文上的适用性。凡无法从记忆直接核实的信息均标 ⚠️ **待核对**。

## 1. 元信息

| 项 | 内容 |
|---|---|
| 标题 | SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers |
| 作者 | Danfeng Hong, Zi-Bang Wang, Jing Yao, Bing Zhang, et al.（⚠️ 待核对完整作者名单） |
| 发表 | IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2022 |
| DOI | ⚠️ 待核对（10.1109/TGRS.2021.3130716 量级） |
| 代码 | github.com/danfenghong/IEEE_TGRS_SpectralFormer（⚠️ 待核对） |
| 本课程对应 | 第 10 章（10.4 线索 1）；分组 token 化思想的来源 |

## 2. 一句话总结

把高光谱的**波段分组**做 token 化（group-wise embedding）+ **跨层自适应加权**（浅层光谱细节直通深层）+ **DSP 下游预训练**，证明纯 Transformer 在 HSI 上能匹配并超越 CNN——前提是解决数据饥饿。

## 3. 问题与缺口

- CNN 的局部归纳偏置在 HSI 上高效（课程第 6 章实测 95.83%），但**感受野受限**：长程光谱关系（相隔 50 波段的吸收谷关联）需要堆很深；
- vanilla Transformer 的 token 化对 HSI 有两个不适配：(a) 逐波段 token 化 N=200+ → O(N²) 代价爆炸（第 10 章实测 16.81M MACs）；(b) 光谱是**连续信号**，逐波段离散化丢失局部连续性先验；
- 缺口：能否设计一种**光谱感知的 token 化**，既享受 attention 的全局建模，又不丢 CNN 的局部先验？

## 4. 方法拆解

```
输入 (1, B, H, W)                    # B = 原始波段数（200 for IP）
├─ 分组嵌入 group-wise embedding      # B 个波段 → G 组（默认 G=4 组）
│   每组经 1×1 Conv + LN → G 个 token # N = G（而非 B！）→ O(G²) ≪ O(B²)
├─ + 可学习位置编码
├─ TransformerEncoder × L 层         # 标准多头自注意力 + FFN
│   └─ ★ 跨层注意力 cross-layer attn  # 每层输出与前面所有层自适应加权融合
├─ 展平 + 分类头
└─ DSP 预训练：掩码波段重建（mask 部分波段 → 重建）→ 下游微调
```

- **创新点 1：group-wise embedding**——把相邻波段分组（而非逐波段 token 化），G 个 token 即保留局部连续性先验（CNN 先验的 Transformer 版），又把 N 从 200+ 压到 G，attention 代价骤降。**这正是课程第 1 章"相邻波段高相关"的知识点变成设计决策的实例**；
- **创新点 2：跨层注意力**——标准 Transformer 每层只看上一层的输出（残差是加法），SpectralFormer 让每层**自适应加权所有前面层的输出**（类似 DenseNet 的思想搬进 attention），浅层光谱细节可以直通深层分类头；
- **创新点 3：DSP 预训练**——自监督掩码重建预训练 + 下游微调，直接攻击第 10 章诊断的"数据饥饿"问题（本章 Transformer 69.10% 的负结果的正解）；
- ⚠️ 待核对：分组的确切实现（1×1 conv 分组还是 reshape 分组？）、跨层注意力的权重机制（可学习标量还是 attention 权重？）、DSP 的掩码比例。

## 5. 实验协议表

| 协议项 | 论文设置 | 备注 |
|---|---|---|
| 数据集 | IP / PU / Houston 2013 | 与本课程 IP 重叠；Houston 未覆盖 |
| 划分 | ⚠️ 待核对（训练率与划分方式） | |
| 输入 | 原始波段（无 PCA）→ 分组 token | 与课程 2D CNN 的 PCA-12 路线不同 |
| 模型规模 | d_model / heads / layers ⚠️ 待核对 | |
| 训练预算 | ⚠️ 待核对 | |
| 结果统计 | ⚠️ 待核对（单次 or 多种子） | |
| 指标口径 | OA / AA / Kappa | 与本课程一致 |

## 6. 结果与声称

- 论文声称：在 IP / PU / Houston 2013 上与 CNN 基线（含 HybridSN、SSRN）及 vanilla Transformer 全面比较，**在有预训练（DSP）时优势更明显**——正是"数据饥饿"问题的正面回应（⚠️ 待核对具体数字）；
- 关键消融：group-wise vs 逐波段 token 化、跨层注意力 vs 标准残差、有/无 DSP——⚠️ 待核对消融表的具体数字。

## 7. 可复现性 checklist

| 检查项 | 论文是否写明 | 影响 |
|---|---|---|
| 分组数 G | ⚠️ 待核对（默认 4？） | 高：G 决定 token 数与 O(G²) |
| 划分比例 | ⚠️ 待核对 | 高 |
| 种子 | ⚠️ 待核对 | 中 |
| 预训练数据量 | ⚠️ 待核对 | 高（DSP 是主要卖点） |
| 代码开源 | 是（github.com/danfenghong/...） | 低（有代码可查） |

## 8. 缺点与可扩展点

1. **分组是均匀的**——相邻波段高度相关（第 1 章），均匀分组合理；但**非均匀分组**（按信息量/相关性自适应分组）是天然扩展点；
2. **group-wise 嵌入丢掉了组内精细结构**——组内再用轻量 1D CNN 提取局部谱形（"CNN 先验 + attention 全局"的混合，HybridSN 思想的 Transformer 版）；
3. **DSP 只做了掩码重建**——对比学习（SimCLR/MoCo 风格）在 HSI 上的预训练是开放方向；
4. **计算量**——即使 G ≪ B，多层 encoder + 跨层注意力的计算仍不便宜（未报告 FLOPs 对比？⚠️ 待核对）。

## 9. 与我的想法的联系

- **与本课程 2D CNN 的互补**：2D CNN（patch 9，OA 95.83%）用 CNN 先验 + 小数据取胜；SpectralFormer 用分组 attention + 预训练取胜。**两者的混合**（CNN 局部特征 + 分组 attention 全局交互）是一个自然的 Capstone 选题；
- **DSP 的启示**：第 10 章的"数据饥饿"负结果（69.10%）可以通过自监督预训练缓解——这是从"诊断问题"到"解决问题"的完整闭环，也是基础模型方向的入口；
- **与第 9 章 SIR 的对照**：SIR 把"patch 内光谱一致"写进损失（CNN 时代的先验注入），DSP 把"波段可重建"写进预训练目标（Transformer 时代的先验注入）——**先验注入的形式从损失函数演化为预训练目标**，这是 2018→2022 的方法论演化主线。
