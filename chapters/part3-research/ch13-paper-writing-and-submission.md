# 第 13 章 论文写作与投稿

> **Paper Writing & Submission**

状态：✅ 已完成

---

## 本章定位

把第 12 章的实验变成一篇可投稿的论文。写作不是实验结束后的"整理"，而是与实验并行的**第二战场**：结构先于内容、图表先于文字、协议先行于数字。本章给三样东西：IMRaD 各节的写法与审稿人视角、IEEE LaTeX 工程的完整骨架（**可编译交付物**：`chapters/part3-research/latex/paper_skeleton.tex`）、投稿与返修的全流程。学完即可把课程记分板上的任何一行实验展开成论文骨架。

## 学习目标 / Learning Objectives

- 掌握 IMRaD 结构与各节的写作任务，建立审稿人视角（每节要回答什么问题）
- 会写"问题—缺口—贡献"三段式引言与标准贡献点
- 会搭 IEEEtran LaTeX 工程并遵守三条排版纪律
- 了解投稿全流程：期刊选择、arXiv 预印本、cover letter 与 rebuttal 的结构

---

## 13.1 论文结构与各节任务

IMRaD 不是格式惯性，而是一套**论证协议**——每节回答一个审稿人必问的问题：

**表 13-1**　各节任务与审稿人视角

| 节 | 回答的问题 | 审稿人最常见的拒稿理由 |
|---|---|---|
| Abstract | 论文值得读吗？ | 贡献模糊、数字缺席 |
| Introduction | 问题重要吗？缺口真实吗？贡献是什么？ | 缺口句引用不到位；贡献点与实验不对应 |
| Related Work | 你读过该读的吗？你的位置在哪？ | 只罗列不对比；贬低前人 |
| Method | 你的方法到底是什么？ | 符号未定义；总览图缺失；关键细节藏在实现里 |
| Experiments | 凭什么信你的数字？ | 协议不公平（第 12 章）；缺消融；缺与经典基线的对比 |
| Conclusion | 带走了什么？什么没做？ | 重复摘要；夸大结论 |

**写作顺序 ≠ 阅读顺序**。推荐写作顺序：**Method → Experiments → Introduction → Related Work → Abstract/Conclusion**——先写你最确定的部分（方法与实验是既成事实），最后写最需要全局视角的部分（摘要与引言是全文的压缩）。

## 13.2 摘要与引言

### 引言的三段式模板

1. **问题段**（为什么重要）：高光谱分类的背景 + 一个具体的痛点（如"稀有类别与标注成本"）；
2. **缺口段**（别人做到哪、缺什么）：2–3 句，每句带引用——"3D 卷积联合建模谱空特征 [Chen2016, Zhong2018]，但计算代价随深度快速增长；2D 卷积高效却丢失光谱细节 [Makantasis2015]。**如何在两者间取得平衡仍是开放问题。**"——最后一句就是第 11 章说的"入场券"；
3. **贡献段**（本文做了什么）：`The main contributions of this paper are summarized as follows:` + 3 条 bullet。

### 贡献点的三条纪律

- **可证伪**：每条贡献对应实验节的某张表/图（审稿人会逐条找）；写"我们提出了 X 模块"就必须有 X 的消融行（第 12 章的消融矩阵就是为此准备的）；
- **量化或定性二选一，别含糊**：好——"在 IP 上以 10% 训练率达到 96.66% OA，较 3D 基线提升 2.1 点"；坏——"取得了很好的效果"；
- **不超过 4 条**：超过 4 条通常说明把实现细节也当成了贡献。

摘要 = 引言的 150 词压缩：背景一句、缺口一句、方法两句（含机制关键词）、数字一句（OA/AA/Kappa + 数据集 + 协议）、意义一句。**数字必须出现在摘要里**——没有数字的摘要在遥感领域会被直接怀疑。

## 13.3 相关工作与方法

### Related Work：从矩阵到文字

第 11 章的对比矩阵在这里兑现：每段 = 一个方法家族（卷积/混合/残差/注意力——第 10 章图 10-4 的分组就是现成的段落划分），段内按思想线索而非年代展开，**每段最后一句指明该家族未解决的问题**——四段的四个"未解决"汇合处，就是你的方法登场的位置。严禁两种写法：只罗列不对比（"XX 提出了…。YY 提出了…。"）；贬低前人（"XX 方法存在明显缺陷"→ 改为 "XX 方法假设……，这在小样本设置下难以满足"）。

### Method：符号系统与总览图

三条纪律：

1. **符号表先行**：每个符号第一次出现时给出定义；张量形状写全（第 8 章表 8-1 就是现成模板——无 padding 导致的 25→17 这类细节，审稿人一定会核对）；
2. **总览图必备**：一张方法总览图（结构 + 数据流 + 创新点高亮）承担一半的说服力——参考本课程图 8-1 / 图 9-1 的画法：真实形状、枢纽位置高亮、残差/注意力连接显式画出；
3. **创新点前置**：把最核心的机制放在方法节第一小节讲透，其余按数据流顺序展开；实现细节（如 padding 选择）放实现细节段或脚注，不与核心机制抢版面。

## 13.4 实验与图表

实验节的固定骨架（本课程全部素材可直接搬入）：

1. **Datasets and Protocols**：数据集表（尺寸/波段/类别/样本量——第 1 章表 + 类别分布图）+ 协议表（划分/预处理/预算/种子——第 12 章 run_config 的论文形态）；
2. **Implementation Details**：框架、硬件、epochs/lr/优化器——与 `run_config.json` 逐项对应；
3. **Main Results**：主表（第 12 章 booktabs 模板：三线表、精度统一、并列最优都加粗、协议写进表注）+ 训练曲线/收敛分析；
4. **Ablation Study**：消融表（一次一变量、mean ± std）——第 12 章的 SIR/残差消融即成品；
5. **Analysis / Discussion**：混淆矩阵分析（第 2 章的三步读法）、失败案例（稀有类）、注意力/特征可视化（第 10 章）——**解释性小节是区分"跑分论文"与"有洞察论文"的分水岭**。

图的三条纪律（全课程约定，直接迁移）：预测图统一 `nipy_spectral` + 背景掩除；训练曲线标注协议锚线与 best epoch；热力图单色系 + 计数标注。**每张图必须在正文被引用且被讨论**——不被讨论的图会被审稿人当作凑数。

## 13.5 LaTeX 工程

**交付物**：[`chapters/part3-research/latex/paper_skeleton.tex`](latex/paper_skeleton.tex)——可编译的 IEEEtran 骨架（Overleaf 直接导入，或本地 TeX Live `pdflatex paper_skeleton.tex`），内含：

- IEEEtran journal 模板结构与全部节的注释模板（每节开头是"本节任务 + 写法要点"，正文是占位符）；
- 第 12 章 booktabs 主表模板（已填入课程协议 C 真实数字，可直接替换）；
- 算法伪代码（`algorithm`/`algorithmic`）与跨引用的规范用法；
- 投稿前 checklist 注释块（编译零错误、引用无 "?"、图表全被引用、协议表注完整）。

工程纪律：**先搭骨架再填肉**（每节先写小标题 + 一句话提纲，全文有了再逐节展开）；交叉引用全部用 `\ref`/`\cite`，禁止手写编号；图表文件与 `.tex` 分目录管理（`figs/`）；每完成一节编译一次，错误早暴露。

## 13.6 投稿与返修

- **期刊选择**：第 11 章表 11-1 的版图反过来用——方法新颖度高、实验完整选 TGRS；短平快的单一创新选 GRSL 快报；应用/系统向选 JSTARS 或 ISPRS Journal。选刊三看：你的贡献类型、论文篇幅（GRSL 上限 5 页）、目标读者；
- **arXiv 预印本**：投稿后（或 before）挂 arXiv 占先权与传播；注意部分期刊对预印本版本更新的规定；
- **Cover letter**：三句话——论文解决什么、为什么适合该刊（引用该刊近期相关文章）、原创性声明；
- **Rebuttal**：审稿意见回应信的结构 = 逐条编号回应 + 开头一段总结主要修改。写法三原则：**先谢后答、有问必答、能用实验说话就不用形容词**——审稿人要求补的对比实验， rebuttal 里给出新表比任何辩护都有力（第 12 章的可复现工程让你能在 rebuttal 期内真的补出来）；
- **Major/Minor revision**：major = 逐条修改说明（response letter）+ 修改稿高亮；拒绝 ≠ 结束：根据意见补实验后可改投（换刊需说明，一稿多投是红线）。

---

## 配套实操 / Hands-on

- `chapters/part3-research/latex/paper_skeleton.tex` —— 可编译骨架；练习：把课程记分板（协议 C）填进主表，把第 12 章消融结果填进消融表，为"HybridSN + 谱注意力"假想的贡献写三条 bullet；
- 练习 1（引言）：为本章假想论文写三段式引言（每段不超过 5 句），缺口句的引用从第 11 章 Zotero 库里取；
- 练习 2（图表）：把 `results/hybridsn/IP/` 的预测图与训练曲线按 13.4 节规范导出为论文级图（300 dpi、字号可读、无多余白边）；
- 练习 3（rebuttal 演练）：假设审稿人问"为什么不与 Transformer 比较？"——用第 10 章的负结果与协议 C 记分板写一段 150 词的回应。

## 本章要点 / Key Takeaways

- 中文：IMRaD 是论证协议——每节回答一个审稿人必问的问题，拒稿理由表就是写作任务清单；写作顺序 Method → Experiments → Introduction → Related Work → Abstract，先写确定的再写全局的；引言三段式（问题—缺口—贡献），贡献点可证伪且逐条对应实验表；实验节五件套（数据集协议/实现细节/主表/消融/分析），表注写协议、图必须被讨论；LaTeX 纪律（先骨架后填肉、全交叉引用、每节一编译），投稿端期刊选择看贡献类型与篇幅，rebuttal 先谢后答、用新实验代替形容词。
- English: IMRaD is an argumentation protocol — every section answers a question a reviewer will ask, and the desk-reject table doubles as your writing checklist. Write Method → Experiments → Introduction → Related Work → Abstract; the introduction runs problem–gap–contribution with falsifiable, table-backed contribution bullets. The experiments section ships five pieces (datasets & protocols, implementation, main table, ablation, analysis); protocols live in table captions and every figure must be discussed. LaTeX discipline: scaffold first, cross-reference everything, compile per section; for submission, match the venue to your contribution type, and answer reviews with new experiments rather than adjectives.

## 自测题 / Self-check

1. 写作顺序为什么是 Method 最先、Abstract 最后？如果反过来写会发生什么？
2. 贡献点"我们提出了一个注意力模块并取得了很好的效果"违反了哪两条纪律？改写成合格版本（数据可从课程记分板取）。
3. 三线表的三条排版纪律各针对审稿人的什么行为？
4. Rebuttal 里审稿人质疑"对比不公平——你的方法训了 40 epochs，基线只有 20"。用第 12 章的哪个概念回应？写出回应的核心两句。

<details>
<summary><strong>参考答案（先自己回答再看）</strong></summary>

1. Method 与 Experiments 是既成事实（实验已经做完、数字已经存在），写作最不容易返工；Abstract/Introduction 需要全文视角（贡献的取舍、数字的挑选），最后写才能与实际内容对齐。反过来写：引言承诺的贡献会随方法节的推进漂移，摘要里的数字在实验定稿后作废——返工量最大的顺序。
2. 违反"可证伪（无对应实验）"与"量化或定性二选一（'很好'含糊）"。改写："We insert a spectral attention module into HybridSN's 3-D stage. Under protocol C on Indian Pines, the modified model reaches 97.1% OA and 94.0% AA, improving the HybridSN baseline by +0.4/+0.7 points (3 seeds, mean ± std)（数字为示意，须由真实消融产生——第 12 章规范）。"
3. 精度统一 → 审稿人横向比较行间数字（不统一会被当作隐瞒）；并列最优都加粗 → 防止"漏标对手最优"的指控（诚信信号）；协议写进表注 → 审稿人核对公平时第一眼看表注，免去翻正文。
4. 用第 12 章的"同预算"原则回应：核心两句——"All methods in Table X were trained under the identical 40-epoch budget with the same data splits and preprocessing; the budget was chosen because every model reached its validation plateau within it (best epochs reported in the appendix)." / "We additionally report the 20-epoch column for all models in the revised Table X, confirming the ranking is unchanged under a shorter budget."——用新表说话，不用形容词辩护。
</details>

## 延伸阅读 / Further Reading

- Simon Peyton Jones, "How to write a great research paper"（经典讲义；写作顺序与贡献点纪律的出处，网上可查）。
- IEEE Author Center（IEEEtran 模板、图表规范、投稿流程官方文档）。
- 本课程交付物：[`latex/paper_skeleton.tex`](latex/paper_skeleton.tex)——IEEEtran 可编译骨架（Overleaf/TeX Live 均可编译）。
- 第 11 章表 11-1（选刊版图）与第 12 章（本章实验节的素材来源）。
