# HSI-Learning 路线图

## 当前优先事项：录课就绪（验收中，非已发布）

原有17章作为基础材料保留。当前按「基础闭集 → 经典方法 → 小样本 → 开集 → 科研写作」整理第一批11个问题驱动单元，重点是方法准确、公式与代码对应、预测与错误图、可复现演示，不继续堆模型或追分。

- **已准备**：11张讲课卡、L01/L09试录脚本、幻灯片提纲、环境入口、公共算法修正、自动测试与串行验收入口。
- **待运行验收**：新协议测试、4套演示资产、修正版00/15/16/17 notebook、根目录/子目录启动和图像目检。工具服务受阻时不把这些项目标为通过。
- **待来源核验**：论文贡献/教学简化边界；未经核实的SIR、DSP等原文归属已列入纠错，不作为录课事实。
- [完整交付状态](teaching/delivery-status.md) · [视频导航](teaching/course-map.md) · [录制清单](teaching/recording-checklist.md)
- 执行入口：`python scripts/prepare_recording.py --stage all`，保留每阶段日志，失败立即停止；不安装依赖、不提交、不推送。

> 下方“已完成”记录的是历史内容建设，不表示已通过本轮方法核验、执行验收和试录。发布资格以当前逐课检查表为准。

目标不是只放几个实验文件，而是把仓库逐步打造成一个「可学习、可复现、可扩展」的高光谱课程仓库：从数据基础到独立撰写和发表论文。

## 阶段 1：把基础打稳（已完成）

- 明确仓库定位、补齐依赖说明、分离源码与实验产物
- README 重写、代码总览文档、路线图文档、学习路径文档
- 数据派生脚本、`HybridSN` 脚本入口
- `src/hsi_learning/` 最小工程化骨架
- 实验产物忽略规则
- 9 个教学 notebook（数据可视化、SVM、HybridSN、1D/2D/3D CNN、Transformer、SSRN、HybridSN Exploring）

## 阶段 2：课程化改造——搭骨架（已完成）

- 参照章节化课程的组织方式，确定「基础篇 → 模型篇 → 科研篇」三部分共 14 章的课程大纲
- 建立 `chapters/part1-foundations/`、`part2-models/`、`part3-research/` 目录与统一章节模板
- 完成全部 14 章的章节骨架（本章定位、学习目标、内容规划、配套实操映射、要点提纲）
- README 重写为课程主页：双语目录总表、学习路径建议、进度状态
- 修复迁移遗留的旧绝对路径链接

## 阶段 3：逐章写实（已完成）

按编号顺序逐章完成正文写作，每一章的交付标准：

- 完整 markdown 正文（含推导、图示、文献引用）
- 配套 notebook 与正文对齐（必要时修订 notebook）
- 本章要点（双语）、自测题与延伸阅读定稿
- 更新 README 目录状态（🚧 → ✍️ → ✅）

写作顺序：ch01 → ch14。其中 ch04、ch08、ch12 写实时可顺带把 notebook 中与 `src/scripts` 重复的逻辑对齐。

当前进度（全部 ✅）：

- ✅ ch01 高光谱成像基础——完整正文（1.1–1.6 共六节）、四张基于真实数据的插图（波段切片、类别光谱曲线、伪彩色 + GT、类别分布）、配套自测题答案、插图生成脚本 `scripts/generate_ch01_figures.py`
- ✅ ch02 数据集、划分与评价指标——完整正文（2.1–2.5 共五节）、两张真实协议插图（30/10/60 划分可视化、RBF-SVM 混淆矩阵，复现 `notebooks/02` 协议的真实指标 OA 85.12% / AA 81.25% / Kappa 0.8292）、共享插图工具 `scripts/_chfigure_utils.py` 与生成脚本 `scripts/generate_ch02_figures.py`、实验规范五条
- ✅ ch03 传统机器学习基线——完整正文（3.1–3.5 共五节）、三张真实运行插图（四模型整图预测对比、SVM C×gamma 网格、PCA 方差解释与主成分图）；四基线双协议（80/20 与 10%）真实指标；建立课程记分板 `docs/benchmark.md`；生成脚本 `scripts/generate_ch03_figures.py`
- ✅ ch04 从像素到 Patch——完整正文（4.1–4.5 共五节，基础篇收官）、三张真实数据插图（patch 邻域对比、三种 patch size 的纯度分布——25×25 均值仅 0.43 的关键发现、真实训练样本张量可视化）、`PatchDataset` 逐行精读、生成脚本 `scripts/generate_ch04_figures.py`
- ✅ ch05 1D CNN——完整正文（5.1–5.5 共五节，模型篇开篇）、四张真实运行插图（卷积=滑动滤波器、训练曲线含"训练不足"诊断、混淆矩阵、整图预测图）；新建可复现训练脚本 `scripts/train_1d_cnn.py`（协议 C：10/10/80 seed 42，真实成绩 1D CNN 15ep 61.10% / 60ep 75.27% vs SVM 锚点 80.70%，已录入记分板协议 C）
- ✅ ch06 2D CNN——完整正文（6.1–6.4 共四节）、四张真实运行插图（patch 与感受野增长、训练曲线、混淆矩阵、1D vs 2D 整图预测对比）；`scripts/train_2d_cnn.py`（2D CNN 10ep 70.01% / 40ep 95.83%，泄漏口径的定量分析：每测试 patch 平均含 ~8 个训练邻居；Oats 类 recall 恒为 0），已录入记分板协议 C
- ✅ ch07 3D CNN——完整正文（7.1–7.4 共四节）、四张真实运行插图（Conv3d 核覆盖范围、参数量 vs MACs 对比（3D 参数最少 19.4k 但计算 4.23M MACs 最贵）、三模型验证曲线、混淆矩阵）；`scripts/train_3d_cnn.py`（3D CNN 10ep 59.82% / 40ep 94.55%，稀有类单次运行不稳定的讨论），已录入记分板协议 C
- ✅ ch08 HybridSN——完整正文（8.1–8.5 共五节，工程化主线核心章）、四张真实运行插图（形状流含 3D→2D 枢纽重排、协议 B 训练曲线、从整图预测重建的混淆矩阵、整图预测图）；精读 `src/hsi_learning/models/hybridsn.py` 与 `train_hybridsn.py` 管线（表 8-2 输出约定）；协议 B 首行 99.20%（饱和口径分析）+ 协议 C HybridSN 40ep 96.66%（四模型演化线闭合，Oats 首次 1.000）
- ✅ ch09 SSRN——完整正文（9.1–9.5 共五节）、三张真实运行插图（双分支结构图、SIR 消融对比、混淆矩阵）；`scripts/train_ssrn.py` 补上论文的 SIR 正则并完成两次消融（20%/5% 训练率均为负结果，如实入板）；协议 D 新块（SSRN 10ep 94.92 / 40ep 98.73）+ 协议 C SSRN 行 97.87%（OA 冠军但稀有类全零，AA 仅 79.84——"OA 冠军 ≠ 全面冠军"）
- ✅ ch10 注意力与 Transformer——完整正文（10.1–10.4 共四节，模型篇收官）、四张真实运行插图（真实注意力矩阵 200×200 可视化、120ep 训练曲线含 SVM 锚线、混淆矩阵、模型篇协议 C 全模型总结图）；`scripts/train_transformer.py`（谱 Transformer 10ep 41.48 / 40ep 57.95 / 120ep 69.10——弱归纳偏置小样本吃亏的负结果教学）；协议 C 追加 Transformer 行
- ✅ ch11 文献检索与论文精读——完整正文（11.1–11.5 共五节，科研篇开篇）、文献地图时间线插图（课程 17 篇核心文献按方法家族分道）；**精读笔记范文** `chapters/part3-research/notes/hybridsn-roy2020-grsl-reading-notes.md`（九节模板 + 复现 checklist 现场 + 课程协议 B/C 复现数据回填，"⚠️ 待核对"标注示范）；复现 checklist 十项三问
- ✅ ch12 科研实验设计与可复现工程——正文 + **多种子消融研究完成**（SIR λ=0/0.1 + 去残差 × 3 种子 = 9 次训练，mean ± std 入板）：H1（SIR）未检测到收益（第 9 章单种子负结果稳定复现）；H2（残差）方向性成立（−0.45 OA，3/3 一致）且去残差组训练不稳定可见（一次 0.60 崩塌），量级受预算限制；新增 `scripts/aggregate_ch12_ablation.py` 与 `generate_ch12_figures.py`（误差棒柱状图 + 种子间波动曲线）；记分板新增协议 D′（mean±std）块
- ✅ ch13 论文写作与投稿——完整正文（13.1–13.6 共六节：IMRaD 审稿人视角、引言三段式、实验节五件套、LaTeX 纪律、投稿与 rebuttal）；**交付可编译 IEEEtran 骨架** `chapters/part3-research/latex/paper_skeleton.tex`（含课程真实数字的 booktabs 主表、消融表模板、投稿前 checklist）
- ✅ ch14 毕业实战 Capstone——完整正文（14.1–14.6 共六阶段：选题立项/基线复现/改进实现/实验消融/论文撰写/模拟投稿，六阶段流程图 + 三个衔接课程资产的示例选题 + 立项书模板含负结果退路）
- ✅ ch15 实战补遗——完整正文（15.1–15.5 共五节）、五张分析图（t-SNE、Grad-CAM、空间分桶、损失权衡、逐类召回）；`train_2d_cnn.py` 新增 `--loss ce/weighted/focal`（engine.fit 支持自定义 criterion）；SA/PU 跨数据集零改动跑通（99.77%/99.66%）；gokriznastic/HybridSN 仓库走读（与课程实现逐层 diff ✓）；SpectralFormer 精读笔记

**课程 15 章全部完成。**

## Part IV：小样本与开集分类（ch16–ch17，已完成）

- ✅ ch16 小样本高光谱图像分类——原型网络 Prototypical Networks（episodic training，N-way K-shot），K-shot 曲线 1/5/10-shot（85.12/92.29/92.33%——K=5 稳定点）
- ✅ ch17 开集高光谱图像分类——MSP 基线 + 距离阈值拒绝，AUROC 0.68 vs 0.96（嵌入距离远优于 softmax 置信度）；已知 12 类 / 未知 4 类（稀有类）
- 📁 `chapters/part4-few-shot-open-set/`

**课程 14 章全部完成。**

## 阶段 4：科研篇配套产出（已完成）

- 第 11 章：HybridSN 论文精读笔记示例 ✅
- 第 13 章：可编译的 IEEEtran 论文骨架 `.tex` ✅
- 第 12 章：多种子 benchmark 表格模板与汇总脚本 ✅

## 阶段 5：毕业实战 Capstone（手册就绪，执行属学习者行为）

- 第 14 章提供六阶段完整作战手册与三个示例选题
- 与学习者确认选题方向（如 HybridSN + 谱注意力、SSRN 的 λ 扫描与空间不相交对照、轻量化 3D-2D 结构）后按手册推进
- 最终产出一份完整的论文稿件

## 工程配套（持续）

- 固定随机种子、统一数据划分逻辑
- 统一模型保存路径与实验结果目录（`results/<model>/<dataset>/`）
- 用脚本替代 notebook 里的重复逻辑，逐步把传统机器学习训练也补成脚本入口
- 给 `results/` 设计统一输出规范和 benchmark 汇总格式
