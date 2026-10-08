# L10 · 一次实验怎样变成可信的研究证据？

目标：从“我加了模块”转到可检验假设、复现记录和受限结论。前置 L02 与任一模型课。材料 ch11–14、`docs/benchmark.md`、`run_config.json`、`history.json`，不现场启动实验矩阵。

## 节奏（25 分钟）

0–3：两组分数差0.5，问“是模块作用还是划分运气？”；3–7：问题—假设—控制变量；7–12：论文/仓库/课程三个版本对齐；12–17：种子配对差值与限制；17–21：把证据写成段落；21–25：半页立项练习。

## 六页幻灯片

1. 论文不是模块拼装清单，是可检验问题的证据链。
2. 假设、baseline、controlled change、metric、failure criterion。
3. 来源表：原文页/官方代码函数/教学差异；未取得证据就写待核验。
4. 三对种子结果先算 delta_i，再看分布；均值差<标准差不等于无效，也不是等效性检验。
5. 输入、预处理、patch、训练预算不同的模型表只能作教学对照，不是架构消融。
6. 方法→实验→限制→写作，用可复现文件组织证据。

## 现场练习

打开某次 run_config，找 seed、split、PCA fit 范围和 best epoch；与另一组逐项比。手算 delta=[-.1,-.2,-.1]：平均负不自动证明总体负效应，n=3 和测试样本相关性限制推断。

## 写作示范

可用：“Under the declared same-scene protocol, all three runs showed a lower mean OA for this setting; this small experiment does not establish a general disadvantage.”
不可用：“SIR was disproved”“our module significantly improves”而没有相应原方法核验、统计程序和对照。自定义空间方差损失只能按自定义扩展报告。

## 练习与答案

立项书四栏：研究问题、经典baseline、唯一变化、失败后能得出什么。答案不预定正收益；若效果相反，先审计实现/协议，不为了得到正结果遍历test。文献选择依据问题相关性而非“热门=适合”。

## 科研产出与 AI 边界

一页实验登记表、原文对照笔记、一段带限制的英文结果描述。AI 用来提出反例和检查遗漏，不替代阅读原文、不生成伪引用/指标。参考 Keshav 2007、Pineau et al. 2021；写作模板的编译状态单列，不再凭文件存在宣称可编译。
