# SpectralFormer 阅读任务卡（来源待核验，不作为定稿精读范文）

旧版笔记把“四个分组token”“DSP掩码预训练”等未核实内容当作方法事实，还推导了其收益；这些说法现已撤回。没有原文证据时，不靠加一个“待核对”脚注保留正文里的确定结论。

## 资料入口

- 题名：SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers。
- [作者仓库入口](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer)
- 作者名单、DOI、版本和原文表格需从正式来源核对；本轮远端工具不可用，未声称已下载或重读原文。

## 九节笔记的实际任务

1. 元信息：从正式文献导出，不猜DOI。
2. 一句话总结：以原文摘要/方法为据，不从模型名字猜“预训练”。
3. 问题与缺口：定位原文如何讨论光谱局部性和跨层信息。
4. 方法：按源码打印token数，确认group-wise spectral embedding是否改变序列长度，区分分组宽度与组数；核对跨层融合机制。
5. 协议：逐项记录数据、标注预算、训练/验证/测试、预处理、超参。
6. 结果：每个数绑定原表和协议，不能把本课程纯谱Transformer成绩当SpectralFormer。
7. 复现：选一个入口跑最小forward，保存shape与源码版本。
8. 限制：只批评已核实的实现/证据，禁止对猜测出的DSP机制做消融结论。
9. 联系：提出可检验的问题；例如局部光谱信息如何进入token，但必须完成前八项再谈改进。

## 课堂使用边界

本材料现在用于示范“如何撤回不可靠笔记并重新核验”，不用于讲授SpectralFormer原理。已经核实的经典Transformer与ProtoNet可按其他课正常准备；该专题不阻塞第一季主线录制。
