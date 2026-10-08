# 经典方法来源与教学边界

本轮远端取文献的工具被服务错误阻止；以下链接是**核验入口**，不冒称本轮已访问成功。代码已直接检查，但原论文的页/表号未核实不填写。正式录制论文贡献部分前须对原文完成核对。

| 方法 | 原始来源入口 | 本课实际实现 | 状态与禁止的表述 |
|---|---|---|---|
| ProtoNet | [Snell et al., 2017](https://arxiv.org/abs/1703.05175) | 类均值 + 负平方欧氏距离；类别不相交模式；独立 train/val/test sampler | 算法待本轮自动测试；不能称“总共只用25个标签” |
| MSP | [Hendrycks & Gimpel, 2017](https://arxiv.org/abs/1610.02136) | 1−max softmax，已知验证集校准 | 不是普适弱检测器；性能依赖模型/未知类选择 |
| OpenMax | [Bendale & Boult, 2016](https://arxiv.org/abs/1511.06233) | 只做原理教学，未实现 EVT 拟合 | 正确分类训练样本激活到类均值的距离尾部，不是拟合误分类样本 |
| Grad-CAM | [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391) | 目标logit梯度空间平均→通道加权→ReLU→插值 | 旧通道均值图不属于 Grad-CAM；新模块待执行验证 |
| HybridSN | [作者仓库](https://github.com/gokriznastic/HybridSN) | 本地 HybridSN 无 padding 三层3D后一层2D | 原文/作者代码/课程的PCA与预算分别说明，不能声称同协议复现数字 |
| SSRN | [论文入口](https://doi.org/10.1109/TGRS.2017.2755542) | notebook 为教学版分轴残差；额外空间方差惩罚属于课程扩展 | **撤回“SIR是原论文贡献”的未经核实归属**；不能用自定义损失结果评价原论文有效性 |
| SpectralFormer | [作者仓库](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer) | 当前纯谱Transformer是教学基线，不是SpectralFormer | 分组嵌入/跨层融合须原文核对；**撤回“4个token、DSP掩码预训练”未经核实描述** |
| Conv/Transformer API | [PyTorch](https://docs.pytorch.org/docs/stable/nn.html) | 本地代码普通Conv2d跨全输入通道；Transformer未设置norm_first=True | 概念核对；默认Post-LN，不按“现代实现”猜架构 |

## 资料核验记录模板

```text
方法：
论文版本/日期/DOI：
作者仓库 commit：
页/节/公式或源码函数：
原文主张：
本课实际做法：
差异与不能推断的结论：
验证命令与结果文件：
状态：待核验 / 已核验 / 阻塞
```

不得让模型补全缺失 DOI、作者或实验协议后就当来源。允许简化，但标题与图注标“教学实现”，保留失败和负结果，不为视频叙事删掉不支持假设的证据。
