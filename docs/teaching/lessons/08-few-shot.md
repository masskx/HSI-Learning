# L08 · 5-shot 的五个样本究竟是什么？

目标：能区分总标注预算、每任务支持集与未见类泛化。前置 L04/L05；notebook 16 `protocol`、`episode`、`prototype-math`、`evaluation`、`maps`；代码 `EpisodeSampler` / `prototype_logits`。

## 节奏（25 分钟）

0–3：“5-way 5-shot=25 个标签，那么 encoder 有没有用过其他标签？”；3–7：meta-train/val/test 类集合；7–12：二维原型手算；12–17：真实 episode 与 support/query 图；17–21：同 encoder 的 K 对照；21–25：map 的支持集边界与论文表述。

## 六页幻灯片

1. 少标注不是自动等于未见类别 few-shot；两种研究问题先分清。
2. 6/5/5 类别集合，meta-training 标签预算单独列；test support 是部署适应标签，不是训练期可见标签。
3. 支持集 {(0,0),(2,0)} 原型 (1,0)，另一类原型 (9,0)，查询 (4,0) 的平方距离 9/25。
4. 负平方距离就是 logits；原型为类均值，episode labels 重映射至 0…N−1。
5. K=1/5/10 用相同冻结 encoder、相同 eligible class pool、配对查询任务；不是三个不同模型随便比。
6. 全图可视化是额外展示，明确原型来自哪些支持样本；不能把整图 GT 当推理输入。

## 现场推导/代码

打印 sampled class IDs、support/query 原图坐标，执行无交集断言；N-way 不足必须报错。展示一个 episode 的图例与分布，再打印手算 logits 与 torch 输出。用独立 val sampler，说明为什么训练过程中看 test 会使实验失去盲测意义。

## 受控改动 / 答案

固定 query，只把每类 support 从1增到5，记录这一次任务精度，而不是预告必定提升。若均值估计更稳定通常有利，但相关样本、异常值、类内多模态都可能造成反例；方差缩小1/K的说法需要独立同分布条件。

## 科研产出

协议图、标注预算表、episode manifest、任务准确率分布。随机同场景 patch 仍可能共享上下文；类别不交不等于空间独立。参考 Snell et al. 2017、Vinyals et al. 2016。AI 若说“只用了25个标签”，追问预训练和 meta-training 标签是否计入。

## 演示安排

完整 episodic training 提前排队，现场做一个任务前向/损失和 map；无兼容 encoder 资产时中止 demo 并给生成命令，不用随机权重输出冒充已训练结果。
