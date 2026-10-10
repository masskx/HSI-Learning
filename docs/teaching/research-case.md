# 研究案例：加权CE改变了哪些错误？

## 问题与预登记

IP类别样本量相差大。固定中心划分、PCA12、9×9 patch、架构、seed和预算，问加权CE是否改变少数类recall，并付出怎样的OA代价。本次只改loss，模型按validation OA选择。

候选解释：频率权重改变了样本对梯度的贡献。热图本身不能证明机制。

## 证据链

1. 配置：classifier-ce / classifier-weighted的run_config。
2. 划分：比较train/val/test ID与预处理参数。
3. 指标：Notebook15重新推理test，报告OA、AA、support/recall。
4. 空间错误：仅test中心计错误；全图仍含训练中心。
5. 梯度图：同一输入更换目标类，不将插值精度当作解释分辨率。

## 学生证据表

|字段|CE|weighted CE|来源|
|---|---|---|---|
|seed、样本量|配置读取|配置读取|protocol|
|checkpoint epoch|cfg读取|cfg读取|run_config|
|OA、AA|实测填入|实测填入|loss-comparison|
|小类support/recall|实测填入|实测填入|逐类报告|
|失败像元ID|固定规则选择|同一位置|maps / CAM|

不预填“必须提高”的答案；不将历史3-seed与当前新预处理单seed混合求平均。

## 结果段落

观察：在[输入/协议/预算]下，OA为[实测]、AA为[实测]；相对baseline差值为[计算]个百分点。

解释假设：变化与类别权重改变优化目标相一致；验证需要记录梯度贡献或增加受控对照。

限制：同场景单seed演示，patch观测相关；不推断独立空间区域、其他数据集或方法上限。

## 多seed迁移

分别运行42/43/44，每个seed内baseline与variant共享split。保存每对差值，不把像元当独立重复。

```python
delta = variant_per_seed - baseline_per_seed
print(delta, delta.mean(), delta.std(ddof=1))
```

均值与std是描述统计；差值小于std、三次同号均不能直接得出显著性、等价或因果结论。无增益也保留配置和失败样本。
