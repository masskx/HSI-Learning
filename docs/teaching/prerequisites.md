# 前置能力自诊断（不以术语数量考核）

每题先预测结果再运行；答错后按补课入口修复，不需要先看完全部深度学习教材。

| 能力 | 快速题 | 通过标准 | 补课入口 |
|---|---|---|---|
| ndarray轴 | `(3,4,5)`切`[:,:,2]`是什么shape？ | 能回答(3,4)并解释轴 | L01 / A01 |
| 标签与特征 | CSV最后一列class应否进入SVM？ | 不进入，且保留空间位置另存 | L03 / A02 |
| 索引 | 原坐标与padded坐标差什么？ | 能手算中心与四角 | L04 / A03 |
| 概率/损失 | logits和softmax概率区别？ | CE输入logits，不重复softmax | L05 / PyTorch CE |
| 自动微分 | 为什么Grad-CAM不能no_grad？ | 要对目标logit求feature梯度 | L07 / A04 |
| train/eval | eval是否等于禁止梯度？ | 不是；BN/Dropout模式与autograd分别控制 | L05/L07 |
| 数据边界 | scaler可以fit test吗？ | 默认只fit训练；其它设定需声明 | L02 |
| episode | 5-shot是不是总训练标签预算？ | 不是，能列meta-training和support/query | L08 / A05 |
| 开集阈值 | test ROC最优点能当校准吗？ | 不能作无泄漏部署阈值 | L09 / A06 |
| 研究结论 | 三次均值高就显著吗？ | 不直接下显著性/因果结论 | L10 |

建议学习路径：前四题未通过先做A01–A03；自动微分/模式题未通过先学L05再进入解释性；episode与开集是进阶，不跳过协议。
