# L05A：模型怎样完成一次参数更新？

入口：Notebook10；先修L04的patch与标签。目标是追踪输入、target、logits、loss、梯度和参数更新。

45分钟：5分钟回顾类别ID；10分钟手算logits=[2,1]的CE；10分钟运行batch/forward-loss/one-update；12分钟漏掉zero_grad的复制模型实验；5分钟比较train/eval与no_grad；3分钟退出题。

关键检查：target为0…15，背景不进入监督；CE直接接收logits。原演示权重未改动。eval仍能求导，Grad-CAM需要梯度。

退出题：一次batch loss下降能否证明泛化改善？eval是否等于no_grad？

作业A03先做完整示例，再填空和迁移。真实checkpoint按validation选，test仅在冻结后报告。
