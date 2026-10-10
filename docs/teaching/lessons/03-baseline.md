# L03 · 上神经网络前，SVM 能告诉我们什么？

当前课堂执行入口：Notebook02：pipeline / kernel-scale / pca-confusion。以下历史入口只用于延伸阅读，逐课顺序见[classroom-guide](../classroom-guide.md)。
目标：跑通一个标注像元 baseline，理解 RBF 的尺度依赖。前置 L02；notebook 02，定位 `SVC` / `train_test_split`，ch03 作为阅读。

## 节奏（20 分钟）

0–2：“不用深度学习，能否先检验数据到底有没有判别信息？”；2–6：向量、标签与位置；6–10：RBF 距离核手算；10–14：小规模拟合/加载既有模型；14–17：看逐类报告和预测 map；17–20：验证集搜索、课后任务。

## 六页幻灯片

1. Baseline 的意义：不是必须打败的稻草人，而是数据与协议的检查器。
2. `(H,W,B)→(HW,B)`；GT 非零筛选，空间位置保留用于回贴。
3. `exp(-gamma * squared_distance)`；令 d²=100，gamma=.01 与 1 得到截然不同的相似度。
4. StandardScaler 只 fit train；Pipeline 让交叉验证内拟合正确，禁止先全数据标准化后 CV。
5. C 与 gamma 网格、验证集选择；最优在边界是继续检查的信号，不自动证明未收敛。
6. 精度、训练耗时、输入信息和结果限制一起记录。

## 代码对应与演示

`SVC(C=...,gamma=...)` 对应核；`class` 列必须从特征移除；整图预测只做显示，test mask 指标另算。展示 GT/预测/测试错误图，而不是把训练像元混进成绩。

## 受控改动与答案

把所有特征乘 10、固定 gamma，会使核指数的距离项扩大 100 倍。调整 gamma/100 可抵消这个尺度变化；用库验证而不是背“默认最好”。课后：标准化 Pipeline 与 raw DN 各用独立验证集选超参，不能拿默认值排名推断模型族强弱。

## 研究产出与边界

产出 baseline config、测试报告和误差图。参考 Melgani & Bruzzone 2004、sklearn SVC/Pipeline。结果可以很低：若没有协议，漂亮数字无价值。AI 建议的 gamma 网格必须结合输入尺度核算。完整搜索提前运行，现场保留一个小例和真实日志。
