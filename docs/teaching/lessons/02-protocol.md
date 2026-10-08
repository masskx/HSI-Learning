# L02 · 99% 准确率，为什么可能没有意义？

目标：能说清数据划分、指标分母与结论适用域。前置 L01。定位 ch02 / notebook 02 的 `train_test_split`、`classification_report`；用 `tests/test_teaching.py::test_kappa_majority` 核对算例。

## 20 分钟安排

0–2：只展示一个高 OA，问“在哪些像元上算的？”；2–6：train/val/test 角色，训练调参不能碰 test；6–11：手算偏斜三类；11–15：分层划分与 GT mask 代码；15–18：两个相邻 patch 的重叠；18–20：写一句有协议的实验结果。

## 六页幻灯片 / 现场推导

1. 高分前先问四件事：分母、划分、预处理、选择模型方式。
2. A/B/C 数量 900/90/10，全预测 A：混淆矩阵只有第一列非零。
3. OA=.9；AA=(1+0+0)/3；Kappa 中 p_e=.9×1=.9，故 Kappa=0。现场 sklearn 断言，纠正“0.47”旧例。
4. 两步切分：val_ratio_within_remainder=.1/.9，不是 .1；明确 rounding。
5. 标注中心不重合，不代表 patch 内容不重合；随机像元实验与空间外推回答不同问题。
6. 指标陈述模板：数据、划分、预处理、seed、checkpoint 选择、指标。

## 改动实验

把错误集中到一个小类，比较 OA 和 AA；固定预测但改变是否 mask 背景，展示分母改变。不要现场调 test 分数寻找“更好结果”。

## 练习 / 答案

“验证准确率最好时保存模型”和“测试准确率最好时保存模型”差在哪？后者把 test 变成模型选择数据。若某类 test support=0，recall 未定义；显式报告缺类与固定 labels 策略，不能悄悄删除后提高 AA。

## 科研产出

协议检查表与一段结果描述。来源：sklearn metrics、Cohen 1960；复用 `split_ground_truth`。图选择混淆矩阵、类别计数和 train/val/test mask，不能只给 OA 大数字。

## 证据与 AI 核验

单次高分不证明泛化。让 AI 解释 Kappa 后用手算和库函数双重核验；“差异大于标准差”等口号不是显著性检验。备用为固定 toy 数组，几秒可运行，不需要网络训练。
