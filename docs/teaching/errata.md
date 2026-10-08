# 录课纠错记录（保留旧实验，不混换证据）

## 已确认的问题

| 原材料 | 问题 | 本轮处理 / 验收状态 |
|---|---|---|
| ch02 | 常量预测多数类的 Kappa 写成0.47 | 正确为0；加入自动算例，待运行 |
| ch01/05/07 | band index、PCA分量与物理波长混用 | 不支持线性换算，录制卡删除未经波长元数据核实的物理解释 |
| ch05/06/07/08 | 普通Conv2d被说成只看单通道；感受野漏池化 | 录制卡明确跨输入通道聚合，池化一起递推 |
| ch06 | PCA-12“丢88%方差” | 维数比例不是解释方差；须实际读取 explained_variance_ratio_ |
| ch09/12 | 把课程空间方差惩罚归给SSRN原论文 | 原文未核验，不再声称论文SIR复现；历史消融保留为自定义扩展实验 |
| ch10 | 默认Transformer称Pre-LN；注意力峰映射红边 | 默认norm_first=False为Post-LN；注意力图不是已证实物理归因 |
| ch12 | 用组均值差/标准差判显著性与等价 | 只作描述统计；配对seed差值、样本单位和n=3限制必须明确 |
| notebook 15 | 通道均值冒充Grad-CAM，层索引与模型变量错位 | 新公共实现真正反传目标logit；旧输出在新执行成功前不覆盖 |
| notebook 16 | test参与训练监控；少量support被称总标签预算 | 新class-disjoint/same-class模式，独立sampler，平方距离，同encoder比较K |
| notebook 17 | shuffle后特征与标签错配；test校准阈值 | 新协议显式类划分、known validation分位数校准、错误图 |
| notebook补丁 | 再生成会丢map/重复插cell | 单一生成入口、稳定cell ID、成功后原子替换 |

## 状态说明

**新算法模块/运行器/讲课卡已写入；Bash 和 WebFetch 在本轮多次因服务不可用被拦截。尚无新测试通过或新训练成功的证据。** 不把现有输出当作修正后的结果，也不宣称所有课已可发布。

复验顺序：

```bash
python -m unittest discover -s tests -v
python scripts/prepare_teaching_artifacts.py --quick --output-dir results/teaching_smoke
python scripts/prepare_teaching_artifacts.py
python scripts/build_lecture_notebooks.py
python scripts/check_teaching_ready.py
```

之后逐张目检新图、在 notebook 子目录复跑、填写 recording-checklist。章节正文的定点纠错与来源核验需逐项检查，不能仅凭有一份讲课卡就把原错误算作已修复。
