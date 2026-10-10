# 学生练习：完整示例、填空、迁移

先选择项目kernel。学生版故意保留None与断言，不声称已完成答案。每题先预测再运行；卡住时教师根据错误回到最小手算例。

|任务|填空Notebook|完整示例定位|迁移重点|
|---|---|---|---|
|A01|[A01_practice](A01_practice.ipynb)|[01_data_reading_and_visualization.ipynb](../notebooks/01_data_reading_and_visualization.ipynb)：spectral-comparison|将toy变成4×6×7；再在IP选两类各3像元画曲线。不提供波长表时只标波段索引。|
|A02|[A02_practice](A02_practice.ipynb)|[02_svm_baseline.ipynb](../notebooks/02_svm_baseline.ipynb)：metrics-toy / pipeline|用Notebook02的空间缓冲划分重新训练SVM。先报告缺席类与预算，解释为什么分数差不是纯泄漏效应。|
|A03|[A03_practice](A03_practice.ipynb)|[11_patch_hybridsn_classroom.ipynb](../notebooks/11_patch_hybridsn_classroom.ipynb)：hybridsn-shapes / reshape-check；另看Notebook10|patch25改21，重新计算每层shape与FC参数；在复制模型上展示漏掉zero_grad的两次backward。|
|A04|[A04_practice](A04_practice.ipynb)|[15_imbalance_analysis_teaching.ipynb](../notebooks/15_imbalance_analysis_teaching.ipynb)：weighted-loss-toy / target-comparison|选一个不同的失败patch，固定输入换两个CAM目标；保留全零响应，写观察、候选解释、限制各一句。|
|A05|[A05_practice](A05_practice.ipynb)|[16_protonet_teaching.ipynb](../notebooks/16_protonet_teaching.ipynb)：prototype-math / episode-update|固定encoder、类集合和query，按嵌套support比较K=1/5/10；报告每episode差值和总监督预算。|
|A06|[A06_practice](A06_practice.ipynb)|[17_openset_teaching.ipynb](../notebooks/17_openset_teaching.ipynb)：calibration / threshold-exercise|接受率.95改.90，通过threshold-exercise重算tau、指标和地图；AUROC为何不变？写200字结果与限制，不引入未执行数字。|

[详细交付要求与评分](../docs/teaching/assignments/README.md)；先提交再查看[教师参考答案](../docs/teaching/assignments/solutions.md)。教师检查协议、实现、解释和复现，不按accuracy排名。每题把观察与候选机制分开，记录NA类别与失败样本。
