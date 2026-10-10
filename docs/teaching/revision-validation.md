# 课堂修订与验收（2026-10-09）

本轮落实课程评审中的教学结构、计算错误、演示一致性、启动路径、实验解释和学生练习问题。当前课堂按[classroom-guide](classroom-guide.md)走；[课件列表](../../slides/README.md)、[学生练习](../../student_workbook/README.md)分别面向教师和学生。

## 已落实

- 12个单元、147页可编辑课件。新增L05A的真实单batch更新；卷积拆为手算、输入轴和逐层RF；HybridSN验证32通道和C×D的元素对应。
- Notebook02重建为train/val/test分离的SVM Pipeline，验证选C，预处理仅fit训练中心；修正类别ID与混淆矩阵名称顺序；另出投影用ID大字号图，并明确颜色是行归一化比例，不是原始数量。旧02在archive保留。
- 00的batch修改与断言共用变量；01补类内/类间光谱，默认关闭wx/OpenGL桌面窗口；旧01–09加入根目录查找。
- 15补加权损失手算与同输入两目标Grad-CAM；16补复制encoder上的meta-training episode更新；17接受率变化后重校准并重算指标/地图，不再要求新阈值等于旧值。
- 空间划分在边界两侧保留patch半径缓冲，明确缺席类、预算和分布变化；72/81只描述一对相邻窗口。没有测试支持的类标NA，序列化为null，AA只平均有支持的类。
- SVM与四个模型包均有配置、数据或资产哈希；数字由实际运行导出，不把旧实验成绩当当前结果。闭集共享中心不代表共享输入信息；few-shot与open-set使用独立任务设定。
- 6份学生填空Notebook配完整示例、迁移任务、交付要求和已有教师答案；每课加入预测练习与退出题。L10用真实加权损失实验串联问题、对照、证据和结果段落。
- ch05–09、ch12、ch15及benchmark清理输入瓶颈、patch归因、单通道卷积、标准差显著性等过度结论；空间方差惩罚明确为课程扩展，旧CLI名称保留兼容。

## 科研极简视觉修订

12份课件统一浅底、深蓝与大图，减少装饰和重复信息；原171页中的24页独立来源/AI提示内容移入讲者备注，当前147页。标题、图表、手算、代码、三线表、课堂预测与退出题分别采用对应版式。历史曲线从保存的20轮训练和4次验证日志重绘，不重训、不改变数字。算法与Notebook执行记录沿用上一轮，本轮重跑课件检查与268项教学就绪检查。

## 验证范围

|项目|结果与范围|
|---|---|
|算法与输出检查|20项unittest通过：类ID/NA、训练拟合范围、空间窗口不共享、阈值、episode、Grad-CAM等|
|课堂Notebook|00/01/02/10/11/12/15/16/17实际执行；9本均从notebooks目录复验；根目录执行也通过|
|历史Notebook03–09|仅路径初始化复验；未重新跑完历史长训练，不把此项称全本执行|
|演示资产|四个模型包重新训练、哈希校验与推理；raw/PCA12 SVM按验证规则重新拟合|
|课件|12份、147页；Artifact Tool最终文件通过包/布局/字体/导入检查；Artifact与桌面PowerPoint均导出全部页|
|人工视觉检查|全部147页以1600×900逐页目检，修正SVM代码折行和历史HybridSN曲线标签重叠；不等同于真实课堂试讲|

本机Python3.13、CPU；具体库版本在`artifacts/teaching/environment-versions.json`。已注册kernel **HSI-Learning (.venv)**。其他电脑需安装自己的环境并准备`results/`中的资产；权重不随Git分发，已执行Notebook可作离线备用。

本地详细报告：`results/classroom_verification/execution-report.json`、`results/teaching_readiness.json`、`slides/previews/validation.json`、`results/slide_revision_build/*validation*.json`。后者分别是执行、教学资产、PPT包和最终导出检查，不能相互替代。最终课件页图在`slides/previews/com-export/`。

## 使用中的限制

真实学生试讲、课堂时长和学习效果尚未实测。课程简化模型不标成原论文逐层复现；完整来源核对状态见[source-audit](source-audit.md)。当前闭集结果为同场景单seed，随机patch可能共享上下文；开集选模型与阈值校准共享known validation。下一步研究迁移需配对种子、空间覆盖与更多数据集证据，不能用这轮教学修订代替这些实验。
