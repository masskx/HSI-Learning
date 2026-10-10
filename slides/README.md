# HSI-Learning课堂课件（2026-10-09）

12份可编辑PPTX，147页。课堂路线、Notebook与cell定位见[使用指南](../docs/teaching/classroom-guide.md)，检查范围见[验收记录](../docs/teaching/revision-validation.md)。历史设计与数字不作为当前结果。

## 打开课件

|课|文件|重点|
|---|---|---|
|L00|[环境](decks/L00-environment.pptx)|kernel、数据、前向|
|L01|[数据](decks/L01-data.pptx)|光谱变化与GT|
|L02|[协议](decks/L02-protocol.pptx)|指标与信息流|
|L03|[SVM](decks/L03-svm.pptx)|训练拟合、验证选参|
|L04|[Patch](decks/L04-patch.pptx)|坐标、padding、空间缓冲|
|L05A|[参数更新](decks/L05A-training.pptx)|logits、loss、梯度|
|L05|[卷积](decks/L05-cnn.pptx)|乘加、轴、参数、RF|
|L06|[HybridSN](decks/L06-hybridsn.pptx)|逐层shape与元素对应|
|L07|[错误分析](decks/L07-errors.pptx)|损失代价、真实Grad-CAM|
|L08|[Few-shot](decks/L08-fewshot.pptx)|episode更新、标签预算|
|L09|[开集](decks/L09-openset.pptx)|校准、误拒与漏检|
|L10|[研究案例](decks/L10-research.pptx)|证据表与结果段落|

## 重建

内容：`src/course_content.py`与`src/classroom_content.py`；作者工具：`scripts/build_course_slides.mjs`调用`@oai/artifact-tool`。`src/deck_design.py`仅保留为历史实现。采用科研极简风格：浅底、深蓝、大图、克制排版，微软雅黑/Consolas。流程与三线表使用可编辑文本和线条；公式沿用已检查的参考PNG。来源、AI辅助提示与Notebook定位保留在讲者备注，重复的独立页面合并入备注。历史HybridSN曲线由保存日志重新绘图，保留数值与历史协议。

```powershell
.\.venv\Scripts\python.exe scripts/build_course_slides.py
.\.venv\Scripts\python.exe scripts/check_course_slides.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/export_deck_previews.ps1
```

构建需要Codex附带的Artifact Tool运行库；其他安装可用`HSI_RUNTIME_ROOT`指向dependencies目录。普通授课只需PowerPoint和项目Python。Notebook图按cell ID与SHA256溯源；参考公式保存在版本控制的`assets/reference-formulas/`，不依赖本机旧课件备份。课前先准备资产并执行Notebook。

构建分别验证包、布局、字体与导入；实际渲染和人工目检另记，真实课堂学习效果需试讲反馈。
