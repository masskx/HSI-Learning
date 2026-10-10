# HSI-Learning课堂课件（2026-10-10 视觉改版）

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

## 视觉设计（v3）

- **三明治结构**：午夜蓝封面与“结论要有边界”收尾页，中间为白底内容页。
- **唯一标识**：光谱色编号（01/02/03 按紫→蓝→青→绿→橙→红取色）；封面与收尾页底部是 Indian Pines 标注像元的**真实平均光谱**（`assets/brand/spectrum-bars.png`，由 `scripts/make_slide_brand_assets.py` 生成），与视频片头同一图形。
- **封面**：短标题 + “本课问题” + 副标题 + 先修，右侧白卡片展示本课第一张 Notebook 图（按图片比例自适应）。
- **按图片比例排版**：近方图左图右读图要点；宽图（宽高比 ≥ 1.9）通栏，要点在下方三栏。
- **其余页型**：交付物卡片 + 课堂入口；要点 + 关键词链；公式卡片（参考 PNG）+ 横向流程；纵向步骤流程；原生可编辑表格（深色表头、斑马行）；深色代码面板；预测卡 + 深色退出题卡。
- **字体**：主题与每个文本 run 的 latin/ea 都设为微软雅黑，代码为 Consolas（中文注释回落到微软雅黑）。
- 页数（147页）、页序与讲者备注内容沿用上一版（封面图改为优先选比例适合封面的本课 Notebook 图，备注中的“封面图源”随之更新）。

## 重建

内容：`src/course_content.py` 与 `src/classroom_content.py`（经 `scripts/export_slide_content.py` 导出）。版式：`scripts/build_course_slides.js`（pptxgenjs，只需 Node.js）。`build_course_slides.mjs`（上一版 Artifact Tool 设计）可用 `--engine artifact-tool` 继续构建；`src/deck_design.py` 仅保留为历史实现。公式沿用已检查的参考PNG；历史HybridSN曲线由保存日志重新绘图，保留数值与历史协议。

```powershell
cd slides; npm install; cd ..                                  # 首次
.\.venv\Scripts\python.exe scripts/build_course_slides.py     # 全部；--only L01 L05A 只建部分
.\.venv\Scripts\python.exe scripts/check_course_slides.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/export_deck_previews.ps1
```

Notebook图按cell ID与SHA256溯源；参考公式保存在版本控制的`assets/reference-formulas/`。课前先准备资产并执行Notebook。

本次改版已用 LibreOffice 渲染全部 147 页逐页目检（无溢出/重叠），尚未在桌面 PowerPoint 中目检；实际课堂效果需试讲反馈。
