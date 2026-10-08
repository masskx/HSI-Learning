# 课程质量加固与PPT交付报告

## 本轮已准备（尚未全部执行验收）

- 用户指定技能 `@user_3b34947d/ppt-generator-skill` 1.0.2 已安装并读取；选择其Python实现，实际复用 `create_content_slide`，不执行带虚构示例数字的main。
- 11份课件的内容源：`slides/src/course_content.py`；每份约13–14页，含理论/手算/关键代码/真实Notebook图/AI操作卡/限制/来源。
- 生成器与PPTX包检查、可选LibreOffice逐页渲染已写好；只在本机运行，未上传课程。
- 六个任务书与评分规范、独立解答、前置自诊断已写好。
- `docs/teaching/ai-assist.md` 将每课AI技巧与验证步骤对应，禁止编造实验/引用。
- 联网图源发现脚本已写好，只收集候选来源页，不把未授权图片自动嵌入。

## 核心漏洞清单（必须闭环，不能用警示语替代修复）

| 项 | 证据 | 状态 |
|---|---|---|
| 感受野/波长解释跨正文和小结不一致 | ch05小结23 vs 正文26；ch06仍漏Pool；ch07把PCA分量换nm | PPT内容源采用纠正值；历史章/图仍需全面收口 |
| 自定义方差惩罚冒归SSRN | ch09/12与benchmark残留原论文收益主张 | 录课主线不采用该归属，历史材料清理未全部完成 |
| 前沿精读机制未核实 | SpectralFormer原文/代码待核验 | 仅保留阅读任务卡；PPT不引用DSP假设 |
| 写作骨架不能宣称可编译 | todo宏冲突、缺bib/alg label | 待修复与TeX验收；PPT不宣称模板已通过编译 |
| 旧01–09启动路径与依赖 | 相对dataset路径、CSV前置、未执行cell | 旧本兼容复验待完成，不能从新四本通过外推 |
| 校准和选择共享val | teaching_runs中的已知val同时选模型和阈值 | PPT/讲稿披露，不声称独立校准保证 |
| 演示包在本地results | manifest静态状态与实际日志不完全同步 | 说明冷启动必须准备资产，禁止声称克隆即具备权重 |
| 作业缺评分交付 | 既有自测多为概念题 | 已补A01–A06任务/文件/断言/rubric/解答 |
| 配图来源 | 部分数据/图片的使用条件未统一 | 只使用当前课程生成图与自绘结构；外图候选需核实后加入 |

## 后续静态修复（尚待运行）

- ch05双语小结和答案同步RF26，删除“充分训练证明输入瓶颈”的强归因。
- ch06正文/小结/答案同步RF12；空间划分差异不直接称净泄漏贡献。
- ch07明确13个PCA分量覆盖不是130nm，核宽是结构选择而非已证实物理尺度匹配。
- ch10延伸阅读同步Post-LN，ch12主要判定段/答案撤回std阈值显著性及曲线骤降的因果解释。
- LaTeX骨架改为英文排版正文、唯一placeholder宏，补最小references.bib与编译README，删除悬空algorithm引用和未经统一协议的示例成绩表。尚未实际编译。
- dataset/README补键、形状、来源与许可边界，不假称镜像许可等于数据授权。
- 测试新增含池化RF的两个算例，尚待执行。

## PPT执行入口（2026-10-08 已运行；v2 重设计完成）

执行结果（日志 `results/slide_builds/20261008T054057253049Z/` 为v1；v2构建后 `slides/lesson-manifest.json` 为准）：

- **v1问题确认并修复**：用户反馈"不美观、公式和符号乱、左下角页码、AI味重"。经桌面PowerPoint COM渲染目检确认：技能骨架的卡片堆栈装饰、灰条纯文本伪公式（如 `R^(H×W×B)`）、页脚、单一模板版式为缺陷主因。
- **v2设计系统**（`slides/src/deck_design.py`）：弃用技能骨架（技能保留安装但不使用）；墨蓝+陶土橙编辑风、左缘光谱渐变识别条、每run设置latin+ea字体、去页脚、术语脊线替代卡片堆栈、代码深色面板、图片导读栏。
- **公式**：18处全部改matplotlib mathtext排版（LaTeX源保留在`course_content.py`的`math`字段），300dpi透明PNG嵌入， κ分式、‖·‖₂、Σ、min_k等均正确呈现。
- **渲染闭环**：`scripts/export_deck_previews.ps1` 用桌面PowerPoint COM导出全部144页PNG；按9种页型抽样目检通过。未做逐页人工确认，渲染不等于排版完美，仍有问题页可用同一流程复验。

新增 `scripts/preview_course_pdfs.py` 仍可用：在PowerPoint/WPS里另存每份PPT为同名PDF，放到 `slides/previews/pdf/` 后执行

```text
python scripts/preview_course_pdfs.py --pdf-dir slides/previews/pdf --only L01 L09
```

它会栅格化页面并核对页数/时间戳，供逐页目检；不会把自动检查冒称为视觉审查。

```text
python scripts/prepare_course_slides.py --install-deps --research
```

该命令仍引用v1流程，尚未更新到v2设计系统；直接使用 `scripts/build_course_slides.py` + `check_course_slides.py` + `export_deck_previews.ps1` 三步代替。联网配图仍未授权，正式课件只使用课程自产图与自绘图形。

## 后续顺序

完成包生成与两份试制目检 → 修布局 → 全套逐页检查 → 配图许可核验/可用图纳入 → 继续历史资料一致性修复和旧Notebook验收 → 更新课程就绪状态。用户试录与真实授课反馈不可由自动测试替代。

本轮未自动提交或推送，不删除历史results/未跟踪用户文件。
