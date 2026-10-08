# HSI-Learning 可编辑课件

## 当前状态（2026-10-08）

**11份可编辑PPTX已生成并通过包完整性检查；144页已全部经桌面 PowerPoint COM 渲染导出，对9种页型逐类目检通过。** 未做每一页的人工逐页确认；发现问题页可参照下方流程重建。

v1 版本使用 `@user_3b34947d/ppt-generator-skill` 的页面骨架，经 PowerPoint 渲染目检后确认排版缺陷明显（右侧装饰性卡片堆栈、底部灰条伪公式 `R^(H×W×B)`、页脚页码、模板味重），**该骨架已整体弃用**。技能仍保留安装，但当前生成链路不依赖它。

## 设计系统（v2）

- 版式引擎：`src/deck_design.py`（python-pptx 全掌控）；内容源：`src/course_content.py`。
- 视觉主题：纸面白底 + 墨蓝正文 + 陶土橙单一强调色；每页左缘一条**高光谱光谱渐变细条**作为课程识别元素。
- 字体：中文/正文 Microsoft YaHei，**每个 run 同时设置 latin 与 ea typeface**（修复此前中英文回退到主题默认字体的混乱）；代码 Consolas 深色面板。
- **公式**：18 处全部改为 matplotlib mathtext（Computer Modern）排版，LaTeX 源保留在 `course_content.py` 的 `math` 字段，渲染为 300dpi 透明 PNG 嵌入（`assets/formulas/`，按内容哈希缓存）。不再使用 `R^(...)`、`‖x‖²` 这类纯文本伪公式。
- 版式：无页脚、无页码；右侧装饰性"卡片+箭头"堆栈改为**术语脊线**（细线+圆点）；要点页左侧文本垂直居中；图片页大图+右侧导读栏+notebook/cell 溯源；代码页深色面板+打开路径+cell 定位。
- **去模板话术**（2026-10-08 第二轮自查）：删除全部 11 份课件里重复的口号文案（"理论—解释机制/实践—运行与检查/研究—证据与边界"、"课堂节奏"、"现场只做"、"把答案变成待验证假设"、"带走方法也带走边界"、"用一个公式、一个函数、一个证据回答"等），封面/目标页改为每课不同的交付物与 Notebook 入口信息，代码页右栏按课程类型（核心四本/其余 Notebook/写作课）区分，练习页右侧改为可书写的预测框，"数据边界"守则移入讲者备注。标题一律使用中性事实性短语（本课交付/代码实操/课堂练习/AI 辅助/小结与边界）。
- **内容充实**（2026-10-08 第三轮，针对"内容不实用"）：L01–L06 每课嵌入 2–3 张**当前仓库真实执行的 notebook 输出图**（共 8 处新引用：真彩色合成、六波段切片、协议B train划分、SVM 原始/PCA 两次实测混淆矩阵、GT 叠加 patch 中心图、预测回贴 mask、2D CNN 训练曲线与三联预测图、HybridSN 带 val 下滑的训练曲线），全部经 notebook+cell 溯源记录在 `assets/sources.json`；11 份课件现共有 16 张真实课程图。旧 notebook 无 ID 的 cell 按 `cellN` 位置引用（记录于 sources.json）。
- **干货重写**（2026-10-08 第四轮，最终定稿标准：每页内容必须是本项目总结出的实证知识）：全部理论页改写为"要点（实测数字）+ 可运行命令 + 真实输出"三件套，新增 22 个 run 条（REPL 式：命令 + 实际输出，全部来自本仓库实测/实算/已执行 notebook 输出）。载入课件的项目独有干货包括：IP 实测统计（10249/10776 标注、Oats 20 vs Soybean-mint 2455）、四套协议表（A 80/20｜B 30/10/60 seed 63466｜C 10/10/80 seed 42｜D 每类 20/10/70 seed 1334）、gamma='scale' 实算 2e-9 与教科书网格塌缩到 24% 的完整因果、patch 相邻窗口共享 72/81 与 60–89% 重叠率、感受野 23→26 与 Kappa 0.47→0 纠错、SVM 0.85 / HybridSN OA 0.9909 (epoch_14 ckpt) / 2D CNN OA 0.7056-AA 0.4957-Oats 0.0000 / 加权 CE 0.9506→0.9440 与 AA 0.8029→0.9345 / 开集 τ=0.2456 与 5.5572、AUROC 0.849 vs 0.851 撤回"距离远胜"。教科书式泛泛陈述不再进入理论页。
- 9 种页型：封面 / 本课交付 / 理论（要点+命令+输出 / 公式）/ 图片导读 / 代码实操 / 课堂练习 / AI辅助 / 小结与边界 / 来源。

## 构建、检查、渲染预览（只在本机）

```powershell
# 1. 生成全部（或 --only L01 试点）
.\.venv\Scripts\python.exe scripts\build_course_slides.py
# 2. 包完整性 / 页数 / 备注 / 越界检查
.\.venv\Scripts\python.exe scripts\check_course_slides.py
# 3. 用桌面 PowerPoint 渲染导出每页 PNG（Windows，需已装 PowerPoint；不做网络上传）
powershell -File scripts\export_deck_previews.ps1            # 全部
powershell -File scripts\export_deck_previews.ps1 -Only L01  # 单份
```

导出目录 `previews/com-export/<deck>-<时间戳>/NN.png`（1600×900）。修改版式后必须重导出并目检受影响页型；PowerPoint 渲染失败（如字体缺失、XML 损坏）会在此步暴露。

## 11份课件

L00 环境、L01 数据、L02 实验协议、L03 SVM、L04 Patch、L05 卷积、L06 HybridSN、L07 不均衡与解释、L08 Few-shot、L09 Open-set、L10 科研写作。

每份约13–14页，包含理论/手算/公式、Notebook cell入口、受控练习、AI提示词及验证、限制与来源页，与 [讲课卡](../docs/teaching/course-map.md) 对齐。标题/流程/文本/代码为可编辑对象，公式为 mathtext 渲染图（LaTeX 源可改后重建），Notebook 结果图为有源 PNG。

## 图源规则

- 课程结果图从**当前Notebook的稳定cell ID**提取，保存PNG哈希、notebook名和使用页号（`assets/sources.json`）。
- 结构图为 PPT 原生图形（术语脊线/步骤条/公式面板），自绘教学示意，不照搬论文版式。
- `python scripts/research_slide_images.py` 联网读取官方资料页，输出 `assets/research/candidates.json`；它只发现候选，不自动授权或嵌入。当前**外部配图仍未授权使用**，正式课件只用课程自产图与自绘图形。

## 可再生文件

- `src/course_content.py`：人工编写的课程内容源（含公式 LaTeX 源），单一事实来源。
- `src/deck_design.py`：版式设计系统（颜色/字体/页型/公式渲染）。
- `decks/*.pptx`：执行构建后生成。
- `lesson-manifest.json`：每课页数、Notebook/cell 映射、设计系统信息与哈希、生成/渲染状态。
- `assets/formulas/`：mathtext 公式渲染缓存（键含字号/DPI版本，可整目录删除重建）。
- `previews/validation.json`：包检查结果；`previews/com-export/`：PowerPoint 渲染导出。

修改核心算法或Notebook输出后须重建受影响课件；修改 `math` 字段后删除对应公式缓存或直接重建（哈希键会变化）。不要上传未经授权的数据或未公开稿件到外部PPT服务。
