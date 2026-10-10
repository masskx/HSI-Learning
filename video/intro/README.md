# 视频片头 / Video Intro（v2，配合 2026-10-10 课件改版）

片头的最后一帧就是课件封面：午夜蓝底、底部光谱条、短标题、“本课问题”、右侧 Notebook 图卡片。位置、字号与配图都来自课件构建（`scripts/build_course_slides.js` 写入 `slides/lesson-manifest.json` 的封面数据），所以从片头切到录屏的封面页几乎无缝。

## 成片（可直接导入剪辑软件）

`renders/` 下 13 个 MP4，均为 1920×1080 · 60fps · H.264 High · yuv420p · AAC 48kHz 立体声 · 7.0 秒，符合 B 站投稿推荐参数：

| 文件 | 用途 |
|---|---|
| `HSI-Learning-intro-1080p60.mp4` | 系列总片头（“高光谱图像分类 从零到一”，合集首集 / 课程介绍） |
| `HSI-Learning-intro-L00-1080p60.mp4` … `L10`，含 `L05A` | 每课片头，结尾停在该课课件封面 |

## 镜头设计（7 秒）

| 时间 | 画面 | 声音 |
|---|---|---|
| 0–1.45 s | 午夜蓝底上，200 条波段柱依次升起，同时一条光谱曲线从紫到红描出——这是 **Indian Pines 标注像元的真实平均光谱** | 五声音阶上行拨弦，从左声道移到右声道 |
| 1.35–2.15 s | 说明文字与 `HSI-LEARNING · 高光谱图像分类课程` 字标淡入 | — |
| 2.8–3.8 s | 整条光谱下沉、拉宽，落到画面底部，变成课件封面底部的光谱条；字标缩放到封面眉题位置 | 柔和气流声 |
| 3.45–4.9 s | 课号徽标弹出，短标题逐行遮罩上升，“本课问题”、副标题、先修依次出现，Notebook 图卡片从右侧滑入 | 温暖的和弦落定 |
| 4.9–7.0 s | 静止停留在封面版式 | 和弦自然衰减 |

**剪辑建议：** 录屏第一页就是课件封面时，片头末尾接 0.3 秒交叉溶解即可衔接。片头是按 LibreOffice 渲染的封面逐像素对过位置的（误差 ≤ 3 px）；PowerPoint 用微软雅黑渲染时行高可能有几像素差异。声音峰值 −6 dBFS，若另配 BGM，可把片头音轨降 6–10 dB 或静音。

## 重新生成

改了课程内容或课件后，先重建课件，再重建片头：

```bash
python scripts/build_course_slides.py         # 课件（同时更新 lesson-manifest.json 里的封面数据）
pip install -r requirements-video.txt
python -m playwright install chromium         # 只需一次
python scripts/build_video_intro.py           # 系列 + 全部 12 课
python scripts/build_video_intro.py --only series L03 --stills   # 只渲部分，并输出关键帧 PNG
python scripts/build_video_intro.py --size 2160                  # 4K
python scripts/build_video_intro.py --audio silent               # 静音轨（仍保留音轨，便于剪辑软件对齐）
```

需要 ffmpeg 在 PATH 上（或 `--ffmpeg 路径`）。每课文字读取 `slides/src/course_content.py`；封面短标题与配图卡片读取 `slides/lesson-manifest.json`；系列片头的文字在 `scripts/build_video_intro.py` 顶部 `SERIES` 中。

**预览/调整：** 直接用浏览器打开 `intro.html`（`?lesson=L03` 看某课，`&t=3.2` 冻结在某一时刻，点击重播）。所有动画都是时间 t 的纯函数，渲染逐帧确定、可复现。

**字体：** 与课件一致优先使用 Microsoft YaHei（微软雅黑）。仓库中的成片是在 Linux 上用 Noto Sans SC（思源黑体）渲染的，字形与雅黑略有差异；在装有雅黑的 Windows 上重跑上面的命令，即可得到与 PPT 封面字形一致的版本。没有雅黑的机器可以用 `--font-css` 传入任意 `@font-face` 样式表。

## 文件

- `intro.html` — 动画本体（设计令牌、时间轴）。底部光谱条直接引用 `slides/assets/brand/spectrum-bars.png`。
- `intro-data.js` — 由脚本生成：每课文字、封面数据、归一化平均光谱，勿手改。
- `heroes/` — 由脚本从课件构建复制的封面配图（每课一张）。
- `spectrum.json` — 平均光谱缓存（没有 scipy 或数据集时使用；课件封面光谱条也读它）。
- `renders/` — 渲染成片。
