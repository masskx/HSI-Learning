# 视频片头 / Video Intro

与课件同一套设计（`slides/src/deck_design.py`）：纸面白底、墨蓝正文、陶土橙单一强调色、左缘高光谱渐变细条。

## 成片（可直接导入剪辑软件）

`renders/` 下 12 个 MP4，均为 1920×1080 · 60fps · H.264 High · yuv420p · AAC 48kHz 立体声 · 7.0 秒，符合 B 站投稿推荐参数：

| 文件 | 用途 |
|---|---|
| `HSI-Learning-intro-1080p60.mp4` | 系列总片头（合集首集 / 课程介绍视频） |
| `HSI-Learning-intro-L00-1080p60.mp4` … `L10` | 每课片头，结尾停在该课封面版式 |

## 镜头设计（7 秒）

| 时间 | 画面 | 声音 |
|---|---|---|
| 0.25–1.45 s | 200 条波段柱依次升起，同时一条光谱曲线从紫到红描出——这是 **Indian Pines 标注像元的真实平均光谱**，不是装饰曲线 | 五声音阶上行拨弦，从左声道移到右声道 |
| 1.35–2.15 s | 曲线下方说明文字与 `HSI-LEARNING · 高光谱图像分类课程` 字标淡入 | — |
| 2.85–3.99 s | 200 条波段像一条丝带依次飞向左缘，拼成课件里的光谱渐变细条；字标同时缩放到封面眉题位置 | 柔和气流声 |
| 3.55–5.05 s | 课号、标题（遮罩上升）、副标题、细线、本课交付、先修依次出现 | 温暖的和弦落定 |
| 5.05–7.0 s | 静止停留在封面版式 | 和弦自然衰减 |

**剪辑建议：** 片头结尾按 `deck_design.cover()` 的坐标与字号换算（1 英寸 = 144 px，1 pt = 2 px），与课件封面布局相同；录屏第一页就是封面时，用 0.3 秒交叉溶解衔接最稳妥（尚未与 PowerPoint 渲染做逐像素对比，行高可能有几像素差异）。声音峰值 −6 dBFS，若另配 BGM，可把片头音轨降 6–10 dB 或静音。

## 重新生成

```bash
pip install -r requirements-video.txt
python -m playwright install chromium        # 只需一次
python scripts/build_video_intro.py          # 系列 + L00–L10，全部重渲染
python scripts/build_video_intro.py --only series L03 --stills   # 只渲部分，并输出关键帧 PNG
python scripts/build_video_intro.py --size 2160                  # 4K
python scripts/build_video_intro.py --audio silent               # 静音轨（仍保留音轨，便于剪辑软件对齐）
```

需要 ffmpeg 在 PATH 上（或 `--ffmpeg 路径`）。课程标题、副标题、交付物、先修**均读取 `slides/src/course_content.py`**：改课件内容后重跑脚本，片头会同步更新，不需要手改片头。系列片头的文字在脚本顶部 `SERIES` 中。

**预览/调整：** 直接用浏览器打开 `intro.html`（`?lesson=L03` 看某课，`&t=3.2` 冻结在某一时刻，点击重播）。所有动画都是时间 t 的纯函数，渲染逐帧确定、可复现。

**字体：** 与课件一致优先使用 Microsoft YaHei（微软雅黑）。仓库中的成片是在 Linux 上用 Noto Sans SC（思源黑体）渲染的，字形与雅黑略有差异；在装有雅黑的 Windows 上重跑上面的命令，即可得到与 PPT 封面字形完全一致的版本。没有雅黑的机器可以用 `--font-css` 传入任意 `@font-face` 样式表。

## 文件

- `intro.html` — 动画本体（设计令牌、时间轴）。
- `intro-data.js` — 由脚本生成：每课文字 + 归一化平均光谱，勿手改。
- `spectrum.json` — 平均光谱缓存（没有 scipy 或数据集时使用）。
- `renders/` — 渲染成片。
