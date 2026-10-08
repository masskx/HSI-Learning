# IEEEtran 教学骨架

状态：已修复明显源码问题，**尚未实测编译**。不要将“静态修复”表述为“编译通过”。

## 使用

在本目录执行（安装可信TeX发行版后）：

```text
pdflatex -interaction=nonstopmode -halt-on-error paper_skeleton.tex
bibtex paper_skeleton
pdflatex -interaction=nonstopmode -halt-on-error paper_skeleton.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_skeleton.tex
```

或者把 `.tex` 与 `references.bib` **一起**上传到你自己使用的编辑环境；本仓库工具不会自动上传论文。

正文为英文，中文只在 `%` 注释内，因此不要求模板自带中文排版包。全文使用唯一的 `\placeholder` 宏，不与todonotes冲突；不引用不存在的算法label或图片。

表格为占位结构，不填课程历史排行榜以免被误当作公平主实验。使用者须填入自己核验的同协议结果。BibTeX仅提供最小经典引用，投稿前核对正式元数据；不填写猜测的DOI。

## 验收

- 编译退出码为0；无Undefined control sequence、undefined references/citations。
- Method的符号与代码对应；贡献逐项有实验验证。
- 结果表所有 `--` 与正文 `TODO` 均清除。
- 加粗最优来自真实数据，不按叙事选择；单次分数不自动证明显著性。
- 图/数据/代码来源与许可、预处理fit范围、split与选择规则写清楚。
