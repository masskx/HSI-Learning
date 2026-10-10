# 从零打开课程（CPU 教学入口）

建议 Python 3.12；环境兼容性以实际验收报告为准，不宣称所有系统已通过。讲解中文，图中文字英文。数据目录含 IP/SA/PU，首批演示只使用 IP。

最新12单元课堂路线见[classroom-guide](teaching/classroom-guide.md)，练习见[student_workbook](../student_workbook/README.md)。本机已验证版本见`artifacts/teaching/environment-versions.json`。

## 一键验收入口

本轮新增串行入口，默认只跑测试，不自动安装依赖、提交或推送：

```bash
python scripts/prepare_recording.py --stage test
# 测试成功后执行全链路（可能需要较长时间）：
python scripts/prepare_recording.py --stage all
```

`all` 依次执行测试、quick冒烟、正式模型与SVM资产、数字导出、两组Notebook执行、子目录启动复验与验收。每个阶段日志保存在 `results/recording_checks/<UTC时间>/`；失败停止，报告中未执行阶段不会伪装为完成。自动检查之后仍需人工目检和两节试录。

## 1. 环境与内核

在项目根目录：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install nbformat nbclient ipykernel
python -m ipykernel install --user --name hsi-learning --display-name "HSI-Learning (.venv)"
python -m jupyter lab
```

PyTorch 应按自己的 CPU/CUDA 环境选择官方安装方式，以上不替代 GPU 安装文档。打开 Notebook 后确认内核并运行 `import sys; print(sys.executable)`。不能只看终端激活标记。

## 2. 第一次成功

打开 [notebook 00](../notebooks/00_environment_check.ipynb)：版本、项目根、`.mat` 键、cube/GT 形状、一次模型前向。初次运行不做长训练。

Notebook 会向父目录查找 `src/hsi_learning` 和 `dataset`。如果不在仓库目录树下，请先切换到仓库；不要硬编码个人绝对路径。根目录和 `notebooks/` 启动都应通过验收。

## 3. demo 与 train

修正版 notebook 15/16/17 的默认 demo 读取 `results/teaching_ready/` 的可信包，训练流程不会偷偷触发。首次准备：

```bash
python scripts/prepare_teaching_artifacts.py
python -m unittest discover -s tests -v
python scripts/check_teaching_ready.py
```

准备命令会串行训练最小参考模型、保留日志、保存 hash/预处理/类别划分/阈值；这是**录制前**操作，不是视频现场操作。`--quick --output-dir results/teaching_smoke` 仅用于冒烟，不可当参考实验成绩。

已有 `results/` 中的历史指标保留；与修正后的运行不同并不意味着退步。数据设定、预处理和验证程序改变后本就不应横比。

## 4. 常见故障

- 找不到包：核对 `sys.executable`，使用同一 Python 的 `-m pip`。
- 找不到 `.mat`：检查 ROOT 与键名，不立刻下载另一个同名文件覆盖。
- 找不到演示包：运行准备命令；不能把随机初始化模型的 map 叫训练结果。
- hash 不一致：更换了权重/预处理/数据，重新准备到新目录，别手改 hash 掩盖差异。
- 字体缺字：图标签保持英文，教学说明仍可中文；不安装来历不明字体。
- 执行失败：保留报错/日志，旧 notebook 输出不覆盖；不要 `allow_errors=True` 后声称完成。

## 5. 课堂与研究材料

[视频导航](teaching/course-map.md) · [录制清单](teaching/recording-checklist.md) · [来源审计](teaching/source-audit.md) · [纠错记录](teaching/errata.md)。课程的完成度分别记录正文核验、代码验证、输出验收、试录通过。
