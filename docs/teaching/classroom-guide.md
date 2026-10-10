# 课堂使用指南（2026-10-09）

面向有Python基础、初学高光谱分类的学生。阅读编号ch01–17、历史Notebook编号和视频编号是三套索引；课堂按下表走。

## 上课前

使用同一个Python，确认kernel的`sys.executable`。本机可用`.venv\Scripts\python.exe`；其他电脑按getting-started安装，不复制个人绝对路径。

本机已注册 **HSI-Learning (.venv)**，打开Notebook时选择此kernel。其他电脑运行`python -m ipykernel install --user --name hsi-learning --display-name "HSI-Learning"`注册自己的环境。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/prepare_teaching_artifacts.py
.\.venv\Scripts\python.exe scripts/prepare_classroom.py
.\.venv\Scripts\python.exe scripts/export_classroom_facts.py
.\.venv\Scripts\python.exe scripts/build_lecture_notebooks.py
.\.venv\Scripts\python.exe scripts/build_classroom_notebooks.py
.\.venv\Scripts\python.exe scripts/check_teaching_ready.py
```

训练在课前完成。四个模型包和SVM保存在`results/`，不会随git clone自动出现。`--quick`是冒烟预算，不作为正式成绩。准备失败时保留日志。

## 课堂顺序与定位

|课|主要演示|关键cell|学生交付|
|---|---|---|---|
|L00|00_environment_check|environment / dataset / forward|kernel与shape报告|
|L01|01_data_reading_and_visualization|spectral-comparison|数据卡、光谱图|
|L02|02_svm_baseline|protocol / metrics-toy / raw-confusion|手算指标与split|
|L03|02_svm_baseline|pipeline / kernel-scale / pca-confusion|验证选参与test报告|
|L04|11_patch_hybridsn_classroom|toy-patches；02的spatial-split|坐标断言、空间覆盖|
|L05A|10_training_step_classroom|batch / one-update / mode-gradient|梯度与参数变化|
|L05|12_convolution_classroom|manual-convolution / three-axes|轴、参数与RF表|
|L06|11_patch_hybridsn_classroom|hybridsn-shapes / reshape-check / pca-change|逐层shape与元素对应|
|L07|15_imbalance_analysis_teaching|weighted-loss-toy / target-comparison|损失对照与两目标CAM|
|L08|16_protonet_teaching|episode / episode-update / evaluation|标签预算与K对照|
|L09|17_openset_teaching|calibration / threshold-exercise|新阈值、指标与地图|
|L10|research-case.md + Notebook15|证据表/结果段落|半页研究登记|

Notebook03–09保留历史训练，不是默认现场队列。Notebook02旧版本在archive/中，错误的旧混淆矩阵输出已移除，须重跑才能使用。

## 协议E：统一闭集课堂实验

- IP、seed42、train/val/test中心1024/1025/8200，比例10/10/80。
- SVM：原始200波段标准化，或训练中心拟合PCA12后标准化；C候选1/10/100，按validation OA选，平手保留先到候选。
- CNN：训练中心拟合PCA12与scaler，9×9 patch；CE/weighted共享中心、seed、架构和预算。
- 全图是显示；成绩只算test中心。GT0是未标注。
- 共享split不代表输入信息相同：SVM看单像元，CNN看邻域；预算也不自动等价。
- Few-shot用显式6/5/5类别划分；开集留出指定未知类，不能称16类闭集E。
- 同场景窗口仍可能共享观测；类别不交不代表空间不交。已知val同时用于选模型与校准阈值，未声称独立校准保证。

结果身份卡：数据哈希、split、输入、预处理fit范围、预算、选模规则、评价集合。本次数字及运行配置见`artifacts/teaching/classroom-facts.json`；跨环境变化保留，不追某个目标分数。

## 45分钟节奏与备用

5分钟回顾预判；10分钟toy手算；10分钟代码；12分钟填空/改动；5分钟错误分析；3分钟退出题。录课可拆段。学生答错时回到最小例，不继续滚代码。

提前在根目录和notebooks目录各用新kernel运行课堂Notebook。现场只做单batch/episode、前向、推理和短校准；缺资产可展示已执行输出并说明它是课前记录。投影检查字号、图例及窗口切换，课前试录。

## 课件构建

新课件在`slides/decks/`。入口`build_course_slides.py`默认使用`scripts/build_course_slides.js`（pptxgenjs，需先在`slides/`运行`npm install`），`--engine artifact-tool`可用旧版JS Artifact Tool；内容源为`slides/src/course_content.py`与`classroom_content.py`。旧deck_design.py为历史实现。逐页录课讲稿见`lesson/`。

```powershell
.\.venv\Scripts\python.exe scripts/build_course_slides.py
.\.venv\Scripts\python.exe scripts/check_course_slides.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/export_deck_previews.ps1
```

包检查、渲染、逐页检查与试讲分别记录。自动PASS不代表真实学生已学会。

参考：[scikit-learn预处理边界](https://scikit-learn.org/stable/common_pitfalls.html)、[Carpentries课堂练习](https://carpentries.github.io/instructor-training/instructor/02-practice-learning.html)。
