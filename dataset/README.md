# 数据集文件与使用边界

本地数据用于教学复现。代码的开源许可**不自动覆盖数据文件**；本页不授予第三方数据的再分发许可。发布课程或分发数据前应分别核对原始提供方条款。

| 代码 | 输入文件 / MAT键 | GT文件 / MAT键 | 本仓库已读取的形状 | 类别 |
|---|---|---|---|---:|
| IP | Indian_pines_corrected.mat / indian_pines_corrected | Indian_pines_gt.mat / indian_pines_gt | 145×145×200 / 145×145 | 16 |
| SA | Salinas_corrected.mat / salinas_corrected | Salinas_gt.mat / salinas_gt | 512×217×204 / 512×217 | 16 |
| PU | PaviaU.mat / paviaU | PaviaU_gt.mat / paviaU_gt | 610×340×103 / 610×340 | 9 |

这些形状来自此前本地加载检查，不是从传感器宣传参数推断。课程加载器以 `src/hsi_learning/data.py::DATASET_SPECS` 为配置入口。

## 来源记录

- 原始场景汇总入口：[EHU Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)。
- 本项目的SA/PU文件此前从 [HybridSN作者仓库](https://github.com/gokriznastic/HybridSN) 数据目录取得；镜像不是原始权属证明。
- IP原始加入来源需继续追溯git历史与原始下载记录，不编造采集日期/许可。
- `Data_Declaration` 是旧资料，包含版本/尺寸/单位不一致的叙述，仅作历史线索，不作为精确shape、波长、反射率单位的依据。

## 关键语义

- GT=0代表未标注位置，不代表该位置没有光谱，不应直接作为一个监督背景类。
- corrected数据的列索引需结合删带映射才能转成物理波长；没有映射就使用band index。
- PCA分量是原始波段的线性组合，不是新的窄波段，不能按每个分量10nm解释。
- 数值量纲必须由该文件的处理说明确认；不能只因值域就断定是DN或反射率。

## 文件完整性与引用

正式教学包的 `run_config.json` 记录数据SHA256；核对哈希可证明使用的是同一文件，不能证明来源或许可。重新下载不同版本时另存并重新验证键/shape/标签数，不直接覆盖旧实验的输入。

论文/课件引用场景来源、提供方以及实际下载入口；许可证、图像使用条件和是否允许再分发目前仍需对原始页面逐项核验。未经确认不声称“公开下载=无限制使用”。
