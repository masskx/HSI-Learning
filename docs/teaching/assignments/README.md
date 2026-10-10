# 六个实践任务：交付证据，不交最高分

配套[学生填空与迁移任务](../../../student_workbook/README.md)。按A01→A06递进。每题100分，评分围绕协议、实现、解释、复现；不按准确率高低排名。提交包统一包含 `README.md`（环境/命令）、最小代码或notebook、配置、输出及限制说明。先做再看 [解答与评分参考](solutions.md)。

## A01 数据卡与切片（L00/L01）

起点：notebook 00 `dataset`；notebook 01 `loadmat`。

任务：为IP制作一页数据卡，输出空间切片、像元光谱、GT和类频率。使用下面最小例子先理解轴：

```python
x = np.arange(3*4*5).reshape(3,4,5)
assert x[:,:,2].shape == (3,4)
assert x[1,2,:].shape == (5,)
```

交付：`data-card.md`、`slices.png`、`checks.py`。断言cube/GT空间尺寸一致、非零标签数量正确、同一像元向量长度等于B。说明未标注不等于无光谱；没有波长表不标nm。

评分：shape/键/标签30，图与注释20，来源与单位边界30，运行说明20。错误例：把 `cube[b,:,:]` 当波段；将显示拉伸写成训练预处理。

科研产出：一段只包含有来源的数据集描述。

## A02 基线与指标（L02/L03）

起点：notebook02 `protocol` / `metrics-toy` / `pipeline`；C由validation选。

任务：先手算900/90/10三类全部预测A的OA/AA/Kappa，再运行SVM Pipeline。训练/验证/测试分别存索引；超参由validation选择。

交付：`protocol.json`、`metrics.json`、`report.txt`、测试错误map、手算笔记。断言背景不参与指标、索引不相交、标准化fit只看train。不要使用全图OA冒充test OA。

评分：信息边界35，指标正确25，baseline配置20，受限解释20。错误例：对全数据先fit scaler；为提高分数调整test_size后只保留最好一次。

科研产出：含输入、划分、预算、seed、选择规则的baseline描述。

## A03 patch与感受野（L04–L06）

起点：`teaching.patches_at`、`PatchDataset.__getitem__`、HybridSN `forward`。

任务：toy数组检查中心与四角；写逐层shape/RF计算；打印HybridSN 3D→2D元素数相等。至少检测一个故意引入的axis错误。

交付：`test_patches.py`、`shapes.csv`、一张窗口示意。验收：奇数size约束、中心像元一致、Pool计入RF、PCA分量不换算波长。1D示例RF26、2D示例RF12（在各自第三卷积后）。

评分：坐标30，shape/RF30，错误测试20，物理解释边界20。错误例：把普通Conv2d说成单通道；把view当任意轴置换。

科研产出：可核算的输入与架构说明。

## A04 不均衡与Grad-CAM（L07）

起点：notebook15 `loss-comparison` / `gradcam` / `maps`。

任务：同split比较CE与weighted；报告逐类recall与support。选同一patch、两个目标类别，执行真实Grad-CAM；保存至少一个失败/全零例，不只挑漂亮热区。

交付：`comparison.csv`、`cam-targets.png`、模型/目标/像元ID清单、解释段落。断言来自目标logit的梯度存在，hook释放，输出尺寸匹配输入；特征/标签配对未变化。

评分：公平对照30，Grad-CAM正确30，失败分析20，可复现20。错误例：通道均值冒充Grad-CAM；热区高就断言“模型因果依赖它”。

科研产出：一段区分观察与机制假说的分析。

## A05 Few-shot协议（L08）

起点：notebook16 `protocol` / `episode` / `prototype-math`。

任务：报告meta-train/val/test类集合与总标签预算；画一个episode的support/query位置。用固定encoder和配对query比较K，不按test选择checkpoint。

交付：`class-split.json`、`episode.npz`、`kshot.csv`、原型手算、全图展示说明。断言类集合不交（class-disjoint模式）、support/query ID无重复、每episode严格N类、候选类池一致。说明same-scene空间依赖。

评分：协议/预算35，原型距离25，采样检查25，解释15。错误例：train/test只是同类不同像元却声称未见类泛化；静默缩小N-way。

科研产出：可复查的N-way K-shot方法与评价设定。

## A06 开集与证据写作（L09/L10）

起点：notebook17 `scores` / `calibration` / `evaluation` / `maps`。

任务：仅known val选择阈值；报告AUROC、已知误拒/正确接受、未知召回/漏检；输出白色unknown、黑背景地图与错误图。接受率.95→.90只用val重校准，写明预期和观察。

交付：`thresholds.json`、`scores.npz`、`open-map.png`、`errors.png`、200词以内的结果与限制段。断言打乱特征标签对后原型不变、改变test标签不改变阈值、unknown ID与0背景不同。

评分：校准信息边界35，指标和映射25，地图20，研究写作20。错误例：从test ROC最佳点选阈值；把白色都解释为正确未知。

科研产出：一份半页研究登记（问题、baseline、唯一变化、失败解释、证据范围）。AI可以帮助润色，但不得新增数值/引用/因果词。
