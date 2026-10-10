# L05 · 1D、2D、3D，究竟在沿哪个轴卷积？

当前课堂执行入口：先L05A/Notebook10，再Notebook12：manual-convolution / three-axes / receptive-field。以下历史入口只用于延伸阅读，逐课顺序见[classroom-guide](../classroom-guide.md)。
目标：讲清输入轴、核与通道，而非背模型排行榜。前置 L04；notebooks 04/05/06，ch05–07。定位 `SpectralCNN1D` / `SpectralSpatialCNN2D` / `SpectralSpatialCNN3D`。

## 25 分钟节奏

0–3：给三个张量，请观众判断各用哪种卷积；3–8：通道与滑动轴；8–13：一层参数/输出手算；13–17：dummy forward 核对；17–21：成本与预测图；21–25：归纳偏置和公平对照练习。

## 六页幻灯片

1. 1D `(N,1,B)`：沿波段索引；2D `(N,B,H,W)`：沿空间；3D `(N,1,B,H,W)`：沿深度和空间。
2. **2D 普通卷积聚合全部输入通道**，不是只看一个波段；3D 区别在沿谱/分量轴局部滑动与共享权重。
3. 参数 `Cout*(Cin/groups*kernel_volume+1)`；depthwise 是特殊设定。
4. 输出长度公式含 stride/padding/dilation；感受野递推 `r'=r+(k−1)j, j'=j*s`，池化也计算。
5. PCA 分量顺序不是波长序列；“13 个分量”不能称 130nm。
6. 比较时列输入、patch、PCA、预算；单次差异不能归因到卷积维数。

## 可运行对应

三种小张量分别 forward，hook 打印 shape；手算 2D `12→32,k=3` 得 3488 参数。屏幕上只突出首层和末层，不逐行读 import。已有预测图用于解释失败结构，必须注明历史协议与单次运行。

## 受控改动 / 答案

同一输入和分类头下将核由 3 改 1：参数下降但感受野收缩。不能在同时改 patch/PCA/训练轮次后说“证明 3D 更好”。现场改 batch 不改空间维，验证参数量不随 batch 变化。

## 科研产出与核验

逐层 shape/MACs 表（计数范围写明）和比较协议表。来源：PyTorch Conv 文档，ch05–07 原始引用需按 source-audit 核验。AI 的感受野答案必须含池化；GPU 墙钟不由参数量单独决定。长训练提前运行，现场只做前向与一小批反向。
