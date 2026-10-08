# L07 · OA 很高，为什么小类仍全错？

目标：把错误诊断接到标准损失和真实类别解释。前置 L02/L05；notebook 15，ch15。定位 `loss-comparison`、`gradcam`、`maps`（修正版 cell ID）。

## 节奏（25 分钟）

0–3：隐藏 OA，只展示 Oats 混淆行；3–7：加权 CE 和 focal 公式；7–11：手算一次权重；11–15：加载 CE/weighted 演示包做全图推理；15–21：真实 Grad-CAM；21–25：证据边界和练习。

## 六页幻灯片

1. 总体好不等于各类好；看 support 和逐类 recall。
2. `w_c=N/(C*n_c)`；与 focal `(1-p_t)^gamma` 区别：类频率 vs 样本难度。
3. 权重/采样改变有效训练目标，不保证 OA 和 AA 同时上升。
4. GT、CE、weighted、测试错误图，用同一 class→color 对应。
5. Grad-CAM：`alpha_k=mean_ij(d logit_c/d A_kij)`，`ReLU(sum alpha_k*A_k)`；不是通道均值。
6. 三种证据：指标、局部解释、受控扰动；没有一张图能单独证明因果机制。

## 理论→代码

`build_criterion` 对应损失；`teaching.grad_cam` 对应梯度加权。显式传目标类别、显式选 CE 或 weighted 模型。取真实最后卷积（shape 由 hook 得到），不硬写 5×5。插值不会增加解释分辨率。

## 受控实验 / 练习

固定同一 patch，分别用预测类与另一个类作为目标，比较 CAM；如果全零也如实显示，不重新挑“好看”的样本。遮挡热区后记录 logit 变化，只作为局部扰动证据，遮挡本身可能产生分布外输入。

答案：两个目标的梯度与热区可以不同；响应均值不随目标变化，不能冒充 Grad-CAM。t-SNE 簇形态也受投影和采样影响，不能只凭图证明泛化。

## 科研产出与执行安排

CE/weighted 配对报告、逐类表、一组正确/错误样本的解释卡。完整训练提前跑；现场仅计算一次损失与反向、加载 checkpoint 后生成 map/CAM。来源：Lin et al. 2017；Selvaraju et al. 2017。AI 生成“可解释性”代码要检查是否真的对目标 logit 求导。
