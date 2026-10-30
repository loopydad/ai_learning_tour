# AI 学习 Wiki

按主题查阅知识，区分课件内容、数学结论与实际验证。

- [当前进度](progress.md)
- [问题追踪](questions.md)：已提出问题、解释进展与待确认之处。

## 机器学习基础

- [模式识别与机器学习绪论](wiki/machine-learning-introduction.md)：学习任务、曲线拟合、正则化与模型选择的主线。
- [模型复杂度、泛化与优化的数学边界](wiki/model-complexity-generalization-optimization.md)：哪些结论可证明，哪些依赖假设，以及实际训练为何可能达不到最优结果。

## 第二讲与专题

- [机器学习概述：第二讲导航](wiki/machine-learning-overview.md)：覆盖98页课件的主线、公式与纠错定位。
- [概率、采样与期望](wiki/probability-sampling-expectation.md)：逆变换采样的百分位直觉及大数定律。
- [线性回归：秩、梯度与链式法则](wiki/linear-regression-matrix-calculus.md)：零空间解释不唯一，统一行雅可比与列梯度。
- [似然、损失函数与归纳偏置](wiki/likelihood-loss-inductive-bias.md)：连接MSE、交叉熵、先验与VAE。
- PAC、样本复杂度与数据质量见[已有泛化页](wiki/model-complexity-generalization-optimization.md#pac与样本复杂度)，不另建重复正文。

## 第三讲：最近邻与决策树

- [第三讲导航与KNN](wiki/nearest-neighbors-decision-trees.md)：分类评价、近邻预测、维度灾难及课件纠错。
- [决策树：递归划分、熵与剪枝](wiki/decision-trees.md)：用同一示例比较基尼与熵，解释信息增益和实现准则的区别。

## 第四讲：线性分类

- [第四讲导航、投影与Fisher](wiki/linear-classification.md)：投影几何、尺度约束、Fisher条件与感知机。
- [高斯推导、Logistic与Softmax](wiki/gaussian-logistic-softmax.md)：贝叶斯后验逐步推导、课件纠错、闭式估计及概率输出函数。

## 生成模型

- [DDPM：噪声预测与逐步生成](wiki/ddpm.md)：区分累计噪声与单步噪声，连接训练目标、后验均值及采样。

## 实践记录

- [手写DDPM：CelebA最小实现与40轮训练](practice/ddpm/README.md)：用户手写实现，含 Codex 辅助；记录数据划分、运行证据、生成样例及验证边界。

## 来源与学习记录

- [DDPM论文与示范代码来源](sources/ddpm/ddpm.source.md)
- [CelebA数据来源与校验](sources/ddpm/celeba.source.md)
- [2026-10-30–31：DDPM手写与训练记录](journal/2026-10-31-DDPM手写与训练.md)

- [绪论课件来源](sources/ml-introduction.source.md)
- [2026-10-28：从课件现象追问数学保证](journal/2026-10-28-机器学习绪论.md)

- [第二讲课件来源及OCR](sources/ml-overview.source.md)
- [VAE原论文局部核对](sources/auto-encoding-variational-bayes.source.md)
- [2026-10-28：第二讲对话复盘](journal/2026-10-28-机器学习概述.md)

- [第三讲课件来源及OCR](sources/nearest-neighbors-decision-trees.source.md)
- [2026-10-28：最近邻与决策树学习记录](journal/2026-10-28-最近邻与决策树.md)

- [第四讲课件来源及OCR](sources/linear-classification.source.md)
- [2026-10-28：线性分类学习记录](journal/2026-10-28-线性分类.md)

