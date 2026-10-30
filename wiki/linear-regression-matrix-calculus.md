# 线性回归：秩、梯度与链式法则

创建：2026-10-28｜更新：2026-10-28

## 统一符号与求导约定

每行一个样本：$\mathbf X\in\mathbb R^{N\times d}$；$\mathbf w\in\mathbb R^{d\times1}$；$\mathbf y\in\mathbb R^{N\times1}$。截距可以通过常数特征合并。课件每列一个样本，转置位置不同。

$$
\mathbf r=\mathbf X\mathbf w-\mathbf y,\qquad E(\mathbf w)=\frac12\mathbf r^T\mathbf r
$$

本页固定约定：**标量函数的梯度为列向量；向量函数的雅可比为“输出数 × 输入数”。** $\partial E/\partial\mathbf x$ 在本页专指行雅可比，不视为所有教材的通用约定。

## 向量平方究竟是什么

这里是平方范数，不是未定义的向量自乘：

$$
\frac12\|\mathbf r\|^2=\frac12\sum_i r_i^2,\qquad \frac{\partial E}{\partial r_i}=r_i
$$

相同的偏导数有两种排列：

$$
\frac{\partial E}{\partial\mathbf r}=[r_1,\ldots,r_N]=\mathbf r^T,
\qquad \nabla_{\mathbf r}E=[r_1,\ldots,r_N]^T=\mathbf r
$$

因此 $\nabla_{\mathbf x}(\mathbf x^T\mathbf x)=2\mathbf x$；行雅可比则为 $2\mathbf x^T$。若把逐元素平方视为向量输出，导数是 $2\operatorname{diag}(\mathbf x)$，与本页标量平方范数不同。

## 为什么链式法则里 X 转置后到了左边

从行雅可比开始，外层乘内层：

$$
\underbrace{\frac{\partial E}{\partial\mathbf w}}_{1\times d}
=\underbrace{\frac{\partial E}{\partial\mathbf r}}_{1\times N}
\underbrace{\frac{\partial\mathbf r}{\partial\mathbf w}}_{N\times d}
=\mathbf r^T\mathbf X
$$

其中 $J_{ij}=\partial r_i/\partial w_j=X_{ij}$。把整体转置成列梯度，使用 $(AB)^T=B^TA^T$：

$$
\boxed{\nabla_{\mathbf w}E=(\mathbf r^T\mathbf X)^T=\mathbf X^T\mathbf r}
$$

**不是随意交换矩阵顺序，而是对整个乘积转置。** 一般列梯度链式法则为 $\nabla_{\mathbf w}E=J^T\nabla_{\mathbf r}E$。

形状的来源是定义：$\mathbf f:\mathbb R^n\to\mathbb R^m$ 的雅可比第 $i$ 行对应输出 $f_i$、第 $j$ 列对应输入 $x_j$，故为 $m\times n$。标量输出的雅可比是一行；梯度再把这一行转成列。

用微分可再次核对：

$$
dE=\mathbf r^T d\mathbf r=\mathbf r^T\mathbf X\,d\mathbf w
=(\mathbf X^T\mathbf r)^T d\mathbf w
$$

与 $dE=(\nabla_{\mathbf w}E)^Td\mathbf w$ 比较即可。维度检查能发现错误，但不能代替求导证明。

分量检查：

$$
\frac{\partial E}{\partial w_j}=\sum_i\frac{\partial E}{\partial r_i}\frac{\partial r_i}{\partial w_j}=\sum_i r_iX_{ij}
$$

这是 $\mathbf X^T\mathbf r$ 的第 $j$ 个分量。这个解释核对数值计算；行列约定解释乘法顺序，二者各回答不同问题。

## 正规方程、凸性与解析解

$$
\nabla E=\mathbf X^T(\mathbf X\mathbf w-\mathbf y),\qquad \nabla^2 E=\mathbf X^T\mathbf X\succeq0
$$

令梯度为零，得正规方程 $\mathbf X^T\mathbf X\mathbf w=\mathbf X^T\mathbf y$。满列秩时：

$$
\mathbf w^*=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y
$$

这是理论表达，实际实现可以用稳定的最小二乘求解器，不必显式求逆。凸性与数值是否容易、是否已经收敛是不同问题，见[优化边界](model-complexity-generalization-optimization.md#多项式拟合为何没有较差的局部最小值)。

### 与条件均值和高斯似然的关系

本页回答“给定平方误差目标，如何求参数”；[似然与损失页](likelihood-loss-inductive-bias.md#mse-与交叉熵的共同来源)回答“在什么概率模型下会得到这个目标”。独立观测、线性均值 $\mu_{\mathbf w}(x)=\mathbf w^Tx$、同一固定方差的高斯条件噪声，使负对数似然成为残差平方和的正常数倍加常数，因此得到相同的最优参数。

但本页梯度、秩和正规方程的代数推导本身不要求噪声高斯。[非高斯噪声下的条件均值解释](likelihood-loss-inductive-bias.md#非高斯噪声不会自动否定最小二乘)进一步说明：要区分“能求最小二乘解”“总体平方损失由条件均值最小化”和“它是正确设定模型的最大似然估计”，三者的前提不同。

## 不满列秩为何一定有无穷多个参数解

直觉例子：$x_2=2x_1$ 时，$w_1x_1+w_2x_2=(w_1+2w_2)x_1$。数据只能确定组合，不能分别确定两个系数。

数学上，$\operatorname{rank}(\mathbf X)<d$ 等价于存在 $\mathbf v\ne0$，使 $\mathbf X\mathbf v=0$。于是：

$$
\mathbf X(\mathbf w+t\mathbf v)=\mathbf X\mathbf w,\qquad \forall t\in\mathbb R
$$

任何最优参数都能沿此方向任意移动，预测与损失不变，所以无约束线性最小二乘的参数最优解有无穷多个。

又有 $\mathbf X^T\mathbf X\mathbf v=0$，因此 $\mathbf X^T\mathbf X$ 不可逆。损失曲面沿这个方向平坦，没有更差的局部最小值。

| 对象 | 不满列秩时 |
| --- | --- |
| 最优参数 | 无穷多个 |
| 最优训练预测 | 唯一：是 $\mathbf y$ 在列空间上的正交投影 |
| 新输入预测 | 未必相同；取决于新输入是否也消去零空间方向 |

若特征关系只是单位换算，未来始终保持，预测仍相同；若共线性只在训练样本中偶然成立，未来预测就可能分歧。

## 岭回归与归一化

全部参数受 L2 惩罚、$\lambda>0$ 时：

$$
J=\frac12\|\mathbf X\mathbf w-\mathbf y\|^2+\frac\lambda2\|\mathbf w\|^2,
\qquad \mathbf w^*=(\mathbf X^T\mathbf X+\lambda I)^{-1}\mathbf X^T\mathbf y
$$

新增 $\lambda I$ 使 Hessian 正定，参数解唯一。若截距不受惩罚，需要相应调整惩罚矩阵，不能照搬全部参数惩罚的论证。

若数据项改成 $\frac1{2N}\|\cdot\|^2$ 而惩罚项不变，解析解中相应变成 $N\lambda I$。跨教材比较前要统一尺度。

同一个 L2 项还可由[高斯参数先验下的 MAP](likelihood-loss-inductive-bias.md#高斯先验与归纳偏置)导出，连接了概率假设与本页解析解；惩罚强度变化的数学性质见[正则化单调性](model-complexity-generalization-optimization.md#正则化增强能证明什么)。前者解释目标的一种来源，后者说明精确最优解如何变化；它们都不单独保证测试误差下降。

多项式用 $\phi(x)=[1,x,\ldots,x^M]^T$ 作特征，仍然对参数线性，因此适用相同的矩阵推导。

## 我的理解与验证

- 用户先追问秩亏的直觉与数学解释；随后连续追问链式法则的顺序、转置到左边、向量平方求导及导数形状。
- 用户明确纠正“不是我得意思是 XT到左边去了这件事情”。实际卡点是导数布局，不能仅用“列点积”解释代替。
- 2026-10-28：逐项确认后，用户报告“都已经完成证明”，相关疑问已关闭。记录为用户自述，未展示证明步骤，Codex尚未核查，见 [Q005](../questions.md#q005秩亏为什么使最小二乘参数不唯一) 与 [Q006](../questions.md#q006链式法则为什么出现左侧的转置矩阵)。

## 来源与重要修订

依据[第二讲来源](../sources/ml-overview.source.md)P58–72；分量推导、零空间证明与形状说明为补充推导。

2026-10-28：纠正此前“不满列秩时参数可能不唯一”的弱表述；在本页明确的无约束线性最小二乘设定中必有无穷多个参数最优解。前次讲解混用了行雅可比与列梯度，本页统一约定，并保留纠正来由于[学习记录](../journal/2026-10-28-机器学习概述.md)。

2026-10-28：补充最小二乘与概率建模、岭回归与高斯先验的联系；综合解释由 Codex 补写，未新增实验或用户掌握证据。
