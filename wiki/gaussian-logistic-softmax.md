# 概率线性分类：高斯推导、Logistic与Softmax

创建：2026-10-28｜更新：2026-10-28

别名：逻辑回归 / Logistic Regression；Softmax Regression；共享协方差高斯分类器。

## 解决的问题与符号

本页对应第四讲P24–44，以及用户对高斯推导和Sigmoid/Softmax的追问。[第四讲导航](linear-classification.md)保存投影、Fisher与感知机。

$\omega_c$ 是类别，不是权重w；$\mu_c\in\mathbb R^D$ 是类均值，$\Sigma\in\mathbb R^{D\times D}$ 是共享协方差，$\pi_c=p(\omega_c)$ 是先验。

生成式建模 $p(x\mid\omega_c)$ 和 $\pi_c$，再求后验；判别式可直接建模 $p(\omega_c\mid x)$。这两种路线可以有同样的后验函数形式，但学习目标与参数估计不同。判别式方法不限于分类，生成能力也不与分类能力等价。

## 为什么 $p(x\mid\omega_c)$ 不等于 $p(x)$

前者只看某一类，后者把各类混合。例如x是重量，“所有水果的重量分布”通常不同于“西瓜的重量分布”。

$$
p(x)=\sum_c p(x\mid\omega_c)\pi_c.
$$

连续x这里是密度，不是取到某个精确点的概率。如果x与类别独立，才会对所有正先验类别有 $p(x\mid\omega_c)=p(x)$，从而 $p(\omega_c\mid x)=\pi_c$，特征不提供分类信息。个别x处偶然相等不意味着全局独立。

## 从贝叶斯公式到Sigmoid：此步不需要高斯

二分类中：

$$
p(\omega_1\mid x)=\frac{p(x\mid\omega_1)\pi_1}
{p(x\mid\omega_1)\pi_1+p(x\mid\omega_2)\pi_2}.
$$

当相关量为正，定义自然对数比：

$$
a=\log\frac{p(x\mid\omega_1)\pi_1}{p(x\mid\omega_2)\pi_2}.
$$

分子分母同除以 $p(x\mid\omega_1)\pi_1$，得到：

$$
p(\omega_1\mid x)=\frac1{1+e^{-a}}=\sigma(a).
$$

这仅把二分类后验改写为对数几率形式，还没证明a对x线性。零概率情况需按极限或单独定义处理。

## 共享协方差高斯：为什么对数比是线性的

假设两类均为高斯，协方差相同且正定，先验均为正：

$$
p(x\mid\omega_c)=\frac1{(2\pi)^{D/2}|\Sigma|^{1/2}}
\exp\left[-\frac12(x-\mu_c)^T\Sigma^{-1}(x-\mu_c)\right].
$$

取对数并相减，共同归一化常数抵消：

$$
a=-\frac12(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)
+\frac12(x-\mu_2)^T\Sigma^{-1}(x-\mu_2)
+\log\frac{\pi_1}{\pi_2}.
$$

由于 $\Sigma^{-1}$ 对称：

$$
(x-\mu_c)^T\Sigma^{-1}(x-\mu_c)
=x^T\Sigma^{-1}x-2\mu_c^T\Sigma^{-1}x+\mu_c^T\Sigma^{-1}\mu_c.
$$

其中 $x^T\Sigma^{-1}\mu_c$ 与 $\mu_c^T\Sigma^{-1}x$ 都是标量且互为转置，因此相等。逐项代入：

$$
\begin{aligned}
a={}&-\tfrac12x^T\Sigma^{-1}x+\mu_1^T\Sigma^{-1}x-\tfrac12\mu_1^T\Sigma^{-1}\mu_1\\
&+\tfrac12x^T\Sigma^{-1}x-\mu_2^T\Sigma^{-1}x+\tfrac12\mu_2^T\Sigma^{-1}\mu_2
+\log\frac{\pi_1}{\pi_2}.
\end{aligned}
$$

两个关于x的二次项一正一负抵消：

$$
a=(\mu_1-\mu_2)^T\Sigma^{-1}x
+\frac12(\mu_2^T\Sigma^{-1}\mu_2-\mu_1^T\Sigma^{-1}\mu_1)
+\log\frac{\pi_1}{\pi_2}=w^Tx+b,
$$

其中：

$$
\boxed{w=\Sigma^{-1}(\mu_1-\mu_2)},\qquad
\boxed{b=\frac12(\mu_2^T\Sigma^{-1}\mu_2-\mu_1^T\Sigma^{-1}\mu_1)+\log\frac{\pi_1}{\pi_2}}.
$$

最终 $p(\omega_1\mid x)=\sigma(w^Tx+b)$。若协方差不同，二次项一般不抵消，边界可能为二次曲面。

### P26常数项纠错

截图将常数项写成 $\frac12(\mu_2-\mu_1)^T\Sigma^{-1}(\mu_2-\mu_1)$，一般错误。正确等价乘积为：

$$
\frac12(\mu_2-\mu_1)^T\Sigma^{-1}(\mu_2+\mu_1).
$$

一维反例：$\mu_1=1,\mu_2=3,\Sigma=1$，正确项为 $(9-1)/2=4$，PPT所列为 $(3-1)^2/2=2$。这是Codex验证公式的示例，不是用户练习。

## 这是不是解析解

要区分两个层次：

1. 已知均值、协方差和先验时，上述是后验分类器参数的解析表达式，尚未做数据拟合。
2. 用数据拟合这个生成模型时，其最大似然估计也有闭式形式：

$$
\hat\mu_c=\frac1{N_c}\sum_{i:y_i=c}x_i,\quad
\hat\pi_c=\frac{N_c}{N},\quad
\hat\Sigma=\frac1N\sum_c\sum_{i:y_i=c}(x_i-\hat\mu_c)(x_i-\hat\mu_c)^T.
$$

这里用共享协方差的MLE归一化，需每类有样本且估计协方差满足相应可逆条件；奇异时不能直接代入逆矩阵。

直接训练Logistic回归则最大化条件似然，通常无一般闭式解，使用数值优化。两条路线即使同样输出Sigmoid，也不保证同一数据上得到同样的w、b。

## 为什么用Sigmoid

二分类标签改用 $y\in\{0,1\}$，令 $p=p(y=1\mid x)$。若建模假设为线性对数几率：

$$
\log\frac p{1-p}=z=w^Tx+b,
$$

解出p必然得到 $p=\sigma(z)=1/(1+e^{-z})$。因此它不仅把实数压进(0,1)，还赋予分数明确含义：z增加1，概率比 $p/(1-p)$ 乘以e；z为0时p为0.5。

高斯生成模型可以导出这个形式；直接训练Logistic时，不必假设输入的类条件分布为高斯。也可用其他映射，例如标准正态CDF的Probit模型，假设与链接不同。Sigmoid不是唯一合法概率映射。

伯努利似然的负对数是二元交叉熵：

$$
\ell=-y\log p-(1-y)\log(1-p),\qquad
\frac{\partial\ell}{\partial z}=p-y.
$$

因此平均损失梯度为 $\nabla_w R=N^{-1}\sum_i(p_i-y_i)x_i$，$\partial R/\partial b=N^{-1}\sum_i(p_i-y_i)$。概率与损失的通用推导、交叉熵和KL的条件关系以[已有似然与损失页](likelihood-loss-inductive-bias.md)为主，不在这里复制独立版本。

补充区别：[决策树](decision-trees.md)的节点熵评价样本类别本身的不确定性；这里的交叉熵评价用模型概率预测实际标签的损失。单次硬标签one-hot不是已知的总体条件分布。

## 为什么用Softmax：推广概率比关系

互斥多分类给每类一个分数 $z_c=w_c^Tx+b_c$，定义：

$$
p_c=\frac{e^{z_c}}{\sum_j e^{z_j}},\qquad
\sum_c p_c=1,\qquad
\log\frac{p_i}{p_j}=z_i-z_j.
$$

指数保证正值，归一化保证总和为1；更重要的是分数差决定对数概率比。二分类时 $p_1=\sigma(z_1-z_2)$，所以Softmax与Sigmoid是同一思路。

若真实类别为k，$\ell=-\log p_k$，对one-hot标签有 $\partial\ell/\partial z_c=p_c-y_c$，梯度形式简洁。预测 $\arg\max p_c=\arg\max z_c$；仅需类别时不必显式计算Softmax概率。

两类候选分界满足 $(w_i-w_j)^Tx+(b_i-b_j)=0$，实际边界还要求这两类优于其余类别，不应统一称作相交射线。

| 任务 | 常用方式 |
| --- | --- |
| 二分类 | 一个Sigmoid，另一类概率为1-p |
| 互斥多分类 | Softmax，总概率为1 |
| 多标签，一个输入可属于多类 | 每个标签一个Sigmoid，不要求跨标签概率和为1 |

这些选择建立概率模型并配合相应损失训练；满足概率约束不保证校准或准确性。每标签Sigmoid本身也不自动描述标签之间的依赖关系。

## 我的理解与验证

2026-10-28：用户从 $p(x\mid\omega_c)$ 与 $p(x)$ 的区别，追问P26推导，随后询问是否属于解析解及为何使用Sigmoid/Softmax。Codex给出推导并指出截图中的常数项错误。状态：接触过；用户独立推导、解释和应用未验证。本轮无训练实验；上一讲完成证明的自述不延伸到本页。

## 来源与重要修订

[课件来源及读取范围](../sources/linear-classification.source.md)P24–44；[实际学习过程](../journal/2026-10-28-线性分类.md)。2026-10-28：首次整理中保留P26常数项纠错，区分生成式闭式参数估计与判别式Logistic优化；对P29、33、38、41、46的表述补充边界条件。
