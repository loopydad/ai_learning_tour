# Auto-Encoding Variational Bayes：局部核对记录

- 作者：Diederik P. Kingma、Max Welling。
- 原文：[arXiv:1312.6114，v11](https://arxiv.org/html/1312.6114v11)，该版本日期2022-12-10；初稿2013-12-20。
- 检索与核对日期：2026-10-28。
- 读取范围：摘要、§2.1–2.2及§2.3相关片段；未阅读全文，不作为完整论文精读记录。
- 未保存全文；下面保留本次使用的公式定位和简短摘记。

## 用于核对的内容

§2.1：生成过程为先抽 $z\sim p(z)$，再抽 $x\sim p_\theta(x\mid z)$；边缘似然需要对 $z$ 积分。

§2.2，式(3)：

$$
\mathcal L(\theta,\phi;x)=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z))
$$

原文将其称为边缘对数似然的下界；本次笔记取其负值表达最小化目标。固定高斯或伯努利解码器与平方误差/BCE的对应关系，按各分布的负对数似然自行展开，不把“VAE=MSE”当作原文结论。

## 处理记录

2026-10-28：用于核对[概率模型与VAE联系](../wiki/likelihood-loss-inductive-bias.md#从无监督密度建模走到-vae)，未展开实现、重参数化推导与实验。
