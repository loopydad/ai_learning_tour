# DDPM：论文与示范代码来源

创建：2026-10-31｜更新：2026-10-31

## 原论文

- 标题：Denoising Diffusion Probabilistic Models。
- 作者：Jonathan Ho、Ajay Jain、Pieter Abbeel。
- 版本：arXiv:2006.11239v2，2020-12-16。
- 地址：https://arxiv.org/html/2006.11239v2
- 核对日期：2026-10-30、2026-10-31。
- 实际读取范围：第 2 节前向过程及高斯后验；第 3.2 节噪声参数化，公式 (4)、(6)–(12)，算法 1–2；简化噪声预测损失的相关讨论。不是全文精读记录。
- 保存范围：未保存正文；本记录保留来源与定位，推导归入 [知识页](../../wiki/ddpm.md)。

## 示范代码

- 标题：RethinkFun/DDPM，`ddpm.py`。
- 机构/作者标识：GitHub 账号 RethinkFun；个人作者身份未另行核实。
- 地址：https://github.com/RethinkFun/DDPM/blob/main/ddpm.py
- 获取日期：2026-10-30；读取了当时 main 分支的整个文件（528 行），未固定 commit，后续 main 可能变化。
- 读取用途：确认 CelebA 图片加载方式；比较网络结构、AMP、梯度裁剪、续训及权重合并。
- 读取边界：未运行该仓库的完整训练；示范不构成所有实现正确或效果优于本地版本的证明。
- 本地 [ddpm.py](../../practice/ddpm/ddpm.py) 是用户手写的学习实现，含 Codex 数据加载与部分工程支持，非该示范代码的原件副本。代码评审与改动范围见 [实践记录](../../practice/ddpm/README.md)。

## 处理记录

2026-10-31：结合本次对话、实际代码及运行结果建立 [DDPM 知识页](../../wiki/ddpm.md)、
[实践记录](../../practice/ddpm/README.md) 与 [学习记录](../../journal/2026-10-31-DDPM手写与训练.md)。
数据来源及 SHA-256 另见 [CelebA 来源记录](celeba.source.md)。

2026-10-31：按用户要求将手写实现与运行产物移至 `practice/ddpm/`，本目录只保留原始数据及来源登记；相关链接已更新。
