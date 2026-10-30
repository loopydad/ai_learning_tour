# AI Learning Tour

个人 AI 学习 Wiki：保存概念解释、数学推导、手写代码、实际实验及学习过程。
知识、用户理解与代码运行证据分别记录，不将 AI 已讲解或训练已完成等同于用户已掌握。

- [知识入口](index.md)
- [当前进度](progress.md)
- [问题追踪](questions.md)
- [手写 DDPM 实践](practice/ddpm/README.md)：用户手写学习实现，数据加载与部分工程支持由 Codex 辅助。

## 目录

- `wiki/`：可复用的概念、机制、推导与学习状态。
- `journal/`：真实学习过程与理解变化。
- `practice/`：代码、实验结果及验证范围。
- `sources/`：来源登记、允许保存的原始材料与提取文本。
- `AGENTS.md`：本 Wiki 的协作与写入约定。

## 本地材料与复现范围

仓库保留代码、笔记、来源说明及已引用的生成图片；不提交 CelebA 数据、模型权重、
环境缓存、凭据和临时处理文件。本地文件不因 Git 忽略而被删除。
某些笔记链接到本机课程 PDF、未提交权重或服务器文件，克隆仓库后不会自动获得这些材料。

DDPM 数据准备与运行方法见 [运行说明](practice/ddpm/usage.md)，
来源及数据使用条件见 [CelebA 来源登记](sources/ddpm/celeba.source.md)。
原始数据默认放在 `sources/ddpm/data/img_align_celeba/`；模型权重需另行获取。
