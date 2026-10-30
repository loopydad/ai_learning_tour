# CelebA 对齐裁剪图片：来源与下载记录

创建：2026-10-30｜更新：2026-10-30

- 标题：CelebFaces Attributes Dataset（CelebA），Align&Cropped Images。
- 作者：Ziwei Liu、Ping Luo、Xiaogang Wang、Xiaoou Tang；香港中文大学 MMLAB。
- 官方来源：https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- 实际下载镜像：https://huggingface.co/datasets/student/celebA/blob/main/Dataset.zip
- 实际文件地址：https://huggingface.co/datasets/student/celebA/resolve/main/Dataset.zip
- 获取日期：2026-10-30（Asia/Shanghai）。镜像上传版本日期未核实。
- 本地压缩包：[celeba_mirror.zip](data/celeba_mirror.zip)。大小：1,414,826,162 字节。
- 本地图片：`data/img_align_celeba/`，共 202,599 张 JPG。
- SHA-256：`9464985fa6cfc84d0fe6b87a6e4f07a275c8bd3dfc8abd713218765de7d62306`。
- 使用范围：官方说明为非商业研究用途，具体条款见官网 Agreement。

## 处理记录与读取范围

2026-10-30：用户要求下载数据。官方 Google Drive 部分请求返回限制页面，
未取得完整官方压缩包；改从上述镜像下载。实际 SHA-256 与镜像 API 返回的
LFS 对象哈希一致。这验证下载文件与镜像一致，未独立证明每张图片与官方文件逐字节一致。

镜像压缩包中 JPG 位于 `Dataset/CelebA_train/img_align_celeba/`。
解压时仅将 JPG 原始字节复制到本地默认图片目录，不缩放、不重编码。
逐文件解压读取到末尾，压缩包 CRC 检查通过；图片文件名集合经检查恰为
`000001.jpg` 至 `202599.jpg`，无重复、无缺号。未导入属性或官方划分文件。

抽查首张、中间张、末张图片：Pillow `verify()` 通过，均为 RGB、178×218。
未逐张做图片解码检查。图片文件总大小为 1,409,859,160 字节。

执行现有数据入口后，首批真实图片输出 `(8, 3, 104, 104)`、`torch.float32`、
范围 `[-1.000, 1.000]`。缩放、裁剪和归一化仅在加载时进行。
未运行 DDPM 训练，未检查生成质量，数据下载不计为算法掌握证据。
