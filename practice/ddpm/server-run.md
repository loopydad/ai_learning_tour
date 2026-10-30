# DDPM 服务器训练记录

创建：2026-10-30｜更新：2026-10-31

## 本次提交

- 执行者：Codex；用户授权提交服务器训练。
- SSH：`ssh -p 30640 root@120.209.70.195`；密码未写入文件。
- 工作目录：`/root/gpufree-data/ddpm`。
- 环境：Ubuntu 22.04、Python 3.11.13、PyTorch 2.7.1+cu128、NumPy 2.2.6、Pillow 11.0.0。
- GPU：NVIDIA L40，驱动报告显存 46068 MiB。
- 已上传当前 `ddpm.py`、`dataloader.py` 及数据来源说明；两个 Python 文件与本地 SHA-256 一致。
- 数据：上传本地 `celeba_mirror.zip`，SHA-256 与 [来源记录](../../sources/ddpm/celeba.source.md) 一致；
  解压后 202,599 张 JPG，文件名集合检查及首/中/末张图片校验通过。

## 配置与运行

沿用当前算法与网络，batch size 8、图片 104×104、T=1000、学习率 1e-4、
总目标 100 轮；训练图片 201575 张，固定验证图片 1024 张。
服务器 `run_server.py` 仅通过 `functools.partial` 将数据加载工作进程数设置为 4；
`train.sh` 设置 `OMP_NUM_THREADS=4`，执行 `/opt/conda/bin/python` 环境中的 `python -u run_server.py`。

后台启动命令（已执行，不要重复提交）：

```bash
cd /root/gpufree-data/ddpm
nohup bash train.sh > train.log 2>&1 < /dev/null &
echo $! > train.pid
```

- 本次后台 shell PID：431；训练 Python PID：433。PID 只对应本次启动。
- 日志：`/root/gpufree-data/ddpm/train.log`。
- 完成或失败时的退出码：`/root/gpufree-data/ddpm/train.exit`；提交时尚无该文件，后续状态见停止记录。
- 权重：`/root/gpufree-data/ddpm/outputs/latest.pt`，每轮训练与验证结束后保存。
- 图片：`/root/gpufree-data/ddpm/outputs/samples_XXXX.png`，每 10 轮及最终轮保存。
- 本地旧进程 PID 1945 保持暂停；没有 checkpoint，服务器从头训练。

连接后查看：

```bash
tail -f /root/gpufree-data/ddpm/train.log
```

## 本次验证边界

正式提交前使用真实图片在 CUDA 上完成单批训练、反向传播、优化器更新及单批验证，
损失均为有限值。正式日志已确认 `cuda`、`Epoch 1/100`、首批损失 1.047571；
GPU 查询显示训练进程实际占用 GPU；随后日志已推进至第 1 轮的 Batch 3300/25197。
这是运行启动证据，不代表完整训练完成、生成质量验证或用户掌握。

## 2026-10-31：检查并停止

用户要求检查是否可早停，若可以则停止。检查时已完成 25 轮，正在第 26 轮。
最近验证 MSE：

| 轮数 | val_loss |
| --- | --- |
| 20 | 0.013953 |
| 21 | 0.013972 |
| 22 | 0.013974 |
| 23 | 0.013894 |
| 24 | 0.013889 |
| 25 | 0.013949 |

第 20 到第 24 轮的验证 MSE 改善约 0.46%，仍有缓慢改善，不能据此证明收敛或严格过拟合。
目视检查第 10、20 轮的 8 张生成图片：存在大片黑白饱和区域、颜色异常和人脸结构失真，
未达到合理的人脸生成质量。按学习试跑的目的判断，先停止并排查生成质量是合理选择；
这不是预先设定 patience 的自动早停，也不声称后续训练一定无益。

Codex 向本次训练 PID 433 及其数据工作进程发送 SIGTERM。
复查训练进程已退出，GPU 使用率 0%、显存占用 0 MiB；`train.exit` 为 143（主动终止）。
未完成的第 26 轮进度没有保存。

最近完整检查点为第 25 轮：train_loss=0.014359115824391196，
val_loss=0.013948555299975851。原 `outputs/latest.pt` 保留，另备份为服务器
`/root/gpufree-data/ddpm/outputs/stopped_epoch0025.pt`，并下载到本地
[stopped_epoch0025.pt](outputs/stopped_epoch0025.pt)。
没有保存第 24 轮独立权重，不能恢复为验证损失最低的那一轮。

## 2026-10-31：使用第 25 轮权重生成

用户要求用停止时权重尝试生成。Codex 在本地 MPS 加载
`outputs/stopped_epoch0025.pt`，从检查点恢复图片尺寸与 betas，重算 alphas、alpha_bars，
用 `UNet(timesteps=1000)` 加载模型；设置 `torch.manual_seed(42)` 和
`torch.mps.manual_seed(42)`，调用现有 `generate(model, 8)`、`save_samples`。
未修改采样算法或权重。1000 步采样与保存耗时 12.58 秒。

结果：[8 张生成图片](outputs/generated_epoch0025_seed42.png)。原始输出均为有限值，
范围 [-1.726148, 1.143618]，约 25.54% 的像素通道值超出 [-1,1]，保存时按原代码裁剪。
目视观察仍有严重人脸结构失真、黑色区域及颜色异常；不能视为生成质量已通过。
这些结果不能单独确定问题源于模型容量、训练过程还是采样行为，需另行排查。

## 2026-10-31：继续训练 5 轮

用户要求再训练几轮。Codex 确认服务器 GPU 空闲、没有原训练进程，且
`outputs/latest.pt` 仍为第 25 轮后，提交 `run_continue.py` 从第 26 轮继续，
总目标设为 30 轮，到目标后自动结束。恢复模型与 Adam 状态，学习率和数据划分不变。
第 25 轮备份保留，不修改原 DDPM 公式或网络。

本次服务器启动入口：`/root/gpufree-data/ddpm/continue.sh`，设置
`OMP_NUM_THREADS=4` 并运行 `python -u run_continue.py`，数据工作进程数仍为 4。
后台 shell PID 14242；日志 `continue_26_30.log`，退出码写入 `continue.exit`。

每轮生成 8 张图，CUDA 采样种子固定为 42；使用 `torch.random.fork_rng`
避免采样随机数设置影响后续训练。不同设备的同种子不保证输出与本地 MPS 相同。
图片保存为 `outputs/samples_0026.png` 至 `samples_0030.png`，
并将对应轮最新完整检查点复制为同名 `.pt`，保留逐轮比较所需的权重。
`outputs/latest.pt` 仍每轮更新。

查看进度：

```bash
tail -f /root/gpufree-data/ddpm/continue_26_30.log
```

本记录只代表提交续训，完成状态和效果需之后核查日志与结果。

### 续训完成核查

2026-10-31，Codex 实时读取服务器日志：第 26–30 轮均完成，`continue.exit=0`，
训练进程已退出，GPU 使用率 0%、显存占用 0 MiB。各轮验证 MSE：

| 轮数 | train_loss | val_loss |
| --- | --- | --- |
| 26 | 0.014474 | 0.013865 |
| 27 | 0.014585 | 0.013871 |
| 28 | 0.014441 | 0.013853 |
| 29 | 0.014326 | 0.013889 |
| 30 | 0.014513 | 0.013845 |

服务器第 26–30 轮的 `.png` 和对应 `.pt` 文件均存在。
本次仅核查训练完成与文件存在，未目视检查新增图片，不据此断言生成质量改善。

## 2026-10-31：继续训练第 31–40 轮

用户要求再训练 10 轮。Codex 核实服务器无活动训练、GPU 空闲、最新检查点为
第 30 轮（val_loss=0.013845154844602803）且独立第 30 轮权重存在后，
基于原续训入口创建 `run_31_40.py`，仅把总目标改为 40 轮。
恢复模型和优化器，保持学习率、数据划分、网络、固定种子采样及逐轮权重保存行为不变。

服务器后台入口 `train_31_40.sh`，shell PID 14797；日志
`/root/gpufree-data/ddpm/train_31_40.log`，退出状态写入 `train_31_40.exit`。
目标完成后自动结束；本次提交不表示这 10 轮已经完成。

```bash
tail -f /root/gpufree-data/ddpm/train_31_40.log
```

### 第31–40轮完成核查（2026-10-31补录）

Codex 在本轮对话中实时读取服务器日志：第 40 轮完成，`train_31_40.exit=0`，
没有活动训练进程，GPU 使用率和显存占用均为 0。`latest.pt`、`samples_0040.pt`、
`samples_0040.png` 均存在。第 40 轮 train_loss=0.014407、val_loss=0.013756；
本段最低 val_loss 为第 39 轮的 0.013704。未目视检查这些新增图片。
完整摘录及当前实验结论统一在 [实践记录](README.md) 维护。
