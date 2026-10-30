"""用户手写 DDPM 学习实现：训练、保存权重和生成图片。

数据加载和部分工程支持由 Codex 辅助完成；不等同于独立完成全部实现。
"""

from pathlib import Path

from dataloader import make_train_val_loaders

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

# 原始数据保留在 Wiki 的 sources/ddpm/data/ 中。
data_root = Path(__file__).resolve().parents[2] / "sources" / "ddpm" / "data" / "img_align_celeba"
image_size = 104
batch_size = 8
T = 1000
lr = 1e-4
num_epochs = 100
num_samples = 8
sample_every = 10
log_every = 100  # 每多少个批次打印一次；首批和末批也会打印。
max_grad_norm = 1.0
resume = True  # 存在 latest.pt 时继续训练；重新实验请设 False 并换 output_dir。
val_size = 1024  # 固定留出这些图片计算验证损失，不参与训练。
split_seed = 42
output_dir = Path(__file__).resolve().parent / "outputs"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

betas = torch.linspace(1e-4,0.02,T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time):
        h = self.conv1(x) + self.time_proj(time)[:, :, None, None]
        return self.conv2(h) + self.skip(x)


class UNet(nn.Module):
    """预测噪声：输入 x [B, 3, H, W] 和 t [B]，输出同形状张量。"""

    def __init__(self, base_ch=32, time_dim=128, timesteps=T):
        super().__init__()
        # t 使用 1,...,T，额外留出索引 0；通道数须为 8 的倍数。
        self.time_embed = nn.Sequential(nn.Embedding(timesteps + 1, time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.input_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        self.enc1 = ResBlock(base_ch, base_ch, time_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.mid = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.dec2 = ResBlock(base_ch * 6, base_ch * 2, time_dim)
        self.dec1 = ResBlock(base_ch * 3, base_ch, time_dim)
        self.output_conv = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, 3, 3, padding=1))

    def forward(self, x, t):
        time = self.time_embed(t)
        x1 = self.enc1(self.input_conv(x), time)  # 104 × 104
        x2 = self.enc2(F.avg_pool2d(x1, 2), time)  # 52 × 52
        h = self.mid(F.avg_pool2d(x2, 2), time)  # 26 × 26
        h = F.interpolate(h, size=x2.shape[-2:], mode="nearest")
        h = self.dec2(torch.cat([h, x2], dim=1), time)
        h = F.interpolate(h, size=x1.shape[-2:], mode="nearest")
        h = self.dec1(torch.cat([h, x1], dim=1), time)
        return self.output_conv(h)


def loss_func(epsilon_predict,epsilon):
    return torch.mean((epsilon - epsilon_predict) ** 2)

def train_ddpm(loader,device,model,optimiser):
    model.train()
    alpha_bars_device = alpha_bars.to(device)
    total_loss = 0.0
    num_images = 0

    for step, x0 in enumerate(loader, start=1):
        B = x0.shape[0]
        x0 = x0.to(device)
        t = torch.randint(1,T+1,(B,),device=x0.device)
        epsilon = torch.randn_like(x0)

        alpha_t_bar = alpha_bars_device[t-1].view(B,1,1,1)
        x_t = torch.sqrt(alpha_t_bar) * x0 + torch.sqrt(1-alpha_t_bar) * epsilon
        epsilon_predict = model(x_t,t)

        loss = loss_func(epsilon_predict,epsilon)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimiser.step()
        total_loss += loss.item() * B
        num_images += B
        if step == 1 or step % log_every == 0 or step == len(loader):
            print(f"  Batch {step}/{len(loader)} | loss={loss.item():.6f}", flush=True)

    return total_loss / num_images


@torch.no_grad()
def evaluate_ddpm(loader, device, model):
    """固定时间步与噪声，便于比较各轮验证 MSE；不更新参数。"""
    model.eval()
    generator = torch.Generator().manual_seed(1234)
    alpha_bars_device = alpha_bars.to(device)
    total_loss = 0.0
    num_images = 0
    for x0 in loader:
        B = x0.shape[0]
        # 使用独立 CPU 随机数生成器，兼容 CPU/CUDA/MPS。
        t = torch.randint(1, T + 1, (B,), generator=generator).to(device)
        epsilon = torch.randn(x0.shape, generator=generator).to(device)
        x0 = x0.to(device)
        alpha_t_bar = alpha_bars_device[t - 1].view(B, 1, 1, 1)
        x_t = alpha_t_bar.sqrt() * x0 + (1 - alpha_t_bar).sqrt() * epsilon
        loss = loss_func(model(x_t, t), epsilon)
        total_loss += loss.item() * B
        num_images += B
    return total_loss / num_images


@torch.no_grad()
def generate(model, num_samples):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(num_samples, 3, image_size, image_size, device=device)

    beta = betas.to(device)
    alpha = alphas.to(device)
    alpha_bar = alpha_bars.to(device)

    for t in range(T, 0, -1):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        epsilon_predict = model(x, t_batch)

        beta_t = beta[t - 1]
        alpha_t = alpha[t - 1]
        alpha_bar_t = alpha_bar[t - 1]
        alpha_bar_prev = alpha_bar[t - 2] if t > 1 else x.new_tensor(1.0)

        # 先估计干净图片；裁剪的是 x0 估计，而不是带噪状态 x_t。
        x0_predict = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_predict) / torch.sqrt(alpha_bar_t)
        x0_predict = x0_predict.clamp(-1, 1)

        # 用裁剪后的 x0 估计计算后验均值，不能再直接用原噪声简化式。
        # t=1 时后验均值就是 x0_predict，直接使用可避免系数舍入误差。
        mu_predict = x0_predict if t == 1 else (
            torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * x0_predict
            + torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
        )
        variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

        x = mu_predict + torch.sqrt(variance) * torch.randn_like(x) if t > 1 else mu_predict
    
    return x


def save_samples(samples, path):
    # 转回 [0, 255]，横向排列保存；不依赖 torchvision。
    images = ((samples.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    grid = torch.cat(list(images), dim=2).permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(path)


def main():
    loader, val_loader = make_train_val_loaders(
        data_root, image_size=image_size, batch_size=batch_size,
        val_size=val_size, split_seed=split_seed,
    )
    data_split = {"seed": split_seed, "val_size": val_size,
                  "total_size": len(loader.dataset) + len(val_loader.dataset)}
    x0 = next(iter(loader))
    print(f"训练图片：{len(loader.dataset)}，验证图片：{len(val_loader.dataset)}")
    print(f"首批形状：{tuple(x0.shape)}，类型：{x0.dtype}")
    print(f"像素范围：[{x0.min().item():.3f}, {x0.max().item():.3f}]")
    print(f"训练设备：{device}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = UNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_path = output_dir / "latest.pt"
    start_epoch = 1
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("data_split") != data_split:
            raise ValueError("检查点的数据划分不同或未记录划分（旧版使用了全部图片训练）。请换 output_dir 开始新实验，避免验证集泄漏。")
        if checkpoint["image_size"] != image_size or not torch.equal(checkpoint["betas"], betas):
            raise ValueError("检查点的图片尺寸或噪声调度与当前配置不同，请恢复原配置或更换 output_dir。")
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"已恢复第 {start_epoch - 1} 轮检查点；优化器沿用保存时的学习率。", flush=True)
    if start_epoch > num_epochs:
        print("已达到 num_epochs，无需继续训练；增加 num_epochs 可继续。")
        return

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}", flush=True)
        loss = train_ddpm(loader, device, model, optimiser)
        val_loss = evaluate_ddpm(val_loader, device, model)
        print(f"Epoch {epoch}/{num_epochs} | train_loss={loss:.6f} | val_loss={val_loss:.6f}", flush=True)
        # 先写临时文件，再替换，避免写入中断损坏上一份检查点。
        temporary_path = output_dir / "latest.tmp"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimiser": optimiser.state_dict(), "betas": betas, "image_size": image_size,
                    "data_split": data_split, "train_loss": loss, "val_loss": val_loss}, temporary_path)
        temporary_path.replace(checkpoint_path)

        if epoch % sample_every == 0 or epoch == num_epochs:
            samples = generate(model, num_samples)
            save_samples(samples, output_dir / f"samples_{epoch:04d}.png")



if __name__ == "__main__":
    main()
