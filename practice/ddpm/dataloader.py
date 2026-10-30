"""读取已解压的 CelebA 图片；仅负责数据，不包含扩散算法。"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CelebADataset(Dataset):
    """每个样本是 float32 的 [3, H, W] 张量，取值在 [-1, 1]。"""

    def __init__(self, root, image_size=104):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"图片目录不存在：{self.root}\n"
                "请先解压 CelebA，并将 data_root 指向直接包含 JPG 的 img_align_celeba 目录。"
            )
        if image_size <= 0:
            raise ValueError("image_size 必须大于 0")
        self.image_size = image_size
        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise ValueError(f"目录中没有 JPG/PNG 图片：{self.root}（不递归搜索子目录）")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")

        # 保持宽高比，把短边缩放到 image_size，再中心裁剪。
        width, height = image.size
        size = self.image_size
        if width <= height:
            new_size = (size, int(height * size / width))
        else:
            new_size = (int(width * size / height), size)
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        left = round((image.width - size) / 2)
        top = round((image.height - size) / 2)
        image = image.crop((left, top, left + size, top + size))

        # HWC -> CHW；uint8 [0, 255] -> float32 [-1, 1]。
        pixels = np.array(image, dtype=np.float32)
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous() / 127.5 - 1.0


def make_dataloader(root, image_size=104, batch_size=8, num_workers=0):
    """返回仅含图片的训练 DataLoader；最后不足 batch_size 的批次也保留。"""
    dataset = CelebADataset(root, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_train_val_loaders(root, image_size=104, batch_size=8, num_workers=0,
                           val_size=1024, split_seed=42):
    """按固定种子留出验证图片；训练、验证图片互不重叠。"""
    dataset = CelebADataset(root, image_size=image_size)
    if not 0 < val_size < len(dataset):
        raise ValueError("val_size 必须大于 0 且小于图片总数")
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    options = dict(batch_size=batch_size, num_workers=num_workers,
                   pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **options)
    val_loader = DataLoader(val_set, shuffle=False,
                            generator=torch.Generator().manual_seed(split_seed), **options)
    return train_loader, val_loader
