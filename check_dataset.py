import torch
from src.datasets import get_datasets, IMAGENET_MEAN, IMAGENET_STD

train_ds, val_ds, test_ds = get_datasets("pneumoniamnist", size=224)
print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

x, y = train_ds[0]
print(f"单样本 图像 {tuple(x.shape)} {x.dtype} | 标签 {y}")
print(f"像素范围 min={x.min():.3f} max={x.max():.3f}   <- 归一化后不再是 0~255")
print(f"各通道均值 {[round(x[c].mean().item(), 3) for c in range(3)]}  <- 接近 0 说明归一化生效")

# 关键验证：同一个样本，train 的增广应该每次都不一样，val 应该每次都一样
print("\n[增广随机性检查]")
a1, _ = train_ds[0]; a2, _ = train_ds[0]
print(f"  train 两次取同一张是否相同: {torch.allclose(a1, a2)}  <- 期望 False")
b1, _ = val_ds[0];   b2, _ = val_ds[0]
print(f"  val   两次取同一张是否相同: {torch.allclose(b1, b2)}  <- 期望 True")

# 反归一化回原图，确认没被搞坏
def denorm(t):
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return (t * 255).clamp(0, 255).byte()

back = denorm(x)
print(f"\n反归一化回 uint8: {tuple(back.shape)} 范围 {back.min()}~{back.max()}  <- 期望 0~255")