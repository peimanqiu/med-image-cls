# 用数字图像处理的视角体检这批胸片
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.datasets import get_datasets, IMAGENET_MEAN, IMAGENET_STD

#  新增：解决中文乱码 
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False    # 解决负号"-"变成方框

train_ds, _, _ = get_datasets("pneumoniamnist", size=224)

# 取 200 张算统计（避开增广的随机性，直接读原始数据）
imgs = np.stack([train_ds[i][0].numpy() for i in range(200)])

print("=" * 56)
print("影像统计体检（200 张）")
print("=" * 56)
print(f"归一化后 均值 {imgs.mean():.4f}  标准差 {imgs.std():.4f}")
print(f"各通道均值 {[round(imgs[:, c].mean(), 4) for c in range(3)]}")
print("  -> 三个通道数值几乎相同，因为灰度图复制成了 3 通道")

# 单张灰度直方图
one = train_ds[0][0][0].numpy()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(one, cmap="gray"); axes[0].set_title("单张胸片（归一化后）"); axes[0].axis("off")
axes[1].hist(one.ravel(), bins=60, color="#1e6fd9", alpha=0.85)
axes[1].set_title("灰度直方图"); axes[1].set_xlabel("归一化像素值"); axes[1].set_ylabel("频数")
plt.tight_layout(); plt.savefig("experiments/debug/histogram.png", dpi=120)
print("\n✅ 直方图已保存 experiments/debug/histogram.png")