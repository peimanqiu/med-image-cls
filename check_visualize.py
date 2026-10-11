# check_visualize.py —— 医学影像长什么样
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.datasets import get_datasets, IMAGENET_MEAN, IMAGENET_STD

#  新增：解决中文乱码 
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False    # 解决负号"-"变成方框

SAVE_DIR = "experiments/debug"
os.makedirs(SAVE_DIR, exist_ok=True)

train_ds, _, _ = get_datasets("pneumoniamnist", size=224)

def denorm(t):
    """把归一化后的 tensor 还原成能显示的 uint8 图像"""
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

# 各取 4 张正常 / 4 张肺炎
idxs = {"normal": [], "pneumonia": []}
for i in range(len(train_ds)):
    _, y = train_ds[i]
    key = "pneumonia" if int(y[0]) == 1 else "normal"
    if len(idxs[key]) < 4:
        idxs[key].append(i)
    if all(len(v) == 4 for v in idxs.values()):
        break

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for row, key in enumerate(["normal", "pneumonia"]):
    for col, i in enumerate(idxs[key]):
        img, y = train_ds[i]
        axes[row][col].imshow(denorm(img), cmap="gray")
        axes[row][col].set_title(f"{key} (label={int(y[0])})", fontsize=11)
        axes[row][col].axis("off")

plt.suptitle("PneumoniaMNIST 224x224 —— 上排：正常，下排：肺炎", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "sample_grid.png"), dpi=120, bbox_inches="tight")
print("✅ 已保存:", os.path.join(SAVE_DIR, "sample_grid.png"))