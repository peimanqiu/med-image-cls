import os, numpy as np
from medmnist import PneumoniaMNIST, INFO

# 定位到项目根目录的 data/raw/
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(ROOT, exist_ok=True)
FLAG = "pneumoniamnist"
info = INFO[FLAG]
print("=" * 60)
print(f"数据集 : {FLAG}")
print(f"任务类型: {info['task']}")
print(f"通道数  : {info['n_channels']}  (1=灰度, 3=彩色)")
print(f"样本数  : train {info['n_samples']['train']} / val {info['n_samples']['val']} / test {info['n_samples']['test']}")
print(f"类别    : {info['label']}")
print(f"许可    : {info['license']}")
print("=" * 60)

# 只需要下载一次，后面直接读本地
for split in ["train", "val", "test"]:
    # 重点：如果已经手动放好npz文件，改成 download=False
    ds = PneumoniaMNIST(root=ROOT, split=split, size=224, download=False, as_rgb=True)
    imgs, labels = ds.imgs, ds.labels
    print(f"{split:5s} | 图像 {imgs.shape} {imgs.dtype} | 标签 {labels.shape} "
          f"| 正样本比例 {labels.mean():.3f}")

print("\n✅ 数据已就绪，存放于:", ROOT)
