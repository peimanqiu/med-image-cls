# make_roc_7cls.py —— 画七分类 ROC 与混淆矩阵
import os, sys, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.visualize import plot_roc, plot_cm

OUT = "experiments/d7_resnet18_ce/eval"
NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

os.makedirs(OUT, exist_ok=True)

# 1 先出评估指标与概率（若已有则直接读）
if not (os.path.exists(f"{OUT}/y_prob.npy") and os.path.exists(f"{OUT}/y_true.npy")):
    print("未找到预测结果，先跑评估 ...")
    from src.evaluate import main as eval_main
    sys.argv = ["evaluate.py",
                "--ckpt", "experiments/d7_resnet18_ce/best.pt",
                "--out", OUT]
    eval_main()

y_prob = np.load(f"{OUT}/y_prob.npy")
y_true = np.load(f"{OUT}/y_true.npy")
m = json.load(open(f"{OUT}/metrics.json", encoding="utf-8"))

# 2 出图
plot_roc(y_true, y_prob, f"{OUT}/roc.png", NAMES,
         "ROC - DermaMNIST 7-class (ResNet18 + CE)")
plot_cm(m["confusion_matrix"], f"{OUT}/cm.png", NAMES,
        "Confusion Matrix - DermaMNIST (normalized)")

print("\n完成，图在", OUT)