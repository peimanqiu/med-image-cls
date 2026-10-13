# make_roc_binary.py —— 画二分类（PneumoniaMNIST）的 ROC 与混淆矩阵
import os, sys, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.visualize import plot_roc, plot_cm

OUT = "experiments/day5_eval"
NAMES = ["normal", "pneumonia"]

os.makedirs(OUT, exist_ok=True)

# 1 若还没有预测结果，先跑一遍评估
if not (os.path.exists(f"{OUT}/y_prob.npy") and os.path.exists(f"{OUT}/y_true.npy")):
    print("未找到预测结果，先跑评估 ...")
    from src.evaluate import main as eval_main
    sys.argv = ["evaluate.py",
                "--ckpt", "experiments/day5_resnet18_ce/best.pt",
                "--data", "pneumoniamnist",
                "--num_classes", "2",
                "--out", OUT]
    eval_main()

# 2 读结果
y_prob = np.load(f"{OUT}/y_prob.npy")
y_true = np.load(f"{OUT}/y_true.npy")
m = json.load(open(f"{OUT}/metrics.json", encoding="utf-8"))

# 3 出图
plot_roc(y_true, y_prob, f"{OUT}/roc.png", NAMES,
         "ROC - PneumoniaMNIST (ResNet18 + CE)")
plot_cm(m["confusion_matrix"], f"{OUT}/cm.png", NAMES,
        "Confusion Matrix - PneumoniaMNIST (normalized)")

print("\n完成，图在", OUT)