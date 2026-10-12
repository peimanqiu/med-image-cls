# src/visualize.py —— ROC 曲线与归一化混淆矩阵
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.metrics import get_roc_data

CLASS_NAMES = {
    "pneumoniamnist": ["normal", "pneumonia"],
    "dermamnist": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}


def plot_roc(y_true, y_prob, save_path, class_names=None, title="ROC"):
    data = get_roc_data(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    fpr, tpr, a = data["macro"]
    plt.plot(fpr, tpr, lw=2.2, color="#1e6fd9", label=f"macro (AUC={a:.4f})")
    if len(data) > 1 and class_names:
        for k, (f, t, ai) in data.items():
            if k == "macro":
                continue
            plt.plot(f, t, lw=1.0, alpha=0.65,
                     label=f"{class_names[int(k)]} (AUC={ai:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    plt.xlabel("False Positive Rate  (1 - Specificity)")
    plt.ylabel("True Positive Rate  (Sensitivity)")
    plt.title(title); plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=130); plt.close()
    print("✔", save_path)


def plot_cm(cm, save_path, class_names, title="Confusion Matrix (normalized)"):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)   # 按行归一化 = 每类的召回
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, cbar_kws={"label": "ratio"})
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title); plt.tight_layout()
    plt.savefig(save_path, dpi=130); plt.close()
    print("✔", save_path)

if __name__ == "__main__":
    OUT = "experiments/eval"
    names = CLASS_NAMES["pneumoniamnist"]

    y_prob = np.load(f"{OUT}/y_prob.npy")
    y_true = np.load(f"{OUT}/y_true.npy")

    plot_roc(y_true, y_prob, f"{OUT}/roc.png", names,
             "ROC - PneumoniaMNIST (ResNet18+CE)")

    import json
    cm = json.load(open(f"{OUT}/metrics.json", encoding="utf-8"))["confusion_matrix"]
    plot_cm(cm, f"{OUT}/cm.png", names, "Confusion Matrix (normalized)")

    print("全部完成，可在experiments/day5_eval 文件夹查看 roc.png 和 cm.png")