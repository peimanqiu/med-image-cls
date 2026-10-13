# check_multilabel_eval.py —— 多标签评估：mAP、每病 AUC、14 条 ROC、阈值对比
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # 让 src 可导入

import json, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, f1_score)

from src.datasets import get_datasets
from src.models import get_model

# 14 种胸腔疾病（ChestMNIST 官方顺序）
CHEST_CLASSES = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass",
    "nodule", "pneumonia", "pneumothorax", "consolidation", "edema",
    "emphysema", "fibrosis", "pleural_thickening", "hernia",
]
NUM_CLS = 14
OUT = "experiments/d8_ml_resnet18"


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{OUT}/best_ml.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 {ckpt_path}，请先完成多标签训练")

    # ---- 1. 数据 ----
    _, _, test_ds = get_datasets("chestmnist", size=224)
    print(f"测试集 {len(test_ds)} 张")

    # ---- 2. 模型 ----
    model = get_model("resnet18", NUM_CLS, pretrained=False).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print("权重加载成功：resnet18（14 类多标签）")

    # ---- 3. 预测（显存不够自动降 batch）----
    def predict(bs):
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
        P, T = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                out = model(imgs.to(device, non_blocking=True))
                P.append(torch.sigmoid(out.float()).cpu().numpy())   # sigmoid，不是 softmax
                T.append(labels.numpy())
        return np.concatenate(P), np.concatenate(T)

    y_prob = y_true = None
    for bs in [8, 4, 2, 1]:
        try:
            y_prob, y_true = predict(bs)
            print(f"[评估 batch = {bs}]")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  显存不足(bs={bs})，降低重试")
                continue
            raise

    np.save(f"{OUT}/ml_y_prob.npy", y_prob)
    np.save(f"{OUT}/ml_y_true.npy", y_true)

    # ---- 4. 指标 ----
    res = {
        "mAP_macro": float(average_precision_score(y_true, y_prob, average="macro")),
        "auc_macro": float(roc_auc_score(y_true, y_prob, average="macro")),
        "auc_micro": float(roc_auc_score(y_true, y_prob, average="micro")),
        "per_class_auc": {},
        "per_class_thr": {},
        "positive_rate": {},
    }

    for i, c in enumerate(CHEST_CLASSES):
        res["positive_rate"][c] = round(float(y_true[:, i].mean()), 4)
        try:
            res["per_class_auc"][c] = round(float(roc_auc_score(y_true[:, i], y_prob[:, i])), 4)
        except ValueError:
            res["per_class_auc"][c] = None
        # 搜索该类 F1 最优阈值
        best_t, best_f = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            f = f1_score(y_true[:, i], (y_prob[:, i] > t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        res["per_class_thr"][c] = {"thr": round(best_t, 2), "f1": round(float(best_f), 4)}

    f1_05 = f1_score(y_true, (y_prob > 0.5).astype(int), average="macro", zero_division=0)
    thrs = np.array([res["per_class_thr"][c]["thr"] for c in CHEST_CLASSES])
    f1_best = f1_score(y_true, (y_prob > thrs).astype(int), average="macro", zero_division=0)
    res["f1_at_0.5"] = round(float(f1_05), 4)
    res["f1_at_best_thr"] = round(float(f1_best), 4)

    json.dump(res, open(f"{OUT}/ml_metrics.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 5. 打印 ----
    print(f"\n===== ChestMNIST 多标签 / test =====")
    print(f"  mAP(macro)   {res['mAP_macro']:.4f}")
    print(f"  AUC(macro)   {res['auc_macro']:.4f}")
    print(f"  AUC(micro)   {res['auc_micro']:.4f}")
    print(f"  F1 @0.5      {res['f1_at_0.5']:.4f}")
    print(f"  F1 @最优阈值 {res['f1_at_best_thr']:.4f}")

    print("\n  每病 AUC（按降序）:")
    items = sorted(res["per_class_auc"].items(), key=lambda x: (x[1] is None, -(x[1] or 0)))
    for c, a in items:
        pr = res["positive_rate"][c]
        print(f"    {c:18s} AUC {str(a):6s}  阳性率 {pr:.3f}  最优阈值 {res['per_class_thr'][c]['thr']}")

    # ---- 6. 14 条 ROC ----
    plt.figure(figsize=(9, 8))
    for i, c in enumerate(CHEST_CLASSES):
        a = res["per_class_auc"][c]
        if a is None:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, lw=1.2, alpha=0.8, label=f"{c} ({a:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC per disease - ChestMNIST (ResNet18, multilabel)")
    plt.legend(fontsize=7, ncol=2, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/roc_multilabel.png", dpi=140)
    plt.close()
    print(f"\n已保存 {OUT}/roc_multilabel.png")

    # ---- 7. 每病 AUC 柱状图 ----
    names = [c for c, _ in items]
    vals = [a or 0 for _, a in items]
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals, color="#1e6fd9", alpha=0.85)
    plt.axhline(0.5, color="red", ls="--", lw=1, label="random (0.5)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("AUC"); plt.ylim(0, 1)
    plt.title("Per-disease AUC - ChestMNIST multilabel")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/auc_per_disease.png", dpi=140)
    plt.close()
    print(f"已保存 {OUT}/auc_per_disease.png")


if __name__ == "__main__":
    main()