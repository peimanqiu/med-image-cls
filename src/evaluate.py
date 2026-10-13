# src/evaluate.py —— 加载 best.pt 在指定 split 上评估，输出四指标
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import get_datasets
from src.models import get_model
from src.metrics import compute_metrics
from src.utils import set_seed

CLASS_NAMES = {
    "pneumoniamnist": ["normal", "pneumonia"],
    "dermamnist": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    "chestmnist": ["atelectasis", "cardiomegaly", "effusion", "infiltration",
                   "mass", "nodule", "pneumonia", "pneumothorax", "consolidation",
                   "edema", "emphysema", "fibrosis", "pleural_thickening", "hernia"],
}


def infer_from_ckpt(sd):
    """从权重的 key 反推架构与类别数，不用手填 --model / --num_classes。"""
    keys = list(sd.keys())
    if any(k.startswith("features.denseblock") for k in keys):
        return "densenet121", sd.get("classifier.weight", torch.zeros(7)).shape[0]
    if any(k.startswith("features.0.0.") for k in keys):
        return "convnext_tiny", sd.get("classifier.2.weight", torch.zeros(7)).shape[0]
    if "conv1.weight" in sd:
        return "resnet18", sd.get("fc.weight", torch.zeros(7)).shape[0]
    return None, None


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    probs, trues = [], []
    for imgs, labels in loader:
        out = model(imgs.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        trues.append(labels.reshape(-1).long().numpy())
    return np.concatenate(probs), np.concatenate(trues)


def predict_auto(model, ds, device, bs):
    """显存不够就自动把 batch 减半重试，最低到 1。"""
    tried = []
    for b in [bs, bs // 2, bs // 4, 1]:
        b = max(1, b)
        if b in tried:
            continue
        tried.append(b)
        try:
            loader = DataLoader(ds, batch_size=b, shuffle=False, num_workers=0)
            yp, yt = _predict(model, loader, device)
            print(f"   [评估 batch = {b}]")
            return yp, yt
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"   显存不足(bs={b})，降低后重试")
                continue
            raise
    raise RuntimeError("batch 降到 1 仍然显存不足，请关闭其他占显存的程序")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dermamnist")
    p.add_argument("--model", default="auto")
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--ckpt", default="experiments/d7_convnext_tiny_ce/best.pt")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default="experiments/d7_convnext_tiny_ce/eval")
    args = p.parse_args()

    set_seed(42)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. 数据 ----
    if args.data not in CLASS_NAMES:
        raise ValueError(f"未知数据集 {args.data}，可选 {list(CLASS_NAMES)}")
    train_ds, val_ds, test_ds = get_datasets(args.data, size=224)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    print(f"数据 {args.data} / {args.split} 共 {len(ds)} 张")

    # ---- 2. 权重 ----
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"找不到权重文件：{args.ckpt}\n请先训练，或检查路径拼写")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]

    arch, nc = infer_from_ckpt(sd)
    print(f"权重识别：架构={arch}  类别数={nc}  来自 epoch {ckpt.get('epoch')}")
    if arch is None:
        raise RuntimeError("无法识别权重架构，请检查 best.pt 是否完整")
    args.model, args.num_classes = arch, nc

    # ---- 3. 模型 ----
    model = get_model(args.model, args.num_classes, pretrained=False).to(device)
    try:
        model.load_state_dict(sd)
        print(f"权重加载成功：{args.model}（{args.num_classes} 类）")
    except RuntimeError as e:
        print("\n架构不匹配。权重前 3 个 key:", list(sd.keys())[:3])
        print("模型前 3 个 key:", list(model.state_dict().keys())[:3])
        raise

    # ---- 4. 预测 ----
    y_prob, y_true = predict_auto(model, ds, device, args.bs)

    # ---- 5. 指标 ----
    res = compute_metrics(y_true, y_prob, class_names=CLASS_NAMES[args.data])
    res["_info"] = {"data": args.data, "split": args.split,
                    "model": args.model, "ckpt": args.ckpt}

    sens = res.get("sensitivity", res.get("sens_macro"))
    spec = res.get("specificity", res.get("spec_macro"))

    print(f"\n===== {args.data} / {args.split} / {args.model} =====")
    print(f"  AUC(macro)   {res['auc_macro']:.4f}")
    print(f"  F1(macro)    {res['f1_macro']:.4f}")
    print(f"  Sensitivity  {sens:.4f}")
    print(f"  Specificity  {spec:.4f}")
    print(f"  Accuracy     {res['accuracy']:.4f}   <- 不平衡数据会虚高，仅参考")

    if "per_class_sens" in res:
        names = CLASS_NAMES[args.data]
        print("\n  每类 Sensitivity：")
        for i, n in enumerate(names):
            print(f"    {n:6s} {res['per_class_sens'][str(i)]:.4f}")

    np.save(os.path.join(args.out, "y_prob.npy"), y_prob)
    np.save(os.path.join(args.out, "y_true.npy"), y_true)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n已保存到 {args.out}")


if __name__ == "__main__":
    main()