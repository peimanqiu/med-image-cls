# 加载 best.pt 在 test split 评估，指标写入 metrics.json（不用 val 冒充）
# src/evaluate.py —— 加载 best.pt 在 test split 评估
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
}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, trues = [], []
    for imgs, labels in loader:
        out = model(imgs.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        trues.append(labels.squeeze().long().numpy())
    return np.concatenate(probs), np.concatenate(trues)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pneumoniamnist")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--ckpt", default="experiments/day5_resnet18_ce/best.pt", help="best.pt 路径")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="experiments/eval")
    args = p.parse_args()

    set_seed(42)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ds = get_datasets(args.data, size=224)
    loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    model = get_model(args.model, args.num_classes, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    y_prob, y_true = predict(model, loader, device)
    names = CLASS_NAMES.get(args.data)
    res = compute_metrics(y_true, y_prob, class_names=names)

    print(f"\n===== {args.data} / {args.split} split =====")
    print(f"AUC(macro)   {res['auc_macro']:.4f}")
    print(f"F1(macro)    {res['f1_macro']:.4f}")
    print(f"Sensitivity  {res.get('sensitivity', res['sens_macro']):.4f}")
    print(f"Specificity  {res.get('specificity', res['spec_macro']):.4f}")
    print(f"Accuracy     {res['accuracy']:.4f}   <- 仅参考，不平衡数据会虚高")

    np.save(os.path.join(args.out, "y_prob.npy"), y_prob)
    np.save(os.path.join(args.out, "y_true.npy"), y_true)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"✔ 已保存 {args.out}/metrics.json 与 y_prob.npy")


if __name__ == "__main__":
    main()