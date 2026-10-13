# 多标签训练：sigmoid + BCEWithLogitsLoss，与单标签共用 datasets/models

# src/train_multilabel.py —— ChestMNIST 多标签训练
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import get_datasets
from src.models import get_model
from src.utils import set_seed, get_logger, save_ckpt, peak_mem, reset_mem

CHEST_CLASSES = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass",
                 "nodule", "pneumonia", "pneumothorax", "consolidation", "edema",
                 "emphysema", "fibrosis", "pleural_thickening", "hernia"]
NUM_CLS = 14


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, P, T = 0.0, [], []
    with torch.enable_grad() if train else torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)   # (B,14) 0/1
            if train:
                optimizer.zero_grad()
            out = model(imgs)                                        # logits, 不用 softmax
            loss = criterion(out, labels)
            if train:
                loss.backward(); optimizer.step()
            tot += loss.item() * imgs.size(0)
            P.append(torch.sigmoid(out.detach().float()).cpu().numpy())
            T.append(labels.cpu().numpy())
    yp, yt = np.concatenate(P), np.concatenate(T)
    return {"loss": tot / len(loader.dataset), "y_prob": yp, "y_true": yt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="chestmnist")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--subset", type=int, default=30000, help="抽取的训练子集大小")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/d8_ml_resnet18")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    logger = get_logger(os.path.join(args.out, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = get_datasets(args.data, size=224)
    logger.info(f"全量 train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    # 抽子集（报告里必须写明抽样方式，否则不可复现）
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(train_ds), size=min(args.subset, len(train_ds)), replace=False)
    train_sub = Subset(train_ds, idx)
    logger.info(f"抽取训练集子集 {len(train_sub)} 张（随机种子 {args.seed}）")

    tr_l = DataLoader(train_sub, batch_size=args.bs, shuffle=True,  num_workers=0)
    va_l = DataLoader(val_ds,    batch_size=args.bs, shuffle=False, num_workers=0)

    model = get_model(args.model, NUM_CLS, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()                       # ← 多标签专用
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        reset_mem(); t0 = time.time()
        tr = run_epoch(model, tr_l, criterion, device, optimizer)
        va = run_epoch(model, va_l, criterion, device)
        dt = time.time() - t0

        from sklearn.metrics import average_precision_score
        map_ = average_precision_score(va["y_true"], va["y_prob"], average="macro")
        logger.info(f"Epoch {ep}/{args.epochs} | {dt:.0f}s | 显存 {peak_mem():.2f}GB")
        logger.info(f"   train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} | mAP {map_:.4f}")

        hist.append({"epoch": ep, "train_loss": round(tr["loss"], 4),
                     "val_loss": round(va["loss"], 4), "mAP": round(float(map_), 4),
                     "sec": round(dt, 1), "peak_mem_gb": round(peak_mem(), 2)})
        if map_ > best:
            best = float(map_)
            save_ckpt(os.path.join(args.out, "best_ml.pt"), model, ep, best)
            logger.info(f"   -> 保存最优 (mAP {best:.4f})")

    json.dump({"args": vars(args), "history": hist, "best_mAP": best},
              open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    logger.info(f"✔ 完成，最优 mAP = {best:.4f}")


if __name__ == "__main__":
    main()