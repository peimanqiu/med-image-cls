# src/train.py —— 单标签训练主程序
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import get_datasets
from src.models import get_model
from src.losses import build_criterion, compute_class_weights
from src.utils import set_seed, get_logger, save_ckpt, peak_mem, reset_mem


def run_one_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    """optimizer 为 None 时是验证模式"""
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, all_prob, all_true = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, labels in loader:                                   # ① 取数据
            imgs = imgs.to(device, non_blocking=True)
            # reshape(-1) 而非 squeeze()：batch=1 时 squeeze 会把标签压没
            labels = labels.reshape(-1).long().to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()                                 # ② 清梯度

            with torch.amp.autocast("cuda", enabled=amp):              # ③ 前向
                out = model(imgs)
                loss = criterion(out, labels)

            if train:                                                 # ④ 反向 ⑤ 更新
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)                  # ⑥ 记账
            all_prob.append(torch.softmax(out.detach().float(), dim=1).cpu().numpy())
            all_true.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_prob)
    y_true = np.concatenate(all_true)
    n = len(loader.dataset)

    res = {"loss": total_loss / n,
           "acc": float(accuracy_score(y_true, y_prob.argmax(1)))}
    try:
        res["auc"] = float(roc_auc_score(y_true, y_prob[:, 1])) if y_prob.shape[1] == 2 \
                     else float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        res["auc"] = float("nan")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dermamnist")
    p.add_argument("--model", default="resnet18",
                   choices=["resnet18", "densenet121", "convnext_tiny"])
    p.add_argument("--loss", default="ce", choices=["ce", "weighted_ce", "focal"])
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true", help="冻结主干只训分类头")
    p.add_argument("--max_samples", type=int, default=0, help=">0 时只取前 N 张（烟雾测试用）")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default="experiments/tmp")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    logger = get_logger(os.path.join(args.out, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = get_datasets(args.data, size=224)
    if args.max_samples > 0:                       # 烟雾测试：只取一小撮
        train_ds = Subset(train_ds, range(min(args.max_samples, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(args.max_samples // 4, len(val_ds))))
        logger.info(f"⚠️ 烟雾模式：只用 {len(train_ds)} 张训练 / {len(val_ds)} 张验证")

    logger.info(f"设备 {device} | {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
    logger.info(f"配置 model={args.model} loss={args.loss} bs={args.bs} lr={args.lr} "
                f"epochs={args.epochs} freeze={args.freeze_backbone} seed={args.seed}")
    logger.info(f"数据 train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = get_model(args.model, args.num_classes, pretrained=True).to(device)

    # 冻结主干（ConvNeXt 在 4GB 卡上的保命选项）
    if args.freeze_backbone:
        for n, prm in model.named_parameters():
            if not n.startswith(("fc", "classifier")):
                prm.requires_grad = False
        logger.info("已冻结主干，只训练分类头")

    # 损失（ce 分支不会用到 cw）
    labels_all = train_ds.labels if hasattr(train_ds, "labels") else \
                 np.array([train_ds[i][1] for i in range(len(train_ds))])
    cw = compute_class_weights(labels_all, args.num_classes)
    logger.info(f"类别权重 {[round(float(x), 3) for x in cw]}")
    logger.info(f"损失函数 {args.loss} -> {type(build_criterion(args.loss, cw, device=device)).__name__}")
    criterion = build_criterion(args.loss, class_weight=cw, device=device)

    optimizer = torch.optim.AdamW(
        [prm for prm in model.parameters() if prm.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_auc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        reset_mem()
        t0 = time.time()
        tr = run_one_epoch(model, train_loader, criterion, device, optimizer, scaler, args.amp)
        va = run_one_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0

        logger.info(f"Epoch {epoch}/{args.epochs} | {dt:.0f}s | 显存峰值 {peak_mem():.2f}GB")
        logger.info(f"   train loss {tr['loss']:.4f} acc {tr['acc']:.4f} auc {tr['auc']:.4f}")
        logger.info(f"   val   loss {va['loss']:.4f} acc {va['acc']:.4f} auc {va['auc']:.4f}")

        history.append({"epoch": epoch, "train": tr, "val": va,
                        "sec": round(dt, 1), "peak_mem_gb": round(peak_mem(), 2)})

        if va["auc"] > best_auc:
            best_auc = va["auc"]
            save_ckpt(os.path.join(args.out, "best.pt"), model, epoch, best_auc)
            logger.info(f"   -> 保存最优 (val AUC {best_auc:.4f})")

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history,
                   "best_val_auc": best_auc}, f, indent=2, ensure_ascii=False)
    logger.info(f"完成，最优 val AUC = {best_auc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()