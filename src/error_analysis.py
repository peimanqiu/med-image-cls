# 按置信度排序导出错判样本图，统计混淆对并归因
# src/error_analysis.py —— 错判样本导出与错误归因
import os, sys, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.datasets import get_datasets, IMAGENET_MEAN, IMAGENET_STD
from src.models import get_model

CLASS_NAMES = {
    "pneumoniamnist": ["normal", "pneumonia"],
    "dermamnist": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}


def infer_from_ckpt(sd):
    """从权重反推架构与类别数"""
    keys = list(sd.keys())
    if any(k.startswith("features.denseblock") for k in keys):
        return "densenet121", sd.get("classifier.weight", torch.zeros(7)).shape[0]
    if any(k.startswith("features.0.0.") for k in keys):
        return "convnext_tiny", sd.get("classifier.2.weight", torch.zeros(7)).shape[0]
    if "conv1.weight" in sd:
        return "resnet18", sd.get("fc.weight", torch.zeros(7)).shape[0]
    return None, None


def denorm(t):
    """归一化后的 tensor 还原成可显示的 uint8 图像"""
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def predict(model, ds, device, bs):
    """显存不够自动降 batch"""
    for b in [bs, bs // 2, bs // 4, 1]:
        b = max(1, b)
        try:
            loader = DataLoader(ds, batch_size=b, shuffle=False, num_workers=0)
            model.eval()
            P = []
            with torch.no_grad():
                for imgs, _ in loader:
                    out = model(imgs.to(device, non_blocking=True))
                    P.append(torch.softmax(out.float(), dim=1).cpu().numpy())
            print(f"   [推理 batch = {b}]")
            return np.concatenate(P)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"   显存不足(bs={b})，降低重试")
                continue
            raise
    raise RuntimeError("batch 降到 1 仍显存不足")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dermamnist")
    p.add_argument("--ckpt", default="experiments/d7_resnet18_ce/best.pt")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--topk", type=int, default=20, help="导出多少张错例")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = CLASS_NAMES[args.data]
    out_dir = os.path.join(os.path.dirname(args.ckpt), "errors")
    single_dir = os.path.join(out_dir, "single")
    os.makedirs(single_dir, exist_ok=True)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"找不到权重：{args.ckpt}")

    # ---- 1. 数据 ----
    _, _, test_ds = get_datasets(args.data, size=224)
    print(f"测试集 {len(test_ds)} 张")

    # ---- 2. 模型 ----
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)["state_dict"]
    arch, nc = infer_from_ckpt(sd)
    print(f"权重识别：{arch}（{nc} 类）")
    model = get_model(arch, nc, pretrained=False).to(device)
    model.load_state_dict(sd)

    # ---- 3. 预测 ----
    y_prob = predict(model, test_ds, device, args.bs)
    y_pred = y_prob.argmax(1)
    y_true = np.array([test_ds[i][1] for i in range(len(test_ds))]).reshape(-1)

    # ---- 4. 挑错：置信度最高的错判 ----
    wrong = np.where(y_pred != y_true)[0]
    conf = y_prob.max(1)
    order = wrong[np.argsort(-conf[wrong])]          # 越自信越靠前
    print(f"\n错判样本 {len(wrong)} / {len(y_true)} = {len(wrong)/len(y_true)*100:.2f}%")

    # ---- 5. 导出单张错例图 ----
    print(f"导出前 {min(args.topk, len(order))} 张到 {single_dir}")
    for rank, idx in enumerate(order[:args.topk], 1):
        img, _ = test_ds[idx]
        fig, ax = plt.subplots(figsize=(3, 3.4))
        ax.imshow(denorm(img))
        ax.set_title(f"#{rank} conf={conf[idx]:.2f}\n"
                     f"True: {names[y_true[idx]]}\nPred: {names[y_pred[idx]]}",
                     fontsize=9, color="darkred")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(single_dir,
                    f"{rank:02d}_true-{names[y_true[idx]]}_pred-{names[y_pred[idx]]}.png"),
                    dpi=110)
        plt.close()

    # ---- 6. 错例网格（2 行 x 5 列 x 若干屏）----
    grid_n = min(20, len(order))
    cols, rows = 5, (grid_n + 4) // 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.1 * rows))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        if k < grid_n:
            idx = order[k]
            img, _ = test_ds[idx]
            axes[r][c].imshow(denorm(img))
            axes[r][c].set_title(f"T:{names[y_true[idx]]} P:{names[y_pred[idx]]}\n"
                                 f"c={conf[idx]:.2f}", fontsize=8, color="darkred")
        axes[r][c].axis("off")
    plt.suptitle(f"Top-{grid_n} most confident errors ({args.data})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_grid.png"), dpi=120)
    plt.close()
    print(f"已保存 {out_dir}/error_grid.png")

    # ---- 7. 统计：每类错误数 + 主要混淆方向 ----
    nc_ = len(names)
    cm = np.zeros((nc_, nc_), dtype=int)
    for t, pr in zip(y_true, y_pred):
        cm[t][pr] += 1
    cls_total = cm.sum(1)
    cls_err = cls_total - np.diag(cm)

    print("\n每类错误情况：")
    stat = {}
    for i, n in enumerate(names):
        if cls_total[i] == 0:
            continue
        # 该类最常被误判成哪个类
        row = cm[i].copy(); row[i] = -1
        top_j = int(row.argmax())
        stat[n] = {
            "样本数": int(cls_total[i]),
            "错误数": int(cls_err[i]),
            "错误率": round(float(cls_err[i] / cls_total[i]), 4),
            "主要误判为": names[top_j],
            "误判数": int(cm[i][top_j]),
        }
        print(f"  {n:6s} 共 {cls_total[i]:4d} 张 | 错 {cls_err[i]:4d} 张 "
              f"({cls_err[i]/cls_total[i]*100:5.1f}%) | 主要误判为 {names[top_j]:6s} ({cm[i][top_j]} 张)")

    # ---- 8. Top 混淆对 ----
    pairs = []
    for i in range(nc_):
        for j in range(nc_):
            if i != j and cm[i][j] > 0:
                pairs.append((names[i], names[j], int(cm[i][j])))
    pairs.sort(key=lambda x: -x[2])
    print("\nTop 混淆对（真实 -> 预测）：")
    for a, b, c in pairs[:8]:
        print(f"  {a:6s} -> {b:6s}  {c} 张")

    # ---- 9. 每类错误数柱状图 ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, [stat.get(n, {}).get("错误率", 0) for n in names],
           color="#c0392b", alpha=0.85)
    ax.set_ylabel("Error rate"); ax.set_ylim(0, 1)
    ax.set_title(f"Error rate per class ({args.data})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_by_class.png"), dpi=130)
    plt.close()
    print(f"已保存 {out_dir}/error_by_class.png")

    # ---- 10. 生成 error_summary.md（可直接粘进报告）----
    md = [f"# 错误分析摘要（{args.data}）\n",
          f"- 测试集样本数：{len(y_true)}",
          f"- 错判样本数：{len(wrong)}（错误率 {len(wrong)/len(y_true)*100:.2f}%）\n",
          "## 每类错误统计\n",
          "| 类别 | 样本数 | 错误数 | 错误率 | 主要误判为 | 误判数 |",
          "|---|---|---|---|---|---|"]
    for n in names:
        if n not in stat:
            continue
        s = stat[n]
        md.append(f"| {n} | {s['样本数']} | {s['错误数']} | {s['错误率']:.4f} | "
                  f"{s['主要误判为']} | {s['误判数']} |")
    md += ["\n## Top 混淆对\n", "| 真实类别 | 预测类别 | 数量 |", "|---|---|---|"]
    for a, b, c in pairs[:8]:
        md.append(f"| {a} | {b} | {c} |")

    md += ["\n## 归因分析\n",
           "| 归因类别 | 典型表现 | 对应改进动作 |",
           "|---|---|---|",
           "| 长尾类别 | 训练样本极少（如 df、vasc 占比约 1%），被大量误判入多数类 nv | Weighted CE / Focal Loss / 过采样 |",
           "| 类间视觉相似 | bkl 与 nv 等良性色素病变特征接近，肉眼亦难区分 | 难样本挖掘、更细粒度的增广 |",
           "| 图像质量 | 模糊、过曝、毛发遮挡、存在伪影 | 预处理（去噪、直方图均衡） |",
           "| 标注噪声 | 标签本身存在医师主观差异 | 标签平滑、噪声鲁棒损失 |\n",
           "## 改进建议\n",
           "1. 对长尾类别采用加权损失或过采样；",
           "2. 降低决策阈值以提升 Sensitivity（多标签实验已验证可提升 F1 约 4.4 倍）；",
           "3. 增加 epoch 并解冻主干（当前 ConvNeXt 受显存限制冻结了主干）；",
           "4. 对高相似类别引入难样本挖掘。"]

    with open(os.path.join(out_dir, "error_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n已保存 {out_dir}/error_summary.md（可直接粘进报告第 5 节）")


if __name__ == "__main__":
    main()