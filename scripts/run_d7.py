# 正式跑 9 组（3 模型 × 3 损失），挂后台
import os, sys, subprocess, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = [
    ("resnet18",      "ce",          8, False),
    ("resnet18",      "weighted_ce", 8, False),
    ("resnet18",      "focal",       8, False),
    ("densenet121",   "ce",          4, False),
    ("densenet121",   "weighted_ce", 4, False),
    ("densenet121",   "focal",       4, False),
    ("convnext_tiny", "ce",          2, True),
    ("convnext_tiny", "weighted_ce", 2, True),
    ("convnext_tiny", "focal",       2, True),
]

t_all = time.time()
for i, (model, loss, bs, freeze) in enumerate(CONFIGS, 1):
    out = f"experiments/d7_{model}_{loss}"
    cmd = [sys.executable, "src/train.py",
           "--data", "dermamnist", "--num_classes", "7",
           "--model", model, "--loss", loss,
           "--epochs", "3", "--bs", str(bs),
           "--out", out]
    if freeze:
        cmd.append("--freeze_backbone")

    print(f"\n[{i}/{len(CONFIGS)}] {model} + {loss}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    dt = time.time() - t0

    if r.returncode == 0 and os.path.exists(f"{out}/best.pt"):
        print(f"   ✅ 完成 {dt/60:.1f} 分钟")
    else:
        out_txt = (r.stdout + r.stderr).strip().splitlines()
        print(f"   ❌ 失败")
        print("\n".join("      " + l for l in out_txt[-5:]))
        # 显存不足自动降 batch 重试一次
        if any("out of memory" in l.lower() for l in out_txt):
            cmd[cmd.index(str(bs))] = str(max(1, bs // 2))
            print(f"   显存不足，降到 bs={max(1,bs//2)} 重试")
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            print("   ✅ 重试成功" if r.returncode == 0 else "   ❌ 重试仍失败")

print(f"\n全部结束，总耗时 {(time.time()-t_all)/60:.1f} 分钟")
print("下一步：python scripts/eval_all.py")