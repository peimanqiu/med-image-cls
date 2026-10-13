# 批量评估 9 组，自动识别架构、自动降 batch
import os, sys, subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JOBS = [
    ("d7_resnet18_ce",          8),
    ("d7_resnet18_weighted_ce", 8),
    ("d7_resnet18_focal",       8),
    ("d7_densenet121_ce",          4),
    ("d7_densenet121_weighted_ce", 4),
    ("d7_densenet121_focal",       4),
    ("d7_convnext_tiny_ce",          2),
    ("d7_convnext_tiny_weighted_ce", 2),
    ("d7_convnext_tiny_focal",       2),
]

for name, bs in JOBS:
    ckpt = f"experiments/{name}/best.pt"
    if not os.path.exists(ckpt):
        print(f"⏭  跳过 {name}（无 best.pt）")
        continue
    cmd = [sys.executable, "src/evaluate.py",
           "--data", "dermamnist",
           "--ckpt", ckpt,
           "--bs", str(bs),
           "--out", f"experiments/{name}/eval"]
    print(f"\n▶ {name}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    txt = r.stdout + r.stderr
    if r.returncode == 0:
        for l in txt.splitlines():
            if any(k in l for k in ["AUC(macro)", "F1(macro)", "Sensitivity", "Specificity", "Accuracy"]):
                print("   " + l.strip())
    else:
        print("   ❌ 失败:")
        print("\n".join("      " + l for l in txt.strip().splitlines()[-5:]))

print("\n下一步：python scripts/collect_results.py")