# 烟雾测试：9 组配置各跑 1 epoch × 300 张，确认无报错
import os, sys, subprocess
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

ok, bad = [], []
for model, loss, bs, freeze in CONFIGS:
    name = f"smoke_{model}_{loss}"
    cmd = [sys.executable, "src/train.py",
           "--data", "dermamnist", "--num_classes", "7",
           "--model", model, "--loss", loss,
           "--epochs", "1", "--bs", str(bs),
           "--max_samples", "300",
           "--out", f"experiments/{name}"]
    if freeze:
        cmd.append("--freeze_backbone")

    print(f"\n▶ {model} + {loss}  (bs={bs}, freeze={freeze})")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode == 0:
        print("   ✅ 通过")
        ok.append(name)
    else:
        out = (r.stdout + r.stderr).strip().splitlines()
        print("   ❌ 失败:")
        print("\n".join("      " + l for l in out[-5:]))
        bad.append(name)

print("\n" + "=" * 50)
print(f"通过 {len(ok)} / {len(CONFIGS)}")
if bad:
    print("失败:", bad)
else:
    print("全部通过，可以启动正式训练")