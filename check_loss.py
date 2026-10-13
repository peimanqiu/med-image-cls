# 30 秒验证：ce 到底有没有被偷偷加权
import numpy as np, torch, sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.losses import build_criterion, compute_class_weights

y = np.array([0]*300 + [1]*150 + [2]*100 + [3]*50 + [4]*20 + [5]*10 + [6]*5)
w = compute_class_weights(y, 7)
print("类别权重:", [round(float(x), 3) for x in w])
print()
ok = True
for n in ["ce", "weighted_ce", "focal"]:
    c = build_criterion(n, class_weight=w, device="cpu")
    wt = getattr(c, "weight", None)
    tag = "None" if wt is None else [round(float(x), 2) for x in wt]
    print(f"{n:12s} -> {type(c).__name__:18s} weight = {tag}")
    if n == "ce" and wt is not None:
        ok = False

print()
print("✅ CE 未加权，bug 已修复" if ok else "❌ CE 仍在加权，检查 losses.py 是否保存")

# 额外确认：三个损失算出来的数值确实不同
torch.manual_seed(0)
logits = torch.randn(8, 7); target = torch.randint(0, 7, (8,))
for n in ["ce", "weighted_ce", "focal"]:
    c = build_criterion(n, class_weight=w, device="cpu")
    print(f"  {n:12s} loss = {c(logits, target).item():.4f}")