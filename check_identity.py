import torch, torch.nn as nn

torch.manual_seed(0)
x = torch.randn(2, 64, 32, 32).abs() + 0.5   # 全为正，排除 ReLU 干扰

# A：普通堆叠，20 层卷积
plain = nn.Sequential(*[l for _ in range(20)
                        for l in (nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())])

# B：残差堆叠，20 个 x + conv(x)
class ResStack(nn.Module):
    def __init__(self, n=20):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(64, 64, 3, padding=1) for _ in range(n)])
    def forward(self, x):
        for c in self.convs:
            x = x + c(x)      # 注意：残差，没有额外 ReLU
        return x

res = ResStack(20)

print("把两组的全部权重和偏置清零 —— 也就是让它们『什么都学不到』\n")
for m in (plain, res):
    for p in m.parameters():
        nn.init.zeros_(p)

with torch.no_grad():
    y_plain = plain(x)
    y_res   = res(x)

print(f"普通堆叠 20 层：与输入的平均误差 = {(y_plain - x).abs().mean():.6f}")
print(f"   -> 输出的绝对值均值 = {y_plain.abs().mean():.6f}   （信息全丢了）\n")
print(f"残差堆叠 20 层：与输入的平均误差 = {(y_res - x).abs().mean():.6f}")
print(f"   -> 输出的绝对值均值 = {y_res.abs().mean():.6f}   （原样保留）")