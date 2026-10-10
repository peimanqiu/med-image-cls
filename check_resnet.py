# 把 resnet18 拆开看清楚
import torch, torch.nn as nn, torchvision, inspect
from torchvision import models

net = models.resnet18(weights=None)
net.eval()

lines = []
def log(s=""):
    print(s); lines.append(s)

log("=" * 66)
log(f"resnet18 结构解剖   torch {torch.__version__} | torchvision {torchvision.__version__}")
log("=" * 66)

# 【1】完整模块树
log("\n【1】完整模块树")
log(str(net))

# 【2】输入 224x224 时逐层的张量形状
log("\n【2】输入 224×224 时，每层的张量形状（B=1）")
targets = [("conv1", net.conv1), ("bn1", net.bn1), ("relu", net.relu),
           ("maxpool", net.maxpool), ("layer1", net.layer1), ("layer2", net.layer2),
           ("layer3", net.layer3), ("layer4", net.layer4),
           ("avgpool", net.avgpool), ("fc", net.fc)]

def make_hook(name):
    def h(m, i, o):
        log(f"  {name:9s} {str(tuple(i[0].shape)):20s} -> {tuple(o.shape)}")
    return h

hooks = [m.register_forward_hook(make_hook(n)) for n, m in targets]
with torch.no_grad():
    y = net(torch.randn(1, 3, 224, 224))
for h in hooks: h.remove()

# 【3】参数量
log("\n【3】参数量统计")
total     = sum(p.numel() for p in net.parameters())
fc_params = sum(p.numel() for p in net.fc.parameters())
log(f"  原始(1000类) 总参数 : {total:,}  ({total/1e6:.2f}M)")
log(f"  其中 fc 层          : {fc_params:,}")
log(f"  主干 backbone       : {total-fc_params:,}  ({(total-fc_params)/1e6:.2f}M)")

for n in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
    m = getattr(net, n)
    p = sum(q.numel() for q in m.parameters())
    log(f"    {n:8s}: {p:>10,}  ({p/1e6:.2f}M)")

net7 = models.resnet18(weights=None)
net7.fc = nn.Linear(net7.fc.in_features, 7)
log(f"  换成 7 类后总参数   : {sum(p.numel() for p in net7.parameters()):,}"
    f"  ({sum(p.numel() for p in net7.parameters())/1e6:.2f}M)   <-- Day5 用这个")

# 【4】downsample 出现在哪
log("\n【4】downsample（捷径上的 1x1 卷积）出现位置")
for name in ["layer1", "layer2", "layer3", "layer4"]:
    layer = getattr(net, name)
    for i, blk in enumerate(layer):
        ds = blk.downsample
        info = "无（直接相加）" if ds is None else f"有 -> {str(ds[0]).split('(')[0]} {ds[0].in_channels}→{ds[0].out_channels}, stride={ds[0].stride}"
        log(f"  {name} block{i}: {info}")

# 【5】源码级确认
log("\n【5】BasicBlock.forward 源码（重点看那个 +=）")
log(inspect.getsource(torchvision.models.resnet.BasicBlock.forward))

with open("report/resnet18_arch.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n✅ 已保存到 report/resnet18_arch.txt")