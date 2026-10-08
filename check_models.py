import torch, torchvision
from torchvision import models
# DermaMNIST 7分类
NUM_CLASSES = 7

def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6

# 只保存模型构造函数，不在外部提前实例化
specs = [
    ("resnet18",      models.resnet18,      "fc"),
    ("densenet121",   models.densenet121,   "classifier"),
    ("convnext_tiny", models.convnext_tiny, "classifier"),
]

print(f"torchvision {torchvision.__version__}\n")

for name, model_func, head in specs:
    # 每次循环新建原始模型，避免复用修改后的对象
    net = model_func(weights=None)

    if name == "resnet18":
        # resnet: 分类头是 fc (Linear)
        net.fc = torch.nn.Linear(net.fc.in_features, NUM_CLASSES)

    elif name == "densenet121":
        # densenet121: 分类头 classifier 直接就是 Linear，不是 Sequential
        in_dim = net.classifier.in_features
        net.classifier = torch.nn.Linear(in_dim, NUM_CLASSES)

    elif name == "convnext_tiny":
        # convnext_tiny: classifier 是 Sequential，Linear 在索引 [2]
        in_dim = net.classifier[2].in_features
        net.classifier[2] = torch.nn.Linear(in_dim, NUM_CLASSES)

    net = net.cuda()
    net.eval()
    x = torch.randn(2, 3, 224, 224).cuda()   # batch_size=2
    with torch.no_grad():
        y = net(x)
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{name:15s} 参数量 {count_params(net):6.2f}M | 输出 {tuple(y.shape)} | bs=2 显存 {mem:.2f}GB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print("\n✅ 三个模型都能加载，Day13/Day14 不会卡在版本问题上")
