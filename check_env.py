import torch, torchvision, platform, sys

print("Python      :", sys.version.split()[0], "|", platform.system())
print("PyTorch     :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("CUDA 可用   :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("显卡型号    :", torch.cuda.get_device_name(0))
    print("显存总量GB  :", torch.cuda.get_device_properties(0).total_memory / 1024**3)
    x = torch.randn(8, 3, 224, 224).cuda()      # 模拟一个 batch 的 224×224 图像
    print("前向测试    :", x.shape, "已在显存上，占用 %.2f GB" % (torch.cuda.memory_allocated()/1024**3))
    del x; torch.cuda.empty_cache()
    print("✅ 环境正常，可以开始 Day2")
else:
    print("❌ 没认到显卡，看下面的排错表")