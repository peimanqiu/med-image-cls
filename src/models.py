# get_model(name, num_classes, pretrained)：resnet18 / densenet121 / convnext_tiny

#src/models.py --get_models():统一三个架构的入口
import torch.nn as nn
from torchvision import models

def get_model(name="resnet18",num_classes=2,pretrained=True):
    """
    resnet18      -> net.fc
    densenet121   -> net.classifier        (本身是 Linear)
    convnext_tiny -> net.classifier[2]     (Sequential 的第 3 层)
    """

    name=name.lower()
    weights="DEFAULT" if pretrained else None

    if name=="resnet18":
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features,num_classes)

    elif name == "densenet121":
        net = models.densenet121(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)

    elif name == "convnext_tiny":
        net = models.convnext_tiny(weights=weights)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"未知模型: {name}，可选 resnet18 / densenet121 / convnext_tiny")

    return net