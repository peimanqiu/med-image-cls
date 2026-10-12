"""
transform与数据集
train:做增广
val/test:只做归一化
"""
import os
import numpy as np
from torchvision import transforms
from medmnist import PneumoniaMNIST, DermaMNIST, ChestMNIST

#ImageNet的均值与标准差 预训练权重使用
IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD = [0.229,0.224,0.225]

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data","raw")

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2,contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
        ])

def get_datasets(flag="pneumoniamnist", size=224):
    cls_map = {"pneumoniamnist": PneumoniaMNIST,
               "dermamnist":     DermaMNIST,
               "chestmnist":     ChestMNIST}
    cls = cls_map[flag]
    train_ds = cls(root=DATA_ROOT, split="train", size=size, download=True,
                   as_rgb=True, transform=get_transforms(train=True))
    val_ds   = cls(root=DATA_ROOT, split="val",   size=size, download=True,
                   as_rgb=True, transform=get_transforms(train=False))
    test_ds  = cls(root=DATA_ROOT, split="test",  size=size, download=True,
                   as_rgb=True, transform=get_transforms(train=False))
    return train_ds, val_ds, test_ds