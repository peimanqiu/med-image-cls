# src/losses.py —— CE / WeightedCE / FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(loss_name="ce", class_weight=None, gamma=2.0, device="cuda"):
    """
    loss_name   : ce / weighted_ce / focal
    class_weight: 一维数组，长度=类别数

    关键：ce 分支绝不使用 class_weight（否则 CE 与 WeightedCE 会完全相同）
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss()          # 括号内为空 = 纯 CE

    w = None
    if class_weight is not None:
        w = torch.as_tensor(class_weight, dtype=torch.float32).to(device)

    if loss_name == "weighted_ce":
        assert w is not None, "weighted_ce 需要 class_weight"
        return nn.CrossEntropyLoss(weight=w)

    if loss_name == "focal":
        return FocalLoss(gamma=gamma, weight=w)

    raise ValueError(f"未知损失: {loss_name}")


class FocalLoss(nn.Module):
    """FL = (1-pt)^gamma * CE，gamma=0 时退化为 CE"""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def compute_class_weights(labels, num_classes):
    """按频率倒数加权，归一化后平均权重=1（学习率不用改）"""
    import numpy as np
    y = np.asarray(labels).ravel().astype(int)
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    return w / w.mean()