# 医学图像分类实战：ResNet / DenseNet / ConvNeXt

## 项目目标
基于 MedMNIST 数据集完成医学图像分类实验：单标签 + 多标签分类，
对比 CE / Weighted CE / Focal Loss，输出 AUC / F1 / Sensitivity / Specificity，
绘制 ROC 曲线与混淆矩阵，并做错误案例分析。

## 环境
- 显卡：NVIDIA GeForce GTX 1650（4GB）
- PyTorch 2.5.1 + CUDA 12.1
- Python 3.9.25 (conda env: dl)

## 数据集
- PneumoniaMNIST (224)：二分类，热身
- DermaMNIST (224)：7 类单标签，主线
- ChestMNIST (224)：14 类多标签，进阶

## 运行顺序


## 进度
- Day1 (10-07)：环境搭建完成，GTX 1650 4GB + PyTorch 2.5.1 + CUDA 可用
- Day2 (10-08)：项目骨架搭建完成，15 个文件就位