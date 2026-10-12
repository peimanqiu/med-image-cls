# src/utils.py —— set_seed、日志、保存 checkpoint、打印显存峰值
import os, random, logging
import numpy as np
import torch


def set_seed(seed=42):
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_path):
    """同时输出到屏幕和文件的日志器。"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_ckpt(path, model, epoch, metric):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metric": metric}, path)


def peak_mem():
    """返回本次训练显存峰值(GB)，写进 summary 用。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0


def reset_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()