# AUC、F1、Sensitivity、Specificity、混淆矩阵（多分类输出 macro + per-class）

# src/metrics.py —— AUC、F1、Sensitivity、Specificity、混淆矩阵
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                            accuracy_score, roc_curve, auc as sk_auc)


def _per_class_sens_spec(cm):
    """从混淆矩阵算每一类的 Sensitivity 与 Specificity。"""
    n = cm.shape[0]
    sens, spec = [], []
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp          # 第 i 行其余 = 被错分到别的类
        fp = cm[:, i].sum() - tp          # 第 i 列其余 = 别的类被错分成 i
        tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        spec.append(tn / (tn + fp) if tn + fp > 0 else 0.0)
    return np.array(sens), np.array(spec)


def compute_metrics(y_true, y_prob, y_pred=None, class_names=None):
    """
    y_true : 真实标签 (N,)
    y_prob : 预测概率 (N, C)  —— 多分类 OvR，二分类取正类
    y_pred : 预测类别 (N,)，不传则自动取 argmax
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)
    if y_pred is None:
        y_pred = y_prob.argmax(1)
    y_pred = np.asarray(y_pred).ravel()

    binary = y_prob.shape[1] == 2
    pos = y_prob[:, 1] if binary else y_prob

    # AUC
    try:
        if binary:
            auc_macro = float(roc_auc_score(y_true, pos))
        else:
            auc_macro = float(roc_auc_score(y_true, y_prob,
                                            multi_class="ovr", average="macro"))
    except ValueError:
        auc_macro = float("nan")

    res = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "auc_macro":   auc_macro,
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro":    float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }

    # 二分类：直接算正类的 Sens / Spec
    if binary:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        res["sensitivity"] = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        res["specificity"] = float(tn / (tn + fp)) if tn + fp > 0 else 0.0
        res["sens_macro"] = res["sensitivity"]
        res["spec_macro"] = res["specificity"]
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(y_prob.shape[1])))
        sens, spec = _per_class_sens_spec(cm)
        res["sens_macro"] = float(sens.mean())
        res["spec_macro"] = float(spec.mean())
        res["per_class_sens"] = {str(i): round(float(sens[i]), 4)
                                 for i in range(len(sens))}
        res["per_class_spec"] = {str(i): round(float(spec[i]), 4)
                                 for i in range(len(spec))}

    res["confusion_matrix"] = cm.tolist()
    if class_names is not None:
        res["class_names"] = list(class_names)
    return res


def get_roc_data(y_true, y_prob, class_names=None):
    """返回 ROC 曲线数据（二分类一条，多分类 macro + 每类）。"""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)
    out = {}

    if y_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        out["macro"] = (fpr, tpr, sk_auc(fpr, tpr))
        return out

    n_cls = y_prob.shape[1]
    y_bin = np.eye(n_cls)[y_true.astype(int)]      # one-hot，OvR
    fprs, tprs = [], []
    all_fpr = np.linspace(0, 1, 100)
    for i in range(n_cls):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        out[str(i)] = (fpr, tpr, sk_auc(fpr, tpr))
        tprs.append(np.interp(all_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    out["macro"] = (all_fpr, mean_tpr, sk_auc(all_fpr, mean_tpr))
    return out