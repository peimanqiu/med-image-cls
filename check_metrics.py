import numpy as np
from src.metrics import compute_metrics

# 故意造一个已知答案的小例子：10 个样本，7 真肺炎 3 真正常
y_true = np.array([1,1,1,1,1,1,1, 0,0,0])
# 模型：抓对 6 个肺炎、漏 1 个；3 个正常里误判 1 个
y_pred = np.array([1,1,1,1,1,1,0, 1,0,0])
y_prob = np.zeros((10,2)); y_prob[:,1] = y_pred   # 简化概率

r = compute_metrics(y_true, y_prob, y_pred)
print(f"Sensitivity = {r['sensitivity']:.4f}   手算: 6/7 = {6/7:.4f}")
print(f"Specificity = {r['specificity']:.4f}   手算: 2/3 = {2/3:.4f}")