# 汇总所有评估结果为 results/summary.csv
import os, json, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP  = os.path.join(ROOT, "experiments")
OUT  = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

rows = []
for name in sorted(os.listdir(EXP)):
    d = os.path.join(EXP, name)
    if not os.path.isdir(d) or name.startswith("smoke_"):
        continue
    mp = os.path.join(d, "eval", "metrics.json")
    if not os.path.exists(mp):
        mp = os.path.join(d, "metrics.json")
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp, encoding="utf-8"))
    if "auc_macro" not in m:
        print(f"  跳过 {name}（未评估）")
        continue

    tm_path = os.path.join(d, "metrics.json")
    tm = json.load(open(tm_path, encoding="utf-8")) if os.path.exists(tm_path) else {}
    a, h = tm.get("args", {}), tm.get("history", [{}])[-1]

    rows.append({
        "实验": name,
        "模型": a.get("model", ""),
        "损失": a.get("loss", "ce"),
        "AUC":  round(m.get("auc_macro", 0), 4),
        "F1":   round(m.get("f1_macro", 0), 4),
        "Sens": round(m.get("sensitivity", m.get("sens_macro", 0)), 4),
        "Spec": round(m.get("specificity", m.get("spec_macro", 0)), 4),
        "Acc":  round(m.get("accuracy", 0), 4),
        "耗时s": h.get("sec", ""),
        "显存GB": h.get("peak_mem_gb", ""),
    })

path = os.path.join(OUT, "summary.csv")
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

print(f"已写出 {path}，共 {len(rows)} 行\n")
for r in rows:
    print(f"{r['实验']:30s} {r['模型']:14s} {r['损失']:12s} "
          f"AUC {r['AUC']:.4f}  F1 {r['F1']:.4f}  Sens {r['Sens']:.4f}  Acc {r['Acc']:.4f}")