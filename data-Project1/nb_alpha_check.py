# -*- coding: utf-8 -*-
"""细化 NB 的 alpha 扫描：确认它在 TF-IDF 特征上的最优量级。

背景：§3.1 主表里 NB 用的是 alpha=1.0（sklearn 默认），得 0.9187，看起来是四个
模型里最差的。但 MultinomialNB 假设输入是**计数**，而 TF-IDF 是归一化后的连续值
（量级 0~1），alpha=1.0 的拉普拉斯平滑对这么小的"计数"来说等于先验压过证据。
本脚本把 alpha 扫一遍，看 NB 到底能到多少。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.naive_bayes import MultinomialNB

import main as M

X, y = M.load_train()
cv = M.make_cv(5, 5)
cfg, tf = M.CLEAN_PRESETS[M.BEST_PRESET], dict(M.BEST_TFIDF)

print("[NB alpha 细化扫描] routing + 50000 维 + sublinear，同一批 25 折")
print("[说明] alpha=1.0 是 sklearn 默认，也是 §3.1 主表用的值\n")

alphas = [0.03, 0.05, 0.1, 0.2, 0.3]
sink, rows = {}, []
for a in alphas:
    key = f"NB_a{a}"
    _, raw = M.run_cv({key: MultinomialNB(alpha=a)}, X, y, cv, cfg=cfg, tfidf=tf,
                      fold_sink=sink, quiet=True)
    r = raw.iloc[0]
    rows.append((a, float(r["macro_f1"]), float(r["macro_f1_std"]), float(r["gap"])))
    print(f"  alpha={a:<5} Macro-F1 = {r['macro_f1']:.4f} ± {r['macro_f1_std']:.4f}  间隙 {r['gap']:.4f}")

pd.DataFrame(sink).to_csv(M.RESULT_DIR / "nb_alpha_fold_scores_5x5.csv", index=False)
pd.DataFrame(rows, columns=["alpha", "Macro-F1", "标准差", "泛化间隙"]).to_csv(
    M.RESULT_DIR / "nb_alpha_sweep_5x5.csv", index=False, encoding="utf-8-sig")

best = max(rows, key=lambda r: r[1])
print("")
print(f"  本轮最优 alpha = {best[0]} (Macro-F1 {best[1]:.4f})")
print(f"  对照：alpha=1.0（默认，主表用值）= 0.9187，alhpa=0.01 = 0.9353")

# 与主表其余模型做配对检验
fin = pd.read_csv(M.RESULT_DIR / "model_fold_scores_routing_best_5x5.csv", index_col=0)
fair = pd.read_csv(M.RESULT_DIR / "tuning_fairness_fold_scores_5x5.csv")

agg = lambda v: np.asarray(v).reshape(5, 5).mean(axis=1)


def cmp(a, b, na, nb):
    d = np.asarray(a) - np.asarray(b)
    m = d.mean()
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=m, scale=stats.sem(d))
    p2 = stats.ttest_rel(agg(a), agg(b)).pvalue
    sig = (lo > 0 or hi < 0) and p2 < 0.05
    print(f"    {na:12s} vs {nb:14s} Δ={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
          f"p(重复聚合)={p2:.2g}  {'显著' if sig else '不显著'}")


print("\n[配对检验] NB 最优配置 与 主表其余模型的对比：")
nb_best = sink[f"NB_a{best[0]}"]
cmp(fin["LinearSVC"].to_numpy(), nb_best, "LinearSVC", f"NB(a={best[0]})")
cmp(fin["MLP"].to_numpy(), nb_best, "MLP(未调)", f"NB(a={best[0]})")
cmp(fair["LR_C100"].to_numpy(), nb_best, "LR(C=100)", f"NB(a={best[0]})")
cmp(fin["NaiveBayes"].to_numpy(), nb_best, "NB(a=1.0,默认)", f"NB(a={best[0]})")
print("\n[记录] results/nb_alpha_sweep_5x5.csv")
