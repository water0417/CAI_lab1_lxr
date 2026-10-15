# -*- coding: utf-8 -*-
"""补齐「各模型调到自身最优后」的完整指标。

起因：报告表 3-1 的上半部分（库默认超参）有完整六项指标，
但下半部分（各自最优）此前只有 Macro-F1 —— 因为那些最优配置是在
tuning_fairness / nb_alpha_sweep 里扫出来的，那两次只落了部分指标。

这里用 `main.run_cv` 原样重跑 LR(C=100) 与 NB(alpha=0.1) 的 25 折，
与主表同一条代码路径、同一批折，把 Accuracy / Weighted-F1 / 训练-MacroF1
也补齐，使上下两部分**逐格可比**。

MLP 的最优配置已有完整指标（mlp_tuned_25fold_5x5.csv，由 mlp_tune.py confirm 产出）。
LinearSVC 的最优即默认（C=1），无需重跑。
"""
from __future__ import annotations

import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import os as _os, sys as _sys
# 本文件位于子目录，而 main.py 在上一级——把它加进 sys.path 才能 import 到 main。
# main.py 里的路径常量（M.RESULT_DIR / M.FIG_DIR / M.TRAIN_CSV）都以 main.py 自身
# 的位置为锚，所以本文件搬家不影响它们。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import main as M

CASES = [
    ("LogisticRegression", "LogisticRegression · C=100（最优）",
     LogisticRegression(C=100, max_iter=1000, random_state=M.SEED)),
    ("NaiveBayes", "NaiveBayes · alpha=0.1（最优）",
     MultinomialNB(alpha=0.1)),
]


def main_cli() -> None:
    X, y = M.load_train()
    cv = M.make_cv(5, 5)
    cfg, tf = M.CLEAN_PRESETS[M.BEST_PRESET], dict(M.BEST_TFIDF)
    print("[补齐最优配置指标] routing + max_features=50000 + sublinear_tf，25 折")
    print("[说明] 与主表同一条代码路径（main.run_cv），故与表 3-1 上半部分逐格可比\n")

    rows = []
    for name, label, clf in CASES:
        t0 = time.time()
        _, raw = M.run_cv({name: clf}, X, y, cv, cfg=cfg, tfidf=tf, quiet=True)
        r = raw.iloc[0]
        rows.append({
            "模型": name, "配置": label,
            "Accuracy": round(float(r["accuracy"]), 6),
            "Macro-F1": round(float(r["macro_f1"]), 6),
            "MacroF1标准差": round(float(r["macro_f1_std"]), 6),
            "Weighted-F1": round(float(r["weighted_f1"]), 6),
            "训练-MacroF1": round(float(r["train_macro_f1"]), 6),
            "泛化间隙": round(float(r["gap"]), 6),
            "耗时(s)": round(time.time() - t0, 1),
        })
        print(f"  {label:34s} Macro-F1={r['macro_f1']:.4f} ± {r['macro_f1_std']:.4f}  "
              f"Accuracy={r['accuracy']:.4f}  间隙={r['gap']:.4f}  ({rows[-1]['耗时(s)']:.0f}s)")

    out = M.RESULT_DIR / "model_metrics_tuned_5x5.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  [记录] {out.relative_to(M.PROJECT_DIR)}")


if __name__ == "__main__":
    main_cli()
