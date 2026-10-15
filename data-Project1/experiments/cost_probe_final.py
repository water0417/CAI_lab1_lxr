# -*- coding: utf-8 -*-
"""表 3-1 的成本列：受控单折测量（四个模型的**最终配置**）。

为什么需要重测：
表 3-1 原来的「25 折耗时」分别来自四次不同的运行，机器负载与休眠情况各不相同，
互相之间不可比（LR 那次 1625 s，与默认配置的 240 s 差近 7 倍，而它只多迭代了
64 vs 46 次——明显被拉长了）。

这里按 §3.5 的做法重来：**同一折、同一进程、背靠背测完四个模型**，
每个模型都用表 3-1 ① 部分的最终配置。

注意：与 §3.5 的探测一样，绝对耗时会随机器负载波动（实测 1.5~2.5 倍），
所以结论只取量级，不引用具体倍数。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

import os as _os, sys as _sys
# 本文件位于子目录，而 main.py 在上一级——把它加进 sys.path 才能 import 到 main。
# main.py 里的路径常量（M.RESULT_DIR / M.FIG_DIR / M.TRAIN_CSV）都以 main.py 自身
# 的位置为锚，所以本文件搬家不影响它们。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import main as M


def main_cli() -> None:
    X, y = M.load_train()
    docs = M.apply_clean(X, M.CLEAN_PRESETS[M.BEST_PRESET])
    tr, _ = next(iter(StratifiedKFold(5, shuffle=True, random_state=M.SEED).split(docs, y)))
    from sklearn.feature_extraction.text import TfidfVectorizer
    t = time.time()
    Xtr = TfidfVectorizer(**M.BEST_TFIDF).fit_transform([docs[i] for i in tr])
    print(f"[受控单折成本] 表 3-1 ① 部分的最终配置 | 第 1 折 训练 {len(tr)} 条 | "
          f"维度 {Xtr.shape[1]:,} | TF-IDF 拟合 {time.time()-t:.1f}s\n")

    # 与表 3-1 ① 部分一一对应的最终配置
    clfs = [
        ("MLP", "hidden=(100,)、alpha=1e-3", MLPClassifier(
            hidden_layer_sizes=(100,), alpha=1e-3, max_iter=300,
            early_stopping=True, random_state=M.SEED)),
        ("LinearSVC", "C=1", LinearSVC(random_state=M.SEED, C=1.0)),
        ("LogisticRegression", "C=100", LogisticRegression(
            C=100, max_iter=1000, random_state=M.SEED)),
        ("NaiveBayes", "alpha=0.1", MultinomialNB(alpha=0.1)),
    ]

    rows = []
    for name, cfg, clf in clfs:
        t0 = time.time()
        clf.fit(Xtr, y[tr])
        el = time.time() - t0
        n_it = getattr(clf, "n_iter_", None)
        n_it = int(np.ravel(n_it)[0]) if n_it is not None else None
        rows.append({"模型": name, "最终配置": cfg, "单折拟合耗时(s)": round(el, 2),
                     "迭代次数": n_it})
        print(f"  {name:20s} {cfg:26s} {el:8.2f}s"
              + (f"   迭代 {n_it}" if n_it else ""))

    df = pd.DataFrame(rows)
    out = M.RESULT_DIR / "model_cost_final_config_5x5.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print()
    svc = float(df[df["模型"] == "LinearSVC"]["单折拟合耗时(s)"].iloc[0])
    mlp = float(df[df["模型"] == "MLP"]["单折拟合耗时(s)"].iloc[0])
    print(f"  对照：LinearSVC {svc:.2f}s，MLP {mlp:.2f}s → MLP 慢约 {mlp/svc:.0f} 倍")
    print("  （绝对耗时随机器负载波动 1.5~2.5 倍，故只用来看量级）")
    print(f"\n  [记录] {out.relative_to(M.PROJECT_DIR)}")


if __name__ == "__main__":
    main_cli()
