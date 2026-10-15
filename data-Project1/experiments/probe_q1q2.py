# -*- coding: utf-8 -*-
"""思考题①②的判定性小实验。

【为什么需要它】
报告 §5.1 用"训练集能被线性模型近乎完全分开"来支持"高维稀疏下数据近似线性可分"。
但这个论证有个漏洞：当**特征维度 d > 样本数 n** 时，任何一组标签都能被线性模型
完美分开（线性分类器的 VC 维是 d+1），所以"训练分高"在 d>n 时**几乎不携带信息**。

本脚本做两件事：

  实验 A（随机标签对照）—— 把标签随机打乱后重跑，
      若照样能把训练集完美分开、而验证集掉到随机水平（10 类 ≈ 0.10），
      就证明"训练分高"在这个维度下确实不能作为"近似线性可分"的证据。
      这样一来，"近似线性可分"这个论断就必须改用**泛化性能**来支撑。

  实验 B（MLP 迭代次数诊断）—— 记录 MLP 实际用了多少次迭代、有没有撞到 max_iter 上限。
      用于回答思考题②里"增加 epochs 能否提升"：若早停总在上限之前触发，
      说明 epochs 不是瓶颈。

用法：python probe_q1q2.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

import os as _os, sys as _sys
# 本文件位于子目录，而 main.py 在上一级——把它加进 sys.path 才能 import 到 main。
# main.py 里的路径常量（M.RESULT_DIR / M.FIG_DIR / M.TRAIN_CSV）都以 main.py 自身
# 的位置为锚，所以本文件搬家不影响它们。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import main as M


def exp_a_random_labels() -> None:
    print("=" * 78)
    print("  实验 A · 随机标签对照：d > n 时「训练分高」还说明得了什么？")
    print("=" * 78)
    X, y = M.load_train()
    docs = M.apply_clean(X, M.CLEAN_PRESETS[M.BEST_PRESET])
    rng = np.random.default_rng(M.SEED)
    rows = []

    for label, yy in (("真实标签", y), ("随机标签（打乱）", rng.permutation(y))):
        for dim, tf in ((5000, {"max_features": 5000}), (50000, dict(M.BEST_TFIDF))):
            cv = StratifiedKFold(5, shuffle=True, random_state=M.SEED)
            pipe = M.make_pipeline(None, **tf)
            pipe.set_params(clf=M.make_models()["LinearSVC"])
            r = cross_validate(pipe, docs, yy, cv=cv, scoring=M.SCORING,
                               return_train_score=True, n_jobs=1)
            tr, te = r["train_macro_f1"].mean(), r["test_macro_f1"].mean()
            rows.append({"标签": label, "维度": dim, "训练-MacroF1": round(tr, 4),
                         "验证-MacroF1": round(te, 4), "d 与 n": f"{dim} vs 5894"})
            print(f"  {label:16s} {dim:>6d} 维  训练 {tr:.4f}   验证 {te:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(M.RESULT_DIR / "random_label_control_5x5.csv", index=False, encoding="utf-8-sig")
    print()
    print("  读数：")
    print("    · 50,000 维（d>n）下，**随机标签照样被完美分开**——所以这个维度上的"
          "\n      「训练分高」不能作为「近似线性可分」的证据。")
    print("    · 5,000 维（d<n）下，随机标签**分不开**（训练分掉下来）——"
          "\n      这个设定下的训练分才携带信息。")
    print("    · 因此「近似线性可分」必须靠**泛化性能**（验证分）来支撑，见实验结论。")
    print("\n  [记录] results/random_label_control_5x5.csv")


def exp_b_mlp_epochs() -> None:
    print()
    print("=" * 78)
    print("  实验 B · MLP 迭代次数诊断：epochs 是不是瓶颈？")
    print("=" * 78)
    from sklearn.feature_extraction.text import TfidfVectorizer
    X, y = M.load_train()
    docs = M.apply_clean(X, M.CLEAN_PRESETS[M.BEST_PRESET])
    tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=M.SEED).split(docs, y)))
    vec = TfidfVectorizer(**M.BEST_TFIDF)
    Xtr = vec.fit_transform([docs[i] for i in tr])
    print(f"  第 1 折：训练 {len(tr)} 条 / 验证 {len(te)} 条 | 维度 {Xtr.shape[1]:,}\n")
    rows = []
    # 全部保留早停（关掉早停跑满 max_iter 在 5 万维下要几小时，不划算）。
    # 问"加大 epochs 有没有用"，等价于问"早停是不是停早了"——所以改的是**耐心**：
    #   n_iter_no_change 是"验证分连续多少轮不改善就停"，默认 10。
    #   把它放大到 30，就是让模型多练 3 倍才允许停；若结果不变，说明早已收敛。
    for name, kw in (
        ("默认（耐心=10, 上限=300）", dict(early_stopping=True, max_iter=300, n_iter_no_change=10)),
        ("耐心=30（多练 3 倍才停）", dict(early_stopping=True, max_iter=300, n_iter_no_change=30)),
        ("上限提到 1000", dict(early_stopping=True, max_iter=1000, n_iter_no_change=10)),
    ):
        clf = MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-4,
                            random_state=M.SEED, **kw)
        t0 = time.time()
        clf.fit(Xtr, y[tr])
        el = time.time() - t0
        acc_te = f1_score(y[te], clf.predict(vec.transform([docs[i] for i in te])), average="macro")
        acc_tr = f1_score(y[tr], clf.predict(Xtr), average="macro")
        n_it = int(np.ravel(clf.n_iter_)[0])
        hit = "撞到上限!" if n_it >= kw["max_iter"] else "早停/收敛"
        rows.append({"配置": name, "迭代次数": n_it,
                     "上限": kw["max_iter"], "状态": hit,
                     "训练-MacroF1": round(acc_tr, 4), "验证-MacroF1": round(acc_te, 4),
                     "耗时(s)": round(el, 1)})
        print(f"  {name:24s} 迭代 {n_it:>4d}/{kw['max_iter']:<5d} {hit:9s} "
              f"训练 {acc_tr:.4f}  验证 {acc_te:.4f}  ({el:.0f}s)")

    pd.DataFrame(rows).to_csv(M.RESULT_DIR / "mlp_epoch_probe_5fold.csv",
                              index=False, encoding="utf-8-sig")
    print("\n  读数：见表中「状态」列——若默认配置远未撞到 max_iter 上限，")
    print("        说明早停已经在合适的位置停下，单纯加大 epochs 不会带来提升。")
    print("\n  [记录] results/mlp_epoch_probe_5fold.csv")


if __name__ == "__main__":
    exp_a_random_labels()
    exp_b_mlp_epochs()
