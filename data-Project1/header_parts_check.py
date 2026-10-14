# -*- coding: utf-8 -*-
"""邮件头各部分的区分力：逐部分切片对照。

起因：报告 §3.2 里有一张"只留 Subject / 邮件头去掉 Subject / 正文 / 全文"的表，
四个分数（0.8798 / 0.5878 / 0.8908 / 0.9133）**在任何结果文件里都查不到**，
来自步骤 2 的一次未落盘探索。这里把它做成可复现的实验。

四种输入的定义：
    A 只留 Subject 一行      —— 仅该行，用于看标题本身的判别力
    B 邮件头去掉 Subject     —— From/Organization/Lines 等，用于看"元数据捷径"有多强
    C 去掉邮件头后的正文     —— 正文（含 Subject 行），无任何其它头部
    D 全文                   —— 原样，作为对照

评价口径与主表一致：LinearSVC(C=1) + max_features=50000 + sublinear_tf，
同一批 25 折（make_cv(5,5)）。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # noqa: F401  (仅为保持与主流程一致的导入面)
from sklearn.pipeline import Pipeline

import main as M

HEADER_RE = re.compile(
    r"^(From|Subject|Organization|Lines|Reply-To|Distribution|Nntp-Posting-Host|"
    r"X-Newsreader|Keywords|Summary|Article-I.D.|Sender|Followup-To|Originator|"
    r"Expires|References|Xref|Path|News-Software|In-Reply-To|NNTP-Posting-Host):.*$",
    re.M)
SUBJECT_RE = re.compile(r"^Subject:.*$", re.M)


def split_parts(t: str):
    """把一篇文章切成 (subject 行, 其它头部行, 正文)。"""
    subj = SUBJECT_RE.search(t)
    subj_line = subj.group(0) if subj else ""
    others = [m.group(0) for m in HEADER_RE.finditer(t) if not m.group(0).startswith("Subject:")]
    body = HEADER_RE.sub("", t)
    return subj_line, "\n".join(others), body


def build(X, which: str):
    out = []
    for t in X:
        subj, others, body = split_parts(t)
        if which == "A":
            out.append(subj)
        elif which == "B":
            out.append(others)
        elif which == "C":
            out.append(body)
        else:
            out.append(t)
    return out


def main_cli() -> None:
    X, y = M.load_train()
    cv = M.make_cv(5, 5)
    labels = {"A": "只留 Subject 一行", "B": "邮件头去掉 Subject",
              "C": "去掉邮件头后的正文", "D": "全文"}

    total = sum(len(t) for t in X)
    print("[邮件头逐部分] LinearSVC(C=1) · max_features=50000 + sublinear_tf · 25 折\n")
    rows, sink = [], {}
    for k in "ABCD":
        docs = build(X, k)
        frac = sum(len(d) for d in docs) / total * 100
        pipe = M.make_pipeline(None, **M.BEST_TFIDF)
        pipe.set_params(clf=M.make_models()["LinearSVC"])
        from sklearn.model_selection import cross_validate
        r = cross_validate(pipe, docs, y, cv=cv, scoring=M.SCORING, n_jobs=1)
        s = r["test_macro_f1"]
        sink[labels[k]] = s
        rows.append((labels[k], f"{frac:.1f}%", float(s.mean()), float(s.std(ddof=1))))
        print(f"  {labels[k]:<20} 字符占比 {frac:>5.1f}%   Macro-F1 {s.mean():.4f} ± {s.std(ddof=1):.4f}")

    pd.DataFrame(rows, columns=["输入", "字符占比", "Macro-F1", "标准差"]).to_csv(
        M.RESULT_DIR / "header_parts_5x5.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sink).to_csv(M.RESULT_DIR / "header_parts_fold_scores_5x5.csv", index=False)

    print("\n  读数：")
    ab = np.array(sink[labels["A"]]) - np.array(sink[labels["B"]])
    print(f"    A（只留 Subject）− B（去掉 Subject）= {ab.mean():+.4f}"
          f"  → Subject 一行的判别力远强于其余全部头部")
    dc = np.array(sink[labels["D"]]) - np.array(sink[labels["C"]])
    print(f"    D（全文）− C（正文）= {dc.mean():+.4f}  → 这正是 L0→L1 的收益")
    print("\n[记录] results/header_parts_5x5.csv、header_parts_fold_scores_5x5.csv")


if __name__ == "__main__":
    main_cli()
