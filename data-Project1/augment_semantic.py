# -*- coding: utf-8 -*-
"""
实验一 · 补充实验：语义级数据增强（词向量近邻替换）

## 为什么要做这个

原实验只做了**结构扰动**（删词、换词序）。任务书要求"增强方法必须符合数据语义"，
而结构扰动被明确列为报告的局限性。本模块补上**语义级**的那一半：
把词替换成它的**分布近邻词**。

## 与结构扰动的关键区别

结构扰动只改"数量/顺序"，不改词本身；近邻替换改的是**词形**，试图让模型见到更多
表达方式。但它**不改变类别之间的词汇重叠**——而错误分析（报告 §四）指出，本任务的
瓶颈恰恰是"四个计算机类共享大量词汇"。所以预期收益仍然有限，本实验的价值在于
**把"增强无收益"这个结论从结构扰动推广到语义级**。

## 三条设计约束

1. **无新增依赖**：词向量由语料自身的 TF-IDF 矩阵经 TruncatedSVD（LSA）得到，
   不需要 gensim、不需要下载预训练词向量。
2. **折内构建，杜绝泄露**：词向量、近邻表、类内关联都**只在当前折的训练部分**上估计。
   若在全量语料上建好再用，验证集的信息会通过词与词的共现结构渗进训练数据——
   这比"先增广再划分"更隐蔽。
3. **只在同类内替换，保证标签仍然成立**：候选近邻必须对该类别有正向关联
   （`P(c|w)/P(c) > 1`）。否则会出现"把汽车类的 drive 换成磁盘类的 disk"这类
   破坏标签的替换——而 drive 恰好同时出现在汽车类与 PC 硬件类里。

## 用法

    cd data-Project1
    python augment_semantic.py                          # 5 折 ×5 重复，约 15 分钟
    python augment_semantic.py --folds 3 --repeats 1    # 快速试跑

输出（文件名与既有结果完全不重名，不影响任何已有文件）：
    ../results/semantic_aug_<tag>.csv               总表
    ../results/semantic_aug_fold_scores_<tag>.csv   逐折分数（可与结构扰动做配对检验）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

import main as M

RESULT_DIR = M.RESULT_DIR

# ---- 词向量与替换的超参（只影响"怎么选近邻"，不参与分类）----
BASIS_FEATURES = 20000     # LSA 基底的词表上限（比分类用的 50000 小，够用且更快）
N_COMPONENTS = 150         # LSA 维度
N_NEIGHBORS = 20           # 每个词保留多少个近邻候选
MIN_SIM = 0.50             # 近邻余弦相似度下限，太不像的不换
AFFINITY_MIN = 1.0         # 类内关留下限：P(c|w)/P(c) 必须大于此值
MIN_DF = 20                # 源词至少要出现在这么多篇文档里才允许被替换
                           # —— 低频词的 LSA 向量不可靠，换出来的"近邻"往往是
                           #    同现噪声（实测出现过把姓氏换成缩写的情况）


# ---------------------------------------------------------------------------
# 词向量与近邻
# ---------------------------------------------------------------------------
def build_word_vectors(docs_tr: list[str], seed: int = M.SEED):
    """在**训练折**上训练 LSA 词向量，返回 (词表, 单位化词向量, 词→下标, 分词器)。

    TF-IDF(文档×词) --TruncatedSVD--> 词向量 = components_ 的转置。
    这是经典的 LSA 词表示，只用语料自身，无需外部资源。

    ⚠️ 必须把 `vec.build_analyzer()` 一并返回：TF-IDF 用的是自己的 token_pattern
    （按非字母数字切分、小写化），与 `str.split()` **不一致**。若替换时用 split 分词，
    `"drive,"` 这类带标点的词元在词表里查不到，会导致一个词都换不掉。
    """
    vec = TfidfVectorizer(max_features=BASIS_FEATURES, sublinear_tf=True)
    X = vec.fit_transform(docs_tr)
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=seed,
                       algorithm="randomized", n_iter=5)
    svd.fit(X)
    W = svd.components_.T                                   # (n_terms, k)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W = W / norms                                           # 单位化后点积即余弦
    terms = vec.get_feature_names_out()
    # 只允许替换"足够常见"的词：低频词的 LSA 向量不可靠，换出来的近邻是噪声
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    replaceable = df >= MIN_DF
    return terms, W, {t: i for i, t in enumerate(terms)}, vec.build_analyzer(), replaceable


def class_affinity(docs_tr: list[str], y_tr: np.ndarray, terms: np.ndarray) -> np.ndarray:
    """词 × 类别的关联矩阵 aff[c, w] = P(c|w) / P(c)。>1 表示该词在类别 c 里更常见。

    用于把替换限制在"同类内"，避免跨类别替换破坏标签。

    ⚠️ 这里的分母是**该类别的文档数**，不是 `C.sum(axis=1)`。
    后者是"该类在所有词上的计数之和"，量级约为 文档数 × 词表大小，
    会让所有亲和度被压小几个数量级、永远过不了阈值（曾因此一个词都换不掉）。
    """
    cv = CountVectorizer(vocabulary=terms, binary=True)
    Xb = cv.fit_transform(docs_tr)                          # 二值文档-词矩阵
    C = np.vstack([np.asarray(Xb[y_tr == c].sum(axis=0)).ravel()
                   for c in range(M.N_CLASSES)])            # (n_classes, n_terms)
    n_c = np.bincount(y_tr, minlength=M.N_CLASSES).reshape(-1, 1)   # 各类文档数
    df = C.sum(axis=0, keepdims=True)                       # 每个词出现在多少篇文档
    n_doc = len(docs_tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        # (C/n_c) = P(w|c)； (df/n_doc) = P(w)； 比值即 P(c|w)/P(c)
        aff = (C / n_c) / (df / n_doc)
    return np.nan_to_num(aff, nan=0.0, posinf=0.0)


def build_neighbor_table(W: np.ndarray) -> list[list[int]]:
    """每个词的近邻下标表，用余弦相似度阈值过滤。"""
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric="cosine").fit(W)
    dist, idx = nn.kneighbors(W)
    sim = 1.0 - dist
    table = []
    for i in range(len(W)):
        # 剔除自身（下标相同），再按相似度阈值过滤
        table.append([int(j) for j, s in zip(idx[i], sim[i]) if j != i and s >= MIN_SIM])
    return table


def replace_doc(text: str, rng: np.random.RandomState, analyzer,
                term2idx: dict, table: list, aff: np.ndarray,
                vocab: np.ndarray, cls: int, rate: float,
                replaceable: np.ndarray | None = None) -> tuple[str, int]:
    """对单篇文档做近邻替换，返回 (新文本, 实际替换的词数)。

    分词用 `analyzer`（= TF-IDF 的 `build_analyzer()`），保证词元与词表对齐。
    替换后按空格拼接——对 TF-IDF 而言特征向量不变（它本来就忽略标点），
    所以不会造成训练/评估的表示不一致。
    """
    toks = analyzer(text)
    if len(toks) < 8:
        return text, 0
    aff_c = aff[cls]
    out, n_rep = [], 0
    for t in toks:
        i = term2idx.get(t)
        if (i is None or rng.rand() >= rate
                or (replaceable is not None and not replaceable[i])):
            out.append(t)
            continue
        cand = [j for j in table[i] if aff_c[j] > AFFINITY_MIN]
        if not cand:
            out.append(t)                       # 没有合格的同类近邻 → 不换
            continue
        out.append(str(vocab[cand[int(rng.randint(len(cand)))] ]))
        n_rep += 1
    return " ".join(out), n_rep


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(tag: str, folds: int, repeats: int, eval_rate: float = 0.10) -> None:
    t0 = time.time()
    X, y = M.load_train()
    cv = M.make_cv(folds, repeats)
    docs_all = np.asarray(M.apply_clean(X, M.CLEAN_PRESETS[M.BEST_PRESET]), dtype=object)

    print("=" * 78)
    print("  语义级数据增强（LSA 词向量近邻替换）")
    print(f"  前提 {M.BEST_PRESET} · 分类器 LinearSVC C=1 · TF-IDF {M.tfidf_label(M.BEST_TFIDF)}")
    print(f"  LSA 基底 {BASIS_FEATURES} 词 × {N_COMPONENTS} 维 · 近邻 {N_NEIGHBORS} 个 · "
          f"相似度下限 {MIN_SIM} · 类内关留下限 {AFFINITY_MIN}")
    print(f"  {folds} 折 × {repeats} 重复 = {folds * repeats} 个估计值")
    print("=" * 78)

    modes = [("不加增强（基线）", 0.0), ("近邻替换 10%", 0.10), ("近邻替换 20%", 0.20)]
    rows, fold_cols = [], {}
    for desc, rate in modes:
        clean_s, pert_s, changed = [], [], []
        t1 = time.time()
        for fi, (tr, te) in enumerate(cv.split(docs_all, y), 1):
            Xtr = [docs_all[i] for i in tr]
            ytr = y[tr]
            if rate > 0:
                # —— 词向量与近邻表只在本折训练部分上构建（防泄露）——
                vocab, W, term2idx, analyzer, repl = build_word_vectors(Xtr)
                aff = class_affinity(Xtr, ytr, vocab)
                table = build_neighbor_table(W)
                rng = np.random.RandomState(M.SEED * 1000 + fi)
                add, n_rep, n_tok = [], 0, 0
                for s, c in zip(Xtr, ytr):
                    r, k = replace_doc(s, rng, analyzer, term2idx, table,
                                       aff, vocab, int(c), rate, repl)
                    n_rep += k
                    n_tok += len(analyzer(s))
                    add.append(r)
                changed.append(n_rep / max(n_tok, 1))
                Xtr = Xtr + add
                ytr = np.concatenate([ytr, ytr])

            pipe = M.make_pipeline(**M.BEST_TFIDF)
            pipe.set_params(clf=M.LinearSVC(random_state=M.SEED, C=1.0))
            pipe.fit(Xtr, ytr)

            te_idx = list(te)
            clean_s.append(f1_score(y[te], pipe.predict([docs_all[i] for i in te_idx]),
                                    average="macro"))
            prng = np.random.RandomState(M.SEED * 7777 + fi)
            Xte_p = [M._augment_text(docs_all[i], prng, "delete", eval_rate) for i in te_idx]
            pert_s.append(f1_score(y[te], pipe.predict(Xte_p), average="macro"))

        c, p = np.array(clean_s), np.array(pert_s)
        fold_cols[desc + "｜干净"] = c
        fold_cols[desc + "｜扰动"] = p
        rows.append({
            "增强方式": desc,
            "干净验证集": M.ms(c.mean(), c.std(ddof=1)),
            "扰动验证集": M.ms(p.mean(), p.std(ddof=1)),
            "降幅": f"{(c - p).mean():.4f}",
            "干净分": float(c.mean()),
            "扰动分": float(p.mean()),
            "实际替换词元比例": "—" if not changed else f"{np.mean(changed):.2%}",
            "耗时(s)": round(time.time() - t1, 1),
        })
        rep = "—" if not changed else f"{np.mean(changed):.2%}"
        print(f"  {desc:<16} 干净={c.mean():.4f}  扰动={p.mean():.4f}  "
              f"降幅={(c - p).mean():.4f}  实际替换率={rep}  ({time.time() - t1:.0f}s)")

    df = pd.DataFrame(rows)
    folds_df = pd.DataFrame(fold_cols)

    def show(title: str, tbl: pd.DataFrame) -> None:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        print(tbl.to_string(index=False))
        print("=" * 78)

    show("语义级增强 · 总表",
         df[["增强方式", "干净验证集", "扰动验证集", "降幅", "实际替换词元比例", "耗时(s)"]])

    base = "不加增强（基线）"
    clean_only = folds_df[[c for c in folds_df.columns if c.endswith("｜干净")]].rename(
        columns=lambda c: c.split("｜")[0])
    show("配对检验 A · 干净验证集（相对基线）",
         M.paired_tests(clean_only, baseline=base, n_splits=folds, n_repeats=repeats))

    drops = pd.DataFrame({
        c.split("｜")[0]: folds_df[c].to_numpy() - folds_df[c.split("｜")[0] + "｜扰动"].to_numpy()
        for c in folds_df.columns if c.endswith("｜干净")})
    show("配对检验 B · 降幅（负值 = 比基线更抗噪）",
         M.paired_tests(drops, baseline=base, n_splits=folds, n_repeats=repeats))

    # —— 与结构扰动做同折配对（折与种子完全相同，可直接配对）——
    old = RESULT_DIR / "augment_robust_fold_scores_5x5.csv"
    if old.exists() and folds == 5 and repeats == 5:
        oldf = pd.read_csv(old, index_col=0)
        sem = folds_df["近邻替换 10%｜干净"].to_numpy()
        cmp_rows = []
        for col, label in [("不做增强（基线）｜干净", "不做增强（基线）"),
                           ("随机删除 10% 词｜干净", "结构扰动：随机删除 10%")]:
            if col not in oldf.columns:
                continue
            other = oldf[col].to_numpy()
            d = sem - other
            _, p25 = stats.ttest_rel(sem, other)
            lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
            cmp_rows.append({
                "比较": f"近邻替换 10% − {label}",
                "Δ 均值": f"{d.mean():+.4f}",
                "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                "p（25 折）": f"{p25:.2e}",
                "结论": "不显著" if p25 >= 0.05 else "显著",
            })
        show("语义级 vs 结构扰动（同一批折，可直接配对）", pd.DataFrame(cmp_rows))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["干净分", "扰动分"]).to_csv(
        RESULT_DIR / f"semantic_aug_{tag}.csv", index=False, encoding="utf-8-sig")
    folds_df.to_csv(RESULT_DIR / f"semantic_aug_fold_scores_{tag}.csv", encoding="utf-8-sig")
    print(f"\n[记录] 已保存 results/semantic_aug_{tag}.csv 及其逐折分数")
    print(f"[耗时] 总计 {time.time() - t0:.1f}s")


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="语义级数据增强（LSA 词向量近邻替换）")
    ap.add_argument("--folds", type=int, default=M.N_SPLITS)
    ap.add_argument("--repeats", type=int, default=M.N_REPEATS)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(tag=args.tag or f"{args.folds}x{args.repeats}",
        folds=args.folds, repeats=args.repeats)


if __name__ == "__main__":
    main_cli()
