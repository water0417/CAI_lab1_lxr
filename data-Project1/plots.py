# -*- coding: utf-8 -*-
"""
实验一：步骤 5 · 可视化

从 results/ 下已保存的结果 CSV 生成报告用图，全部输出到 figs/。

设计约定（见 README「图表规范」）：
  * 配色取自验证过的分类调色板，按固定槽位顺序取用，**不循环**；
    4 色相邻对已通过 CVD 与正常视觉分离度检验。
  * 柱状条上**直接标数值**：调色板中 aqua / yellow 两色对底色的对比度低于 3:1，
    按"补偿规则"必须用可见标签补齐，不能只靠颜色传达信息。
  * 极简网格与坐标轴，墨色使用中性色，**数值与标签不着系列色**。
  * 顺序编码（混淆矩阵）只用单一蓝色由浅到深，绝不用彩虹色。

用法：
    cd data-Project1
    python plots.py                # 生成全部图
    python plots.py --only fig4    # 只生成某一张
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import main as M

FIG_DIR = M.FIG_DIR
RESULT_DIR = M.RESULT_DIR

# ---- 调色板（light 模式；已通过 validate_palette.js 六项检查）----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
POS, NEG = "#2a78d6", "#e34948"          # 发散配色：蓝 ↔ 红
# 单一蓝色由浅到深（顺序编码，用于混淆矩阵）
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _load(name: str, **kw) -> pd.DataFrame | None:
    p = RESULT_DIR / name
    if not p.exists():
        print(f"  [跳过] 缺少 {name}")
        return None
    return pd.read_csv(p, **kw)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  [输出] figs/{name}")


def _title(ax, title: str, subtitle: str | None = None) -> None:
    """标题 + 副标题。

    副标题放在坐标区上方 1.5% 处，标题则靠 pad 抬到它上面——pad 必须给足，
    否则标题会压在副标题上（前者默认贴坐标区顶边）。
    """
    ax.set_title(title, loc="left", pad=30 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=9.5, color=INK_SECONDARY, va="bottom")


# 清洗配置描述的短标签。完整描述太长会把坐标区挤变形，报告里也用不上全称。
SHORT_LABELS = {
    "保留全部邮件头": "H0 保留全部邮件头",
    "删路由头(留Subject)": "H1 删路由头（留 Subject）",
    "邮件头全删": "H2 邮件头全删",
    "保留全部邮件头 + 去引用": "I2 仅去引用",
    "保留全部邮件头 + 去签名": "I3 仅去签名",
    "保留全部邮件头 + 去标点": "I4 仅去标点",
    "保留全部邮件头 + 去停用词": "I5 仅去停用词",
    "保留全部邮件头 + stemming": "I6 仅 stemming",
    "删路由头(留Subject) + 去引用": "L2 + 去引用",
    "删路由头(留Subject) + 去引用 + 去签名": "L3 + 去签名",
    "删路由头(留Subject) + 去引用 + 去签名 + 去标点": "L4 + 去标点",
    "删路由头(留Subject) + 去引用 + 去签名 + 去标点 + 去停用词": "L5 + 去停用词",
    "删路由头(留Subject) + 去引用 + 去签名 + 去标点 + 去停用词 + stemming": "L6 + stemming",
}


def short_label(desc: str) -> str:
    return SHORT_LABELS.get(desc, desc)


# ==========================================================================
# 图 1 · 最优配置下的四模型对比
# ==========================================================================
def fig1_model_comparison():
    """四模型对比画**点区间图**而不是柱状图。

    原因：柱状图用"长度"编码数值，纵轴一旦不从 0 起就会放大差异，是经典反模式；
    但若从 0 起，0.94 与 0.92 的差别又完全看不出来。点区间图用"位置"编码，
    截断坐标轴不构成失真，同时还能把 95% CI 直接画出来。与 fig3/fig6 的森林图同一套视觉语言。
    """
    df = _load("model_fold_scores_routing_best_5x5.csv", index_col=0)
    if df is None:
        return
    from scipy import stats
    means = df.mean().sort_values()
    ci = [stats.t.interval(0.95, len(df) - 1, loc=means[m], scale=stats.sem(df[m]))
          for m in means.index]
    lo = np.array([c[0] for c in ci])
    hi = np.array([c[1] for c in ci])

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(means))[::-1]
    for i, (yy, m, l, h) in enumerate(zip(y, means.values, lo, hi)):
        c = SERIES[i % len(SERIES)]
        ax.plot([l, h], [yy, yy], color=c, linewidth=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot([m], [yy], "o", color=c, markersize=11,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.text(h + (hi.max() - lo.min()) * 0.035, yy, f"{m:.4f}", va="center",
                fontsize=11, color=INK_PRIMARY, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(means.index, color=INK_PRIMARY, fontsize=11)
    ax.set_xlim(lo.min() - (hi.max() - lo.min()) * 0.10,
                hi.max() + (hi.max() - lo.min()) * 0.16)
    ax.set_xlabel("Macro-F1（5 折 ×5 重复，横线为 95% 置信区间）")
    ax.set_ylim(-0.6, len(means) - 0.4)
    ax.grid(axis="y", visible=False)
    _title(ax, "四个模型在最优配置下的对比",
           "前提 routing · TF-IDF max_features=50000 + sublinear_tf · LinearSVC C=1 · "
           "MLP 与 LinearSVC 的区间重叠，差异不显著")
    _save(fig, "fig1_model_comparison.png")


# ==========================================================================
# 图 2 · 文本清洗的累积阶梯
# ==========================================================================
def fig2_cleaning_ladder():
    """阶梯画的是**相对基线的变化量**，不是绝对分数。

    绝对分数的差异只有 0.006~0.031，画在 0.86~0.94 的纵轴上基本看不出高低；
    直接画 Δ 才能让"每加一步就往下走"这个结论一眼可见。
    绝对分数用次要标注补在 Δ 下方，两者都不丢。
    """
    df = _load("ablation_clean_5x5.csv")
    if df is None:
        return
    ladder = M.ablation_groups()["累积阶梯（逐级叠加，每级只多一个因素）"]
    labels = [short_label(desc) for _, cfg in ladder
              for desc in [M.describe(cfg)]]
    vals = []
    for _, cfg in ladder:
        row = df[df["配置"] == M.describe(cfg)]
        vals.append(float(row["MacroF1均值"].iloc[0]))
    vals = np.array(vals)
    base = vals[0]
    delta = vals - base

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(vals))
    colors = [SERIES[0] if d >= 0 else SERIES[7] for d in delta]
    ax.bar(x, delta, width=0.6, color=colors, zorder=3)
    for xi, v, d in zip(x, vals, delta):
        off = 0.0022 if d >= 0 else -0.0022
        va = "bottom" if d >= 0 else "top"
        ax.text(xi, d + off, f"{d:+.4f}", ha="center", va=va,
                fontsize=10, color=INK_PRIMARY, fontweight="bold")
        ax.text(xi, d + (0.0068 if d >= 0 else -0.0068), f"({v:.4f})",
                ha="center", va=va, fontsize=9, color=INK_SECONDARY)

    ax.axhline(0, color=BASELINE, linewidth=1.6, zorder=2)
    ax.set_ylim(min(delta) - 0.020, max(delta) + 0.017)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("Macro-F1 相对基线的变化 Δ")
    ax.grid(axis="x", visible=False)
    _title(ax, "文本清洗的累积阶梯：每加一步，分数就往下走",
           "前提 raw 起步 · LinearSVC · max_features=5000 · 括号内为绝对分数，25 折")
    _save(fig, "fig2_cleaning_ladder.png")


# ==========================================================================
# 图 3 · 清洗消融的效应森林图
# ==========================================================================
def _forest(ax, labels, deltas, los, his, highlight: set[int] | None = None,
            xlabel: str = "Macro-F1 变化（Δ）"):
    """横向森林图：每行一个配置，点估计 + 95% CI。

    highlight 里的下标用实心强调色，其余用浅色——把"显著"从"不显著"里分出来，
    但两类都保留形状与标签，不靠颜色单独承载信息。
    """
    y = np.arange(len(labels))[::-1]
    for i, (yy, d, lo, hi) in enumerate(zip(y, deltas, los, his)):
        significant = highlight is None or i in highlight
        color = (POS if d >= 0 else NEG) if significant else "#c9c8c2"
        ax.plot([lo, hi], [yy, yy], color=color, linewidth=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot([d], [yy], "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.text(hi + abs(hi - lo) * 0.06 + 0.0006, yy, f"{d:+.4f}", va="center",
                fontsize=9.5, color=INK_PRIMARY, fontweight="bold" if significant else "normal")
    ax.axvline(0, color=BASELINE, linewidth=1.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK_PRIMARY, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def fig3_cleaning_forest():
    df = _load("ablation_clean_5x5.csv")
    tests = _load("paired_tests_5x5.csv")
    if df is None or tests is None:
        return
    tests = tests.set_index("配置")
    labels, deltas, los, his, sig = [], [], [], [], set()
    for _, r in df.iterrows():
        name = r["配置"]
        if name == "保留全部邮件头":
            continue
        t = tests.loc[name]
        if t["Δ 均值"] == "—":
            continue
        d = float(t["Δ 均值"])
        lo, hi = [float(v) for v in t["Δ 的 95% CI"].strip("[]").split(",")]
        labels.append(short_label(name))
        deltas.append(d)
        los.append(lo)
        his.append(hi)
        if t["结论"] in ("显著", "显著（仅单层检验）"):
            sig.add(len(labels) - 1)

    order = np.argsort(deltas)
    labels = [labels[i] for i in order]
    deltas = [deltas[i] for i in order]
    los = [los[i] for i in order]
    his = [his[i] for i in order]
    sig = {i for i, o in enumerate(order) if o in sig}

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    _forest(ax, labels, deltas, los, his, sig)
    _title(ax, "各清洗配置相对原文基线的效应（配对检验）",
           "实心 = 统计显著，灰色 = 不显著（95% CI 跨 0）· LinearSVC · 25 折配对 t 检验")
    _save(fig, "fig3_cleaning_forest.png")


# ==========================================================================
# 图 4 · max_features 曲线
# ==========================================================================
def fig4_max_features():
    """max_features 曲线。

    每个维度取**该维度下可得的最高分**：2000 与 20000 只有"单独改 max_features"的
    结果；5000 之外还测过 sublinear_tf 的组合，这里取其中的较高者，
    以免"因为没配 sublinear 而显得低"——那是配置差异，不是维度差异。
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ticks = ["2000", "5000", "20000", "50000", "100000", "全词表\n107,882"]
    picks = {
        2000: ["max_features=2000"],
        5000: ["基线（全部默认）", "sublinear_tf=True"],
        20000: ["max_features=20000"],
        50000: ["max_features=50000、sublinear_tf=True"],
        100000: ["max_features=100000、sublinear_tf=True"],
        107882: ["max_features=None、sublinear_tf=True"],
    }
    xi = np.arange(len(ticks))
    series = []
    for i, p in enumerate(["raw", "routing"]):
        df = _load(f"tfidf_sweep_{p}_5x5.csv")
        if df is None:
            return
        vals = []
        for x, keys in picks.items():
            cand = [float(df[df["参数"] == k]["MacroF1均值"].iloc[0])
                    for k in keys if (df["参数"] == k).any()]
            vals.append(max(cand))
        series.append((p, np.array(vals)))
        ax.plot(xi, vals, "-o", color=SERIES[i], linewidth=2, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label="原文（raw）" if p == "raw" else "删路由头（routing）")
        ax.annotate(f"{vals[-1]:.4f}", (xi[-1], vals[-1]), textcoords="offset points",
                    xytext=(-8, 12 if i == 0 else -20), ha="right",
                    fontsize=9.5, color=INK_PRIMARY, fontweight="bold")
    ax.set_xticks(xi)
    ax.set_xticklabels(ticks, fontsize=9.5)
    ax.set_xlabel("TF-IDF 词表上限 max_features")
    ax.set_ylabel("Macro-F1（25 折）")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(min(min(v) for _, v in series) - 0.006,
                max(max(v) for _, v in series) + 0.006)
    _title(ax, "max_features 越多越好，直到用尽全词表才饱和",
           "LinearSVC · 每个维度取该维度下可得的最高分 · 50000 之后收益已不足 0.001")
    _save(fig, "fig4_max_features.png")


# ==========================================================================
# 图 5 · 模型超参曲线
# ==========================================================================
def fig5_model_hyperparams():
    df = _load(f"model_sweep_{M.BEST_PRESET}_5x5.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # 左：LinearSVC 的 C
    ax = axes[0]
    sub = df[df["模型"] == "LinearSVC"]
    cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    vals = [float(sub[sub["超参"] == f"C={c}"]["MacroF1均值"].iloc[0]) for c in cs]
    stds = [float(sub[sub["超参"] == f"C={c}"]["MacroF1标准差"].iloc[0]) for c in cs]
    xi = np.arange(len(cs))
    ax.errorbar(xi, vals, yerr=stds, fmt="-o", color=SERIES[0], linewidth=2,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                ecolor=INK_MUTED, capsize=5, zorder=3)
    best = int(np.argmax(vals))
    ax.plot(xi[best], vals[best], "o", color=SERIES[0], markersize=13,
            markeredgecolor=SURFACE, markeredgewidth=2.5, zorder=4)
    # 标注抬到误差棒之上，否则会被误差棒盖住
    top = max(v + s for v, s in zip(vals, stds))
    ax.set_ylim(min(v - s for v, s in zip(vals, stds)) - 0.004, top + 0.010)
    ax.annotate(f"最优 C={cs[best]}\n{vals[best]:.4f}", (xi[best], vals[best]),
                textcoords="offset points", xytext=(0, 34), ha="center",
                fontsize=9.5, color=INK_PRIMARY, fontweight="bold")
    ax.set_xticks(xi)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("LinearSVC 的 C（惩罚力度，越大正则越弱）")
    ax.set_ylabel("Macro-F1（25 折）")
    _title(ax, "LinearSVC：C 呈 U 形", "纵轴仅覆盖 0.90~0.95，为放大 C 的影响")

    # 右：LR 的 C × penalty
    ax = axes[1]
    sub = df[df["模型"] == "LogisticRegression"]
    for i, pen in enumerate(["l2", "l1"]):
        vals = [float(sub[sub["超参"] == f"C={c}、penalty={pen}"]["MacroF1均值"].iloc[0])
                for c in cs]
        ax.plot(xi, vals, "-o", color=SERIES[i], linewidth=2, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label=f"penalty={pen}")
        ax.annotate(f"{vals[-1]:.4f}", (xi[-1], vals[-1]), textcoords="offset points",
                    xytext=(-6, 10 if i == 0 else -18), ha="right",
                    fontsize=9.5, color=INK_PRIMARY, fontweight="bold")
    l1 = [float(sub[sub["超参"] == f"C={c}、penalty=l1"]["MacroF1均值"].iloc[0]) for c in cs]
    ax.annotate(f"L1 在 C=0.01 塌到 {l1[0]:.4f}\n（10 类随机猜是 0.10）", (xi[0], l1[0]),
                textcoords="offset points", xytext=(24, 26), ha="left",
                fontsize=9, color=NEG, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=NEG, linewidth=1.4))
    ax.set_xticks(xi)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("LogisticRegression 的 C（正则强度的倒数）")
    ax.set_ylabel("Macro-F1（25 折）")
    ax.legend(loc="lower right", fontsize=10)
    _title(ax, "LR：L1 全区间劣于 L2",
           "纵轴覆盖 0~0.95 以容纳 L1 的塌陷，故与左图不可比陡峭度")

    fig.tight_layout(w_pad=3.5)
    _save(fig, "fig5_model_hyperparams.png")


# ==========================================================================
# 图 6 · TF-IDF 参数效应森林图（双前提）
# ==========================================================================
def fig6_tfidf_forest():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    order = ["max_features=20000", "max_features=2000", "ngram_range=(1, 2)",
             "sublinear_tf=True", "min_df=10", "min_df=5", "min_df=2",
             "max_df=0.9", "max_df=0.5"]
    for ax, p in zip(axes, ["raw", "routing"]):
        t = _load(f"tfidf_paired_tests_{p}_5x5.csv")
        if t is None:
            return
        t = t.set_index("配置")
        labels, deltas, los, his, sig = [], [], [], [], set()
        for k in order:
            if k not in t.index:
                continue
            r = t.loc[k]
            if r["Δ 均值"] == "—":
                continue
            d = float(r["Δ 均值"])
            lo, hi = [float(v) for v in r["Δ 的 95% CI"].strip("[]").split(",")]
            labels.append(k)
            deltas.append(d); los.append(lo); his.append(hi)
            if r["结论"] in ("显著", "显著（仅单层检验）"):
                sig.add(len(labels) - 1)
        _forest(ax, labels, deltas, los, his, sig)
        ax.set_title(f"前提 {p}（{'原文' if p == 'raw' else '删路由头'}）",
                     loc="left", fontsize=11.5, pad=8)
        ax.set_xlim(-0.052, 0.030)
    axes[0].set_ylabel("")
    fig.suptitle("TF-IDF 各参数的效应（相对默认值，配对检验）",
                 x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK_PRIMARY)
    fig.text(0.012, 0.925, "实心 = 统计显著，灰色 = 不显著（95% CI 跨 0）· LinearSVC · 25 折配对 t 检验",
             ha="left", fontsize=9.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3.5)
    _save(fig, "fig6_tfidf_forest.png")


# ==========================================================================
# 折外预测（fig7 混淆矩阵与 fig8 逐类别 F1 共用，只算一次）
# ==========================================================================
# 语料是 20 Newsgroups 风格的新闻组帖子，助教未给类别名。
# 下面这些名字是**从各类别 TF-IDF 高权重词反推**出来的，仅用于让图好读；
# 举例：类别 2 的 top 词是 drive/card/ide/scsi/controller → "PC 硬件"。
CLASS_NAMES = {
    0: "图形/3D", 1: "Windows/DOS", 2: "PC 硬件", 3: "汽车", 4: "摩托车",
    5: "棒球", 6: "加密/密码学", 7: "电子电路", 8: "医学", 9: "中东政治",
}
# 混淆矩阵显示出的高混淆簇：这四个类别的帖子词汇高度重叠
COMPUTER_CLUSTER = {0, 1, 2, 7}

_OOF: dict = {}


def oof_predictions(n_splits: int = 5):
    """用最优配置跑一次 5 折交叉验证，返回折外预测。

    折外预测是错误分析的正确做法：每个样本的预测都来自没见过它的模型，
    因此逐类别指标不会因为"模型见过这条样本"而虚高。
    """
    if n_splits in _OOF:
        return _OOF[n_splits]
    from sklearn.model_selection import StratifiedKFold

    X, y = M.load_train()
    docs = M.apply_clean(X, M.CLEAN_PRESETS[M.BEST_PRESET])
    pipe = M.make_pipeline(**M.BEST_TFIDF)
    pipe.set_params(clf=M.LinearSVC(random_state=M.SEED, C=1.0))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=M.SEED)

    y_pred = np.empty_like(y)
    print(f"  [计算] {n_splits} 折折外预测…")
    for tr, te in cv.split(docs, y):
        pipe.fit([docs[i] for i in tr], y[tr])
        y_pred[te] = pipe.predict([docs[i] for i in te])
    _OOF[n_splits] = (y, y_pred)
    return y, y_pred


# ==========================================================================
# 图 7 · 混淆矩阵
# ==========================================================================
def fig7_confusion_matrix(n_splits: int = 5):
    from sklearn.metrics import confusion_matrix

    y, y_pred = oof_predictions(n_splits)
    cm = confusion_matrix(y, y_pred, labels=range(M.N_CLASSES))
    cmn = cm / cm.sum(axis=1, keepdims=True)      # 行归一化：每类的召回率视角

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    cmap = LinearSegmentedColormap.from_list("blues", [SURFACE] + BLUE_RAMP)
    im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(M.N_CLASSES):
        for j in range(M.N_CLASSES):
            v = cmn[i, j]
            if v == 0:
                continue
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 0.55 else INK_PRIMARY,
                    fontweight="bold" if i == j else "normal")
    labels = [f"{c} {CLASS_NAMES[c]}" for c in range(M.N_CLASSES)]
    ax.set_xticks(range(M.N_CLASSES))
    ax.set_yticks(range(M.N_CLASSES))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("该类样本被判给各类的比例", color=INK_SECONDARY, fontsize=9.5)
    cb.outline.set_edgecolor(BASELINE)
    acc = (y_pred == y).mean()
    _title(ax, "最优模型的混淆矩阵（5 折折外预测）",
           f"LinearSVC C=1 · {M.describe(M.CLEAN_PRESETS[M.BEST_PRESET])} · "
           f"max_features=50000 + sublinear_tf · 折外准确率 {acc:.4f}")
    _save(fig, "fig7_confusion_matrix.png")


# ==========================================================================
# 图 8 · 逐类别 F1
# ==========================================================================
def fig8_per_class_f1(n_splits: int = 5):
    """逐类别 F1 横向条形图。

    这里用条形是合适的：要读的是"哪些类比其它类差得多"，条形比点更为直观；
    而且纵轴不是截断的连续轴，而是 10 个独立类别，不存在截断失真的问题。
    高混淆簇用强调色标出，其余用中性灰——颜色承载的是"是否属于同一词汇簇"，
    不是排名。
    """
    from sklearn.metrics import f1_score

    y, y_pred = oof_predictions(n_splits)
    f1 = {c: f1_score(y == c, y_pred == c) for c in range(M.N_CLASSES)}
    order = sorted(f1, key=lambda c: f1[c])
    vals = [f1[c] for c in order]
    names = [f"{c} {CLASS_NAMES[c]}" for c in order]
    colors = [NEG if c in COMPUTER_CLUSTER else "#c9c8c2" for c in order]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    yy = np.arange(len(order))[::-1]
    ax.barh(yy, vals, height=0.62, color=colors, zorder=3)
    for yv, v in zip(yy, vals):
        ax.text(v + 0.004, yv, f"{v:.4f}", va="center", fontsize=10,
                color=INK_PRIMARY, fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=10.5, color=INK_PRIMARY)
    ax.set_xlim(0.80, 1.02)
    ax.set_xlabel("F1（5 折折外预测）")
    ax.grid(axis="y", visible=False)
    ax.axvline(min(vals), color=BASELINE, linestyle="--", linewidth=1.2, zorder=2)
    _title(ax, "错误集中在计算机相关的四个类别上",
           "红色 = 图形/3D、Windows、PC 硬件、电子电路——四个类的词汇高度重叠，两两互相混淆；"
           "灰色类别的错误率低得多")
    _save(fig, "fig8_per_class_f1.png")


FIGS = {
    "fig1": fig1_model_comparison,
    "fig2": fig2_cleaning_ladder,
    "fig3": fig3_cleaning_forest,
    "fig4": fig4_max_features,
    "fig5": fig5_model_hyperparams,
    "fig6": fig6_tfidf_forest,
    "fig7": fig7_confusion_matrix,
    "fig8": fig8_per_class_f1,
}


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="实验一 步骤 5：生成报告图表")
    ap.add_argument("--only", nargs="+", choices=list(FIGS), default=None,
                    help="只生成指定的图，例如 --only fig1 fig4")
    args = ap.parse_args()

    setup_style()
    targets = args.only or list(FIGS)
    print(f"[绘图] 输出目录 {FIG_DIR}")
    for k in targets:
        print(f"\n[{k}]")
        try:
            FIGS[k]()
        except Exception as e:                      # 单张失败不影响其余
            print(f"  [失败] {k}: {type(e).__name__}: {e}")
    print(f"\n[完成] 共处理 {len(targets)} 张图")


if __name__ == "__main__":
    main_cli()
