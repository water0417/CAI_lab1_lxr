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

# 模型 → 固定色位。颜色跟随**实体**而不是排名：同一个模型在任何一张图里都是同一个颜色，
# 换一张图、换一次排序，颜色都不会跳（否则读者会把颜色误当成"名次"来记）。
MODEL_COLORS = {
    "NaiveBayes": SERIES[0],
    "LogisticRegression": SERIES[1],
    "LinearSVC": SERIES[2],
    "MLP": SERIES[3],
}


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


# ==========================================================================
# 图 9~12 · 结果总览（供报告「结果与评估」章节用）
# ==========================================================================
def _pm(s) -> tuple[float, float]:
    """把 "0.7451 ± 0.0332" 拆成 (均值, 标准差)。

    learning_curve.csv 里只有"验证-MacroF1"带 ±，"泛化间隙"是纯浮点，
    所以没有 ± 时标准差记 0（画出来就是没有误差带）。
    """
    s = str(s)
    if "±" in s:
        a, b = s.split("±")
        return float(a.strip()), float(b.strip())
    return float(s), 0.0


def _mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """均值与 t 分布置信区间（与 fig1 同一口径）。"""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    m = x.mean()
    lo, hi = stats.t.interval(level, len(x) - 1, loc=m, scale=stats.sem(x))
    return m, lo, hi


def fig9_overview():
    """总览：**基线 → 最终**，四个模型同图。

    哑铃图（dumbbell）。灰点是旧配置、彩点是最终配置，中间连线带出"移动了多少"。
    用位置编码而不是柱长——两者的分数差 0.03 左右，柱状图配截断轴会失真，
    配从 0 起的轴又什么都看不出来。
    """
    base = _load("cv_5x5_raw.csv")                       # 原文 + 5000 维
    fin = _load("model_metrics_routing_best_5x5.csv")    # routing + 50000 维 + sublinear
    if base is None or fin is None:
        return
    b = dict(zip(base["Model"], base["macro_f1"]))
    f = dict(zip(fin["模型"], fin["MacroF1均值"]))
    models = sorted(f, key=f.get)                        # 升序 → 最好的画在最上面

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    y = np.arange(len(models))[::-1]
    for yy, m in zip(y, models):
        c = MODEL_COLORS[m]
        ax.plot([b[m], f[m]], [yy, yy], color=c, linewidth=2.2, alpha=0.42,
                solid_capstyle="round", zorder=2)
        ax.plot([b[m]], [yy], "o", color=BASELINE, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4)
        ax.plot([f[m]], [yy], "o", color=c, markersize=11.5,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.text(f[m] + 0.0014, yy, f"{f[m]:.4f}", va="center", fontsize=10.5,
                color=INK_PRIMARY, fontweight="bold")
        ax.text(f[m] + 0.0108, yy, f"+{f[m]-b[m]:.4f}", va="center", fontsize=10,
                color=c, fontweight="bold")
        ax.text(b[m] - 0.0014, yy, f"{b[m]:.4f}", va="center", ha="right",
                fontsize=9.5, color=INK_MUTED)

    lo, hi = min(b.values()), max(f.values())
    ax.set_xlim(lo - 0.008, hi + 0.019)
    ax.set_yticks(y)
    ax.set_yticklabels(models, color=INK_PRIMARY, fontsize=11)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Macro-F1（5 折 ×5 重复）")
    # 图例必须放在坐标区**外面**：放里面会压住最下面那行（MLP 的点与标签）。
    # 两个点的含义也不能只靠图例，正文里同时写明。
    ax.plot([], [], "o", color=BASELINE, markersize=9, label="基线：原文 + 5000 维")
    ax.plot([], [], "o", color=INK_MUTED, markersize=9,
            label="最终：删路由头 + 50000 维 + sublinear_tf")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2,
              fontsize=9.5, labelcolor=INK_SECONDARY)
    _title(ax, "最终成绩 vs 基线：四个模型都涨了，但涨幅不同",
           "右端数字为最终 Macro-F1，后面的 +x 为相对基线的提升")
    _save(fig, "fig9_overview.png")


def fig10_gain_waterfall():
    """增益分解瀑布：0.9156 →(清洗)→ 0.9189 →(TF-IDF 调参)→ 0.9434。

    三段分数**全部从逐折分数现算**（model_fold_scores_*），不手抄任何数字——
    手抄的值一旦和 CSV 对不上，图就成了新的错误来源。
    两个增量柱上画的是配对差的 95% CI（同一批折逐折做差），不是独立均值的区间。
    """
    raw = _load("model_fold_scores_raw_5x5.csv", index_col=0)
    mid = _load("model_fold_scores_routing_5x5.csv", index_col=0)
    fin = _load("model_fold_scores_routing_best_5x5.csv", index_col=0)
    if raw is None or mid is None or fin is None:
        return
    s_raw, s_mid, s_fin = raw["LinearSVC"], mid["LinearSVC"], fin["LinearSVC"]
    lv = [s_raw.mean(), s_mid.mean(), s_fin.mean()]
    d1, d1lo, d1hi = _mean_ci((s_mid - s_raw).to_numpy())
    d2, d2lo, d2hi = _mean_ci((s_fin - s_mid).to_numpy())

    floor = lv[0] - 0.005
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    xs = np.arange(4)
    # 两端的"锚"用中性灰，中间的增量用蓝——颜色区分的是**性质**（水平 vs 增量）
    ax.bar(0, lv[0] - floor, bottom=floor, width=0.55, color=BASELINE, zorder=3)
    ax.bar(1, d1, bottom=lv[0], width=0.55, color=POS, zorder=3)
    ax.bar(2, d2, bottom=lv[1], width=0.55, color=POS, zorder=3)
    ax.bar(3, lv[2] - floor, bottom=floor, width=0.55, color=INK_SECONDARY, zorder=3)
    for x, (lo, hi) in ((1, (d1lo, d1hi)), (2, (d2lo, d2hi))):
        ax.plot([x, x], [lv[x - 1] + lo, lv[x - 1] + hi], color=INK_PRIMARY,
                linewidth=1.4, zorder=4)
        ax.plot([x - 0.09, x + 0.09], [lv[x - 1] + lo] * 2, color=INK_PRIMARY,
                linewidth=1.4, zorder=4)
        ax.plot([x - 0.09, x + 0.09], [lv[x - 1] + hi] * 2, color=INK_PRIMARY,
                linewidth=1.4, zorder=4)
    # 虚线把每级台阶接到下一根柱；**数值标签就贴在虚线上**。
    # 早先标签放在柱顶，与增量柱的 95% CI 竖线叠在一起（渲染出来才发现）——
    # 挪到虚线上既不挡竖线，也不挡柱顶。
    for x, lvl in ((0, lv[0]), (1, lv[1]), (2, lv[2])):
        ax.plot([x + 0.28, x + 0.72], [lvl, lvl], color=INK_MUTED,
                linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.text(x + 0.5, lvl + 0.0005, f"{lvl:.4f}", ha="center", va="bottom",
                fontsize=10.5, color=INK_PRIMARY, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                          edgecolor="none"))
    for x, hi in ((1, d1hi), (2, d2hi)):
        ax.text(x, lv[x - 1] + hi + 0.0022, f"+{[d1, d2][x - 1]:.4f}", ha="center",
                va="bottom", fontsize=11, color=POS, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(["基线\n原文 + 5000 维", "+ 删路由头\n（留 Subject）",
                        "+ TF-IDF 调参\n50000 维 + sublinear", "最终\n交付配置"],
                       fontsize=10)
    ax.set_ylim(floor, lv[2] + 0.006)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Macro-F1（LinearSVC，25 折）")
    _title(ax, "最终成绩是两步挣来的，其中九成来自 TF-IDF 调参",
           "纵轴从 %.3f 起（只影响柱高观感，标注的都是真实数值）· 增量柱上的竖线为配对差的 95%% CI"
           % floor)
    _save(fig, "fig10_gain_waterfall.png")


def fig11_learning_curve():
    """学习曲线：左=验证分，右=泛化间隙，横轴都是训练样本量。

    拆成两个子图而不是画双纵轴——双纵轴（两个 y 刻度）是最常见的图表错误，
    两条线的交点会被读成"相等"，而它没有意义。
    """
    d = _load("learning_curve.csv")
    if d is None:
        return
    sizes = sorted(d["训练样本量"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    for m in MODEL_COLORS:
        sub = d[d["模型"] == m].set_index("训练样本量").reindex(sizes)
        val = np.array([_pm(v) for v in sub["验证-MacroF1"]])
        gap = np.array([_pm(v) for v in sub["泛化间隙"]])
        c = MODEL_COLORS[m]
        for ax, arr, ylab in ((axes[0], val, "验证 Macro-F1"),
                              (axes[1], gap, "泛化间隙（训练分 − 验证分）")):
            ax.plot(sizes, arr[:, 0], color=c, linewidth=2, marker="o",
                    markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=m, zorder=3)
            ax.fill_between(sizes, arr[:, 0] - arr[:, 1], arr[:, 0] + arr[:, 1],
                            color=c, alpha=0.13, linewidth=0, zorder=2)
            ax.set_xlabel("训练样本量")
            ax.set_ylabel(ylab)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes])
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=9.5, labelcolor=INK_PRIMARY, ncols=2)
    _title(axes[0], "验证分：数据越多越好，但 MLP 的追赶最猛",
           "阴影为 5 折的标准差")
    _title(axes[1], "泛化间隙：随数据量单调下降",
           "600 条时 MLP 过拟合最重，全量时反而最小")
    _save(fig, "fig11_learning_curve.png")


def fig12_model_cost():
    """成本：5000 维 → 50000 维的单折拟合耗时。

    横轴是**对数**轴，因为四个模型跨越四个数量级（0.01s ~ 283s），线性轴会把
    除 MLP 外的三个压成一条线。用点而不是柱——点用位置编码，对数轴上不构成失真；
    每个点都直接标了秒数，信息不靠颜色或长度单独承载。
    """
    d = _load("model_cost_single_fold.csv")
    if d is None:
        return
    if "模型" in d.columns:
        d = d.set_index("模型")
    cols = [c for c in d.columns if "维" in c]
    c5, c50 = cols[0], cols[1]
    models = ["NaiveBayes", "LogisticRegression", "LinearSVC", "MLP"]
    models = [m for m in models if m in d.index]

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    y = np.arange(len(models))[::-1]
    ax.set_xscale("log")
    for yy, m in zip(y, models):
        v5, v50 = float(d.loc[m, c5]), float(d.loc[m, c50])
        c = MODEL_COLORS[m]
        ax.plot([v5, v50], [yy, yy], color=c, linewidth=2.2, alpha=0.42,
                solid_capstyle="round", zorder=2)
        ax.plot([v5], [yy], "o", color=BASELINE, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4)
        ax.plot([v50], [yy], "o", color=c, markersize=11.5,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        # 标签一律放在**远离另一个点**的那一侧。LinearSVC 的 50000 维反而更快，
        # 它的点落在基线点左边——固定"标签往右"会正好压到另一个点上。
        to_right = v50 >= v5
        ax.annotate(f"{v50:.2f}s", (v50, yy), textcoords="offset points",
                    xytext=(11 if to_right else -11, 0), va="center",
                    ha="left" if to_right else "right", fontsize=10.5,
                    color=INK_PRIMARY, fontweight="bold")
        ax.annotate(f"{v5:.2f}s", (v5, yy), textcoords="offset points",
                    xytext=(-11 if to_right else 11, 0), va="center",
                    ha="right" if to_right else "left", fontsize=9.5,
                    color=INK_MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(models, color=INK_PRIMARY, fontsize=11)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlim(0.003, 700)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("单折拟合耗时（秒，对数轴）")
    ax.plot([], [], "o", color=BASELINE, markersize=9, label="5000 维（基线）")
    ax.plot([], [], "o", color=INK_MUTED, markersize=9, label="50000 维（最优）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2,
              fontsize=9.5, labelcolor=INK_SECONDARY)
    _title(ax, "只有 MLP 的成本随维度爆炸：慢两个数量级",
           "同一折、同一进程、背靠背测量；重复测量本身有 1.5~2.5 倍抖动，故只看量级")
    _save(fig, "fig12_model_cost.png")


FIGS = {
    "fig1": fig1_model_comparison,    "fig2": fig2_cleaning_ladder,
    "fig3": fig3_cleaning_forest,
    "fig4": fig4_max_features,
    "fig5": fig5_model_hyperparams,
    "fig6": fig6_tfidf_forest,
    "fig7": fig7_confusion_matrix,
    "fig8": fig8_per_class_f1,
    "fig9": fig9_overview,
    "fig10": fig10_gain_waterfall,
    "fig11": fig11_learning_curve,
    "fig12": fig12_model_cost,
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
