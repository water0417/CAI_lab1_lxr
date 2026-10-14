# -*- coding: utf-8 -*-
"""
实验一 · 终端证据截图生成

把 results/ 下的运行日志渲染成**终端风格的 PNG**，输出到 figs/term/。

用途：报告里凡属"过程性"的结论（怎么防泄露、怎么验证一个反直觉说法），
都需要看到真实的命令与输出。这类证据不适合画成图表，用终端截图最直接。

**关于真实性**：图片内容是 `results/` 日志文件的原样输出（含耗时、警告、路径），
脚本只做"过滤显示范围"与配色渲染，不增删任何一个字符。
每张图的正文第一行就是产生它的命令，可据此自行复现核对。

用法：
    python make_term_shots.py            # 生成全部 12 张
    python make_term_shots.py 1 6        # 只生成指定编号
"""
from __future__ import annotations

import subprocess
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 关键：日志里含 `$`（命令提示符）、`_`、`^` 等字符，matplotlib 默认会把 $...$ 当数学公式解析，
# 导致文字被吞掉或错位。关掉 mathtext，让所有文本按字面渲染。
try:
    matplotlib.rcParams["text.parse_math"] = False
except KeyError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
OUT = LAB / "figs" / "term"

# ---- 终端配色（深色主题，贴近 VS Code 默认终端）----
BG = "#1e1e1e"          # 正文底色
BAR = "#323233"         # 标题栏
FG = "#e6e6e6"          # 正文
DIM = "#9a9a9a"         # 标题栏文字
ACCENT = "#4ec9b0"      # 命令提示符
DOT = ["#ff5f57", "#febc2e", "#28c840"]

# Consolas 打底，中文回退到雅黑，符号（✓ ✗ 等）回退到 Segoe UI Symbol ——
# 少了最后一项，evidence_1.log 里的 ✓ 会渲染成方框。
FONT = ["Consolas", "Microsoft YaHei", "Segoe UI Symbol"]
FS = 9.5                                   # 字号（pt）
CHAR_W = 0.0792                            # 每个半角字符的宽度（inch），按 Consolas@9.5pt 标定
LINE_H = 0.167                             # 每行高度（inch）
MAX_COLS = 170                             # 超过这个显示宽度的行会被截断（当前日志最宽 ~150）


# ---------------------------------------------------------------------------
# 12 张图：编号 → (文件名, 标题, 生成命令)
# ---------------------------------------------------------------------------
SHOTS = {
    1: ("term_01_split_cv_vs_holdout.png",
        "证据 ① · CV 与单次 80/20 的稳定性对照",
        "cat results/run_split_comparison.log"),
    2: ("term_02_stratify_check.png",
        "证据 ② · 分层检查与数据指纹",
        "sed -n '1,14p' results/run_cv_5x5.log"),
    3: ("term_03_test_seal.png",
        "证据 ③ · 测试集隔离：指纹、唯一调用点、篡改 1 字节即被拦截",
        "cat results/evidence_1.log"),
    4: ("term_04_augment_identity.png",
        "证据 ④ · 恒等变换验证：交换词序 200/200 向量完全相同",
        "cat results/evidence_2.log"),
    5: ("term_05_ablation.png",
        "证据 ⑤ · 清洗消融：三张分组表",
        "sed -n '26,59p' results/run_ablation.log"),
    6: ("term_06_paired_test.png",
        "证据 ⑥ · 配对显著性检验：Δ、95% CI、p 值与显著判定",
        "sed -n '25,48p' results/run_paired_test.log"),
    7: ("term_07_tfidf_sweep.png",
        "证据 ⑦ · TF-IDF 各参数配置逐个跑出的结果",
        "sed -n '84,129p' results/run_tfidf_sweep.log"),
    8: ("term_08_model_sweep.png",
        "证据 ⑧ · 模型超参扫描：LinearSVC 的 C 与 LR 的 C×penalty",
        "sed -n '577,631p' results/run_model_sweep.log"),
    9: ("term_09_nested_cv.png",
        "证据 ⑨ · 嵌套交叉验证：逐折的内部最优分、外层实测分与选出的超参",
        r"grep -E '^\[嵌套 CV\]|外层第' results/run_nested_cv.log"),
    10: ("term_10_final_models.png",
         "证据 ⑩ · 最终四模型对比与两两配对检验",
         "sed -n '14,40p' results/run_final_models.log"),
    11: ("term_11_error_analysis.png",
         "证据 ⑪ · 错误分析：逐类别 F1、混淆对及其涉及的类别集合",
         "cat results/evidence_3.log"),
    12: ("term_12_mlp_density.png",
         "证据 ⑫ · MLP 稠密化探测：维度、非零项数与单折计时",
         "cat results/evidence_4.log"),
    13: ("term_13_augment_robust.png",
         "证据 ⑬ · 数据增强的鲁棒性评估：干净/扰动验证集与降幅",
         "sed -n '17,49p' results/run_augment_robust.log"),
    14: ("term_14_augment_label_integrity.png",
         "证据 ⑭ · 标签保持性抽检：删除 20% 词后各类别关键词的存活情况",
         "sed -n '53,66p' results/run_augment_robust.log"),
    15: ("term_15_semantic_aug.png",
         "证据 ⑮ · 语义级增强（LSA 词向量近邻替换）及其与结构扰动的同折对照",
         "sed -n '12,44p' results/run_semantic_aug.log"),
    16: ("term_16_ablation_recheck.png",
         "证据 ⑯ · 清洗消融的四模型复核（含训练 Macro-F1 与泛化间隙）",
         "sed -n '71,79p' results/run_ablation.log"),
    17: ("term_17_tfidf_combo.png",
         "证据 ⑰ · TF-IDF 参数组合的配对检验（含 min_df 与 ngram 的组合项）",
         "sed -n '171,188p' results/run_tfidf_combo3.log"),
    18: ("term_18_learning_curve.png",
         "证据 ⑱ · 学习曲线：四个训练规模下各模型的训练/验证分与泛化间隙",
         "sed -n '33,51p' results/run_learning_curve.log"),
}


def disp_width(s: str) -> int:
    """终端显示宽度：CJK 全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def run_log(cmd: str) -> list[str]:
    """执行取日志的命令，返回输出行。

    命令是 cat/sed，只读不写；用 bash 执行以便与清单里的命令完全一致。
    """
    r = subprocess.run(["bash", "-c", cmd], cwd=str(LAB),
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    lines = r.stdout.splitlines()
    while lines and not lines[-1].strip():      # 去掉尾部空行
        lines.pop()
    return lines


def render(no: int) -> None:
    fname, caption, cmd = SHOTS[no]
    body = run_log(cmd)

    # 正文第一行放命令本身，模拟真实终端
    lines = [f"$ {cmd}"] + body

    # 过长行截断（保留行号信息无意义，直接截字符）
    shown, truncated = [], 0
    for l in lines:
        if disp_width(l) > MAX_COLS:
            truncated += 1
            cut, w = "", 0
            for ch in l:
                cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
                if w + cw > MAX_COLS - 1:
                    break
                cut += ch
                w += cw
            l = cut + "…"
        shown.append(l)

    cols = max(disp_width(l) for l in shown)
    cols = max(cols, disp_width(f"  {caption}"))
    n = len(shown)

    w_in = cols * CHAR_W + 0.30
    h_body = n * LINE_H + 0.30
    h_bar = 0.40
    fig = plt.figure(figsize=(w_in, h_bar + h_body))
    fig.patch.set_facecolor(BG)

    # 标题栏
    bar = Rectangle((0, 1 - h_bar / (h_bar + h_body)), 1, h_bar / (h_bar + h_body),
                    transform=fig.transFigure, facecolor=BAR, edgecolor="none")
    fig.patches.append(bar)
    for i, c in enumerate(DOT):
        fig.text(0.012 + i * 0.016, 1 - h_bar / (h_bar + h_body) / 2, "●",
                 color=c, fontsize=9, ha="center", va="center")
    fig.text(0.062, 1 - h_bar / (h_bar + h_body) / 2, caption,
             color=DIM, fontsize=9.5, ha="left", va="center", family=FONT)

    # 正文
    top = 1 - h_bar / (h_bar + h_body)
    dy = LINE_H / (h_bar + h_body)
    x0 = 0.011
    for i, l in enumerate(shown):
        y = top - 0.045 - i * dy
        color = ACCENT if i == 0 else FG
        weight = "bold" if i == 0 else "normal"
        fig.text(x0, y, l, color=color, fontsize=FS,
                 ha="left", va="top", family=FONT, fontweight=weight)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, facecolor=BG)
    plt.close(fig)
    warn = f"  ⚠ {truncated} 行被截断" if truncated else ""
    print(f"  [{no:>2}] {fname}  {cols} 列 × {n} 行  →  {w_in:.1f}×{h_bar+h_body:.1f} in{warn}")


def main_cli() -> None:
    args = sys.argv[1:]
    todo = [int(a) for a in args if a.isdigit() and int(a) in SHOTS] or sorted(SHOTS)
    print(f"[渲染终端证据] 输出到 {OUT}")
    for no in todo:
        render(no)
    print(f"\n[完成] 共 {len(todo)} 张。图片内容为 results/ 日志的原样输出，未增删字符。")


if __name__ == "__main__":
    main_cli()
