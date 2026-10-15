# -*- coding: utf-8 -*-
"""MLP 超参搜索 —— 补上四个模型里最后一块"调参对等"的拼图。

已完成：12 组（4 种结构 × 3 个 alpha）全部跑完，结果存 mlp_screen_5fold.csv
   （5 折单次）；最优配置另做了 25 折确认，存 mlp_tuned_25fold_5x5.csv。
   结论：默认结构 (100,) 即最优；加宽/加深都更差；调参收益为 0。
   （本脚本有断点续跑：中断后再跑会自动跳过已完成的组合。）

背景：§3.1 主表里四个模型都用库默认参数，但默认值离各自最优的远近差别极大。
LinearSVC(C=1)、LR(C=100)、NB(alpha=0.1) 都已查明并调过；**只有 MLP 还是默认配置**，
所以 §3.1 的"调参对等"表里它是唯一空着的一格。

搜索区间取自助教的 `TUNING.md`：
    hidden_layer_sizes ∈ {(50,), (100,), (200,), (100,50)}
    alpha              ∈ {1e-4, 1e-3, 1e-2}

两阶段（MLP 在 50000 维下每折要 3~5 分钟，全网格 ×25 折代价不可接受）：
    阶段 1  5 折单次，跑完 12 个组合 → 选出最优
    阶段 2  最优组合跑 25 折（与主表同口径）→ 这个数才写进报告

用法：
    python mlp_tune.py screen      # 阶段 1
    python mlp_tune.py confirm 200 1e-3   # 阶段 2，参数为已选出的最优

性能说明：这台机器上 MLP 的 BLAS **不会自动并行**（实测只用到约 0.5 个核），
所以阶段 1 改成**折级并行**——启动前把 `OMP_NUM_THREADS=1`（每个 worker 单线程，
避免超订），再用 `JOBS=5` 让 5 折同时跑，实测吞吐约为串行的 5~10 倍。
阶段 2（确认）仍用 `n_jobs=1` + 默认线程，与主表完全同口径。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

import os as _os, sys as _sys
# 本文件位于子目录，而 main.py 在上一级——把它加进 sys.path 才能 import 到 main。
# main.py 里的路径常量（M.RESULT_DIR / M.FIG_DIR / M.TRAIN_CSV）都以 main.py 自身
# 的位置为锚，所以本文件搬家不影响它们。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import main as M

GRID_HIDDEN = [(50,), (100,), (200,), (100, 50)]
GRID_ALPHA = [1e-4, 1e-3, 1e-2]
SCREEN_CSV = M.RESULT_DIR / "mlp_screen_5fold.csv"
FOLD_CSV = M.RESULT_DIR / "mlp_screen_fold_scores.csv"
JOBS = int(os.environ.get("JOBS", "5"))


def _mlp(hidden, alpha) -> MLPClassifier:
    """与 make_models() 完全一致，只换 hidden_layer_sizes 与 alpha。"""
    return MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=300,
                         early_stopping=True, random_state=M.SEED)


def _tag(hidden, alpha) -> str:
    return f"MLP · hidden={'x'.join(map(str, hidden))}、alpha={alpha:g}"


def screen() -> None:
    X, y = M.load_train()
    cv = StratifiedKFold(5, shuffle=True, random_state=M.SEED)   # 单次 5 折，与主表第 1 次重复同折
    done = pd.read_csv(SCREEN_CSV) if SCREEN_CSV.exists() else None
    have = set(done["配置"]) if done is not None else set()
    sink = pd.read_csv(FOLD_CSV) if FOLD_CSV.exists() else pd.DataFrame()

    combos = [(h, a) for h in GRID_HIDDEN for a in GRID_ALPHA]
    print(f"[阶段 1] MLP 超参筛选：{len(combos)} 个组合 × 5 折（单次，与主表第 1 次重复同折）")
    print(f"[阶段 1] 配置：routing + max_features=50000 + sublinear_tf；"
          f"折级并行 n_jobs={JOBS}（OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','未设')}）；"
          f"跳过已完成 {len(have)} 个\n")

    rows = done.to_dict("records") if done is not None else []
    for hidden, alpha in combos:
        tag = _tag(hidden, alpha)
        if tag in have:
            print(f"  [跳过] {tag}")
            continue
        t0 = time.time()
        pipe = M.make_pipeline(M.CLEAN_PRESETS[M.BEST_PRESET], **M.BEST_TFIDF)
        pipe.set_params(clf=_mlp(hidden, alpha))
        r = cross_validate(pipe, X, y, cv=cv, scoring=M.SCORING,
                           return_train_score=True, n_jobs=JOBS)
        te = r["test_macro_f1"]
        rows.append({"配置": tag, "hidden": str(hidden), "alpha": alpha,
                     "Macro-F1": float(te.mean()), "标准差": float(te.std(ddof=1)),
                     "训练-MacroF1": float(r["train_macro_f1"].mean()),
                     "泛化间隙": float(r["train_macro_f1"].mean() - te.mean()),
                     "耗时(s)": round(time.time() - t0, 1)})
        sink[tag] = te
        pd.DataFrame(rows).to_csv(SCREEN_CSV, index=False, encoding="utf-8-sig")
        sink.to_csv(FOLD_CSV, index=False)
        print(f"  [{len(rows):>2}/{len(combos)}] {tag:44s} Macro-F1={te.mean():.4f} "
              f"±{te.std(ddof=1):.4f} 间隙={rows[-1]['泛化间隙']:.4f} "
              f"({rows[-1]['耗时(s)']:.0f}s)")

    df = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)
    print("\n[阶段 1 结果]")
    print(df[["配置", "Macro-F1", "标准差", "泛化间隙"]].to_string(index=False))
    best = df.iloc[0]
    print(f"\n  最优：{best['配置']}  Macro-F1={best['Macro-F1']:.4f}")
    print(f"  下一步：python mlp_tune.py confirm {best['hidden'].replace(' ','')} {best['alpha']:g}")


def confirm(hidden_str: str, alpha: float) -> None:
    hidden = tuple(int(v) for v in hidden_str.strip("()").split(",") if v.strip())
    X, y = M.load_train()
    cv = M.make_cv(5, 5)
    tag = _tag(hidden, alpha)
    print(f"[阶段 2] 25 折确认：{tag}")
    print("[阶段 2] 与主表同界面（make_cv(5,5)、n_jobs=1、默认线程设置）\n")
    t0 = time.time()
    sink = {}
    df, raw = M.run_cv({"MLP": _mlp(hidden, alpha)}, X, y, cv,
                       cfg=M.CLEAN_PRESETS[M.BEST_PRESET], tfidf=M.BEST_TFIDF,
                       fold_sink=sink, quiet=False)
    r = raw.iloc[0]
    out = pd.DataFrame([{"配置": tag, "Accuracy": r["accuracy"], "Macro-F1": r["macro_f1"],
                         "Weighted-F1": r["weighted_f1"], "训练-MacroF1": r["train_macro_f1"],
                         "泛化间隙": r["gap"], "耗时(s)": round(time.time() - t0, 1)}])
    out.to_csv(M.RESULT_DIR / "mlp_tuned_25fold_5x5.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sink).to_csv(M.RESULT_DIR / "mlp_tuned_fold_scores_5x5.csv", index=False)
    print(f"\n  Macro-F1 = {r['macro_f1']:.4f} ± {r['macro_f1_std']:.4f}  间隙 {r['gap']:.4f}")
    print("  [记录] results/mlp_tuned_25fold_5x5.csv、mlp_tuned_fold_scores_5x5.csv")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if arg == "screen":
        screen()
    elif arg == "confirm":
        confirm(sys.argv[2], float(sys.argv[3]))
    else:
        print(__doc__)
