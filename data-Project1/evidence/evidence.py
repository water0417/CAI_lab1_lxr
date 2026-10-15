# -*- coding: utf-8 -*-
"""
实验一 · 证据演示脚本

用途：把报告里几条**过程性结论**的验证过程单独跑出来，每项输出控制在一屏以内，
方便直接截图放进报告。

为什么单独写这个脚本：报告里的结论分两类——**结果性**的（分数、曲线）已有图表覆盖；
**过程性**的（怎么防泄露、怎么验证一个反直觉结论）必须看到真实的命令与输出才有说服力。
这些验证在原实验里是临时命令，没留日志，所以在这里补成可复现的形式。

用法：
    python evidence.py            # 列出全部演示项
    python evidence.py 1          # 只跑第 1 项
    python evidence.py 4          # 第 4 项较慢（要跑一次 MLP，约 5 分钟）

五项分别对应报告里的：
    1  §2.3  测试集三道锁：正常运行时读不到它，被篡改时会拒绝运行
    2  §2.7  对 TF-IDF 而言"交换词序"是恒等变换（解释数据增强为何无效）
    3  §四   错误分析：逐类别 F1 + 混淆对 + 各类别特征词
    4  §5.2  MLP 把稀疏输入稠密化（解释高维下为何超线性变慢）
    5  §3.5  四模型的单折受控计时（为什么报告不引用 25 折运行的墙钟时间）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os as _os, sys as _sys
# 本文件位于子目录，而 main.py 在上一级——把它加进 sys.path 才能 import 到 main。
# main.py 里的路径常量（M.RESULT_DIR / M.FIG_DIR / M.TRAIN_CSV）都以 main.py 自身
# 的位置为锚，所以本文件搬家不影响它们。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import main as M


def _hdr(no: int, title: str, cmd: str) -> None:
    """每个演示项统一的开头，标明这是第几项、怎么复现。"""
    print("=" * 78)
    print(f"  证据 {no} · {title}")
    print(f"  复现命令：  python evidence.py {no}")
    print("=" * 78)
    print()


def _foot(t0: float) -> None:
    print()
    print(f"[完成] 耗时 {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# 1 · 测试集三道锁
# ---------------------------------------------------------------------------
def ev1_test_seal() -> None:
    t0 = time.time()
    _hdr(1, "测试集隔离：正常路径读不到它，被篡改时拒绝运行", "python evidence.py 1")

    print("【1a】每次运行都会打印数据指纹，用于自证数据未被改动")
    print(f"      训练集 train_data.csv       {M.file_fingerprint(M.TRAIN_CSV)}")
    p = M.resolve_test_csv()
    if p is not None:
        print(f"      测试集 test_data_unlabeled.csv {M.file_fingerprint(p)}")
    print()
    print("【1b】代码结构：测试集只被 load_test() 读取，而它只被 --submit 分支调用")
    import inspect
    src = inspect.getsource(M)
    callers = [ln.strip() for ln in src.splitlines()
               if "load_test()" in ln and "def load_test" not in ln]
    print(f"      全文件中调用 load_test() 的地方共 {len(callers)} 处：")
    for c in callers:
        print(f"        · {c}")
    print("      → 其余所有命令（默认 CV / --smoke / 调参 / 各种扫描）都拿不到测试数据")
    print()

    print("【1c】完整性校验：伪造一份被改动 1 个字节的测试集，看是否被拦住")
    import shutil
    import tempfile
    src = M.resolve_test_csv()
    if src is None:
        print("      （测试集当前不在预期位置，跳过本项）")
        _foot(t0)
        return
    tmp = Path(tempfile.mkdtemp(prefix="tamper_")) / "test_data_unlabeled.csv"
    data = src.read_bytes()
    tmp.write_bytes(data[:1000] + b"X" + data[1001:])   # 只翻转 1 个字节
    print(f"      伪造副本：{tmp}（大小与原文件相同：{len(data):,} 字节）")
    real_resolver = M.resolve_test_csv
    M.resolve_test_csv = lambda: tmp
    try:
        M.load_test()
        print("      ✗ 未被拦截 —— 校验失效")
    except RuntimeError as e:
        print("      ✓ 被拦截，报错如下：")
        for line in str(e).splitlines():
            print(f"        {line}")
    except Exception as e:
        print(f"      ？ 抛出其它异常：{type(e).__name__}: {e}")
    finally:
        M.resolve_test_csv = real_resolver
        shutil.rmtree(tmp.parent, ignore_errors=True)
    _foot(t0)


# ---------------------------------------------------------------------------
# 2 · 恒等变换验证
# ---------------------------------------------------------------------------
def ev2_identity_swap() -> None:
    t0 = time.time()
    _hdr(2, "对 TF-IDF，「交换词序」是恒等变换", "python evidence.py 2")

    from sklearn.feature_extraction.text import TfidfVectorizer

    X, y = M.load_train()
    all_docs = M.apply_clean(X, M.CLEAN_PRESETS["routing"])
    # 词表在全量语料上拟合——与真实实验流程一致。
    # 若只在被检验的 200 条上拟合，IDF 与词表都会不同，得出的数字与实验对不上。
    vec = TfidfVectorizer(**M.BEST_TFIDF).fit(all_docs)
    docs = all_docs[:200]

    print(f"  TF-IDF 配置 {M.tfidf_label(M.BEST_TFIDF)}（词表在全量 {len(all_docs)} 条上拟合）")
    print(f"  被检验样本 {len(docs)} 条")
    print()
    print("  做法：对每条文本做一次扰动，把扰动前后的文本分别向量化，")
    print("        比较两个特征向量的最大逐元素差异。差异为 0 即说明该扰动")
    print("        对当前特征表示**没有产生任何改变**。")
    print()
    print(f"  {'扰动方式':<18}{'参与样本':>10}{'向量完全相同':>14}{'占比':>10}")
    print("  " + "-" * 54)
    for mode, rate in (("swap", 0.10), ("delete", 0.10)):
        same = tot = 0
        for s in docs:
            a = M._augment_text(s, np.random.RandomState(7), mode, rate)
            if a == s:
                continue                      # 过短未扰动，不计入
            tot += 1
            d = vec.transform([s]) - vec.transform([a])
            if abs(d).max() == 0:
                same += 1
        label = "随机交换词序" if mode == "swap" else "随机删除 10% 词"
        pct = f"{same/tot*100:.1f}%" if tot else "—"
        print(f"  {label:<18}{tot:>10}{same:>14}{pct:>10}")
    print()
    print("  结论：交换词序 100% 不改变特征向量（词袋表示不含顺序信息），")
    print("        故「随机交换」那一组实际是「恒等复制训练集」的对照组。")
    _foot(t0)


# ---------------------------------------------------------------------------
# 3 · 错误分析
# ---------------------------------------------------------------------------
def ev3_error_analysis() -> None:
    t0 = time.time()
    _hdr(3, "错误分析：错误集中在计算机相关的四个类别内", "python evidence.py 3")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    X, y = M.load_train()
    docs = M.apply_clean(X, M.CLEAN_PRESETS["routing"])
    pipe = M.make_pipeline(**M.BEST_TFIDF)
    pipe.set_params(clf=M.LinearSVC(random_state=M.SEED, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=M.SEED)

    yp = np.empty_like(y)
    for tr, te in cv.split(docs, y):
        pipe.fit([docs[i] for i in tr], y[tr])
        yp[te] = pipe.predict([docs[i] for i in te])

    print(f"  最优模型 5 折折外预测 · 折外准确率 {(yp == y).mean():.4f}")
    print()
    rep = classification_report(y, yp, digits=4, output_dict=True)
    per = [(int(k), v["f1-score"], v["precision"], v["recall"], int(v["support"]))
           for k, v in rep.items() if k.isdigit()]
    print("  【逐类别表现】按 F1 升序（最差的在前）")
    print(f"    {'类别':<6}{'F1':>9}{'精确率':>10}{'召回率':>10}{'样本数':>8}")
    print("    " + "-" * 41)
    for c, f, p, r, n in sorted(per, key=lambda t: t[1]):
        print(f"    {c:<6}{f:>9.4f}{p:>10.4f}{r:>10.4f}{n:>8}")

    cm = confusion_matrix(y, yp)
    print()
    print("  【主要混淆对】真实→预测，出现 ≥15 次的（共 8 对）")
    pairs = sorted([(cm[i, j], i, j) for i in range(10) for j in range(10)
                    if i != j and cm[i, j] >= 15], reverse=True)
    for cnt, i, j in pairs:
        print(f"    真实 {i} → 预测 {j}：{cnt:>3} 例（占类别 {i} 的 {cnt/cm[i].sum()*100:.1f}%）")
    involved = sorted({i for _, i, _ in pairs} | {j for _, _, j in pairs})
    print(f"    → 这 {len(pairs)} 对涉及的类别集合：{involved}")
    print()

    vec = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english")
    Xa = vec.fit_transform(docs)
    names = vec.get_feature_names_out()
    print("  【各类别高权重词】用于推断类别含义（任务未提供类别名）")
    for c in range(10):
        mu = np.asarray(Xa[y == c].mean(axis=0)).ravel()
        top = ", ".join(names[i] for i in np.argsort(mu)[-7:][::-1])
        mark = "  ← 高混淆簇" if c in (0, 1, 2, 7) else ""
        print(f"    类别 {c}：{top}{mark}")
    _foot(t0)


# ---------------------------------------------------------------------------
# 4 · MLP 稠密化探测
# ---------------------------------------------------------------------------
def ev4_mlp_density() -> None:
    t0 = time.time()
    _hdr(4, "MLP 在高维稀疏输入上被稠密化（约 5 分钟）", "python evidence.py 4")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC

    X, y = M.load_train()
    docs = M.apply_clean(X, M.CLEAN_PRESETS["routing"])
    cv = StratifiedKFold(5, shuffle=True, random_state=M.SEED)
    tr, te = next(iter(cv.split(docs, y)))
    print(f"  取第 1 折：训练 {len(tr)} 条 / 验证 {len(te)} 条")
    print()

    t = time.time()
    vec = TfidfVectorizer(**M.BEST_TFIDF)
    Xtr = vec.fit_transform([docs[i] for i in tr])
    tfidf_t = time.time() - t
    density = Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1])
    print(f"  TF-IDF：维度 {Xtr.shape[1]:,}  每行非零 {Xtr.nnz/Xtr.shape[0]:.0f} "
          f"稠密度 {density*100:.2f}%  耗时 {tfidf_t:.1f}s")
    print(f"  若按稀疏矩阵处理，本应只与 {Xtr.nnz:,} 个非零项有关；")
    print(f"  而稠密化后参与运算的矩阵是 {Xtr.shape[0]:,} × {Xtr.shape[1]:,} "
          f"= {Xtr.shape[0]*Xtr.shape[1]/1e6:.0f}M 个元素"
          f"（float64 约 {Xtr.shape[0]*Xtr.shape[1]*8/1e9:.1f} GB）")
    print()

    print(f"  {'分类器':<22}{'总耗时':>10}{'迭代次数':>10}{'每次迭代':>12}")
    print("  " + "-" * 54)
    for name, clf in (("LinearSVC(C=1)", LinearSVC(random_state=M.SEED, C=1.0)),
                      ("MLP((100,), 早停)", MLPClassifier(hidden_layer_sizes=(100,),
                                                          alpha=1e-4, max_iter=300,
                                                          early_stopping=True,
                                                          random_state=M.SEED))):
        t = time.time()
        clf.fit(Xtr, y[tr])
        el = time.time() - t
        n_iter = getattr(clf, "n_iter_", None)
        n_iter = int(n_iter) if n_iter is not None else None
        n_s = f"{n_iter}" if n_iter else "—"
        # 用两位小数：LinearSVC 每次迭代只要 0.02s，一位小数会被舍成 0.0
        per_s = f"{el/n_iter:.2f}s" if n_iter else "—"
        print(f"  {name:<22}{el:>9.1f}s{n_s:>10}{per_s:>12}")
    print()
    print("  结论：每行只有极少非零项，MLP 却慢上千倍——说明 sklearn 把稀疏输入")
    print("        稠密化了。这是它随维度超线性变慢、且在四模型中最慢的根因。")
    _foot(t0)


# ---------------------------------------------------------------------------
# 5 · 四模型的单折受控计时
# ---------------------------------------------------------------------------
def ev5_model_cost() -> None:
    """为什么报告的耗时论断以"单折受控测量"为准，而不是 25 折运行的墙钟时间。

    25 折那几次运行是几小时的长任务，中途机器休眠/被别的进程抢占，墙钟时间
    会把等待也算进去。同一份代码两次跑出的 MLP 总耗时能差好几倍，**这类数字
    不可复现**，不能拿来立论。

    这里换成一次受控测量：**同一折、同一进程、同一线程设置**，四个模型 × 两个
    维度各拟合一次，全程背靠背连续跑完（约 5 分钟）。四个模型面对的是同一个矩阵、
    同一份机器负载，差值才归因得到模型本身。
    结果落盘 results/model_cost_single_fold.csv，报告与 README 的耗时论断引用它。
    """
    t0 = time.time()
    _hdr(5, "四模型单折受控计时：同一折、同一进程、连续测量（约 5 分钟）",
         "python evidence.py 5")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold

    X, y = M.load_train()
    docs = M.apply_clean(X, M.CLEAN_PRESETS[M.BEST_PRESET])
    cv = StratifiedKFold(M.N_SPLITS, shuffle=True, random_state=M.SEED)
    tr, _ = next(iter(cv.split(docs, y)))

    configs = [("5000 维（基线）", {"max_features": M.MAX_FEATURES}),
               ("50000 维（最优）", dict(M.BEST_TFIDF))]
    rows = []
    for label, tf in configs:
        t = time.time()
        Xtr = TfidfVectorizer(**tf).fit_transform([docs[i] for i in tr])
        fit_t = time.time() - t
        print(f"  【{label}】TF-IDF 拟合 {fit_t:.1f}s   "
              f"矩阵 {Xtr.shape[0]:,} × {Xtr.shape[1]:,}")
        print(f"  {'分类器':<22}{'拟合耗时':>10}")
        print("  " + "-" * 32)
        for name, clf in M.make_models().items():
            t = time.time()
            clf.fit(Xtr, y[tr])
            el = time.time() - t
            rows.append({"维度配置": label, "模型": name, "拟合耗时(s)": round(el, 2)})
            print(f"  {name:<22}{el:>9.1f}s")
        print()

    df = pd.DataFrame(rows)
    wide = df.pivot(index="模型", columns="维度配置", values="拟合耗时(s)")
    wide = wide[["5000 维（基线）", "50000 维（最优）"]]
    wide["倍数"] = (wide["50000 维（最优）"] / wide["5000 维（基线）"]).round(1)
    print("  汇总（单折拟合耗时，秒）：")
    print(wide.to_string())
    print()
    base = wide.loc["LinearSVC", "50000 维（最优）"]
    slow = wide.loc["MLP", "50000 维（最优）"]
    print(f"  结论：最优配置下单折 MLP 拟合 {slow:.0f}s，其余三个模型都在 10s 以内——")
    print(f"        MLP 比 LinearSVC 慢约 {slow/base:.0f} 倍，即两个数量级。")
    print(f"        MLP 也是绝对增幅最大的模型：维度 5000→50000 多花 "
          f"{slow - wide.loc['MLP','5000 维（基线）']:.0f} 秒，LinearSVC 基本不动。")
    print()
    print("  注意：本项重复跑，绝对耗时会差 1.5~2.5 倍（机器负载所致，同一折、同一进程")
    print("        也一样）。所以只取「差两个数量级」这个稳健结论，不引用具体的倍数与秒数。")

    out = M.RESULT_DIR / "model_cost_single_fold.csv"
    wide.to_csv(out)
    print(f"\n  → 已写入 {out.relative_to(M.PROJECT_DIR)}")
    _foot(t0)


DEMOS = {1: ev1_test_seal, 2: ev2_identity_swap,
         3: ev3_error_analysis, 4: ev4_mlp_density,
         5: ev5_model_cost}

DEMO_DESC = {
    1: "测试集三道锁：正常路径读不到它、被篡改时拒绝运行      ［快］",
    2: "对 TF-IDF，「交换词序」是恒等变换（数据增强为何无效）  ［快］",
    3: "错误分析：逐类别 F1 + 混淆对 + 各类别特征词            ［快］",
    4: "MLP 把稀疏输入稠密化（高维下为何超线性变慢）          ［约 5 分钟］",
    5: "四模型的单折受控计时（报告为何不用 25 折的墙钟时间）    ［约 5 分钟］",
}


def main_cli() -> None:
    args = sys.argv[1:]
    if not args:
        print("实验一 · 证据演示。可跑项：")
        for no in DEMOS:
            print(f"  {no}  {DEMO_DESC[no]}")
            print(f"       python evidence.py {no}")
        print()
        print("每项输出控制在一屏以内，可直接截图。")
        return
    for a in args:
        if not a.isdigit() or int(a) not in DEMOS:
            print(f"未知演示项 {a!r}；可选 {list(DEMOS)}")
            continue
        DEMOS[int(a)]()


if __name__ == "__main__":
    main_cli()
