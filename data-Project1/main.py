# -*- coding: utf-8 -*-
"""
实验一：新闻文本 10 分类 —— 主实验脚本
=====================================

本文件是实验的**唯一入口**，覆盖从数据划分到提交文件生成的全部流程。
其余脚本（`experiments/`、`evidence/`）只做补充验证，都复用本文件的函数与路径常量。

## 实验流程与对应开关

| 阶段 | 主要开关 | 做什么 |
|---|---|---|
| 自检 | `--smoke` | 600 条样本走通全流程，**不写任何文件** |
| 1 · 数据划分 | `--compare-splits` | 单次 80/20 vs 5 折 CV 的估计稳定性对照 |
| 2 · 清洗与增强 | `--ablation`、`--paired-test`、`--augment`、`--augment-robust`、`--show-clean` | 13 个清洗配置的消融、配对检验、数据增强与鲁棒性评估 |
| 3 · TF-IDF 调参 | `--tfidf-sweep`、`--tfidf-combo`、`--tfidf-report` | 单参数敏感性扫描 + 参数组合验证 |
| 4 · 模型调参 | `--model-sweep`、`--nested-cv`、`--model-paired-test` | 超参扫描、嵌套 CV 无偏估计、四模型两两配对检验 |
| 5 · 评估 | `--learning-curve`、`--ablation-report` | 学习曲线（过拟合随数据量的变化） |
| 6 · 提交 | `--submit` | **唯一读取测试集**的路径，写出预测文件 |

完整开关见 `python main.py --help`（共 25 个）。

## 设计原则（对应《实验一要求》第 2~5 页的实验规范）

1. **防数据泄露**：任何 fit 只发生在训练数据上。
   - TF-IDF 绑定在 Pipeline 内，交叉验证时**每折独立拟合**一份词表与 IDF；
   - 测试集"气隙隔离"：只有 `--submit` 会读取，其余路径根本不加载它；
   - 数据增强**只在 CV 折内做**——先增广再划分会造成隐蔽泄露（同一文本的原始版
     与扰动版分落训练/验证折，常规查重发现不了）。
2. **控制变量**：所有模型共用同一套 CV 划分、同一套 TF-IDF 配置、同一个随机种子。
   比较模型时 TF-IDF 参数必须完全一致，否则模型间的差异无法归因。
3. **可复现**：SEED 固定；数据指纹（大小 + SHA256）写进每份日志；
   环境版本记录在 requirements.txt；结果一律落盘到 `results/`。
4. **一切结论都过配对检验**：逐折分数落盘到 `results/*fold_scores*.csv`，
   因此任意两个配置都能**事后重算**配对 t 检验，不依赖当时打印的结论。

## 长跑保护（这台机器会不定期休眠）

几小时的长任务很容易被系统冻结，因此每个扫描都做了两件事：

- **逐配置落盘 + 断点续跑**：中断后重跑会跳过已完成的配置；
- **写盘前与旧文件取并集**（`_union_with_prev`）：只请求部分配置时不会把已有结果抹掉。
  —— 这是早期踩过的坑：续跑逻辑是覆盖式的，只跑子集会把上次的结果从 CSV 里删掉，
  而且**没有任何报错**。

## 用法

    conda activate AILab

    python main.py --smoke                     # 自检，不写任何文件
    python main.py                             # 基线：四模型 × 25 折（约 31 分钟）
    python main.py --ablation                  # 清洗消融（13 配置，约 85 分钟）
    python main.py --tfidf-sweep --clean-preset both   # TF-IDF 参数扫描
    python main.py --model-sweep               # 模型超参扫描
    python main.py --nested-cv                 # 嵌套交叉验证
    python main.py --model-paired-test --tfidf-preset best   # 四模型主结果 + 两两配对检验
    python main.py --learning-curve            # 学习曲线
    python main.py --submit --models LinearSVC # 生成提交文件（唯一读取测试集）
    python main.py --help                      # 全部开关

## 输出

结果一律写到 `Lab1/results/`（`RESULT_DIR` 以**本文件位置**为锚，不受调用目录影响）：

    cv_5x5.csv                          基线交叉验证（四模型 × 25 折）
    ablation_clean_5x5.csv              清洗消融（13 配置）
    tfidf_sweep_{raw,routing}_5x5.csv   TF-IDF 参数扫描
    model_sweep_routing_5x5.csv         模型超参扫描
    nested_cv_5x5.csv                   嵌套 CV 无偏估计
    model_metrics_routing_best_5x5.csv  四模型主结果（+ 配对检验）
    learning_curve.csv                  学习曲线
    *_fold_scores_*.csv                 逐折分数（配对检验的唯一输入）
    run_*.log                           运行日志（部分被渲染成报告里的终端截图）

    ./predictions.csv                   仅 --submit 生成（2457 行、单列、无表头、取值 0~9）

## 为什么不用单次 80/20 划分

单次划分的估计值对"抽到哪 1474 条"高度敏感。实测换随机种子，LinearSVC 的单次
holdout 估计标准差为 0.0070（极差 0.0207），而 5 折 CV 只有 0.0025（极差 0.0073）
——**单次划分的波动比模型之间的真实差距还大**。
因此这里以重复分层交叉验证为主：5 折 × 5 次重复 = 25 个估计值，报告"均值 ± 标准差"，
直接响应任务书第 5 页"存在随机性时报告波动"的要求。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 控制台默认按本地代码页（GBK）输出中文，在 VS Code / Windows Terminal
# 等 UTF-8 终端下会显示为乱码。这里显式统一为 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nltk.stem import PorterStemmer
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# 全局配置：全部集中在这里，方便报告里"实验设置"一节直接引用
# --------------------------------------------------------------------------
SEED = 42                 # 固定随机种子，保证完全可复现
N_SPLITS = 5              # 交叉验证折数
N_REPEATS = 5             # 重复次数 → 共 5 × 5 = 25 个估计值
TEST_SIZE = 0.2           # 仅 --holdout 时使用的单次划分比例
N_CLASSES = 10            # 新闻文本 10 分类
MAX_FEATURES = 5000       # TF-IDF 词表上限，与 draft_main.py 保持一致，作为基线配置

HERE = Path(__file__).resolve().parent      # data-Project1/
PROJECT_DIR = HERE.parent                   # Lab1/
FIG_DIR = PROJECT_DIR / "figs"
RESULT_DIR = PROJECT_DIR / "results"

TRAIN_CSV = HERE / "train_data.csv"
PREDICTION_CSV = HERE / "predictions.csv"

# ---- 测试集：气隙隔离 ----
# 测试集被移到项目外的 sealed/ 目录，物理上离开工作区；同时只有 --submit 这一条
# 代码路径会读它。两道锁叠加：一道靠目录隔离，一道靠代码结构。
SEALED_DIR = PROJECT_DIR / "sealed"
TEST_CSV = SEALED_DIR / "test_data_unlabeled.csv"
TEST_CSV_LEGACY = HERE / "test_data_unlabeled.csv"   # 万一又被移回原位时的兜底

# 助教原始测试集的指纹，2026-10-09 记录。--submit 前会逐字节校验，
# 对不上就直接拒绝运行——这是"测试集自始至终未被修改"唯一的硬证据。
EXPECTED_TEST_SIZE = 4_499_622
EXPECTED_TEST_SHA256 = "4882c04b008c055f351f1c9b25c45d601a454077af37e79375601f2465b41615"


# --------------------------------------------------------------------------
# 0. 工具
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    """流式计算文件 SHA256，避免把大文件整份读进内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> str:
    """返回文件的 (字节数, SHA256 前16位)，用于在运行日志里留痕。

    最终提交时可以比对完整哈希，证明测试集自始至终未被修改——
    这是"我没有触碰测试集"这句话唯一的可验证证据。
    """
    return f"{path.stat().st_size} bytes, sha256:{sha256_of(path)[:16]}"


def resolve_test_csv() -> Path | None:
    """定位被隔离的测试集：优先 sealed/，其次原位兜底。"""
    for p in (TEST_CSV, TEST_CSV_LEGACY):
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------
# 1. 数据加载
# --------------------------------------------------------------------------
def load_train():
    """只读训练集。

    测试集刻意不在此处加载——这是"代码层气隙"：任何不经过 --submit 的运行路径
    都拿不到测试数据，从结构上杜绝误用。
    """
    df = pd.read_csv(TRAIN_CSV)
    return df["text"].astype(str).tolist(), df["target"].to_numpy()


def load_test():
    """读取无标签测试集。**只允许被 --submit 路径调用。**

    读取前逐字节校验 SHA256：对不上就抛异常拒绝运行。
    这保证了提交结果一定对应助教发放的原始测试集，而不是某个被改动过的版本。
    """
    path = resolve_test_csv()
    if path is None:
        raise FileNotFoundError(
            f"找不到测试集。已查找：\n  {TEST_CSV}\n  {TEST_CSV_LEGACY}\n"
            f"请把它放回 {SEALED_DIR}/ 再运行 --submit。"
        )

    size, digest = path.stat().st_size, sha256_of(path)
    if size != EXPECTED_TEST_SIZE or digest != EXPECTED_TEST_SHA256:
        raise RuntimeError(
            "测试集完整性校验失败，拒绝运行。\n"
            f"  路径    : {path}\n"
            f"  实际    : {size} bytes, sha256:{digest}\n"
            f"  期望    : {EXPECTED_TEST_SIZE} bytes, sha256:{EXPECTED_TEST_SHA256}\n"
            "该文件与助教发放的原始版本不一致，可能是被改动或替换过。"
        )

    df = pd.read_csv(path)
    return df["text"].astype(str).tolist(), len(df), path


# --------------------------------------------------------------------------
# 2. 文本清洗（步骤 2）
# --------------------------------------------------------------------------
# 清洗是**无状态**变换（规则固定，不依赖数据统计量），因此它本身不会造成数据
# 泄露。但仍然把它放进 Pipeline，理由有两个：
#   1. 让"预处理 + 向量化 + 分类"成为一个整体，任何 fit 都只在训练折内发生；
#   2. 保证训练折与验证折施加的是逐字相同的变换。
#
# 语料是 Usenet 新闻组帖子，结构固定：
#     From: xxx@yyy.edu          ← 邮件头，首个空行之前
#     Subject: Re: something
#     Organization: Some University
#     Lines: 17
#     (空行)
#     正文……（含 > 引用、In article <...> writes: 出处行、-- 之后的签名档）
_HEADER_END = re.compile(r"^(.*?)\n[ \t]*\n", re.S)

# 邮件头里唯一保留的一行。Subject 是帖子的标题，属于正文内容；
# 而 From / Organization / NNTP-Posting-Host 这类是路由元数据 ——
# 实测 Organization 与类别近乎一一对应（某些组织的帖子只属于一个新闻组），
# 只留它就能把准确率刷到很高，属于典型的"捷径特征"。
_SUBJECT_PREFIX = "subject:"

_QUOTE_LINE = re.compile(r"^\s*>")                      # > 开头的引用行
_ATTRIBUTION = re.compile(r"^\s*In article\s+<[^>]+>.*writes:?\s*$", re.I)  # 引用出处行
_SIG_SEP = re.compile(r"\n--[ \t]*\n")                  # Usenet 标准签名分隔符
_NON_ALNUM = re.compile(r"[^A-Za-z0-9\s]+")             # 非字母数字 → 空格

_STEMMER = PorterStemmer()
_STOPWORDS = frozenset(w.lower() for w in ENGLISH_STOP_WORDS)


@dataclass(frozen=True)
class CleanConfig:
    """一套清洗规则的开关组合。

    header: "keep"         保留整个邮件头（不做任何处理）
            "drop_routing" 只删路由元数据行，保留 Subject
            "drop_all"     整个邮件头全删（含 Subject）
    其余字段是逐项开关，按 header → quotes → signature → punct → stopwords → stem
    的顺序依次施加。
    """
    header: str = "keep"
    quotes: bool = False
    signature: bool = False
    punct: bool = False
    stopwords: bool = False
    stem: bool = False

    def key(self) -> tuple:
        """用于去重：开关组合相同的配置视为同一个实验组。"""
        return (self.header, self.quotes, self.signature,
                self.punct, self.stopwords, self.stem)


def _split_header_body(s: str) -> tuple[str, str]:
    """按首个空行切出邮件头与正文。

    加一道保险：只有在首个空行之前的块**看起来确实像邮件头**（不超过 2000 字符，
    且以 `Key: value` 开头）时才认定它是邮件头，避免把正文里的段落分隔误判。
    """
    m = _HEADER_END.match(s)
    if not m:
        return "", s
    head = m.group(1)
    if len(head) > 2000 or not re.match(r"^[A-Za-z][A-Za-z0-9\-]*\s*:", head):
        return "", s
    return head, s[m.end():]


def _drop_headers(s: str, mode: str) -> str:
    if mode == "keep":
        return s
    head, body = _split_header_body(s)
    if mode == "drop_all" or not head:
        return body
    # drop_routing：只保留 Subject 行，其余 Key: value 全部丢弃
    kept = [ln for ln in head.split("\n") if ln.lower().startswith(_SUBJECT_PREFIX)]
    return "\n".join(kept) + "\n\n" + body


def _drop_quotes(s: str) -> str:
    """去掉 > 引用行与 'In article <...> writes:' 出处行。

    引用内容来自别人，既重复又常与本来类别无关，是噪声的主要来源之一。
    """
    out = []
    kept_header_done = False
    for ln in s.split("\n"):
        if _QUOTE_LINE.match(ln) or _ATTRIBUTION.match(ln):
            continue
        out.append(ln)
    return "\n".join(out)


def _drop_signature(s: str) -> str:
    """丢弃第一个 '--' 分隔行及其之后的所有内容（标准 Usenet 签名区）。"""
    m = _SIG_SEP.search(s)
    return s[:m.start()] if m else s


def _drop_punct(s: str) -> str:
    """把非字母数字字符替换为空格。保留数字（年份、版本号等仍有信息量）。"""
    return _NON_ALNUM.sub(" ", s)


def _drop_stopwords(s: str) -> str:
    """去掉英文停用词。用 sklearn 内置词表（318 词），无需联网下载。"""
    return " ".join(t for t in s.split() if t.lower() not in _STOPWORDS)


def _stem(s: str) -> str:
    """Porter 词干化，把 running/runs/ran 归并到同一词干，压缩词表。"""
    return " ".join(_STEMMER.stem(t) for t in s.split())


def clean_text(text: str, cfg: CleanConfig) -> str:
    """按配置依次施加清洗规则。顺序固定，方便消融实验逐级对照。"""
    s = _drop_headers(text, cfg.header)
    if cfg.quotes:
        s = _drop_quotes(s)
    if cfg.signature:
        s = _drop_signature(s)
    if cfg.punct:
        s = _drop_punct(s)
    if cfg.stopwords:
        s = _drop_stopwords(s)
    if cfg.stem:
        s = _stem(s)
    return s


def apply_clean(docs, cfg: CleanConfig):
    """批量清洗，供 FunctionTransformer 调用（partial 绑定 cfg 后可被 joblib 序列化）。"""
    return [clean_text(d, cfg) for d in docs]


def describe(cfg: CleanConfig) -> str:
    """把开关组合翻译成人类可读的处理说明，用于结果表。"""
    parts = {"keep": "保留全部邮件头",
             "drop_routing": "删路由头(留Subject)",
             "drop_all": "邮件头全删"}[cfg.header]
    for flag, label in ((cfg.quotes, "去引用"), (cfg.signature, "去签名"),
                        (cfg.punct, "去标点"), (cfg.stopwords, "去停用词"),
                        (cfg.stem, "stemming")):
        if flag:
            parts += " + " + label
    return parts


# 消融实验选出的两个候选前提（步骤 3 起将在这两种前提下各跑一遍对照）：
#   raw     —— L0 原文，不做任何清洗，与步骤 1 基线完全一致
#   routing —— L1 删路由元数据、保留 Subject，步骤 2 中唯一有显著正收益的配置
CLEAN_PRESETS: dict[str, CleanConfig] = {
    "raw": CleanConfig(),
    "routing": CleanConfig(header="drop_routing"),
}


# --------------------------------------------------------------------------
# 3. 模型与管道
# --------------------------------------------------------------------------
def make_pipeline(cfg: CleanConfig | None = None, **tfidf_kwargs) -> Pipeline:
    """构造 (清洗 → TF-IDF → 分类器) 管道，分类器由调用方通过 set_params 指定。

    必须使用 Pipeline：TfidfVectorizer 只在 fit 时所见的那份数据上学习词表与 IDF，
    验证集/测试集只会被 transform。交叉验证时每折都会构造并拟合一份全新的管道，
    折与折之间不共享任何统计量——这不是靠人工小心，是结构上做不到泄露。

    cfg=None 表示不做任何清洗（步骤 1 的基线）；
    步骤 2 的消融实验通过传入不同的 CleanConfig 来切换清洗规则，
    **TF-IDF 的参数在所有配置下保持完全一致**，否则模型差异无法归因。
    """
    params = {"max_features": MAX_FEATURES}
    params.update(tfidf_kwargs)
    steps = []
    if cfg is not None:
        steps.append(("clean", FunctionTransformer(partial(apply_clean, cfg=cfg))))
    steps += [
        ("tfidf", TfidfVectorizer(**params)),
        ("clf", MultinomialNB()),   # 占位，由 make_models 覆盖
    ]
    return Pipeline(steps)


def make_models() -> dict[str, object]:
    """四个待比较的分类器，全部使用默认/温和参数作为基线。

    调参区间见 TUNING.md，步骤 4 会用 GridSearchCV 覆盖这里的默认值。
    """
    return {
        "NaiveBayes": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        # LinearSVC(LIBLINEAR, OvR) 与 SVC(kernel='linear')(libsvm, OvO) 在文本任务上
        # 结果接近；基线用前者是为了在大特征数下控制训练时间。
        "LinearSVC": LinearSVC(random_state=SEED),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,),   # 单隐层，先建立简单基线
            alpha=1e-4,
            max_iter=300,
            # 早停是 MLP 的标准训练协议：它从**训练折内部**再切出 10% 作为内部验证集，
            # 不接触外层验证集，因此不构成泄露。实测可把单折耗时从 ~108s 降到 ~31s。
            early_stopping=True,
            random_state=SEED,
        ),
    }


# --------------------------------------------------------------------------
# 4. 划分
# --------------------------------------------------------------------------
def make_cv(n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS):
    """重复分层交叉验证。

    为什么用重复 CV 而不是单次 80/20：
      - 分层保证每折的类别比例与全集一致；
      - 重复用不同的随机划分跑多轮，把"划分运气"平均掉，
        估计值的标准差从 0.0070 降到 0.0025（LinearSVC 实测）；
      - 每个样本会被验证 n_repeats 次，充分利用了全部数据。
    """
    if n_repeats == 1:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=SEED)


def check_cv_folds(y, cv) -> None:
    """检查交叉验证的每一折是否都保持了分层。

    分层是"划分规范"的直接证据，报告里可以放这张汇总。
    """
    sizes, devs = [], []
    p_all = np.array([(y == c).mean() for c in np.unique(y)])
    classes = np.unique(y)
    n_folds = 0
    for _, va in cv.split(np.zeros(len(y)), y):
        n_folds += 1
        sizes.append(len(va))
        p_fold = np.array([(y[va] == c).mean() for c in classes])
        devs.append(np.abs(p_fold - p_all).max())
    print(f"[划分检查] 共 {n_folds} 折 | 每折验证集 {min(sizes)}~{max(sizes)} 条"
          f" | 类别占比相对全集的最大偏差 {max(devs):.4f}\n")


# --------------------------------------------------------------------------
# 5. 评估
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict[str, float]:
    """统一评估口径：Accuracy / Macro-F1 / Weighted-F1。

    三指标同时报告，因为 Macro-F1 更能暴露少数类上的失效。
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted"),
    }


def ms(mean: float, std: float) -> str:
    """把「均值 ± 标准差」格式化成报告可直接引用的字符串。"""
    return f"{mean:.4f} ± {std:.4f}"


SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}


def run_cv(models: dict, X, y, cv, n_jobs: int = 1,
           cfg: CleanConfig | None = None, quiet: bool = False,
           label: str = "", fold_sink: dict | None = None,
           tfidf: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """交叉验证，输出各指标的均值 ± 标准差。

    同时取回训练分数（return_train_score=True），用于计算泛化间隙
    （训练分数 − 验证分数），这是判断过拟合程度的直接证据。

    fold_sink：若传入 dict，会把每个模型在**每一折**上的 Macro-F1 存进去，
    供配对显著性检验使用（配对检验需要逐折的原始分数，均值不够）。

    返回 (展示用表格, 数值用表格)。展示表把指标格式化成 "均值 ± 标准差" 字符串，
    数值表保留 float 供程序化选优使用——不要从字符串里反解数字。
    """
    rows, raw = [], []
    for name, clf in models.items():
        pipe = make_pipeline(cfg, **(tfidf or {}))
        pipe.set_params(clf=clf)
        t = time.time()
        r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING,
                           return_train_score=True, n_jobs=n_jobs)
        cost = time.time() - t
        if fold_sink is not None:
            fold_sink[name] = r["test_macro_f1"]
        gap = float((r["train_macro_f1"] - r["test_macro_f1"]).mean())
        key = {"accuracy": "test_accuracy", "macro_f1": "test_macro_f1",
               "weighted_f1": "test_weighted_f1", "train_macro_f1": "train_macro_f1"}
        s = {k: r[v] for k, v in key.items()}
        n = len(s["macro_f1"])
        raw.append({
            "Model": name,
            **{k: float(v.mean()) for k, v in s.items()},
            "macro_f1_std": float(s["macro_f1"].std(ddof=1)),
            "n_estimates": n,
            "gap": gap,
            "耗时(s)": round(cost, 1),
        })
        rows.append({
            "Model": name,
            "Accuracy": ms(s["accuracy"].mean(), s["accuracy"].std(ddof=1)),
            "Macro-F1": ms(s["macro_f1"].mean(), s["macro_f1"].std(ddof=1)),
            "Weighted-F1": ms(s["weighted_f1"].mean(), s["weighted_f1"].std(ddof=1)),
            "训练-MacroF1": ms(s["train_macro_f1"].mean(), s["train_macro_f1"].std(ddof=1)),
            "泛化间隙": f"{gap:.4f}",
            "折数": n,
            "耗时(s)": round(cost, 1),
        })
        if not quiet:
            print(f"  - {label}{name:<20} MacroF1={rows[-1]['Macro-F1']}  "
                  f"间隙={rows[-1]['泛化间隙']}  ({cost:.1f}s)")
    return pd.DataFrame(rows), pd.DataFrame(raw)


def run_holdout(models: dict, X, y) -> pd.DataFrame:
    """可选的 80/20 单次划分对照。

    只作为"快速自检"保留，**不作为选优依据**：实测它的估计波动（标准差 0.0070）
    比模型间的真实差距还大。
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    rows = []
    for name, clf in models.items():
        pipe = make_pipeline()
        pipe.set_params(clf=clf)
        t = time.time()
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)
        rows.append({"Model": name, **evaluate(y_val, y_pred),
                     "耗时(s)": round(time.time() - t, 1)})
    print(f"[holdout] train {len(X_tr)} / val {len(X_val)}（stratify=y, seed={SEED}）")
    return pd.DataFrame(rows)


def compare_splits(X, y, n_holdout_seeds: int = 10, n_cv_seeds: int = 6) -> pd.DataFrame:
    """对照实验：单次 holdout 与 5 折 CV 的**估计稳定性**。

    做法是固定同一个模型，只改随机种子看估计值晃多少：
      - holdout：换 n_holdout_seeds 个种子各做一次 80/20 划分，得到 n 个估计值；
      - CV：换 n_cv_seeds 个种子各做一次 5 折 CV，得到 n 个估计值。
    比较两组估计值的标准差与极差，就得到"哪种划分方案更稳"的直接证据。

    固定用 LinearSVC：这里比较的是划分方案本身，与模型无关，用最快的模型即可。
    """
    def svc_pipe():
        p = make_pipeline()
        p.set_params(clf=LinearSVC(random_state=SEED))
        return p

    holdout = []
    for s in range(n_holdout_seeds):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=s)
        p = svc_pipe()
        p.fit(X_tr, y_tr)
        holdout.append(f1_score(y_val, p.predict(X_val), average="macro"))

    cvs = []
    for s in range(n_cv_seeds):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        r = cross_val_score(svc_pipe(), X, y, cv=cv, scoring="f1_macro")
        cvs.append(r.mean())

    holdout, cvs = np.array(holdout), np.array(cvs)
    summary = pd.DataFrame([
        {"划分方案": f"单次 holdout 80/20（{n_holdout_seeds} 个种子）", "估计值个数": len(holdout),
         "均值": holdout.mean(), "标准差": holdout.std(ddof=1), "极差": holdout.max() - holdout.min()},
        {"划分方案": f"{N_SPLITS} 折 CV（{n_cv_seeds} 个种子）", "估计值个数": len(cvs),
         "均值": cvs.mean(), "标准差": cvs.std(ddof=1), "极差": cvs.max() - cvs.min()},
    ])
    detail = pd.DataFrame({"seed": list(range(max(len(holdout), len(cvs)))),
                           "holdout估计": pd.Series(holdout),
                           "CV估计": pd.Series(cvs)})
    return summary, detail


# --------------------------------------------------------------------------
# 5.5 步骤 2：文本清洗消融
# --------------------------------------------------------------------------
def ablation_groups() -> dict[str, list[tuple[str, CleanConfig]]]:
    """三组对照实验的配置。

    组间刻意共享配置：累积阶梯的 L0/L1 就是邮件头分级的 H0/H1，"仅去邮件头"
    与分级的 H2 也是同一个开关组合。运行时会按开关组合去重，
    每个组合只跑一次 CV，再分别填进各组的表——省掉重复计算，结论完全一致。
    """
    L0 = CleanConfig()
    L1 = CleanConfig(header="drop_routing")
    L2 = replace(L1, quotes=True)
    L3 = replace(L2, signature=True)
    L4 = replace(L3, punct=True)
    L5 = replace(L4, stopwords=True)
    L6 = replace(L5, stem=True)
    drop_all = CleanConfig(header="drop_all")
    return {
        "累积阶梯（逐级叠加，每级只多一个因素）": [
            ("L0 · 原文（基线）", L0),
            ("L1 · 删路由头，留 Subject", L1),
            ("L2 · + 去引用行", L2),
            ("L3 · + 去签名档", L3),
            ("L4 · + 去标点", L4),
            ("L5 · + 去停用词", L5),
            ("L6 · + Porter stemming", L6),
        ],
        "邮件头分级（单轴对照）": [
            ("H0 · 保留全部邮件头", L0),
            ("H1 · 只删路由头，留 Subject", L1),
            ("H2 · 邮件头全删（含 Subject）", drop_all),
        ],
        "单项独立（各自单独作用于原文，互不叠加）": [
            ("I1 · 仅去邮件头（全删）", drop_all),
            ("I2 · 仅去引用行", CleanConfig(quotes=True)),
            ("I3 · 仅去签名档", CleanConfig(signature=True)),
            ("I4 · 仅去标点", CleanConfig(punct=True)),
            ("I5 · 仅去停用词", CleanConfig(stopwords=True)),
            ("I6 · 仅 Porter stemming", CleanConfig(stem=True)),
        ],
    }


def _ckpt_paths(tag: str) -> tuple[Path, Path]:
    return (RESULT_DIR / f"ablation_clean_{tag}.csv",
            RESULT_DIR / f"fold_scores_{tag}.csv")


def run_ablation(X, y, cv, n_jobs: int = 1, model_name: str = "LinearSVC",
                 cv_desc: str = "", tag: str = "5x5", resume: bool = True,
                 checkpoint: bool = True):
    """对每个清洗配置跑一次交叉验证。

    实现说明：清洗是**无状态**变换（规则固定，不依赖任何跨文档统计量），
    因此可以先把全量文本统一清洗、再送进 CV；无论清洗放在 CV 前还是放在
    Pipeline 里，每折拿到的数据逐字相同。真正会拟合的 TF-IDF 依然严格留在
    每折训练集内部，泄露防线没有被削弱。这样做避免了把清洗重复 25 遍。

    **断点续跑**：每算完一个配置就立刻把结果和每折分数写盘。重新运行时会读回
    已完成的配置并跳过它们。长跑任务在这台机器上会被系统休眠冻结，这一步
    保证了即使中途关机也不会全部白跑。
    """
    groups = ablation_groups()
    clf = make_models()[model_name]

    unique: dict[tuple, CleanConfig] = {}
    for items in groups.values():
        for _, cfg in items:
            unique.setdefault(cfg.key(), cfg)
    desc_to_key = {describe(cfg): cfg.key() for cfg in unique.values()}

    clean_path, fold_path = _ckpt_paths(tag)
    done: dict[str, dict] = {}
    prev_fold: pd.DataFrame | None = None
    if resume and checkpoint and clean_path.exists() and fold_path.exists():
        prev, prev_fold = pd.read_csv(clean_path), pd.read_csv(fold_path, index_col=0)
        for _, r in prev.iterrows():
            desc = r["配置"]
            if desc in desc_to_key:          # 配置清单变了就忽略旧结果
                row = r.to_dict()
                row["key"] = desc_to_key[desc]
                done[desc] = row
        if done:
            print(f"[续跑] 发现已完成的 {len(done)} 个配置，将跳过：")
            for d in done:
                print(f"         · {d}")

    n_txt = cv_desc or type(cv).__name__
    print(f"\n[消融] {len(unique)} 个配置 × {model_name} × {n_txt}"
          f"（n_jobs={n_jobs}，待跑 {len(unique) - len(done)} 个）\n")

    rows: list[dict] = []
    fold_cols: dict[str, np.ndarray] = {}
    for i, (key, cfg) in enumerate(unique.items(), 1):
        desc = describe(cfg)

        if desc in done:                                   # ---- 复用已完成 ----
            rows.append(done[desc])
            fold_cols[desc] = prev_fold[desc].to_numpy(dtype=float)
            print(f"  [{i:>2}/{len(unique)}] 跳过（已完成）  {desc}")
            continue

        docs = apply_clean(X, cfg)
        # 规模指标：字符数会被"标点→空格"的替换撑住（字符数不变），
        # 所以真正能反映"模型看到多少内容"的是词数与去重词数。
        avg_len = float(np.mean([len(d) for d in docs]))
        avg_tok = float(np.mean([len(d.split()) for d in docs]))
        vocab = len({t for d in docs for t in d.split()})
        sink: dict = {}
        t = time.time()
        _, raw = run_cv({model_name: clf}, docs, y, cv, n_jobs=n_jobs, quiet=True,
                        fold_sink=sink)
        cost = time.time() - t
        fold_cols[desc] = sink[model_name]
        r = raw.iloc[0]
        rows.append({
            "key": key,
            "配置": desc,
            "MacroF1均值": r["macro_f1"],
            "MacroF1标准差": r["macro_f1_std"],
            "Macro-F1": ms(r["macro_f1"], r["macro_f1_std"]),
            "Accuracy": r["accuracy"],
            "Weighted-F1": r["weighted_f1"],
            "训练-MacroF1": r["train_macro_f1"],
            "泛化间隙": r["gap"],
            "平均字符数": avg_len,
            "平均词数": avg_tok,
            "去重词数": vocab,
            "耗时(s)": round(cost, 1),
        })
        print(f"  [{i:>2}/{len(unique)}] MacroF1={rows[-1]['Macro-F1']}  "
              f"词数={avg_tok:6.0f}  词表={vocab:6d}  {desc}")

        # ---- 每完成一个配置就落盘，保证断点可续 ----
        if checkpoint:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).drop(columns=["key"]).to_csv(
                clean_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(fold_cols).to_csv(fold_path, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    base_tok = float(df.loc[df["配置"] == describe(CleanConfig()), "平均词数"].iloc[0])
    df["词数保留率"] = df["平均词数"] / base_tok
    fold_df = pd.DataFrame(fold_cols)
    return df, groups, fold_df


def group_baseline(gname: str) -> CleanConfig:
    """每组算 Δ 时相对谁。

    累积阶梯 / 邮件头分级：组内第一行本身就是基线（原文 / 保留全部邮件头）。
    单项独立：每一项都是"单独作用于原文"，所以基线必须是**原文**——
    组内第一行是"仅去邮件头"，那是一个处理，不是基线。取错会让整组数字失真。
    """
    if gname.startswith("单项独立"):
        return CleanConfig()
    return ablation_groups()[gname][0][1]


def ablation_tables(df: pd.DataFrame, groups: dict) -> dict[str, pd.DataFrame]:
    """把扁平结果按三组实验重排成便于放进报告的表。

    Δ 一律相对**该组的基线**（见 group_baseline），而不是相对上一行——
    三张表的 Δ 列因此含义一致，都能读成"这一步相对本组基线带来多少变化"。
    按"配置"描述字符串查表，这样既能用内存里的结果，也能直接读回 CSV 重新出表。
    """
    by_desc = {r["配置"]: r for _, r in df.iterrows()}
    # 词数保留率**现算**，不依赖 CSV 里有没有那一列：
    # run_ablation 只在收尾时把它加进内存中的 df，而落盘的是循环内的检查点
    # （那时还没算）。若这里直接取 r["词数保留率"]，读回 CSV 重渲染时就会 KeyError。
    base_tok = float(by_desc[describe(CleanConfig())]["平均词数"])
    tables = {}
    for gname, items in groups.items():
        base_cfg = group_baseline(gname)
        base = by_desc[describe(base_cfg)]["MacroF1均值"]
        rows = []
        for label, cfg in items:
            r = by_desc[describe(cfg)]
            is_base = cfg.key() == base_cfg.key()
            rows.append({
                "步骤": label,
                "Macro-F1": r["Macro-F1"],
                "Δ vs 组内基线": "—" if is_base else f"{r['MacroF1均值'] - base:+.4f}",
                "泛化间隙": f"{r['泛化间隙']:.4f}",
                "平均词数": f"{r['平均词数']:.0f}",
                "词数保留率": f"{float(r['平均词数']) / base_tok:.1%}",
                "去重词数": int(r["去重词数"]),
            })
        tables[gname] = pd.DataFrame(rows)
    return tables


def show_clean(X, n: int = 2, preview: int = 700) -> None:
    """打印前 n 条样本在累积阶梯各级的输出，用来肉眼核对清洗规则是否符合预期。

    动辄一小时的消融跑之前，先跑这个看一眼——正则写错了在这里能立刻发现。
    """
    ladder = ablation_groups()["累积阶梯（逐级叠加，每级只多一个因素）"]
    for i in range(min(n, len(X))):
        print("=" * 100)
        print(f"样本 {i} | 原文长度 {len(X[i])}")
        print("-" * 100)
        print("【原文】" + X[i][:preview])
        for label, cfg in ladder[1:]:
            out = clean_text(X[i], cfg)
            print("-" * 100)
            print(f"【{label}】长度 {len(out)}（保留 {len(out)/max(len(X[i]),1):.1%}）")
            print(out[:preview])
    print("=" * 100)
    print(f"以上共 {min(n, len(X))} 条样本 × {len(ladder)} 级配置\n")


def _paired_verdict(a: np.ndarray, b: np.ndarray, n_splits: int, n_repeats: int):
    """给一对逐折分数算配对检验：Δ 均值、95% CI、两种 p 值、结论。

    分两层判读：25 折检验偏乐观（各折不独立），按重复聚合后偏保守。
    两者都显著才判"显著"。若本来就没做重复（n_repeats == 1），
    第二层无从计算，此时只依据第一层，并在结论里注明——
    不能因为"第二层算不出来"就一律判存疑，那会把真实结论埋掉。
    """
    d = a - b
    t25, p25 = stats.ttest_rel(a, b)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    if n_repeats > 1:
        ra = a.reshape(n_repeats, n_splits).mean(axis=1)
        rb = b.reshape(n_repeats, n_splits).mean(axis=1)
        _, p5 = stats.ttest_rel(ra, rb)
        p5_s = f"{p5:.3f}"
    else:
        p5 = None
        p5_s = "—（未做重复）"
    if p25 >= 0.05:
        verdict = "不显著"
    elif p5 is None:
        verdict = "显著（仅单层检验）"
    elif p5 < 0.05:
        verdict = "显著"
    else:
        verdict = "存疑"
    return d, p25, lo, hi, p5_s, verdict


def paired_tests(fold_df: pd.DataFrame, baseline: str | None = None,
                 n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """各配置与基线做**配对** t 检验。

    为什么要配对：同一批折里每一折的难度不同，而这个难度是各配置**共同**的。
    逐折配对做差可以把这份共同方差消掉，因此比"两个独立的均值 ± 标准差有没有重叠"
    灵敏得多——后者基本判不出 0.003 这个量级的差异。

    必须诚实说明的局限：重复交叉验证的各折不独立（同一批样本被反复用作训练数据），
    直接对全部折做配对 t 检验会让 p 值偏乐观。因此同时给出按重复聚合后的检验，
    它保守得多。只有两者都显著才判"显著"，避免过度声称。
    """
    base = describe(CleanConfig()) if baseline is None else baseline
    n_folds = len(fold_df)
    pcol, rcol = f"p（{n_folds} 折）", f"p（{n_repeats} 次重复聚合）"
    rows = []
    for name in fold_df.columns:
        if name == base:
            rows.append({"配置": name, "Δ 均值": "—", "Δ 的 95% CI": "—",
                         pcol: "—", rcol: "—", "结论": "基线"})
            continue
        a = fold_df[name].to_numpy(dtype=float)
        b = fold_df[base].to_numpy(dtype=float)
        d, p25, lo, hi, p5_s, verdict = _paired_verdict(a, b, n_splits, n_repeats)
        rows.append({
            "配置": name,
            "Δ 均值": f"{d.mean():+.4f}",
            "Δ 的 95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
            pcol: f"{p25:.1e}",
            rcol: p5_s,
            "结论": verdict,
        })
    return pd.DataFrame(rows)


def run_model_fold_scores(X, y, cv, cfg: CleanConfig | None = None, n_jobs: int = 1,
                          tfidf: dict | None = None,
                          ckpt_path: Path | None = None,
                          metrics_path: Path | None = None,
                          resume: bool = True):
    """采集四个模型的**逐折分数**与**完整指标**，返回 (逐折表, 指标表)。

    两张表各有用途，不要混：
      * **逐折分数**（每折一个 Macro-F1）→ 配对检验的唯一输入，均值不够用；
      * **指标汇总**（Accuracy / Macro-F1 / Weighted-F1 / 训练分 / 间隙 / 耗时）→ 最终结果表。

    **逐模型落盘**：MLP 在 50000 维稀疏输入上极慢（sklearn 会走低效路径，实测单折 10 秒/迭代），
    全量 25 折要两小时以上。所以每跑完一个模型立刻把该模型的逐折分数与指标写入磁盘，
    重跑时按列比对、只补缺失的模型。没有这层保护，一次关机就白等两小时。
    """
    docs = X if cfg is None else apply_clean(X, cfg)

    fold_df = None
    metrics_df = None
    if resume and ckpt_path is not None and ckpt_path.exists():
        fold_df = pd.read_csv(ckpt_path, index_col=0)
        print(f"[续跑] {ckpt_path.name} 已有模型：{list(fold_df.columns)}")
    if metrics_path is not None and metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    # 已完成的判定必须**两张表都有**该模型：只有逐折分数、没有指标表时，
    # 若按"逐折分数里有这一列"就跳过，会永远补不出指标（旧版只存了逐折分数）。
    have = set(fold_df.columns) if fold_df is not None else set()
    if metrics_df is not None and "模型" in metrics_df.columns:
        have &= set(metrics_df["模型"])
    else:
        have = set()
    # 若有模型需要重算，就把**已有的逐折分数一并丢掉**再重来。
    # 否则新算出的列会与旧列重名，pandas 会把它们改名为 `xxx.1` 并保留两份，
    # 后面的配对检验就会把每个模型当成两个模型、算出 28 组两两比较。
    if not have:
        fold_df = None

    for name, clf in make_models().items():
        if name in have:
            print(f"  - {name:<20} 跳过（已完成）")
            continue
        pipe = make_pipeline(cfg, **(tfidf or {}))
        pipe.set_params(clf=clf)
        t = time.time()
        r = cross_validate(pipe, docs, y, cv=cv, scoring=SCORING,
                           return_train_score=True, n_jobs=n_jobs)
        cost = time.time() - t

        col = pd.Series(r["test_macro_f1"], name=name)
        fold_df = col.to_frame() if fold_df is None else pd.concat([fold_df, col], axis=1)

        row = pd.DataFrame([{
            "模型": name,
            "MacroF1均值": float(r["test_macro_f1"].mean()),      # 数值列，供程序化排序
            "Accuracy": ms(r["test_accuracy"].mean(), r["test_accuracy"].std(ddof=1)),
            "Macro-F1": ms(r["test_macro_f1"].mean(), r["test_macro_f1"].std(ddof=1)),
            "Weighted-F1": ms(r["test_weighted_f1"].mean(), r["test_weighted_f1"].std(ddof=1)),
            "训练-MacroF1": ms(r["train_macro_f1"].mean(), r["train_macro_f1"].std(ddof=1)),
            "泛化间隙": f"{(r['train_macro_f1'] - r['test_macro_f1']).mean():.4f}",
            "折数": len(r["test_macro_f1"]),
            "耗时(s)": round(cost, 1),
        }])
        metrics_df = row if metrics_df is None else pd.concat([metrics_df, row], ignore_index=True)
        print(f"  - {name:<20} MacroF1={row.iloc[0]['Macro-F1']}  ({cost:.1f}s)")

        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            fold_df.to_csv(ckpt_path, encoding="utf-8-sig")
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    return fold_df, metrics_df


def pairwise_model_tests(fold_df: pd.DataFrame, n_splits: int = N_SPLITS,
                         n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """对四个模型做两两配对 t 检验。

    为什么必须配对而不是看各自的均值±标准差：同一折的难度是四个模型共有的，
    配对做差消掉这份共同方差后，才会显出 0.003 这种量级的真实差异。
    判读规则与消融检验一致：25 折与按重复聚合两种检验都显著才判"显著"。
    """
    names = list(fold_df.columns)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = fold_df[names[i]].to_numpy(dtype=float)
            b = fold_df[names[j]].to_numpy(dtype=float)
            d, p25, lo, hi, p5_s, verdict = _paired_verdict(a, b, n_splits, n_repeats)
            rows.append({
                "模型 A": names[i],
                "模型 B": names[j],
                "Δ (A−B)": f"{d.mean():+.4f}",
                "Δ 的 95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                f"p（{len(d)} 折）": f"{p25:.1e}",
                f"p（{n_repeats} 次重复聚合）": p5_s,
                "结论": verdict,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5.6 步骤 3：TF-IDF 参数调优
# --------------------------------------------------------------------------
# 基线值即 scikit-learn 的默认值（max_features 除外，我们用 5000）。
# 单参数敏感性分析时，每次只动一个参数，其余固定在基线，这样才能把
# "这个参数的效果"与"别的参数的效果"分开——否则又变成多个变量一起动。
TFIDF_BASELINE: dict = {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0,
                        "max_features": MAX_FEATURES, "sublinear_tf": False}
TFIDF_LEVELS: dict = {
    "ngram_range": [(1, 1), (1, 2)],
    "min_df": [1, 2, 5, 10],
    "max_df": [0.5, 0.9, 1.0],
    "max_features": [2000, 5000, 20000],
    "sublinear_tf": [False, True],
}


def tfidf_key(kw: dict) -> tuple:
    """把参数字典变成可做去重键的元组。"""
    return tuple(sorted((k, repr(v)) for k, v in kw.items()))


def tfidf_sweep_configs() -> dict[str, list[tuple[str, dict]]]:
    """构造单参数敏感性扫描：每个参数单独变化，其余保持基线。"""
    groups = {}
    for param, levels in TFIDF_LEVELS.items():
        items = []
        for lv in levels:
            kw = dict(TFIDF_BASELINE)
            kw[param] = lv
            items.append((f"{param} = {lv}", kw))
        groups[param] = items
    return groups


def tfidf_label(kw: dict) -> str:
    """把参数字典压成一行可读标签，只显示相对基线有变化的项。"""
    diff = [f"{k}={v!r}" for k, v in kw.items() if TFIDF_BASELINE.get(k) != v]
    return "、".join(diff) if diff else "基线（全部默认）"


def _parse_literal(v: str):
    """把命令行里的字符串还原成合适的 Python 字面量。"""
    low = v.lower()
    if low == "none":
        return None
    if low in ("true", "false"):
        return low == "true"
    if v.startswith("(") and v.endswith(")"):
        return tuple(int(x) for x in v[1:-1].split(","))
    try:
        return int(v)
    except ValueError:
        return v


def parse_combo(spec: str) -> dict:
    """把 'max_features=20000,sublinear_tf=True' 解析成完整参数字典（未提及的项取基线值）。

    用于在单参数扫描之后测试**参数组合**——单参数扫描看不出交互效应，
    而 max_features 与 sublinear_tf 各自有收益，组合起来是否叠加必须单独验证。

    注意不能简单按逗号切分：`ngram_range=(1,2)` 的值里**本身就含逗号**，
    会让朴素切分把 `2)` 当成一个片段。这里用正则把"带括号的值"整体识别出来。
    """
    kw = dict(TFIDF_BASELINE)
    pairs = re.findall(r"(\w+)\s*=\s*(\([^)]*\)|[^,]+)", spec)
    if not pairs:
        raise ValueError(f"无法从 {spec!r} 解析出任何 key=value 片段")
    for k, v in pairs:
        k = k.strip()
        if k not in TFIDF_BASELINE:
            raise ValueError(f"未知 TF-IDF 参数 {k!r}；可选：{list(TFIDF_BASELINE)}")
        kw[k] = _parse_literal(v.strip())
    return kw


def run_tfidf_sweep(X, y, cv, preset: str, n_jobs: int = 1,
                    model_name: str = "LinearSVC", tag: str = "5x5",
                    checkpoint: bool = True, extra: list[dict] | None = None):
    """对 TF-IDF 参数做单参数敏感性扫描 + 每折分数采集。

    与消融实验共用同一套做法：清洗无状态可提前做，TF-IDF 的拟合严格留在折内；
    每完成一个配置立刻落盘，支持断点续跑。
    """
    docs = X if preset == "raw" else apply_clean(X, CLEAN_PRESETS[preset])
    groups = tfidf_sweep_configs()
    clf = make_models()[model_name]

    unique: dict[tuple, dict] = {}
    for items in groups.values():
        for _, kw in items:
            unique.setdefault(tfidf_key(kw), kw)
    for kw in (extra or []):                    # 额外的参数组合
        unique.setdefault(tfidf_key(kw), kw)

    ck_path = RESULT_DIR / f"tfidf_sweep_{preset}_{tag}.csv"
    fold_path = RESULT_DIR / f"tfidf_fold_scores_{preset}_{tag}.csv"
    done: dict[str, dict] = {}
    prev_fold = None
    if checkpoint and ck_path.exists() and fold_path.exists():
        prev, prev_fold = pd.read_csv(ck_path), pd.read_csv(fold_path, index_col=0)
        for _, r in prev.iterrows():
            if r["参数"] in prev_fold.columns:
                done[r["参数"]] = r.to_dict()
        if done:
            print(f"[续跑] 已完成 {len(done)} 个参数配置，将跳过")

    print(f"\n[TF-IDF 扫描] 清洗前提 {preset}（{describe(CLEAN_PRESETS[preset])}）"
          f" × {model_name} × {len(unique)} 个配置"
          f"（待跑 {len(unique) - len(done)} 个）\n")

    rows, fold_cols = [], {}
    # ── 关键：把"上次跑过、但本次没有请求"的配置一并带上 ──
    # 断点续跑的正面是省时，反面是**静默覆盖**：一次只请求部分组合的运行，
    # 若输出时只写本次请求的那些，就会把先前算好的其它组合从 CSV 里悄悄抹掉。
    # 这个坑本项目踩过三次（每次都是"代码看起来对、结果少几列"），所以这里显式写清楚。
    requested = {tfidf_label(kw) for kw in unique.values()}
    for _lab, _row in done.items():
        if _lab not in requested:
            rows.append(_row)
            fold_cols[_lab] = prev_fold[_lab].to_numpy(dtype=float)
    for i, (key, kw) in enumerate(unique.items(), 1):
        label = tfidf_label(kw)
        if label in done:
            rows.append(done[label])
            fold_cols[label] = prev_fold[label].to_numpy(dtype=float)
            print(f"  [{i:>2}/{len(unique)}] 跳过（已完成）  {label}")
            continue

        t = time.time()
        pipe = make_pipeline()
        pipe.set_params(clf=clf, **{f"tfidf__{k}": v for k, v in kw.items()})
        r = cross_validate(pipe, docs, y, cv=cv, scoring=SCORING,
                           return_train_score=True, n_jobs=n_jobs)
        cost = time.time() - t
        fold_cols[label] = r["test_macro_f1"]
        rows.append({
            "参数": label,
            "MacroF1均值": float(r["test_macro_f1"].mean()),
            "MacroF1标准差": float(r["test_macro_f1"].std(ddof=1)),
            "Macro-F1": ms(r["test_macro_f1"].mean(), r["test_macro_f1"].std(ddof=1)),
            "Accuracy": float(r["test_accuracy"].mean()),
            "训练-MacroF1": float(r["train_macro_f1"].mean()),
            "泛化间隙": float((r["train_macro_f1"] - r["test_macro_f1"]).mean()),
            "耗时(s)": round(cost, 1),
        })
        print(f"  [{i:>2}/{len(unique)}] MacroF1={rows[-1]['Macro-F1']}  "
              f"({cost:.1f}s)  {label}")

        if checkpoint:
            # 写盘前先与已有文件取并集——保证文件在任何时刻都不会缩水
            out_df, out_fold = _union_with_prev(pd.DataFrame(rows),
                                                pd.DataFrame(fold_cols),
                                                ck_path, fold_path)
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(ck_path, index=False, encoding="utf-8-sig")
            out_fold.to_csv(fold_path, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    fold_df = pd.DataFrame(fold_cols)
    if checkpoint:
        df, fold_df = _union_with_prev(df, fold_df, ck_path, fold_path, verbose=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(ck_path, index=False, encoding="utf-8-sig")
        fold_df.to_csv(fold_path, encoding="utf-8-sig")
    return df, groups, fold_df


def tfidf_sweep_tables(df: pd.DataFrame, groups: dict) -> dict[str, pd.DataFrame]:
    """按参数分组成表，Δ 一律相对**基线配置**，便于横向读每个参数的效果。"""
    by_label = {r["参数"]: r for _, r in df.iterrows()}
    base_label = tfidf_label(TFIDF_BASELINE)
    base = by_label.get(base_label)
    tables = {}
    for param, items in groups.items():
        rows = []
        for label, kw in items:
            r = by_label[tfidf_label(kw)]
            is_base = tfidf_label(kw) == base_label
            delta = "—"
            if not is_base and base is not None:
                delta = f"{r['MacroF1均值'] - base['MacroF1均值']:+.4f}"
            rows.append({
                "取值": str(kw[param]),
                "Macro-F1": r["Macro-F1"],
                "Δ vs 基线": delta,
                "泛化间隙": f"{r['泛化间隙']:.4f}",
                "耗时(s)": r["耗时(s)"],
            })
        tables[param] = pd.DataFrame(rows)
    return tables


# --------------------------------------------------------------------------
# 5.7 步骤 4：模型超参调优
# --------------------------------------------------------------------------
# 固定使用步骤 3 选出的最优 TF-IDF 配置与 routing 清洗前提。
# 调模型超参时其余条件必须锁死，否则差异无法归因到超参上。
BEST_TFIDF: dict = {"max_features": 50000, "sublinear_tf": True}
BEST_PRESET = "routing"

MODEL_SWEEP_SPACE: dict[str, list[dict]] = {
    "LinearSVC": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "LogisticRegression": [{"C": c, "penalty": p}
                           for p in ("l2", "l1")
                           for c in (0.01, 0.1, 1.0, 10.0, 100.0)],
}
# 每个模型的"默认"配置，作为配对检验的基准
MODEL_SWEEP_BASE: dict[str, dict] = {
    "LinearSVC": {"C": 1.0},
    "LogisticRegression": {"C": 1.0, "penalty": "l2"},
}


def model_label(model_name: str, params: dict) -> str:
    """模型 + 超参的一行标签，例如 'LogisticRegression · C=0.1、penalty=l1'。"""
    return model_name + " · " + "、".join(f"{k}={v}" for k, v in sorted(params.items()))


def make_sweep_estimator(model_name: str, params: dict):
    """按超参构造分类器。

    LogisticRegression 的 L1 与 L2 **统一用 liblinear 求解器**：lbfgs 不支持 L1，
    若 L1 用 liblinear、L2 用 lbfgs，就把"惩罚项"和"求解器"两个变量混在了一起。
    实测 l2 下两种求解器结果几乎相同（0.9282 vs 0.9287），所以统一用 liblinear 是安全的。
    """
    p = dict(params)
    if model_name == "LinearSVC":
        return LinearSVC(random_state=SEED, **p)
    if model_name == "LogisticRegression":
        p.setdefault("max_iter", 1000)
        p["solver"] = "liblinear"
        return LogisticRegression(random_state=SEED, **p)
    raise ValueError(f"模型超参扫描暂不支持 {model_name!r}")


def run_model_sweep(X, y, cv, models: list[str] | None = None,
                    tfidf: dict | None = None, preset: str = BEST_PRESET,
                    n_jobs: int = 1, tag: str = "5x5", checkpoint: bool = True):
    """扫描 LinearSVC 的 C 与 LogisticRegression 的 C × penalty。

    与前面几个扫描共用同一套做法：清洗无状态可提前做、TF-IDF 拟合严格留在折内、
    每完成一个配置立刻落盘以便断点续跑。
    """
    tf = dict(BEST_TFIDF if tfidf is None else tfidf)
    models = models or list(MODEL_SWEEP_SPACE)
    docs = apply_clean(X, CLEAN_PRESETS[preset])

    items = [(model_label(m, p), m, p)
             for m in models for p in MODEL_SWEEP_SPACE[m]]

    ck_path = RESULT_DIR / f"model_sweep_{preset}_{tag}.csv"
    fold_path = RESULT_DIR / f"model_sweep_fold_scores_{preset}_{tag}.csv"
    done: dict[str, dict] = {}
    prev_fold = None
    if checkpoint and ck_path.exists() and fold_path.exists():
        prev, prev_fold = pd.read_csv(ck_path), pd.read_csv(fold_path, index_col=0)
        for _, r in prev.iterrows():
            if r["配置"] in prev_fold.columns:
                done[r["配置"]] = r.to_dict()
        if done:
            print(f"[续跑] 已完成 {len(done)} 个模型配置，将跳过")

    print(f"\n[模型超参扫描] 前提 {preset}，TF-IDF 固定为 {tfidf_label(tf)}")
    print(f"共 {len(items)} 个配置（待跑 {len(items) - len(done)} 个）\n")

    rows, fold_cols = [], {}
    for i, (label, mname, params) in enumerate(items, 1):
        if label in done:
            rows.append(done[label])
            fold_cols[label] = prev_fold[label].to_numpy(dtype=float)
            print(f"  [{i:>2}/{len(items)}] 跳过（已完成）  {label}")
            continue

        pipe = make_pipeline(**tf)
        pipe.set_params(clf=make_sweep_estimator(mname, params))
        t = time.time()
        r = cross_validate(pipe, docs, y, cv=cv, scoring=SCORING,
                           return_train_score=True, n_jobs=n_jobs)
        cost = time.time() - t
        fold_cols[label] = r["test_macro_f1"]
        rows.append({
            "模型": mname,
            "配置": label,
            "超参": "、".join(f"{k}={v}" for k, v in sorted(params.items())),
            "MacroF1均值": float(r["test_macro_f1"].mean()),
            "MacroF1标准差": float(r["test_macro_f1"].std(ddof=1)),
            "Macro-F1": ms(r["test_macro_f1"].mean(), r["test_macro_f1"].std(ddof=1)),
            "Accuracy": float(r["test_accuracy"].mean()),
            "训练-MacroF1": float(r["train_macro_f1"].mean()),
            "泛化间隙": float((r["train_macro_f1"] - r["test_macro_f1"]).mean()),
            "耗时(s)": round(cost, 1),
        })
        print(f"  [{i:>2}/{len(items)}] MacroF1={rows[-1]['Macro-F1']}  "
              f"间隙={rows[-1]['泛化间隙']:.4f}  ({cost:.1f}s)  {label}")

        if checkpoint:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(ck_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(fold_cols).to_csv(fold_path, encoding="utf-8-sig")

    return pd.DataFrame(rows), pd.DataFrame(fold_cols)


def model_sweep_table(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """单个模型的超参对照表，Δ 相对该模型的默认配置。"""
    sub = df[df["模型"] == model_name].copy()
    base_params = MODEL_SWEEP_BASE.get(model_name)
    if base_params is None:
        return sub
    base_label = model_label(model_name, base_params)
    base = df.loc[df["配置"] == base_label, "MacroF1均值"]
    base_v = float(base.iloc[0]) if len(base) else None
    out = []
    for _, r in sub.iterrows():
        delta = "—"
        if r["配置"] != base_label and base_v is not None:
            delta = f"{r['MacroF1均值'] - base_v:+.4f}"
        out.append({
            "超参": r["超参"],
            "Macro-F1": r["Macro-F1"],
            "Δ vs 默认": delta,
            "泛化间隙": f"{r['泛化间隙']:.4f}",
            "耗时(s)": r["耗时(s)"],
        })
    return pd.DataFrame(out)


def nested_cv(X, y, model_name: str, grid: dict, tfidf: dict | None = None,
              preset: str = BEST_PRESET, outer_splits: int = 5,
              inner_splits: int = 3, n_jobs: int = 1):
    """嵌套交叉验证：外层估性能、内层调超参。

    为什么必须嵌套：如果"用同一份 CV 选出最优 C、又把那份 CV 的分数当作最终性能"，
    选优过程本身已经把分数"喂高"了——报出来的是同一批数据的第二次使用，
    偏乐观。嵌套 CV 让调参永远看不到外层验证折，给出的才是**调参流程**的无偏估计。

    返回 (外层各折 Macro-F1, 内层各折选出的最优 CV 分数, 各折选出的最优超参)。

    **关于乐观偏差的正确度量**：不能拿"嵌套 CV 的外层分数"去比"另一次独立扫描的最优分数"——
    两者折结构不同、不配对，测的不是同一件事。正确的做法是**在每个外层折内部**比较：
    内层 CV 选出的最优分数 `best_score_`（乐观，因为是从同一批数据里挑出来的最大值）
    与外层测试分数（无偏，该折数据没参与选优）。同一份训练数据 → 天然配对。
    """
    tf = dict(BEST_TFIDF if tfidf is None else tfidf)
    docs = np.asarray(apply_clean(X, CLEAN_PRESETS[preset]), dtype=object)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
    param_grid = {f"clf__{k}": v for k, v in grid.items()}

    scores, inner_best, chosen = [], [], []
    for fold_i, (tr, te) in enumerate(outer.split(docs, y), 1):
        pipe = make_pipeline(**tf)
        pipe.set_params(clf=make_sweep_estimator(model_name, {}))
        gs = GridSearchCV(pipe, param_grid, cv=inner, scoring="f1_macro", n_jobs=n_jobs)
        gs.fit(docs[tr], y[tr])
        pred = gs.predict(docs[te])
        s = f1_score(y[te], pred, average="macro")
        scores.append(float(s))
        inner_best.append(float(gs.best_score_))
        best = {kk.replace("clf__", ""): vv for kk, vv in gs.best_params_.items()}
        chosen.append(best)
        print(f"  - 外层第 {fold_i}/{outer_splits} 折  内层最优={gs.best_score_:.4f}  "
              f"外层实测={s:.4f}  虚高={gs.best_score_ - s:+.4f}  选出 {best}")
    return np.array(scores), np.array(inner_best), chosen


def sweep_grid(model_name: str) -> dict:
    """从扫描空间反推出网格字典，保证嵌套 CV 与扫描搜索的是同一组超参。"""
    grid: dict = {}
    for p in MODEL_SWEEP_SPACE[model_name]:
        for k, v in p.items():
            grid.setdefault(k, [])
            if v not in grid[k]:
                grid[k].append(v)
    return grid


# --------------------------------------------------------------------------
# 5.8 数据增强
# --------------------------------------------------------------------------
# 为什么不用同义词替换/回译：前者要下载 WordNet 语料，后者要翻译模型，
# 都引入体积大、且与"文本分类"本身无关的依赖。这里只用**无依赖的结构扰动**：
# 随机删除与随机交换词序——它们不改动任何词，因而不需要额外的语义资源，
# 也不会因同义词替换不当而破坏标签语义。
AUG_MODES: list[tuple[str, str, float]] = [
    ("none", "不做增强（基线）", 0.0),
    ("delete", "随机删除 10% 词", 0.10),
    ("delete20", "随机删除 20% 词", 0.20),
    ("swap", "随机交换 10% 词序", 0.10),
]


def _augment_text(s: str, rng: np.random.RandomState, mode: str, rate: float) -> str:
    """对单条文本做一次结构扰动。过短的文本不动，避免把它打烂。"""
    toks = s.split()
    if len(toks) < 8:
        return s
    n = max(1, int(len(toks) * rate))
    if mode.startswith("delete"):
        drop = set(rng.choice(len(toks), size=n, replace=False).tolist())
        return " ".join(t for i, t in enumerate(toks) if i not in drop)
    if mode == "swap":
        for _ in range(n):
            i, j = int(rng.randint(len(toks))), int(rng.randint(len(toks)))
            toks[i], toks[j] = toks[j], toks[i]
        return " ".join(toks)
    return s


def run_augmentation(X, y, cv, preset: str = BEST_PRESET, tfidf: dict | None = None,
                     modes: list | None = None, model_name: str = "LinearSVC"):
    """对比几种数据增强方式（只看干净验证集上的准确率）。

    **必须在交叉验证的每一折内部做增广**：只增广该折的训练部分，
    验证部分是原始文本、一条都不动。若先增广再划分，同一篇文本的原始版与
    扰动版会同时落进训练和验证，等于让模型在验证集上见到训练样本——
    这是数据增强最容易踩的泄露陷阱，比 TF-IDF 泄露更隐蔽。

    注意：本函数只测**泛化能力**（干净验证集上的准确率）。数据增强的另一个
    经典目的是**鲁棒性**，那需要 `run_augmentation_robust()`——它额外在
    被扰动的验证集上评估，看的是"降幅"。
    """
    tf = dict(BEST_TFIDF if tfidf is None else tfidf)
    modes = modes or AUG_MODES
    docs = apply_clean(X, CLEAN_PRESETS[preset])

    rows, fold_cols = [], {}
    for mode, desc, rate in modes:
        scores = []
        t0 = time.time()
        for fi, (tr, te) in enumerate(cv.split(docs, y), 1):
            Xtr = [docs[i] for i in tr]
            ytr = y[tr]
            if mode != "none":
                # 每折用固定种子重建 rng，保证结果可复现
                rng = np.random.RandomState(SEED * 1000 + fi)
                Xtr = Xtr + [_augment_text(s, rng, mode, rate) for s in Xtr]
                ytr = np.concatenate([ytr, ytr])
            pipe = make_pipeline(**tf)
            pipe.set_params(clf=LinearSVC(random_state=SEED, C=1.0))
            pipe.fit(Xtr, ytr)
            scores.append(f1_score(y[te], pipe.predict([docs[i] for i in te]),
                                   average="macro"))
        cost = time.time() - t0
        arr = np.array(scores)
        fold_cols[desc] = arr
        rows.append({
            "增强方式": desc,
            "MacroF1均值": float(arr.mean()),
            "MacroF1标准差": float(arr.std(ddof=1)),
            "Macro-F1": ms(arr.mean(), arr.std(ddof=1)),
            "训练集规模": "7368（不变）" if mode == "none" else "×2（原始 + 一份扰动）",
            "耗时(s)": round(cost, 1),
        })
        print(f"  {desc:<20} MacroF1={rows[-1]['Macro-F1']}  ({cost:.1f}s)")
    return pd.DataFrame(rows), pd.DataFrame(fold_cols)


def class_keywords(docs, y, n_classes: int = N_CLASSES, top_k: int = 40):
    """每个类别的"特征词"：类内文档频率高、但全局不算通用的词。

    两个用途：
      1. 「类内插入」增强——把该类自己的特征词插进文档，**按构造保证标签不变**；
      2. 标签保持性检查——看删除扰动后，这些词还剩几个。
    """
    from collections import Counter
    global_df: Counter = Counter()
    for d in docs:
        global_df.update(set(d.split()))
    n_doc = max(len(docs), 1)

    kw = {}
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        df: Counter = Counter()
        for i in idx:
            df.update(set(docs[i].split()))
        lo = max(3, int(0.05 * len(idx)))
        cand = [(w, k) for w, k in df.items()
                if k >= lo and global_df[w] < 0.5 * n_doc and len(w) > 2]
        kw[c] = [w for w, _ in sorted(cand, key=lambda t: -t[1])[:top_k]]
    return kw


def _augment_insert(text: str, rng, words: list[str], n: int = 0, rate: float = 0.10) -> str:
    """类内词插入：把该类别的特征词追加到文档末尾。

    按构造保标签——插入的词本来就来自同一类别，不会把文档的类别语义带偏。
    与删除类扰动相比，它只增不减，不会破坏原有的类别线索。
    """
    if not words:
        return text
    toks = text.split()
    k = n or max(1, int(len(toks) * rate))
    add = [words[int(rng.randint(len(words)))] for _ in range(k)]
    return text + " " + " ".join(add)


def run_augmentation_robust(X, y, cv, preset: str = BEST_PRESET,
                            tfidf: dict | None = None, modes: list | None = None,
                            eval_rate: float = 0.10, model_name: str = "LinearSVC"):
    """数据增强的**鲁棒性**评估——训练用不用增强 × 评估在干净/扰动输入上。

    为什么原来的实验不够：原设计只在**干净的验证集**上评估，测的是泛化能力。
    但数据增强的经典目的还有**鲁棒性**——让模型对输入损坏更耐受。
    正确测法是看**降幅**（干净分 − 扰动分），而不是干净分本身：
      * 增强有效 ⟹ 受过扰动训练的模型，在扰动输入上掉分更少
      * 只看干净分，是测不出鲁棒性的

    验证集只在**评估时**被扰动，不参与任何拟合，因此不构成泄露；
    且评估用的扰动种子与训练用的不同，不会与训练时的扰动重合。

    返回 (总表, 逐折分数表)。
    """
    tf = dict(BEST_TFIDF if tfidf is None else tfidf)
    modes = modes if modes is not None else AUG_MODES + [("insert", "类内词插入 10%", 0.10)]
    docs = np.asarray(apply_clean(X, CLEAN_PRESETS[preset]), dtype=object)

    rows, fold_cols = [], {}
    for mode, desc, rate in modes:
        clean_s, pert_s = [], []
        t0 = time.time()
        for fi, (tr, te) in enumerate(cv.split(docs, y), 1):
            Xtr = [docs[i] for i in tr]
            ytr = y[tr]
            if mode != "none":
                rng = np.random.RandomState(SEED * 1000 + fi)
                if mode == "insert":
                    kw = class_keywords([docs[i] for i in tr], ytr)
                    add = [_augment_insert(s, rng, kw[int(c)], rate=rate)
                           for s, c in zip(Xtr, ytr)]
                else:
                    add = [_augment_text(s, rng, mode, rate) for s in Xtr]
                Xtr = Xtr + add
                ytr = np.concatenate([ytr, ytr])
            pipe = make_pipeline(**tf)
            pipe.set_params(clf=LinearSVC(random_state=SEED, C=1.0))
            pipe.fit(Xtr, ytr)

            te_idx = list(te)
            clean_s.append(f1_score(y[te], pipe.predict([docs[i] for i in te_idx]),
                                    average="macro"))
            # 评估时的扰动：换一个种子，与训练扰动不重合
            prng = np.random.RandomState(SEED * 7777 + fi)
            Xte_p = [_augment_text(docs[i], prng, "delete", eval_rate) for i in te_idx]
            pert_s.append(f1_score(y[te], pipe.predict(Xte_p), average="macro"))

        c, p = np.array(clean_s), np.array(pert_s)
        fold_cols[desc + "｜干净"] = c
        fold_cols[desc + "｜扰动"] = p
        rows.append({
            "训练增强": desc,
            "干净验证集": ms(c.mean(), c.std(ddof=1)),
            "扰动验证集": ms(p.mean(), p.std(ddof=1)),
            "降幅（干净−扰动）": f"{(c - p).mean():.4f}",
            "干净分": float(c.mean()),
            "扰动分": float(p.mean()),
            "降幅": float((c - p).mean()),
            "耗时(s)": round(time.time() - t0, 1),
        })
        print(f"  {desc:<20} 干净={c.mean():.4f}  扰动={p.mean():.4f}  "
              f"降幅={(c - p).mean():.4f}")
    return pd.DataFrame(rows), pd.DataFrame(fold_cols)


def label_integrity(docs, y, rate: float = 0.20, top_k: int = 5) -> pd.DataFrame:
    """标签保持性抽检：删除扰动后，该类的特征词还剩几个。

    这是回应"增强方法必须保证标签在变换后仍然成立"的直接检查。
    若一类文档的关键词被删光，其标签就名存实亡，等于往训练集里掺噪声。
    """
    kw = class_keywords(list(docs), y, top_k=40)
    rng = np.random.RandomState(SEED)
    rows = []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        keys = set(kw[c][:top_k])
        lost_all = 0
        remain = []
        for i in idx:
            toks = docs[i].split()
            before = len(keys & set(toks))
            after = len(keys & set(
                _augment_text(docs[i], np.random.RandomState(int(i) + 1), "delete", rate).split()))
            remain.append(after / before if before else 1.0)
            if before and after == 0:
                lost_all += 1
        rows.append({
            "类别": c,
            "样本数": len(idx),
            "关键词保留率": f"{np.mean(remain):.1%}",
            "关键词全丢的文档数": lost_all,
            "占比": f"{lost_all/len(idx):.1%}",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5.10 学习曲线：过拟合程度随训练样本量怎么变
# --------------------------------------------------------------------------
LEARNING_SIZES = [600, 1500, 3000, 0]      # 0 表示全量


def _stratified_subsample(y, n: int, seed: int = SEED) -> np.ndarray:
    """按类别比例抽 n 个样本，返回下标（已排序）。"""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    take = np.floor(counts / counts.sum() * n).astype(int)
    while take.sum() < n:                      # 补齐取整造成的差额
        take[int(rng.randint(len(classes)))] += 1
    idx = [rng.choice(np.where(y == c)[0], size=min(k, int((y == c).sum())),
                      replace=False)
           for c, k in zip(classes, take)]
    return np.sort(np.concatenate(idx))


def run_learning_curve(X, y, folds: int = N_SPLITS, sizes: list | None = None,
                       tfidf: dict | None = None, preset: str = BEST_PRESET,
                       n_jobs: int = 1):
    """在不同训练样本量下跑同一套 CV，记录训练分与**泛化间隙**。

    用途：支撑"过拟合程度随数据量变化"这一论断。
    报告初稿引用过一组 600 条规模的间隙数字，但那来自一次临时运行、没有落盘证据；
    这里把它做成可复现的实验（结果存 results/learning_curve.csv）。

    注意：这是**单次** 5 折 CV（不是 5×5），因为目的是看趋势、不是精确比较，
    且要跑 4 个规模 × 4 个模型，重复 5 次代价过高。趋势结论不依赖重复。
    """
    tf = dict(BEST_TFIDF if tfidf is None else tfidf)
    sizes = sizes or LEARNING_SIZES
    docs = np.asarray(apply_clean(X, CLEAN_PRESETS[preset]), dtype=object)

    rows = []
    for n in sizes:
        if n and n < len(y):
            idx = _stratified_subsample(y, n)
            d_sub, y_sub = docs[idx], y[idx]
        else:
            d_sub, y_sub = docs, y
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        print(f"\n  --- 训练样本量 {len(d_sub)} ---")
        for name, clf in make_models().items():
            pipe = make_pipeline(**tf)
            pipe.set_params(clf=clf)
            r = cross_validate(pipe, d_sub, y_sub, cv=cv, scoring=SCORING,
                               return_train_score=True, n_jobs=n_jobs)
            tr, te = r["train_macro_f1"], r["test_macro_f1"]
            rows.append({
                "训练样本量": len(d_sub),
                "模型": name,
                "验证-MacroF1": ms(te.mean(), te.std(ddof=1)),
                "训练-MacroF1": ms(tr.mean(), tr.std(ddof=1)),
                "泛化间隙": f"{tr.mean() - te.mean():.4f}",
                "gap": float(tr.mean() - te.mean()),
            })
            print(f"    {name:<20} 训练={tr.mean():.4f}  验证={te.mean():.4f}  "
                  f"间隙={tr.mean() - te.mean():.4f}")
    return pd.DataFrame(rows)


def _union_with_prev(df: pd.DataFrame, fold_df: pd.DataFrame,
                     ck_path: Path, fold_path: Path, verbose: bool = False):
    """把"已保存结果里有、本次内存里没有"的配置补回来。

    **为什么需要它**：断点写入是覆盖式的，而中途的滚动结果天然不完整——
    如果一次只请求部分配置（例如只补跑 2 个组合），写盘时就会把先前算好的
    其它配置从 CSV 里抹掉。本项目因此丢过三次数据，且**失败时不报错**。
    改成每次写盘前都先与已有文件取并集，文件就**在任何时刻都不会缩水**，
    正确性由结构保证，不依赖调用方记得请求哪些配置。
    """
    if not (ck_path.exists() and fold_path.exists()):
        return df, fold_df
    prev_all = pd.read_csv(ck_path)
    prev_fold_all = pd.read_csv(fold_path, index_col=0)
    have = set(df["参数"]) if "参数" in df.columns else set()
    add = [r for _, r in prev_all.iterrows() if r["参数"] not in have]
    if add:
        df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
        if verbose:
            print(f"[续跑] 补回 {len(add)} 个本次未请求的配置：" +
                  "、".join(str(r["参数"]) for r in add))
    for c in prev_fold_all.columns:
        if c not in fold_df.columns:
            fold_df[c] = prev_fold_all[c].to_numpy()
    return df, fold_df


def print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 100 + "\n")


# --------------------------------------------------------------------------
# 6. 主流程
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="实验一：新闻文本 10 分类")
    parser.add_argument("--smoke", action="store_true",
                        help="小样本自检，只验证流程是否走得通，不写任何文件")
    parser.add_argument("--smoke-size", type=int, default=600, help="自检样本数（默认 600）")
    parser.add_argument("--folds", type=int, default=N_SPLITS, help=f"折数（默认 {N_SPLITS}）")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help=f"重复次数（默认 {N_REPEATS}；设 1 退化为普通分层 CV）")
    parser.add_argument("--holdout", action="store_true",
                        help="额外附一次 80/20 单次划分做对照（默认关，因为它的估计不稳定）")
    parser.add_argument("--models", nargs="+", default=None,
                        help="只运行指定模型，例如 --models LinearSVC LogisticRegression")
    parser.add_argument("--jobs", type=int, default=1,
                        help="并行度（默认 1；设 -1 用满 CPU，但会与 BLAS 线程竞争）")
    parser.add_argument("--submit", action="store_true",
                        help="【唯一会读取测试集的开关】在全体训练数据上重训最优模型并写 predictions.csv")
    parser.add_argument("--final-config", choices=["best", "baseline"], default="best",
                        help="--submit 用哪套配置：best（默认）= 正文选定的 routing + "
                             "max_features=50000 + sublinear_tf；baseline = 原文 + 5000 维")
    parser.add_argument("--compare-splits", action="store_true",
                        help="跑划分方案对照实验：单次 80/20 与 5 折 CV 的估计稳定性（约 2 分钟）")
    parser.add_argument("--ablation", action="store_true",
                        help="步骤 2：跑文本清洗消融（累积阶梯 + 邮件头分级 + 单项独立）")
    parser.add_argument("--ablation-report", action="store_true",
                        help="读取已保存的消融结果 CSV 重新渲染三张表（不重新训练）")
    parser.add_argument("--paired-test", action="store_true",
                        help="步骤 2 补充：对消融配置做配对显著性检验（复用已保存的每折分数）")
    parser.add_argument("--model-paired-test", action="store_true",
                        help="四模型之间做两两配对显著性检验（约 10 分钟，含 MLP）")
    parser.add_argument("--clean-preset", choices=list(CLEAN_PRESETS) + ["both"],
                        default="routing",
                        help="清洗前提：raw = L0 原文，routing = L1 删路由头留 Subject（默认），"
                             "both = 两者都跑（步骤 3 的双前提对照用）")
    parser.add_argument("--tfidf-preset", choices=["baseline", "best"], default="baseline",
                        help="TF-IDF 配置：baseline = max_features=5000（默认），"
                             "best = 步骤 3 选出的最优配置（max_features=50000 + sublinear_tf）")
    parser.add_argument("--tfidf-sweep", action="store_true",
                        help="步骤 3：TF-IDF 参数单参数敏感性扫描（配合 --clean-preset both 跑双前提）")
    parser.add_argument("--tfidf-report", action="store_true",
                        help="读取已保存的 TF-IDF 逐折分数，渲染敏感性表并做配对检验（秒级）")
    parser.add_argument("--model-sweep", action="store_true",
                        help="步骤 4：扫描 LinearSVC 的 C 与 LR 的 C×penalty（TF-IDF 固定为最优配置）")
    parser.add_argument("--nested-cv", action="store_true",
                        help="步骤 4：嵌套交叉验证，给出调参流程的无偏性能估计")
    parser.add_argument("--augment", action="store_true",
                        help="数据增强对照：只看干净验证集上的准确率")
    parser.add_argument("--augment-robust", action="store_true",
                        help="数据增强的鲁棒性评估：训练用不用增强 × 在干净/扰动验证集上评估，看降幅")
    parser.add_argument("--learning-curve", action="store_true",
                        help="学习曲线：在 600/1500/3000/全量 四个规模上记录泛化间隙")
    parser.add_argument("--tfidf-combo", action="append", default=None, metavar='"k=v,k=v"',
                        help='额外测试的参数组合，逗号分隔、可重复，例如 '
                             '--tfidf-combo "max_features=20000,sublinear_tf=True"')
    parser.add_argument("--no-recheck", action="store_true",
                        help="消融后跳过四模型复核环节")
    parser.add_argument("--show-clean", type=int, default=None, metavar="N",
                        help="打印前 N 条样本在各清洗配置下的效果预览，然后退出（不训练）")
    args = parser.parse_args()

    t0 = time.time()

    # ---- 6.1 加载（只碰训练集）----
    X_all, y_all = load_train()
    print(f"[数据] 训练集 {len(X_all)} 条 | 类别数 {len(np.unique(y_all))}")
    print(f"[指纹] train_data.csv  {file_fingerprint(TRAIN_CSV)}")
    print(f"[说明] 测试集已被隔离到 {SEALED_DIR.relative_to(PROJECT_DIR)}/，本流程不会读取它")

    if args.smoke:
        rng = np.random.RandomState(SEED)
        idx = np.sort(rng.choice(len(X_all), size=min(args.smoke_size, len(X_all)), replace=False))
        X_all = [X_all[i] for i in idx]
        y_all = y_all[idx]
        print(f"[自检] 已下采样至 {len(X_all)} 条，跳过所有文件写入")

    # ---- 对照实验模式：只回答"哪种划分更稳"，不训练最终模型 ----
    if args.compare_splits:
        print("\n[对照] 单次 holdout 80/20 vs 5 折 CV 的估计稳定性（模型固定为 LinearSVC）")
        summary, detail = compare_splits(X_all, y_all)
        print_table("划分方案稳定性对照", summary)
        print("各次重复的估计值：")
        print(detail.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if not args.smoke:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            summary.to_csv(RESULT_DIR / "split_comparison.csv", index=False, encoding="utf-8-sig")
            detail.to_csv(RESULT_DIR / "split_comparison_detail.csv", index=False, encoding="utf-8-sig")
            print("\n[记录] 已保存 results/split_comparison.csv 与 split_comparison_detail.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 从已保存的结果重新渲染消融表（不训练）----
    if args.ablation_report:
        tag = f"{args.folds}x{args.repeats}"
        path = RESULT_DIR / f"ablation_clean_{tag}.csv"
        if not path.exists():
            parser.error(f"找不到 {path}，请先运行 python main.py --ablation")
        df = pd.read_csv(path)
        print(f"[重渲染] 读取 {path}（{len(df)} 个配置）")
        for gname, tbl in ablation_tables(df, ablation_groups()).items():
            print_table(gname, tbl)
        return

    # ---- 步骤 2 补充：配对显著性检验 ----
    if args.paired_test:
        tag = f"{args.folds}x{args.repeats}"
        fpath = RESULT_DIR / f"fold_scores_{tag}.csv"
        expected = {describe(cfg) for items in ablation_groups().values()
                    for _, cfg in items}
        have: set = set()
        # 自检模式下数据被下采样过，缓存的每折分数不可复用
        if fpath.exists() and not args.smoke:
            have = set(pd.read_csv(fpath, index_col=0).columns)

        if expected <= have:
            fold_df = pd.read_csv(fpath, index_col=0)
            print(f"[配对检验] 复用已保存的每折分数：{fpath.name}"
                  f"（{fold_df.shape[0]} 折 × {fold_df.shape[1]} 配置）")
        else:
            # 注意：只检查"文件是否存在"是不够的——断点续跑会留下不完整的文件。
            # 必须比对配置集合，否则会在半份数据上空跑一遍还给出错误的结论。
            if have:
                print(f"[配对检验] {fpath.name} 只含 {len(have)}/{len(expected)} 个配置，"
                      f"继续跑完剩余部分")
            else:
                print(f"[配对检验] 没有可用的 {fpath.name}，先跑一遍消融以采集每折分数")
            cv_abl = make_cv(args.folds, args.repeats)
            _, _, fold_df = run_ablation(
                X_all, y_all, cv_abl, n_jobs=args.jobs,
                cv_desc=f"{args.folds} 折 × {args.repeats} 次重复",
                tag=tag, checkpoint=not args.smoke)
            if not args.smoke:
                RESULT_DIR.mkdir(parents=True, exist_ok=True)
                fold_df.to_csv(fpath, encoding="utf-8-sig")
                print(f"[记录] 已保存 results/{fpath.name}")

        tbl = paired_tests(fold_df, n_splits=args.folds, n_repeats=args.repeats)
        print_table(f"步骤 2 · 配对显著性检验（基准 = 原文 L0，配对 {len(fold_df)} 折）", tbl)
        print("判读：只有 25 折与 5 次重复聚合两种检验都显著才判「显著」；")
        print("      两者结论不一致时判「存疑」，避免因各折不独立而过度声称。\n")
        if not args.smoke:
            tbl.to_csv(RESULT_DIR / f"paired_tests_{tag}.csv", index=False, encoding="utf-8-sig")
            print(f"[记录] 已保存 results/paired_tests_{tag}.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 四模型配对显著性检验 ----
    if args.model_paired_test:
        cfg = CLEAN_PRESETS[args.clean_preset]
        use_cfg = None if args.clean_preset == "raw" else cfg
        tf = BEST_TFIDF if getattr(args, "tfidf_preset", "baseline") == "best" else None
        tf_desc = tfidf_label(BEST_TFIDF) if tf else f"max_features={MAX_FEATURES}（基线）"
        cv_m = make_cv(args.folds, args.repeats)
        tag = f"{args.folds}x{args.repeats}"
        stem = f"{args.clean_preset}_{getattr(args, 'tfidf_preset', 'baseline')}_{tag}"
        print(f"\n[四模型对比] 清洗前提 {args.clean_preset}（{describe(cfg)}）")
        print(f"[四模型对比] TF-IDF {tf_desc}")
        print(f"[CV] {args.folds} 折 × {args.repeats} 次重复，四个模型共用同一批折\n")

        fold_df, metrics_df = run_model_fold_scores(
            X_all, y_all, cv_m, cfg=use_cfg, n_jobs=args.jobs, tfidf=tf,
            ckpt_path=(None if args.smoke else
                       RESULT_DIR / f"model_fold_scores_{stem}.csv"),
            metrics_path=(None if args.smoke else
                          RESULT_DIR / f"model_metrics_{stem}.csv"))

        if metrics_df is not None and "MacroF1均值" in metrics_df.columns:
            order = metrics_df.sort_values("MacroF1均值", ascending=False).drop(
                columns=["MacroF1均值"])
            print_table("最终结果 · 四模型完整指标（按 Macro-F1 降序）", order)

        tbl = pairwise_model_tests(fold_df, args.folds, args.repeats)
        print_table(f"四模型两两配对检验 · 前提 {args.clean_preset} · "
                    f"TF-IDF {getattr(args, 'tfidf_preset', 'baseline')}", tbl)
        print("判读：25 折检验偏乐观、重复聚合偏保守，两者都显著才判「显著」。\n")
        if not args.smoke:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            tbl.to_csv(RESULT_DIR / f"model_paired_tests_{stem}.csv",
                       index=False, encoding="utf-8-sig")
            print(f"[记录] 已保存 results/model_fold_scores_{stem}.csv、"
                  f"model_metrics_{stem}.csv、model_paired_tests_{stem}.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 步骤 3：读取已保存的逐折分数，渲染表格 + 配对检验 ----
    if args.tfidf_report:
        tag = f"{args.folds}x{args.repeats}"
        presets = ["raw", "routing"] if args.clean_preset == "both" else [args.clean_preset]
        for preset in presets:
            fpath = RESULT_DIR / f"tfidf_fold_scores_{preset}_{tag}.csv"
            cpath = RESULT_DIR / f"tfidf_sweep_{preset}_{tag}.csv"
            if not (fpath.exists() and cpath.exists()):
                print(f"[跳过] 缺少 {fpath.name} 或 {cpath.name}")
                continue
            fold_df = pd.read_csv(fpath, index_col=0)
            df = pd.read_csv(cpath)
            print(f"\n[重渲染] 前提 {preset}：{df.shape[0]} 个配置 × {fold_df.shape[0]} 折")
            for pname, tbl in tfidf_sweep_tables(df, tfidf_sweep_configs()).items():
                print_table(f"TF-IDF 单参数敏感性 · 前提 {preset} · {pname}", tbl)
            tbl = paired_tests(fold_df, baseline=tfidf_label(TFIDF_BASELINE),
                               n_splits=args.folds, n_repeats=args.repeats)
            print_table(f"TF-IDF 参数配对检验 · 前提 {preset}（基准 = 全部默认值）", tbl)
            if not args.smoke:
                tbl.to_csv(RESULT_DIR / f"tfidf_paired_tests_{preset}_{tag}.csv",
                           index=False, encoding="utf-8-sig")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 步骤 3：TF-IDF 参数单参数敏感性扫描 ----
    if args.tfidf_sweep:
        presets = ["raw", "routing"] if args.clean_preset == "both" else [args.clean_preset]
        extra = [parse_combo(s) for s in (args.tfidf_combo or [])]
        if extra:
            print(f"[额外组合] {len(extra)} 个：" +
                  "；".join(tfidf_label(kw) for kw in extra))
        cv_t = make_cv(args.folds, args.repeats)
        tag = f"{args.folds}x{args.repeats}"
        print(f"\n[CV] {args.folds} 折 × {args.repeats} 次重复 = "
              f"{args.folds * max(args.repeats, 1)} 个估计值")
        for preset in presets:
            df, groups, fold_df = run_tfidf_sweep(
                X_all, y_all, cv_t, preset, n_jobs=args.jobs, tag=tag,
                checkpoint=not args.smoke, extra=extra)
            for pname, tbl in tfidf_sweep_tables(df, groups).items():
                print_table(f"TF-IDF 单参数敏感性 · 前提 {preset} · {pname}", tbl)
            tbl = paired_tests(fold_df, baseline=tfidf_label(TFIDF_BASELINE),
                               n_splits=args.folds, n_repeats=args.repeats)
            print_table(f"TF-IDF 参数配对检验 · 前提 {preset}（基准 = 全部默认值）", tbl)
            if not args.smoke:
                RESULT_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(RESULT_DIR / f"tfidf_sweep_{preset}_{tag}.csv",
                          index=False, encoding="utf-8-sig")
                tbl.to_csv(RESULT_DIR / f"tfidf_paired_tests_{preset}_{tag}.csv",
                           index=False, encoding="utf-8-sig")
                print(f"[记录] 已保存 results/tfidf_sweep_{preset}_{tag}.csv 与 "
                      f"tfidf_paired_tests_{preset}_{tag}.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 步骤 4：模型超参扫描 ----
    if args.model_sweep:
        cv_s = make_cv(args.folds, args.repeats)
        tag = f"{args.folds}x{args.repeats}"
        print(f"\n[CV] {args.folds} 折 × {args.repeats} 次重复 = "
              f"{args.folds * max(args.repeats, 1)} 个估计值")
        df, fold_df = run_model_sweep(X_all, y_all, cv_s, n_jobs=args.jobs,
                                      tag=tag, checkpoint=not args.smoke)
        for mname in df["模型"].unique():
            print_table(f"步骤 4 · {mname} 超参对照（前提 routing，"
                        f"TF-IDF = {tfidf_label(BEST_TFIDF)}）",
                        model_sweep_table(df, mname))
            cols = [c for c in fold_df.columns if c.startswith(mname)]
            base_label = model_label(mname, MODEL_SWEEP_BASE[mname])
            tbl = paired_tests(fold_df[cols], baseline=base_label,
                               n_splits=args.folds, n_repeats=args.repeats)
            print_table(f"步骤 4 · {mname} 配对检验（基准 = 默认配置）", tbl)
            if not args.smoke:
                RESULT_DIR.mkdir(parents=True, exist_ok=True)
                tbl.to_csv(RESULT_DIR / f"model_paired_{mname}_{tag}.csv",
                           index=False, encoding="utf-8-sig")
        if not args.smoke:
            print(f"[记录] 已保存 results/model_sweep_*_{tag}.csv 与 model_paired_*_{tag}.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 步骤 4：嵌套交叉验证（无偏估计）----
    if args.nested_cv:
        tag = f"{args.folds}x{args.repeats}"
        print(f"\n[嵌套 CV] 外层 {args.folds} 折 / 内层 3 折 / "
              f"前提 {BEST_PRESET} / TF-IDF = {tfidf_label(BEST_TFIDF)}")
        rows = []
        for mname in MODEL_SWEEP_SPACE:
            grid = sweep_grid(mname)
            print(f"\n[嵌套 CV] {mname}  网格 {grid}")
            outer_s, inner_s, chosen = nested_cv(
                X_all, y_all, mname, grid, outer_splits=args.folds, n_jobs=args.jobs)
            rows.append({
                "模型": mname,
                "嵌套 CV（无偏估计）": ms(outer_s.mean(), outer_s.std(ddof=1)),
                "内层最优（参考）": ms(inner_s.mean(), inner_s.std(ddof=1)),
                "各折选出的最优超参": " | ".join(str(c) for c in chosen),
            })
        tbl = pd.DataFrame(rows)
        print_table("步骤 4 · 嵌套交叉验证", tbl)
        print("判读：**嵌套 CV 的外层分数是调参流程的无偏估计**——每折的超参只由该折训练数据")
        print("      内部选出来，外层测试折从未参与选优，所以不存在「选优虚高」。")
        print("      这一列才是报告里应该引用的性能数字。\n")
        print("      注意「内层最优」列**不能**当作乐观偏差：内层只用外层训练折的 1/2~2/3，")
        print("      训练数据更少，本身就带悲观偏差，两个方向的偏差混在一起没法分离。\n")

        # 与扫描最优对照。两者折结构不同、不配对，只能看量级。
        sweep_path = RESULT_DIR / f"model_sweep_{BEST_PRESET}_{tag}.csv"
        if sweep_path.exists():
            sw = pd.read_csv(sweep_path)
            print("[对照] 嵌套 CV 无偏估计 vs 扫描取最优（折结构不同，非配对，仅看量级）：")
            for r in rows:
                sub = sw[sw["模型"] == r["模型"]]
                if not len(sub):
                    continue
                best = float(sub["MacroF1均值"].max())
                nested = float(r["嵌套 CV（无偏估计）"].split(" ±")[0])
                note = ("嵌套 CV 反而更高，说明本例的选优虚高可忽略"
                        if nested >= best else f"扫描最优高估了约 {best - nested:.4f}")
                print(f"  {r['模型']:<22} 扫描最优 {best:.4f}   嵌套 CV {nested:.4f}   → {note}")
            print()
        if not args.smoke:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            tbl.to_csv(RESULT_DIR / f"nested_cv_{tag}.csv", index=False, encoding="utf-8-sig")
            print(f"[记录] 已保存 results/nested_cv_{tag}.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 清洗效果预览：不训练，只肉眼看规则对不对 ----
    if args.show_clean is not None:
        show_clean(X_all, args.show_clean)
        return

    # ---- 步骤 2：文本清洗消融 ----
    # ---- 数据增强对照（只看干净验证集）----
    if args.augment:
        cv_a = make_cv(args.folds, args.repeats)
        tag = f"{args.folds}x{args.repeats}"
        print(f"\n[数据增强] 前提 {BEST_PRESET}，TF-IDF {tfidf_label(BEST_TFIDF)}，LinearSVC C=1")
        print("[数据增强] 增广只在每折训练部分内做，验证部分保持原始文本\n")
        df, fold_df = run_augmentation(X_all, y_all, cv_a)
        print_table("数据增强对照", df)
        base_desc = df.loc[df["增强方式"].str.startswith("不做"), "增强方式"].iloc[0]
        test = paired_tests(fold_df, baseline=base_desc,
                            n_splits=args.folds, n_repeats=args.repeats)
        print_table("数据增强 · 配对检验（基准 = 不做增强）", test)
        if not args.smoke:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(RESULT_DIR / f"augmentation_{tag}.csv", index=False, encoding="utf-8-sig")
            fold_df.to_csv(RESULT_DIR / f"augmentation_fold_scores_{tag}.csv",
                           encoding="utf-8-sig")
            test.to_csv(RESULT_DIR / f"augmentation_paired_{tag}.csv",
                        index=False, encoding="utf-8-sig")
            print(f"[记录] 已保存 results/augmentation_*.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 数据增强 · 鲁棒性评估 ----
    if args.augment_robust:
        cv_r = make_cv(args.folds, args.repeats)
        tag = f"{args.folds}x{args.repeats}"
        print(f"\n[鲁棒性评估] 前提 {BEST_PRESET}，TF-IDF {tfidf_label(BEST_TFIDF)}，LinearSVC C=1")
        print("[鲁棒性评估] 训练增强只在训练折内做；评估在**干净**与**被扰动**的验证集上各做一次")
        print("[鲁棒性评估] 评估扰动 = 随机删除 10% 词（种子与训练扰动不同，不重合）")
        print("[鲁棒性评估] 关键指标是**降幅**（干净分 − 扰动分），不是分数本身\n")
        df, fold_df = run_augmentation_robust(X_all, y_all, cv_r)
        cols = ["训练增强", "干净验证集", "扰动验证集", "降幅（干净−扰动）", "耗时(s)"]
        print_table("数据增强 · 鲁棒性评估", df[cols])

        # 配对检验一：扰动验证集上的分数（增强是否让模型抗噪更强）
        pert_cols = [c for c in fold_df.columns if c.endswith("｜扰动")]
        t_pert = paired_tests(fold_df[pert_cols].rename(columns=lambda c: c.split("｜")[0]),
                              baseline="不做增强（基线）",
                              n_splits=args.folds, n_repeats=args.repeats)
        print_table("配对检验 A · 扰动验证集上的分数（越高越鲁棒）", t_pert)

        # 配对检验二：降幅本身（越小越鲁棒）
        clean_cols = [c for c in fold_df.columns if c.endswith("｜干净")]
        drops = pd.DataFrame({
            c.split("｜")[0]: fold_df[c].to_numpy() - fold_df[c.split("｜")[0] + "｜扰动"].to_numpy()
            for c in clean_cols})
        t_drop = paired_tests(drops, baseline="不做增强（基线）",
                              n_splits=args.folds, n_repeats=args.repeats)
        print_table("配对检验 B · 降幅（负值 = 比基线掉分更少，即更鲁棒）", t_drop)

        # 标签保持性抽检
        docs_all = apply_clean(X_all, CLEAN_PRESETS[BEST_PRESET])
        print_table("标签保持性抽检 · 删除 20% 词后，各类别关键词还剩几个",
                    label_integrity(docs_all, y_all, rate=0.20))

        if not args.smoke:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(RESULT_DIR / f"augment_robust_{tag}.csv", index=False, encoding="utf-8-sig")
            fold_df.to_csv(RESULT_DIR / f"augment_robust_fold_scores_{tag}.csv",
                           encoding="utf-8-sig")
            t_pert.to_csv(RESULT_DIR / f"augment_robust_paired_pert_{tag}.csv",
                          index=False, encoding="utf-8-sig")
            t_drop.to_csv(RESULT_DIR / f"augment_robust_paired_drop_{tag}.csv",
                          index=False, encoding="utf-8-sig")
            label_integrity(docs_all, y_all, rate=0.20).to_csv(
                RESULT_DIR / f"augment_label_integrity_{tag}.csv",
                index=False, encoding="utf-8-sig")
            print(f"[记录] 已保存 results/augment_robust_*.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 学习曲线：过拟合随数据量的变化 ----
    if args.learning_curve:
        print(f"\n[学习曲线] 前提 {BEST_PRESET}，TF-IDF {tfidf_label(BEST_TFIDF)}，"
              f"{args.folds} 折单次 CV")
        print("[学习曲线] 每个规模下的样本按类别比例抽取，随机种子固定")
        df = run_learning_curve(X_all, y_all, folds=args.folds, n_jobs=args.jobs)
        print_table("学习曲线 · 训练/验证 Macro-F1 与泛化间隙", df)
        if not args.smoke:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            df.drop(columns=["gap"]).to_csv(RESULT_DIR / "learning_curve.csv",
                                            index=False, encoding="utf-8-sig")
            print(f"[记录] 已保存 results/learning_curve.csv")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    if args.ablation:
        cv_abl = make_cv(args.folds, args.repeats)
        n_est = args.folds * max(args.repeats, 1)
        print(f"\n[CV] {args.folds} 折 × {args.repeats} 次重复 = {n_est} 个估计值")
        check_cv_folds(y_all, cv_abl)

        abl_df, groups, fold_df = run_ablation(
            X_all, y_all, cv_abl, n_jobs=args.jobs,
            cv_desc=f"{args.folds} 折 × {args.repeats} 次重复",
            tag=f"{args.folds}x{args.repeats}", checkpoint=not args.smoke)
        for gname, tbl in ablation_tables(abl_df, groups).items():
            print_table(gname, tbl)

        uniq = {}
        for items in groups.values():
            for _, cfg in items:
                uniq.setdefault(cfg.key(), cfg)
        best_cfg = uniq[tuple(abl_df.loc[abl_df["MacroF1均值"].idxmax(), "key"])]
        print(f"[最优] {describe(best_cfg)}  "
              f"MacroF1={abl_df['MacroF1均值'].max():.4f}")

        if not args.smoke:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            tag = f"{args.folds}x{args.repeats}"
            abl_df.drop(columns=["key"]).to_csv(
                RESULT_DIR / f"ablation_clean_{tag}.csv", index=False, encoding="utf-8-sig")
            fold_df.to_csv(RESULT_DIR / f"fold_scores_{tag}.csv", encoding="utf-8-sig")
            print(f"[记录] 已保存 results/ablation_clean_{tag}.csv 与 fold_scores_{tag}.csv")

        # ---- 四模型复核：结论是否只对 LinearSVC 成立？ ----
        if not args.no_recheck:
            print(f"\n[复核] 用全部 4 个模型跑最优配置：{describe(best_cfg)}")
            if "MLP" in make_models() and n_est > 10:
                print(f"[提示] 复核含 MLP，{n_est} 折约需 20 分钟")
            docs = apply_clean(X_all, best_cfg)
            re_df, re_raw = run_cv(make_models(), docs, y_all, cv_abl, n_jobs=args.jobs)
            print_table(f"四模型复核 · {describe(best_cfg)}", re_df)
            if not args.smoke:
                re_df.to_csv(RESULT_DIR / f"ablation_recheck_{tag}.csv",
                             index=False, encoding="utf-8-sig")
                print(f"[记录] 已保存 results/ablation_recheck_{tag}.csv")

        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    # ---- 6.2 模型清单 ----
    models = make_models()
    if args.models:
        unknown = set(args.models) - set(models)
        if unknown:
            parser.error(f"未知模型 {sorted(unknown)}；可选：{list(models)}")
        models = {k: v for k, v in models.items() if k in args.models}
    print(f"[模型] {list(models)}")

    # --submit 时，CV 与最终模型都改用正文选定的配置，保证"评估的东西"和"提交的东西"是同一个。
    # 其余运行路径保持默认（原文 + 5000 维基线），以免与 README/报告里的历史结果错位。
    if args.submit and args.final_config == "best":
        cv_cfg, cv_tf = CLEAN_PRESETS[BEST_PRESET], dict(BEST_TFIDF)
        print(f"[提交] 配置 best：清洗 = {describe(cv_cfg)} | TF-IDF = {tfidf_label(cv_tf)}")
    else:
        cv_cfg, cv_tf = None, {}

    # ---- 6.3 交叉验证 ----
    cv = make_cv(args.folds, args.repeats)
    n_est = args.folds * max(args.repeats, 1)
    print(f"\n[CV] {args.folds} 折 × {args.repeats} 次重复 = {n_est} 个估计值"
          f"（每折内部独立拟合 TF-IDF，n_jobs={args.jobs}）")
    if "MLP" in models and n_est > 10:
        print(f"[提示] MLP 单折 25~50s（早停触发早晚不定），{n_est} 折实测约 20 分钟；"
              f"如只需快速结果可加 --models 跳过它")
    check_cv_folds(y_all, cv)
    cv_df, cv_raw = run_cv(models, X_all, y_all, cv, n_jobs=args.jobs,
                           cfg=cv_cfg, tfidf=cv_tf)
    print_table(f"{args.folds} 折 × {args.repeats} 次重复交叉验证 · 均值 ± 标准差", cv_df)

    # ---- 6.4 可选的单次划分对照 ----
    holdout_df = None
    if args.holdout:
        holdout_df = run_holdout(models, X_all, y_all)
        print_table("对照 · 单次 80/20 划分（估计不稳定，仅作参考）", holdout_df)

    # ---- 6.5 选择方案 ----
    best_name = cv_raw.loc[cv_raw["macro_f1"].idxmax(), "Model"]
    print(f"[选择] 依据 {n_est} 次交叉验证的 Macro-F1 均值选出：{best_name}")

    # ---- 6.6 落盘 ----
    if args.smoke:
        print("[自检] 流程走通，未写入任何文件")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cv_tag = f"{args.folds}x{args.repeats}"
    # cv_5x5.csv 是 README/报告引用的"四模型基线表"。只有跑**完整默认配置**时才写它；
    # --submit 或 --models 子集都会改变表的内容，直接写进去等于把基线表覆盖掉且无法恢复。
    canonical = (not args.submit) and cv_cfg is None and not cv_tf \
        and len(models) == len(make_models())
    if canonical:
        cv_df.to_csv(RESULT_DIR / f"cv_{cv_tag}.csv", index=False, encoding="utf-8-sig")
        cv_raw.to_csv(RESULT_DIR / f"cv_{cv_tag}_raw.csv", index=False, encoding="utf-8-sig")
        print(f"[记录] 已保存 results/cv_{cv_tag}.csv 与 cv_{cv_tag}_raw.csv")
    else:
        why = "submit 诊断" if args.submit else "非基线配置"
        cv_df.to_csv(RESULT_DIR / f"cv_{cv_tag}_submitdiag.csv",
                     index=False, encoding="utf-8-sig")
        print(f"[记录] {why}，改写 results/cv_{cv_tag}_submitdiag.csv"
              f"（不覆盖基线表 cv_{cv_tag}.csv）")
    if holdout_df is not None:
        holdout_df.to_csv(RESULT_DIR / "holdout_val.csv", index=False, encoding="utf-8-sig")
        print("[记录] 已保存 results/holdout_val.csv")

    # ---- 6.7 提交（唯一会触碰测试集的分支）----
    if not args.submit:
        print("\n[提交] 未指定 --submit，跳过测试集预测。")
        print("[提交] 确认方案后运行：python main.py --submit")
        print(f"[耗时] {time.time() - t0:.1f}s")
        return

    print("\n" + "!" * 100)
    print("即将打开气隙：读取测试集并写 predictions.csv")
    print("!" * 100)
    X_test, n_test, test_path = load_test()
    print(f"[测试集] 位置 {test_path}")
    print(f"[测试集] {n_test} 条 | 完整性校验通过 {file_fingerprint(test_path)}")

    # 最终模型必须用**正文选定的那套配置**重建。
    # 早先这里写的是 make_pipeline()（无参数），等于固定用"原文 + 5000 维"的 baseline，
    # 与报告里选出的 routing + 50000 维 + sublinear_tf 不是一回事——
    # 提交的东西和论证过的东西必须是同一个，否则整份调参白做。
    print(f"[提交] 最终配置 {args.final_config}："
          f"清洗 = {describe(cv_cfg) if cv_cfg else '不清理'} | "
          f"TF-IDF = {tfidf_label(cv_tf) if cv_tf else 'max_features=5000（基线）'}")

    # 词表在全部训练数据上重新拟合（依然只接触训练数据）
    final = make_pipeline(cv_cfg, **cv_tf)
    final.set_params(clf=make_models()[best_name])
    final.fit(X_all, y_all)
    preds = final.predict(X_test)
    print(f"[提交] 最终模型 {best_name} 的词表规模：{len(final.named_steps['tfidf'].vocabulary_)}")

    # ---- 6.8 提交前校验 ----
    assert len(preds) == n_test, f"预测数量 {len(preds)} 与测试集 {n_test} 不一致"
    assert set(np.unique(preds)) <= set(range(N_CLASSES)), "预测值超出 0~9 范围"
    pd.Series(preds).to_csv(PREDICTION_CSV, index=False, header=False)
    print(f"[提交] 已写入 {PREDICTION_CSV}（{len(preds)} 行，无表头）")
    dist = {int(k): int(v) for k, v in zip(*np.unique(preds, return_counts=True))}
    print(f"[提交] 预测类别分布：{dist}")

    # ---- TODO（后续步骤）----
    # [步骤 2] 文本清洗消融：邮件头 / 引用 / 签名 / 标点 / 停用词 / stemming 逐项对照
    # [步骤 3] TF-IDF 调参：ngram_range, min_df, max_df, max_features, sublinear_tf
    # [步骤 4] 模型调参：SVC.C/gamma、LR.C/penalty、MLP.alpha/hidden_layer_sizes
    #          必须用嵌套 CV（外层 5 折评估 + 内层 GridSearchCV 调参），
    #          否则调参过程会"喂高"评估分数，报出来的性能偏乐观
    # [步骤 5] 可视化：模型对比条形图、C 值曲线、混淆矩阵、MLP loss 曲线 → 存入 figs/

    print(f"[耗时] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
