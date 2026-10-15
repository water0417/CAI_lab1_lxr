# data-Project1/ —— 代码与数据

## 目录结构

```
data-Project1/
├── main.py                主实验（唯一入口）
├── plots.py               结果图表生成
├── train_data.csv         训练集（7368 条）
├── predictions.csv        提交文件（2457 行）
├── experiments/           补充实验：每个回答一个具体问题
├── evidence/              证据生成：过程性演示 + 终端截图
└── reference/             助教原件（原样保留，未改动）
```

**只看 `main.py` + `plots.py` 就能读懂全实验**；其余目录是为回答具体问题而补做的。

### 根目录 · 主实验

| 文件 | 行数 | 作用 |
|---|---|---|
| **`main.py`** | 2208 | **实验全流程**：划分、清洗调参、评估、四模型对比、嵌套 CV、学习曲线、`--submit` |
| `plots.py` | 834 | 结果性图表生成（12 张 → `../figs/`） |

### `experiments/` · 补充实验

| 文件 | 行数 | 回答什么问题 |
|---|---|---|
| `experiments/augment_semantic.py` | 303 | 语义级增强（LSA 词向量近邻替换）到底有没有用 |
| `experiments/probe_q1q2.py` | 115 | 思考题①「高维下训练分算不算证据」+ ②「加大 epochs 有没有用」 |
| `experiments/header_parts_check.py` | 96 | 邮件头里哪一部分有判别力 |
| `experiments/nb_alpha_check.py` | 67 | NaiveBayes 的 `alpha` 扫到多少最优 |
| `experiments/mlp_tune.py` | 134 | MLP 的结构 × 正则 12 组网格（两阶段，有断点续跑） |
| `experiments/mlp_run_all.py` | 50 | ↑ 的守护进程：阶段 1 跑完自动挑最优并接阶段 2 |
| `experiments/cost_probe_final.py` | 75 | 表 3-1 的成本行（四模型同折受控计时） |
| `experiments/tuned_metrics.py` | 64 | 补齐 LR / NB 调参后的完整指标 |

### `evidence/` · 证据生成

| 文件 | 行数 | 作用 |
|---|---|---|
| `evidence/evidence.py` | 360 | 过程性证据演示（5 项，秒级~6 分钟；落盘 `../results/evidence_*.log`） |
| `evidence/make_term_shots.py` | 217 | 把 `../results/` 的日志渲染成终端风格 PNG（20 张 → `../figs/term/`） |

### `reference/` · 助教原件（原样保留，未改动）

| 文件 | 作用 |
|---|---|
| `draft_main.py` | 助教给的脚手架 |
| `TUNING.md` | 助教的调参指南（本实验超参搜索区间的依据） |

### 数据

| 文件 | 说明 |
|---|---|
| `train_data.csv` | 7368 条，列 `text, target` |
| `predictions.csv` | 提交文件（由 `--submit` 生成，2457 行、单列、无表头） |

> **路径锚点**：`experiments/` 与 `evidence/` 下的脚本在文件头插入了 3 行 `sys.path` 垫片，
> 因此**无论在哪个目录执行**，`M.RESULT_DIR` / `M.FIG_DIR` 都指向 `Lab1/results/` 与
> `Lab1/figs/`——这些常量在 `main.py` 里以**它自己的位置**为锚，不随脚本搬家而错位。

---

## 运行前提

```bash
conda activate AILab        # Python 3.10.22，依赖版本见 ../requirements.txt
cd data-Project1
python main.py --smoke      # 先跑自检：600 条样本走通全流程，不写任何文件
```

**脚本内部按 `__file__` 解析路径**，因此从其他目录调用也不会读错数据；但为清晰起见建议在本目录执行。

---

## `main.py` 的命令行入口（按实验阶段）

### 基础

| 命令 | 作用 | 耗时 |
|---|---|---|
| `--smoke` | 600 条样本自检，**不写任何文件** | 1 分钟 |
| （默认） | 基线：四模型 × 5 折 ×5 重复 | 31 分钟 |
| `--models a b` | 只跑指定模型 | 视模型 |
| `--repeats 1` | 退化为 5 折单次 | 4~7 分钟 |
| `--holdout` | 附带一次 80/20 单次划分对照 | +2 分钟 |

### 步骤 1 · 数据划分

| 命令 | 作用 |
|---|---|
| `--compare-splits` | **划分方案稳定性对照**：单次 80/20 vs 5 折 CV（截图 ①） |

### 步骤 2 · 清洗与增强

| 命令 | 作用 |
|---|---|
| `--show-clean N` | 打印前 N 条样本在各清洗配置下的效果，**不训练** |
| `--ablation` | **清洗消融** 13 配置 + 四模型复核（85 分钟） |
| `--ablation --no-recheck` | 只跑消融阶梯，跳过四模型复核 |
| `--ablation-report` | 从 CSV 重渲染三张表（**秒级**，不训练） |
| `--paired-test` | 清洗配置的**配对显著性检验**（复用缓存的逐折分数） |
| `--augment` | 数据增强对照（只看干净验证集） |
| `--augment-robust` | **鲁棒性评估**：训练用不用增强 × 干净/扰动验证集 |

### 步骤 3 · TF-IDF 调参

| 命令 | 作用 |
|---|---|
| `--tfidf-sweep` | **单参数敏感性扫描**（配合 `--clean-preset both` 跑双前提） |
| `--tfidf-combo "k=v,k=v"` | 追加**参数组合**测试，可重复 |
| `--tfidf-report` | 从 CSV 重渲染 + 配对检验（**秒级**） |

### 步骤 4 · 模型调参

| 命令 | 作用 |
|---|---|
| `--model-sweep` | LinearSVC 的 `C`、LR 的 `C×penalty` 扫描 |
| `--nested-cv` | **嵌套交叉验证**：调参流程的无偏估计 |

### 评估

| 命令 | 作用 | 耗时 |
|---|---|---|
| `--model-paired-test` | **四模型对比 + 两两配对检验**（主结果） | 2.5~5.5 小时（MLP 占绝大部分） |
| `--tfidf-preset best` | 配合上一项：用最优 TF-IDF 配置而非基线 | — |
| `--learning-curve` | **学习曲线**：4 规模 × 4 模型的泛化间隙 | 28 分钟 |
| `--submit` | ⚠ **唯一会读取测试集的命令**，生成 `predictions.csv`。默认用正文选定的最优配置（见下） | 1 分钟（配 `--models LinearSVC`） |
| `--final-config best\|baseline` | 配合 `--submit`：用哪套配置。**默认 `best`** = routing + 50000 维 + `sublinear_tf` | — |

> **实际生成提交文件用的命令**（也是唯一跑过 `--submit` 的一次）：
> ```bash
> python main.py --submit --models LinearSVC
> ```
> 日志在 `../results/run_submit.log`。`--models LinearSVC` 是因为选型已由前面的
> 四模型对比决定，没必要为了选型再跑一遍含 MLP 的 25 折（约 2~5 小时）。
>
> ⚠ **这里曾经有个 bug**：`--submit` 分支原先写的是 `make_pipeline()`（无参数），
> 等于固定用"原文 + 5000 维"的 baseline，**与正文选出的配置无关**。
> 后果是提交文件与报告里论证的方案不是一回事——而且因为 `--tfidf-preset` 默认
> `baseline`，从命令行上完全看不出来。现已改为读 `--final-config`（默认 `best`），
> 并把实际配置打印在日志里。同一次修复还加了一道护栏：非默认配置不再覆盖
> `../results/cv_5x5.csv`（那是 README 引用的四模型基线表）。

### 通用开关

| 开关 | 说明 |
|---|---|
| `--folds N` / `--repeats N` | 覆盖 CV 折数与重复次数（默认 5 / 5） |
| `--clean-preset {raw,routing,both}` | 清洗前提（默认 `routing`） |
| `--tfidf-preset {baseline,best}` | TF-IDF 配置（默认 `baseline`） |
| `--jobs N` | 并行度（默认 1；`-1` 用满 CPU，但会与 BLAS 竞争） |

---

## 其他脚本

### `plots.py` —— 结果性图表

```bash
python plots.py                 # 全部 12 张
python plots.py --only fig4     # 只生成某一张
```

从 `../results/` 的 CSV 读取，输出到 `../figs/`。图表规范见 `../figs/README.md`。

### `evidence/evidence.py` —— 过程性证据演示

```bash
python evidence/evidence.py            # 列出全部 5 项
python evidence/evidence.py 1          # 测试集三道锁（0.1 秒）
python evidence/evidence.py 2          # 恒等变换验证（0.8 秒）
python evidence/evidence.py 3          # 错误分析（23 秒）
python evidence/evidence.py 4          # MLP 稠密化探测（约 3 分钟）
python evidence/evidence.py 5          # 四模型单折受控计时（约 6 分钟，写 model_cost_single_fold.csv）
```

每项输出控制在一屏内，供截图使用。**可以现跑现截**——不像主实验那样耗时。
落盘方式与其他实验一致，例如 `python evidence/evidence.py 5 > ../results/evidence_5.log`；
报告里的截图 ③④⑪⑫⑳ 都取自这些日志文件。

### `experiments/augment_semantic.py` —— 语义级增强

```bash
python experiments/augment_semantic.py              # 约 15 分钟
python experiments/augment_semantic.py --folds 3 --repeats 1   # 快速试跑
```

**刻意做成独立模块**：不修改 `main.py` 一行，输出 `semantic_aug_*.csv`，与既有结果零重名。三条设计约束（无新依赖 / 折内构建 / 只在同类内替换）写在模块头部的 docstring 里。

### `evidence/make_term_shots.py` —— 终端证据截图

```bash
python evidence/make_term_shots.py          # 全部 20 张
python evidence/make_term_shots.py 3 6      # 只生成指定编号
```

把 `../results/` 的运行日志渲染成终端风格 PNG（深色底 + 标题栏 + 命令提示符），输出到 `../figs/term/`。

**内容不增删一个字符**——脚本只做"过滤显示范围 + 配色渲染"，每张图正文第一行就是产生它的命令。

---

## 测试集隔离（重要）

`../sealed/test_data_unlabeled.csv` 是助教发放的原始测试集，**不在本目录**。四道锁：

1. **目录隔离** —— 移出工作目录，放在 `../sealed/`
2. **代码隔离** —— 全项目中只有 `load_test()` 一个函数读它，且只被 `--submit` 分支调用
3. **完整性校验** —— 读取前逐字节比对 SHA256，对不上直接报错退出
4. **版本库隔离** —— 已在 `../.gitignore` 中排除，不会入库或上传

即使用 `--submit`，校验失败也会拒绝运行。这一点的证据是 `../results/evidence_1.log`（截图 ③）。
