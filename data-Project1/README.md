# data-Project1/ —— 代码与数据

## 文件清单

| 文件 | 行数 | 作用 | 来源 |
|---|---|---|---|
| **`main.py`** | 2208 | **实验全流程**：划分、清洗、调参、评估、四模型对比、学习曲线 | 自己写 |
| `plots.py` | 834 | 结果性图表生成（12 张，输出到 `../figs/`） | 自己写 |
| `evidence.py` | 322 | 过程性证据演示（5 项，秒级~6 分钟可重跑，供截图） | 自己写 |
| `augment_semantic.py` | 303 | 语义级数据增强（LSA 词向量近邻替换），**独立模块** | 自己写 |
| `make_term_shots.py` | 217 | 把运行日志渲染成终端风格 PNG（20 张，输出到 `../figs/term/`） | 自己写 |
| `nb_alpha_check.py` | 67 | **调参对等性核查**：细化扫描 NB 的 alpha（见 `results/README.md` §9） | 自己写 |
| `header_parts_check.py` | 96 | **邮件头逐部分对照**：Subject 行 / 其余头部 / 正文 / 全文各切一次 | 自己写 |
| `mlp_tune.py` | 134 | **MLP 超参搜索**（两阶段；⚠ 只跑了 12 组里的 6 组，见文件头说明） | 自己写 |
| `mlp_run_all.py` | 50 | 同上：等阶段 1 结束自动挑最优并跑阶段 2（守护进程） | 自己写 |
| `draft_main.py` | 56 | 助教给的脚手架 | 助教，**原样保留** |
| `TUNING.md` | — | 助教的调参指南（调参区间的依据） | 助教，**原样保留** |
| `train_data.csv` | — | 7368 条，列 `text, target` | 助教 |
| `predictions.csv` | — | 提交文件（由 `--submit` 生成，2457 行） | 自己生成 |

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

### `evidence.py` —— 过程性证据演示

```bash
python evidence.py            # 列出全部 5 项
python evidence.py 1          # 测试集三道锁（0.1 秒）
python evidence.py 2          # 恒等变换验证（0.8 秒）
python evidence.py 3          # 错误分析（23 秒）
python evidence.py 4          # MLP 稠密化探测（约 3 分钟）
python evidence.py 5          # 四模型单折受控计时（约 6 分钟，写 model_cost_single_fold.csv）
```

每项输出控制在一屏内，供截图使用。**可以现跑现截**——不像主实验那样耗时。
落盘方式与其他实验一致，例如 `python evidence.py 5 > ../results/evidence_5.log`；
报告里的截图 ③④⑪⑫⑳ 都取自这些日志文件。

### `augment_semantic.py` —— 语义级增强

```bash
python augment_semantic.py              # 约 15 分钟
python augment_semantic.py --folds 3 --repeats 1   # 快速试跑
```

**刻意做成独立模块**：不修改 `main.py` 一行，输出 `semantic_aug_*.csv`，与既有结果零重名。三条设计约束（无新依赖 / 折内构建 / 只在同类内替换）写在模块头部的 docstring 里。

### `make_term_shots.py` —— 终端证据截图

```bash
python make_term_shots.py          # 全部 18 张
python make_term_shots.py 3 6      # 只生成指定编号
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
