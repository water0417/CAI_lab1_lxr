# results/ —— 实验结果索引

本目录是**全部结论的数据来源**。共 86 个文件：**54 份结果 CSV + 32 份运行日志**。

## 命名规则

```
<实验名>_<前提>_<CV 方案>.csv     结果数据
run_<实验名>.log                 运行日志（原始输出，部分被用作终端证据截图）
```

| 片段 | 取值 | 含义 |
|---|---|---|
| **前提** | `raw` / `routing` | 清洗前提：保留全部邮件头 / 删路由头保留 Subject |
| **CV 方案** | `5x5` | 5 折 ×5 重复 = 25 个估计值 |
| `_raw` 后缀 | — | 同一次运行的**机器可读版**（主表是格式化版；`_raw` 是未格式化的浮点原值） |
| `_fold_scores` | — | 每折的原始分数，**配对检验的唯一输入**（均值不够用） |

> **`run_tfidf_final.log` / `run_tfidf_refill.log` / `run_tfidf_restore.log` 是补数据的运行记录**：
> 整理过程中发现逐折分数表缺了几列（续跑逻辑的静默覆盖问题），这三次运行是分步补齐的过程。
> 保留它们是为了让"数据是怎么补回来的"也可追溯，而非隐藏。

---

## 1. 数据划分

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `split_comparison.csv` | 单次 80/20 vs 5 折 CV 的估计稳定性对照 | `python main.py --compare-splits` |
| `split_comparison_detail.csv` | 15 次重复的逐次估计值 | 同上 |
| `cv_5x5.csv` | 四模型基线（5000 维 + raw，25 折） | `python main.py` |
| `cv_5x5_raw.csv` | 同上，机器可读版 | 同上 |
| `run_cv_5x5.log` | 默认运行日志（含分层检查与数据指纹） | 同上 |

**对应报告**：§2.2 数据划分 ｜ **截图 ①②**

---

## 2. 清洗消融

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `ablation_clean_5x5.csv` | **13 个清洗配置的结果**（主表） | `python main.py --ablation` |
| `paired_tests_5x5.csv` | 各配置相对原文的**配对 t 检验** | `python main.py --paired-test` |
| `header_parts_5x5.csv` | **邮件头逐部分**对照：Subject 行 / 其余头部 / 正文 / 全文 | `python header_parts_check.py` |
| `header_parts_fold_scores_5x5.csv` | 上者的逐折分数 | 同上 |
| `fold_scores_5x5.csv` | 各配置的逐折分数 | `--ablation` |
| `ablation_recheck_5x5.csv` | 最优配置上的**四模型复核** | `--ablation` |
| `run_ablation.log` | 运行日志（含三张分组表 + 四模型复核表） | 同上 |
| `run_paired_test.log` | 配对检验输出 | `--paired-test` |

**对应报告**：§3.2 ｜ **截图 ⑤⑥⑯**

> `header_parts_*` 是 2026-10-15 补做的：§3.2 原有的"邮件头逐部分"表里的四个分数
> 在任何结果文件中都查不到（来自一次未落盘探索），已重跑并落盘。

---

## 3. 数据增强

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `augmentation_5x5.csv` | 三种结构扰动的结果（只看干净验证集） | `python main.py --augment` |
| `augmentation_paired_5x5.csv` | 相对基线的配对检验 | 同上 |
| `augmentation_fold_scores_5x5.csv` | 逐折分数 | 同上 |
| `augment_robust_5x5.csv` | **鲁棒性评估**：干净/扰动验证集与降幅 | `python main.py --augment-robust` |
| `augment_robust_paired_pert_5x5.csv` | 配对检验 A：扰动集上的分数 | 同上 |
| `augment_robust_paired_drop_5x5.csv` | 配对检验 B：**降幅** | 同上 |
| `augment_label_integrity_5x5.csv` | **标签保持性抽检**：删除 20% 后各类别关键词存活 | 同上 |
| `augment_robust_fold_scores_5x5.csv` | 逐折分数 | 同上 |
| `semantic_aug_5x5.csv` | **语义级增强**（LSA 词向量近邻替换） | `python augment_semantic.py` |
| `semantic_aug_fold_scores_5x5.csv` | 逐折分数（可与结构扰动做同折配对） | 同上 |
| `run_augment.log` / `run_augment_robust.log` / `run_semantic_aug.log` | 三者的运行日志 | 同上 |

**对应报告**：§3.2 ｜ **截图 ④⑬⑭⑮**

---

## 4. TF-IDF 调参

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `tfidf_sweep_{raw,routing}_5x5.csv` | **16 个配置的结果**（含参数组合） | `python main.py --tfidf-sweep --clean-preset both` |
| `tfidf_paired_tests_{raw,routing}_5x5.csv` | 各配置相对默认值的配对检验 | 同上 |
| `tfidf_fold_scores_{raw,routing}_5x5.csv` | 逐折分数 | 同上 |
| `run_tfidf_sweep.log` | 单参数敏感性扫描日志 | 同上 |
| `run_tfidf_combo3.log` | 参数组合扫描日志（含完整 16 配置） | 同上 |
| `run_tfidf_final.log`、`run_tfidf_refill.log`、`run_tfidf_restore.log` | 补齐缺失列的三次运行记录 | 见上方说明 |

**对应报告**：§3.3 ｜ **截图 ⑦⑰**

---

## 5. 模型调参

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `model_sweep_routing_5x5.csv` | LinearSVC 的 `C`、LR 的 `C×penalty`（15 个配置） | `python main.py --model-sweep` |
| `model_paired_LinearSVC_5x5.csv` | LinearSVC 各 `C` 相对默认的配对检验 | 同上 |
| `model_paired_LogisticRegression_5x5.csv` | LR 各配置的配对检验 | 同上 |
| `model_sweep_fold_scores_routing_5x5.csv` | 逐折分数 | 同上 |
| `nested_cv_5x5.csv` | **嵌套 CV 的无偏估计**与逐折选出的超参 | `python main.py --nested-cv` |
| `run_model_sweep.log` / `run_nested_cv.log` | 运行日志 | 同上 |

**对应报告**：§3.4 ｜ **截图 ⑧⑨**

---

## 6. 四模型对比（主结果）

| 文件 | 内容 | 产生命令 |
|---|---|---|
| **`model_metrics_routing_best_5x5.csv`** | **最终结果表**：Accuracy / Macro-F1 / Weighted-F1 / 训练分 / 泛化间隙 / 耗时 | `python main.py --model-paired-test --tfidf-preset best` |
| `model_fold_scores_routing_best_5x5.csv` | 逐折分数（配对检验输入） | 同上 |
| `model_paired_tests_routing_best_5x5.csv` | 四模型两两配对检验 | 同上 |
| `model_fold_scores_{raw,routing}_5x5.csv` | 5000 维下（raw / routing）的四模型逐折分数 | `--model-paired-test` |
| `model_paired_tests_{raw,routing}_5x5.csv` | 对应的两两配对检验 | 同上 |
| `run_final_models.log` / `run_final_models_full.log` | **同一配置的两次独立运行**（详见下方） | 同上 |
| `run_model_paired_{raw,routing}.log` | 5000 维两前提的运行日志 | 同上 |

**对应报告**：§3.1 主结果 ｜ **截图 ⑩⑲**

> **`routing_best` 里的 "best" 指的是 TF-IDF 配置最优**（routing + `max_features=50000` + `sublinear_tf=True`），
> **不是"模型超参最优"**。⚠️ 该表里四个模型的超参**都是 sklearn 默认值**：
> LinearSVC `C=1`、LR `C=1`、NB `alpha=1.0`、MLP `(100,)`、`alpha=1e-4`。
>
> 这些默认值**离各自最优的远近差别极大**——LinearSVC 与 MLP 的默认恰好就在最优附近，
> 而 LR 与 NB 的默认偏离很远（详见 `../实验报告.md` §3.1 的 ⚠️ 警示块）。
>
> **本文件自 2026-10-14 首次生成后未再改动，这是刻意的**：它记录的是**原配置口径**的结果，
> 与报告 §3.1 主表一一对应。各模型调到自身最优后的结果**另存于**：
> `tuning_fairness_5x5.csv`（LR → C=100）、`nb_alpha_sweep_5x5.csv`（NB → alpha=0.1）、
> `mlp_tuned_25fold_5x5.csv`（MLP → alpha=1e-3，25 折确认 **0.94445，比默认值低 0.00005**，即调参无提升）。
>
> **两次运行的意义**：`run_final_models.log` 与 `run_final_models_full.log` 是**同一个配置跑了两遍**。
> 两次的 Macro-F1 逐位相同（0.9445 / 0.9434 / 0.9280 / 0.9187），但耗时相差 1.6~2.9 倍
> （MLP 7890 s vs 19457 s，LinearSVC 228 s vs 79 s）。
> 这正是报告 §3.1 判定"**分数可复现、墙钟时间不可复现**"的直接依据——
> 因此正文的耗时论断一律引用受控的单折测量（`evidence_4.log` / `evidence_5.log`），
> 不引用这两次长任务的墙钟时间。

---

## 7. 过拟合 / 学习曲线

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `learning_curve.csv` | 4 个训练规模 × 4 模型的训练分、验证分与泛化间隙 | `python main.py --learning-curve` |
| `run_learning_curve.log` | 运行日志 | 同上 |

**对应报告**：§5.2 思考题二 ｜ **截图 ⑱**

---

## 8. 过程性证据演示

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `evidence_1.log` | 测试集三道锁：指纹、唯一调用点、篡改 1 字节被拦截 | `python evidence.py 1` |
| `evidence_2.log` | 恒等变换验证：交换词序 200/200 向量完全相同 | `python evidence.py 2` |
| `evidence_3.log` | 错误分析：逐类别 F1、混淆对、类别特征词 | `python evidence.py 3` |
| `evidence_4.log` | MLP 稠密化探测：维度、非零项数、单折计时 | `python evidence.py 4` |
| `evidence_5.log` | 四模型单折受控计时（5000 维 vs 50000 维） | `python evidence.py 5` |
| `model_cost_single_fold.csv` | 上者的结果表 | 同上 |

**对应报告**：§2.3 / §3.2 / §三 / §3.5 / §5.2 ｜ **截图 ③④⑪⑫⑳**

> **为什么耗时只看 `evidence_4/5.log`，不看 `run_final_models*.log`**：
> 后者是几小时的长任务，墙钟时间把机器休眠也算了进去——同一配置跑两遍差 1.6~2.9 倍
> （详见 §6 与 `../README.md` §4.6）。前者是 3~6 分钟的受控测量，同一折、同一进程。

---

## 9. 提交文件生成

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `run_submit.log` | 最终提交的完整日志：配置、25 折 CV 复核、测试集 SHA256 校验、预测分布 | `python main.py --submit --models LinearSVC` |
| `cv_5x5_submitdiag.csv` | 提交时那次 CV 的诊断表（**不覆盖**基线表 `cv_5x5.csv`） | 同上 |

**对应报告**：附录 复现说明 ｜ 提交物 `../data-Project1/predictions.csv`

> 提交模型的 CV 复现了 **0.9434 ± 0.0062**，与 §3.1 表格逐位一致；
> 说明"评估用的配置"和"提交用的配置"确实是同一个。

---

## 9. 调参对等性核查（2026-10-14 补）

§6 的四模型主表四行**用的都是各库默认超参数**，但"默认值离各自最优有多远"差别极大
（LinearSVC 的默认恰好命中，LR 与 NB 的默认差很远），于是补做了这一组对照
——**其余条件完全不变，只改超参**：

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `tuning_fairness_5x5.csv` | LR 在 C=1/10/100、NB 在 alpha=1.0/0.1/0.01 下的结果 | `python -` (见 `run_tuning_fairness.log`) |
| `tuning_fairness_fold_scores_5x5.csv` | 上者的逐折分数（配对检验输入） | 同上 |
| `nb_alpha_sweep_5x5.csv` | **NB 的 alpha 细化扫描**（0.03~0.3，最优 0.1） | `python nb_alpha_check.py` |
| `nb_alpha_fold_scores_5x5.csv` | 上者的逐折分数 | 同上 |
| `run_tuning_fairness.log` / `run_nb_alpha_fine.log` | 运行日志 | 同上 |
| `mlp_tuned_25fold_5x5.csv` | **MLP 最优配置的 25 折确认**（`(100,)`, alpha=1e-3） | `python mlp_tune.py confirm "(100,)" 1e-3` |
| `mlp_tuned_fold_scores_5x5.csv` | 上者的逐折分数 | 同上 |
| `run_mlp_confirm.log` | 运行日志 | 同上 |
| `model_metrics_routing_baseline_5x5.csv` 等 3 份 | **5000 维**下的四模型完整指标 / 逐折分数 / 配对检验 | `python main.py --model-paired-test` |
| `run_final_models_5000dim.log` | 上者的运行日志 | 同上 |
| `mlp_screen_5fold.csv` | **MLP 超参完整网格**：4 种结构 × 3 个 alpha = 12 组 | `python mlp_tune.py screen` |
| `mlp_screen_fold_scores.csv` | 上者的逐折分数 | 同上 |
| `run_mlp_screen.log` | 运行日志 | 同上 |

**结论（也是本实验最值得记的教训）**：

| 模型 | 主表值 | 自身最优 | 调参后 | 变化 |
|---|---|---|---|---|
| MLP | 0.9445 | 完整网格最优仍是 `(100,)`，alpha=1e-3 | 0.9445 | **−0.00005（无提升）** |
| LinearSVC | 0.9434 | C=1 | 0.9434 | — |
| LogisticRegression | 0.9280 | C=100 | **0.9405** | **+0.0125** |
| NaiveBayes | 0.9187 | alpha=0.1 | **0.9391** | **+0.0204** |

**极差 0.0258 → 0.0054**；LR 与 NB 由"显著不同"变为"不可区分"。
→ **主表量到的"模型差距"，约八成是"调参程度差距"。**

> 这两次运行用的是 `main.run_cv` 原样调用，与主表同一条代码路径、同一批 25 折，
> 因此得到的分数与主表**逐位可比**。
>
> **MLP 已跑完整网格**（12 组 = 4 种结构 × 3 个 alpha）：
> `(100,)` 0.9450~**0.9455** ＞ `(50,)` 最高 0.9437 ＞ `(100,50)` 最高 0.9434 ＞ `(200,)` 最高 0.9426。
> **增大容量（加宽或加深）反而更差**；`alpha` 同结构内极差 ≤0.002。
> 默认配置 `(100, 1e-4)` 就在最优旁边（差 0.0005）。
> 注：上表是 **5 折单次**口径，不与主表 25 折直接可比；最优配置的 25 折确认另跑。

---

## 10. 思考题①②的判定性实验（2026-10-15）

| 文件 | 内容 | 产生命令 |
|---|---|---|
| `random_label_control_5x5.csv` | **随机标签对照**：真实 vs 打乱标签 × 5000/50000 维的训练分与验证分 | `python probe_q1q2.py`（实验 A） |
| `mlp_epoch_probe_5fold.csv` | **MLP 迭代次数诊断**：三种训练预算的迭代数、训练分、验证分 | 同上（实验 B） |
| `run_random_label_control.log` / `run_probe_epochs.log` | 两者的运行日志 | 同上 |

**结论一（随机标签对照）**：50000 维（`d > n`）下**随机标签也能被分到训练 0.9939**，
验证分掉到 0.0912；真实标签同设定下验证 **0.9452**。
→ **高维下"训练分高"不构成证据**，"近似线性可分"只能靠泛化性能支撑（§5.1）。

**结论二（迭代次数）**：默认 18/300 次迭代就早停；把耐心放大 3 倍（练到 38 轮）
与把上限提到 1000，三种预算给出**逐位相同**的 0.9929 / 0.9421。
→ **加大 epochs 没有收益**，模型早已收敛（§5.2）。

---

## 数据可信度

- **可复现**：所有脚本固定随机种子 `SEED=42`；环境版本锁定在 `../requirements.txt`
- **可追溯**：每份 CSV 都能对应到一条命令；每份日志都记录了数据指纹（`train_data.csv` 的 SHA256）
- **配对性**：`*_fold_scores*.csv` 保留逐折原始分数，因此任意两个配置都能**事后重算配对检验**，不依赖当时打印的结论

> 本目录中的日志**未被修改**。整理过程中只在"过滤显示范围"时用 `sed`/`grep`，不改内容。
