# results/ —— 实验结果索引

本目录是**全部结论的数据来源**。共 60 个文件：**41 份结果 CSV + 19 份运行日志**。

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
| `fold_scores_5x5.csv` | 各配置的逐折分数 | `--ablation` |
| `ablation_recheck_5x5.csv` | 最优配置上的**四模型复核** | `--ablation` |
| `run_ablation.log` | 运行日志（含三张分组表 + 四模型复核表） | 同上 |
| `run_paired_test.log` | 配对检验输出 | `--paired-test` |

**对应报告**：§3.2 ｜ **截图 ⑤⑥⑯**

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
| `run_final_models.log` / `run_final_models_full.log` | 运行日志 | 同上 |
| `run_model_paired_{raw,routing}.log` | 5000 维两前提的运行日志 | 同上 |

**对应报告**：§3.1 主结果 ｜ **截图 ⑩**

> `routing_best` = 最优配置（routing + `max_features=50000` + `sublinear_tf=True` + LinearSVC C=1）。

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

**对应报告**：§2.3 / §3.2 / §四 / §5.2 ｜ **截图 ③④⑪⑫**

---

## 数据可信度

- **可复现**：所有脚本固定随机种子 `SEED=42`；环境版本锁定在 `../requirements.txt`
- **可追溯**：每份 CSV 都能对应到一条命令；每份日志都记录了数据指纹（`train_data.csv` 的 SHA256）
- **配对性**：`*_fold_scores*.csv` 保留逐折原始分数，因此任意两个配置都能**事后重算配对检验**，不依赖当时打印的结论

> 本目录中的日志**未被修改**。整理过程中只在"过滤显示范围"时用 `sed`/`grep`，不改内容。
