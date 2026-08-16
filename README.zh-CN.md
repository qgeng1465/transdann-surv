# TransDANN Failure Analysis: Cross-Population Cancer Survival Prediction

## When Gradient-Reversal Adversarial Training (GRL/DANN) Fails: Population Distributions as the Root Domain Fingerprint (Missingness Is Its Visible Surface)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**中文** · [**English**](README.md)

> **中文摘要。** 在多机构癌症生存预测中，*队列级协变量的分布*（年龄、分期、事件率）——其中缺失模式只是最显眼的表面——构成了根本性的**域指纹（domain fingerprint）**。我们提供大规模机制性证据表明：**梯度反转域对抗训练（GRL/DANN）**——以及更一般地，所有试图**将域身份从学习到的表征中解耦**的方法——都无法在 **110,640 例肝癌患者 / 5 个队列 / 3 种癌症类型**上改进跨人群生存预测：域分类器可达到 97–98% 的准确率，且无法被梯度反转所压制；受控插补（缺失率 → 0%）之后其准确率仍为 97.7%；而无泄漏的队列内插补无法填补在队列中结构性不可观测的字段（SEER 的 Grade，100% 缺失），此时域指纹被完美保留。相比之下，**输入级增强（mixup）是唯一通过 Benjamini–Hochberg FDR 校正的方法**（Δ=+0.019，q=0.012，在 5 队列划分中），而基于对齐/不变性的方法没有一致收益——其中一些（GroupDRO、V-REx、Fish）在异构多域数据上显著*有害*（Δ=−0.05 至 −0.07，q<0.05）。失败是结构性的：域身份编码在协变量分布本身之中，因此**对抗式特征解耦**无法将其消除，而**输入级增强**则绕过了它。

---

## 目录

- [背景](#背景)
- [核心假设](#核心假设)
- [实验概览](#实验概览)
- [关键发现](#关键发现)
- [机制验证](#机制验证)
- [数据](#数据)
- [方法](#方法)
- [结果：图表](#结果图表)
- [项目结构](#项目结构)
- [安装](#安装)
- [复现命令](#复现命令)
- [引用](#引用)
- [许可证](#许可证)
- [参考文献](#参考文献)

---

## 背景

### 医学领域中领域自适应的“误用”

领域自适应（Domain adaptation，DA）在计算机视觉领域取得了巨大成功，但其在医疗与健康数据上的应用面临着独特挑战。特别是**域对抗训练**（通过梯度反转层 GRL 实现）试图学习域不变表征，而我们发现：

1. **医疗数据中的人群/队列分布高度分歧**——不同数据库在采集标准、随访流程与人群构成（年龄/分期/事件率）上存在根本性差异；
2. **缺失模式只是域信号的“可见表面”**——仅使用缺失指示器即可使域分类器达到 **90.3%** 的准确率（跨 36 个数据源 / 158,585 例样本清单；在 LIHC 5 队列定义下为 97%），但受控实验表明，在完全插补（缺失率降至 0%）之后域分类器仍可达到 97.7%，证明根本原因在于**协变量取值分布本身**；
3. **特征解耦失败**——GRL 对抗过程无法迫使编码器丢弃嵌入在分布中的这些“域指纹”，跨域泛化没有得到改善（甚至发生退化）；
4. **插补等单一补救手段无法去除域指纹**——缺失只是人群分布的一个投影；平滑缺失并不能平滑人群本身。

### 为什么是负结果？

顶级期刊（Nature Methods、JAMIA、Briefings in Bioinformatics）日益重视**可复现性研究**与**方法学警示性研究**。本文的核心贡献并非提出新方法，而是一次系统性的证伪：

| 贡献 | 说明 |
|------|------|
| 系统性验证 | 域对抗方法在 110,640 例样本、5 个 HCC 队列、3 种癌症类型上失败 |
| 机制阐明 | 人群分布而非缺失才是根本域指纹（插补反证 + MINE/Wasserstein 量化 + SHAP 可解释性） |
| 全面方法比较 | 9 种 DA/DG 方法均未超越 ERM（实验 J） |
| 临床评估 | IBS / 校准曲线 / DCA（实验 K） |
| 跨癌种验证 | 在 COAD 或 LUAD 上没有任何方法超越 ERM（实验 L/M） |
| 领域方向 | 先匹配人群分布，再进行自适应；以因果推断作为对抗训练的替代 |

---

## 核心假设

$$\text{人群分布} \gg \text{缺失} \quad \text{(作为域可判别性的根源；缺失是其表面)}$$

缺失模式对域分类器具有高度判别性（在 36 个数据源 / 158,585 例清单上为 90.3%；在 LIHC 5 队列定义下为 97%），但它们是**结果而非原因**：真正编码域身份的是人群/队列自身的协变量取值分布（受控插补证明，缺失率降至 0% 后域分类器仍可达 97.7%；`scripts/05_exp_d_imputation.py`；插补后域探针 AUC 仍为 0.970，`scripts/10_exp_i_domain_quantify.py`）。

### 假设验证路径

```
数据收集 → 计算每个数据库的缺失模式 → 训练域分类器
                   （仅凭缺失模式预测数据来源）
                                     ↓
                    域分类器准确率 > 90% ?
                                     ↓
                             ✅ 假设成立：
                                缺失模式是域指纹的“可见表面”
                              （实验 D 反证：缺失率降至 0% 后
                                域分类器仍达到 97.7%，
                                真正根本原因是人群分布）
                                     ↓
                           加入 GRL → 域分类器仍 > 80%
                                     ↓
                            ✅ GRL 无法解耦
                                     ↓
                           方法比较（实验 J）：DANN 无增益；
                             没有任何方法稳定超越 ERM
                              （GroupDRO/V-REx/Fish 在多域数据上显著更差）
                                     ↓
                           域偏移量化（实验 I）：
                             域指纹可测且不可移除
                                     ↓
                           跨癌种验证（实验 L/M）：结论可推广
```

---

## 实验概览

| ID | 名称 | 状态 | 脚本 | 核心输出 |
|---|---|---|---|---|
| A | TCGA 对比 SEER（Level 4 极端缺失） | ✅ | `scripts/02_train_dann_lihc.py` | `results/tables/table2_main_results.csv`；Fig 2 |
| B | TCGA 对比外部 HCC（Level 2，4 个域） | ✅ | `scripts/02_train_dann_lihc.py` | `results/tables/table2_main_results.csv`；Fig 2 |
| C | 全部 5 个队列（多域） | ✅ | `scripts/02_train_dann_lihc.py` | `results/tables/table2_main_results.csv`；Fig 2 |
| D | 插补“复活”（KNN k=7，缺失率 → 0%） | ✅ | `scripts/05_exp_d_imputation.py` | `results/tables/table2_main_results.csv`；Fig 3 |
| E | Level-0 干净数据对照（BRCA） | ✅ | `scripts/06_exp_e_clean_control.py` | `results/tables/table2_main_results.csv`；Fig 4 |
| F | 1000× 分层自助法 | ✅ | `scripts/04_exp_f_bootstrap.py` / `04b` | Fig 4 |
| G | SHAP 可解释性 | ✅ | `scripts/07_exp_g_shap.py` | Fig 5 |
| H | K-M 临床分层 + log-rank 检验 | ✅ | `scripts/08_exp_h_km_curves.py` | Fig 6 |
| **I** | **域偏移量化**（Wasserstein / propensity / MINE / probe） | ✅ | `scripts/10_exp_i_domain_quantify.py` | `results/tables/table5_domain_quantify.csv`；Fig 8 |
| **J** | **综合方法比较（9 种方法）** | ✅ | `scripts/11_exp_j_method_comparison.py` | `results/tables/table3_method_comparison.csv`；Fig 7 |
| **K** | **临床评估指标**（IBS / 校准 / DCA） | ✅ | `scripts/12_exp_k_clinical_metrics.py` | `results/tables/table4_clinical_metrics.csv`；Fig S4/S5 |
| **L** | **COAD 跨癌种验证** | ✅ | `scripts/13_exp_lm_multicancer_validation.py --cancers COAD` | `results/tables/table6_multicancer_validation.csv` |
| **M** | **LUAD 跨癌种验证** | ✅ | `scripts/13_exp_lm_multicancer_validation.py --cancers LUAD` | `results/tables/table6_multicancer_validation.csv` |

**审修补充实验（2026-08-02，投稿前完成）**

| ID | 名称 | 状态 | 脚本 | 核心输出 |
|---|---|---|---|---|
| R1 | 非神经网络基线（XGB-Cox / RSF） | ✅ | `scripts/17_baseline_xgboost_rsf.py` | `results/tables/table7_tree_baselines.csv` |
| R2 | BH-FDR 多重比较校正（35 次检验） | ✅ | `scripts/18_fdr_correction.py` | `results/tables/table3_method_comparison.csv`（q 值） |
| J′ | SEER 降采样敏感性（10,000 → 309，6 个层级） | ✅ | `scripts/11_exp_j_method_comparison.py --seer_subsample` | `results/tables/table3b_seer_subsample.csv` |
| D′ | 插补策略对照（队列内中位数 / 随机填充） | ✅ | `scripts/05_exp_d_imputation.py --impute_strategy all` + `scripts/23_exp_dprime_retrain.py` | **全量重训**：Δ=−0.0007，域分类器全程 ≈1.0（`results/tables/table10_dprime_retrain.csv`） |
| **α** | **GRL α 敏感性（固定三个水平 {0.1, 0.5, 1.0} × A/B/C × 3 个种子）** | ✅ | `scripts/19_exp_alpha_sensitivity.py` | `results/tables/table8_alpha_sensitivity.csv` |
| **IT** | **真实独立测试集 3 折重训（A/B/C × 3 折 × 3 个种子）** | ✅ | `scripts/21_exp_independent_test.py` | `results/tables/table9_independent_test.csv` |
| **D′R** | **D′ 队列内中位数插补，全量重训（3 个种子 × {DANN, Baseline}）** | ✅ | `scripts/23_exp_dprime_retrain.py` | `results/tables/table10_dprime_retrain.csv`（Δ=−0.0007；后期 DANN 域分类器 ≈1.0） |
| **IPW** | **IPW-ERM 原型（倾向加权 ERM；A/B/C × 3 个种子 × 4 种模式）** | ✅ | `scripts/24_exp_ipw_erm.py` | `results/tables/table11_ipw_erm.csv`（目标队列 TCGA 验证 C-index +0.016 / +0.031 / +0.055；8/9 配对为正；量级较小） |

**收尾补充实验（2026-08-02）**

| ID | 名称 | 状态 | 脚本 | 核心输出 |
|---|---|---|---|---|
| D′RF | D′ 随机填充插补，全量重训（3 个种子 × {DANN, Baseline}） | ✅ | `scripts/25_exp_dprime_randomfill_retrain.py` | `results/tables/table12_dprime_randomfill.csv`（Δ=+0.0004；域分类器 ≈1.0） |
| IPW×M | IPW × 倾向得分匹配（A/B/C × 3 个种子 × 5 种模式） | ✅ | `scripts/26_exp_ipw_matching.py` | `results/tables/table13_ipw_matching.csv`（B 共同支撑覆盖率 0.27；匹配以整体表现为代价换取目标队列；ipw_matching 二者皆失） |
| RMST | RMST 分析（观测 + 模型预测校准） | ✅ | `scripts/27_exp_rmst.py` | `results/tables/table14_rmst.csv`（TCGA 对比 SEER RMST@36 ≈+8 个月；DANN ≈ ERM 预测校准） |
| SHAP-BC | 在 B/C 上的 SHAP 可解释性（Baseline vs DANN） | ✅ | `scripts/28_exp_shap_bc.py` | `results/tables/table15_shap_bc.csv`（多域下 Stage 排名第一；在 B 中 Stage +1.37） |
| DeLong | 时间依赖 AUC 的 DeLong 显著性检验（A/B/C 部分） | ✅ | `scripts/29_exp_delong_td_auc.py` | `results/tables/table16_delong_td_auc.csv`（DANN 差异 ≤0.009；方向在不同设置间不一致） |
| IPW×Mx | IPW × 输入级 mixup（A/B/C × 3 个种子 × 4 种模式） | ✅ | `scripts/30_exp_ipw_mixup.py` | `results/tables/table17_ipw_mixup.csv`（无协同效应；目标队列增益最大） |
| SHAP-CI | SHAP B/C 自助法置信区间（B=500） | ✅ | `scripts/31_exp_shap_ci.py` | `results/tables/table18_shap_ci.csv`（B 中 Stage +1.36 [0.71, 2.13]，显著） |
| KM-TCGA | TCGA 目标队列临床特征刻画（Stage 分层 + 风险组 + RMST 差异） | ✅ | `scripts/32_exp_km_tcga.py` | `results/tables/table19_km_tcga.csv`（Stage log-rank p=0.0078；TCGA−SEER RMST@36 +8.15 个月，p<0.001） |

> **标签说明**：在早期报告中保留了“G=SHAP，H=K-M”的标签；2026-07-31 完成的新实验在 H 之后连续编号（I–M），以避免标签冲突。审修补充实验复用 R1/R2/J′/D′/α/IT/D′R/IPW 标签。四个收尾补充实验（D′RF / IPW×M / RMST / SHAP-BC）与另外四个收尾补充实验（DeLong / IPW×Mixup / SHAP-CI / TCGA-KM）汇总于上表。

---

## 关键发现

| # | 发现 | 证据 | 实验 / 图 |
|---|---|---|---|
| 1 | **DANN 零增益** | Δ ∈ [−0.0032, +0.0007]；Baseline 在 9/9 个时间点上更优 | A/B/C；Fig 2 |
| 2 | **域分类器无法被击败** | 2 域设置在 5 个 epoch 内达到 97%；α 无法压制它；多域训练震荡而不收敛 | A/B/C；Fig 2 |
| 3 | **插补无法复活 GRL** | 缺失率降至 0% 后，域分类器仍达到 97.7%；Δ 不显著 | D；Fig 3 |
| 4 | **GRL 在干净数据上收益最大——但仍不显著** | 域分类器仅 64.3%；Δ=+0.0074（p=0.058，**在 α=0.05 下不显著**；仅为一个理论上界） | E；Fig 4 |
| 5 | **对抗解耦失败；FDR 后仅输入级 mixup 为正向** | DANN 在任何划分中均无增益（p>0.65）；经 35 个 p 值的 BH-FDR 校正后仅 Mixup（+0.019，q=0.012）存活；IRM（p=0.043）未通过；GroupDRO/V-REx/Fish 显著有害（V-REx B：q=0.023） | J；Fig 7；`scripts/18_fdr_correction.py` |
| 6 | **域指纹可被定量刻画** | Wasserstein W1 = 8.6 年 / 0.15；倾向重叠 0.18；探针 AUC > 0.96（插补后不变） | I；Fig 8 |
| 7 | **临床评估** | 模型优于 KM 零假设（IBS 0.202 vs 0.207），ECE ≈ 0.03–0.08，DCA 接近 | K；Fig S4/S5 |
| 8 | **跨癌种泛化** | COAD/LUAD 上没有任何方法超越 ERM | L/M；Fig 8 |

---

## 机制验证

### 因果链（实验 D/E/F/G/H）

```
原假设：缺失模式 ──(最强指纹)──▶ 域分类器 97% ──▶ GRL 失败
实验 D 证伪：插补后缺失率=0%，域分类器仍达 97.7%           ✗ 缺失并非根本原因
实验 E 支持：干净数据上域分类器仅 64%，GRL Δ=+0.0074       ✓ 指纹较弱时 GRL 有效
实验 F 收口：5 个实验中 Δ 全部 p>0.05，C 显著为负            ✓ 统计上无收益
实验 G 佐证：SHAP 显示 DANN 未破坏特征注意力（Stage +26%）   ✓ 失败 ≠ “学到了错误的东西”
实验 H 佐证：两种模型的 K-M 分层完全一致（P ≈ 1e-25~1e-28）  ✓ 临床能力无损失
实验 I 量化：Wasserstein/propensity/MINE/probe 刻画分布级指纹 ✓ 可测，插补无法移除
实验 J 推广：DANN 无增益；没有任何方法稳定超越 ERM；
              GroupDRO/V-REx/Fish 显著更差                    ✓ 范式级问题
实验 L/M 普适：COAD/LUAD 上出现同样的失败                    ✓ 跨癌种成立
```

### 关键数字一览

| 指标 | 值 |
|------|:----:|
| LIHC 数据规模 | 110,640 例 × 5 个队列 |
| DANN C-index Δ（A/B/C） | +0.0007 / −0.0032 / −0.0011 |
| GRL α 敏感性（固定三个水平 {0.1, 0.5, 1.0} × 3 个种子） | 三个 α 水平全部**不显著**（perm-p ≥ 0.2）；α 上最佳 Δ = +0.0005 / +0.0016 / +0.0049（C·α=1.0 是 9 组比较中最大者，p=0.20，量级 <0.01——无临床意义）；随着 α 升高，B/C 中域分类器被压制到 0.43/0.64——**GRL 机制确实在工作，但生存预测毫无所得**（见 table8） |
| 真实独立测试集 3 折重训（A/B/C × 3 折 × 3 个种子） | 测试集 Δ(DANN−Base) = −0.0002 / +0.0043 / +0.0033；符号翻转置换 p = 0.71 / 0.46 / 0.23（全部 95% CI 包含 0，**不显著**）；最优验证集性能高估了测试性能（B 上差距 ≈ +0.029）——在严格无泄漏协议下，负结论仍然成立（见 table9） |
| D′ 队列内中位数插补，全量重训（3 个种子 × {DANN, Baseline}） | 最优验证 C-index：DANN 0.6259 ± 0.0022 vs Baseline 0.6266 ± 0.0024，Δ = **−0.0007（不显著）**；**后期 DANN 域分类器 ≈1.0**（0.9997 / 1.0 / 1.0）——无泄漏的队列内插补保留了结构性指纹（SEER Grade 99.7% 缺失），GRL 在生存预测上仍毫无所得（见 table10） |
| IPW-ERM 原型（倾向加权，A/B/C × 3 个种子 × 4 种模式） | 机制方向成立：**目标队列 TCGA 验证 C-index +0.016 / +0.031 / +0.055**（8/9 配对为正），但以整体（SEER 主导）性能为代价：A/C 中整体性能略有下降（−0.007 / −0.004），B 中持平（+0.006）；量级很小，处于 46 例噪声带之内，无法单独挽救跨人群预测（结构性不可观测的字段无法通过重加权修复）（见 table11） |
| D′ 随机填充插补，全量重训（3 个种子 × {DANN, Baseline}） | 最优验证 C-index：DANN 0.6062 ± 0.0018 vs Baseline 0.6058 ± 0.0015，Δ = **+0.0004（不显著）**；随机填充后探针 AUC 仍 = 1.000，**后期 DANN 域分类器 ≈1.0**——与队列内中位数重训（Δ=−0.0007）一致：三种插补策略（KNN / 队列内中位数 / 随机填充）均通过探针与全量重训两项检验，且都无法消除结构性指纹或使 GRL 获益（见 table12） |
| IPW × 倾向得分匹配（A/B/C × 3 个种子 × 5 种模式） | 匹配诊断：B 的共同支撑覆盖率仅为 0.27（外部队列与 TCGA 的倾向得分几乎不重叠），匹配对 logit 距离 7–12——结构性不可观测性导致共同支撑非常稀薄；结果在 table13 中如实报告（**匹配以整体表现为代价换取目标队列；ipw_matching 二者皆失**），进一步印证简单的分布匹配无法两全 |
| RMST 分析（KM 观测 + 模型预测校准） | 观测：**TCGA RMST@36 = 26.42 个月 vs SEER 18.43 个月（≈+8 个月）**，AMC 34.52——结局分布是域身份中最难处理的分量之一；模型：DANN 与 ERM 在 TCGA 上预测的 RMST@36 几乎相同（27.83 vs 27.96，均将观测值 23.77 高估约 4 个月）——在 RMST 尺度上 GRL 同样无法改善目标队列的校准（见 table14） |
| 扩展至 B/C 的 SHAP 可解释性 | 在 B 中，DANN 将 Stage 的均值\|SHAP\| 提升 +1.37（7.45 vs 6.08），目标队列 TCGA 的 Stage 由 10.66 → 17.58；在 C 中，DANN 使各特征整体下降，但 Stage 仍排名第一——**在多域设置下，GRL 从未将关键临床特征挤出榜首**（“失败是结构性的，而非学到了错误的东西”在多域设置下依然成立）（见 table15） |
| DeLong 时间依赖 AUC 检验（`scripts/29_exp_delong_td_auc.py`） | **DANN 与 ERM 的 AUC 差异 ≤0.009（验证集几乎全部 ≤0.002），且方向在不同设置间不一致**（在 A 的后期验证中略为负，在 C 的验证与测试中略为正）；B 部分正确复现了已知显著的方法（Mixup C Δ=+0.023，p=0.003；V-REx B Δ=−0.098，p=0.002）；A/B 的真实测试集均不显著——**在每一项指标下，GRL 都没有提供可复现的实质性增益**（见 table16） |
| IPW × mixup（`scripts/30_exp_ipw_mixup.py`） | **叠加无协同效应**：ipw_mixup 的整体最优验证在 A/C 中比单独 mixup 更差（−0.024 / −0.019），在 B 中持平；但其目标队列 TCGA 验证增益（+0.060 / +0.029 / +0.094）是四种模式中最大的——重加权与输入插值都以整体表现为代价换取目标队列，因此不能简单叠加（见 table17） |
| SHAP B/C 自助法置信区间（`scripts/31_exp_shap_ci.py`） | B 中 **Stage +1.36 的 95% CI 为 [0.71, 2.13]，显著不同于 0**（点估计升级为显著结论）；Grade −2.01 显著；在 C 中四个特征均显著为负，但排名仍保持 Stage 第一——SHAP 结论如今有了显著性支撑（见 table18） |
| TCGA 目标队列临床特征刻画（`scripts/32_exp_km_tcga.py`） | TCGA 临床 Stage 分层 log-rank **p=0.0078**（结局分布本身就是一种域指纹）；ERM/DANN 风险组 K-M 曲线均显著分离（p<0.001）；**TCGA−SEER RMST@36 差异 +8.15 个月 [6.75, 9.66]，p<0.001**，TCGA−MSK 边缘显著 +1.58 个月，p=0.06——结局分布偏移从点估计升级为显著性检验（见 table19） |
| 时间依赖 AUC（12/36/60，三个实验） | Baseline 在 9/9 个时间点上更优 |
| 域分类器峰值（2 域 / 插补后 / 干净数据） | 97.3% / 97.7% / **64.3%** |
| 域探针 AUC（LIHC / 插补后 / 干净数据） | >0.96 / 0.970 / **0.709** |
| Wasserstein W1（Age / Stage） | 8.59 / 0.153 |
| 倾向得分重叠系数 | 0.18 |
| 方法比较（9 种方法 vs ERM） | DANN 全部 p>0.65；**经 BH-FDR（35 次检验）后仅 Mixup（C：+0.019，q=0.012）为正、V-REx（B：−0.067，q=0.023）为负**；其余名义上显著的方法（如 IRM）未通过（见 table3） |
| 树模型基线（相同 4 个特征，XGB-Cox / RSF） | A：0.642/0.624，B：0.617/0.624，C：**0.657**/0.644——与 Transformer ERM（0.633–0.639）处于同一平台，证明 0.63 的天花板是特征集的属性（`scripts/17_baseline_xgboost_rsf.py`） |
| IBS（ERM/DANN vs KM 零假设） | 0.202 / 0.203 vs 0.207 |

---

## 数据

### 5 个 LIHC 队列（实验 A/B/C/D）

| 队列 | 类型 | N | 事件率 | 显著缺失 |
|---|---|---|---|---|
| TCGA_LIHC | 基因组数据库 | 309 | 42.7% | Stage 20.7% |
| US_SEER | 人群登记数据库 | 108,638 | 78.1% | Grade 100% |
| hcc_msk_2024 | MSK 独立队列 | 1,357 | 59.1% | Age 100% |
| lihc_amc_prv | 韩国 AMC | 231 | 16.0% | Sex/Stage/Grade 100% |
| hcc_meric_2021 | 瑞士巴塞尔 | 105 | 78.1% | Stage/Grade 100% |

> 训练期间，SEER 使用 `RandomState(42)` 降采样至 10,000 例。完整缺失矩阵见 `results/tables/table1_cohort_summary.csv`。

### 其他数据（实验 E/L/M）

- E：TCGA-BRCA（来源 1,098）对比 METABRIC（来源 2,509），特征为 Age + Sex + Stage（排除 Grade）；过滤生存时间/结局后，2,995 例进入分析（BRCA 1,015 / METABRIC 1,980）。
- L：TCGA-COAD 对比 CPTAC-COAD（原计划的 `coadread_dfci_2016` 无生存数据，因此改用 CPTAC）。
- M：TCGA-LUAD 对比 MSKCC-2020。

### 数据可用性

原始数据与已处理数据**不**随本仓库分发，因为 TCGA、SEER 与 cBioPortal 数据受各自许可条款约束。相反，数据通过运行下载脚本获取，再重新生成处理后的文件：

- `scripts/download_tcga.py` —— 下载 TCGA GDC 数据（LIHC、BRCA、COAD、LUAD）。
- `scripts/download_additional_data.py` —— 下载 SEER、MSK（`hcc_msk_2024`）、AMC（`lihc_amc_prv`）、METABRIC、CPTAC-COAD、MSKCC-2020 及其它 cBioPortal/登记数据库队列。
- SEER 需要签署数据使用协议；`scripts/process_seer.py` 记录了预期的 SEER 导出格式，并将其转换为统一模式。

大型、可再生的人工产物——模型权重（`*.pth`）、中间特征（`*.npy`）与原始数据——通过 `.gitignore` 排除在仓库之外，运行下面的流程即可生成。`scripts/paths.py` 集中管理数据根目录的解析（支持环境变量 `TRANSDANN_DATA_ROOT` 覆盖；默认回退到本地 `results/` 目录）。

| 组件 | 是否分发 | 说明 |
|---|---|---|
| 代码（`scripts/`） | 是 | 完整流程与全部实验脚本，MIT 许可 |
| 论文图集（`results/figures/paper/`） | 是 | Figure 1–8 + S1–S5（PNG/PDF）+ 图注 + 补充表 |
| 论文数据表（`results/tables/`） | 是 | table1–table19 + table3b CSV |
| 原始/已处理数据（`data_raw/`、`data_processed/`） | 否 | 受许可限制；通过 `scripts/download_*.py` 获取 |
| 模型权重 / 中间特征（`*.pth`、`*.npy`） | 否 | 通过 `.gitignore` 排除；运行实验重新生成 |

---

## 方法

### 统一数据模式

`Age`（连续）+ `Sex` / `Stage` / `Grade`（分类）+ `Survival_Months` / `Vital_Status`。

### 模型：TransDANNSurvV3

```
Input features → continuous MLP embedding / categorical Embedding(padding_idx=missing)
              → [CLS] + positional encoding → Transformer (4 layers × 8 heads, d_model=128)
              → CLS representation → survival head (DeepHit, 32 bins)
                        └→ GradientReversal(α) → domain-classification head
```

- 总参数量 ≈ 818K；DANN 与 Baseline 仅在 GRL + 域损失上不同。
- 训练：AdamW 5e-4，CosineAnnealing，早停耐心值 = 40，损失按队列平均。
- 引擎范围：主实验 A/B/C（`02`）使用**单一种子（42）**；方法比较 J 与多癌种实验 L/M（`11`/`13`）使用 **3 个种子（42/43/44）**取平均。两种编号方式的差异 <0.002，结论一致。
- 全部 9 种比较方法共用该骨干网络（实现见 `scripts/method_utils.py`）。

---

## 结果：图表

### 论文图表集（Figure 1–8 + S1–S5，`results/figures/paper/`）

一套高密度、达到投稿质量的图表集，按论文叙事顺序重新组织（由 `scripts/16_generate_paper_figures.py` 生成，PNG 300 dpi + PDF 矢量，附可直接用于投稿的图注）：

| 章节 | 图 | 文件 | 面板 | 叙事功能 |
|---|---|---|---|---|
| 3.1 数据刻画 | **Fig 1** | `Figure_01_domain_fingerprint` | 2×2 | 域指纹：缺失是表面，分布才是本质 |
| 3.2 主要结果 | **Fig 2** | `Figure_02_main_results` | 2×2 | C-index / 时间 AUC / 合并分队列 / 域分类器：零到负增益 |
| 3.3 机制 I | **Fig 3** | `Figure_03_imputation` | 2×2 | 插补不能移除域指纹（证伪“缺失是根本原因”） |
| 3.4 机制 II + 统计 | **Fig 4** | `Figure_04_clean_data_bootstrap` | 2×3 | 干净数据是 GRL 的最佳场景；bootstrap 仍不显著 |
| 3.5 可解释性 | **Fig 5** | `Figure_05_shap` | 2×2 | SHAP：GRL 并未破坏特征注意力 |
| 3.6 临床验证 | **Fig 6** | `Figure_06_km_clinical` | 2×2 | 等价的 K-M 分层 |
| 3.7 方法比较 | **Fig 7** | `Figure_07_methods` | 2×3 | 9 种方法的系统性失败 |
| 3.8 量化 + 普适性 | **Fig 8** | `Figure_08_quantify_multicancer` | 2×3 | Wasserstein / propensity / MINE / 多癌种 |
| 附录 | **S1–S5** | `supplementary/Figure_S1..S5_*` | — | t-SNE / 训练动态 / GRL 调度 / 校准 / DCA |

配套文档与补充数据表（均位于 `results/figures/paper/` 下）：

- `Figure_Legends.md` —— 全部 13 张图的完整图注（可直接复制进投稿材料）。
- `README_figures.md` —— 图-章节对应表、数据来源清单，以及相对早期草稿的修正说明。
- `supplementary/Supp_Table_S1..S7_*.csv` —— 补充数据表：**S1 全部 36 个数据源的完整清单**（样本量 / 事件率 / 癌种 / 每个数据源用于哪些实验），**S2 队列汇总**（基线表，含缺失率），S3 基线结果（DANN vs Baseline），S4 9 种方法比较，S5 临床指标，S6 域量化，S7 多癌种验证。

> 一条命令即可复现：`scripts/16_generate_paper_figures.py all`。图内数字读取自 `results/tables/*.csv`；少量坐标轴标签与峰值硬编码为与这些表保持一致（例如域分类器峰值 97.3%/97.7%）。论文表中的全部数字均可追溯到相应的实验脚本。

### 论文数据表（`results/tables/`）

| 数据表 | 内容 |
|---|---|
| `table1_cohort_summary.csv` | 队列汇总（缺失率/事件率） |
| `table2_main_results.csv` | A–H 主要结果（主引擎，单种子） |
| `table3_method_comparison.csv` | 9 种方法比较（C-index/IBS/Δ/p + BH-FDR q 值） |
| `table3b_seer_subsample.csv` | J′ SEER 降采样敏感性（6 个层级） |
| `table4_clinical_metrics.csv` | IBS/ECE |
| `table5_domain_quantify.csv` | MINE/探针 |
| `table6_multicancer_validation.csv` | COAD/LUAD |
| `table7_tree_baselines.csv` | R1 XGBoost-Cox / RSF 基线 |
| `table8_alpha_sensitivity.csv` | GRL α 敏感性（固定三个水平 {0.1, 0.5, 1.0} × A/B/C × 3 个种子） |
| `table9_independent_test.csv` | 真实独立测试集 3 折重训（DANN vs Baseline 测试 C-index + 符号翻转置换 p） |
| `table10_dprime_retrain.csv` | D′ 队列内中位数插补，全量重训（最优验证 C-index + 后期域分类器准确率） |
| `table11_ipw_erm.csv` | IPW-ERM 原型（erm/dann/erm_pooled/ipw_erm C-index + 分队列验证） |
| `table12_dprime_randomfill.csv` | D′ 随机填充插补，全量重训（Δ=+0.0004，域分类器 ≈1.0） |
| `table13_ipw_matching.csv` | IPW × 倾向得分匹配（A/B/C × 3 个种子 × 5 种模式 + 匹配诊断） |
| `table14_rmst.csv` | RMST 分析（观测 KM RMST + 模型预测 vs 观测校准） |
| `table15_shap_bc.csv` | SHAP B/C 特征重要性（Baseline vs DANN，分队列） |
| `table16_delong_td_auc.csv` | DeLong 时间依赖 AUC 检验（DANN/各方法与 ERM 的 ΔAUC + DeLong p + bootstrap p + 两种 AUC 范围） |
| `table17_ipw_mixup.csv` | IPW × mixup（erm/mixup/ipw_erm/ipw_mixup C-index + 分队列验证） |
| `table18_shap_ci.csv` | SHAP 均值\|SHAP\| 自助法 95% CI（Baseline/DANN/Δ，B/C） |
| `table19_km_tcga.csv` | TCGA K-M（临床 Stage 分层 + RMST@36 差异 bootstrap 检验） |

---

## 项目结构

```
TransDANN_Liver_Cancer/
├── README.md                    # 项目概览（本文件）
├── LICENSE                      # MIT 许可证
├── requirements.txt             # Python 依赖
├── .gitignore                   # 排除原始数据、权重、内部文档、日志、遗留归档
├── run_lihc_pipeline.sh         # 全流程启动器
├── start_training.sh            # 训练启动器
│
├── scripts/                     # 全部 Python 代码（按实验/输出顺序编号）
│   ├── paths.py                 # 数据路径接口（数据根目录解析）
│   ├── download_tcga.py         # TCGA GDC 数据下载
│   ├── download_additional_data.py  # SEER / cBioPortal / 额外队列下载
│   ├── process_seer.py          # SEER 导出解析 → 统一模式
│   ├── 01_prepare_lihc_data.py  # 数据准备 → lihc_all_cohorts.csv
│   ├── 02_train_dann_lihc.py    # 实验 A/B/C（DANN vs Baseline 训练引擎）
│   ├── 03_visualize_lihc.py     # 主图可视化
│   ├── 04/04b_exp_f_bootstrap.py    # 实验 F：1000× 分层自助法
│   ├── 05_exp_d_imputation.py   # 实验 D：KNN 插补 + 重训
│   ├── 06_exp_e_clean_control.py# 实验 E：干净数据对照
│   ├── 07_exp_g_shap.py         # 实验 G：SHAP 可解释性
│   ├── 08_exp_h_km_curves.py    # 实验 H：K-M 分层
│   ├── 09_make_figure_package.py# 遗留图池（历史来源池）
│   ├── 10_exp_i_domain_quantify.py    # 实验 I：域偏移量化
│   ├── 11_exp_j_method_comparison.py  # 实验 J：方法比较（9 种方法）
│   ├── 12_exp_k_clinical_metrics.py   # 实验 K：临床指标
│   ├── 13_exp_lm_multicancer_validation.py  # 实验 L/M：多癌种验证
│   ├── 15_make_summary_tables.py      # 生成表 1/2
│   ├── 16_generate_paper_figures.py   # 论文图集（Figure 1–8 + S1–S5 + 图注 + 补充表）
│   ├── 17_baseline_xgboost_rsf.py     # R1：非神经网络基线（XGB-Cox/RSF）
│   ├── 18_fdr_correction.py           # R2：BH-FDR 多重比较校正
│   ├── 19_exp_alpha_sensitivity.py    # GRL α 敏感性（固定三个水平 {0.1, 0.5, 1.0}）
│   ├── 20_alpha_sweep_table.py        # α 敏感性汇总 → table8
│   ├── 21_exp_independent_test.py     # 真实独立测试集 3 折重训（A/B/C × 3 折 × 3 个种子）
│   ├── 22_independent_test_table.py   # 独立测试汇总 → table9（符号翻转置换检验）
│   ├── 23_exp_dprime_retrain.py       # D′ 队列内中位数插补，全量重训 → table10
│   ├── 24_exp_ipw_erm.py              # IPW-ERM 原型（逆倾向加权 ERM）→ table11
│   ├── 25_exp_dprime_randomfill_retrain.py  # D′ 随机填充插补，全量重训 → table12
│   ├── 26_exp_ipw_matching.py         # IPW × 倾向得分匹配 → table13
│   ├── 27_exp_rmst.py                 # RMST 分析（观测 + 模型校准）→ table14
│   ├── 28_exp_shap_bc.py              # SHAP 可解释性扩展至 B/C → table15
│   ├── 29_exp_delong_td_auc.py        # DeLong 时间依赖 AUC 显著性 → table16
│   ├── 30_exp_ipw_mixup.py            # IPW × mixup → table17
│   ├── 31_exp_shap_ci.py              # SHAP 均值|SHAP| 自助法 CI → table18
│   ├── 32_exp_km_tcga.py              # TCGA K-M + RMST@36 差异 → table19
│   ├── method_utils.py          # 9 种方法的损失/惩罚实现
│   ├── transdann_utils.py       # 共享模型/工具（TransDANNSurvV3）
│   ├── lihc_recon.py            # LIHC 重建辅助函数
│   └── fast_cindex.py           # 快速一致性指数实现
│
└── results/                     # 论文交付物
    ├── tables/                  # table1–table19 + table3b CSV
    └── figures/paper/           # 论文图集：Figure 1–8 + S1–S5 + 图注 + 补充表
```

---

## 安装

```bash
# Linux / GPU 环境（已验证：PyTorch 2.3.1+cu121、CUDA、Tesla V100 32GB）
cd /path/to/TransDANN_Liver_Cancer
pip install -r requirements.txt
pip install scikit-survival>=0.22 lifelines>=0.27 shap   # 额外依赖
```

依赖：`torch>=2.0, numpy, pandas, scikit-learn, scikit-survival, lifelines, matplotlib, seaborn, openpyxl, pyarrow, shap`。

---

## 复现命令

```bash
# 0) 下载原始数据（TCGA / SEER / cBioPortal——受许可条款约束；见“数据”一节）
python3 scripts/download_tcga.py
python3 scripts/download_additional_data.py
python3 scripts/process_seer.py                         # 如有 SEER 数据可用

# 数据准备
python3 scripts/01_prepare_lihc_data.py                # → data_processed/lihc_all_cohorts.csv

# 主实验 A/B/C
python3 scripts/02_train_dann_lihc.py --experiment ALL --epochs 200 --surv_type deephit \
    --d_model 128 --n_layers 4 --dropout 0.15 --lr 5e-4 --domain_weight 0.3
python3 scripts/03_visualize_lihc.py                   # 主图（遗留）

# 补充实验 D–H
python3 scripts/05_exp_d_imputation.py --epochs 200    # D
python3 scripts/06_exp_e_clean_control.py --epochs 200 # E
python3 scripts/04_exp_f_bootstrap.py                  # F（全量集合范围）
python3 scripts/04b_exp_f_val_bootstrap.py             # F（验证集范围）
python3 scripts/07_exp_g_shap.py                       # G
python3 scripts/08_exp_h_km_curves.py                  # H

# 实验 I–M
python3 scripts/10_exp_i_domain_quantify.py            # I：域偏移量化
python3 scripts/11_exp_j_method_comparison.py --experiments A B C --seeds 42 43 44  # J：方法比较
python3 scripts/12_exp_k_clinical_metrics.py           # K：临床指标（先运行 J 的 A）
python3 scripts/13_exp_lm_multicancer_validation.py --cancers COAD LUAD --seeds 42 43 44  # L/M

# 审修补充实验
python3 scripts/17_baseline_xgboost_rsf.py             # R1：非神经网络基线
python3 scripts/18_fdr_correction.py                   # R2：BH-FDR 校正
python3 scripts/19_exp_alpha_sensitivity.py            # GRL α 敏感性（A/B/C × {0.1, 0.5, 1.0} × 3 个种子）
python3 scripts/21_exp_independent_test.py             # 真实独立测试集 3 折重训（A/B/C × 3 折 × 3 个种子）
python3 scripts/23_exp_dprime_retrain.py               # D′ 队列内中位数插补，全量重训（3 个种子 × {DANN, Baseline}）
python3 scripts/24_exp_ipw_erm.py                      # IPW-ERM 原型（A/B/C × 3 个种子 × 4 种模式）

# 收尾补充实验
python3 scripts/25_exp_dprime_randomfill_retrain.py    # D′ 随机填充插补，全量重训（3 个种子 × {DANN, Baseline}）
python3 scripts/26_exp_ipw_matching.py                 # IPW × 倾向得分匹配（A/B/C × 3 个种子 × 5 种模式）
python3 scripts/27_exp_rmst.py                         # RMST 分析（观测 KM + 模型校准）
python3 scripts/28_exp_shap_bc.py                      # SHAP 可解释性扩展至 B/C
python3 scripts/29_exp_delong_td_auc.py                # DeLong 时间依赖 AUC 显著性（C 部分需要 GPU）
python3 scripts/30_exp_ipw_mixup.py                    # IPW × mixup（A/B/C × 3 个种子 × 4 种模式）
python3 scripts/31_exp_shap_ci.py                      # SHAP 均值|SHAP| 自助法 CI（先运行 28）
python3 scripts/32_exp_km_tcga.py                      # TCGA K-M + RMST@36 差异

# 数据表与图集
python3 scripts/15_make_summary_tables.py              # table1/2
python3 scripts/20_alpha_sweep_table.py                # α 敏感性汇总 → table8
python3 scripts/22_independent_test_table.py           # 独立测试汇总 → table9
python3 scripts/16_generate_paper_figures.py all       # 论文图集：Figure 1–8 + S1–S5 + 图注 + 补充表

# 查看结果
ls results/tables/                                     # 论文数据表
ls results/figures/paper/                              # 论文图集
```

---

## 引用

本工作目前作为一篇负结果方法学论文正在审稿中。如果您使用本仓库或认为这些发现有用，请引用：

```bibtex
@misc{transdann_failure_2026,
  author = {Geng, Qiushuo},
  title  = {When Domain Adversarial Training Fails: Population Distributions as the Fundamental
            Domain Fingerprint in Cancer Survival Prediction},
  year   = {2026},
  note   = {110,640 LIHC patients, 5 HCC cohorts, 3 cancer types},
}
```

预印本可能发布于 arXiv（cs.LG / stat.ML / q-bio.QM）；可通过为 GitHub release 打标签在 Zenodo 生成 DOI。

---

## 许可证

本仓库基于 **MIT 许可证**（见 `LICENSE`）发布。请注意，原始数据（TCGA / SEER / cBioPortal）受各自数据使用协议约束，不在此重新分发。

---

## 参考文献

1. Ganin, Y., et al. "Domain-adversarial training of neural networks." *JMLR* 2016.
2. Arjovsky, M., et al. "Invariant Risk Minimization." *arXiv* 2019.
3. Sagawa, S., et al. "Distributionally Robust Neural Networks for Group Shifts." *NeurIPS* 2020.
4. Krueger, D., et al. "Out-of-Distribution Generalization via Risk Extrapolation (V-REx)." *ICML* 2021.
5. Shi, Y., et al. "Gradient Matching for Domain Generalization." *arXiv* 2021.
6. Sun, B., Saenko, K. "Deep CORAL." *arXiv* 2016.
7. Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." *ICLR* 2018.
8. Li, D., et al. "Learning to Generalize: Meta-Learning for Domain Generalization." *AAAI* 2018.
9. Katzman, J.L., et al. "DeepSurv." *BMC Med Res Methodol* 2018.
10. Lee, C., et al. "DeepHit." *AAAI* 2018.
11. Curtis, C., et al. "The genomic and transcriptomic architecture of 2,000 breast tumours." *Nature* 2012 (METABRIC).
12. Colaprico, A., et al. "TCGAbiolinks." *Nucleic Acids Res* 2016.
13. Cerami, E., et al. "The cBio Cancer Genomics Portal." *Cancer Discov* 2012.
14. The Cancer Genome Atlas Research Network. "Comprehensive and Integrative Genomic Characterization of Hepatocellular Carcinoma." *Cell* 2017.
15. Chen, T., Guestrin, C. "XGBoost." *KDD* 2016.
16. Ishwaran, H., et al. "Random survival forests." *Ann Appl Stat* 2008.
17. Belghazi, M.I., et al. "MINE: Mutual Information Neural Estimation." *ICML* 2018.
18. Vickers, A.J., Elkin, E.B. "Decision curve analysis." *Med Decis Making* 2006.
19. Rosenbaum, P.R., Rubin, D.B. "The central role of the propensity score in observational studies for causal effects." *Biometrika* 1983.
20. Shimodaira, H. "Improving predictive inference under covariate shift by weighting the log-likelihood function." *J. Stat. Plan. Inference* 2000.
21. Sugiyama, M., Krauledat, M., Müller, K.-R. "Covariate shift adaptation by importance weighted cross validation." *JMLR* 2007.
22. Benjamini, Y., Hochberg, Y. "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *JRSS-B* 1995.

---

> 在科学中，知道什么*行不通*有时与知道什么*行得通*同样有价值。这项对 110,640 例肝癌患者、另外两种癌种、五个队列以及九种域自适应/域泛化方法的系统性研究表明：**人群分布的差异是医学数据最根本的域指纹——插补无法将其消除，对抗训练无法将其分离——这正是域对抗方法在跨人群生存预测中系统性失败的原因**。
>
> **核心建议**：未来的医学域自适应研究应首先解决**人群分布匹配 / 标准化**问题，而不是仅仅在算法上进行创新。只有当域身份与生存信号不再深度纠缠时，迁移学习才有意义。
