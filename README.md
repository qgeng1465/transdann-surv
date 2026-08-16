# TransDANN Failure Analysis: Cross-Population Cancer Survival Prediction

## When Gradient-Reversal Adversarial Training (GRL/DANN) Fails: Population Distributions as the Root Domain Fingerprint (Missingness Is Its Visible Surface)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21967922.svg)](https://doi.org/10.5281/zenodo.21967922)

**English** · [**中文 (Chinese)**](README.zh-CN.md)

> **English abstract.** In multi-institutional cancer survival prediction, the *distribution of
> cohort-level covariates* (age, stage, event rate), of which missingness patterns are only the most
> visible surface, constitutes the fundamental **domain fingerprint**. We provide large-scale
> mechanistic evidence that **gradient-reversal domain-adversarial training (GRL/DANN)**, and more
> generally methods that try to **decouple domain identity from the learned representation**, fail to
> improve cross-population survival prediction across **110,640 HCC patients / 5 cohorts / 3 cancer
> types**: a domain classifier reaches 97–98% and cannot be suppressed by gradient reversal; controlled
> imputation (missingness → 0%) leaves it at 97.7%; and a leakage-free within-cohort imputation cannot
> fill fields that are structurally unobservable in a cohort (SEER grade, 100% missing), where the
> fingerprint is retained perfectly. By contrast, **input-level augmentation (mixup) is the only method
> that survives Benjamini–Hochberg FDR correction** (Δ=+0.019, q=0.012, in the 5-cohort split), whereas
> alignment/invariance-based methods yield no consistent benefit. Some (GroupDRO, V-REx, Fish) are
> significantly *harmful* on heterogeneous multi-domain data (Δ=−0.05 to −0.07, q<0.05). The failure is
> structural: domain identity is encoded in the covariate distribution itself, so **adversarial feature
> decoupling** cannot remove it, while **input-level augmentation** sidesteps it.

---

## Table of Contents

- [Background](#background)
- [Core Hypothesis](#core-hypothesis)
- [Experiment Overview](#experiment-overview)
- [Key Findings](#key-findings)
- [Mechanism Verification](#mechanism-verification)
- [Data](#data)
- [Methods](#methods)
- [Results: Figures and Tables](#results-figures-and-tables)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Reproduction Commands](#reproduction-commands)
- [Citation](#citation)
- [License](#license)
- [References](#references)

---

## Background

### The "Misuse" of Domain Adaptation in Medicine

Domain adaptation (DA) has been highly successful in computer vision, but its application to medical
and health data faces unique challenges. In particular, **domain-adversarial training** (implemented
via a gradient-reversal layer, GRL) tries to learn domain-invariant representations, and we find that:

1. **Population/cohort distributions in medical data are highly divergent**: different databases
   differ fundamentally in acquisition standards, follow-up procedures, and population composition
   (Age/Stage/event rate);
2. **Missingness patterns are only the "visible surface" of the domain signal**: a domain classifier
   reaches **90.3%** accuracy using only missingness indicators alone (across a 36-source /
   158,585-case inventory; 97% under the LIHC 5-cohort definition), yet controlled experiments show
   that after complete imputation (0% missingness) the domain classifier still reaches 97.7%,
   demonstrating that the root cause is the **covariate-value distribution itself**;
3. **Feature decoupling fails**: the GRL adversarial procedure cannot force the encoder to discard
   these "domain fingerprints" embedded in the distributions, and cross-domain generalization does not
   improve (it even degrades);
4. **Single remedies such as imputation cannot remove the domain fingerprint**: missingness is only
   one projection of the population distribution; smoothing the missingness does not smooth the
   population itself.

### Why a Negative Result?

Top-tier venues (Nature Methods, JAMIA, Briefings in Bioinformatics) increasingly value
**reproducibility studies** and **methodological cautionary notes**. The core contribution of this
paper is not a new method but a systematic falsification:

| Contribution | Description |
|------|------|
| Systematic validation | Domain-adversarial methods fail on 110,640 cases, 5 HCC cohorts, 3 cancer types |
| Mechanism elucidation | Population distributions, not missingness, are the root domain fingerprint (imputation counter-evidence + MINE/Wasserstein quantification + SHAP interpretability) |
| Comprehensive method comparison | None of 9 DA/DG methods surpasses ERM (Experiment J) |
| Clinical evaluation | IBS / calibration curves / DCA (Experiment K) |
| Cross-cancer validation | No method surpasses ERM on COAD or LUAD (Experiments L/M) |
| Direction for the field | Match population distributions first, then adapt; causal inference as an alternative to adversarial training |

---

## Core Hypothesis

$$\text{Population Distributions} \gg \text{Missingness} \quad \text{(as the root of domain discriminability; missingness is its surface)}$$

Missingness patterns are highly discriminative for a domain classifier (90.3% across a
36-source / 158,585-case inventory; 97% under the LIHC 5-cohort definition), but they are the
**outcome, not the cause**: what actually encodes domain identity is the covariate-value
distribution of the population/cohort itself (controlled imputation demonstrates that the domain
classifier still reaches 97.7% after missingness drops to 0%;
`scripts/05_exp_d_imputation.py`; the domain-probe AUC remains 0.970 after imputation,
`scripts/10_exp_i_domain_quantify.py`).

### Hypothesis-Verification Path

```
Data collection → compute the missingness pattern of each database → train a domain classifier
                   (predict data source from missingness pattern alone)
                                     ↓
                    domain classifier accuracy > 90% ?
                                     ↓
                             ✅ Hypothesis holds:
                                missingness pattern is the "visible surface"
                                of the domain fingerprint
                              (Experiment D counter-evidence: after 0% missingness
                               the domain classifier still reaches 97.7%,
                               the true root cause is the population distribution)
                                     ↓
                            add GRL → domain classifier still > 80%
                                     ↓
                            ✅ GRL cannot decouple
                                     ↓
                            method comparison (Experiment J): DANN shows no gain;
                              no method consistently surpasses ERM
                              (GroupDRO/V-REx/Fish significantly worse on multi-domain data)
                                     ↓
                            domain-offset quantification (Experiment I):
                              the domain fingerprint is measurable and not removable
                                     ↓
                            cross-cancer validation (Experiments L/M): the conclusion generalizes
```

---

## Experiment Overview

| ID | Name | Status | Script | Core Outputs |
|---|---|---|---|---|
| A | TCGA vs SEER (Level 4 extreme missingness) | ✅ | `scripts/02_train_dann_lihc.py` | `results/tables/table2_main_results.csv`; Fig 2 |
| B | TCGA vs external HCC (Level 2, 4 domains) | ✅ | `scripts/02_train_dann_lihc.py` | `results/tables/table2_main_results.csv`; Fig 2 |
| C | All 5 cohorts (multi-domain) | ✅ | `scripts/02_train_dann_lihc.py` | `results/tables/table2_main_results.csv`; Fig 2 |
| D | Imputation "revival" (KNN k=7, missingness → 0%) | ✅ | `scripts/05_exp_d_imputation.py` | `results/tables/table2_main_results.csv`; Fig 3 |
| E | Level-0 clean-data control (BRCA) | ✅ | `scripts/06_exp_e_clean_control.py` | `results/tables/table2_main_results.csv`; Fig 4 |
| F | 1000× stratified bootstrap | ✅ | `scripts/04_exp_f_bootstrap.py` / `04b` | Fig 4 |
| G | SHAP interpretability | ✅ | `scripts/07_exp_g_shap.py` | Fig 5 |
| H | K-M clinical stratification + log-rank | ✅ | `scripts/08_exp_h_km_curves.py` | Fig 6 |
| **I** | **Domain-offset quantification** (Wasserstein / propensity / MINE / probe) | ✅ | `scripts/10_exp_i_domain_quantify.py` | `results/tables/table5_domain_quantify.csv`; Fig 8 |
| **J** | **Comprehensive method comparison (9 methods)** | ✅ | `scripts/11_exp_j_method_comparison.py` | `results/tables/table3_method_comparison.csv`; Fig 7 |
| **K** | **Clinical evaluation metrics** (IBS / calibration / DCA) | ✅ | `scripts/12_exp_k_clinical_metrics.py` | `results/tables/table4_clinical_metrics.csv`; Fig S4/S5 |
| **L** | **COAD cross-cancer validation** | ✅ | `scripts/13_exp_lm_multicancer_validation.py --cancers COAD` | `results/tables/table6_multicancer_validation.csv` |
| **M** | **LUAD cross-cancer validation** | ✅ | `scripts/13_exp_lm_multicancer_validation.py --cancers LUAD` | `results/tables/table6_multicancer_validation.csv` |

**Revision supplements (2026-08-02, completed before submission)**

| ID | Name | Status | Script | Core Outputs |
|---|---|---|---|---|
| R1 | Non-neural baselines (XGB-Cox / RSF) | ✅ | `scripts/17_baseline_xgboost_rsf.py` | `results/tables/table7_tree_baselines.csv` |
| R2 | BH-FDR multiple-comparison correction (35 tests) | ✅ | `scripts/18_fdr_correction.py` | `results/tables/table3_method_comparison.csv` (q-values) |
| J′ | SEER downsampling sensitivity (10,000 → 309, 6 levels) | ✅ | `scripts/11_exp_j_method_comparison.py --seer_subsample` | `results/tables/table3b_seer_subsample.csv` |
| D′ | Imputation-strategy control (within-cohort median / random fill) | ✅ | `scripts/05_exp_d_imputation.py --impute_strategy all` + `scripts/23_exp_dprime_retrain.py` | **full retrain**: Δ=−0.0007, domain classifier ≈1.0 throughout (`results/tables/table10_dprime_retrain.csv`) |
| **α** | **GRL α sensitivity (fixed three levels {0.1, 0.5, 1.0} × A/B/C × 3 seeds)** | ✅ | `scripts/19_exp_alpha_sensitivity.py` | `results/tables/table8_alpha_sensitivity.csv` |
| **IT** | **True independent test-set 3-fold retrain (A/B/C × 3 folds × 3 seeds)** | ✅ | `scripts/21_exp_independent_test.py` | `results/tables/table9_independent_test.csv` |
| **D′R** | **D′ within-cohort median imputation, full retrain (3 seeds × {DANN, Baseline})** | ✅ | `scripts/23_exp_dprime_retrain.py` | `results/tables/table10_dprime_retrain.csv` (Δ=−0.0007; DANN domain classifier ≈1.0 in the late phase) |
| **IPW** | **IPW-ERM prototype (propensity-weighted ERM; A/B/C × 3 seeds × 4 modes)** | ✅ | `scripts/24_exp_ipw_erm.py` | `results/tables/table11_ipw_erm.csv` (target-cohort TCGA validation C-index +0.016 / +0.031 / +0.055; 8/9 paired positive; small magnitude) |

**Closing supplements (2026-08-02)**

| ID | Name | Status | Script | Core Outputs |
|---|---|---|---|---|
| D′RF | D′ random-fill imputation, full retrain (3 seeds × {DANN, Baseline}) | ✅ | `scripts/25_exp_dprime_randomfill_retrain.py` | `results/tables/table12_dprime_randomfill.csv` (Δ=+0.0004; domain classifier ≈1.0) |
| IPW×M | IPW × propensity-score matching (A/B/C × 3 seeds × 5 modes) | ✅ | `scripts/26_exp_ipw_matching.py` | `results/tables/table13_ipw_matching.csv` (B common-support coverage 0.27; matching trades overall performance for the target cohort; ipw_matching loses both) |
| RMST | RMST analysis (observed + model-predicted calibration) | ✅ | `scripts/27_exp_rmst.py` | `results/tables/table14_rmst.csv` (TCGA vs SEER RMST@36 ≈+8 months; DANN ≈ ERM predicted calibration) |
| SHAP-BC | SHAP interpretability on B/C (Baseline vs DANN) | ✅ | `scripts/28_exp_shap_bc.py` | `results/tables/table15_shap_bc.csv` (Stage ranks first under multi-domain; Stage +1.37 in B) |
| DeLong | Time-dependent-AUC DeLong significance testing (Parts A/B/C) | ✅ | `scripts/29_exp_delong_td_auc.py` | `results/tables/table16_delong_td_auc.csv` (DANN difference ≤0.009; direction inconsistent across settings) |
| IPW×Mx | IPW × input-level mixup (A/B/C × 3 seeds × 4 modes) | ✅ | `scripts/30_exp_ipw_mixup.py` | `results/tables/table17_ipw_mixup.csv` (no synergy; target-cohort gain largest) |
| SHAP-CI | SHAP B/C bootstrap confidence intervals (B=500) | ✅ | `scripts/31_exp_shap_ci.py` | `results/tables/table18_shap_ci.csv` (B Stage +1.36 [0.71, 2.13], significant) |
| KM-TCGA | TCGA target-cohort clinical characterization (Stage stratification + risk groups + RMST difference) | ✅ | `scripts/32_exp_km_tcga.py` | `results/tables/table19_km_tcga.csv` (Stage log-rank p=0.0078; TCGA−SEER RMST@36 +8.15 months, p<0.001) |

> **Labeling note**: In earlier reports the "G=SHAP, H=K-M" labels were kept; new experiments completed
> on 2026-07-31 are numbered contiguously after H (I–M) to avoid label conflicts. Revision supplements
> reuse the R1/R2/J′/D′/α/IT/D′R/IPW labels. The four closing supplements (D′RF / IPW×M / RMST /
> SHAP-BC) and the four further closing supplements (DeLong / IPW×Mixup / SHAP-CI / TCGA-KM) are
> summarized in the tables listed above.

---

## Key Findings

| # | Finding | Evidence | Experiment / Figure |
|---|---|---|---|
| 1 | **DANN yields zero gain** | Δ ∈ [−0.0032, +0.0007]; Baseline is superior at 9/9 time points | A/B/C; Fig 2 |
| 2 | **The domain classifier is undefeatable** | 97% within 5 epochs in the 2-domain setting; α cannot suppress it; multi-domain training oscillates without converging | A/B/C; Fig 2 |
| 3 | **Imputation cannot revive GRL** | After missingness → 0%, the domain classifier still reaches 97.7%; Δ non-significant | D; Fig 3 |
| 4 | **GRL benefit is largest, but non-significant, on clean data** | Domain classifier only 64.3%; Δ=+0.0074 (p=0.058, **not significant at α=0.05**; a theoretical bound only) | E; Fig 4 |
| 5 | **Adversarial decoupling fails; only input-level mixup is positive after FDR** | DANN shows no gain in any split (p>0.65); after BH-FDR on 35 p-values only Mixup (+0.019, q=0.012) survives; IRM (p=0.043) does not pass; GroupDRO/V-REx/Fish are significantly harmful (V-REx B: q=0.023) | J; Fig 7; `scripts/18_fdr_correction.py` |
| 6 | **The domain fingerprint is quantitatively characterizable** | Wasserstein W1 = 8.6 yr / 0.15; propensity overlap 0.18; probe AUC > 0.96 (unchanged after imputation) | I; Fig 8 |
| 7 | **Clinical evaluation** | Model outperforms the KM null (IBS 0.202 vs 0.207), ECE ≈ 0.03–0.08, DCA close | K; Fig S4/S5 |
| 8 | **Cross-cancer generalization** | No method surpasses ERM on COAD/LUAD | L/M; Fig 8 |

---

## Mechanism Verification

### Causal Chain (Experiments D/E/F/G/H)

```
Original hypothesis: missingness pattern ──(strongest fingerprint)──▶ domain classifier 97% ──▶ GRL fails
Experiment D falsifies: after imputation missingness=0%, the domain classifier still reaches 97.7%   ✗ missingness is not the root cause
Experiment E supports: on clean data the domain classifier is only 64%, GRL Δ=+0.0074               ✓ when the fingerprint is weak, GRL works
Experiment F closes: Δ in all 5 experiments p>0.05, C significantly negative                        ✓ statistically, no benefit
Experiment G corroborates: SHAP shows DANN does not destroy feature attention (Stage +26%)           ✓ failure ≠ "learned the wrong thing"
Experiment H corroborates: the two models stratify K-M identically (P ≈ 1e-25~1e-28)                 ✓ no loss of clinical capability
Experiment I quantifies: Wasserstein/propensity/MINE/probe characterize the distribution-level fingerprint ✓ measurable, not removable by imputation
Experiment J generalizes: DANN shows no gain; no method consistently beats ERM;
                          GroupDRO/V-REx/Fish are significantly worse                                ✓ a paradigm-level problem
Experiments L/M universal: the same failure on COAD/LUAD                                            ✓ holds across cancer types
```

### Key Numbers at a Glance

| Metric | Value |
|------|:----:|
| LIHC data scale | 110,640 cases × 5 cohorts |
| DANN C-index Δ (A/B/C) | +0.0007 / −0.0032 / −0.0011 |
| GRL α sensitivity (fixed three levels {0.1, 0.5, 1.0} × 3 seeds) | All three α levels **non-significant** (perm-p ≥ 0.2); best-over-α Δ = +0.0005 / +0.0016 / +0.0049 (C·α=1.0 is the largest of the 9 comparisons, p=0.20, magnitude <0.01, no clinical meaning); the domain classifier in B/C is suppressed to 0.43/0.64 as α rises, **the GRL mechanism works, but survival gains nothing** (see table8) |
| True independent test-set 3-fold retrain (A/B/C × 3 folds × 3 seeds) | Test-set Δ(DANN−Base) = −0.0002 / +0.0043 / +0.0033; sign-flip permutation p = 0.71 / 0.46 / 0.23 (all 95% CIs include 0, **non-significant**); best-val overestimates test performance (gap ≈ +0.029 on B); the negative conclusion holds under a strict, leakage-free protocol (see table9) |
| D′ within-cohort median imputation, full retrain (3 seeds × {DANN, Baseline}) | Best-val C-index DANN 0.6259 ± 0.0022 vs Baseline 0.6266 ± 0.0024, Δ = **−0.0007 (non-significant)**; **DANN domain classifier ≈1.0 in the late phase** (0.9997 / 1.0 / 1.0); leakage-free within-cohort imputation preserves the structural fingerprint (SEER Grade 99.7% missing), and GRL still gains nothing in survival (see table10) |
| IPW-ERM prototype (propensity-weighted, A/B/C × 3 seeds × 4 modes) | The mechanism direction holds: **target-cohort TCGA validation C-index +0.016 / +0.031 / +0.055** (8/9 paired positive), but at the cost of overall (SEER-dominated) performance, which drops slightly in A/C (−0.007 / −0.004) and is flat in B (+0.006); magnitudes are small, within the 46-case noise band, and cannot alone rescue cross-population prediction (structurally unobservable fields cannot be repaired by reweighting) (see table11) |
| D′ random-fill imputation, full retrain (3 seeds × {DANN, Baseline}) | Best-val C-index DANN 0.6062 ± 0.0018 vs Baseline 0.6058 ± 0.0015, Δ = **+0.0004 (non-significant)**; after random fill the probe AUC remains = 1.000 and **the DANN domain classifier is ≈1.0 in the late phase**: consistent with the within-cohort-median retrain (Δ=−0.0007): all three imputation strategies (KNN / within-cohort median / random fill) pass both the probe and the full-retrain checks, and none can erase the structural fingerprint or make GRL benefit (see table12) |
| IPW × propensity-score matching (A/B/C × 3 seeds × 5 modes) | Matching diagnostics: B's common-support coverage is only 0.27 (the external cohort and TCGA propensity scores barely overlap), matched-pair logit distance 7–12; structural unobservability leaves very thin common support; results are reported honestly in table13 (**matching trades overall performance for the target cohort; ipw_matching loses both**), reinforcing that simple distribution matching cannot have it both ways |
| RMST analysis (KM observed + model-predicted calibration) | Observed: **TCGA RMST@36 = 26.42 mo vs SEER 18.43 mo (≈+8 mo)**, AMC 34.52; the outcome distribution is one of the hardest components of domain identity; model: DANN and ERM predict nearly identical RMST@36 on TCGA (27.83 vs 27.96, both overestimating the observed 23.77 by ≈4 mo); GRL does not improve target-cohort calibration on the RMST scale either (see table14) |
| SHAP interpretability extended to B/C | In B, DANN pushes Stage mean\|SHAP\| up +1.37 (7.45 vs 6.08), target-cohort TCGA Stage 10.66 → 17.58; in C, DANN decreases features overall but Stage still ranks first; **under multi-domain settings GRL never displaces the key clinical features from the top rank** ("the failure is structural, not about learning the wrong thing" holds under multi-domain settings) (see table15) |
| DeLong time-dependent-AUC test (`scripts/29_exp_delong_td_auc.py`) | **DANN vs ERM AUC difference ≤0.009 (validation almost entirely ≤0.002), with direction inconsistent across settings** (slightly negative in A's late validation, slightly positive in C's validation and test); Part B correctly reproduces known-significant methods (Mixup C Δ=+0.023, p=0.003; V-REx B Δ=−0.098, p=0.002); A/B true test sets are all non-significant; **under every metric, GRL offers no reproducible meaningful gain** (see table16) |
| IPW × mixup (`scripts/30_exp_ipw_mixup.py`) | **No synergy from stacking**: ipw_mixup's overall best-val is worse than mixup alone in A/C (−0.024 / −0.019) and flat in B; however its target-cohort TCGA val gain (+0.060 / +0.029 / +0.094) is the largest of the four modes; both reweighting and input interpolation buy the target cohort at the cost of overall performance, so they do not simply stack (see table17) |
| SHAP B/C bootstrap CI (`scripts/31_exp_shap_ci.py`) | B's **Stage +1.36 has 95% CI [0.71, 2.13], significantly different from 0** (point estimate upgraded to a significant conclusion); Grade −2.01 significant; in C all four features are significantly negative but the ranking keeps Stage first; the SHAP conclusions now have significance backing (see table18) |
| TCGA target-cohort clinical characterization (`scripts/32_exp_km_tcga.py`) | TCGA clinical Stage stratification log-rank **p=0.0078** (the outcome distribution is a domain fingerprint); ERM/DANN risk-group K-M curves both separate significantly (p<0.001); **TCGA−SEER RMST@36 difference +8.15 mo [6.75, 9.66], p<0.001**, TCGA−MSK marginal +1.58 mo, p=0.06; the outcome-distribution shift is upgraded from a point estimate to a significance test (see table19) |
| Time-dependent AUC (12/36/60, three experiments) | Baseline superior at 9/9 time points |
| Domain-classifier peak (2-domain / after imputation / clean) | 97.3% / 97.7% / **64.3%** |
| Domain-probe AUC (LIHC / after imputation / clean) | >0.96 / 0.970 / **0.709** |
| Wasserstein W1 (Age / Stage) | 8.59 / 0.153 |
| Propensity-score overlap coefficient | 0.18 |
| Method comparison (9 methods vs ERM) | DANN all p>0.65; **after BH-FDR (35 tests) only Mixup (C: +0.019, q=0.012) is positive and V-REx (B: −0.067, q=0.023) is negative**; nominally significant others (e.g., IRM) fail (see table3) |
| Tree baselines (same 4 features, XGB-Cox / RSF) | A: 0.642/0.624, B: 0.617/0.624, C: **0.657**/0.644; on the same platform as Transformer ERM (0.633–0.639), proving that the 0.63 ceiling is an attribute of the feature set (`scripts/17_baseline_xgboost_rsf.py`) |
| IBS (ERM/DANN vs KM null) | 0.202 / 0.203 vs 0.207 |

---

## Data

### The 5 LIHC Cohorts (Experiments A/B/C/D)

| Cohort | Type | N | Event rate | Notable missingness |
|---|---|---|---|---|
| TCGA_LIHC | Genomic repository | 309 | 42.7% | Stage 20.7% |
| US_SEER | Population registry | 108,638 | 78.1% | Grade 100% |
| hcc_msk_2024 | MSK independent cohort | 1,357 | 59.1% | Age 100% |
| lihc_amc_prv | Korea AMC | 231 | 16.0% | Sex/Stage/Grade 100% |
| hcc_meric_2021 | Basel, Switzerland | 105 | 78.1% | Stage/Grade 100% |

> During training, SEER is subsampled to 10,000 cases with `RandomState(42)`. The full missingness
> matrix is in `results/tables/table1_cohort_summary.csv`.

### Other Data (Experiments E/L/M)

- E: TCGA-BRCA (source 1,098) vs METABRIC (source 2,509), features Age + Sex + Stage (Grade excluded);
  after filtering survival time/outcome, 2,995 cases enter analysis (BRCA 1,015 / METABRIC 1,980).
- L: TCGA-COAD vs CPTAC-COAD (the originally planned `coadread_dfci_2016` has no survival data, so we
  switched to CPTAC).
- M: TCGA-LUAD vs MSKCC-2020.

### Data Availability

Raw and processed data are **not** shipped in this repository, because TCGA, SEER, and cBioPortal data
are governed by their respective license terms. Instead, data are obtained by running the download
scripts, then regenerating the processed files:

- `scripts/download_tcga.py`: downloads TCGA GDC data (LIHC, BRCA, COAD, LUAD).
- `scripts/download_additional_data.py`: downloads SEER, MSK (`hcc_msk_2024`), AMC (`lihc_amc_prv`),
  METABRIC, CPTAC-COAD, MSKCC-2020, and other cBioPortal/registry cohorts.
- SEER requires a signed data-use agreement; `scripts/process_seer.py` documents the expected SEER
  export format and converts it to the unified schema.

Large, regenerable artifacts, such as model weights (`*.pth`), intermediate features (`*.npy`), and raw
data, are excluded from the repository via `.gitignore` and are produced by running the pipeline below.
`scripts/paths.py` centralizes data-root resolution (env var `TRANSDANN_DATA_ROOT` override; falls
back to the local `results/` directory).

| Component | Distributed | Notes |
|---|---|---|
| Code (`scripts/`) | Yes | Full pipeline and all experiment scripts, MIT license |
| Paper figure set (`results/figures/paper/`) | Yes | Figure 1–8 + S1–S5 (PNG/PDF) + Figure legends + supplementary tables |
| Paper tables (`results/tables/`) | Yes | table1–table19 + table3b CSV |
| Raw/processed data (`data_raw/`, `data_processed/`) | No | License-restricted; obtain via `scripts/download_*.py` |
| Model weights / intermediate features (`*.pth`, `*.npy`) | No | Excluded via `.gitignore`; regenerate by running the experiments |

---

## Methods

### Unified Schema

`Age` (continuous) + `Sex` / `Stage` / `Grade` (categorical) + `Survival_Months` / `Vital_Status`.

### Model: TransDANNSurvV3

```
Input features → continuous MLP embedding / categorical Embedding(padding_idx=missing)
              → [CLS] + positional encoding → Transformer (4 layers × 8 heads, d_model=128)
              → CLS representation → survival head (DeepHit, 32 bins)
                        └→ GradientReversal(α) → domain-classification head
```

- Total parameters ≈ 818K; DANN and Baseline differ only by the GRL + domain loss.
- Training: AdamW 5e-4, CosineAnnealing, early stopping patience = 40, loss averaged per cohort.
- Engine scope: the main experiments A/B/C (`02`) use a **single seed (42)**; the method comparison J
  and the multi-cancer experiments L/M (`11`/`13`) use **3 seeds (42/43/44)** averaged. The two
  numberings differ by <0.002, with identical conclusions.
- All 9 compared methods share this backbone (implementations in `scripts/method_utils.py`).

---

## Results: Figures and Tables

### Publication Figure Set (Figure 1–8 + S1–S5, `results/figures/paper/`)

A dense, publication-grade figure set reorganized to follow the paper narrative (generated by
`scripts/16_generate_paper_figures.py`, PNG 300 dpi + PDF vector, with ready-to-submit figure
legends):

| Section | Figure | File | Panels | Narrative function |
|---|---|---|---|---|
| 3.1 Data characterization | **Fig 1** | `Figure_01_domain_fingerprint` | 2×2 | Domain fingerprint: missingness is the surface, distributions are the essence |
| 3.2 Main results | **Fig 2** | `Figure_02_main_results` | 2×2 | C-index / time-AUC / pooled per-cohort / domain classifier: zero-to-negative gain |
| 3.3 Mechanism I | **Fig 3** | `Figure_03_imputation` | 2×2 | Imputation does not remove the domain fingerprint (falsifies "missingness is the root cause") |
| 3.4 Mechanism II + statistics | **Fig 4** | `Figure_04_clean_data_bootstrap` | 2×3 | Clean data is GRL's best-case scenario; bootstrap still non-significant |
| 3.5 Interpretability | **Fig 5** | `Figure_05_shap` | 2×2 | SHAP: GRL does not destroy feature attention |
| 3.6 Clinical validation | **Fig 6** | `Figure_06_km_clinical` | 2×2 | Equivalent K-M stratification |
| 3.7 Method comparison | **Fig 7** | `Figure_07_methods` | 2×3 | Systematic failure of 9 methods |
| 3.8 Quantification + universality | **Fig 8** | `Figure_08_quantify_multicancer` | 2×3 | Wasserstein / propensity / MINE / multi-cancer |
| Supp | **S1–S5** | `supplementary/Figure_S1..S5_*` | n/a | t-SNE / training dynamics / GRL schedule / calibration / DCA |

Supporting documents and supplementary data tables (all under `results/figures/paper/`):

- `Figure_Legends.md`: complete legends for all 13 figures (ready to copy into a submission).
- `README_figures.md`: figure-to-section correspondence table, data-source list, and corrections
  relative to earlier drafts.
- `supplementary/Supp_Table_S1..S7_*.csv`: supplementary data tables: **S1 the full inventory of all
  36 data sources** (sample size / event rate / cancer type / which experiments use each), **S2 cohort
  summary** (baseline table, with missingness rates), S3 baseline results (DANN vs Baseline), S4 the
  9-method comparison, S5 clinical metrics, S6 domain quantification, S7 multi-cancer validation.

> Reproduced in one command with `scripts/16_generate_paper_figures.py all`. Numbers inside the
> figures are read from `results/tables/*.csv`; a few axis labels and peaks are hard-coded
> consistently with those tables (e.g., domain-classifier peaks 97.3%/97.7%). All numbers in the paper
> tables are traceable to the corresponding experiment scripts.

### Paper Tables (`results/tables/`)

| Table | Contents |
|---|---|
| `table1_cohort_summary.csv` | Cohort summary (missingness/event rates) |
| `table2_main_results.csv` | A–H main results (main engine, single seed) |
| `table3_method_comparison.csv` | 9-method comparison (C-index/IBS/Δ/p + BH-FDR q-values) |
| `table3b_seer_subsample.csv` | J′ SEER downsampling sensitivity (6 levels) |
| `table4_clinical_metrics.csv` | IBS/ECE |
| `table5_domain_quantify.csv` | MINE/probe |
| `table6_multicancer_validation.csv` | COAD/LUAD |
| `table7_tree_baselines.csv` | R1 XGBoost-Cox / RSF baselines |
| `table8_alpha_sensitivity.csv` | GRL α sensitivity (fixed three levels {0.1, 0.5, 1.0} × A/B/C × 3 seeds) |
| `table9_independent_test.csv` | True independent test-set 3-fold retrain (DANN vs Baseline test C-index + sign-flip permutation p) |
| `table10_dprime_retrain.csv` | D′ within-cohort median imputation, full retrain (best-val C-index + late-phase domain-classifier accuracy) |
| `table11_ipw_erm.csv` | IPW-ERM prototype (erm/dann/erm_pooled/ipw_erm C-index + per-cohort validation) |
| `table12_dprime_randomfill.csv` | D′ random-fill imputation, full retrain (Δ=+0.0004, domain classifier ≈1.0) |
| `table13_ipw_matching.csv` | IPW × propensity-score matching (A/B/C × 3 seeds × 5 modes + matching diagnostics) |
| `table14_rmst.csv` | RMST analysis (observed KM RMST + model-predicted vs observed calibration) |
| `table15_shap_bc.csv` | SHAP B/C feature importance (Baseline vs DANN, per cohort) |
| `table16_delong_td_auc.csv` | DeLong time-dependent-AUC tests (DANN/each method vs ERM ΔAUC + DeLong p + bootstrap p + both AUC scopes) |
| `table17_ipw_mixup.csv` | IPW × mixup (erm/mixup/ipw_erm/ipw_mixup C-index + per-cohort validation) |
| `table18_shap_ci.csv` | SHAP mean\|SHAP\| bootstrap 95% CI (Baseline/DANN/Δ, B/C) |
| `table19_km_tcga.csv` | TCGA K-M (clinical Stage stratification + RMST@36 difference bootstrap test) |

---

## Project Structure

```
TransDANN_Liver_Cancer/
├── README.md                    # Project overview (this file)
├── LICENSE                      # MIT license
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes raw data, weights, internal docs, logs, legacy archives
├── run_lihc_pipeline.sh         # Full pipeline launcher
├── start_training.sh            # Training launcher
│
├── scripts/                     # All Python code (numbered in experiment/output order)
│   ├── paths.py                 # Data-path interface (data-root resolution)
│   ├── download_tcga.py         # TCGA GDC data download
│   ├── download_additional_data.py  # SEER / cBioPortal / additional cohort download
│   ├── process_seer.py          # SEER export parsing → unified schema
│   ├── 01_prepare_lihc_data.py  # Data preparation → lihc_all_cohorts.csv
│   ├── 02_train_dann_lihc.py    # Experiments A/B/C (DANN vs Baseline training engine)
│   ├── 03_visualize_lihc.py     # Main-figure visualization
│   ├── 04/04b_exp_f_bootstrap.py    # Experiment F: 1000× stratified bootstrap
│   ├── 05_exp_d_imputation.py   # Experiment D: KNN imputation + retrain
│   ├── 06_exp_e_clean_control.py# Experiment E: clean-data control
│   ├── 07_exp_g_shap.py         # Experiment G: SHAP interpretability
│   ├── 08_exp_h_km_curves.py    # Experiment H: K-M stratification
│   ├── 09_make_figure_package.py# Legacy figure pooling (historical source pool)
│   ├── 10_exp_i_domain_quantify.py    # Experiment I: domain-offset quantification
│   ├── 11_exp_j_method_comparison.py  # Experiment J: method comparison (9 methods)
│   ├── 12_exp_k_clinical_metrics.py   # Experiment K: clinical metrics
│   ├── 13_exp_lm_multicancer_validation.py  # Experiments L/M: multi-cancer validation
│   ├── 15_make_summary_tables.py      # Generates tables 1/2
│   ├── 16_generate_paper_figures.py   # Publication figure set (Figure 1–8 + S1–S5 + legends + supp tables)
│   ├── 17_baseline_xgboost_rsf.py     # R1: non-neural baselines (XGB-Cox/RSF)
│   ├── 18_fdr_correction.py           # R2: BH-FDR multiple-comparison correction
│   ├── 19_exp_alpha_sensitivity.py    # GRL α sensitivity (fixed three levels {0.1, 0.5, 1.0})
│   ├── 20_alpha_sweep_table.py        # α sensitivity summary → table8
│   ├── 21_exp_independent_test.py     # True independent test-set 3-fold retrain (A/B/C × 3 folds × 3 seeds)
│   ├── 22_independent_test_table.py   # Independent-test summary → table9 (sign-flip permutation test)
│   ├── 23_exp_dprime_retrain.py       # D′ within-cohort median imputation, full retrain → table10
│   ├── 24_exp_ipw_erm.py              # IPW-ERM prototype (inverse-propensity-weighted ERM) → table11
│   ├── 25_exp_dprime_randomfill_retrain.py  # D′ random-fill imputation, full retrain → table12
│   ├── 26_exp_ipw_matching.py         # IPW × propensity-score matching → table13
│   ├── 27_exp_rmst.py                 # RMST analysis (observed + model calibration) → table14
│   ├── 28_exp_shap_bc.py              # SHAP interpretability extended to B/C → table15
│   ├── 29_exp_delong_td_auc.py        # DeLong time-dependent-AUC significance → table16
│   ├── 30_exp_ipw_mixup.py            # IPW × mixup → table17
│   ├── 31_exp_shap_ci.py              # SHAP mean|SHAP| bootstrap CI → table18
│   ├── 32_exp_km_tcga.py              # TCGA K-M + RMST@36 difference → table19
│   ├── method_utils.py          # Loss/penalty implementations for the 9 methods
│   ├── transdann_utils.py       # Shared model/utilities (TransDANNSurvV3)
│   ├── lihc_recon.py            # LIHC reconstruction helpers
│   └── fast_cindex.py           # Fast concordance-index implementation
│
└── results/                     # Paper deliverables
    ├── tables/                  # table1–table19 + table3b CSV
    └── figures/paper/           # Publication figure set: Figure 1–8 + S1–S5 + legends + supp tables
```

---

## Installation

```bash
# Linux / GPU environment (validated: PyTorch 2.3.1+cu121, CUDA, Tesla V100 32GB)
cd /path/to/TransDANN_Liver_Cancer
pip install -r requirements.txt
pip install scikit-survival>=0.22 lifelines>=0.27 shap   # additional dependencies
```

Dependencies: `torch>=2.0, numpy, pandas, scikit-learn, scikit-survival, lifelines, matplotlib,
seaborn, openpyxl, pyarrow, shap`.

---

## Reproduction Commands

```bash
# 0) Download raw data (TCGA / SEER / cBioPortal; license terms apply; see the Data section)
python3 scripts/download_tcga.py
python3 scripts/download_additional_data.py
python3 scripts/process_seer.py                         # if SEER is available

# Data preparation
python3 scripts/01_prepare_lihc_data.py                # → data_processed/lihc_all_cohorts.csv

# Main experiments A/B/C
python3 scripts/02_train_dann_lihc.py --experiment ALL --epochs 200 --surv_type deephit \
    --d_model 128 --n_layers 4 --dropout 0.15 --lr 5e-4 --domain_weight 0.3
python3 scripts/03_visualize_lihc.py                   # main figures (legacy)

# Supplement experiments D–H
python3 scripts/05_exp_d_imputation.py --epochs 200    # D
python3 scripts/06_exp_e_clean_control.py --epochs 200 # E
python3 scripts/04_exp_f_bootstrap.py                  # F (full-set scope)
python3 scripts/04b_exp_f_val_bootstrap.py             # F (validation-set scope)
python3 scripts/07_exp_g_shap.py                       # G
python3 scripts/08_exp_h_km_curves.py                  # H

# Experiments I–M
python3 scripts/10_exp_i_domain_quantify.py            # I: domain-offset quantification
python3 scripts/11_exp_j_method_comparison.py --experiments A B C --seeds 42 43 44  # J: method comparison
python3 scripts/12_exp_k_clinical_metrics.py           # K: clinical metrics (run J's A first)
python3 scripts/13_exp_lm_multicancer_validation.py --cancers COAD LUAD --seeds 42 43 44  # L/M

# Revision supplements
python3 scripts/17_baseline_xgboost_rsf.py             # R1: non-neural baselines
python3 scripts/18_fdr_correction.py                   # R2: BH-FDR correction
python3 scripts/19_exp_alpha_sensitivity.py            # GRL α sensitivity (A/B/C × {0.1, 0.5, 1.0} × 3 seeds)
python3 scripts/21_exp_independent_test.py             # True independent test-set 3-fold retrain (A/B/C × 3 folds × 3 seeds)
python3 scripts/23_exp_dprime_retrain.py               # D′ within-cohort median imputation, full retrain (3 seeds × {DANN, Baseline})
python3 scripts/24_exp_ipw_erm.py                      # IPW-ERM prototype (A/B/C × 3 seeds × 4 modes)

# Closing supplements
python3 scripts/25_exp_dprime_randomfill_retrain.py    # D′ random-fill imputation, full retrain (3 seeds × {DANN, Baseline})
python3 scripts/26_exp_ipw_matching.py                 # IPW × propensity-score matching (A/B/C × 3 seeds × 5 modes)
python3 scripts/27_exp_rmst.py                         # RMST analysis (observed KM + model calibration)
python3 scripts/28_exp_shap_bc.py                      # SHAP interpretability extended to B/C
python3 scripts/29_exp_delong_td_auc.py                # DeLong time-dependent-AUC significance (Part C needs GPU)
python3 scripts/30_exp_ipw_mixup.py                    # IPW × mixup (A/B/C × 3 seeds × 4 modes)
python3 scripts/31_exp_shap_ci.py                      # SHAP mean|SHAP| bootstrap CI (run 28 first)
python3 scripts/32_exp_km_tcga.py                      # TCGA K-M + RMST@36 difference

# Tables and figure set
python3 scripts/15_make_summary_tables.py              # table1/2
python3 scripts/20_alpha_sweep_table.py                # α sensitivity summary → table8
python3 scripts/22_independent_test_table.py           # Independent-test summary → table9
python3 scripts/16_generate_paper_figures.py all       # Publication figure set: Figure 1–8 + S1–S5 + legends + supp tables

# View results
ls results/tables/                                     # Paper tables
ls results/figures/paper/                              # Publication figure set
```

---

## Citation

This work is currently under review as a negative-result methods paper. If you use this repository or
find the findings useful, please cite:

```bibtex
@misc{transdann_failure_2026,
  author = {Geng, Qiushuo},
  title  = {When Domain Adversarial Training Fails: Population Distributions as the Fundamental
            Domain Fingerprint in Cancer Survival Prediction},
  year   = {2026},
  note   = {110,640 LIHC patients, 5 HCC cohorts, 3 cancer types},
  doi    = {10.5281/zenodo.21967922},
}
```

A DOI is minted by Zenodo for each tagged GitHub release. Current release:
[10.5281/zenodo.21967922](https://doi.org/10.5281/zenodo.21967922). A preprint may be posted to
arXiv (cs.LG / stat.ML / q-bio.QM).

---

## License

This repository is released under the **MIT License** (see `LICENSE`). Note that the raw data
(TCGA / SEER / cBioPortal) are governed by their respective data-use agreements and are not
redistributed here.

---

## References

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

> In science, knowing what *does not* work is sometimes as valuable as knowing what *does*. This
> systematic study of 110,640 liver-cancer patients, two additional cancer types, five cohorts, and
> nine domain-adaptation/domain-generalization methods shows that **differences in population
> distributions are the fundamental domain fingerprint of medical data, irreducible by imputation and
> inseparable by adversarial training, which is why domain-adversarial methods systematically fail in
> cross-population survival prediction**.
>
> **Core recommendation**: Future medical domain-adaptation research should solve the
> **population-distribution matching / standardization** problem before innovating on algorithms alone.
> Transfer learning is only meaningful once domain identity and the survival signal stop being deeply
> entangled.
