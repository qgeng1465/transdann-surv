#!/usr/bin/env python3
"""
09_make_figure_package.py; Organise every figure into a paper-ready package
============================================================================
  - Collects all PNGs from results/figures/lihc/ (14 main) and
    results/figures/extended/ (30 supplementary).
  - Copies them into results/figures/figure_package/ renamed Figure_01.png …
    Figure_44.png (zero-padded, logical paper order).
  - Writes Figure_Legends.xlsx with one row per figure: number, new file,
    original file, experiment, content, an English figure-legend draft and a
    Chinese caption draft.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LHC = BASE_DIR / "results" / "figures" / "lihc"
EXT = BASE_DIR / "results" / "figures" / "extended"
PKG = BASE_DIR / "results" / "figures" / "figure_package"
PKG.mkdir(parents=True, exist_ok=True)

# (original file, experiment, title, content, legend EN, caption ZH)
FIGURES = [
    # ---------------- main results (A/B/C) ----------------
    ("01_missingness_heatmap.png", "A/B/C",
     "Missingness heatmap (5 cohorts × 4 features)",
     "Fraction of missing values per cohort and feature; Grade is 100% missing in SEER/external cohorts.",
     "Missingness patterns across five liver-cancer cohorts (rows) and four clinical features (columns); "
     "colour encodes the fraction of missing values. Grade is 100% missing in the SEER and external-HCC "
     "cohorts while TCGA-LIHC is nearly complete. Missingness alone almost perfectly separates the cohorts.",
     "Missingness heatmap (5 cohorts × 4 clinical features). Grade is 100% missing in the SEER and "
     "external-HCC cohorts while TCGA-LIHC is nearly complete; the missingness pattern alone almost "
     "perfectly separates the cohorts; the most direct evidence of the domain fingerprint."),
    ("02_tsne_A_tcga_vs_seer.png", "A",
     "t-SNE of CLS embeddings (TCGA vs SEER)",
     "Baseline vs DANN latent space on Exp A test set.",
     "t-SNE projection of the transformer CLS embeddings on the experiment-A test set (TCGA-LIHC vs US-SEER), "
     "for the Baseline (left) and DANN (right) models; points are coloured by cohort. The adversarial objective "
     "visibly pulls the two cohorts together in the embedding space, yet this does not translate into any "
     "survival-prediction benefit (Δ C-index = +0.0007).",
     "t-SNE of the CLS latent space on the experiment-A test set (left: Baseline, right: DANN, coloured by cohort). "
     "The adversarial objective does pull the two cohorts together in the latent space, yet this brings no "
     "survival-prediction benefit (Δ C-index = +0.0007); mixing representations is not the same as disentangling them."),
    ("02_tsne_B_tcga_vs_external.png", "B",
     "t-SNE of CLS embeddings (TCGA vs 3 external cohorts)",
     "Baseline vs DANN latent space on Exp B test set.",
     "As in Figure 2, for experiment B (TCGA-LIHC vs three external HCC cohorts, four domains).",
     "As in Figure 2, experiment B: TCGA-LIHC vs three external HCC cohorts (4 domains)."),
    ("02_tsne_C_all_cohorts.png", "C",
     "t-SNE of CLS embeddings (all 5 cohorts)",
     "Baseline vs DANN latent space on Exp C test set.",
     "As in Figure 2, for experiment C (all five LIHC cohorts combined, five domains).",
     "As in Figure 2, experiment C: all five LIHC cohorts (5 domains)."),
    ("03_cindex_comparison.png", "A/B/C",
     "Validation C-index: DANN vs Baseline",
     "Overall concordance index across the three main experiments.",
     "Concordance index on the held-out validation set for the DANN and Baseline models in experiments A, B and C. "
     "Differences are within ±0.003; domain-adversarial training provides no survival-prediction benefit.",
     "Validation-set C-index for the three main experiments (DANN vs Baseline). Differences are within "
     "±0.003; domain-adversarial training yields no survival-prediction benefit."),
    ("03_per_cohort_cindex_A_tcga_vs_seer.png", "A",
     "Per-cohort C-index (TCGA vs SEER)",
     "C-index per cohort for Exp A.",
     "Per-cohort concordance index for experiment A. DANN lowers C-index on the TCGA reference cohort (−0.034) "
     "and is neutral on SEER; the adversarial re-weighting hurts the cohort it is supposed to transfer from.",
     "Per-cohort C-index for experiment A. DANN lowers the TCGA reference cohort by 0.034 while SEER stays "
     "flat; the adversarial re-weighting hurts the reference domain it is supposed to transfer from."),
    ("03_per_cohort_cindex_B_tcga_vs_external.png", "B",
     "Per-cohort C-index (TCGA vs external)",
     "C-index per cohort for Exp B.",
     "Per-cohort concordance index for experiment B. DANN hurts the TCGA reference cohort (−0.014) and gains only "
     "in the smallest cohort (AMC, +0.046), a net zero effect.",
     "Per-cohort C-index for experiment B. DANN hurts TCGA (−0.014) and gains only in the smallest cohort "
     "(AMC, +0.046), a net-zero effect."),
    ("03_per_cohort_cindex_C_all_cohorts.png", "C",
     "Per-cohort C-index (all cohorts)",
     "C-index per cohort for Exp C.",
     "Per-cohort concordance index for experiment C (five cohorts); the net effect across cohorts is zero.",
     "Per-cohort C-index for experiment C (5 cohorts); the net effect across cohorts is zero."),
    ("04_domain_accuracy_comparison.png", "A/B/C",
     "Domain-classifier accuracy during adversarial training",
     "Peak domain accuracy across experiments.",
     "Peak domain-classifier accuracy during GRL training in experiments A/B/C. The classifier reaches 97% in the "
     "two-domain setting and oscillates without converging in the multi-domain setting; the gradient-reversal "
     "signal cannot suppress domain identity.",
     "Peak domain-classifier accuracy during adversarial training across the three experiments. It reaches "
     "97% in the two-domain setting and oscillates without converging in the multi-domain setting; the GRL "
     "signal cannot suppress domain identity."),
    ("04_training_dynamics_A_tcga_vs_seer.png", "A",
     "Training dynamics (TCGA vs SEER)",
     "Surv/domain loss, ValC, domain acc, α over epochs.",
     "Training dynamics of the DANN model in experiment A (survival loss, domain loss, validation C-index, domain "
     "accuracy and the ramp coefficient α). Domain accuracy saturates near 97% within five epochs and stays there "
     "while α ramps to 1.",
     "Training dynamics of the DANN model in experiment A (survival loss / domain loss / validation C-index / "
     "domain accuracy / α). Domain accuracy reaches 97% within the first 5 epochs and stays there while α "
     "ramps to 1; it cannot be suppressed."),
    ("04_training_dynamics_B_tcga_vs_external.png", "B",
     "Training dynamics (TCGA vs external)",
     "Domain loss oscillates; survival flat.",
     "As in Figure 10 for experiment B. The domain loss oscillates (≈0.3↔3.6) and never converges while survival "
     "loss and C-index remain flat; the adversarial and survival objectives are effectively decoupled.",
     "As in Figure 10, experiment B. The domain loss oscillates between 0.3 and 3.6 without converging, while "
     "the survival loss and C-index remain flat; the adversarial and survival tracks are effectively decoupled."),
    ("04_training_dynamics_C_all_cohorts.png", "C",
     "Training dynamics (all cohorts)",
     "Multi-domain training curves.",
     "As in Figure 10 for experiment C (five domains).",
     "As in Figure 10, experiment C (5 domains)."),
    ("05_training_schedule.png", "A/B/C",
     "GRL α and domain-weight schedules",
     "Ramp curves for α and domain_weight.",
     "Scheduling of the gradient-reversal coefficient α = 2/(1+exp(−10p))−1 and the domain-loss weight 0.3·p over "
     "the training epoch; the standard gentle-ramp recipe recommended in the DANN literature.",
     "Gentle-ramp scheduling of the gradient-reversal coefficient α and the domain-loss weight 0.3p; the "
     "setting widely regarded as most stable in the DANN literature, which still fails to revive GRL."),
    ("06_time_auc_comparison.png", "A/B/C",
     "Time-dependent AUC (12/36/60 months)",
     "AUC at 12, 36, 60 months, DANN vs Baseline.",
     "Time-dependent AUC at 12, 36 and 60 months for DANN vs Baseline in experiments A/B/C. Baseline equals or "
     "exceeds DANN at all nine evaluation points.",
     "Time-dependent AUC at 12/36/60 months for experiments A/B/C. Baseline ≥ DANN at all 9/9 evaluation points."),
    # ---------------- Experiment D ----------------
    ("D1_missingness_before_after.png", "D",
     "Missingness before vs after KNN imputation",
     "All four features drop to 0% missing.",
     "Missingness rates per cohort and feature before and after KNN imputation (k = 7, pooled across cohorts); "
     "every rate is reduced to 0%, physically erasing the missingness fingerprint.",
     "Missingness rates per cohort before and after KNN imputation (k=7): every rate drops to 0%, physically "
     "erasing the missingness fingerprint."),
    ("D2_domain_separability_probe.png", "D",
     "Domain-separability probe (original vs imputed)",
     "Gradient-boosting AUC/balanced-acc unchanged after imputation.",
     "Domain-separability probe: gradient-boosting AUC and balanced accuracy for predicting the cohort from "
     "feature values, on the original (A) and fully imputed (D) data. Removing all missingness leaves separability "
     "essentially unchanged (AUC 0.9544 → 0.9603); missingness is a visible symptom, not the cause of domain identity.",
     "Domain-separability probe (gradient-boosting AUC/balanced accuracy predicting the cohort from feature "
     "values): essentially unchanged after full imputation (0.9544→0.9603); missingness is a symptom, not the cause."),
    ("D3_domain_acc_trajectory.png", "D",
     "Domain-classifier accuracy trajectory (A vs D)",
     "Peak stays at 97.7% even with zero missingness.",
     "Domain-classifier accuracy trajectory in experiment A vs experiment D (imputed). Peak accuracy remains "
     "97.7% even with 0% missingness; the adversarial head still reads the cohort off the covariate-value "
     "distributions.",
     "Domain-classifier accuracy trajectories for experiment A vs the imputed variant (D): even with zero "
     "missingness the peak stays at 97.7%; the adversarial head still reads cohort identity off the "
     "covariate-value distributions."),
    ("D4_cindex_original_vs_imputed.png", "D",
     "C-index on original vs imputed data",
     "Δ shifts +0.0007 → +0.0027, never significant.",
     "Validation C-index of DANN vs Baseline on the original (A) and fully imputed (D) data. Imputation shifts Δ "
     "from +0.0007 to +0.0027 but the bootstrap confidence interval still contains zero; imputation alone cannot "
     "resurrect the adversarial benefit.",
     "Validation C-index of DANN vs Baseline on the original and fully imputed data. Δ shifts from +0.0007 to "
     "+0.0027 but the bootstrap interval still contains 0; imputation alone cannot resurrect GRL."),
    # ---------------- Experiment E ----------------
    ("E0_survival_dist.png", "E",
     "Survival distributions (TCGA-BRCA vs METABRIC)",
     "Clean-control cohorts, distinct event rates.",
     "Survival-time distributions of the two clean-control cohorts TCGA-BRCA and METABRIC, whose event rates "
     "(13.8% vs 45.6%) differ; the residual domain signal to be transferred.",
     "Survival-time distributions of the clean control cohorts TCGA-BRCA and METABRIC (event rates 13.8% vs "
     "45.6%); the residual domain signal to be transferred."),
    ("E1_domain_acc_clean.png", "E",
     "Domain-classifier accuracy on clean data",
     "Peak only 64.3% vs 97%+ on LIHC.",
     "Domain-classifier accuracy on the clean control data (TCGA-BRCA vs METABRIC, Age/Sex/Stage complete): the "
     "peak is only 64.3% against 97%+ on the LIHC data; the domain fingerprint is far weaker when clinical "
     "fields are complete.",
     "Domain-classifier accuracy on the clean data (TCGA-BRCA vs METABRIC, near-complete clinical fields): the "
     "peak is only 64.3%, versus 97%+ on LIHC; the domain fingerprint weakens substantially when clinical "
     "fields are complete."),
    ("E2_per_cohort_cindex.png", "E",
     "Per-cohort C-index (clean control)",
     "DANN wins METABRIC, loses TCGA-BRCA.",
     "Per-cohort concordance index in experiment E. DANN improves METABRIC (+0.003) and slightly worsens "
     "TCGA-BRCA (−0.005), for a net Δ of +0.0074; the largest (still non-significant, p = 0.058) adversarial "
     "benefit observed anywhere in this study.",
     "Per-cohort C-index in experiment E. DANN wins on METABRIC (+0.003) and is slightly negative on "
     "TCGA-BRCA (−0.005), for a net Δ of +0.0074; the largest (still non-significant, p=0.058) adversarial "
     "benefit in the study."),
    # ---------------- Experiment F ----------------
    ("F_bootstrap_CIs.png", "F",
     "Bootstrap 95% CIs for Δ across all five experiments",
     "All intervals include zero.",
     "Forest plot of the 95% bootstrap confidence intervals for Δ = C-index(DANN) − C-index(Baseline) across "
     "experiments A–E (validation-set bootstrap, B = 1000, stratified by cohort). All intervals include zero; the "
     "multi-cohort experiment C is significantly negative on the full-set bootstrap (p = 0.03).",
     "Forest plot of the 95% bootstrap confidence intervals for Δ across the five experiments (validation-set "
     "measure, B=1000, stratified by cohort). All intervals include zero; under the full-set measure, "
     "experiment C is significantly negative (p=0.03)."),
    ("F_bootstrap_hist_A.png", "F", "Bootstrap Δ distribution; Exp A", "p_two = 0.898.",
     "Bootstrap distribution of Δ for experiment A; the two-sided p-value is 0.898 (no significant benefit).",
     "Bootstrap distribution of Δ for experiment A; two-sided p=0.898 (no significant benefit)."),
    ("F_bootstrap_hist_B.png", "F", "Bootstrap Δ distribution; Exp B", "p_two = 0.808.",
     "Bootstrap distribution of Δ for experiment B; p = 0.808.",
     "Bootstrap distribution of Δ for experiment B; p=0.808."),
    ("F_bootstrap_hist_C.png", "F", "Bootstrap Δ distribution; Exp C", "p_two = 0.744 (val).",
     "Bootstrap distribution of Δ for experiment C; p = 0.744 on the validation set.",
     "Bootstrap distribution of Δ for experiment C (validation-set p=0.744)."),
    ("F_bootstrap_hist_D.png", "F", "Bootstrap Δ distribution; Exp D (imputed)", "p_two = 0.320.",
     "Bootstrap distribution of Δ for experiment D (fully imputed); p = 0.320.",
     "Bootstrap distribution of Δ for experiment D (fully imputed); p=0.320."),
    ("F_bootstrap_hist_E.png", "F", "Bootstrap Δ distribution; Exp E (clean)", "p_two = 0.058.",
     "Bootstrap distribution of Δ for experiment E (clean control); p = 0.058; borderline positive, the only "
     "experiment where DANN approaches significance.",
     "Bootstrap distribution of Δ for experiment E (clean control); p=0.058; the only experiment approaching "
     "significance, in the positive direction."),
    # ---------------- Experiment G (SHAP) ----------------
    ("G1_shap_beeswarm_baseline.png", "G",
     "SHAP beeswarm; Baseline (Exp A test set)",
     "Stage is the strongest contributor.",
     "SHAP beeswarm of the Baseline model on the experiment-A test set (1,546 patients). Each dot is a patient; "
     "the x-position is the SHAP contribution to the risk score (negative expected lifetime; positive = higher "
     "risk) and the colour is the feature value. Stage is the strongest contributor (mean |SHAP| = 8.31), "
     "followed by Age (4.53), Grade (3.05) and Sex (1.45).",
     "SHAP beeswarm of the Baseline model on the experiment-A test set (1,546 patients). The x-axis is the SHAP "
     "contribution to risk (negative expected lifetime), coloured by feature value. "
     "Stage contributes most (mean|SHAP|=8.31), followed by Age (4.53), Grade (3.05) and Sex (1.45)."),
    ("G2_shap_beeswarm_dann.png", "G",
     "SHAP beeswarm; DANN (Exp A test set)",
     "Stage importance rises to 10.53.",
     "SHAP beeswarm of the DANN model on the same test set. Stage remains the strongest contributor and its mean "
     "|SHAP| rises to 10.53 (+26.6% vs Baseline); the adversarial head does not destroy the model's attention to "
     "the key clinical feature.",
     "SHAP beeswarm of the DANN model on the same test set. Stage still ranks first and its importance rises to "
     "10.53 (+26.6% vs Baseline); the adversarial head does not destroy the model's attention to the key "
     "clinical feature."),
    ("G3_shap_importance_compare.png", "G",
     "Feature importance: Baseline vs DANN",
     "DANN re-weights Age→Stage/Grade, destroys nothing.",
     "Grouped bar chart of mean |SHAP| per feature for the Baseline and DANN models. DANN re-weights importance "
     "away from Age (−17.7%) toward Stage (+26.6%) and Grade (+16.0%) rather than destroying any feature; the "
     "interpretability structure of the model is preserved.",
     "Grouped bar chart of per-feature mean|SHAP| for Baseline vs DANN. DANN re-balances importance from Age "
     "(−17.7%) to Stage (+26.6%) and Grade (+16.0%) rather than destroying any feature; the model's "
     "interpretability structure is preserved."),
    # ---------------- Experiment H (K-M) ----------------
    ("H1_km_baseline_vs_dann.png", "H",
     "Kaplan–Meier curves stratified by predicted risk (SEER test)",
     "Both models separate high/low risk strongly and equivalently.",
     "Kaplan–Meier survival curves for the SEER test set (1,500 patients, 1,162 events) stratified by the median "
     "predicted risk into High- and Low-Risk groups, for Baseline (left) and DANN (right). Both models separate "
     "the groups strongly and equivalently (log-rank P = 3.97e-25 vs 1.83e-28; median survival 8 vs 25 months); "
     "the adversarial variant neither enhances nor harms clinical risk stratification.",
     "Kaplan–Meier curves for the SEER test set (1,500 patients, 1,162 events) split into high/low-risk groups "
     "by the median predicted risk (left: Baseline, right: DANN). "
     "Both models stratify significantly and equivalently (log-rank P=3.97e-25 vs 1.83e-28; median survival 8 vs "
     "25 months); the adversarial variant neither enhances nor harms clinical risk stratification."),
    # ---------------- Experiment I (domain quantification) ----------------
    ("I1_wasserstein_matrix.png", "I",
     "Wasserstein-1 distance matrices (Age / Stage)",
     "Quantified distribution-level domain shift.",
     "Pairwise Wasserstein-1 distances between the five LIHC cohorts for Age (continuous) and the Stage "
     "distribution. Mean pairwise distances of 8.6 years (Age) and 0.15 (Stage EMD) confirm that the cohorts "
     "differ at the level of the covariate-value distributions themselves.",
     "Pairwise Wasserstein-1 distances between the five LIHC cohorts for Age (continuous) and the Stage "
     "distribution. Mean distances of 8.6 years (Age) and 0.15 (Stage EMD); the cohorts genuinely differ at "
     "the level of the covariate-value distributions."),
    ("I2_propensity_overlap.png", "I",
     "Propensity-score overlap (5-class logistic regression)",
     "Overlap coefficient ≈ 0.18; cohorts barely overlap.",
     "Left: pairwise overlap coefficients of the per-domain propensity-score densities; mean overlap is 0.18, "
     "meaning the covariate distributions almost completely separate the cohorts. Right: propensity density per cohort.",
     "Left: pairwise overlap coefficients of the per-domain propensity-score densities (mean 0.18, barely "
     "overlapping; the covariate distributions almost completely separate the cohorts); right: propensity "
     "density per cohort."),
    ("I3_mine_mutual_info.png", "I",
     "MINE mutual information I(domain; ·) per experiment",
     "Representation MI high for multi-domain, low for clean control.",
     "Mutual-information estimates (MINE) between domain identity and the survival outcome, and between domain "
     "identity and the learned CLS representation, for experiments A–E. Representation MI is high for the "
     "multi-domain LIHC experiments (B/C) and drops for the clean breast control (E); the learned representation "
     "carries domain signal whose strength tracks the missingness/heterogeneity level.",
     "MINE mutual-information estimates: I(domain; survival outcome) and I(domain; learned representation), "
     "estimated separately for experiments A–E. Representation MI is high for the multi-domain LIHC experiments "
     "(B/C) and low for the clean breast-cancer control (E); the domain signal carried by the learned "
     "representation tracks the degree of data heterogeneity."),
    ("I4_probe_auc_vs_cindex.png", "I",
     "Domain-separability AUC vs DANN C-index gain",
     "Higher separability ↔ no GRL benefit.",
     "Scatter of the domain-separability AUC (2-layer MLP on ERM hidden features) against the DANN C-index gain "
     "for experiments A–E. Experiments with near-perfect separability (AUC > 0.96) show no adversarial benefit; "
     "the only experiment with weak separability (E) shows the largest (still non-significant) gain.",
     "Scatter of the domain-separability probe AUC (2-layer MLP on ERM hidden features) against the DANN C-index "
     "gain (experiments A–E). Experiments with near-perfect separability (AUC>0.96) show no adversarial benefit; "
     "the only experiment with weak separability (E) shows the largest (still non-significant) gain."),
    # ---------------- Experiment J (method comparison) ----------------
    ("J1_method_comparison_cindex.png", "J",
     "C-index across 9 DA/DG methods (experiments A/B/C)",
     "All methods ≈ ERM.",
     "Validation C-index of ERM, DANN, CORAL, IRM, GroupDRO, V-REx, Fish, Mixup and MLDG on the A/B/C data "
     "splits (3 seeds each). Every method lands within a hair of ERM; the whole domain-adaptation family fails "
     "to improve cross-population survival prediction.",
     "Validation C-index of the 9 methods (ERM/DANN/CORAL/IRM/GroupDRO/V-REx/Fish/Mixup/MLDG) on the A/B/C "
     "splits (3 seeds each). "
     "Every method lands within a hair of ERM; the whole domain-adaptation family fails on cross-population "
     "survival prediction."),
    ("J2_method_comparison_ibs.png", "J",
     "Integrated Brier Score across methods",
     "IBS also ≈ ERM.",
     "Integrated Brier Score of the same nine methods on the A/B/C splits. As with C-index, no method consistently "
     "improves the calibration/discrimination trade-off over ERM.",
     "Integrated Brier Score of the same nine methods on the A/B/C splits. As with C-index, no method "
     "consistently outperforms ERM."),
    ("J3_method_comparison_bootstrap_forest.png", "J",
     "Δ vs ERM: 1000× stratified bootstrap",
     "All CIs include zero (some significantly negative).",
     "Forest plot of Δ = C-index(method) − C-index(ERM) with 95% bootstrap CIs (B = 1000 per seed, pooled over 3 "
     "seeds, stratified by cohort) for experiments A/B/C. No method's interval excludes zero; where the interval "
     "is tight the effect is negative, i.e. the DA/DG methods are no better and often slightly worse than ERM.",
     "Forest plot of the 95% bootstrap CIs for Δ=C-index(method)−C-index(ERM) (1000 draws per seed, pooled over "
     "3 seeds, stratified by cohort; experiments A/B/C). "
     "No method's interval excludes 0; the tighter the interval, the more negative the effect; the DA/DG "
     "methods are no better than, and often slightly worse than, ERM."),
    # ---------------- Experiment K (clinical metrics) ----------------
    ("K1_ibs_comparison.png", "K",
     "IBS: ERM/DANN/CORAL/IRM vs KM null model",
     "All models beat the null; no method dominates.",
     "Integrated Brier Score on the experiment-A validation population for ERM, DANN, CORAL and IRM versus the "
     "Kaplan–Meier null model. All four models beat the null (models carry information) but no DA/DG method "
     "meaningfully improves over ERM.",
     "Integrated Brier Score on the experiment-A validation set for ERM/DANN/CORAL/IRM vs the Kaplan–Meier null "
     "model. All four models beat the null (the models carry information), "
     "but no domain-adaptation method significantly outperforms ERM."),
    ("K2_brier_score_curves.png", "K",
     "Time-dependent Brier curves (overall + per cohort)",
     "Flat, near-identical curves.",
     "Time-dependent Brier score at 6/12/24/36/48/60 months for the four methods, overall (left) and per cohort "
     "(right). Curves are close across methods and cohorts.",
     "Time-dependent Brier-score curves at 6/12/24/36/48/60 months for the four methods (left: overall, right: "
     "per cohort). Curves are close across methods and cohorts."),
    ("K3_calibration_curves.png", "K",
     "Calibration curves @ 12/36/60 months",
     "ECE ≈ 0.03–0.08; reasonable calibration.",
     "Calibration of the four methods at 12/36/60 months (predicted vs observed event rate per risk decile). "
     "Expected Calibration Error is 0.03–0.08; the models are reasonably calibrated and the DA/DG variants do not "
     "worsen calibration.",
     "Calibration curves at 12/36/60 months for the four methods (predicted vs observed event rate per risk "
     "decile). ECE is 0.03–0.08; the models are well calibrated and the domain-adaptation variants do not "
     "worsen calibration."),
    ("K4_decision_curve_analysis.png", "K",
     "Decision-curve analysis @ 36 months (IPCW net benefit)",
     "Clinical utility close across methods.",
     "Decision-curve analysis at 36 months comparing the four methods against Treat-All and Treat-None (IPCW net "
     "benefit). Net-benefit curves are close across methods; domain-adaptation offers no measurable clinical "
     "utility gain over ERM.",
     "Decision-curve analysis at 36 months (IPCW net benefit) comparing the four methods against "
     "Treat-All/Treat-None. The net-benefit curves are close across methods; domain adaptation offers no "
     "measurable clinical-utility gain."),
    # ---------------- Experiments L/M (multi-cancer validation) ----------------
    ("L1_coad_cindex_comparison.png", "L",
     "COAD validation: C-index (TCGA-COAD vs CPTAC-COAD)",
     "ERM best; DA/DG no better.",
     "Validation C-index of ERM/DANN/CORAL/IRM for colorectal cancer (TCGA-COAD vs CPTAC-COAD, 3 seeds). ERM is "
     "best or tied; DANN/CORAL/IRM are no better (all bootstrap Δ non-significant).",
     "Validation C-index for colorectal cancer (TCGA-COAD vs CPTAC-COAD, 3 seeds). ERM is best or tied; "
     "DANN/CORAL/IRM provide no gain (all bootstrap Δ non-significant)."),
    ("M1_luad_cindex_comparison.png", "M",
     "LUAD validation: C-index (TCGA-LUAD vs MSKCC-2020)",
     "DA/DG no better than ERM.",
     "Validation C-index of ERM/DANN/CORAL/IRM for lung adenocarcinoma (TCGA-LUAD vs MSKCC-2020, 3 seeds). No "
     "method significantly outperforms ERM.",
     "Validation C-index for lung adenocarcinoma (TCGA-LUAD vs MSKCC-2020, 3 seeds). No method significantly "
     "outperforms ERM."),
]

def main():
    rows = []
    for i, (fname, exp, title, content, leg_en, cap_zh) in enumerate(FIGURES, start=1):
        new_name = f"Figure_{i:02d}.png"
        src = (LHC if (LHC / fname).exists() else EXT) / fname
        assert src.exists(), f"missing {src}"
        shutil.copy2(src, PKG / new_name)
        rows.append({
            "Figure": i,
            "New File": new_name,
            "Original File": fname,
            "Experiment": exp,
            "Title": title,
            "Content": content,
            "Figure Legend (EN)": leg_en,
            "Figure Legend (Chinese draft)": cap_zh,
        })
        print(f"{i:2d} {new_name}  <-  {fname}")

    df = pd.DataFrame(rows)
    xlsx = PKG / "Figure_Legends.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Figures")
        ws = writer.sheets["Figures"]
        widths = {"A": 6, "B": 16, "C": 34, "D": 10,
                  "E": 42, "F": 46, "G": 80, "H": 80}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"
    print(f"\nCopied {len(FIGURES)} figures -> {PKG}")
    print(f"Legend table -> {xlsx}")


if __name__ == "__main__":
    main()
