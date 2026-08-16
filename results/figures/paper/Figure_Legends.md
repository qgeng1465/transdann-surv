# Figure Legends for the TransDANN Manuscript

> Generated 2026-07-31 (decluttered re-layout 2026-08-02) by
> `scripts/16_generate_paper_figures.py`. Every number below was read directly
> from the verified result artifacts (`results/*.json`, `results/tables/*.csv`,
> `data_processed/*.csv`); annotations that appeared in earlier drafts were
> cross-checked against the data and corrected where they were inaccurate.
> 2026-08-02 re-layout: removed redundant panel boxes/borders, merged the three
> per-cohort panels of Figure 2 into one, replaced the embedded SHAP PNGs with
> natively-computed box-free beeswarms, moved forest-plot p-values to a
> dedicated column, and colour-coded BH-FDR significance (green/red) in
> Figure 7b.
>
> Figures: 8 main (`results/figures/paper/Figure_01..08_*.png/.pdf`) + 5
> supplementary (`results/figures/paper/supplementary/Figure_S1..S5_*.png/.pdf`).
> Supplementary data tables (CSV) live in `results/figures/paper/supplementary/`.

---

## Figure 1. The Domain Fingerprint in Multi-Institutional HCC Data

**(a)** Missingness heatmap across five cohorts (rows) and four clinical
features (Age, Sex, Stage, Grade). Color intensity encodes the proportion of
missing values, revealing that missingness patterns alone almost perfectly
distinguish cohorts (e.g., Age 100% missing only in hcc_msk_2024; Grade 100%
missing in every non-TCGA cohort). **(b)** Age distribution by cohort (violin
with quartile marks). SEER is older (median 64 y) than TCGA-LIHC (median
61 y); lihc_amc_prv is the youngest (median 55 y) and hcc_meric_2021 the
oldest (median 69 y). hcc_msk_2024 has no age data at all (100% missing),
itself a domain signal. **(c)** Stage distribution by cohort (100% stacked
bar). Stage was harmonized to I/II/III/IV/Unknown from the heterogeneous raw
labels. The proportion of unknown stage varies from 20.7% (TCGA-LIHC) to 100%
(hcc_msk_2024, lihc_amc_prv, hcc_meric_2021), reflecting the different staging
practices of genomic databases versus clinical registries (SEER has 48.2%
unknown stage). **(d)** Event rate (bars, left axis) and sample size (open
circles, right axis, log scale) by cohort. Event rates span 16.0% (lihc_amc_prv)
to 78.1% (US_SEER, hcc_meric_2021), and sample sizes span 105–108,638; the
joint distribution of covariates and outcomes encodes domain identity far
beyond what missingness alone explains.

---

## Figure 2. DANN Yields Zero to Negative Gain Across All Experimental Settings

**(a)** Validation C-index for DANN (red) vs Baseline (blue) across experiments
A (TCGA vs SEER), B (TCGA vs external HCC), C (all five cohorts). The gains are
Δ = +0.0007, −0.0032, −0.0011 (two of three negative), with bootstrap p = 0.898,
0.808, 0.744; all within noise. **(b)** Time-dependent AUC at 12/36/60 months.
Baseline (solid) is ≥ DANN (dashed) at all 9 time points across the three
experiments (markers denote experiments; legend at right). **(c)** Per-cohort
C-index for experiments A, B, and C combined in one panel (light separators
between experiments). DANN consistently degrades performance on the reference
cohort TCGA-LIHC (Δ annotated in red, −0.034 to −0.024) while yielding
inconsistent micro-gains on smaller external cohorts; adversarial
representation redistribution rather than true generalization. **(d)** Domain
classifier accuracy trajectory in experiment A. The classifier reaches 93.8%
by epoch 5 and stabilizes at 97.3%, demonstrating that the domain fingerprint
is structurally uneraseable by GRL (the α schedule is shown in Figure S3).

---

## Figure 3. Imputation Does Not Remove the Domain Fingerprint

**(a)** Missingness rates before (blue) and after (orange) KNN imputation
(k = 7) across the four clinical features. All missing values are physically
removed post-imputation (0%). **(b)** Domain-separability probe AUC before and
after imputation. Contrary to the hypothesis that missingness is the primary
domain fingerprint, the probe AUC slightly *increases* from 0.954 to 0.960
after imputation; the domain fingerprint is not carried by missingness.
**(c)** Domain classifier accuracy trajectories for the original (blue, peak
97.3%) and imputed (orange, peak 97.7%) data. Both curves saturate near 97%,
confirming that the classifier relies on covariate value distributions rather
than missing indicators. **(d)** Validation C-index before and after imputation.
The DANN gain shifts from Δ = +0.0007 (p = 0.898) to Δ = +0.0027 (p = 0.320);
neither reaches significance. Together these panels show the causal chain
"imputation → missingness removal → fingerprint removal → GRL revival" breaks
at the second link.

---

## Figure 4. Clean Data Reveals the Upper Bound of GRL Benefit

**(a)** Domain classifier accuracy trajectories for LIHC experiment A (blue,
peaks at 97.3%) and the clean-data experiment E, TCGA-BRCA vs METABRIC (orange,
peaks at 64.3%). The dramatic drop in domain separability under clean data
suggests GRL has more room to operate when the domain fingerprint is weaker.
**(b)** Per-cohort C-index for experiment E. DANN yields Δ = +0.0074, the
largest positive gain observed, yet still fails to reach significance
(p = 0.058; METABRIC Δ = +0.0026, TCGA-BRCA Δ = −0.0047). **(c)** Scatter of GRL
gain (Δ C-index) vs domain-classifier peak accuracy across experiments A–E.
A negative trend emerges (regression with 95% CI): the weaker the fingerprint,
the larger the potential benefit, but even the best case (E, red) is
non-significant. **(d)** Forest plot of bootstrap 95% CIs for the validation-set
Δ across all five experiments (B = 1000, stratified by cohort). All five CIs
cross zero (p > 0.05); as a robustness check on the full training set,
experiment C is significantly *worse* with DANN (p = 0.030). **(e–f)** Bootstrap
distributions of the validation Δ for the two extremes: the flat null
distribution of experiment A (e, p = 0.898) and the marginally right-skewed
distribution of experiment E (f, p = 0.058); the empirical upper bound of GRL
benefit in this study.

---

## Figure 5. SHAP Analysis Shows GRL Does Not Corrupt Feature Attention

**(a–b)** SHAP beeswarm plots for the Baseline (a) and DANN (b) models on a
300-patient subsample of the experiment-A validation set (recomputed natively
with Expected Gradients; colour = feature value, grey = missing). DANN does not
distort the ranking of feature importance; it increases the spread of Stage
SHAP values. **(c)** Mean |SHAP| comparison (Baseline blue vs DANN red). DANN
increases Stage importance by +26.6% (8.31 → 10.53) while reducing Age
importance by −17.7% (4.53 → 3.73), consistent with a rebalancing of attention
toward the most prognostically relevant feature rather than a corruption of
interpretability. **(d)** Mean |SHAP| of Stage by cohort (TCGA-LIHC vs US_SEER,
validation set). DANN amplifies Stage reliance in both cohorts (TCGA +47%,
SEER +26%).

---

## Figure 6. Kaplan-Meier Stratification and Clinical Utility

**(a–b)** Kaplan-Meier survival curves for high-risk (red) and low-risk (blue)
groups stratified at the median predicted risk from the Baseline (a) and DANN
(b) models on the SEER test set (n = 1,500, 1,162 events). Both models achieve
highly significant separation (log-rank P = 3.97 × 10⁻²⁵ and 1.83 × 10⁻²⁸,
respectively) with identical median survival estimates (high-risk 8 months,
low-risk 25 months). **(c)** Summary of stratification statistics: C-index,
log-rank P, median survival (high/low) and group sizes are nearly identical
between the two models; GRL neither enhances nor degrades clinically
actionable risk separation. **(d)** Scatter of discrimination (C-index) vs
stratification strength (−log₁₀ log-rank P). The two points overlap almost
completely, demonstrating that both models carry equivalent prognostic
information despite DANN's adversarial objective.

---

## Figure 7. Systematic Failure of Domain Generalization Methods

**(a)** C-index across nine methods (ERM, DANN, CORAL, IRM, GroupDRO, V-REx,
Fish, Mixup, MLDG) on experiment A data. Error bars = SD over three seeds. No
method significantly outperforms ERM. **(b)** Forest plot of bootstrap 95% CIs
for the gain (Δ C-index) of each method relative to ERM across experiments
A/B/C (p-values in the right column; green/red = survives Benjamini–Hochberg
FDR correction across all 35 tests, grey = not significant). After FDR only
Mixup (+0.019, q = 0.012) and V-REx in split B (−0.067, q = 0.023) survive;
the nominally significant IRM (+0.013, p = 0.043 in split C) does not; the positive signal is split-dependent and confined to input-level mixup,
while risk-distribution DG methods are actively harmful. **(c)** Integrated
Brier score (IBS; lower = better) for ERM/DANN/CORAL/IRM with the Kaplan-Meier
null model as a reference: all methods are informative (IBS 0.202–0.205 vs
0.207 KM null) with no clinically meaningful separation. **(d)**
Representation-level domain separability (probe AUC) for the baseline vs DANN
encoder across experiments A–E. Adversarial training does not reduce the
separability of the learned representation (A: 0.972 → 0.972; E: 0.709 →
0.706); GRL cannot erase the fingerprint. **(e)** Training time vs C-index
(experiment A; colour legend, no per-point labels). Adversarial training adds
non-trivial cost (DANN ≈ 31 s vs ERM ≈ 25 s) for zero gain; the slowest method
is Fish (≈ 136 s), which is not more accurate. **(f)** Adversarial oscillation
in experiment B (4 domains): the domain loss oscillates wildly (0.4 → 3.6 →
0.4, blue, left axis) while the validation C-index stays flat (red dashed,
right axis); the adversarial and survival optimization tracks are decoupled.

---

## Figure 8. Quantifying Domain-Survival Entanglement and Cross-Cancer Generalization

**(a)** Pairwise Wasserstein-1 distance matrix across the five cohorts computed
on Age (years). Large distances (e.g., hcc_meric_2021 vs lihc_amc_prv, W1 =
13.3 y; TCGA vs SEER, 5.0 y) confirm that covariate distribution shifts are
substantial and structured. **(b)** Propensity-score densities for TCGA-LIHC
(blue) and US_SEER (red) from a 5-class logistic-regression domain classifier
on Age + coded Sex/Stage/Grade. The near-zero overlap (overlap coefficient =
0.05; mean pairwise 0.18) quantifies the population shift beyond what
missingness alone explains. **(c)** MINE mutual-information estimates
(Donsker–Varadhan lower bound) across experiments A–E. Note the correct
interpretation: representation-level entanglement I(domain; representation) is
*high* for the multi-domain experiments B (0.83) and C (0.47) and *lowest* for
the clean-data experiment E (0.07); raw survival-level differences
I(domain; survival) are largest for E (0.32). GRL benefit tracks the
representation-level quantity, not the raw survival difference. **(d)** Scatter
of baseline representation probe AUC (domain separability) vs GRL gain across
A–E (regression with 95% CI): the more separable the representation, the more
negative or nil the GRL gain. **(e–f)** Cross-cancer validation in COAD
(TCGA-COAD vs CPTAC-COAD, e) and LUAD (TCGA-LUAD vs LUAD-MSKCC-2020, f). In
both cancers DANN fails to outperform ERM (COAD Δ = −0.025, p = 0.50; LUAD
Δ = +0.005, p = 0.29), and CORAL/IRM show no consistent gain; the systematic
failure generalizes beyond hepatocellular carcinoma.

---

## Figure S1. t-SNE Embeddings of Latent Representations

**(a–c)** Two-dimensional t-SNE projections of CLS-token embeddings for
experiments A, B, and C (≤3,000 subsampled patients per panel). Points are
colored by cohort and shaped by model (circles = DANN, triangles = Baseline).
In every experiment the DANN and Baseline embeddings overlap almost perfectly
and cohorts remain well-separated: adversarial training produces no
domain-invariant clustering.

---

## Figure S2. Training Dynamics Across Experiments

Each row shows one experiment (A, B, C). Within each row: survival loss (green),
domain loss (orange), domain classifier accuracy (red dashed) and validation
C-index (blue, right axis). In experiment A the domain loss collapses while
domain accuracy saturates at ~97%. In experiment B the domain loss oscillates
strongly (minimax instability, 0.4–3.6) without improving the flat validation
C-index; the two optimization tracks decouple.

---

## Figure S3. GRL Scheduling and Hyperparameter Trajectories

The adversarial weight α (red, left axis) follows the standard sigmoid warm-up
from 0 to 1, and the domain-loss weight (orange dashed, right axis) ramps
linearly from 0 to 0.3 over training. These are the recommended practice in the
DANN literature; the failure reported here is therefore not attributable to
poorly tuned GRL scheduling.

---

## Figure S4. Calibration Curves at Multiple Time Points

**(a–c)** Calibration plots at 12, 36, and 60 months for ERM and DANN on the
experiment-A validation set. Predicted probabilities are binned into deciles;
the diagonal dashed line is perfect calibration. **(d)** Expected Calibration
Error (ECE) summary across time points. ECE ≈ 0.03–0.08 for both models and is
comparable between them; DANN does not degrade calibration.

---

## Figure S5. Decision Curve Analysis

Decision curve analysis at 36 months on the experiment-A validation set. Net
benefit is plotted against threshold probability for ERM (blue), DANN (red),
CORAL (green) and IRM (orange), with treat-all (grey dashed) and treat-none
(black dotted) as references. The model curves nearly coincide, indicating
equivalent clinical utility and net benefit across methods.

---

## Supplementary data tables (CSV)

| Table | File | Content |
|---|---|---|
| S1 | `Supp_Table_S1_data_inventory.csv` | All 36 harmonized data sources: sample size, event rate, cancer type, experiment usage |
| S2 | `Supp_Table_S2_cohort_summary.csv` | Cohort-level summary for the 5 LIHC + E/L/M cohorts (N, events, event rate, age median, per-feature missingness, experiments used in) |
| S3 | `Supp_Table_S3_baseline_results.csv` | DANN vs Baseline validation C-index, Δ, bootstrap p and 95% CI for experiments A–H |
| S4 | `Supp_Table_S4_method_comparison.csv` | 9-method comparison (C-index, IBS, Δ vs ERM, bootstrap p, training time) for A/B/C |
| S5 | `Supp_Table_S5_clinical_metrics.csv` | IBS, KM-null IBS, ECE@12/36/60 for ERM/DANN/CORAL/IRM |
| S6 | `Supp_Table_S6_domain_quantify.csv` | MINE mutual information and representation probe AUC per experiment |
| S7 | `Supp_Table_S7_multicancer_validation.csv` | COAD/LUAD multi-cancer validation results |
