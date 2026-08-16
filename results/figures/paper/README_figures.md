# Paper Figure Package; README

Generated 2026-07-31 by `scripts/16_generate_paper_figures.py`.

- **8 main figures** → `results/figures/paper/Figure_01..08_*.png/.pdf` (300 dpi PNG + vector PDF)
- **5 supplementary figures** → `results/figures/paper/supplementary/Figure_S1..S5_*.png/.pdf`
- **Full captions** → `results/figures/paper/Figure_Legends.md`
- **Supplementary data tables (CSV)** → `results/figures/paper/supplementary/Supp_Table_S1..S7_*.csv`

## Figure ↔ manuscript-section correspondence

| Section | Figure | File | Panels | Narrative role |
|---|---|---|---|---|
| 3.1 Data characteristics | **Fig 1** | `Figure_01_domain_fingerprint` | 2×2 | Establishes the *domain fingerprint*: missingness is only the visible surface; distributions are the essence |
| 3.2 Main results | **Fig 2** | `Figure_02_main_results` | 2×3 | C-index, time-AUC, per-cohort, domain classifier; zero to negative gain |
| 3.3 Mechanism I | **Fig 3** | `Figure_03_imputation` | 2×2 | Refutes "missingness is the cause": imputation does not remove the fingerprint |
| 3.4 Mechanism II + statistics | **Fig 4** | `Figure_04_clean_data_bootstrap` | 2×3 | Clean data is the best case for GRL; bootstrap shows it is still not significant |
| 3.5 Interpretability | **Fig 5** | `Figure_05_shap` | 2×2 | GRL does not corrupt feature attention (Stage importance rises) |
| 3.6 Clinical validation | **Fig 6** | `Figure_06_km_clinical` | 2×2 | K-M stratification: the two models are clinically equivalent |
| 3.7 Methodological comparison | **Fig 7** | `Figure_07_methods` | 2×3 | Systematic failure of DA/DG methods, not just DANN |
| 3.8 Quantification + generalization | **Fig 8** | `Figure_08_quantify_multicancer` | 2×3 | Wasserstein / propensity / MINE / cross-cancer |
| Supplementary | **S1** | `supplementary/Figure_S1_tsne` | 1×3 | t-SNE of latent representations |
| Supplementary | **S2** | `supplementary/Figure_S2_training_dynamics` | 3×1 | Training dynamics A–C |
| Supplementary | **S3** | `supplementary/Figure_S3_grl_schedule` | 1×1 | GRL α / domain-weight scheduling |
| Supplementary | **S4** | `supplementary/Figure_S4_calibration` | 2×2 | Calibration curves at 12/36/60 mo |
| Supplementary | **S5** | `supplementary/Figure_S5_dca` | 1×1 | Decision curve analysis at 36 mo |

## Source data for each figure

| Figure | Primary sources |
|---|---|
| Fig 1 | `data_processed/lihc_all_cohorts.csv`, `results/lihc_missingness.csv` |
| Fig 2 | `results/lihc_experiments/{A,B,C}_*/comparison.json` + `dann|baseline/results.json`, `results/F_bootstrap_val_results.json` |
| Fig 3 | `results/D_imputation_probe.json`, Exp A/D histories, `results/F_bootstrap_val_results.json` |
| Fig 4 | Exp A/E histories, Exp E `comparison.json`, `results/F_bootstrap_val_results.json`, recomputed val bootstrap (B=1000, seed=777) |
| Fig 5 | `results/G_shap_results.json`, `results/figures/extended/G1/G2_shap_beeswarm_*.png` |
| Fig 6 | recomputed K-M from Exp A saved models (`best_model.pth`) on the SEER test set (n=1,500), `results/H_km_results.json` |
| Fig 7 | `results/experiments/exp_j/method_comparison.json`, `results/tables/table3_method_comparison.csv`, `results/experiments/exp_k/clinical_metrics.json`, `results/experiments/exp_i/domain_quantify.json` (probe) |
| Fig 8 | `results/experiments/exp_i/domain_quantify.json`, `results/experiments/exp_l/coad_results.json`, `results/experiments/exp_m/luad_results.json` |
| S1 | `results/lihc_experiments/{A,B,C}_*/dann|baseline/features.npy` |
| S2 | Exp A/B/C `dann/results.json` histories |
| S3 | Exp A `dann/results.json` history (α) |
| S4/S5 | `results/experiments/exp_k/clinical_metrics.json` |

## Data-accuracy corrections applied relative to the original v2 prompt

1. **Fig 8(c) / MINE narrative**: the draft claimed *"A highest, E lowest"*
   for I(domain; survival). The verified data show the opposite:
   `mi_domain_representation` is highest for B (0.83) / C (0.47) and **lowest for
   E (0.07)**; `mi_domain_survival` is highest for E (0.32), lowest for A/D
   (0.016). The figure plots both quantities and the legend now states the
   correct reading: GRL benefit tracks the representation-level entanglement.
2. **Fig 7(e) training time**: the draft claimed *"DANN is the slowest"*.
   Verified: Fish is the slowest (136 s in Exp A; 229 s in Exp C), DANN ≈ 31 s
   vs ERM ≈ 25 s. Annotation corrected to "DANN adds cost, no gain; Fish is the
   slowest."
3. **Fig 1(b/c) cohort statistics**: age medians and stage-unknown shares were
   recomputed from the raw harmonized data (e.g., SEER stage-unknown 48.2% vs
   the draft's 41.5%, which counted only the raw `NaN`; SEER/TCGA age medians
   64/61 y).
4. **Fig 5(c) SHAP deltas**: the +26.6% / −17.7% values are reported on the
   overall validation set (n = 1,546, the population on which SHAP was
   computed); the SEER-only subset is +25.7%. The legend now states this
   precisely.

## Reproduce

```bash
# regenerate all figures + legends + tables
python scripts/16_generate_paper_figures.py all
# or individually
python scripts/16_generate_paper_figures.py fig1   # … fig8, s1 … s5, tables
```

Runtime: ~5 min (Fig 6 K-M inference + Fig S1 t-SNE are the slow parts; GPU/CUDA
is used when available).
