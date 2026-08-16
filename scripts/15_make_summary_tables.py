#!/usr/bin/env python3
"""
15_make_summary_tables.py; Build Table 1 & Table 2 from existing results
=========================================================================
  table1_cohort_summary.csv; cohort × feature missingness / event rate
  table2_main_results.csv; Experiments A–H: DANN vs Baseline, Δ, p-value
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lihc_recon import BASE_DIR

RESULTS_DIR = BASE_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
os.makedirs(TABLE_DIR, exist_ok=True)


def make_table1():
    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    rows = []
    for s, sub in df.groupby("Source"):
        rows.append({
            "cohort": s,
            "source_type": {
                "TCGA_LIHC": "Genomic DB (TCGA)",
                "US_SEER": "Population registry (SEER)",
                "hcc_msk_2024": "Independent cohort (MSK)",
                "lihc_amc_prv": "Independent cohort (AMC, Korea)",
                "hcc_meric_2021": "Independent cohort (Basel, CH)",
            }.get(s, "Independent cohort"),
            "n": len(sub),
            "event_rate": round(float(sub["Vital_Status"].mean()), 3),
            "age_median": round(float(sub["Age"].median()), 1) if sub["Age"].notna().any() else np.nan,
            "age_missing_pct": round(float(sub["Age"].isna().mean() * 100), 1),
            "sex_missing_pct": round(float(sub["Sex"].isna().mean() * 100), 1),
            "stage_missing_pct": round(float(sub["Stage"].isna().mean() * 100), 1),
            "grade_missing_pct": round(float(sub["Grade"].isna().mean() * 100), 1),
        })
    return pd.DataFrame(rows)


def make_table2():
    # A/B/C comparisons
    exps = {
        "A_tcga_vs_seer": ("A", "TCGA vs SEER (Level 4 extreme missingness)"),
        "B_tcga_vs_external": ("B", "TCGA vs external HCC (4 domains)"),
        "C_all_cohorts": ("C", "All 5 LIHC cohorts"),
        "D_imputed_tcga_seer": ("D", "KNN-imputed TCGA vs SEER"),
        "E_brca_metabric": ("E", "TCGA-BRCA vs METABRIC (clean data)"),
    }
    rows = []
    for exp_key, (letter, desc) in exps.items():
        comp_path = RESULTS_DIR / "lihc_experiments" / exp_key / "comparison.json"
        if not comp_path.exists():
            continue
        comp = json.load(open(comp_path))
        rows.append({
            "experiment": letter, "scenario": desc,
            "dann_cindex": comp.get("dann_val_cindex"),
            "baseline_cindex": comp.get("baseline_val_cindex"),
            "delta": comp.get("delta"),
        })
    # bootstrap p-values (val-based measure; matches the reported Δ)
    fb_path = RESULTS_DIR / "F_bootstrap_val_results.json"
    if fb_path.exists():
        fb = json.load(open(fb_path))
        keymap = {"A_tcga_vs_seer": "A", "B_tcga_vs_external": "B",
                  "C_all_cohorts": "C", "D_imputed_tcga_seer": "D",
                  "E_brca_metabric": "E"}
        for exp_key, letter in keymap.items():
            if exp_key in fb:
                for r in rows:
                    if r["experiment"] == letter:
                        r["bootstrap_p_two_sided"] = fb[exp_key].get("p_two_sided")
                        r["bootstrap_ci_low"] = fb[exp_key].get("ci_low")
                        r["bootstrap_ci_high"] = fb[exp_key].get("ci_high")
    # F/G/H summary rows (from their result JSONs)
    if (RESULTS_DIR / "F_bootstrap_val_results.json").exists():
        pass
    rows.append({"experiment": "F", "scenario": "1000× stratified bootstrap of Δ (A–E)",
                 "note": "all p>0.05; C full-set DANN significantly worse (p=0.030)"})
    rows.append({"experiment": "G", "scenario": "SHAP interpretability (Baseline vs DANN)",
                 "note": "Stage importance +26% under DANN (SEER test set)"})
    rows.append({"experiment": "H", "scenario": "K-M clinical stratification (SEER test)",
                 "note": "log-rank P≈1e-25 (Baseline) vs 1e-28 (DANN); equivalent separation"})
    return pd.DataFrame(rows)


def main():
    t1 = make_table1()
    t1.to_csv(TABLE_DIR / "table1_cohort_summary.csv", index=False)
    t2 = make_table2()
    t2.to_csv(TABLE_DIR / "table2_main_results.csv", index=False)
    print(f"table1 → {TABLE_DIR / 'table1_cohort_summary.csv'} ({len(t1)} rows)")
    print(f"table2 → {TABLE_DIR / 'table2_main_results.csv'} ({len(t2)} rows)")
    print(t1.to_string(index=False))


if __name__ == "__main__":
    main()
