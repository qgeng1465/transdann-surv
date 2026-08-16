#!/usr/bin/env python3
"""
08_exp_h_km_curves.py; Kaplan-Meier stratification on the Exp A test set (SEER)
===============================================================================
For Experiment A (TCGA_LIHC vs US_SEER), use the held-out val/test SEER cohort,
stratify patients into High-Risk / Low-Risk by the median predicted risk from
each model (Baseline and DANN), draw K-M survival curves and report the
log-rank test P-value.

Outputs:
  results/figures/extended/H1_km_baseline_vs_dann.png
  results/H_km_results.json
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lihc_recon import reconstruct_full_data, _load_train_02
from transdann_utils import TransDANNSurvV3, deep_hit_risk, DEVICE

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = BASE_DIR / "results" / "lihc_experiments" / "A_tcga_vs_seer"
FIG_DIR = BASE_DIR / "results" / "figures" / "extended"
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    full_df, times, events, domains, sources = reconstruct_full_data("A_tcga_vs_seer")

    # Rebuild data dict to recover the val block and model architecture params.
    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[(df["Survival_Months"] > 0) & (df["Vital_Status"].isin([0, 1]))]
    exp_df = df[df["Source"].isin(["TCGA_LIHC", "US_SEER"])].copy()
    data = _load_train_02().load_and_preprocess(
        exp_df, surv_type="deephit", n_bins=32, subsample_seer=10000)
    n_train = len(data["train_df"])
    val_mask = np.arange(n_train, len(full_df))
    assert len(val_mask) == len(data["val_df"]), "val block mismatch"

    val_df = full_df.iloc[val_mask].reset_index(drop=True)
    val_t = times[val_mask]
    val_e = events[val_mask]
    val_src = sources[val_mask]

    # SEER test subset
    seer_mask = val_df["Source"].values == "US_SEER"
    km_t, km_e = val_t[seer_mask], val_e[seer_mask]
    print(f"SEER test set: {int(seer_mask.sum())} patients, "
          f"{int((km_e == 1).sum())} events")

    results = {"n_seer_test": int(seer_mask.sum()), "n_events": int((km_e == 1).sum()),
               "models": {}}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150, sharey=True)

    for ax, mode in zip(axes, ("baseline", "dann")):
        model = TransDANNSurvV3(
            num_continuous=len(data["cont_features"]),
            num_categorical=len(data["cat_features"]),
            cat_cardinalities=data["cat_cardinalities"],
            d_model=128, n_heads=8, n_layers=4, dropout=0.15,
            num_domains=data["num_domains"],
            surv_head_type="deephit", n_bins=32,
        ).to(DEVICE)
        model.load_state_dict(torch.load(
            EXP_DIR / mode / "best_model.pth", map_location=DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            x_cont = torch.tensor(val_df[data["cont_features"]].values,
                                  dtype=torch.float32).to(DEVICE)
            x_cat = torch.tensor(val_df[data["cat_features"]].values,
                                 dtype=torch.long).to(DEVICE)
            surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
            risk_all = deep_hit_risk(surv_out, data["bin_centers"]).cpu().numpy()
        risk = risk_all[seer_mask]

        # C-index on SEER test (higher risk = worse outcome => pass -risk)
        ci = concordance_index(km_t, -risk, km_e)

        # Stratify high/low at the median risk
        med = np.median(risk)
        high = risk >= med
        low = ~high

        kmf_hi = KaplanMeierFitter().fit(km_t[high], km_e[high],
                                         label=f"High Risk (n={int(high.sum())})")
        kmf_lo = KaplanMeierFitter().fit(km_t[low], km_e[low],
                                         label=f"Low Risk (n={int(low.sum())})")

        kmf_hi.plot_survival_function(ax=ax, color="#C44E52", ci_show=True, lw=2)
        kmf_lo.plot_survival_function(ax=ax, color="#4C72B0", ci_show=True, lw=2)

        # Log-rank test
        lr = logrank_test(km_t[high], km_t[low], km_e[high], km_e[low])
        p = lr.p_value
        med_hi = kmf_hi.median_survival_time_
        med_lo = kmf_lo.median_survival_time_

        ax.set_title(f"{mode.upper()}; SEER test\nC-index = {ci:.4f} | "
                     f"log-rank P = {p:.2e}")
        ax.set_xlabel("Time (months)")
        ax.set_xlim(0, 120)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

        results["models"][mode] = {
            "cindex_seer_test": round(float(ci), 4),
            "risk_median": round(float(med), 5),
            "n_high": int(high.sum()),
            "n_low": int(low.sum()),
            "logrank_p": float(p),
            "median_survival_high_months": round(float(med_hi), 2)
            if med_hi == med_hi else None,
            "median_survival_low_months": round(float(med_lo), 2)
            if med_lo == med_lo else None,
        }
        print(f"[{mode}] C-index={ci:.4f} log-rank P={p:.2e} "
              f"med_surv high={med_hi:.1f} low={med_lo:.1f}")

    fig.suptitle("Kaplan-Meier survival stratified by predicted risk "
                 "(Exp A, SEER test set, median split)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "H1_km_baseline_vs_dann.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")

    with open(BASE_DIR / "results" / "H_km_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {BASE_DIR / 'results' / 'H_km_results.json'}")


if __name__ == "__main__":
    main()
