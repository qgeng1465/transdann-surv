#!/usr/bin/env python3
"""
13_exp_lm_multicancer_validation.py; Experiments L/M: multi-cancer validation (COAD & LUAD)
================================================================================
Generalisability check: does the HCC story hold in two more cancers?

  Exp L; COAD/READ:  TCGA-COAD (461) vs coadread_dfci_2016 (619)
  Exp M; LUAD:       TCGA-LUAD (585) vs luad_mskcc_2020 (604)

Runs ERM / DANN / CORAL / IRM (the same backbone & config as Experiment J) on
the two-cancer feature set Age + Sex + Stage + Grade, and reports C-index, IBS,
per-cohort C-index and a 1000-sample stratified bootstrap of Δ vs ERM.

Outputs
-------
  results/experiments/exp_l/{results.json, coad_results.json}
  results/experiments/exp_m/{results.json, luad_results.json}
  results/logs/exp_lm_training.log
  results/figures/extended/L1_coad_cindex_comparison.png
  results/figures/extended/M1_luad_cindex_comparison.png
  results/tables/table6_multicancer_validation.csv
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lihc_recon import BASE_DIR, build_data

RESULTS_DIR = BASE_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
for d in (TABLE_DIR, LOG_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)


def _load_exp_j():
    spec = importlib.util.spec_from_file_location(
        "exp_j_mod", str(Path(__file__).resolve().parent / "11_exp_j_method_comparison.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPJ = _load_exp_j()
train_method = EXPJ.train_method
compute_bootstraps = EXPJ.compute_bootstraps
setup_logger = EXPJ.setup_logger
METHODS = ["ERM", "DANN", "CORAL", "IRM"]
logger = setup_logger("exp_lm")


# ===================================================================
# Data loading; two-cancer harmonised feature set (Age + Sex + Stage + Grade)
# ===================================================================
def load_cancer(cancer):
    """Load TCGA + an external cohort for one cancer.

    COAD: TCGA-COAD vs CPTAC-COAD (cBioPortal proteogenomic cohort; the
          originally proposed coadread_dfci_2016 has NO survival data and
          coad_silu_2022 is not downloaded; CPTAC-COAD is the same-cancer,
          different-database control that *does* carry stage + follow-up).
    LUAD: TCGA-LUAD vs luad_mskcc_2020 (as planned).
    """
    if cancer == "COAD":
        tcga = pd.read_csv(BASE_DIR / "data_processed" / "harmonized_COAD.csv")
        ext = pd.read_csv(BASE_DIR / "data_processed" / "harmonized_cptac_coad.csv")
        tcga = tcga.assign(Source="TCGA-COAD")
        ext = ext.assign(Source="CPTAC-COAD")
        label = "COAD"
    elif cancer == "LUAD":
        tcga = pd.read_csv(BASE_DIR / "data_processed" / "harmonized_LUAD.csv")
        ext = pd.read_csv(BASE_DIR / "data_processed" / "harmonized_luad_mskcc_2020.csv")
        tcga = tcga.assign(Source="TCGA-LUAD")
        ext = ext.assign(Source="MSKCC-2020")
        label = "LUAD"
    else:
        raise ValueError(cancer)
    cols = ["Source", "Age", "Sex", "Stage", "Grade", "Survival_Months", "Vital_Status"]
    df = pd.concat([tcga[cols], ext[cols]], ignore_index=True)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    return df, label


# ===================================================================
# Main
# ===================================================================
def run_cancer(cancer, seeds, epochs, patience):
    df, label = load_cancer(cancer)
    logger.info(f"\n{'#'*70}\nEXPERIMENT {label}: {df['Source'].value_counts().to_dict()}\n{'#'*70}")
    # Drop any feature that is 100 % missing in EVERY cohort (would only encode
    # a constant missing marker; same rationale as Exp E dropping TCGA-BRCA Grade).
    cat_feats = [c for c in ["Sex", "Stage", "Grade"]
                 if not all(df[df["Source"] == s][c].isna().mean() == 1.0
                            for s in df["Source"].unique())]
    data = build_data(df, ["Age"], cat_feats, surv_type="deephit", n_bins=32)
    val_df = data["val_df"]

    config = {
        "surv_type": "deephit", "d_model": 128, "n_layers": 4, "dropout": 0.15,
        "lr": 5e-4, "n_bins": 32, "epochs": epochs, "patience": patience,
        "per_cohort_loss": True, "domain_weight_max": 0.3,
    }
    exp_out = RESULTS_DIR / ("experiments/exp_l" if cancer == "COAD" else "experiments/exp_m")
    os.makedirs(exp_out, exist_ok=True)

    agg, risks_by_seed = {}, {}
    for method in METHODS:
        agg[method], risks_by_seed[method] = {}, {}
        for seed in seeds:
            out_dir = exp_out / f"{method}_seed{seed}"
            logger.info(f"[{label}][{method}] seed={seed} start")
            t0 = time.time()
            try:
                _, res = train_method(method, config, data, seed, out_dir)
                rfile = out_dir / "risks.npy"
                if rfile.exists():
                    risks_by_seed[method][seed] = np.load(rfile)
                logger.info(f"[{label}][{method}] seed={seed} done C={res['best_val_cindex']} "
                            f"IBS={res['ibs']} ({time.time()-t0:.0f}s)")
            except Exception as ex:
                import traceback; traceback.print_exc()
                logger.error(f"[{label}][{method}] seed={seed} FAILED: {ex}")
                res = None
            agg[method][seed] = res

    # aggregate
    out = {"cancer": cancer, "cohorts": list(data["cohort_map"].keys()),
           "cohort_map": data["cohort_map"], "n_samples": int(len(df))}
    for method in METHODS:
        ok = [r for r in agg[method].values() if r is not None]
        if ok:
            cinds = [r["best_val_cindex"] for r in ok]
            ibss = [r["ibs"] for r in ok if r["ibs"] is not None]
            out[method] = {
                "cindex_mean": round(float(np.mean(cinds)), 4),
                "cindex_std": round(float(np.std(cinds)), 4),
                "cindex_per_seed": cinds,
                "ibs_mean": round(float(np.mean(ibss)), 4) if ibss else None,
                "elapsed_sec": round(float(np.mean([r["elapsed_sec"] for r in ok])), 1),
            }
        else:
            out[method] = None
    boot = compute_bootstraps(cancer, METHODS, val_df, risks_by_seed, B=1000)
    out["bootstrap_vs_erm"] = boot
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancers", nargs="+", default=["COAD", "LUAD"],
                        choices=["COAD", "LUAD"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    all_results = {}
    for cancer in args.cancers:
        all_results[cancer] = run_cancer(cancer, args.seeds, args.epochs, args.patience)

    # save per-cancer JSONs
    for cancer in ["COAD", "LUAD"]:
        if cancer not in all_results:
            continue
        d = "exp_l" if cancer == "COAD" else "exp_m"
        fn = "coad_results.json" if cancer == "COAD" else "luad_results.json"
        with open(RESULTS_DIR / "experiments" / d / fn, "w") as f:
            json.dump(all_results[cancer], f, indent=2, default=str)

    # combined CSV
    rows = []
    for cancer, r in all_results.items():
        boot = r.get("bootstrap_vs_erm", {})
        for method in METHODS:
            m = r.get(method)
            b = boot.get(method, {})
            rows.append({
                "cancer": cancer, "method": method,
                "cindex_mean": m["cindex_mean"] if m else None,
                "cindex_std": m["cindex_std"] if m else None,
                "ibs": m["ibs_mean"] if m else None,
                "delta_vs_erm": b.get("delta_mean"),
                "ci_low": b.get("ci_low"), "ci_high": b.get("ci_high"),
                "p_two_sided": b.get("p_two_sided"),
            })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table6_multicancer_validation.csv", index=False)

    _make_figures(all_results)
    print("\nSUMMARY (C-index mean ± std):")
    for cancer, r in all_results.items():
        print(f"  [{cancer}]")
        for method in METHODS:
            m = r.get(method)
            if m:
                print(f"    {method:6s}: {m['cindex_mean']:.4f} ± {m['cindex_std']:.4f}  "
                      f"IBS={m['ibs_mean']}")
    print("\n✅ Experiments L/M complete.")


def _make_figures(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ERM": "#4c72b0", "DANN": "#c44e52", "CORAL": "#55a868",
              "IRM": "#8172b2"}
    for cancer, fname in [("COAD", "L1_coad_cindex_comparison.png"),
                          ("LUAD", "M1_luad_cindex_comparison.png")]:
        if cancer not in all_results:
            continue
        r = all_results[cancer]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        x = np.arange(len(METHODS))
        means = [r.get(m, {}).get("cindex_mean", np.nan) if r.get(m) else np.nan
                 for m in METHODS]
        ax.bar(x, means, 0.6, color=[colors[m] for m in METHODS])
        for xi, v in zip(x, means):
            ax.text(xi, v + 0.005, f"{v:.3f}" if not np.isnan(v) else "",
                    ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=9)
        ax.set_ylabel("C-index (val)")
        ax.set_ylim(0.4, 0.8)
        ax.set_title(f"Experiment {'L' if cancer=='COAD' else 'M'}; {cancer} "
                     f"multi-cancer validation")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
