#!/usr/bin/env python3
"""
32_exp_km_tcga.py; Kaplan–Meier + RMST analysis on TCGA-LIHC
=============================================================

Exp H already showed on the SEER test set that both models' risk stratification
significantly separates survival (K-M + log-rank).
This script adds the same perspective on the **target cohort TCGA** plus a
significance test for the outcome-distribution shift:

Part 1; TCGA clinical Stage-stratified K-M + log-rank
    Plot K-M on TCGA by coarse clinical stage (I / II / III+IV, dropping the 20.7%
    missing), log-rank-test the survival differences across stages; translating
    "the outcome distribution is the domain fingerprint" into clinical evidence in
    the target cohort.

Part 2; risk-group K-M (ERM vs DANN, exp_j seed 42, median split)
    Predict risk with both models on the full TCGA cohort (dev), then K-M +
    log-rank of high/low risk groups.
    Test: does GRL break the target cohort's clinical stratification ability
    (expected: both separate significantly and are close).
    Note: TCGA is in the training set (in-sample, descriptive use; different
    definition from Exp H's SEER val; already honestly labelled).

Part 3; bootstrap significance of the RMST@36 difference (TCGA vs other cohorts)
    Between-cohort difference of the observed KM RMST@36 (same definition as
    script 27) + within-cohort resampling 95% CI and two-sided p; upgrading
    "the TCGA vs SEER outcome-distribution shift of +8 months" from a point
    estimate to a significance test.

Outputs (results/experiments/exp_km_tcga/):
  - km_tcga_results.json / summary.txt
  - results/tables/table19_km_tcga.csv
  - results/figures/extended/H2_km_tcga_risk_groups.png
  - logs/32_exp_km_tcga.log

Usage:
  python3 scripts/32_exp_km_tcga.py
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_km_tcga"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "32_exp_km_tcga.log", mode="w"),
    ],
)
log = logging.getLogger("km_tcga")

spec02 = importlib.util.spec_from_file_location(
    "train_engine", SCRIPTS_DIR / "02_train_dann_lihc.py"
)
train_engine = importlib.util.module_from_spec(spec02)
spec02.loader.exec_module(train_engine)
EXPERIMENTS = train_engine.EXPERIMENTS

from lihc_recon import build_data  # noqa: E402
from transdann_utils import (  # noqa: E402
    DEVICE, TransDANNSurvV3, deep_hit_risk,
)

TAU = 36
N_BOOT = 500
SEED = 0
TAU_MAIN_KM = TAU


# ===================================================================
# 1. K-M helpers (lifelines for log-rank; sksurv for KM steps)
# ===================================================================
def km_curve_rmst(kt, ks, tau):
    t = np.concatenate([[0.0], kt])
    s = np.concatenate([[1.0], ks])
    m = int(np.searchsorted(t, tau, side="right"))
    if m == 0:
        return float(tau)
    ends = np.concatenate([t[1:m], [tau]])
    return float(np.sum(s[:m] * (ends - t[:m])))


def cohort_rmst(times, events, tau=TAU):
    from sksurv.nonparametric import kaplan_meier_estimator
    kt, ks = kaplan_meier_estimator(np.asarray(events, bool), np.asarray(times, float))
    if len(kt) == 0:
        return None
    return km_curve_rmst(kt, ks, tau)


# ===================================================================
# Part 3: RMST@36 difference TCGA vs each other cohort (bootstrap CI + p)
# ===================================================================
def rmst_diff_bootstrap(t_a, e_a, t_b, e_b, B=N_BOOT, seed=SEED):
    """RMST(τ)_a − RMST(τ)_b, resampling each cohort independently."""
    rng = np.random.RandomState(seed)
    n_a, n_b = len(t_a), len(t_b)
    draws = []
    for _ in range(B):
        ia = rng.randint(0, n_a, size=n_a)
        ib = rng.randint(0, n_b, size=n_b)
        r_a = cohort_rmst(t_a[ia], e_a[ia])
        r_b = cohort_rmst(t_b[ib], e_b[ib])
        if r_a is not None and r_b is not None:
            draws.append(r_a - r_b)
    draws = np.asarray(draws)
    if len(draws) == 0:
        return None
    return {
        "delta_rmst": round(float(draws.mean()), 2),
        "ci_low": round(float(np.percentile(draws, 2.5)), 2),
        "ci_high": round(float(np.percentile(draws, 97.5)), 2),
        "p_two_sided": round(min(2 * min(float(np.mean(draws > 0)),
                                         float(np.mean(draws < 0))), 1.0), 4),
        "n_draws": int(len(draws)),
    }


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="TCGA cohort clinical characterisation")
    parser.add_argument("--no-models", action="store_true",
                        help="skip Part 2 (risk-group K-M), Part 1/3 only")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    tcga = df[df["Source"] == "TCGA_LIHC"].copy()
    log.info(f"TCGA cohort: {len(tcga)} patients, "
             f"{int((tcga['Vital_Status'] == 1).sum())} events")

    out = {"tau": TAU, "n_boot": N_BOOT, "generated":
           datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # ---- Part 1: clinical Stage strata K-M + log-rank on TCGA ----
    def coarse_stage(s):
        # NaN → None; string checks ordered longest-prefix first so that
        # 'IIIA'/'IVA' don't collide with 'I'
        if pd.isna(s) or not isinstance(s, str):
            return None
        s = s.strip().upper()
        if s.startswith("IV"):
            return "III+IV"
        if s.startswith("III"):
            return "III+IV"
        if s.startswith("II"):
            return "II"
        if s.startswith("I"):
            return "I"
        if s.startswith("0"):
            return "I"  # '0' → early stage
        return None

    tcga = tcga.copy()
    tcga["stage_coarse"] = tcga["Stage"].apply(coarse_stage)
    # merge the single '0' into I; merge I/II/III+IV for the test
    stage_map = {"I": "I", "II": "II", "III+IV": "III+IV"}
    tcga["stage_grp"] = tcga["stage_coarse"].map(stage_map)
    part1 = {"coarse_map": stage_map, "n_with_stage": int(tcga["stage_grp"].notna().sum()),
             "n_missing_stage": int(tcga["stage_grp"].isna().sum())}
    from lifelines.statistics import logrank_test
    part1["groups"] = {}
    stage_rows = []
    for g in ["I", "II", "III+IV"]:
        m = tcga["stage_grp"] == g
        part1["groups"][g] = {"n": int(m.sum()),
                              "events": int((tcga["Vital_Status"][m] == 1).sum())}
        stage_rows.append({"cohort": "TCGA_LIHC", "part": "stage_strata",
                           "group": g, "n": int(m.sum()),
                           "events": int((tcga["Vital_Status"][m] == 1).sum())})
    ok_groups = [g for g in ["I", "II", "III+IV"] if part1["groups"][g]["n"] >= 10]
    if len(ok_groups) >= 2:
        from lifelines.statistics import multivariate_logrank_test
        gt_t = np.concatenate([tcga.loc[tcga["stage_grp"] == g,
                                        "Survival_Months"].values for g in ok_groups])
        gt_g = np.concatenate([[g] * int((tcga["stage_grp"] == g).sum())
                               for g in ok_groups])
        gt_e = np.concatenate([tcga.loc[tcga["stage_grp"] == g,
                                        "Vital_Status"].values for g in ok_groups])
        lr = multivariate_logrank_test(gt_t, gt_g, event_observed=gt_e)
        part1["logrank_p"] = round(float(lr.p_value), 6)
        part1["groups_tested"] = ok_groups
        log.info(f"[Part 1] TCGA Stage strata {ok_groups}: "
                 f"log-rank p={part1['logrank_p']:.4g}")
    else:
        part1["logrank_p"] = None
        log.info("[Part 1] too few Stage strata; skip log-rank")
    out["part1_stage_km"] = part1

    # ---- Part 2: risk-group K-M (ERM vs DANN) on TCGA dev ----
    part2 = None
    if not args.no_models:
        data = build_data(df[df["Source"].isin(["TCGA_LIHC", "US_SEER"])].copy(),
                          ["Age"], ["Sex", "Stage", "Grade"],
                          surv_type="deephit", n_bins=32, subsample_seer=10000)
        dev = pd.concat([data["train_df"], data["val_df"]], ignore_index=True)
        tcga_dev = dev[dev["Source"] == "TCGA_LIHC"].reset_index(drop=True)
        t_t = torch.tensor(tcga_dev[["Age"]].values, dtype=torch.float32)
        c_t = torch.tensor(tcga_dev[["Sex", "Stage", "Grade"]].values, dtype=torch.long)
        tcga_times = tcga_dev["Survival_Months"].values.astype(float)
        tcga_events = tcga_dev["Vital_Status"].values.astype(bool)

        part2 = {"population": "TCGA full dev (in-sample, descriptive)",
                 "n_tcga": len(tcga_dev), "models": {}}
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        for method, mode in (("ERM", "baseline"), ("DANN", "dann")):
            ckpt = (RESULTS_DIR / "experiments" / "exp_j" / f"exp_A_{method}"
                    / "seed_42" / "best_model.pth")
            if not ckpt.exists():
                log.error(f"[Part 2] missing {ckpt}")
                continue
            model = TransDANNSurvV3(
                num_continuous=len(data["cont_features"]),
                num_categorical=len(data["cat_features"]),
                cat_cardinalities=data["cat_cardinalities"],
                d_model=128, n_heads=8, n_layers=4, dropout=0.15,
                num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32,
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                             weights_only=True))
            model.eval()
            with torch.no_grad():
                surv_out, _, _ = model(t_t.to(DEVICE), c_t.to(DEVICE), alpha=0.0)
                risks = deep_hit_risk(surv_out, data["bin_centers"]).cpu().numpy()
            med = float(np.median(risks))
            high = risks >= med
            low = ~high
            lr = logrank_test(tcga_times[high], tcga_times[low],
                              tcga_events[high], tcga_events[low])
            part2["models"][method] = {
                "median_risk": round(med, 4),
                "n_high": int(high.sum()), "n_low": int(low.sum()),
                "logrank_p": round(float(lr.p_value), 6),
            }
            log.info(f"[Part 2] {method} risk-group log-rank p={lr.p_value:.4g} "
                     f"(n_high={high.sum()}, n_low={(~high).sum()})")
        out["part2_risk_group_km"] = part2

    # ---- Part 3: RMST@36 diff TCGA vs others (bootstrap CI + p) ----
    part3 = {"ref": "TCGA_LIHC"}
    tcga_t = tcga["Survival_Months"].values.astype(float)
    tcga_e = tcga["Vital_Status"].values.astype(bool)
    tcga_rmst = cohort_rmst(tcga_t, tcga_e)
    part3["rmst_tcga"] = round(float(tcga_rmst), 2) if tcga_rmst is not None else None
    part3["pairs"] = {}
    rmst_rows = []
    for src in ["US_SEER", "lihc_amc_prv", "hcc_msk_2024", "hcc_meric_2021"]:
        g = df[df["Source"] == src]
        if len(g) < 10:
            continue
        gt = g["Survival_Months"].values.astype(float)
        ge = g["Vital_Status"].values.astype(bool)
        r_other = cohort_rmst(gt, ge)
        d = rmst_diff_bootstrap(tcga_t, tcga_e, gt, ge)
        part3["pairs"][src] = {
            "rmst_other": round(float(r_other), 2) if r_other is not None else None,
            "delta_tcga_minus_other": d,
        }
        rmst_rows.append({"cohort": "TCGA_LIHC", "part": "rmst_diff",
                          "other": src, "rmst_tcga": part3["rmst_tcga"],
                          "rmst_other": part3["pairs"][src]["rmst_other"],
                          "delta": d["delta_rmst"] if d else None,
                          "ci_low": d["ci_low"] if d else None,
                          "ci_high": d["ci_high"] if d else None,
                          "p_two_sided": d["p_two_sided"] if d else None})
        if d:
            log.info(f"[Part 3] RMST@36 TCGA−{src} = {d['delta_rmst']:+.2f} "
                     f"[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] p={d['p_two_sided']}")
    out["part3_rmst_diff"] = part3

    # ---- risk-group K-M figure (H2) ----
    if part2 and part2["models"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150, sharey=True)
        for ax, method in zip(axes, ("ERM", "DANN")):
            ckpt = (RESULTS_DIR / "experiments" / "exp_j" / f"exp_A_{method}"
                    / "seed_42" / "best_model.pth")
            model = TransDANNSurvV3(
                num_continuous=len(data["cont_features"]),
                num_categorical=len(data["cat_features"]),
                cat_cardinalities=data["cat_cardinalities"],
                d_model=128, n_heads=8, n_layers=4, dropout=0.15,
                num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32,
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                             weights_only=True))
            model.eval()
            with torch.no_grad():
                surv_out, _, _ = model(t_t.to(DEVICE), c_t.to(DEVICE), alpha=0.0)
                risks = deep_hit_risk(surv_out, data["bin_centers"]).cpu().numpy()
            med = float(np.median(risks))
            for label, m in (("High risk", risks >= med), ("Low risk", risks < med)):
                kmf = KaplanMeierFitter()
                kmf.fit(tcga_times[m], event_observed=tcga_events[m], label=label)
                kmf.plot_survival_function(ax=ax, ci_show=True)
            p = part2["models"][method]["logrank_p"]
            ax.set_title(f"{method} risk groups on TCGA dev\nlog-rank p = {p:.3g}")
            ax.set_xlabel("months")
            if method == "ERM":
                ax.set_ylabel("survival probability")
            ax.legend(fontsize=8)
        fig.suptitle("K-M risk-group stratification on TCGA (Exp A models, "
                     "in-sample descriptive)", fontsize=11)
        fig.tight_layout()
        out_fig = RESULTS_DIR / "figures" / "extended" / "H2_km_tcga_risk_groups.png"
        fig.savefig(out_fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info(f"Figure → {out_fig}")

    # ---- save ----
    with open(EXP_OUT / "km_tcga_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # table19 (stage strata + rmst diff)
    rows = stage_rows + rmst_rows
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "table19_km_tcga.csv",
                              index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table19_km_tcga.csv'} "
             f"({len(rows)} rows)")

    # ---- summary.txt ----
    lines = ["\n" + "=" * 90, "TCGA COHORT CLINICAL CHARACTERISATION",
             "=" * 90]
    lines.append("\n[1] Clinical Stage strata K-M on TCGA:")
    lines.append(f"    stage-group n/events: " + ", ".join(
        f"{g}={part1['groups'][g]['n']}/{part1['groups'][g]['events']}"
        for g in part1["groups"]))
    lines.append(f"    log-rank p = {part1['logrank_p']}")
    if part2:
        lines.append("\n[2] Risk-group K-M (median split, TCGA dev):")
        for method, v in part2["models"].items():
            lines.append(f"    {method:<5} high={v['n_high']} low={v['n_low']} "
                         f"log-rank p={v['logrank_p']}")
        lines.append("    (in-sample, descriptive; TCGA was in training)")
    lines.append("\n[3] RMST@36 difference vs TCGA (bootstrap 95% CI, p):")
    for src, v in part3["pairs"].items():
        d = v["delta_tcga_minus_other"]
        lines.append(f"    TCGA−{src:<16} {d['delta_rmst']:+.2f} "
                     f"[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] p={d['p_two_sided']}"
                     if d else f"    TCGA−{src}: n/a")
    lines.append("=" * 90)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {EXP_OUT}/km_tcga_results.json, summary.txt, table19")
    log.info("\n✅ TCGA K-M analysis complete!")


if __name__ == "__main__":
    main()
