#!/usr/bin/env python3
"""
27_exp_rmst.py; Restricted mean survival time (RMST) analysis
==============================================================

Two complementary parts:

(1) Outcome-distribution shift quantification (Observed RMST)
    For each cohort, compute the restricted mean survival time at
    τ = 12/24/36/48/60 months from KM survival curves,
    RMST(τ) = ∫₀^τ S(t)dt, with bootstrap SE (within-cohort resampling, B=200).
    For all A/B/C cohorts.
    Purpose: translate the "population/outcome distribution is the fundamental
    domain fingerprint" argument into a single clinical number; the RMST
    difference between TCGA and SEER (e.g. "SEER RMST@36 is X months lower than
    TCGA") directly quantifies the outcome-distribution shift, complementing the
    single-number views of Exp I (domain shift quantification) and Exp H (K-M curves).

(2) Model RMST calibration (Predicted vs Observed, experiment A validation set)
    Use Exp J's ERM/DANN/CORAL/IRM (seed 42) to predict survival curves on the A
    validation set, compute predicted RMST@36/60, and compare against the per-cohort
    KM-observed RMST on the same validation set. Test: does DANN (GRL) improve the
    predicted calibration of the target cohort TCGA on the RMST scale (expected:
    no improvement / parity, exactly as on C-index; cross-population prediction
    failure is a robust conclusion on the RMST scale).

Outputs (results/experiments/exp_rmst/):
  - rmst_results.json / summary.txt
  - results/tables/table14_rmst.csv
  - results/figures/extended/RMST1_predicted_vs_observed.png
  - logs/27_exp_rmst.log

Usage:
  python3 scripts/27_exp_rmst.py          # full (observed + model)
  python3 scripts/27_exp_rmst.py --no-models   # observed part only
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_rmst"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "27_exp_rmst.log", mode="w"),
    ],
)
log = logging.getLogger("rmst")

spec = importlib.util.spec_from_file_location(
    "train_engine", SCRIPTS_DIR / "02_train_dann_lihc.py"
)
train_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_engine)
EXPERIMENTS = train_engine.EXPERIMENTS

from transdann_utils import (  # noqa: E402
    DEVICE, TransDANNSurvV3, deep_hit_risk,
)
from lihc_recon import build_data, CONT_FEATURES, CAT_FEATURES  # noqa: E402
from method_utils import (  # noqa: E402
    deephit_survival_probabilities, survival_at_times,
)

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
TAUS = [12, 24, 36, 48, 60]
BOOT_SE_TAUS = [36, 60]
N_BOOT = 200
EXP_KEY = {"A": "A_tcga_vs_seer", "B": "B_tcga_vs_external", "C": "C_all_cohorts"}
METHODS = ["ERM", "DANN", "CORAL", "IRM"]


# ===================================================================
# 1. KM RMST
# ===================================================================
def km_curve_rmst(kt, ks, tau):
    """Area under the right-continuous KM step function up to tau."""
    t = np.concatenate([[0.0], kt])
    s = np.concatenate([[1.0], ks])
    m = int(np.searchsorted(t, tau, side="right"))  # first index with t > tau
    if m == 0:
        return float(tau)
    # intervals [t[i], t[i+1]) height s[i] for i < m-1; last [t[m-1], tau] height s[m-1]
    ends = np.concatenate([t[1:m], [tau]])
    area = float(np.sum(s[:m] * (ends - t[:m])))
    return area


def km_rmst_with_se(times, events, tau, n_boot=N_BOOT, seed=7):
    """KM RMST(τ) on the whole sample + bootstrap SE (within-sample resampling)."""
    from sksurv.nonparametric import kaplan_meier_estimator
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)

    def _once(idx):
        kt, ks = kaplan_meier_estimator(events[idx], times[idx])
        if len(kt) == 0:
            return tau if idx.sum() > 0 else 0.0
        return km_curve_rmst(kt, ks, tau)

    point = _once(np.arange(len(times)))
    rng = np.random.RandomState(seed)
    boots = []
    n = len(times)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            boots.append(_once(idx))
        except Exception:
            pass
    se = float(np.std(boots)) if boots else None
    return round(point, 2), (round(se, 2) if se is not None else None)


def cohort_observed_rmst(df, tau):
    """Per-cohort RMST(τ) with bootstrap SE, on a dataframe (dev = train+val)."""
    out = {}
    for src, grp in df.groupby("Source"):
        if len(grp) < 5:
            continue
        point, se = km_rmst_with_se(grp["Survival_Months"].values,
                                    grp["Vital_Status"].values, tau)
        out[src] = {"rmst": point, "se": se, "n": int(len(grp))}
    return out


# ===================================================================
# 2. Predicted RMST from a DeepHit model
# ===================================================================
def predict_rmst(model, loader, bin_edges, bin_centers, taus):
    """Return predicted RMST@taus per val sample: (n_val, len(taus))."""
    model.eval()
    bin_edges = bin_edges.to(DEVICE)
    bin_centers_np = bin_centers.cpu().numpy()
    # bin widths from centers
    widths = np.diff(np.concatenate([[0.0], (bin_edges[:-1] + bin_edges[1:]).cpu().numpy() / 2.0]))
    # simpler: widths of each bin centre interval
    centers = bin_centers_np
    half = np.diff(np.concatenate([[0.0], (centers[1:] + centers[:-1]) / 2.0, [centers[-1] + (centers[-1] - centers[-2])]]))
    rows = []
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in loader:
            surv_out, _, _ = model(x_cont.to(DEVICE), x_cat.to(DEVICE), alpha=0.0)
            surv, _centers = deephit_survival_probabilities(surv_out, bin_edges)
            S = surv.cpu().numpy()  # (B, n_bins) S at bin centers
            r = np.zeros((S.shape[0], len(taus)))
            for j, tau in enumerate(taus):
                # predicted RMST@tau = Σ_{center_k ≤ tau} S(center_k)·width_k
                mask = centers <= tau
                if mask.sum() == 0:
                    r[:, j] = 0.0
                    continue
                r[:, j] = S[:, mask] @ half[mask]
                last_c = centers[mask][-1]
                if last_c < tau:
                    r[:, j] += S[:, mask][:, -1] * (tau - last_c)
            rows.append(r)
    return np.concatenate(rows, axis=0)


def load_expA_data():
    t02 = train_engine
    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    exp_cfg = t02.EXPERIMENTS["A_tcga_vs_seer"]
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    return build_data(exp_df, CONT_FEATURES, CAT_FEATURES,
                      surv_type="deephit", n_bins=32, subsample_seer=10000)


def load_expj_model(method, data):
    model_path = (RESULTS_DIR / "experiments" / "exp_j"
                  / f"exp_A_{method}" / "seed_42" / "best_model.pth")
    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=128, n_heads=8, n_layers=4, dropout=0.15,
        num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def km_rmst_simple(times, events, tau):
    """KM RMST on a val subset (no SE; small n, honest single number)."""
    from sksurv.nonparametric import kaplan_meier_estimator
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    kt, ks = kaplan_meier_estimator(events, times)
    if len(kt) == 0:
        return float(tau)
    return km_curve_rmst(kt, ks, tau)


# ===================================================================
# 3. Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="RMST analysis")
    parser.add_argument("--no-models", action="store_true",
                        help="skip the Exp-J model RMST part (observed only)")
    parser.add_argument("--taus", nargs="+", type=int, default=TAUS)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    log.info(f"Loaded {len(df)} rows, {df['Source'].nunique()} cohorts")

    # ---------- (1) observed RMST across all cohorts ----------
    observed = {}
    for letter, ek in EXP_KEY.items():
        exp_cfg = EXPERIMENTS[ek]
        exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
        # full dev = train+val (build_data keeps the same SEER subsample); reuse it
        data = build_data(exp_df, CONT_FEATURES, CAT_FEATURES,
                          surv_type="deephit", n_bins=32, subsample_seer=10000)
        dev = pd.concat([data["train_df"], data["val_df"]], ignore_index=True)
        for tau in args.taus:
            per_coh = cohort_observed_rmst(dev, tau)
            observed.setdefault(ek, {})[str(tau)] = per_coh
        log.info(f"[Observed] {ek}: RMST@36 = "
                 + ", ".join(f"{c}={v['rmst']}" for c, v in
                             observed[ek]["36"].items()))

    # ---------- (2) model RMST on A val ----------
    model_part = None
    if not args.no_models:
        data = load_expA_data()
        val_df = data["val_df"]
        sources = val_df["Source"].values
        model_part = {"eval_population": "Exp A val (TCGA_LIHC vs US_SEER)",
                      "methods": METHODS, "taus": [36, 60], "per_method": {}}
        for method in METHODS:
            model = load_expj_model(method, data)
            pred = predict_rmst(model, data["val_loader"], data["bin_edges"],
                                data["bin_centers"], [36, 60])
            per_cohort = {}
            for src in sorted(set(sources)):
                m = sources == src
                if m.sum() < 5:
                    continue
                obs = km_rmst_simple(val_df.loc[m, "Survival_Months"].values,
                                     val_df.loc[m, "Vital_Status"].values, 36)
                per_cohort[src] = {
                    "pred_36": round(float(np.mean(pred[m, 0])), 2),
                    "pred_60": round(float(np.mean(pred[m, 1])), 2),
                    "obs_36_km": round(obs, 2),
                    "n_val": int(m.sum()),
                }
            model_part["per_method"][method] = per_cohort
            log.info(f"[Model] {method}: " + ", ".join(
                f"{c} pred36={v['pred_36']} obs36={v['obs_36_km']}"
                for c, v in per_cohort.items()))

    out = {
        "experiment": "rmst",
        "taus": args.taus,
        "observed_rmst_per_cohort": observed,
        "model_rmst_a_val": model_part,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXP_OUT / "rmst_results.json", "w") as f:
        json.dump(out, f, indent=2)

    write_table(out, args.taus)
    write_figure(out, args.taus)

    # ---- summary.txt ----
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("RMST ANALYSIS  (tau = %s)" % args.taus)
    lines.append("=" * 90)
    lines.append("\n[1] Observed KM RMST(τ) per cohort (bootstrap SE):")
    for ek in EXP_KEY.values():
        lines.append(f"\n  {ek}:")
        for tau in args.taus:
            parts = []
            for c, v in observed[ek][str(tau)].items():
                se = f"±{v['se']}" if v["se"] is not None else ""
                parts.append(f"{c}={v['rmst']}{se}")
            lines.append(f"    RMST@{tau:<3}" + ", ".join(parts))
    if model_part:
        lines.append("\n[2] Model predicted vs KM-observed RMST (Exp A val):")
        lines.append(f"    {'method':<7}" + "".join(f"{c:>26}" for c in
                      sorted(model_part["per_method"]["ERM"].keys())))
        for method in METHODS:
            parts = []
            for c, v in model_part["per_method"][method].items():
                parts.append(f"pred36={v['pred_36']:>5} obs={v['obs_36_km']:>5}")
            lines.append(f"    {method:<7}" + "".join(f"{p:>26}" for p in parts))
    lines.append("=" * 90)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {EXP_OUT}/rmst_results.json, summary.txt")
    log.info("\n✅ RMST analysis complete!")


def write_table(out, taus):
    rows = []
    # observed part
    for ek in EXP_KEY.values():
        for tau in taus:
            for c, v in out["observed_rmst_per_cohort"][ek][str(tau)].items():
                rows.append({"block": "observed", "experiment": ek,
                             "cohort": c, "tau": tau, "rmst": v["rmst"],
                             "se": v["se"], "n": v["n"]})
    # model part
    mp = out.get("model_rmst_a_val")
    if mp:
        for method in METHODS:
            for c, v in mp["per_method"][method].items():
                rows.append({"block": "model_A_val", "experiment": "A_tcga_vs_seer",
                             "cohort": c, "tau": 36, "method": method,
                             "pred_rmst": v["pred_36"], "obs_km": v["obs_36_km"],
                             "n_val": v["n_val"]})
                rows.append({"block": "model_A_val", "experiment": "A_tcga_vs_seer",
                             "cohort": c, "tau": 60, "method": method,
                             "pred_rmst": v["pred_60"], "obs_km": None,
                             "n_val": v["n_val"]})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "table14_rmst.csv", index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table14_rmst.csv'} ({len(rows)} rows)")


def write_figure(out, taus):
    mp = out.get("model_rmst_a_val")
    if not mp:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cohorts = sorted(mp["per_method"]["ERM"].keys())
    colors = {"ERM": "#4c72b0", "DANN": "#c44e52", "CORAL": "#55a868", "IRM": "#8172b2"}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(cohorts))
    w = 0.18
    for j, method in enumerate(METHODS):
        vals = [mp["per_method"][method][c]["pred_36"] for c in cohorts]
        ax.bar(x + (j - 1.5) * w, vals, w, label=method, color=colors[method])
    obs = [mp["per_method"]["ERM"][c]["obs_36_km"] for c in cohorts]
    ax.plot(x, obs, "k--o", ms=5, lw=1.5, label="KM observed RMST@36")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, fontsize=8)
    ax.set_ylabel("RMST@36 (months)")
    ax.set_title("Predicted vs KM-observed RMST@36 on Exp A val, per cohort")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_fig = RESULTS_DIR / "figures" / "extended" / "RMST1_predicted_vs_observed.png"
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    log.info(f"Figure → {out_fig}")


if __name__ == "__main__":
    main()
