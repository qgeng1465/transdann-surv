#!/usr/bin/env python3
"""
12_exp_k_clinical_metrics.py; Experiment K: clinical evaluation metrics (Clinical Metrics)
================================================================================
AIM-appropriate clinical evaluation on the Experiment-A validation population
(TCGA_LIHC vs US_SEER), using the already-trained ERM / DANN / CORAL / IRM
models from Experiment J (seed 42).

Metrics
-------
  1. Integrated Brier Score (IBS); vs the Kaplan–Meier null model
  2. Time-dependent Brier-score curves at 6/12/24/36/48/60 months (per cohort)
  3. Calibration curves + Expected Calibration Error (ECE) at 12/36/60 months
  4. Decision-curve analysis (IPCW net benefit) at 36 months vs Treat-All/None

Outputs
-------
  results/experiments/exp_k/clinical_metrics.json
  results/tables/table4_clinical_metrics.csv
  results/logs/exp_k_training.log
  results/figures/extended/K1_ibs_comparison.png
  results/figures/extended/K2_brier_score_curves.png
  results/figures/extended/K3_calibration_curves.png
  results/figures/extended/K4_decision_curve_analysis.png
"""

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from lifelines.utils import concordance_index

from lihc_recon import BASE_DIR, build_data, _load_train_02
from transdann_utils import TransDANNSurvV3, deep_hit_risk, DEVICE
from method_utils import deephit_survival_probabilities, survival_at_times

RESULTS_DIR = BASE_DIR / "results"
EXP_OUT = RESULTS_DIR / "experiments" / "exp_k"
TABLE_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
for d in (EXP_OUT, TABLE_DIR, LOG_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

METHODS = ["ERM", "DANN", "CORAL", "IRM"]
EVAL_TIMES = [12, 36, 60]
BRIER_TIMES = [6, 12, 24, 36, 48, 60]
DCA_TIME = 36


def _load_exp_j():
    spec = importlib.util.spec_from_file_location(
        "exp_j_mod", str(Path(__file__).resolve().parent / "11_exp_j_method_comparison.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPJ = _load_exp_j()
logger = EXPJ.setup_logger("exp_k")


# ===================================================================
# Data + model loading
# ===================================================================
def load_expA_data():
    t02 = _load_train_02()
    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    exp_cfg = t02.EXPERIMENTS["A_tcga_vs_seer"]
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    data = build_data(exp_df, ["Age"], ["Sex", "Stage", "Grade"],
                      surv_type="deephit", n_bins=32, subsample_seer=10000)
    return data


def load_model(method, data):
    model_path = (RESULTS_DIR / "experiments" / "exp_j"
                  / f"exp_A_{method}" / "seed_42" / "best_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found; run Experiment J first")
    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=128, n_heads=8, n_layers=4, dropout=0.15,
        num_domains=data["num_domains"],
        surv_head_type="deephit", n_bins=32,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict_risk_and_surv(model, data):
    """Return (risks, surv_probs) on the validation set."""
    model.eval()
    val_df = data["val_df"]
    times = val_df["Survival_Months"].values.astype(float)
    events = val_df["Vital_Status"].values.astype(float)
    domains = val_df["Domain_Label"].values.astype(int)
    sources = val_df["Source"].values

    risks, probs = [], []
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in data["val_loader"]:
            surv_out, _, _ = model(x_cont.to(DEVICE), x_cat.to(DEVICE), alpha=0.0)
            risks.append(deep_hit_risk(surv_out, data["bin_centers"]).cpu().numpy())
            surv, centers = deephit_survival_probabilities(surv_out, data["bin_edges"].to(DEVICE))
            S = survival_at_times(surv, centers,
                                  torch.tensor(BRIER_TIMES + [DCA_TIME] + EVAL_TIMES,
                                               dtype=torch.float32, device=DEVICE))
            probs.append(S.cpu().numpy())
    risks = np.concatenate(risks)
    probs = np.concatenate(probs)
    return times, events, domains, sources, risks, probs


def survival_probs_at_times(model, data, grid):
    """S(t|x) at an arbitrary time grid on the val set (n_test, n_grid)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in data["val_loader"]:
            surv_out, _, _ = model(x_cont.to(DEVICE), x_cat.to(DEVICE), alpha=0.0)
            surv, centers = deephit_survival_probabilities(surv_out, data["bin_edges"].to(DEVICE))
            S = survival_at_times(surv, centers, torch.tensor(grid, dtype=torch.float32, device=DEVICE))
            probs.append(S.cpu().numpy())
    return np.concatenate(probs)


# ===================================================================
# 1. Integrated Brier Score (vs KM null model)
# ===================================================================
def compute_ibs(model, data, method):
    from sksurv.metrics import integrated_brier_score
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.util import Surv
    train_times = data["train_df"]["Survival_Months"].values
    train_events = data["train_df"]["Vital_Status"].values.astype(bool)
    val_times = data["val_df"]["Survival_Months"].values
    val_events = data["val_df"]["Vital_Status"].values.astype(bool)

    lo = max(float(np.min(train_times)), float(np.min(val_times)), 0.1)
    hi = min(float(np.quantile(train_times, 0.9)), float(np.max(val_times)) * 0.999)
    if hi <= lo:
        return None, None
    grid = np.linspace(lo, hi, 40)
    probs = survival_probs_at_times(model, data, grid)

    y_train = Surv.from_arrays(train_events, train_times)
    y_test = Surv.from_arrays(val_events, val_times)
    try:
        ibs = float(integrated_brier_score(y_train, y_test, probs, grid))
    except Exception as e:
        logger.warning(f"[K][{method}] IBS failed: {e}")
        ibs = None

    # KM null model
    kt, ks = kaplan_meier_estimator(train_events, train_times)
    S_null = np.interp(grid, kt, ks)
    null_est = np.tile(S_null, (len(val_times), 1))
    try:
        ibs_null = float(integrated_brier_score(y_train, y_test, null_est, grid))
    except Exception:
        ibs_null = None
    return ibs, ibs_null


# ===================================================================
# 2. Time-dependent Brier curves (per cohort)
# ===================================================================
def compute_brier_curves(model, data, method):
    from sksurv.metrics import brier_score
    from sksurv.util import Surv
    train_times = data["train_df"]["Survival_Months"].values
    train_events = data["train_df"]["Vital_Status"].values.astype(bool)
    val_df = data["val_df"]
    val_times = val_df["Survival_Months"].values
    val_events = val_df["Vital_Status"].values.astype(bool)
    sources = val_df["Source"].values

    grid = np.array(BRIER_TIMES, dtype=float)
    grid = grid[(grid > 0) & (grid < np.max(val_times) * 0.95)]
    probs = survival_probs_at_times(model, data, grid)
    y_train = Surv.from_arrays(train_events, train_times)
    y_test = Surv.from_arrays(val_events, val_times)
    out_t, out_bs = brier_score(y_train, y_test, probs, grid)
    per_cohort = {}
    for s in sorted(set(sources)):
        m = sources == s
        if m.sum() < 30:
            continue
        y_sub = Surv.from_arrays(val_events[m], val_times[m])
        try:
            _, bs_sub = brier_score(y_train, y_sub, probs[m], grid)
            per_cohort[s] = np.asarray(bs_sub).tolist()
        except Exception:
            per_cohort[s] = None
    return out_t, np.asarray(out_bs), per_cohort


# ===================================================================
# 3. Calibration curves + ECE at 12/36/60 months
# ===================================================================
def compute_calibration(model, data, method):
    from sksurv.nonparametric import kaplan_meier_estimator
    val_df = data["val_df"]
    times = val_df["Survival_Months"].values
    events = val_df["Vital_Status"].values.astype(bool)
    out = {}
    for t_eval in EVAL_TIMES:
        # predicted risk r = 1 - S(t)
        r = 1.0 - survival_probs_at_times(model, data, [t_eval])[:, 0]
        # deciles
        dec = pd.qcut(pd.Series(r), 10, labels=False, duplicates="drop")
        cal = []
        for d in range(int(dec.max()) + 1):
            m = dec.values == d
            if m.sum() < 10:
                continue
            pred = float(np.mean(r[m]))
            # observed event rate by time t within decile (KM)
            kt, ks = kaplan_meier_estimator(events[m], times[m])
            idx = np.searchsorted(kt, t_eval, side="right") - 1
            obs = 1.0 - ks[max(idx, 0)] if 0 <= idx < len(ks) else 1.0
            cal.append({"decile": int(d), "n": int(m.sum()),
                        "predicted": round(pred, 4), "observed": round(float(obs), 4)})
        ece = float(np.mean([abs(c["predicted"] - c["observed"]) for c in cal]))
        out[str(t_eval)] = {"calibration": cal, "ece": round(ece, 4)}
    return out


# ===================================================================
# 4. Decision-curve analysis (IPCW net benefit at 36 months)
# ===================================================================
def compute_dca(model, data, method):
    from sksurv.nonparametric import kaplan_meier_estimator
    val_df = data["val_df"]
    times = val_df["Survival_Months"].values
    events = val_df["Vital_Status"].values.astype(bool)
    t = DCA_TIME

    # IPCW weights: for cases (event ≤ t) w = 1/G(T), else w = 1
    cens = ~events
    kt_c, ks_c = kaplan_meier_estimator(cens, times)
    G = lambda tt: np.interp(min(tt, kt_c[-1]), kt_c, ks_c)
    case = (events & (times <= t))
    w = np.ones(len(times))
    for i in np.where(case)[0]:
        w[i] = 1.0 / max(G(times[i]), 0.05)

    r = 1.0 - survival_probs_at_times(model, data, [t])[:, 0]
    thresholds = np.linspace(0.01, 0.99, 50)
    nb = []
    for p in thresholds:
        tp = (r >= p) & case
        fp = (r >= p) & ~case
        nb.append((w[tp].sum() - (p / (1 - p)) * w[fp].sum()) / len(r))
    return thresholds, np.array(nb)


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("EXPERIMENT K; Clinical Metrics (on Experiment-A val population)")
    print("=" * 70)
    data = load_expA_data()

    results = {"eval_population": "Exp A val (TCGA_LIHC vs US_SEER)",
               "methods": METHODS, "eval_times": EVAL_TIMES,
               "brier_times": BRIER_TIMES, "dca_time": DCA_TIME,
               "models": {}}
    for method in METHODS:
        logger.info(f"[K] loading model {method} ...")
        try:
            model = load_model(method, data)
        except FileNotFoundError as e:
            logger.error(f"[K] {e}")
            results["models"][method] = {"error": str(e)}
            continue
        ibs, ibs_null = compute_ibs(model, data, method)
        b_times, b_bs, b_cohort = compute_brier_curves(model, data, method)
        cal = compute_calibration(model, data, method)
        d_thr, d_nb = compute_dca(model, data, method)

        times, events, domains, sources, risks, _ = predict_risk_and_surv(model, data)
        try:
            cindex = round(float(concordance_index(times, -risks, events)), 4)
        except Exception:
            cindex = None

        results["models"][method] = {
            "cindex": cindex,
            "ibs": ibs,
            "ibs_km_null": ibs_null,
            "brier_curve": {"times": b_times.tolist(), "brier": [round(float(x), 4) for x in b_bs],
                            "per_cohort": b_cohort},
            "calibration": cal,
            "dca": {"thresholds": [round(float(x), 4) for x in d_thr],
                    "net_benefit": [round(float(x), 6) for x in d_nb]},
        }
        logger.info(f"[K] {method}: C-index={cindex} IBS={ibs} "
                    f"(KM-null={ibs_null}) ECE@12/36/60="
                    f"{[cal[str(k)]['ece'] for k in EVAL_TIMES]}")

    with open(EXP_OUT / "clinical_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # table4
    rows = []
    for method in METHODS:
        m = results["models"].get(method) or {}
        rows.append({
            "method": method,
            "cindex": m.get("cindex"),
            "ibs": m.get("ibs"),
            "ibs_km_null": m.get("ibs_km_null"),
            "ece_12": (m.get("calibration") or {}).get("12", {}).get("ece"),
            "ece_36": (m.get("calibration") or {}).get("36", {}).get("ece"),
            "ece_60": (m.get("calibration") or {}).get("60", {}).get("ece"),
        })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table4_clinical_metrics.csv", index=False)

    # Treat-all / treat-none reference for DCA (from ERM's counts; same population)
    _make_figures(results, data)

    print(f"\n✅ Experiment K complete → {EXP_OUT / 'clinical_metrics.json'}")
    for method in METHODS:
        m = results["models"].get(method) or {}
        if "ibs" in m and m["ibs"] is not None:
            print(f"  {method:6s}: C-index={m['cindex']}  IBS={m['ibs']}  "
                  f"KM-null IBS={m['ibs_km_null']}")


# ===================================================================
# Figures
# ===================================================================
def _make_figures(results, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ERM": "#4c72b0", "DANN": "#c44e52", "CORAL": "#55a868", "IRM": "#8172b2"}

    # K1: IBS bar chart (model vs KM null)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ibss = []
    for method in METHODS:
        m = results["models"].get(method) or {}
        ibss.append(m.get("ibs"))
    x = np.arange(len(METHODS))
    bars = ax.bar(x, ibss, 0.5, color=[colors[m] for m in METHODS])
    for xi, v in zip(x, ibss):
        if v is not None:
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    null_ref = None
    for method in METHODS:
        m = results["models"].get(method) or {}
        if m.get("ibs_km_null"):
            null_ref = m["ibs_km_null"]; break
    if null_ref:
        ax.axhline(null_ref, color="k", ls="--", lw=1.2, label=f"KM null model = {null_ref:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(METHODS)
    ax.set_ylabel("Integrated Brier Score")
    ax.set_ylim(0, max([v for v in ibss if v is not None] or [0.2]) * 1.25 + 0.01)
    ax.set_title("Experiment K; IBS (lower better)")
    if null_ref:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "K1_ibs_comparison.png", dpi=200)
    plt.close(fig)

    # K2: Brier curves (overall + per cohort)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for method in METHODS:
        m = results["models"].get(method) or {}
        bc = m.get("brier_curve")
        if bc:
            ax.plot(bc["times"], bc["brier"], "o-", ms=4, label=method, color=colors[method])
    ax.set_xlabel("months"); ax.set_ylabel("Brier score")
    ax.set_title("Overall time-dependent Brier score")
    ax.legend(fontsize=7)
    ax = axes[1]
    sources = sorted(set(data["val_df"]["Source"].values))
    for si, src in enumerate(sources):
        for method in METHODS:
            m = results["models"].get(method) or {}
            bc = m.get("brier_curve")
            if bc and bc.get("per_cohort") and bc["per_cohort"].get(src):
                ax.plot(bc["times"], bc["per_cohort"][src], marker="o", ms=3,
                        ls="-", color=colors[method], alpha=0.85,
                        label=f"{method} ({src})" if method == "ERM" else None)
    ax.set_xlabel("months"); ax.set_ylabel("Brier score")
    ax.set_title("Per-cohort Brier score")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, fontsize=6)
    fig.suptitle("Experiment K; time-dependent Brier scores", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "K2_brier_score_curves.png", dpi=200)
    plt.close(fig)

    # K3: calibration curves (12/36/60)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for k, t_eval in enumerate(EVAL_TIMES):
        ax = axes[k]
        for method in METHODS:
            m = results["models"].get(method) or {}
            cal = (m.get("calibration") or {}).get(str(t_eval), {}).get("calibration")
            if cal:
                ax.plot([c["predicted"] for c in cal], [c["observed"] for c in cal],
                        "o-", ms=4, label=method, color=colors[method])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("predicted event rate")
        ax.set_ylabel("observed event rate")
        ax.set_title(f"Calibration @ {t_eval} mo")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Experiment K; calibration curves (ECE labels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "K3_calibration_curves.png", dpi=200)
    plt.close(fig)

    # K4: DCA at 36 months
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for method in METHODS:
        m = results["models"].get(method) or {}
        dca = m.get("dca")
        if dca:
            ax.plot(dca["thresholds"], dca["net_benefit"], label=method, color=colors[method])
    # Treat-all / treat-none references (IPCW)
    val_df = data["val_df"]
    times = val_df["Survival_Months"].values
    events = val_df["Vital_Status"].values.astype(bool)
    t = DCA_TIME
    from sksurv.nonparametric import kaplan_meier_estimator
    cens = ~events
    kt_c, ks_c = kaplan_meier_estimator(cens, times)
    G = lambda tt: np.interp(min(tt, kt_c[-1]), kt_c, ks_c)
    case = (events & (times <= t))
    n = len(times)
    nb_all = float(np.sum([1.0 / max(G(times[i]), 0.05) for i in np.where(case)[0]]) / n)
    # Treat-none net benefit is 0 by definition
    thr = np.linspace(0.01, 0.99, 50)
    ax.plot(thr, np.full_like(thr, nb_all), "k--", lw=1, label="Treat all")
    ax.plot(thr, np.zeros_like(thr), "k:", lw=1, label="Treat none")
    ax.set_xlabel("threshold probability (36-month risk)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Experiment K; Decision Curve Analysis @ 36 months")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "K4_decision_curve_analysis.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
