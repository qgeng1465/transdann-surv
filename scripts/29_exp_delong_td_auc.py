#!/usr/bin/env python3
"""
29_exp_delong_td_auc.py; DeLong test for correlated cumulative/dynamic AUCs
============================================================================

Reviewer risk: the previously reported time-dependent AUC (Exp J, at 12/36/60
months) has only point estimates; no significance test was done on "the AUC
difference between two models on the same population". The DeLong test is the
standard approach.

Design (rigorous logic, consistent definition)
----------------------------------------------
1. **AUC definition exactly as in the paper**: cumulative/dynamic C/D AUC @ τ,
   case = {T ≤ τ, δ=1} (event observed before τ), control = {T > τ} (survived past τ);
   subjects censored before τ are excluded (definition identical to
   `transdann_utils.time_dependent_auc`).
2. **DeLong test** (DeLong et al. 1988, U-statistic variance method):
   for two models (A, B) on the same subjects at the same τ,
   z = (AUC_A − AUC_B)/√(Var_A + Var_B − 2·Cov), two-sided p.
   The covariance is estimated via the V10/V01 structural components; the standard
   test for "the difference of two correlated AUCs".
3. **Paired stratified bootstrap cross-validation** (B=300/seed, within-cohort
   resampling, merged across seeds):
   uses the resampling distribution to re-test whether ΔAUC contains 0; a robust
   fallback against censoring.
   If the DeLong p and the bootstrap p agree, the conclusion is more stable.
4. Four parts:
   - Part A: DANN vs Baseline, A/B/C validation sets, τ ∈ {12,24,36,48,60}, 3 seeds
     (one DeLong p per seed; bootstrap p merged across seeds).
   - Part B: all 9 methods vs ERM at τ=36 with DeLong (reported merged across 3 seeds),
     + Mixup vs ERM at 12/36/60 (Mixup is the only method significantly positive
     after BH-FDR).
   - Part C: **true independent test set** (Exp IT, 3 folds × 3 seeds × {baseline, dann}),
     merge the 3-fold test sets by seed into one block of subjects that never took
     part in training/early-stopping/selection,
     then run DANN vs Baseline with DeLong; the strictest "out-of-sample AUC
     significance".
   - (RMST difference significance testing is covered in scripts/32_exp_km_tcga.py)

Honesty statement (avoiding over-interpretation):
  - each per-seed DeLong p is a conditional test "on the two risk-score sets
    corresponding to that training seed", not a marginal test treating the seed as
    a random variable; merging across seeds via bootstrap gives a quantity closer
    to the marginal one.
  - the case/control split at fixed τ does not change across models, so DeLong's
    correlation structure is valid;
    but C/D AUC itself ignores pre-τ censoring (excluded), a naive definition
    consistent with the paper.

Outputs (results/experiments/exp_delong_td_auc/):
  - part_a_dann_vs_base.json / part_b_methods.json / part_c_independent_test.json
  - runs.json (all per-run detail)
  - summary.txt
  - results/tables/table16_delong_td_auc.csv
  - results/figures/extended/D1_delong_dann_vs_base_forest.png
  - logs/29_exp_delong_td_auc.log

Usage:
  python3 scripts/29_exp_delong_td_auc.py
  python3 scripts/29_exp_delong_td_auc.py --no-bootstrap    # skip bootstrap (fast)
  python3 scripts/29_exp_delong_td_auc.py --parts A B       # run A/B only
  python3 scripts/29_exp_delong_td_auc.py --parts C         # run C only (true independent test set)
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
import scipy.stats
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_delong_td_auc"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "29_exp_delong_td_auc.log", mode="w"),
    ],
)
log = logging.getLogger("delong")

# ---------------------------------------------------------------------------
# Reuse: train engine for EXPERIMENTS / build_data, and script-21 fold helpers
# ---------------------------------------------------------------------------
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

spec21 = importlib.util.spec_from_file_location(
    "exp_it_21", SCRIPTS_DIR / "21_exp_independent_test.py"
)
exp_it21 = importlib.util.module_from_spec(spec21)
spec21.loader.exec_module(exp_it21)
make_folds = exp_it21.make_folds
prepare_fold = exp_it21.prepare_fold

EXP_KEY = {"A": "A_tcga_vs_seer", "B": "B_tcga_vs_external", "C": "C_all_cohorts"}
METHODS = ["ERM", "DANN", "CORAL", "IRM", "GroupDRO", "V-REx", "Fish", "Mixup", "MLDG"]
SEEDS = [42, 43, 44]
TAUS_A = [12, 24, 36, 48, 60]
TAU_MAIN = 36
N_BOOT = 300
EXPJ_DIR = RESULTS_DIR / "experiments" / "exp_j"
EXPIT_DIR = RESULTS_DIR / "experiments" / "exp_independent_test"
SUBSAMPLE_SEER = 10000


# ===================================================================
# 1. DeLong test for two correlated cumulative/dynamic AUCs at fixed τ
# ===================================================================
def delong_td_auc(scores_a, scores_b, times, events, tau):
    """DeLong test of ΔAUC = AUC(scores_a) − AUC(scores_b) at fixed tau.

    case = (t <= tau) & (e == 1); control = (t > tau).  Same C/D definition as
    transdann_utils.time_dependent_auc (naive, censored-before-tau excluded).
    Returns None if case/control too small.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    case = (t <= tau) & (e == 1)
    ctrl = t > tau
    n1, n0 = int(case.sum()), int(ctrl.sum())
    if n1 < 5 or n0 < 5:
        return None

    idx = np.concatenate([np.where(case)[0], np.where(ctrl)[0]])
    sc_a = sa[idx[:n1]]
    sc__a = sa[idx[n1:]]
    sc_b = sb[idx[:n1]]
    sc__b = sb[idx[n1:]]

    def _components(sc, sc_ctl):
        # V10 per case: fraction of controls strictly lower (+0.5 ties)
        s_c = np.sort(sc_ctl)
        n_lt = np.searchsorted(s_c, sc, side="left")
        n_le = np.searchsorted(s_c, sc, side="right")
        v10 = (n_lt + 0.5 * (n_le - n_lt)).astype(float) / n0
        # V01 per control: fraction of cases strictly higher (+0.5 ties)
        s_k = np.sort(sc)
        n_le2 = np.searchsorted(s_k, sc_ctl, side="right")
        n_lt2 = np.searchsorted(s_k, sc_ctl, side="left")
        v01 = (n1 - n_le2 + 0.5 * (n_le2 - n_lt2)).astype(float) / n1
        auc = float(np.mean(v10))
        var = float(np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n0)
        return auc, var, v10, v01

    auc_a, var_a, v10_a, v01_a = _components(sc_a, sc__a)
    auc_b, var_b, v10_b, v01_b = _components(sc_b, sc__b)
    cov = float(np.cov(v10_a, v10_b, ddof=1)[0, 1] / n1
                + np.cov(v01_a, v01_b, ddof=1)[0, 1] / n0)
    se = float(np.sqrt(max(var_a + var_b - 2.0 * cov, 0.0)))
    delta = auc_a - auc_b
    if se > 0:
        z = delta / se
        p = float(2.0 * (1.0 - scipy.stats.norm.cdf(abs(z))))
    else:
        z, p = float("nan"), float("nan")
    return {
        "tau": tau, "n_case": n1, "n_control": n0,
        "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
        "auc_a_paper": paper_consistent_auc(sa, t, e, tau),
        "auc_b_paper": paper_consistent_auc(sb, t, e, tau),
        "delta": round(delta, 4), "se": round(se, 4),
        "z": round(z, 3), "p_delong": round(p, 4),
    }


def paper_consistent_auc(scores, times, events, tau):
    """Cross-reference: the paper's reported td-AUC (transdann_utils.
    time_dependent_auc); concordance_index over the pooled case∪control set.
    Includes within-case concordant pairs, hence differs slightly from the
    U-statistic C/D AUC that the DeLong test targets.  Returns None on failure.
    """
    from lifelines.utils import concordance_index
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    s = np.asarray(scores, dtype=float)
    case = (t <= tau) & (e == 1)
    ctrl = t > tau
    if case.sum() < 5 or ctrl.sum() < 5:
        return None
    sub_t = np.concatenate([t[case], t[ctrl]])
    sub_s = np.concatenate([s[case], s[ctrl]])
    sub_e = np.concatenate([np.ones(case.sum()), np.zeros(ctrl.sum())])
    try:
        return round(float(concordance_index(sub_t, -sub_s, sub_e)), 4)
    except Exception:
        return None


def td_auc_single(scores, times, events, tau):
    """Plain C/D AUC (paper's definition) for one score vector."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    s = np.asarray(scores, dtype=float)
    case = (t <= tau) & (e == 1)
    ctrl = t > tau
    if case.sum() < 5 or ctrl.sum() < 5:
        return float("nan")
    c_ = np.sort(s[ctrl])
    n_lt = np.searchsorted(c_, s[case], side="left")
    n_le = np.searchsorted(c_, s[case], side="right")
    return float(np.mean(n_lt + 0.5 * (n_le - n_lt)) / ctrl.sum())


# ===================================================================
# 2. Paired stratified bootstrap of ΔAUC (within-cohort resampling)
# ===================================================================
def bootstrap_delta_auc(times, events, sa, sb, domains, taus, B=N_BOOT, seed=777,
                        stat="std"):
    """Bootstrap ΔAUC = AUC(sb) − AUC(sa) at each tau, within-cohort resampling.

    stat="std"   → standard C/D AUC (case vs control, DeLong-consistent).
    stat="paper" → the paper's reported concordance td-AUC (concordance_index
                   over the pooled case∪control set); a cross-check on the
                   exact numbers published in Exp J / §6.3.
    Returns {str(tau): {"delta_mean", "ci_low", "ci_high", "p_two_sided",
    "frac_pos", "n_draws"}}.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    d = np.asarray(domains, dtype=int)
    rng = np.random.RandomState(seed)
    uniq = np.unique(d)
    draws = {str(tau): [] for tau in taus}
    stat_fn = td_auc_single if stat == "std" else paper_consistent_auc
    for _ in range(B):
        idx = []
        for u in uniq:
            m = np.where(d == u)[0]
            if len(m) == 0:
                continue
            idx.append(m[rng.randint(0, len(m), size=len(m))])
        idx = np.concatenate(idx)
        for tau in taus:
            a1 = stat_fn(sa[idx], t[idx], e[idx], tau)
            a2 = stat_fn(sb[idx], t[idx], e[idx], tau)
            if not np.isnan(a1) and not np.isnan(a2):
                draws[str(tau)].append(a2 - a1)
    out = {}
    for tau, dd in draws.items():
        dd = np.asarray(dd)
        if len(dd) == 0:
            out[tau] = None
            continue
        out[tau] = {
            "delta_mean": round(float(np.mean(dd)), 4),
            "ci_low": round(float(np.percentile(dd, 2.5)), 4),
            "ci_high": round(float(np.percentile(dd, 97.5)), 4),
            "frac_pos": round(float(np.mean(dd > 0)), 4),
            "p_two_sided": round(min(2 * min(float(np.mean(dd > 0)),
                                             float(np.mean(dd < 0))), 1.0), 4),
            "n_draws": int(len(dd)),
        }
    return out


# ===================================================================
# 3. Data / risk loading helpers
# ===================================================================
def load_exp_data(letter):
    exp_cfg = EXPERIMENTS[EXP_KEY[letter]]
    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    data = build_data(exp_df, ["Age"], ["Sex", "Stage", "Grade"],
                      surv_type="deephit", n_bins=32, subsample_seer=SUBSAMPLE_SEER)
    val_df = data["val_df"]
    return data, val_df


def load_expj_risks(letter, method, seed):
    p = EXPJ_DIR / f"exp_{letter}_{method}" / f"seed_{seed}" / "risks.npy"
    if not p.exists():
        return None
    return np.load(p)


# ===================================================================
# Part A: DANN vs Baseline, A/B/C val, taus, 3 seeds
# ===================================================================
def run_part_a(do_bootstrap=True):
    out = {"taus": TAUS_A, "experiments": {}}
    for letter in ["A", "B", "C"]:
        data, val_df = load_exp_data(letter)
        times = val_df["Survival_Months"].values.astype(float)
        events = val_df["Vital_Status"].values.astype(float)
        domains = val_df["Domain_Label"].values.astype(int)
        exp_entries = {}
        for tau in TAUS_A:
            rows = []
            boot_draws = {"base": [], "dann": []}
            for seed in SEEDS:
                r_base = load_expj_risks(letter, "ERM", seed)
                r_dann = load_expj_risks(letter, "DANN", seed)
                if r_base is None or r_dann is None or len(r_base) != len(times):
                    continue
                res = delong_td_auc(r_dann, r_base, times, events, tau)
                if res is not None:
                    # delong_td_auc(a, b): auc_a=AUC(a)=DANN, auc_b=AUC(b)=ERM,
                    # delta = auc_a − auc_b = DANN − ERM.
                    rows.append({**res, "seed": seed,
                                 "auc_dann": res["auc_a"], "auc_erm": res["auc_b"],
                                 "delta_dann_minus_erm": res["delta"]})
            # bootstrap ΔAUC pooled across seeds (within-cohort resampling),
            # on BOTH the standard C/D AUC and the paper's concordance statistic
            boot = boot_paper = None
            if do_bootstrap and rows:
                acc = acc_p = None
                for seed in SEEDS:
                    rb = load_expj_risks(letter, "ERM", seed)
                    rd = load_expj_risks(letter, "DANN", seed)
                    if rb is None or rd is None:
                        continue
                    b = bootstrap_delta_auc(times, events, rb, rd, domains,
                                            [tau], B=N_BOOT, stat="std")[str(tau)]
                    bp = bootstrap_delta_auc(times, events, rb, rd, domains,
                                             [tau], B=N_BOOT, stat="paper")[str(tau)]
                    if acc is None:
                        acc, acc_p = b, bp
                    else:
                        acc["n_draws"] += b["n_draws"]
                        acc_p["n_draws"] += bp["n_draws"]
                boot, boot_paper = acc, acc_p
            exp_entries[str(tau)] = {
                "per_seed_delong": rows, "bootstrap": boot,
                "bootstrap_paper_stat": boot_paper,
            }
        out["experiments"][EXP_KEY[letter]] = exp_entries
        log.info(f"[Part A] {letter}: "
                 + "; ".join(f"τ={tau} n_seeds={len(v['per_seed_delong'])}"
                             for tau, v in exp_entries.items()))
    return out


# ===================================================================
# Part B: all 9 methods vs ERM at τ=36 (+ Mixup at 12/36/60), pooled seeds
# ===================================================================
def run_part_b(do_bootstrap=True):
    out = {"tau_main": TAU_MAIN, "experiments": {}}
    for letter in ["A", "B", "C"]:
        data, val_df = load_exp_data(letter)
        times = val_df["Survival_Months"].values.astype(float)
        events = val_df["Vital_Status"].values.astype(float)
        domains = val_df["Domain_Label"].values.astype(int)
        per_tau_rows = {str(tau): [] for tau in [12, 36, 60]}
        per_method_36 = {}
        for method in METHODS:
            if method == "ERM":
                continue
            rows36 = []
            for seed in SEEDS:
                r_erm = load_expj_risks(letter, "ERM", seed)
                r_m = load_expj_risks(letter, method, seed)
                if r_erm is None or r_m is None:
                    continue
                res = delong_td_auc(r_m, r_erm, times, events, TAU_MAIN)
                if res is not None:
                    rows36.append({**res, "seed": seed, "method": method})
            boot36 = None
            if do_bootstrap and rows36:
                acc = None
                for seed in SEEDS:
                    r_erm = load_expj_risks(letter, "ERM", seed)
                    r_m = load_expj_risks(letter, method, seed)
                    if r_erm is None or r_m is None:
                        continue
                    b = bootstrap_delta_auc(times, events, r_erm, r_m, domains,
                                            [TAU_MAIN], B=N_BOOT)[str(TAU_MAIN)]
                    if acc is None:
                        acc = b
                    else:
                        acc["n_draws"] += b["n_draws"]
            per_method_36[method] = {
                "per_seed_delong": rows36,
                "bootstrap": boot36,
                "mean_auc_delta": (round(float(np.mean([r["delta"] for r in rows36])), 4)
                                   if rows36 else None),
                "mean_p_delong": (round(float(np.mean([r["p_delong"] for r in rows36])), 4)
                                  if rows36 else None),
            }
            if method == "Mixup":
                for tau in [12, 36, 60]:
                    rows = []
                    for seed in SEEDS:
                        r_erm = load_expj_risks(letter, "ERM", seed)
                        r_m = load_expj_risks(letter, "Mixup", seed)
                        if r_erm is None or r_m is None:
                            continue
                        res = delong_td_auc(r_m, r_erm, times, events, tau)
                        if res is not None:
                            rows.append({**res, "seed": seed})
                    per_tau_rows[str(tau)] = rows
        out["experiments"][EXP_KEY[letter]] = {
            "methods_vs_erm_at_36": per_method_36,
            "mixup_vs_erm_taus": per_tau_rows,
        }
        log.info(f"[Part B] {letter}: "
                 + ", ".join(f"{m} Δ36={v['mean_auc_delta']}" for m, v in
                             per_method_36.items() if v["mean_auc_delta"] is not None))
    return out


# ===================================================================
# Part C: true independent test set, DANN vs Baseline, pooled across folds
# ===================================================================
def reconstruct_it_test(letter):
    """Rebuild the Exp-IT test population: pool the 3 disjoint test folds.

    Returns per-seed dict {seed: {times, events, domains, sources, r_base, r_dann}}.
    """
    exp_key = EXP_KEY[letter]
    exp_cfg = EXPERIMENTS[exp_key]
    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    if "US_SEER" in exp_df["Source"].values:
        seer_idx = exp_df[exp_df["Source"] == "US_SEER"].index
        if len(seer_idx) > SUBSAMPLE_SEER:
            keep = np.random.RandomState(42).choice(seer_idx, SUBSAMPLE_SEER,
                                                    replace=False)
            drop = seer_idx.difference(pd.Index(keep))
            exp_df = exp_df.drop(drop).copy()
    all_idx = exp_df.index
    folds = make_folds(exp_df, n_splits=3, seed=42)

    per_seed = {s: {"times": [], "events": [], "domains": [], "sources": [],
                    "r_base": [], "r_dann": []} for s in SEEDS}
    for fold in [0, 1, 2]:
        test_idx = folds[fold]
        dev_idx = all_idx.difference(pd.Index(test_idx)).values
        data, test_loader, sizes = prepare_fold(exp_df, dev_idx, test_idx, fold,
                                                n_bins=32)
        test_sources = exp_df.loc[test_idx]["Source"].values
        for seed in SEEDS:
            for mode in ("baseline", "dann"):
                ckpt = (EXPIT_DIR / "runs" / exp_key / f"fold_{fold}"
                        / f"seed_{seed}" / mode / "best_model.pth")
                if not ckpt.exists():
                    log.error(f"  [Part C] missing {ckpt}")
                    return None
                model = TransDANNSurvV3(
                    num_continuous=len(data["cont_features"]),
                    num_categorical=len(data["cat_features"]),
                    cat_cardinalities=data["cat_cardinalities"],
                    d_model=128, n_heads=8, n_layers=4, dropout=0.15,
                    num_domains=data["num_domains"], surv_head_type="deephit",
                    n_bins=32,
                ).to(DEVICE)
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                                 weights_only=True))
                model.eval()
                risks = []
                with torch.no_grad():
                    for x_cont, x_cat, t, e, d in test_loader:
                        surv_out, _, _ = model(x_cont.to(DEVICE), x_cat.to(DEVICE),
                                               alpha=0.0)
                        risks.append(deep_hit_risk(
                            surv_out, data["bin_centers"]).cpu().numpy())
                risks = np.concatenate(risks)
                if mode == "baseline":
                    per_seed[seed]["r_base"].extend(risks.tolist())
                else:
                    per_seed[seed]["r_dann"].extend(risks.tolist())
            # times/events/sources/domains are shared across seeds for a fold
            t_fold, e_fold, d_fold = [], [], []
            with torch.no_grad():
                for x_cont, x_cat, t, e, d in test_loader:
                    t_fold.append(t.numpy()); e_fold.append(e.numpy())
                    d_fold.append(d.numpy())
            per_seed[seed]["times"].extend(np.concatenate(t_fold).tolist())
            per_seed[seed]["events"].extend(np.concatenate(e_fold).tolist())
            per_seed[seed]["domains"].extend(np.concatenate(d_fold).tolist())
            per_seed[seed]["sources"].extend(test_sources.tolist())
    # convert to arrays and check alignment
    for seed in SEEDS:
        for k in ("times", "events", "domains", "sources", "r_base", "r_dann"):
            per_seed[seed][k] = np.asarray(per_seed[seed][k])
        if len(per_seed[seed]["r_base"]) != len(per_seed[seed]["times"]):
            log.error(f"  [Part C] {letter} seed={seed} misalignment")
            return None
    return per_seed


def run_part_c(do_bootstrap=True):
    out = {"taus": [12, 36, 60], "experiments": {}}
    for letter in ["A", "B", "C"]:
        per_seed = reconstruct_it_test(letter)
        if per_seed is None:
            log.error(f"[Part C] {letter}: reconstruction failed; skip")
            out["experiments"][EXP_KEY[letter]] = {"error": "reconstruction failed"}
            continue
        exp_entries = {}
        for tau in [12, 36, 60]:
            rows = []
            acc_boot = None
            for seed in SEEDS:
                times = per_seed[seed]["times"]
                events = per_seed[seed]["events"]
                domains = per_seed[seed]["domains"]
                res = delong_td_auc(per_seed[seed]["r_dann"],
                                    per_seed[seed]["r_base"],
                                    times, events, tau)
                if res is not None:
                    rows.append({**res, "seed": seed})
                if do_bootstrap:
                    b = bootstrap_delta_auc(times, events,
                                            per_seed[seed]["r_base"],
                                            per_seed[seed]["r_dann"],
                                            domains, [tau], B=N_BOOT)[str(tau)]
                    if acc_boot is None:
                        acc_boot = b
                    else:
                        acc_boot["n_draws"] += b["n_draws"]
            exp_entries[str(tau)] = {
                "per_seed_delong": rows,
                "bootstrap": acc_boot,
                "n_test_total": int(len(per_seed[SEEDS[0]]["times"])),
            }
        out["experiments"][EXP_KEY[letter]] = exp_entries
        log.info(f"[Part C] {letter}: n_test={exp_entries['36']['n_test_total']} "
                 + ", ".join(f"τ={tau} meanΔ36="
                             f"{np.mean([r['delta'] for r in v['per_seed_delong']]) if v['per_seed_delong'] else None:.4f}"
                             for tau, v in exp_entries.items()))
    return out


# ===================================================================
# Summarise → JSON / table16 / figure
# ===================================================================
def summarize(part_a, part_b, part_c, do_bootstrap):
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("DeLong TEST FOR TIME-DEPENDENT AUC  (DANN vs Baseline on val, "
                 f"{'bootstraps ON' if do_bootstrap else 'no bootstrap'})")
    lines.append("=" * 100)
    lines.append("  (AUC = standard C/D AUC, case vs control; the DeLong-tested "
                 "quantity; paper-consistent concordance AUCs in the CSV)")
    part_a = part_a or {}
    if "experiments" not in part_a:
        lines.append("\n  (Part A not run)")
    for ek in EXP_KEY.values():
        if "experiments" not in part_a:
            break
        exp_entries = part_a["experiments"][ek]
        lines.append(f"\n{ek}  (val set, seeds {SEEDS}):")
        lines.append(f"  {'τ':>4} {'AUC_ERM':>9} {'AUC_DANN':>9} {'Δ(D−E)':>8} "
                     f"{'DeLong p/seed':>28}  {'bootstrap p (std/paper)':>26}")
        for tau, v in exp_entries.items():
            rows = v["per_seed_delong"]
            if not rows:
                continue
            auc_erm = np.mean([r["auc_erm"] for r in rows])
            auc_dann = np.mean([r["auc_dann"] for r in rows])
            delta = np.mean([r["delta_dann_minus_erm"] for r in rows])
            ps = "[" + ", ".join(f"{r['p_delong']:.3f}" for r in rows) + "]"
            b = v["bootstrap"]
            bp = v["bootstrap_paper_stat"]
            bstr = (f"{b['p_two_sided']:.3f}/{bp['p_two_sided']:.3f}"
                    if b and bp else "; ")
            lines.append(f"  {tau:>4} {auc_erm:>9.4f} {auc_dann:>9.4f} "
                         f"{delta:>+8.4f} {ps:>28}  {bstr:>26}")
    lines.append("\n" + "=" * 100)
    lines.append("METHODS vs ERM  (DeLong ΔAUC @36, 3-seed mean p)")
    lines.append("=" * 100)
    part_b = part_b or {}
    if "experiments" not in part_b:
        lines.append("\n  (not run)")
    for ek in EXP_KEY.values():
        if "experiments" not in part_b:
            break
        d = part_b["experiments"][ek]["methods_vs_erm_at_36"]
        lines.append(f"\n{ek}: " + " | ".join(
            f"{m}={v['mean_auc_delta']:+} (p={v['mean_p_delong']})"
            for m, v in d.items() if v["mean_auc_delta"] is not None))
    lines.append("\n" + "=" * 100)
    lines.append("TRUE INDEPENDENT TEST SET  (DANN vs Baseline, pooled 3 folds)")
    lines.append("=" * 100)
    part_c = part_c or {}
    if "experiments" not in part_c:
        lines.append("\n  (not run)")
    for ek in EXP_KEY.values():
        if "experiments" not in part_c:
            break
        exp_entries = part_c["experiments"][ek]
        if "error" in exp_entries:
            lines.append(f"\n{ek}: {exp_entries['error']}")
            continue
        lines.append(f"\n{ek}  (n_test={exp_entries['36']['n_test_total']}):")
        for tau, v in exp_entries.items():
            rows = v["per_seed_delong"]
            if not rows:
                continue
            delta = np.mean([r["delta"] for r in rows])
            ps = "[" + ", ".join(f"{r['p_delong']:.3f}" for r in rows) + "]"
            b = v["bootstrap"]
            bstr = (f"Δ={b['delta_mean']:+.4f} p={b['p_two_sided']}" if b else "; ")
            lines.append(f"  τ={tau:>3} mean ΔAUC(D−E)={delta:+.4f} "
                         f"DeLong p {ps}  bootstrap {bstr}")
    lines.append("=" * 100)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")


def write_table(part_a, part_b, part_c):
    rows = []
    part_a = part_a or {}
    part_b = part_b or {}
    part_c = part_c or {}
    # Part A
    for ek, exp_entries in (part_a.get("experiments") or {}).items():
        for tau, v in exp_entries.items():
            for r in v["per_seed_delong"]:
                b = v["bootstrap"]
                bp = v["bootstrap_paper_stat"]
                rows.append({
                    "part": "A_val", "experiment": ek, "tau": r["tau"],
                    "seed": r["seed"], "pair": "DANN_vs_ERM",
                    "auc_method": r["auc_dann"], "auc_ref": r["auc_erm"],
                    "auc_method_paper": r.get("auc_a_paper"),
                    "auc_ref_paper": r.get("auc_b_paper"),
                    "delta": r["delta_dann_minus_erm"], "p_delong": r["p_delong"],
                    "n_case": r["n_case"], "n_control": r["n_control"],
                    "bootstrap_delta": b["delta_mean"] if b else None,
                    "bootstrap_p": b["p_two_sided"] if b else None,
                    "bootstrap_paperstat_p": bp["p_two_sided"] if bp else None,
                })
    # Part B (methods vs ERM @36)
    for ek, exp_entries in (part_b.get("experiments") or {}).items():
        for method, v in exp_entries["methods_vs_erm_at_36"].items():
            for r in v["per_seed_delong"]:
                b = v["bootstrap"]
                rows.append({
                    "part": "B_val", "experiment": ek, "tau": r["tau"],
                    "seed": r["seed"], "pair": f"{method}_vs_ERM",
                    "auc_method": r["auc_a"], "auc_ref": r["auc_b"],
                    "delta": r["delta"], "p_delong": r["p_delong"],
                    "n_case": r["n_case"], "n_control": r["n_control"],
                    "bootstrap_delta": b["delta_mean"] if b else None,
                    "bootstrap_p": b["p_two_sided"] if b else None,
                })
    # Part C
    for ek, exp_entries in (part_c.get("experiments") or {}).items():
        if "error" in exp_entries:
            continue
        for tau, v in exp_entries.items():
            for r in v["per_seed_delong"]:
                b = v["bootstrap"]
                rows.append({
                    "part": "C_test", "experiment": ek, "tau": r["tau"],
                    "seed": r["seed"], "pair": "DANN_vs_ERM",
                    "auc_method": r["auc_a"], "auc_ref": r["auc_b"],
                    "delta": r["delta"], "p_delong": r["p_delong"],
                    "n_case": r["n_case"], "n_control": r["n_control"],
                    "bootstrap_delta": b["delta_mean"] if b else None,
                    "bootstrap_p": b["p_two_sided"] if b else None,
                })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "table16_delong_td_auc.csv",
                              index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table16_delong_td_auc.csv'} "
             f"({len(rows)} rows)")


def write_runs_json(part_a, part_b, part_c):
    """Flatten every per-seed DeLong row from the three parts into runs.json."""
    runs = []
    for ek, exp_entries in (part_a or {}).get("experiments", {}).items():
        for tau, v in exp_entries.items():
            for r in v.get("per_seed_delong", []):
                b = v.get("bootstrap")
                bp = v.get("bootstrap_paper_stat")
                runs.append({
                    "part": "A_val", "experiment": ek, "tau": r["tau"],
                    "seed": r["seed"], "pair": "DANN_vs_ERM",
                    "auc_dann": r.get("auc_dann"), "auc_erm": r.get("auc_erm"),
                    "delta_dann_minus_erm": r.get("delta_dann_minus_erm"),
                    "p_delong": r["p_delong"], "n_case": r["n_case"],
                    "n_control": r["n_control"],
                    "bootstrap_p_std": b["p_two_sided"] if b else None,
                    "bootstrap_p_paperstat": bp["p_two_sided"] if bp else None,
                })
    for ek, exp_entries in (part_b or {}).get("experiments", {}).items():
        for method, v in exp_entries.get("methods_vs_erm_at_36", {}).items():
            for r in v.get("per_seed_delong", []):
                b = v.get("bootstrap")
                runs.append({
                    "part": "B_val", "experiment": ek, "tau": r["tau"],
                    "seed": r["seed"], "pair": f"{method}_vs_ERM",
                    "auc_method": r["auc_a"], "auc_ref": r["auc_b"],
                    "delta_method_minus_erm": r["delta"],
                    "p_delong": r["p_delong"], "n_case": r["n_case"],
                    "n_control": r["n_control"],
                    "bootstrap_p_std": b["p_two_sided"] if b else None,
                })
    for ek, exp_entries in (part_c or {}).get("experiments", {}).items():
        if "error" in exp_entries:
            continue
        for tau, v in exp_entries.items():
            for r in v.get("per_seed_delong", []):
                b = v.get("bootstrap")
                runs.append({
                    "part": "C_test", "experiment": ek, "tau": r["tau"],
                    "seed": r["seed"], "pair": "DANN_vs_ERM",
                    "auc_dann": r["auc_a"], "auc_erm": r["auc_b"],
                    "delta_dann_minus_erm": r["delta"],
                    "p_delong": r["p_delong"], "n_case": r["n_case"],
                    "n_control": r["n_control"],
                    "bootstrap_p_std": b["p_two_sided"] if b else None,
                })
    with open(EXP_OUT / "runs.json", "w") as f:
        json.dump(runs, f, indent=2)
    log.info(f"runs.json → {EXP_OUT / 'runs.json'} ({len(runs)} records)")


def write_figure(part_a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for ek, exp_entries in part_a["experiments"].items():
        for tau, v in exp_entries.items():
            rs = v["per_seed_delong"]
            if not rs:
                continue
            # pooled Δ (DANN − ERM) from 3 seeds
            deltas = np.array([r["delta_dann_minus_erm"] for r in rs])
            b = v["bootstrap"]
            lo = b["ci_low"] if b else deltas.min() - 0.002
            hi = b["ci_high"] if b else deltas.max() + 0.002
            mean = float(np.mean(deltas))
            sig = (b["p_two_sided"] < 0.05) if b else False
            rows.append((ek, tau, mean, lo, hi, sig, b["p_two_sided"] if b else None))
    if not rows:
        return
    rows.sort(key=lambda x: (x[0], x[1]))
    labels = [f"{e[:1]}:τ={t}" for e, t, *_ in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, (ek, tau, mean, lo, hi, sig, p) in enumerate(rows):
        ax.plot([lo, hi], [y[i], y[i]], color="#4c72b0", lw=1.8)
        ax.plot(mean, y[i], "o", ms=7, color="#c44e52" if sig else "#4c72b0")
        tag = f"{mean:+.4f}" + (" ✱" if sig else "")
        ax.text(hi + 0.0006, y[i], tag, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("Δ time-dependent AUC (DANN − ERM), bootstrap 95% CI; ✱ = p<0.05")
    ax.set_title("DeLong td-AUC: DANN vs Baseline on A/B/C val (τ=12…60)")
    ax.set_xlim(min(r[2] for r in rows) - 0.02, max(r[3] for r in rows) + 0.02)
    fig.tight_layout()
    out = RESULTS_DIR / "figures" / "extended" / "D1_delong_dann_vs_base_forest.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info(f"Figure → {out}")


def main():
    parser = argparse.ArgumentParser(description="DeLong test for time-dependent AUC")
    parser.add_argument("--parts", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="skip bootstrap (faster, DeLong only)")
    args = parser.parse_args()

    do_boot = not args.no_bootstrap
    part_a = part_b = part_c = None
    if "A" in args.parts:
        part_a = run_part_a(do_boot)
        with open(EXP_OUT / "part_a_dann_vs_base.json", "w") as f:
            json.dump(part_a, f, indent=2, default=str)
    if "B" in args.parts:
        part_b = run_part_b(do_boot)
        with open(EXP_OUT / "part_b_methods.json", "w") as f:
            json.dump(part_b, f, indent=2, default=str)
    if "C" in args.parts:
        part_c = run_part_c(do_boot)
        with open(EXP_OUT / "part_c_independent_test.json", "w") as f:
            json.dump(part_c, f, indent=2, default=str)

    # load any already-saved parts (so --parts A C still summarises all)
    if part_a is None:
        p = EXP_OUT / "part_a_dann_vs_base.json"
        part_a = json.load(open(p)) if p.exists() else {}
    if part_b is None:
        p = EXP_OUT / "part_b_methods.json"
        part_b = json.load(open(p)) if p.exists() else {}
    if part_c is None:
        p = EXP_OUT / "part_c_independent_test.json"
        part_c = json.load(open(p)) if p.exists() else {}

    summarize(part_a, part_b, part_c, do_boot)
    write_table(part_a, part_b, part_c)
    write_runs_json(part_a, part_b, part_c)
    if part_a and part_a.get("experiments"):
        write_figure(part_a)
    log.info("\n✅ DeLong td-AUC analysis complete!")


if __name__ == "__main__":
    main()
