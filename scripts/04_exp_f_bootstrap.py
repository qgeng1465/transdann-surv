#!/usr/bin/env python3
"""
04_exp_f_bootstrap.py; Experiment F: Statistical Significance Testing
======================================================================
Bootstraps (B=1000, stratified by cohort) the C-index difference
Δ = C-index(DANN) − C-index(Baseline) for all five experiments
A, B, C, D (imputed), E (clean breast control).

C-index uses ``fast_lifelines_cindex``; a bit-exact vectorised port of
lifelines' concordance_index (validated diff == 0.0), so the bootstrap Δ is
directly comparable to the reported lifelines C-index values.

For each experiment:
  1. Reconstruct the exact train+val ordering (deterministic) so saved
     risks.npy aligns with each sample's time/event/domain.
  2. Sanity-check: recomputed per-cohort C-index must match results.json /
     comparison.json.
  3. Stratified bootstrap → 95% CI + two-sided p-value (H0: Δ=0) overall and
     per cohort.

Outputs:
  - results/F_bootstrap_results.json
  - results/figures/extended/F_bootstrap_CIs.png      (forest plot)
  - results/figures/extended/F_bootstrap_hist_{A,B,C,D,E}.png
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fast_cindex import fast_lifelines_cindex
from lihc_recon import (
    reconstruct_full_data, reconstruct_exp_d, reconstruct_exp_e, BASE_DIR,
)

RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
os.makedirs(FIG_DIR, exist_ok=True)


def cindex(times, events, risks):
    if len(times) < 2 or (events > 0).sum() < 1:
        return np.nan
    return fast_lifelines_cindex(times, -risks, events)


def strat_bootstrap_delta(times, events, r_dann, r_base, domains, B=1000, seed=123):
    rng = np.random.RandomState(seed)
    uniq = np.unique(domains)
    deltas = np.zeros(B)
    for b in range(B):
        idx = []
        for d in uniq:
            m = np.where(domains == d)[0]
            idx.append(m[rng.randint(0, len(m), size=len(m))])
        idx = np.concatenate(idx)
        deltas[b] = cindex(times[idx], events[idx], r_dann[idx]) \
            - cindex(times[idx], events[idx], r_base[idx])
    return deltas


def bootstrap_one_cohort(times, events, r_dann, r_base, mask, B=1000, seed=999):
    t, e, a, b = times[mask], events[mask], r_dann[mask], r_base[mask]
    if (e > 0).sum() < 8 or len(t) < 25:
        return None
    rng = np.random.RandomState(seed)
    deltas = np.zeros(B)
    for i in range(B):
        idx = rng.randint(0, len(t), size=len(t))
        deltas[i] = cindex(t[idx], e[idx], a[idx]) - cindex(t[idx], e[idx], b[idx])
    return deltas


def summarize(deltas, obs):
    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_sup = float(np.mean(d > 0))
    p_inf = float(np.mean(d < 0))
    return {
        "delta_obs": round(float(obs), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "mean_boot": round(float(d.mean()), 4),
        "p_dann_superior": round(p_sup, 4),
        "p_dann_inferior": round(p_inf, 4),
        "p_two_sided": round(min(2.0 * min(p_sup, p_inf), 1.0), 4),
    }


EXPS = ["A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts",
        "D_imputed_tcga_seer", "E_brca_metabric"]


def load_risks(exp_dir):
    r_d = np.load(exp_dir / "dann" / "risks.npy")
    r_b = np.load(exp_dir / "baseline" / "risks.npy")
    return r_d, r_b


def main():
    B = 1000
    overall = {}
    rows = []

    for exp_key in EXPS:
        if exp_key in ("A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts"):
            full_df, times, events, domains, sources = reconstruct_full_data(exp_key)
        elif exp_key == "D_imputed_tcga_seer":
            full_df, times, events, domains, sources = reconstruct_exp_d()
        else:
            full_df, times, events, domains, sources = reconstruct_exp_e()

        exp_dir = RESULTS_DIR / "lihc_experiments" / exp_key
        r_dann, r_base = load_risks(exp_dir)
        assert len(r_dann) == len(full_df), f"{exp_key}: risks/df length mismatch"

        # ---- cohort map + reported values ----
        comp = json.load(open(exp_dir / "comparison.json"))
        cohort_map = comp["cohort_map"]
        rev = {int(v): k for k, v in cohort_map.items()}
        reported_delta = comp.get("delta")

        # ---- sanity check: per-cohort C-index vs comparison.json ----
        ok = True
        for dom_id in sorted(rev.keys()):
            m = domains == dom_id
            c_d = cindex(times[m], events[m], r_dann[m])
            c_b = cindex(times[m], events[m], r_base[m])
            exp_d = comp["dann_per_cohort"].get(str(dom_id))
            exp_b = comp["baseline_per_cohort"].get(str(dom_id))
            if exp_d is not None and exp_b is not None:
                if abs(c_d - float(exp_d)) > 0.02 or abs(c_b - float(exp_b)) > 0.02:
                    ok = False
                    print(f"  [WARN] {exp_key} {rev[dom_id]}: recon "
                          f"DANN={c_d:.4f} saved={exp_d}, Base={c_b:.4f} saved={exp_b}")
        print(f"[{exp_key}] alignment {'PASS' if ok else 'FAIL'} "
              f"({len(full_df)} samples)")

        obs_d = cindex(times, events, r_dann)
        obs_b = cindex(times, events, r_base)
        obs = obs_d - obs_b
        deltas = strat_bootstrap_delta(times, events, r_dann, r_base, domains, B=B)

        overall[exp_key] = {
            "cohort_map": {str(k): v for k, v in cohort_map.items()},
            "n_samples": int(len(full_df)),
            "reported_delta": reported_delta,
            "overall": {
                "dann_cindex": round(float(obs_d), 4),
                "baseline_cindex": round(float(obs_b), 4),
                **summarize(deltas, obs),
            },
            "per_cohort": {},
        }
        rows.append((exp_key, obs, np.percentile(deltas, [2.5, 97.5])))

        for dom_id in sorted(rev.keys()):
            m = domains == dom_id
            name = rev[dom_id]
            r = bootstrap_one_cohort(times, events, r_dann, r_base, m)
            if r is None:
                overall[exp_key]["per_cohort"][name] = {"skipped": True}
                continue
            d_obs = cindex(times[m], events[m], r_dann[m]) - cindex(times[m], events[m], r_base[m])
            overall[exp_key]["per_cohort"][name] = summarize(r, d_obs)
            rows.append((f"  {name}", d_obs, np.percentile(r, [2.5, 97.5])))

        # histogram of overall Δ
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.hist(deltas, bins=40, color="#4c72b0", alpha=0.85)
        ax.axvline(obs, color="#c44e52", lw=2, label=f"obs Δ={obs:+.4f}")
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        ax.axvline(lo, color="#c44e52", ls="--", lw=1)
        ax.axvline(hi, color="#c44e52", ls="--", lw=1)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set_xlabel("Δ C-index (DANN − Baseline)")
        ax.set_ylabel("bootstrap count")
        ax.set_title(f"Exp {exp_key[0]}; 1000 stratified bootstrap")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"F_bootstrap_hist_{exp_key[0]}.png", dpi=150)
        plt.close(fig)

    # ---- forest plot ----
    fig, ax = plt.subplots(figsize=(7.4, max(5.0, 0.4 * len(rows))))
    for i, (label, obs, (lo, hi)) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color="#4c72b0", lw=3, solid_capstyle="round")
        ax.plot(obs, y, "o", color="#c44e52", ms=7)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Δ C-index (DANN − Baseline), 95% bootstrap CI")
    ax.set_title("Experiment F; Bootstrap significance of Δ (B=1000)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F_bootstrap_CIs.png", dpi=150)
    plt.close(fig)

    with open(RESULTS_DIR / "F_bootstrap_results.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(json.dumps(overall, indent=2))
    print(f"\nSaved → {RESULTS_DIR / 'F_bootstrap_results.json'}")


if __name__ == "__main__":
    main()
