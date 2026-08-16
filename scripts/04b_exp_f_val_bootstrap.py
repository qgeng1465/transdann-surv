#!/usr/bin/env python3
"""
04b_exp_f_val_bootstrap.py; Validation-Only Bootstrap (matches reported Δ)
============================================================================
The headline Δ in the report is computed on the HELD-OUT validation split.
A full-set bootstrap (04) dilutes it because training samples dominate. This
script bootstraps the Δ on the validation rows only, which is the exact same
population the reported C-index came from.

Full_df is ordered [train..., val...], so val rows = the tail block.
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

from fast_cindex import fast_lifelines_cindex
from lihc_recon import (
    BASE_DIR, CONT_FEATURES, CAT_FEATURES, build_data,
    reconstruct_full_data, reconstruct_exp_d, reconstruct_exp_e, EXPERIMENTS,
    _load_train_02,
)

RESULTS_DIR = BASE_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def cindex(t, e, r):
    if len(t) < 2 or (e > 0).sum() < 1:
        return np.nan
    return fast_lifelines_cindex(t, -r, e)


def val_delta_bootstrap(times, events, r_dann, r_base, domains, mask, B=1000, seed=777):
    """Bootstrap Δ on validation samples only (stratified by cohort)."""
    t, e, a, b, d = times[mask], events[mask], r_dann[mask], r_base[mask], domains[mask]
    rng = np.random.RandomState(seed)
    uniq = np.unique(d)
    deltas = np.zeros(B)
    for i in range(B):
        idx = []
        for u in uniq:
            m = np.where(d == u)[0]
            idx.append(m[rng.randint(0, len(m), size=len(m))])
        idx = np.concatenate(idx)
        deltas[i] = cindex(t[idx], e[idx], a[idx]) - cindex(t[idx], e[idx], b[idx])
    return deltas


def main():
    B = 1000
    out = {}

    for exp_key, kind in [
        ("A_tcga_vs_seer", "abc"),
        ("B_tcga_vs_external", "abc"),
        ("C_all_cohorts", "abc"),
        ("D_imputed_tcga_seer", "d"),
        ("E_brca_metabric", "e"),
    ]:
        if kind == "abc":
            full_df, times, events, domains, sources = reconstruct_full_data(exp_key)
        elif kind == "d":
            full_df, times, events, domains, sources = reconstruct_exp_d()
        else:
            full_df, times, events, domains, sources = reconstruct_exp_e()
        exp_dir = RESULTS_DIR / "lihc_experiments" / exp_key
        r_dann = np.load(exp_dir / "dann" / "risks.npy")
        r_base = np.load(exp_dir / "baseline" / "risks.npy")
        comp = json.load(open(exp_dir / "comparison.json"))

        # Rebuild the train/val split to recover the val block length.
        if kind == "abc":
            df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
            df = df[df["Survival_Months"] > 0]
            df = df[df["Vital_Status"].isin([0, 1])]
            df = df[df["Source"].isin(EXPERIMENTS[exp_key]["cohorts"])].copy()
            data = _load_train_02().load_and_preprocess(
                df, surv_type="deephit", n_bins=32, subsample_seer=10000)
        elif exp_key == "D_imputed_tcga_seer":
            from lihc_recon import _load_exp_d
            raw = _load_exp_d().load_raw()
            imputed = _load_exp_d().impute_knn(raw)
            data = build_data(imputed, CONT_FEATURES, CAT_FEATURES, n_bins=32, subsample_seer=10000)
        else:
            from lihc_recon import _load_exp_e
            breast = _load_exp_e().load_breast()
            data = build_data(breast, ["Age"], ["Sex", "Stage"], n_bins=32)

        n_train = len(data["train_df"])
        val_mask = np.arange(n_train, len(full_df))
        assert len(val_mask) == len(data["val_df"]), f"{exp_key}: val block mismatch"

        t_v, e_v = times[val_mask], events[val_mask]
        d_v = domains[val_mask]
        a_v, b_v = r_dann[val_mask], r_base[val_mask]

        obs_a = cindex(t_v, e_v, a_v)
        obs_b = cindex(t_v, e_v, b_v)
        obs = obs_a - obs_b
        deltas = val_delta_bootstrap(times, events, r_dann, r_base, domains, val_mask, B=B)

        out[exp_key] = {
            "n_val": int(len(val_mask)),
            "reported_delta": comp.get("delta"),
            "dann_val_cindex": round(float(obs_a), 4),
            "baseline_val_cindex": round(float(obs_b), 4),
            "delta_obs": round(float(obs), 4),
            "ci_low": round(float(np.percentile(deltas, 2.5)), 4),
            "ci_high": round(float(np.percentile(deltas, 97.5)), 4),
            "p_dann_superior": round(float(np.mean(deltas > 0)), 4),
            "p_dann_inferior": round(float(np.mean(deltas < 0)), 4),
            "p_two_sided": round(min(2 * min(float(np.mean(deltas > 0)),
                                             float(np.mean(deltas < 0))), 1.0), 4),
        }
        print(f"[{exp_key}] n_val={len(val_mask)} obsΔ={obs:+.4f} "
              f"CI=[{out[exp_key]['ci_low']:+}, {out[exp_key]['ci_high']:+}] "
              f"p_two={out[exp_key]['p_two_sided']}")

    with open(RESULTS_DIR / "F_bootstrap_val_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR / 'F_bootstrap_val_results.json'}")


if __name__ == "__main__":
    main()
