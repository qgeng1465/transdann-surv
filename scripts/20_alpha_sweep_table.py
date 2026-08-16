#!/usr/bin/env python3
"""
20_alpha_sweep_table.py; Summarise α-sensitivity results → table8 + domain-accuracy stats
==========================================================================================

Reads the output of `19_exp_alpha_sensitivity.py` (results/experiments/exp_alpha_sensitivity/),
and for each (exp, α, seed) extracts from its results.json:
  - best_val_cindex
  - domain-classifier accuracy min / peak / final (checks whether the GRL pathway is active)

Outputs:
  - results/tables/table8_alpha_sensitivity.csv   manuscript table
  - console summary (including the best-over-α argument)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SWEEP_DIR = BASE_DIR / "results" / "experiments" / "exp_alpha_sensitivity"
TABLE_DIR = BASE_DIR / "results" / "tables"

EXP_ORDER = ["A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts"]
EXP_SHORT = {"A_tcga_vs_seer": "A", "B_tcga_vs_external": "B", "C_all_cohorts": "C"}
ALPHAS = [0.1, 0.5, 1.0]
SEEDS = [42, 43, 44]


def load_run(exp, alpha, seed):
    """alpha=None → baseline; else fixed α. Returns dict or None."""
    sub = "baseline" if alpha is None else f"alpha_{alpha}"
    rj = SWEEP_DIR / exp / sub / f"seed_{seed}" / "results.json"
    if not rj.exists():
        return None
    with open(rj) as f:
        return json.load(f)


def main():
    if not SWEEP_DIR.exists():
        print(f"❌ sweep dir not found: {SWEEP_DIR}")
        sys.exit(1)

    rows = []
    print(f"{'exp':<4}{'α':<9}{'C-index':<14}{'per-seed':<24}"
          f"{'domacc min/peak/final'}")
    print("-" * 80)
    for exp in EXP_ORDER:
        for alpha in [None] + ALPHAS:
            cidx = []
            dom_stats = []
            for seed in SEEDS:
                res = load_run(exp, alpha, seed)
                if res is None:
                    print(f"  !! missing {exp} {alpha} seed {seed}")
                    continue
                cidx.append(float(res["best_val_cindex"]))
                hist = res.get("history", {})
                da = [float(x) for x in hist.get("domain_acc", []) if x is not None]
                if da:
                    dom_stats.append({
                        "min": float(np.min(da)), "peak": float(np.max(da)),
                        "final": da[-1],
                    })
            if not cidx:
                continue
            mean = float(np.mean(cidx))
            std = float(np.std(cidx, ddof=1)) if len(cidx) > 1 else 0.0
            per_seed = ", ".join(f"{x:.4f}" for x in cidx)
            dmin = f"{min(d['min'] for d in dom_stats):.3f}" if dom_stats else "-"
            dpeak = f"{max(d['peak'] for d in dom_stats):.3f}" if dom_stats else "-"
            dfinal = f"{np.mean([d['final'] for d in dom_stats]):.3f}" if dom_stats else "-"
            rows.append({
                "experiment": EXP_SHORT[exp], "alpha": alpha,
                "alpha_label": "baseline" if alpha is None else f"α={alpha}",
                "cindex_mean": round(mean, 4), "cindex_std": round(std, 4),
                "cindex_per_seed": [round(x, 4) for x in cidx],
                "n_seeds": len(cidx),
                "domacc_min": dmin, "domacc_peak": dpeak, "domacc_final": dfinal,
            })
            label = "baseline" if alpha is None else f"α={alpha}"
            print(f"{EXP_SHORT[exp]:<4}{label:<9}{mean:.4f}±{std:.4f}  "
                  f"[{per_seed}]  {dmin}/{dpeak}/{dfinal}")

    # ---- best-over-α argument ----
    print("\n" + "=" * 80)
    print("BEST-OVER-α argument (reviewer-friendly: even at the best α, does DANN beat Baseline?)")
    print("=" * 80)
    for exp in EXP_ORDER:
        base = [r for r in rows if r["experiment"] == EXP_SHORT[exp]
                and r["alpha"] is None]
        danns = [r for r in rows if r["experiment"] == EXP_SHORT[exp]
                 and r["alpha"] is not None]
        if not base or not danns:
            continue
        b = base[0]["cindex_mean"]
        best = max(danns, key=lambda r: r["cindex_mean"])
        delta = best["cindex_mean"] - b
        spread = {r["alpha_label"]: round(r["cindex_mean"], 4) for r in danns}
        print(f"  Exp {EXP_SHORT[exp]}: baseline={b:.4f} | "
              f"best-over-α ({best['alpha_label']}) = {best['cindex_mean']:.4f} "
              f"(Δ={delta:+.4f}) | all-α: {spread}")

    # ---- save table8 ----
    TABLE_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df["delta_vs_baseline"] = np.nan
    for exp in EXP_ORDER:
        base = df[(df["experiment"] == EXP_SHORT[exp]) & (df["alpha"].isna())]
        if base.empty:
            continue
        bm = base.iloc[0]["cindex_mean"]
        mask = (df["experiment"] == EXP_SHORT[exp]) & df["alpha"].notna()
        df.loc[mask, "delta_vs_baseline"] = df.loc[mask, "cindex_mean"] - bm
    out_path = TABLE_DIR / "table8_alpha_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ table8 → {out_path}")


if __name__ == "__main__":
    main()
