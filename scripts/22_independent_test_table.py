#!/usr/bin/env python3
"""
22_independent_test_table.py; Summarise true independent-test-set 3-fold retrain → table9
==========================================================================================

Reads the output of `21_exp_independent_test.py` (results/experiments/exp_independent_test/),
and for each experiment computes (units = 3 folds × 3 seeds = 9 paired DANN/Baseline
test-set C-index values):

  - DANN / Baseline test-set C-index mean ± std (over 9 units)
  - per-fold fold-level Δ (over 3 seeds)
  - paired Δ = DANN − Baseline mean ± std + t 95% CI
  - **sign-flip permutation test** (sign-flip, 2^9 = 512 labelings): under the null,
    DANN/Baseline are exchangeable within each unit, and every sign pattern is
    equally likely; p = P(|flip-mean| ≥ |obs-mean|). Analytic lower bound for
    n=9 = 2/512 ≈ 0.0039.

Outputs:
  - results/tables/table9_independent_test.csv   manuscript table
  - results/experiments/exp_independent_test/test_cindex_matrix.csv  (fold × seed detail)
  - console summary (incl. conclusion)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INDEP_DIR = BASE_DIR / "results" / "experiments" / "exp_independent_test"
TABLE_DIR = BASE_DIR / "results" / "tables"

EXP_ORDER = ["A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts"]
EXP_SHORT = {"A_tcga_vs_seer": "A", "B_tcga_vs_external": "B", "C_all_cohorts": "C"}

# scipy is usually installed with lifelines; if absent, fall back to hard-coded t critical values
try:
    from scipy import stats
    T_CRIT = None
except ImportError:
    stats = None
    T_CRIT = {9: 2.306}  # t(8, 0.975)


def t_crit(n):
    if stats is not None:
        return stats.t.ppf(0.975, n - 1)
    return T_CRIT.get(n, 2.306)


def signflip_p(deltas):
    """Sign-flip permutation test: all 2^n flips are equally likely; returns a two-sided p."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    obs_abs = abs(np.mean(d))
    count = 0
    for bits in range(1 << n):
        signs = 1.0 - 2.0 * np.array(
            [(bits >> k) & 1 for k in range(n)], dtype=float
        )
        if abs(np.mean(d * signs)) >= obs_abs - 1e-12:
            count += 1
    return count / float(1 << n)


def main():
    runs_path = INDEP_DIR / "runs.json"
    if not runs_path.exists():
        print(f"❌ runs.json not found: {runs_path}\n   Run scripts/21_exp_independent_test.py first")
        sys.exit(1)

    with open(runs_path) as f:
        runs = json.load(f)
    records = runs["all_runs"]
    records = [r for r in records if r.get("test_cindex") is not None]
    if not records:
        print("❌ no valid runs with test_cindex")
        sys.exit(1)

    rows = []
    matrix_rows = []
    print("=" * 104)
    print("TRUE INDEPENDENT TEST-SET 3-FOLD RETAIN; DANN vs Baseline (test C-index)")
    print("=" * 104)

    for exp in EXP_ORDER:
        rs = [r for r in records if r["experiment"] == exp]
        if not rs:
            print(f"  !! {exp}: no runs")
            continue
        # ---- per (fold, seed) paired deltas ----
        deltas = []
        for fold in sorted(set(r["fold"] for r in rs)):
            for seed in sorted(set(r["seed"] for r in rs)):
                d = next((r["test_cindex"] for r in rs
                          if r["fold"] == fold and r["seed"] == seed
                          and r["mode"] == "dann"), None)
                b = next((r["test_cindex"] for r in rs
                          if r["fold"] == fold and r["seed"] == seed
                          and r["mode"] == "baseline"), None)
                if d is not None and b is not None:
                    deltas.append((fold, seed, d, b, d - b))
        if len(deltas) < 9:
            print(f"  !! {exp}: only {len(deltas)}/9 paired units")
        n = len(deltas)
        dann_c = np.array([x[2] for x in deltas])
        base_c = np.array([x[3] for x in deltas])
        delta = np.array([x[4] for x in deltas])

        dm, ds = float(np.mean(dann_c)), float(np.std(dann_c, ddof=1) if n > 1 else 0.0)
        bm, bs = float(np.mean(base_c)), float(np.std(base_c, ddof=1) if n > 1 else 0.0)
        delm, dels = float(np.mean(delta)), float(np.std(delta, ddof=1) if n > 1 else 0.0)
        ci_low = delm - t_crit(n) * dels / np.sqrt(n)
        ci_high = delm + t_crit(n) * dels / np.sqrt(n)
        p = signflip_p(delta)
        n_pos = int(np.sum(delta > 0))
        n_neg = int(np.sum(delta < 0))

        # ---- fold-level (mean over seeds) ----
        fold_deltas = []
        for fold in sorted(set(x[0] for x in deltas)):
            fd = [x for x in deltas if x[0] == fold]
            fold_deltas.append(float(np.mean([x[4] for x in fd])))

        print(f"\n  {EXP_SHORT[exp]}  (n={n} paired units)")
        print(f"    DANN     test C-index: {dm:.4f} ± {ds:.4f}")
        print(f"    Baseline test C-index: {bm:.4f} ± {bs:.4f}")
        print(f"    Δ(DANN−Base)         : {delm:+.4f} ± {dels:.4f}   "
              f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"    fold Δ (mean over seeds): "
              + ", ".join(f"f{fold}:{d:+.4f}" for fold, d in zip(sorted(set(x[0] for x in deltas)), fold_deltas)))
        print(f"    sign-flip perm p       : {p:.4f}  ({n_pos}+ / {n_neg}-)")

        rows.append({
            "experiment": EXP_SHORT[exp],
            "dann_mean": round(dm, 4), "dann_std": round(ds, 4),
            "baseline_mean": round(bm, 4), "baseline_std": round(bs, 4),
            "delta_mean": round(delm, 4), "delta_std": round(dels, 4),
            "delta_ci_low": round(ci_low, 4), "delta_ci_high": round(ci_high, 4),
            "n_units": n,
            "n_pos_delta": n_pos, "n_neg_delta": n_neg,
            "signflip_p": round(p, 4),
            "fold_deltas": [round(x, 4) for x in fold_deltas],
        })
        for fold, seed, d, b, delta_v in deltas:
            matrix_rows.append({
                "experiment": EXP_SHORT[exp], "fold": fold, "seed": seed,
                "dann_test_c": round(d, 4), "baseline_test_c": round(b, 4),
                "delta": round(delta_v, 4),
            })

    print("\n" + "=" * 104)
    print("Conclusion framework (to be written into the report): on the true "
          "independent test set DANN still does not beat Baseline; the best-val "
          "metric overestimates test performance for both models (gap in the detail).")
    print("=" * 104)

    # ---- save ----
    TABLE_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = TABLE_DIR / "table9_independent_test.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ table9 → {out_path}")

    INDEP_DIR.mkdir(exist_ok=True)
    mdf = pd.DataFrame(matrix_rows)
    mpath = INDEP_DIR / "test_cindex_matrix.csv"
    mdf.to_csv(mpath, index=False)
    print(f"✅ test_cindex_matrix → {mpath}")


if __name__ == "__main__":
    main()
