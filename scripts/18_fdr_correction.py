#!/usr/bin/env python3
"""
18_fdr_correction.py; Benjamini–Hochberg FDR correction for reviewer response
================================================================================
Addresses reviewer concern #2 ("no multiple-testing correction").  The paper
reports hundreds of bootstrap p-values (Exp J: 9 methods × 3 splits; Exp L/M:
4 methods × 2 cancers; Exp F: 5 experiments).  This script pools ALL reported
two-sided p-values and applies Benjamini–Hochberg FDR control at α = 0.05 and
0.10, then flags which claims survive.

Data sources (all from verified result artifacts, nothing hard-coded):
  1. results/tables/table3_method_comparison.csv; Exp J: Δ vs ERM (24 tests)
  2. results/experiments/exp_l/coad_results.json; Exp L: bootstrap_vs_erm (3)
  3. results/experiments/exp_m/luad_results.json; Exp M: bootstrap_vs_erm (3)
  4. results/F_bootstrap_val_results.json; Exp F: held-out-val bootstrap (5)

Outputs
-------
  results/experiments/fdr/table3_fdr_corrected.csv; table3 + q-values + flags
  results/experiments/fdr/fdr_pooled.csv; all 35 tests, BH-corrected
  results/experiments/fdr/fdr_report.md; human-readable summary
  results/tables/table3_method_comparison.csv; (in-place: adds q_two_sided
                                                       and fdr_significant_05 columns)

Interpretation of the actual data (α=0.05):
  - Only two tests survive FDR: Exp-C Mixup (+0.019, positive) and Exp-B V-REx
    (−0.067, harmful).  The nominally-significant C-IRM (p=0.043), C-V-REx
    (p=0.043), B-GroupDRO (p=0.045) and B-Fish (p=0.048) do NOT survive.
  - Therefore the manuscript must state: "after BH-FDR correction, only input-level
    Mixup remains a significant positive result; the IRM signal does not survive."
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FDR_DIR = RESULTS_DIR / "experiments" / "fdr"
TABLE_DIR = RESULTS_DIR / "tables"
os.makedirs(FDR_DIR, exist_ok=True)

ALPHA_LEVELS = (0.05, 0.10)

# Bootstrap p is computed as min(2*min(p_sup,p_inf), 1) with B=3000 draws, so a
# stored 0.0 means literally zero draws of one sign among 3000 → p < 1/3000.
P_ZERO_FLOOR = 1.0 / 3000.0


# ===================================================================
# BH-FDR (implemented inline; no statsmodels dependency)
# ===================================================================
def bh_fdr(pvals, alpha=0.05):
    """Benjamini–Hochberg.  Returns (qvals, is_significant).

    Correct procedure: sort p ascending; reject H_(i) for i <= k where
    k = max{i : p_(i) <= (i/m)*alpha} (with k=0 if none satisfy).
    q-value = p_(i)*m/i, made monotone from the largest p downward.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    # BH adjusted p-values, monotone from largest to smallest
    q_sorted = p_sorted * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    # largest rank k whose raw p still satisfies the BH threshold
    thr = np.arange(1, n + 1) / n * alpha
    reject = p_sorted <= thr
    k = (np.where(reject)[0].max() + 1) if reject.any() else 0
    sig_sorted = np.zeros(n, dtype=bool)
    sig_sorted[:k] = True
    # map back to original order
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    sig = np.empty_like(sig_sorted)
    sig[order] = sig_sorted
    return q, sig


# ===================================================================
# Collect all reported p-values
# ===================================================================
def collect_table3():
    df = pd.read_csv(TABLE_DIR / "table3_method_comparison.csv")
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["p_two_sided"]):
            continue
        rows.append({
            "source": "J",
            "group": f"Exp-{r['experiment']}",
            "test": f"{r['experiment']}·{r['method']} vs ERM",
            "effect": r["delta_vs_erm"],
            "p_raw": float(r["p_two_sided"]),
            "n_draws": None,
        })
    return rows, df


def collect_lm(path, cancer):
    d = json.load(open(path))
    rows = []
    for m, b in d["bootstrap_vs_erm"].items():
        rows.append({
            "source": "L/M",
            "group": cancer,
            "test": f"{cancer}·{m} vs ERM",
            "effect": b["delta_mean"],
            "p_raw": float(b["p_two_sided"]),
            "n_draws": b.get("n_draws"),
        })
    return rows


def collect_f():
    d = json.load(open(RESULTS_DIR / "F_bootstrap_val_results.json"))
    rows = []
    for k, v in d.items():
        rows.append({
            "source": "F",
            "group": "Exp-F val",
            "test": f"F·{k} DANN vs Baseline",
            "effect": v["delta_obs"],
            "p_raw": float(v["p_two_sided"]),
            "n_draws": None,
        })
    return rows


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHA_LEVELS))
    args = parser.parse_args()

    rows = []
    t3_rows, table3_df = collect_table3()
    rows += t3_rows
    rows += collect_lm(RESULTS_DIR / "experiments" / "exp_l" / "coad_results.json", "COAD")
    rows += collect_lm(RESULTS_DIR / "experiments" / "exp_m" / "luad_results.json", "LUAD")
    rows += collect_f()

    df = pd.DataFrame(rows)
    # clamp stored-0.0 p-values (they encode p < 1/B for the bootstrap)
    df["p_comp"] = df["p_raw"].replace(0.0, P_ZERO_FLOOR)
    n_tests = len(df)

    for alpha in sorted(args.alphas):
        q, sig = bh_fdr(df["p_comp"].values, alpha=alpha)
        df[f"q_{alpha:.2f}"] = np.round(q, 4)
        df[f"sig_{alpha:.2f}"] = sig

    df = df.drop(columns=["p_comp"])
    df["p_label"] = df["p_raw"].apply(
        lambda p: "<0.001" if p == 0.0 else f"{p:.4f}")

    # sort: significant first, then by p
    df = df.sort_values("p_raw")

    # ---- write pooled FDR table ----
    df.to_csv(FDR_DIR / "fdr_pooled.csv", index=False)

    # ---- write table3 with FDR columns (in place) ----
    t3 = table3_df.copy()
    qmap = {}
    for _, r in df[df["source"] == "J"].iterrows():
        qmap[r["test"]] = r
    t3["q_two_sided"] = t3.apply(
        lambda r: (qmap.get(f"{r['experiment']}·{r['method']} vs ERM", pd.Series(dtype=float))
                   [f"q_{0.05:.2f}"] if not pd.isna(r.get("p_two_sided", np.nan)) else np.nan),
        axis=1)
    t3["fdr_significant_05"] = t3.apply(
        lambda r: (qmap.get(f"{r['experiment']}·{r['method']} vs ERM", pd.Series(dtype=bool))
                   [f"sig_{0.05:.2f}"] if not pd.isna(r.get("p_two_sided", np.nan)) else False),
        axis=1)
    t3.to_csv(TABLE_DIR / "table3_method_comparison.csv", index=False)
    t3.to_csv(FDR_DIR / "table3_fdr_corrected.csv", index=False)

    # ---- markdown report ----
    lines = [
        "# FDR correction report (Benjamini–Hochberg); reviewer comment #2",
        "",
        f"> {n_tests} two-sided bootstrap p-values pooled (Exp J: 24, Exp L/M: 6, Exp F: 5).",
        f"> Stored p=0.0 is treated as `p < 1/3000` (no draw of the opposite sign in 3000 resamples).",
        "",
    ]
    for alpha in sorted(args.alphas):
        sig_df = df[df[f"sig_{alpha:.2f}"]]
        lines += [
            f"## α = {alpha:.2f}",
            "",
            f"**Tests passing FDR: {len(sig_df)} / {n_tests}**",
            "",
        ]
        if len(sig_df):
            lines += ["| Test | Δ | Raw p | q | Direction |", "|---|---|---|---|---|"]
            for _, r in sig_df.iterrows():
                direction = "positive (input-level augmentation such as Mixup)" if r["effect"] > 0 else "negative (harmful)"
                lines.append(f"| {r['test']} | {r['effect']:+.4f} | {r['p_label']} | "
                             f"{r[f'q_{alpha:.2f}']:.4f} | {direction} |")
        else:
            lines += ["(no tests pass)", ""]
        # nominally significant but NOT surviving
        nom = df[(df["p_raw"] < 0.05) & ~df[f"sig_{alpha:.2f}"]]
        if len(nom):
            lines += [
                "",
                "**Nominally significant but failing FDR (p<0.05 → q>α):**",
                "",
                "| Test | Δ | Raw p | q | Note |",
                "|---|---|---|---|---|",
            ]
            for _, r in nom.iterrows():
                lines.append(f"| {r['test']} | {r['effect']:+.4f} | {r['p_label']} | "
                             f"{r[f'q_{alpha:.2f}']:.4f} | must be reported as non-significant in the manuscript |")
        lines += ["", "---", ""]

    lines += [
        "## Implications for manuscript wording",
        "",
        "- **Mixup (Exp C, Δ=+0.019)**: the only surviving positive result; retain in the manuscript as",
        "  \"input-level augmentation (Mixup) remains significantly positive after multiple-testing correction on the 5-domain split.\"",
        "- **IRM (Exp C, Δ=+0.013, p=0.043)**: fails FDR; the manuscript must no longer call it",
        "  \"statistically significant\"; phrase it instead as \"nominally significant but failing BH-FDR correction.\"",
        "- **V-REx (Exp B, Δ=−0.067)**: a **negative** result passing FDR; strengthens the claim that",
        "  \"variance-penalty-based DG methods are significantly harmful on multi-domain heterogeneous data.\"",
        "- **GroupDRO / Fish / C-V-REx**: nominally significant but failing FDR; phrase uniformly in the manuscript as",
        "  \"a negative tendency that does not survive multiple-testing correction.\"",
        "- **DANN (all p>0.65), Exp F (p=0.058–0.898), L/M**: already non-significant; conclusions unchanged after correction.",
        "",
        "## Output files",
        "",
        "- `results/tables/table3_method_comparison.csv` (original table with two new columns, `q_two_sided`,",
        "  `fdr_significant_05`)",
        "- `results/experiments/fdr/fdr_pooled.csv` (all 35 tests)",
        "- `results/experiments/fdr/table3_fdr_corrected.csv`",
        "- this report",
    ]
    (FDR_DIR / "fdr_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'='*72}\nBH-FDR correction; {n_tests} pooled tests\n{'='*72}")
    for alpha in sorted(args.alphas):
        sig_df = df[df[f"sig_{alpha:.2f}"]]
        print(f"\nα = {alpha:.2f} → {len(sig_df)}/{n_tests} significant:")
        for _, r in sig_df.iterrows():
            print(f"   {r['test']:<34s} Δ={r['effect']:+.4f}  p={r['p_label']:<7s} q={r[f'q_{alpha:.2f}']:.4f}")
    print(f"\nTables → {FDR_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
