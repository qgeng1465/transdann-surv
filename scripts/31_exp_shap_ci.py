#!/usr/bin/env python3
"""
31_exp_shap_ci.py; Bootstrap confidence intervals for SHAP (Exp G figures)
===========================================================================

Adds an uncertainty measure to exp_shap_bc (script 28): for each feature's
mean|SHAP| and its **DANN−Baseline difference**, compute a within-cohort stratified
bootstrap 95% CI.

- per-sample SHAP is saved by script 28 as {exp}_agg_{mode}.npy (n_val × n_features,
  row order matching val_df).
- bootstrap: resample within each cohort (Source) (same recipe as Exp F/J, B=500),
  recompute mean|SHAP| and Δ(DANN−Base); because the two modes are on the same
  subjects, resampling uses the **same index** (paired bootstrap), so the CI of Δ
  directly answers whether "B's Stage +1.37" is significantly different from 0.

Outputs (results/experiments/exp_shap_bc/):
  - ci_summary.json / summary_ci.txt
  - results/tables/table18_shap_ci.csv
  - results/figures/extended/G6_shap_ci_compare.png
  - logs/31_exp_shap_ci.log

Usage:
  python3 scripts/31_exp_shap_ci.py            # B + C
  python3 scripts/31_exp_shap_ci.py --exps B   # run B only
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

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_shap_bc"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "31_exp_shap_ci.log", mode="w"),
    ],
)
log = logging.getLogger("shap_ci")

spec28 = importlib.util.spec_from_file_location(
    "exp_shap_28", SCRIPTS_DIR / "28_exp_shap_bc.py"
)
exp28 = importlib.util.module_from_spec(spec28)
spec28.loader.exec_module(exp28)
FEATURES = exp28.FEATURES
EXP_KEYS = exp28.EXP_KEYS

from lihc_recon import _load_train_02  # noqa: E402

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
B = 500
SEED = 0


def load_val_sources(exp_key):
    t02 = _load_train_02()
    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    exp_cfg = t02.EXPERIMENTS[exp_key]
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    data = t02.load_and_preprocess(exp_df, surv_type="deephit", n_bins=32,
                                   subsample_seer=10000)
    return data["val_df"]["Source"].values


def bootstrap_ci(agg_base, agg_dann, sources, B=B, seed=SEED):
    """Within-cohort paired bootstrap of mean|SHAP| and its DANN−Baseline diff.

    Returns per-feature dicts for 'baseline', 'dann', and 'delta'.
    """
    n = len(agg_base)
    rng = np.random.RandomState(seed)
    uniq = np.unique(sources)
    draws_base = np.zeros((B, len(FEATURES)))
    draws_dann = np.zeros((B, len(FEATURES)))
    for i in range(B):
        idx = []
        for u in uniq:
            m = np.where(sources == u)[0]
            idx.append(m[rng.randint(0, len(m), size=len(m))])
        idx = np.concatenate(idx)
        draws_base[i] = np.abs(agg_base[idx]).mean(axis=0)
        draws_dann[i] = np.abs(agg_dann[idx]).mean(axis=0)
    out = {}
    for mode, draws in (("baseline", draws_base), ("dann", draws_dann)):
        out[mode] = {
            f: {
                "mean": round(float(draws[:, j].mean()), 5),
                "ci_low": round(float(np.percentile(draws[:, j], 2.5)), 5),
                "ci_high": round(float(np.percentile(draws[:, j], 97.5)), 5),
            }
            for j, f in enumerate(FEATURES)
        }
    out["delta_dann_minus_baseline"] = {
        f: {
            "mean": round(float((draws_dann[:, j] - draws_base[:, j]).mean()), 5),
            "ci_low": round(float(np.percentile(draws_dann[:, j] - draws_base[:, j], 2.5)), 5),
            "ci_high": round(float(np.percentile(draws_dann[:, j] - draws_base[:, j], 97.5)), 5),
            "ci_excludes_0": bool((np.percentile(draws_dann[:, j] - draws_base[:, j], 2.5) > 0)
                                  or (np.percentile(draws_dann[:, j] - draws_base[:, j], 97.5) < 0)),
        }
        for j, f in enumerate(FEATURES)
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="SHAP mean|SHAP| bootstrap CI (B/C)")
    parser.add_argument("--exps", nargs="+", default=list(EXP_KEYS),
                        choices=list(EXP_KEYS))
    args = parser.parse_args()

    all_out = {}
    rows = []
    for letter in args.exps:
        ek = EXP_KEYS[letter]
        agg_base = np.load(EXP_OUT / f"{ek}_agg_baseline.npy")
        agg_dann = np.load(EXP_OUT / f"{ek}_agg_dann.npy")
        sources = load_val_sources(ek)
        n_val = len(sources)
        assert len(agg_base) == n_val and len(agg_dann) == n_val, \
            f"[{ek}] agg/shap size mismatch"
        res = bootstrap_ci(agg_base, agg_dann, sources)
        all_out[ek] = {"n_val": n_val, "n_boot": B, "features": FEATURES, **res}
        log.info(f"[{ek}] n_val={n_val}:")
        for f in FEATURES:
            d = res["delta_dann_minus_baseline"][f]
            mark = " *" if d["ci_excludes_0"] else ""
            log.info(f"  Δ(D−B) {f:<6}= {d['mean']:+.5f} "
                     f"[{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]{mark}")
        for mode in ("baseline", "dann", "delta_dann_minus_baseline"):
            for f in FEATURES:
                v = res[mode][f]
                rows.append({
                    "experiment": ek, "mode": mode, "feature": f,
                    "mean_abs": v["mean"], "ci_low": v["ci_low"],
                    "ci_high": v["ci_high"],
                    "ci_excludes_0": v.get("ci_excludes_0"),
                })

    with open(EXP_OUT / "ci_summary.json", "w") as f:
        json.dump({"experiments": args.exps, "n_boot": B, "seed": SEED,
                   "results": all_out,
                   "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=2)

    # G6 figure: mean|SHAP| with bootstrap CI, Baseline vs DANN, per exp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows = len(args.exps)
    fig, axes = plt.subplots(1, n_rows, figsize=(6.2 * n_rows, 4.4), dpi=150)
    if n_rows == 1:
        axes = [axes]
    x = np.arange(len(FEATURES))
    for ax, letter in zip(axes, args.exps):
        ek = EXP_KEYS[letter]
        res = all_out[ek]
        b = [res["baseline"][f] for f in FEATURES]
        d = [res["dann"][f] for f in FEATURES]
        w = 0.35
        yerr_b = np.array([[v["mean"] - v["ci_low"], v["ci_high"] - v["mean"]]
                           for v in b]).T  # (2, n_features)
        yerr_d = np.array([[v["mean"] - v["ci_low"], v["ci_high"] - v["mean"]]
                           for v in d]).T
        ax.errorbar(x - w / 2, [v["mean"] for v in b], yerr=yerr_b,
                    fmt="o", ms=5, color="#4C72B0", capsize=3, label="Baseline")
        ax.errorbar(x + w / 2, [v["mean"] for v in d], yerr=yerr_d,
                    fmt="o", ms=5, color="#C44E52", capsize=3, label="DANN")
        ax.set_xticks(x); ax.set_xticklabels(FEATURES)
        ax.set_ylabel("mean |SHAP|")
        ax.set_title(f"{ek} (n_val={res['n_val']})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("SHAP feature importance with within-cohort bootstrap 95% CI",
                 fontsize=12)
    fig.tight_layout()
    out_fig = RESULTS_DIR / "figures" / "extended" / "G6_shap_ci_compare.png"
    fig.savefig(out_fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Figure → {out_fig}")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "table18_shap_ci.csv",
                              index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table18_shap_ci.csv'} "
             f"({len(rows)} rows)")

    # summary txt
    lines = ["\n" + "=" * 88,
             "SHAP BOOTSTRAP CI; mean|SHAP| (B/C val, within-cohort B=500)",
             "=" * 88]
    for ek in args.exps:
        res = all_out[EXP_KEYS[ek]]
        lines.append(f"\n{EXP_KEYS[ek]} (n_val={res['n_val']}):")
        lines.append(f"  {'feature':<7}{'Baseline mean[95%CI]':<32}"
                     f"{'DANN mean[95%CI]':<32}Δ(D−B) mean[95%CI]")
        for f in FEATURES:
            b, d, dd = res["baseline"][f], res["dann"][f], res["delta_dann_minus_baseline"][f]
            mark = " *" if dd["ci_excludes_0"] else ""
            lines.append(
                f"  {f:<7}{b['mean']:.4f}[{b['ci_low']:.4f},{b['ci_high']:.4f}]"
                f"{d['mean']:.4f}[{d['ci_low']:.4f},{d['ci_high']:.4f}]"
                f"{dd['mean']:+.4f}[{dd['ci_low']:+.4f},{dd['ci_high']:+.4f}]{mark}")
    lines.append("\n* = Δ CI excludes 0")
    lines.append("=" * 88)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary_ci.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {EXP_OUT}/ci_summary.json, summary_ci.txt, table18")
    log.info("\n✅ SHAP CI complete!")


if __name__ == "__main__":
    main()
