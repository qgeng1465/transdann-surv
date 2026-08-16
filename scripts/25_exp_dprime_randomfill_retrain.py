#!/usr/bin/env python3
"""
25_exp_dprime_randomfill_retrain.py; Experiment D′ random-fill imputation full retrain
=======================================================================================

Background
----------
The three-strategy D′ probe (results/D_imputation_strategy_probe.json) shows the
domain separability under each imputation:
cross-cohort KNN → 0% missing → probe AUC 0.9603; within-cohort median / random
fill → SEER Grade stays ~100% missing because it is structurally unobservable →
probe AUC 1.000 (domain fingerprint perfectly preserved). `23_..._retrain.py`
already did the **full retrain** under **within-cohort median** (Δ=−0.0007, DANN
domain classifier ends at ≈1.0). This script applies the same full-retrain protocol
to the **random-fill** imputed data, completing the last of the three-strategy
retrains, consistent with the within-cohort-median conclusion (expected: DANN still
gives no survival benefit, domain classifier ≈1.0), thereby fully closing reviewer
P0-4 under the **retrain protocol**.

Random-fill definition (05_exp_d_imputation.impute_random_fill): each cohort fills
its missing cells with its own observed values (random draw with replacement);
structurally unobservable cells (SEER Grade 100% missing; no observed values exist
within that cohort) **remain missing**: so after random fill Grade is still ~99.7%
missing, consistent with within-cohort median, and the missing-indicator is fully
preserved in both the probe and the adversarial training. The filled values of
Age/Sex/Stage differ between the two strategies (median/mode vs random draw), but
the domain fingerprint (missing indicator) is unaffected.

Reference: Exp D′R (within-cohort-median retrain, table10): Δ=−0.0007, DANN domain classifier ends at [0.9997,1,1].

Outputs (results/experiments/exp_dprime_randomfill_retrain/):
  - runs.json / randomfill_summary.json / summary.txt
  - results/tables/table12_dprime_randomfill.csv
  - logs/25_exp_dprime_randomfill_retrain.log

Usage:
  python3 scripts/25_exp_dprime_randomfill_retrain.py            # 3 seeds
  python3 scripts/25_exp_dprime_randomfill_retrain.py --seeds 42 # single-seed smoke
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

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_dprime_randomfill_retrain"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "25_exp_dprime_randomfill_retrain.log", mode="w"),
    ],
)
log = logging.getLogger("dprime_randomfill_retrain")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "train_engine", SCRIPTS_DIR / "02_train_dann_lihc.py"
)
train_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_engine)
train_model = train_engine.train_model

spec05 = importlib.util.spec_from_file_location(
    "exp_d_05", SCRIPTS_DIR / "05_exp_d_imputation.py"
)
exp_d_05 = importlib.util.module_from_spec(spec05)
spec05.loader.exec_module(exp_d_05)
load_raw = exp_d_05.load_raw
impute_random_fill = exp_d_05.impute_random_fill
probe_domain_sep = exp_d_05.probe_domain_sep

from lihc_recon import build_data, CONT_FEATURES, CAT_FEATURES  # noqa: E402

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
DEFAULT_SEEDS = [42, 43, 44]
COHORTS = ["TCGA_LIHC", "US_SEER"]
RANDOM_FILL_SEED = 42  # same default seed as impute_random_fill()


def run_retrain(seeds, epochs=200, patience=40):
    if not DATA_PATH.exists():
        log.error(f"Data not found: {DATA_PATH}")
        sys.exit(1)

    raw = load_raw()
    log.info(f"Raw samples: {len(raw)}  |  {raw['Source'].value_counts().to_dict()}")

    # --- Random-fill imputation (leakage-free control, same protocol as D′R) ---
    imputed = impute_random_fill(raw, seed=RANDOM_FILL_SEED)
    miss_after = {c: round(float(imputed[c].isna().mean() * 100), 1)
                  for c in CONT_FEATURES + CAT_FEATURES}
    log.info(f"Missingness after random-fill (%): {miss_after}")
    assert miss_after["Grade"] > 90, \
        "expected SEER Grade to remain structurally missing under random fill"

    # Fingerprint meter on the exact imputed frame (probe, held-out 30% split)
    probe_after = probe_domain_sep(imputed)
    log.info(f"Domain-separability probe after random-fill: {probe_after}")

    data = build_data(imputed, CONT_FEATURES, CAT_FEATURES,
                      surv_type="deephit", n_bins=32, subsample_seer=10000)
    log.info(f"Domains: {data['cohort_map']}")
    log.info(f"Train: {len(data['train_df'])}, Val: {len(data['val_df'])}")

    base_cfg = {
        "surv_type": "deephit", "d_model": 128, "n_layers": 4,
        "dropout": 0.15, "lr": 5e-4, "n_bins": 32,
        "epochs": epochs, "patience": patience,
        "per_cohort_loss": True, "domain_weight_max": 0.3,
    }

    records = []
    for seed in seeds:
        for mode in ["dann", "baseline"]:
            out_dir = EXP_OUT / "runs" / f"seed_{seed}" / mode
            cfg = {**base_cfg, "domain_weight_max": (0.3 if mode == "dann" else 0.0),
                   "seed": seed}
            log.info(f"\n>>> {mode.upper()} seed={seed}")
            _, res, _ = train_model(cfg, data, mode=mode, output_dir=out_dir)
            hist = res.get("history", {})
            dom_acc = [float(x) for x in hist.get("domain_acc", []) if x is not None]
            records.append({
                "mode": mode,
                "seed": seed,
                "imputation": "random_fill",
                "best_val_cindex": res.get("best_val_cindex"),
                "best_epoch": res.get("best_epoch"),
                # normalise per-cohort keys to str (engine returns int keys in-memory)
                "per_cohort_cindex": {str(k): v for k, v in
                                      res.get("per_cohort_cindex", {}).items()},
                "time_dependent_auc": res.get("time_dependent_auc", {}),
                "domain_acc_peak": max(dom_acc) if dom_acc else None,
                "domain_acc_final": dom_acc[-1] if dom_acc else None,
            })

    # --- Aggregate ---
    summary = {}
    for mode in ["dann", "baseline"]:
        cids = [r["best_val_cindex"] for r in records if r["mode"] == mode]
        summary[mode] = {
            "cindex_mean": round(float(np.mean(cids)), 4) if cids else None,
            "cindex_std": round(float(np.std(cids, ddof=1)), 4) if len(cids) > 1 else 0.0,
            "cindex_each": [round(x, 4) for x in cids],
        }
    summary["delta_mean"] = (round(float(summary["dann"]["cindex_mean"] -
                                         summary["baseline"]["cindex_mean"]), 4)
                             if summary["dann"]["cindex_mean"] is not None
                             and summary["baseline"]["cindex_mean"] is not None else None)
    d_acc = [r["domain_acc_final"] for r in records if r["mode"] == "dann"
             and r["domain_acc_final"] is not None]
    summary["dann_domain_acc_final_each"] = [round(x, 4) for x in d_acc]
    summary["dann_domain_acc_final_mean"] = round(float(np.mean(d_acc)), 4) if d_acc else None

    rev = {v: k for k, v in data["cohort_map"].items()}
    for mode in ["dann", "baseline"]:
        per_coh = {}
        for dom_id in sorted(rev.keys()):
            vals = []
            for r in records:
                if r["mode"] != mode:
                    continue
                v = r["per_cohort_cindex"].get(str(dom_id))
                if v is not None:
                    vals.append(float(v))
            per_coh[rev[dom_id]] = (round(float(np.mean(vals)), 4) if vals else None)
        summary[f"{mode}_per_cohort_mean"] = per_coh

    out = {
        "experiment": "Dprime_random_fill_retrain",
        "imputation": "random_fill (each cohort's own observed values, with "
                      "replacement; SEER Grade structurally unfillable, remains missing)",
        "random_fill_seed": RANDOM_FILL_SEED,
        "cohorts": COHORTS,
        "seeds": seeds,
        "epochs": epochs,
        "patience": patience,
        "remaining_missingness": miss_after,
        "probe_after_random_fill": probe_after,
        "probe_context": {
            "random_fill_probe_auc": 1.0,
            "within_cohort_median_probe_auc": 1.0,
            "cross_cohort_knn_probe_auc": 0.9603,
            "source": "results/D_imputation_strategy_probe.json",
        },
        "comparison_dprime_within_median": {
            "delta": -0.0007,
            "dann_domain_acc_final_mean": 0.9999,
            "source": "results/experiments/exp_dprime_retrain/ (table10)",
        },
        "summary": summary,
        "all_runs": records,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXP_OUT / "runs.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    with open(EXP_OUT / "randomfill_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    write_table_csv(summary, seeds)

    lines = []
    lines.append("\n" + "=" * 84)
    lines.append("EXP D′; RANDOM-FILL IMPUTATION: FULL RETRAIN "
                 "(TCGA_LIHC vs US_SEER, %d seeds %s)" % (len(seeds), seeds))
    lines.append("=" * 84)
    lines.append(f"Remaining missingness after random fill: {miss_after}")
    lines.append(f"Domain-separability probe after random fill: {probe_after}")
    lines.append(f"Probe AUC (context): random_fill = 1.000  |  "
                 f"within-cohort median = 1.000  |  cross-cohort KNN = 0.9603")
    lines.append("-" * 84)
    lines.append(f"{'mode':<10}{'C-index mean':<14}{'±std':<10}{'per-seed'}")
    for mode in ["dann", "baseline"]:
        s = summary[mode]
        rows = "[" + ", ".join(f"{x:.4f}" for x in s["cindex_each"]) + "]"
        lines.append(f"{mode:<10}{s['cindex_mean']:.4f}{'':<4}{s['cindex_std']:.4f}"
                     f"{'':<4}{rows}")
    lines.append(f"\nΔ (DANN − Baseline) = {summary['delta_mean']:+.4f}")
    lines.append(f"DANN domain-classifier accuracy (final epoch, per seed): "
                 f"{summary['dann_domain_acc_final_each']}"
                 f"  (mean={summary['dann_domain_acc_final_mean']:.4f})")
    lines.append("-" * 84)
    lines.append("Per-cohort C-index on FULL dev (train+val, engine metric; mean over seeds):")
    for mode in ["dann", "baseline"]:
        lines.append(f"  {mode:<10}" + ", ".join(
            f"{k}={v:.4f}" for k, v in summary[f"{mode}_per_cohort_mean"].items()))
    lines.append("=" * 84)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {EXP_OUT}/runs.json, randomfill_summary.json, summary.txt")
    log.info("\n✅ Exp D′ random-fill full retrain complete!")
    return out


TABLE_DIR = RESULTS_DIR / "tables"


def write_table_csv(summary, seeds):
    """Write results/tables/table12_dprime_randomfill.csv (same pattern as table10)."""
    rows = []
    for mode in ["dann", "baseline"]:
        s = summary[mode]
        rows.append({
            "mode": mode,
            "cindex_mean": s["cindex_mean"],
            "cindex_std": s["cindex_std"],
            "cindex_each": "[" + ", ".join(f"{x:.4f}" for x in s["cindex_each"]) + "]",
            "TCGA_LIHC_per_cohort": summary.get(f"{mode}_per_cohort_mean", {}).get("TCGA_LIHC"),
            "US_SEER_per_cohort": summary.get(f"{mode}_per_cohort_mean", {}).get("US_SEER"),
        })
    rows.append({
        "mode": "delta_dann_minus_baseline",
        "cindex_mean": summary["delta_mean"],
        "cindex_std": None,
        "cindex_each": str([round(a - b, 4) for a, b in zip(
            summary["dann"]["cindex_each"], summary["baseline"]["cindex_each"])]),
    })
    rows.append({
        "mode": "dann_domain_acc_final_mean",
        "cindex_mean": summary["dann_domain_acc_final_mean"],
        "cindex_std": None,
        "cindex_each": str([round(x, 4) for x in summary["dann_domain_acc_final_each"]]),
    })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table12_dprime_randomfill.csv", index=False)
    log.info(f"Table → {TABLE_DIR / 'table12_dprime_randomfill.csv'} ({len(rows)} rows, "
             f"seeds={seeds})")


def main():
    parser = argparse.ArgumentParser(description="Exp D′ random-fill full retrain")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    run_retrain(args.seeds, args.epochs, args.patience)


if __name__ == "__main__":
    main()
