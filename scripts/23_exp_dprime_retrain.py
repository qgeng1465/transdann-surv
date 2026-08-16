#!/usr/bin/env python3
"""
23_exp_dprime_retrain.py; Experiment D′ within-cohort-median full retrain (DANN vs Baseline)
=============================================================================================

Background
----------
In the revision, the reviewer questioned whether Exp D (cross-cohort KNN imputation)
carries cross-cohort information (P0-4). A three-strategy **probe** was already run
(`--impute_strategy all`, `results/D_imputation_strategy_probe.json`):
cross-cohort KNN → 0% missing → probe AUC 0.9603; within-cohort median / random
fill → SEER Grade stays 100% missing because it is structurally unobservable →
probe AUC 1.000 (domain fingerprint perfectly preserved). But the probe is only a
"fingerprint measurement"; it does not do a **full retrain** (a DANN vs Baseline
survival-prediction comparison).

This script closes that gap: on the **within-cohort median imputation**
(leakage-free control) data, following the same protocol as Exp D (SEER subsample
10000, 15% stratified validation set, DeepHit 32 bins, same engine and hyper-parameters),
it runs a full 3-seed retrain of DANN / Baseline and reports:
  1. best-val C-index (DANN vs Baseline, mean±std over seeds); whether the
     erasure of missingness is enough to revive GRL holds under the **retrain
     protocol**;
  2. DANN domain-classifier accuracy trajectory (peak / final); whether the
     structural fingerprint (SEER Grade missing-indicator) is still perfectly
     classified in real adversarial training after the within-cohort median
     (expected ≈100%);
  3. per-cohort C-index and time-dependent AUC.

Reference: Exp D original report (cross-cohort KNN, single seed): Δ=+0.0027, domain classifier 97.7%.

Outputs (results/experiments/exp_dprime_retrain/):
  - runs.json              per (mode, seed) records
  - dprime_summary.json    aggregate (mean±std)
  - summary.txt            human-readable summary
  - logs/23_exp_dprime_retrain.log

Usage:
  python3 scripts/23_exp_dprime_retrain.py                     # 3 seeds
  python3 scripts/23_exp_dprime_retrain.py --seeds 42          # single-seed smoke
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

EXP_OUT = RESULTS_DIR / "experiments" / "exp_dprime_retrain"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "23_exp_dprime_retrain.log", mode="w"),
    ],
)
log = logging.getLogger("dprime_retrain")

# ---------------------------------------------------------------------------
# Import the pieces we need
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
impute_within_median = exp_d_05.impute_within_median

from lihc_recon import build_data, CONT_FEATURES, CAT_FEATURES  # noqa: E402

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
DEFAULT_SEEDS = [42, 43, 44]
COHORTS = ["TCGA_LIHC", "US_SEER"]


def run_retrain(seeds, epochs=200, patience=40):
    if not DATA_PATH.exists():
        log.error(f"Data not found: {DATA_PATH}")
        sys.exit(1)

    # --- Same raw data as Exp D (TCGA_LIHC + US_SEER) ---
    raw = load_raw()
    log.info(f"Raw samples: {len(raw)}  |  {raw['Source'].value_counts().to_dict()}")

    # --- Within-cohort median imputation (leakage-free control) ---
    imputed = impute_within_median(raw)
    miss_after = {c: round(float(imputed[c].isna().mean() * 100), 1)
                  for c in CONT_FEATURES + CAT_FEATURES}
    log.info(f"Missingness after within-cohort median (%): {miss_after}")
    assert miss_after["Grade"] > 90, \
        "expected SEER Grade to remain structurally missing under within-cohort median"

    # --- Build data exactly like Exp D's retrain (build_data, SEER subsample 10000) ---
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
                "imputation": "within_cohort_median",
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

    # Per-cohort mean C-index (keyed by cohort name)
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
        "experiment": "Dprime_within_cohort_median_retrain",
        "imputation": "within_cohort_median (each cohort's own median/mode; "
                      "SEER Grade structurally unfillable, remains missing)",
        "cohorts": COHORTS,
        "seeds": seeds,
        "epochs": epochs,
        "patience": patience,
        "remaining_missingness": miss_after,
        "probe_context": {
            "within_cohort_median_probe_auc": 1.0,
            "cross_cohort_knn_probe_auc": 0.9603,
            "source": "results/D_imputation_strategy_probe.json",
        },
        "summary": summary,
        "all_runs": records,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXP_OUT / "runs.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    with open(EXP_OUT / "dprime_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    write_table_csv(summary, seeds)

    lines = []
    lines.append("\n" + "=" * 84)
    lines.append("EXP D′; WITHIN-COHORT MEDIAN IMPUTATION: FULL RETRAIN "
                 "(TCGA_LIHC vs US_SEER, %d seeds %s)" % (len(seeds), seeds))
    lines.append("=" * 84)
    lines.append(f"Remaining missingness after imputation: {miss_after}")
    lines.append(f"Probe AUC (context): within-cohort median = 1.000  |  "
                 f"cross-cohort KNN = 0.9603")
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
    log.info(f"\nSaved: {EXP_OUT}/runs.json, dprime_summary.json, summary.txt")
    log.info("\n✅ Exp D′ within-cohort-median full retrain complete!")
    return out


TABLE_DIR = RESULTS_DIR / "tables"


def write_table_csv(summary, seeds):
    """Write results/tables/table10_dprime_retrain.csv (same pattern as table8/9)."""
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
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table10_dprime_retrain.csv", index=False)
    log.info(f"Table → {TABLE_DIR / 'table10_dprime_retrain.csv'} ({len(rows)} rows, "
             f"seeds={seeds})")


def summarize_existing(seeds):
    """Re-aggregate summary.txt / dprime_summary.json from an existing runs.json
    (no re-training). Used after code edits that change only labels/aggregation."""
    records = json.load(open(EXP_OUT / "runs.json"))
    imputed = impute_within_median(load_raw())
    miss_after = {c: round(float(imputed[c].isna().mean() * 100), 1)
                  for c in CONT_FEATURES + CAT_FEATURES}
    data = build_data(imputed, CONT_FEATURES, CAT_FEATURES,
                      surv_type="deephit", n_bins=32, subsample_seer=10000)
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
            vals = [float(r["per_cohort_cindex"][str(dom_id)])
                    for r in records if r["mode"] == mode
                    and str(dom_id) in r["per_cohort_cindex"]]
            per_coh[rev[dom_id]] = round(float(np.mean(vals)), 4) if vals else None
        summary[f"{mode}_per_cohort_mean"] = per_coh
    out = {
        "experiment": "Dprime_within_cohort_median_retrain",
        "imputation": "within_cohort_median (each cohort's own median/mode; "
                      "SEER Grade structurally unfillable, remains missing)",
        "cohorts": COHORTS,
        "seeds": seeds,
        "epochs": None, "patience": None,
        "remaining_missingness": miss_after,
        "probe_context": {
            "within_cohort_median_probe_auc": 1.0,
            "cross_cohort_knn_probe_auc": 0.9603,
            "source": "results/D_imputation_strategy_probe.json",
        },
        "summary": summary,
        "all_runs": records,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXP_OUT / "dprime_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    write_table_csv(summary, seeds)
    lines = ["\n" + "=" * 84,
             f"EXP D′; WITHIN-COHORT MEDIAN: FULL RETRAIN SUMMARY "
             f"(re-aggregated from runs.json, seeds {seeds})",
             "=" * 84,
             f"Remaining missingness after imputation: {miss_after}",
             f"Δ (DANN − Baseline) = {summary['delta_mean']:+.4f}",
             f"DANN domain-classifier accuracy (final epoch, per seed): "
             f"{summary['dann_domain_acc_final_each']}"
             f"  (mean={summary['dann_domain_acc_final_mean']:.4f})",
             "Per-cohort C-index on FULL dev (train+val, engine metric; mean over seeds):"]
    for mode in ["dann", "baseline"]:
        lines.append(f"  {mode:<10}" + ", ".join(
            f"{k}={v:.4f}" for k, v in summary[f"{mode}_per_cohort_mean"].items()))
    lines.append("=" * 84)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Re-aggregated {EXP_OUT}/summary.txt + dprime_summary.json "
             f"from {len(records)} runs (no re-training).")


def main():
    parser = argparse.ArgumentParser(description="Exp D′ within-cohort median full retrain")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--summarize-only", action="store_true",
                        help="re-aggregate from existing runs.json (no training)")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_existing(args.seeds)
    else:
        run_retrain(args.seeds, args.epochs, args.patience)


if __name__ == "__main__":
    main()
