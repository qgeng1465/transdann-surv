#!/usr/bin/env python3
"""
19_exp_alpha_sensitivity.py; GRL α (gradient-reversal scale) sensitivity experiment
====================================================================================

Closes the reviewer's "GRL was not tuned" criticism: change α (the gradient-reversal
scale of GradientReversal) from the default 0→1 schedule to **three fixed levels**
{0.1, 0.5, 1.0}, retrain on experiments A/B/C, and compare against a **Baseline
re-run under the same code and seeds**.

If even **tuning α (taking the best over all α values)** cannot make DANN beat
Baseline, then the "GRL systematically fails" conclusion is robust to GRL's most
critical hyper-parameter.

Outputs (results/experiments/exp_alpha_sensitivity/):
  - results.json            per (exp, α, seed): best_val_cindex / domain-classifier accuracy, etc.
  - alpha_sensitivity.csv   summary table (mean±std over seeds)
  - summary.txt             human-readable summary
  - logs/19_exp_alpha_sensitivity.log

Usage:
  python3 scripts/19_exp_alpha_sensitivity.py                 # A/B/C × {0.1,0.5,1.0} × {42,43,44}
  python3 scripts/19_exp_alpha_sensitivity.py --alphas 0.1 1.0 --seeds 42   # custom
  python3 scripts/19_exp_alpha_sensitivity.py --experiments A  # run A only (for timing)
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
DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"

ALPHA_LOG_DIR = RESULTS_DIR / "experiments" / "exp_alpha_sensitivity" / "logs"
ALPHA_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ALPHA_LOG_DIR / "19_exp_alpha_sensitivity.log", mode="w"),
    ],
)
log = logging.getLogger("alpha_sweep")

# ---------------------------------------------------------------------------
# Import the training engine (02); filename starts with a digit, use importlib
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "train_engine", SCRIPTS_DIR / "02_train_dann_lihc.py"
)
train_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_engine)
EXPERIMENTS = train_engine.EXPERIMENTS
load_and_preprocess = train_engine.load_and_preprocess
train_model = train_engine.train_model

DEFAULT_ALPHAS = [0.1, 0.5, 1.0]
DEFAULT_SEEDS = [42, 43, 44]
ALPHA_LABEL = {0.1: "weak", 0.5: "mid", 1.0: "strong"}


def run_sweep(exp_key, df, alphas, seeds, epochs, patience, out_root):
    """Run the α-sweep for one experiment. Returns a list of per-run records."""
    exp_cfg = EXPERIMENTS[exp_key]
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    log.info(f"\n{'#'*70}")
    log.info(f"EXPERIMENT: {exp_cfg['name']}  (cohorts={exp_cfg['cohorts']})")
    log.info(f"Samples: {len(exp_df)}")
    log.info(f"{'#'*70}")

    # Preprocess ONCE per experiment; identical split for every α/seed
    data = load_and_preprocess(
        exp_df, surv_type="deephit", n_bins=32, subsample_seer=10000
    )

    base_cfg = {
        "surv_type": "deephit", "d_model": 128, "n_layers": 4,
        "dropout": 0.15, "lr": 5e-4, "n_bins": 32,
        "epochs": epochs, "patience": patience,
        "per_cohort_loss": True, "domain_weight_max": 0.3,
    }

    exp_out = out_root / exp_key
    os.makedirs(exp_out, exist_ok=True)

    records = []

    # ---- Baseline (no GRL); re-run under identical seeded code ----
    for seed in seeds:
        out_dir = exp_out / "baseline" / f"seed_{seed}"
        cfg = {**base_cfg, "domain_weight_max": 0.0, "seed": seed}
        log.info(f"\n>>> BASELINE seed={seed}")
        _, res, _ = train_model(cfg, data, mode="baseline", output_dir=out_dir)
        records.append({
            "experiment": exp_key, "alpha": None, "alpha_label": "baseline",
            "seed": seed,
            "best_val_cindex": res.get("best_val_cindex"),
            "best_epoch": res.get("best_epoch"),
            "per_cohort_cindex": res.get("per_cohort_cindex", {}),
            "domain_acc_peak": None,
            "domain_acc_final": None,
        })

    # ---- DANN with fixed α ----
    for alpha in alphas:
        for seed in seeds:
            out_dir = exp_out / f"alpha_{alpha}" / f"seed_{seed}"
            cfg = {**base_cfg, "alpha_fixed": alpha, "seed": seed}
            log.info(f"\n>>> DANN α={alpha} ({ALPHA_LABEL.get(alpha,'?')}) seed={seed}")
            _, res, _ = train_model(cfg, data, mode="dann", output_dir=out_dir)
            hist = res.get("history", {})
            dom_acc = [float(x) for x in hist.get("domain_acc", []) if x is not None]
            records.append({
                "experiment": exp_key, "alpha": alpha,
                "alpha_label": ALPHA_LABEL.get(alpha, "?"),
                "seed": seed,
                "best_val_cindex": res.get("best_val_cindex"),
                "best_epoch": res.get("best_epoch"),
                "per_cohort_cindex": res.get("per_cohort_cindex", {}),
                "domain_acc_peak": max(dom_acc) if dom_acc else None,
                "domain_acc_final": dom_acc[-1] if dom_acc else None,
            })

    # Save per-run detail
    with open(exp_out / "runs.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    return records


def aggregate(records):
    """Aggregate per-seed records into (exp, alpha) → mean/std stats."""
    rows = []
    by_key = {}
    for r in records:
        k = (r["experiment"], r["alpha_label"])
        by_key.setdefault(k, []).append(r)
    for (exp, label), rs in sorted(by_key.items()):
        cids = [x["best_val_cindex"] for x in rs if x["best_val_cindex"] is not None]
        mean = float(np.mean(cids)) if cids else None
        std = float(np.std(cids, ddof=1)) if len(cids) > 1 else 0.0
        rows.append({
            "experiment": exp, "alpha_label": label,
            "cindex_mean": mean, "cindex_std": std,
            "n_seeds": len(cids),
            "cindex_each": [round(x, 4) for x in cids],
            "domain_acc_peak_each": [
                round(x["domain_acc_peak"], 4) if x["domain_acc_peak"] is not None else None
                for x in rs
            ],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="GRL α sensitivity sweep")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C"],
                        help="experiments to run (A/B/C)")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS,
                        help="fixed α values (default: 0.1 0.5 1.0)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="seeds (default: 42 43 44)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    out_root = RESULTS_DIR / "experiments" / "exp_alpha_sensitivity"

    if not DATA_PATH.exists():
        log.error(f"Data not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    log.info(f"Loaded {len(df)} rows, {df['Source'].nunique()} cohorts")

    exp_keys = [f"{e}_" + {"A": "tcga_vs_seer", "B": "tcga_vs_external", "C": "all_cohorts"}[e]
                for e in args.experiments]

    all_records = []
    for ek in exp_keys:
        all_records.extend(run_sweep(
            ek, df, args.alphas, args.seeds, args.epochs, args.patience, out_root
        ))

    # ---- Aggregate & summarize ----
    summary = aggregate(all_records)
    with open(out_root / "alpha_sensitivity.csv", "w") as f:
        pd.DataFrame(summary).to_csv(f, index=False)

    out_json = {
        "alphas": args.alphas,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "all_runs": all_records,
    }
    with open(out_root / "results.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)

    # ---- Human-readable table ----
    lines = []
    lines.append("\n" + "=" * 96)
    lines.append("GRL α-SENSITIVITY SUMMARY  (fixed α ∈ %s, seeds %s)"
                 % (args.alphas, args.seeds))
    lines.append("=" * 96)
    lines.append(f"{'exp':<8}{'α':<10}{'mean C-idx':<14}{'±std':<10}{'per-seed':<24}{'dom-acc peak':<24}")
    for s in summary:
        rows = ""
        if s["cindex_each"]:
            rows = "[" + ", ".join(f"{x:.4f}" for x in s["cindex_each"]) + "]"
        dap = ""
        if s["domain_acc_peak_each"] and any(x is not None for x in s["domain_acc_peak_each"]):
            dap = "[" + ", ".join(f"{x:.3f}" if x is not None else "-" for x in s["domain_acc_peak_each"]) + "]"
        lines.append(
            f"{s['experiment']:<8}{s['alpha_label']:<10}"
            f"{s['cindex_mean']:.4f}{'':<4}{s['cindex_std']:.4f}{'':<4}{rows:<24}{dap}"
        )
    lines.append("=" * 96)

    # Max-over-α argument (reviewer-friendly: even best-tuned DANN ≤ baseline)
    lines.append("\nBest-over-α check (does ANY α rescue DANN?):")
    by_exp = {}
    for s in summary:
        by_exp.setdefault(s["experiment"], []).append(s)
    for exp, ss in by_exp.items():
        base = next((x for x in ss if x["alpha_label"] == "baseline"), None)
        danns = [x for x in ss if x["alpha_label"] != "baseline"]
        if base is None or not danns:
            continue
        bmean = base["cindex_mean"]
        best = max(danns, key=lambda x: x["cindex_mean"])
        lines.append(
            f"  {exp:<8} baseline={bmean:.4f} | DANN best-over-α "
            f"({best['alpha_label']}) = {best['cindex_mean']:.4f} "
            f"(Δ={best['cindex_mean'] - bmean:+.4f}) | "
            f"all-α means: " + ", ".join(
                f"{x['alpha_label']}={x['cindex_mean']:.4f}" for x in danns
            )
        )
    lines.append("=" * 96)

    txt = "\n".join(lines)
    log.info(txt)
    with open(out_root / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {out_root}/results.json, alpha_sensitivity.csv, summary.txt")
    log.info("\n✅ α-sensitivity sweep complete!")


if __name__ == "__main__":
    main()
