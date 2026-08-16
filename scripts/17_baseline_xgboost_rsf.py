#!/usr/bin/env python3
"""
17_baseline_xgboost_rsf.py; Reviewer point #5: is the Transformer the bottleneck?
==================================================================================
The reviewer argues that C-index ≈ 0.63 is a low ceiling, and that DANN might
gain on a *stronger* baseline (e.g. with AFP / Child-Pugh features reaching
0.75+).  We cannot add unmeasured clinical variables, but we CAN test whether
the 0.63 plateau is an artefact of the neural backbone or a property of the
4-variable feature set itself.

This script trains two classical survival baselines on the EXACT same
4 features and the EXACT same train/val split as Experiments A/B/C:

    XGBoost survival (Cox objective); if `xgboost` is installed
    Random Survival Forest (sksurv); always available (scikit-survival)

It then reports the validation C-index and time-dependent AUC (12/36/60 mo)
against the Transformer ERM value saved in results/lihc_experiments/*/.

Interface with existing code
---------------------------
  - `lihc_recon.build_data` is reused verbatim, so the SEER downsampling
    (RandomState(42) → 10,000), the stratified per-cohort 85/15 split and the
    label encodings are bit-identical to the neural runs.
  - Categoricals are re-one-hot-encoded for the trees (trees should not treat
    Stage/Grade codes as ordinal).

Outputs
-------
  results/experiments/baseline_trees/baseline_trees.json
  results/tables/table7_tree_baselines.csv
  results/figures/extended/T1_tree_vs_transformer.png

Run:  python scripts/17_baseline_xgboost_rsf.py [--experiments A B C] [--seeds 42]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lihc_recon import BASE_DIR, build_data, EXPERIMENTS

from lifelines.utils import concordance_index

RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = RESULTS_DIR / "experiments" / "baseline_trees"
TABLE_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CONT_FEATURES = ["Age"]
CAT_FEATURES = ["Sex", "Stage", "Grade"]
EXP_KEYS = {"A": "A_tcga_vs_seer", "B": "B_tcga_vs_external", "C": "C_all_cohorts"}


def _try_xgboost():
    try:
        import xgboost
        return xgboost
    except ImportError:
        return None


def _onehot(df, columns=None):
    """Age (as-is, scaling is irrelevant for trees) + one-hot categoricals.

    If `columns` is given (fitted on train), the frame is reindexed to exactly
    those columns so train/val have identical feature matrices.
    """
    X = pd.DataFrame(index=df.index)
    X["Age"] = df["Age"].fillna(df["Age"].median()).astype(float)
    for c in CAT_FEATURES:
        dummies = pd.get_dummies(df[c].fillna("__MISSING__").astype(str),
                                 prefix=c, dtype=float)
        X = X.join(dummies)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def _cindex(times, events, risk):
    """C-index robust to risk-sign convention (higher risk = shorter survival)."""
    if len(times) < 2 or events.sum() < 1:
        return float("nan")
    c1 = concordance_index(times, -risk, events)
    c2 = concordance_index(times, risk, events)
    return float(max(c1, c2))


def _time_auc(times, events, risk):
    from transdann_utils import time_dependent_auc
    auc = time_dependent_auc(times, events, risk, eval_times=(12, 36, 60))
    return {str(k): round(float(v), 4) if not np.isnan(v) else None
            for k, v in auc.items()}


def run_letter(letter, df, XGBSurvival, seed=42):
    exp_key = EXP_KEYS[letter]
    cfg = EXPERIMENTS[exp_key]
    exp_df = df[df["Source"].isin(cfg["cohorts"])].copy()
    data = build_data(exp_df, CONT_FEATURES, CAT_FEATURES, surv_type="deephit",
                      n_bins=32, subsample_seer=10000)
    tr, va = data["train_df"], data["val_df"]

    X_tr = _onehot(tr)
    cols = list(X_tr.columns)
    X_tr, X_va = X_tr.values, _onehot(va, columns=cols).values
    t_tr, e_tr = tr["Survival_Months"].values.astype(float), tr["Vital_Status"].values.astype(int)
    t_va, e_va = va["Survival_Months"].values.astype(float), va["Vital_Status"].values.astype(int)

    out = {"experiment": letter, "n_train": len(tr), "n_val": len(va)}

    # ---------- XGBoost survival (Cox, native API) ----------
    if XGBSurvival is not None:
        # survival:cox labels: +time for events, −time for censored
        label = np.where(e_tr == 1, t_tr, -t_tr)
        dtr = XGBSurvival.DMatrix(X_tr, label=label)
        dte = XGBSurvival.DMatrix(X_va)
        bst = XGBSurvival.train(
            {"objective": "survival:cox", "max_depth": 4, "eta": 0.05,
             "subsample": 0.8, "seed": seed, "nthread": -1},
            dtr, num_boost_round=200, evals=[(dtr, "train")],
            early_stopping_rounds=20, verbose_eval=False)
        risk_xgb = bst.predict(dte)  # log hazard ratio (higher = worse)
        out["xgboost"] = {
            "cindex_val": round(_cindex(t_va, e_va, risk_xgb), 4),
            "time_auc": _time_auc(t_va, e_va, risk_xgb),
            "n_boost_round": bst.best_iteration,
        }
    else:
        out["xgboost"] = {"error": "xgboost not installed; install with pip install xgboost"}

    # ---------- Random Survival Forest ----------
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    y_tr = Surv.from_arrays(tr["Vital_Status"].values.astype(bool),
                            tr["Survival_Months"].values)
    rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=10,
                               max_features="sqrt", random_state=seed, n_jobs=-1)
    rsf.fit(X_tr, y_tr)
    risk_rsf = rsf.predict(X_va)
    out["rsf"] = {
        "cindex_val": round(_cindex(t_va, e_va, risk_rsf), 4),
        "time_auc": _time_auc(t_va, e_va, risk_rsf),
    }

    # ---------- Transformer ERM reference (from saved comparison.json) ----------
    comp = json.load(open(RESULTS_DIR / "lihc_experiments" / exp_key / "comparison.json"))
    out["transformer_erm_cindex"] = comp.get("baseline_val_cindex")
    out["transformer_dann_cindex"] = comp.get("dann_val_cindex")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    XGBSurvival = _try_xgboost()
    if XGBSurvival is None:
        print("⚠  xgboost not installed; RSF only (pip install xgboost for the Cox baseline).")

    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()

    results = {letter: run_letter(letter, df, XGBSurvival, seed=args.seed)
               for letter in args.experiments}
    with open(OUT_DIR / "baseline_trees.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------- CSV table ----------
    rows = []
    for letter, r in results.items():
        rows.append({"experiment": letter,
                     "model": "Transformer-ERM", "cindex_val": r["transformer_erm_cindex"],
                     "note": "TransDANNSurvV3, 4×8-head, DeepHit 32-bin"})
        rows.append({"experiment": letter,
                     "model": "Transformer-DANN", "cindex_val": r["transformer_dann_cindex"],
                     "note": "same backbone + GRL"})
        for model in ("xgboost", "rsf"):
            m = r.get(model, {})
            if "cindex_val" in m:
                rows.append({"experiment": letter, "model": model.upper(),
                             "cindex_val": m["cindex_val"],
                             "note": "4 features, same train/val split"})
            else:
                rows.append({"experiment": letter, "model": model.upper(),
                             "cindex_val": None, "note": m.get("error", "")})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table7_tree_baselines.csv", index=False)

    # ---------- Figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    letters = args.experiments
    x = np.arange(len(letters))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    def _get(l, k):
        r = results.get(l, {})
        v = r.get(k)
        return float(v) if isinstance(v, (int, float)) else None
    for j, (key, color, lab) in enumerate([
            ("transformer_erm_cindex", "#1F77B4", "Transformer ERM"),
            ("transformer_dann_cindex", "#D62728", "Transformer DANN"),
            ("rsf", "#2CA02C", "Random Survival Forest")]):
        vals = [_get(l, key) if key != "rsf" else results[l]["rsf"]["cindex_val"]
                for l in letters]
        ax.bar(x + (j - 1) * w, vals, w, label=lab, color=color,
               edgecolor="white", linewidth=0.3)
    if XGBSurvival is not None:
        vals = [results[l]["xgboost"]["cindex_val"] for l in letters]
        ax.bar(x + 1.5 * w, vals, w, label="XGBoost-Cox", color="#FF7F0E",
               edgecolor="white", linewidth=0.3)
    ax.axhline(0.63, color="grey", ls="--", lw=0.8)
    ax.text(len(letters) - 0.5, 0.6308, "≈0.63 (4-feature ceiling)", fontsize=7,
            color="#555", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"Exp {l}" for l in letters])
    ax.set_ylabel("Validation C-index")
    ax.set_ylim(0.55, 0.70)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "T1_tree_vs_transformer.png", dpi=200)
    plt.close(fig)

    print(f"\n{'='*70}\nTree baselines vs Transformer (val C-index)\n{'='*70}")
    for letter in args.experiments:
        r = results[letter]
        print(f"  Exp {letter}: ERM={r['transformer_erm_cindex']}  "
              f"DANN={r['transformer_dann_cindex']}  "
              f"RSF={r['rsf'].get('cindex_val')}  "
              f"XGB={r['xgboost'].get('cindex_val') if XGBSurvival else 'n/a'}")
    print(f"\nTable → {TABLE_DIR / 'table7_tree_baselines.csv'}")
    print(f"JSON  → {OUT_DIR / 'baseline_trees.json'}")


if __name__ == "__main__":
    main()
