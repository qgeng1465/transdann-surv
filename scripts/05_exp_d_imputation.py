#!/usr/bin/env python3
"""
05_exp_d_imputation.py; Experiment D: Post-Imputation "Resurrection" Test
===========================================================================
Core hypothesis of the paper: *missingness pattern IS the dominant domain
fingerprint* that kills GRL. If true, then after completely imputing the
missing clinical fields, that fingerprint disappears and GRL should start
working (DANN ≥ Baseline).

Design (mirrors Experiment A, TCGA_LIHC vs US_SEER):
  1. Load TCGA_LIHC + US_SEER, same base filters as Exp A.
  2. Impute ALL missing values (KNN imputation on one-hot encoded
     Sex/Stage/Grade + scaled Age, pooled across both cohorts).
  3. QUANTIFY the fingerprint removal:
       - missingness matrix pre/post
       - domain-separability probe: GradientBoosting AUC/balanced-accuracy
         to predict cohort from features, pre vs post imputation.
  4. Retrain DANN (GRL) + Baseline (no GRL); identical hyperparameters to
     Exp A; and compare C-index, time-AUC, and the DANN domain-classifier
     accuracy trajectory.
  5. Diff D vs A: did removing missingness rescue GRL?

Outputs:
  - results/lihc_experiments/D_imputed_tcga_seer/{dann,baseline}/...
  - results/lihc_experiments/D_imputed_tcga_seer/comparison.json
  - results/D_imputation_probe.json
  - results/figures/extended/D_*.png
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from lihc_recon import BASE_DIR, build_data, _load_train_02
from transdann_utils import DEVICE

RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
EXP_DIR = RESULTS_DIR / "lihc_experiments" / "D_imputed_tcga_seer"
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CONT_FEATURES = ["Age"]
CAT_FEATURES = ["Sex", "Stage", "Grade"]
COHORTS = ["TCGA_LIHC", "US_SEER"]

STAGE_ORDER = ["I", "II", "III", "IV"]
GRADE_ORDER = [1.0, 2.0, 3.0, 4.0]


# ===================================================================
# 1. Load + filter (identical to Exp A)
# ===================================================================
def load_raw():
    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    return df[df["Source"].isin(COHORTS)].copy()


# ===================================================================
# 2. Imputation
# ===================================================================
def impute_knn(df):
    """KNN-impute Age/Sex/Stage/Grade (one-hot categorical + scaled Age)."""
    work = df[["Age", "Sex", "Stage", "Grade"]].copy()

    # Age: standardised continuous
    scaler_age = StandardScaler()
    age_std = scaler_age.fit_transform(work[["Age"]]).ravel()

    # Categorical one-hots (NaN → all-zero row)
    def onehot(series):
        cats = sorted(series.dropna().astype(str).unique())
        out = pd.DataFrame(
            np.zeros((len(series), len(cats))), index=series.index, dtype=float
        )
        for j, c in enumerate(cats):
            mask = series.astype(str) == c
            out.iloc[mask.values, j] = 1.0
        return out

    sex_oh = onehot(work["Sex"])
    stage_oh = onehot(work["Stage"])
    grade_oh = onehot(work["Grade"])

    M = np.column_stack([age_std, sex_oh.values, stage_oh.values, grade_oh.values])
    imp = KNNImputer(n_neighbors=7, weights="distance")
    M_imp = imp.fit_transform(M)

    # --- reconstruct ---
    age_imp = (M_imp[:, 0] * scaler_age.scale_[0] + scaler_age.mean_[0])

    def argmax_cat(block, labels):
        # imputed probs → nearest category; all ≤ tiny → fall back to most common
        out = []
        for row in block:
            if row.max() < 1e-6:
                out.append(np.nan)
            else:
                out.append(labels[int(np.argmax(row))])
        return out

    n_sex = sex_oh.shape[1]
    n_stage = stage_oh.shape[1]
    n_grade = grade_oh.shape[1]

    sex_imp = argmax_cat(M_imp[:, 1:1 + n_sex], sorted(work["Sex"].dropna().astype(str).unique()))
    stage_imp = argmax_cat(M_imp[:, 1 + n_sex:1 + n_sex + n_stage],
                           sorted(work["Stage"].dropna().astype(str).unique()))
    grade_imp = argmax_cat(M_imp[:, 1 + n_sex + n_stage:],
                           sorted(work["Grade"].dropna().astype(str).unique()))

    imputed = df.copy()
    imputed["Age"] = age_imp
    imputed["Sex"] = [str(v) for v in sex_imp]
    imputed["Stage"] = [str(v) for v in stage_imp]
    imputed["Grade"] = [str(v) for v in grade_imp]
    return imputed


# ===================================================================
# 2b. Reviewer point #4; imputation-strategy controls (Exp D′)
#
#   cross_cohort_knn : current Exp D imputation (KNN pooled across cohorts; #                      SEER's 100%-missing Grade is filled from TCGA neighbours;
#                      the imputer itself carries cross-cohort information).
#   within_cohort_median : each cohort filled with ITS OWN median/mode.  This is
#                      the honest control the reviewer asks for: if a field is
#                      structurally unobservable in a cohort (SEER Grade 100%),
#                      it stays missing; within-cohort imputation CANNOT erase
#                      that cell, so any surviving domain signal is legitimately
#                      attributable to population distribution, not leakage.
#   random_fill      : extreme control; NaNs replaced by random draws from the
#                      cohort's own observed values.  Removes any *systematic*
#                      filler, keeps only what the cohort itself can express.
# ===================================================================
def impute_within_median(df):
    """Fill each feature's NaNs with the cohort's own median (Age) / mode (cats).

    A feature with 100% missingness inside a cohort (e.g. SEER Grade) has no
    within-cohort estimate and is intentionally left missing; the missing
    indicator therefore survives in the probe, which is the whole point of the
    control.
    """
    imputed = df.copy()
    for coh in df["Source"].unique():
        coh_mask = (imputed["Source"] == coh)
        for col in ["Age"]:
            vals = df.loc[coh_mask, col]
            if vals.notna().any():
                med = vals.median()
                fill = coh_mask & imputed[col].isna()
                imputed.loc[fill, col] = med
        for col in ["Sex", "Stage", "Grade"]:
            vals = df.loc[coh_mask, col]
            if vals.notna().any():
                mode = vals.mode().iloc[0]
                fill = coh_mask & imputed[col].isna()
                imputed.loc[fill, col] = mode
    return imputed


def impute_random_fill(df, seed=42):
    """Fill NaNs with random draws (with replacement) from the cohort's own
    observed values of that feature.  Structurally unobservable cells (100%
    missing in a cohort) are left missing."""
    rng = np.random.RandomState(seed)
    imputed = df.copy()
    for coh in df["Source"].unique():
        coh_mask = (imputed["Source"] == coh)
        for col in ["Age", "Sex", "Stage", "Grade"]:
            miss = coh_mask & imputed[col].isna()
            if int(miss.sum()) == 0:
                continue
            pool = df.loc[coh_mask & df[col].notna(), col].values
            if len(pool) == 0:
                continue
            imputed.loc[miss, col] = pool[rng.randint(0, len(pool), size=int(miss.sum()))]
    return imputed


IMPUTE_STRATEGIES = {
    "cross_cohort_knn": impute_knn,
    "within_cohort_median": impute_within_median,
    "random_fill": impute_random_fill,
}


def impute_by_strategy(df, strategy):
    if strategy not in IMPUTE_STRATEGIES:
        raise ValueError(f"unknown impute strategy {strategy!r}; "
                         f"choose from {list(IMPUTE_STRATEGIES)}")
    return IMPUTE_STRATEGIES[strategy](df)


# ===================================================================
# 3. Domain-separability probe (fingerprint meter)
# ===================================================================
def probe_domain_sep(df, subsample_seer=10000):
    """Fit GradientBoosting to predict cohort from features.

    Returns {auc, balanced_acc} on a held-out 30% split.
    """
    p = df.copy()
    if "US_SEER" in p["Source"].values:
        seer_idx = p[p["Source"] == "US_SEER"].index
        if len(seer_idx) > subsample_seer:
            keep = np.random.RandomState(42).choice(seer_idx, subsample_seer, replace=False)
            p = p.drop(seer_idx.difference(pd.Index(keep))).copy()

    feats = pd.DataFrame(index=p.index)
    feats["Age"] = p["Age"].fillna(50.0).astype(float)
    for c in ["Sex", "Stage", "Grade"]:
        codes, _ = pd.factorize(p[c].fillna("__MISSING__").astype(str))
        feats[f"{c}_code"] = codes
        feats[f"{c}_known"] = p[c].notna().astype(int)

    X = feats.values.astype(float)
    y = (p["Source"] == "US_SEER").astype(int).values

    rng = np.random.RandomState(7)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[:int(0.7 * n)], idx[int(0.7 * n):]

    clf = GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.1,
        random_state=7, subsample=0.8,
    )
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], proba)
    bacc = balanced_accuracy_score(y[te], (proba >= 0.5).astype(int))
    return {"auc": round(float(auc), 4), "balanced_acc": round(float(bacc), 4)}


# ===================================================================
# 4. Main
# ===================================================================
def run_strategy_probe_comparison(raw, probe_before, miss_before):
    """Exp D′ (reviewer point #4): compare the domain-separability probe AUC
    under three imputation strategies; cross-cohort KNN (Exp D original),
    within-cohort median, and random fill.  Probe-only, no retraining."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = ["Age", "Sex", "Stage", "Grade"]
    strategies = list(IMPUTE_STRATEGIES.keys())
    out = {"before": probe_before,
           "missingness_before": miss_before,
           "strategies": {}}
    print("\n" + "=" * 72)
    print("EXP D′; imputation-strategy controls (domain-separability probe)")
    print("=" * 72)
    for strat in strategies:
        imputed = impute_by_strategy(raw, strat)
        miss_after = {c: float(imputed[c].isna().mean() * 100) for c in feats}
        probe_after = probe_domain_sep(imputed)
        seer_grade = miss_after.get("Grade")
        out["strategies"][strat] = {
            "probe": probe_after,
            "remaining_missingness": {k: round(v, 1) for k, v in miss_after.items()},
        }
        print(f"  {strat:<22s} probe AUC={probe_after['auc']:.4f}  "
              f"bal-acc={probe_after['balanced_acc']:.4f}  "
              f"remaining missingness={ {k: f'{v:.0f}%' for k, v in miss_after.items() if v > 0} or '0%' }")

    with open(RESULTS_DIR / "D_imputation_strategy_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR / 'D_imputation_strategy_probe.json'}")

    # ---- D0: probe AUC under each strategy ----
    labels = ["Raw", "KNN (cross-cohort)", "Median (within-cohort)", "Random fill"]
    aucs = [probe_before["auc"]] + [out["strategies"][s]["probe"]["auc"] for s in strategies]
    baccs = [probe_before["balanced_acc"]] + \
            [out["strategies"][s]["probe"]["balanced_acc"] for s in strategies]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    w = 0.36
    b1 = ax.bar(x - w / 2, aucs, w, label="Probe AUC", color="#4c72b0")
    b2 = ax.bar(x + w / 2, baccs, w, label="Balanced accuracy", color="#55a868")
    for xi, a, b in zip(x, aucs, baccs):
        ax.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Predict cohort from features")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp D′: domain separability after 3 imputation strategies")
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D0_strategy_comparison.png", dpi=200)
    plt.close(fig)
    print(f"Figure → {FIG_DIR / 'D0_strategy_comparison.png'}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--domain_weight", type=float, default=0.3)
    parser.add_argument("--skip_train", action="store_true", help="only probe + impute")
    parser.add_argument(
        "--impute_strategy", type=str, default="cross_cohort_knn",
        choices=list(IMPUTE_STRATEGIES) + ["all"],
        help="Imputation strategy for the resurrection test. 'all' runs the "
             "Exp-D′ probe comparison across all three strategies (probe-only). "
             "Default: cross_cohort_knn (the original Exp D imputer).")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT D; Post-Imputation Resurrection Test (TCGA_LIHC vs US_SEER)")
    print("=" * 70)

    raw = load_raw()
    print(f"\nSamples: {len(raw)}  |  {raw['Source'].value_counts().to_dict()}")

    # --- missingness before ---
    miss_before = {}
    for c in ["Age", "Sex", "Stage", "Grade"]:
        miss_before[c] = {coh: float(raw[raw["Source"] == coh][c].isna().mean() * 100)
                          for coh in COHORTS}

    print("\nMissingness BEFORE imputation (%):")
    print(pd.DataFrame(miss_before).round(1).to_string())

    # --- domain separability BEFORE ---
    probe_before = probe_domain_sep(raw)
    print(f"\nDomain-separability BEFORE imputation: {probe_before}")

    # ---- Exp D′ (reviewer point #4): compare all three imputation strategies ----
    if args.impute_strategy == "all":
        run_strategy_probe_comparison(raw, probe_before, miss_before)
        print("\nExp D′ probe comparison complete (no retraining in 'all' mode).\n"
              "To train on a single strategy: --impute_strategy {cross_cohort_knn, "
              "within_cohort_median, random_fill}")
        return

    # --- impute (single strategy) ---
    imputed = impute_by_strategy(raw, args.impute_strategy)
    miss_after = {c: float(imputed[c].isna().mean() * 100) for c in ["Age", "Sex", "Stage", "Grade"]}
    print(f"\nMissingness AFTER imputation ({args.impute_strategy}): {miss_after}")
    if args.impute_strategy == "cross_cohort_knn":
        assert all(v == 0 for v in miss_after.values()), "cross-cohort KNN left NaNs!"
    elif any(v > 0 for v in miss_after.values()):
        print("  (note: some missingness survives by design; structurally "
              "unobservable cells are left unfilled)")

    # --- domain separability AFTER ---
    probe_after = probe_domain_sep(imputed)
    print(f"Domain-separability AFTER imputation:  {probe_after}")

    probe = {
        "before": probe_before,
        "after": probe_after,
        "missingness_before": miss_before,
        "missingness_after": miss_after,
        "strategy": args.impute_strategy,
        "n_neighbors": 7,
        "method": IMPUTE_STRATEGIES[args.impute_strategy].__doc__.strip().splitlines()[0]
                  if args.impute_strategy in IMPUTE_STRATEGIES else str(args.impute_strategy),
    }
    with open(RESULTS_DIR / "D_imputation_probe.json", "w") as f:
        json.dump(probe, f, indent=2)
    print(f"\nProbe saved → {RESULTS_DIR / 'D_imputation_probe.json'}")

    # ---- figures: probe + missingness ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # D1: missingness before/after (per cohort × feature)
    feats = ["Age", "Sex", "Stage", "Grade"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (coh, label) in zip(axes, [(COHORTS[0], COHORTS[0]), (COHORTS[1], COHORTS[1])]):
        before = [miss_before[f][coh] for f in feats]
        after = [miss_after[f] for f in feats]
        x = np.arange(len(feats))
        ax.bar(x - 0.18, before, 0.36, label="before imputation", color="#dd8452")
        ax.bar(x + 0.18, after, 0.36, label="after imputation", color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels(feats)
        ax.set_title(f"{label} missingness (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
    fig.suptitle("Experiment D; KNN imputation removes the missingness fingerprint")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D1_missingness_before_after.png", dpi=150)
    plt.close(fig)

    # D2: domain separability probe
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    keys = ["before", "after"]
    aucs = [probe_before["auc"], probe_after["auc"]]
    baccs = [probe_before["balanced_acc"], probe_after["balanced_acc"]]
    x = np.arange(2)
    ax.bar(x - 0.18, aucs, 0.36, label="AUC", color="#4c72b0")
    ax.bar(x + 0.18, baccs, 0.36, label="balanced acc", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(["pre-imputation", "post-imputation"])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    for xi, a, b in zip(x, aucs, baccs):
        ax.text(xi - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("predict cohort from features")
    ax.set_title("Domain-separability fingerprint meter")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D2_domain_separability_probe.png", dpi=150)
    plt.close(fig)
    print(f"\nFigures: {FIG_DIR / 'D1_missingness_before_after.png'}, "
          f"{FIG_DIR / 'D2_domain_separability_probe.png'}")

    if args.skip_train:
        print("\n--skip_train: probe only. Done.")
        return

    # ===================================================================
    # Retrain on imputed data (same hyperparams as Experiment A)
    # ===================================================================
    t02 = _load_train_02()
    train_model = t02.train_model
    full_evaluation = t02.full_evaluation

    base_config = {
        "surv_type": "deephit",
        "d_model": 128, "n_layers": 4, "dropout": 0.15,
        "lr": 5e-4, "n_bins": 32,
        "epochs": args.epochs, "patience": 40,
        "per_cohort_loss": True, "domain_weight_max": args.domain_weight,
    }

    print("\n" + "=" * 70)
    print("Building data from IMPUTED frames (subsample_seer=10000, same as Exp A)")
    print("=" * 70)
    data = build_data(
        imputed, CONT_FEATURES, CAT_FEATURES,
        surv_type="deephit", n_bins=32, subsample_seer=10000,
    )
    print(f"Domains: {data['cohort_map']}")

    results_all = {}
    for mode, dw in [("dann", args.domain_weight), ("baseline", 0.0)]:
        cfg = {**base_config, "domain_weight_max": dw}
        out_dir = EXP_DIR / mode
        print(f"\n{'─'*60}\nTRAINING: {mode.upper()}\n{'─'*60}")
        model, res, _ = train_model(cfg, data, mode=mode, output_dir=out_dir)
        results_all[mode] = res

    # Comparison summary
    comp = {
        "experiment": "D_imputed_tcga_seer",
        "cohorts": COHORTS,
        "imputation": "KNNImputer k=7, pooled one-hot + scaled Age",
        "cohort_map": data["cohort_map"],
        "dann_val_cindex": results_all["dann"]["best_val_cindex"],
        "baseline_val_cindex": results_all["baseline"]["best_val_cindex"],
        "delta": round(float(results_all["dann"]["best_val_cindex"]
                             - results_all["baseline"]["best_val_cindex"]), 4),
        "dann_per_cohort": results_all["dann"]["per_cohort_cindex"],
        "baseline_per_cohort": results_all["baseline"]["per_cohort_cindex"],
        "dann_time_auc": results_all["dann"]["time_dependent_auc"],
        "baseline_time_auc": results_all["baseline"]["time_dependent_auc"],
        "probe": probe,
    }
    with open(EXP_DIR / "comparison.json", "w") as f:
        json.dump(comp, f, indent=2)

    # Figure: DANN domain-acc trajectory on imputed data (vs Exp A reference)
    hist_d = results_all["dann"]["history"]
    hist_a = json.load(open(RESULTS_DIR / "lihc_experiments" / "A_tcga_vs_seer" / "dann" / "results.json"))["history"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(hist_a["epoch"], hist_a["domain_acc"], "o-", ms=4, color="#c44e52",
            label="Exp A (missingness intact)")
    ax.plot(hist_d["epoch"], hist_d["domain_acc"], "s-", ms=4, color="#4c72b0",
            label="Exp D (imputed)")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("domain classifier accuracy")
    ax.set_title("DANN domain classifier: does imputation defuse it?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D3_domain_acc_trajectory.png", dpi=150)
    plt.close(fig)

    # Figure: C-index comparison (A original vs D imputed, DANN vs Baseline)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    exp_a = json.load(open(RESULTS_DIR / "lihc_experiments" / "A_tcga_vs_seer" / "comparison.json"))
    d_c = comp["dann_val_cindex"]; b_c = comp["baseline_val_cindex"]
    a_d = exp_a["dann_val_cindex"]; a_b = exp_a["baseline_val_cindex"]
    groups = ["A: missingness intact", "D: imputed"]
    dann = [a_d, d_c]
    base = [a_b, b_c]
    x = np.arange(2)
    ax.bar(x - 0.18, dann, 0.36, label="DANN (GRL)", color="#c44e52")
    ax.bar(x + 0.18, base, 0.36, label="Baseline", color="#4c72b0")
    for xi, da, ba in zip(x, dann, base):
        ax.text(xi - 0.18, da + 0.002, f"{da:.4f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, ba + 0.002, f"{ba:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("val C-index")
    ax.set_ylim(0.5, 0.75)
    ax.legend(fontsize=8)
    ax.set_title("Does removing missingness let GRL help? (val C-index)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D4_cindex_original_vs_imputed.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures: {FIG_DIR / 'D3_domain_acc_trajectory.png'}, "
          f"{FIG_DIR / 'D4_cindex_original_vs_imputed.png'}")
    print(f"\nComparison saved → {EXP_DIR / 'comparison.json'}")
    print(json.dumps({k: v for k, v in comp.items() if k != "probe"}, indent=2))
    print("\n✅ Experiment D complete!")


if __name__ == "__main__":
    main()
