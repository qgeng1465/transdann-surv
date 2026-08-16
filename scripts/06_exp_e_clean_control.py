#!/usr/bin/env python3
"""
06_exp_e_clean_control.py; Experiment E: Clean-Data Control (Level 0)
=======================================================================
Goal: prove GRL *itself* is not broken; on data WITHOUT extreme missingness,
the domain-adversarial mechanism should behave normally (moderate domain
classifier accuracy, no instant 97%+ fingerprint, and if the missingness
hypothesis holds, DANN ≥ Baseline).

Uses the classic clean-cancer transfer pair the user proposed:
  - Source-ish: TCGA-BRCA   (harmonized_BRCA.csv,   1,098 patients)
  - Target-ish: METABRIC    (harmonized_metabric.csv, 2,509 patients)
Both breast-cancer cohorts with near-complete clinical fields.

Design (mirrors Exp A/B/C):
  - Features: Age (continuous), Sex (categorical), Stage (categorical,
    harmonized to 0–4).  Grade is EXCLUDED: 100% missing in TCGA-BRCA; including it would re-inject a missingness fingerprint.
  - Same TransDANNSurvV3 + DeepHit config, per-cohort loss, domain_weight=0.3.
  - Track domain classifier accuracy (should stay far below Exp A's 97%).
  - Compare C-index, time-AUC: DANN vs Baseline.

Outputs:
  - results/lihc_experiments/E_brca_metabric/{dann,baseline}/...
  - results/lihc_experiments/E_brca_metabric/comparison.json
  - results/E_clean_probe.json
  - results/figures/extended/E_*.png
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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from lihc_recon import BASE_DIR, build_data, _load_train_02

RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
EXP_DIR = RESULTS_DIR / "lihc_experiments" / "E_brca_metabric"
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CONT_FEATURES = ["Age"]
CAT_FEATURES = ["Sex", "Stage"]

BRCA_PATH = BASE_DIR / "data_processed" / "harmonized_BRCA.csv"
METABRIC_PATH = BASE_DIR / "data_processed" / "harmonized_metabric.csv"


# ===================================================================
# 1. Load + harmonize
# ===================================================================
def normalize_stage(x):
    """Unify stage encodings: TCGA 'IA/IIB/...' vs METABRIC numeric 1-4."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return int(v) if 0 <= v <= 4 else np.nan
    s = str(x).strip().upper()
    if s in ("0",):
        return 0
    if s == "X":
        return np.nan
    if s == "IV":
        return 4
    if s.startswith("III"):
        return 3
    if s.startswith("II"):
        return 2
    if s.startswith("I"):
        return 1
    return np.nan


def load_breast():
    b = pd.read_csv(BRCA_PATH)
    m = pd.read_csv(METABRIC_PATH)

    def to_common(df, source):
        out = pd.DataFrame({
            "Source": source,
            "Age": pd.to_numeric(df["Age"], errors="coerce"),
            "Sex": pd.to_numeric(df["Sex"], errors="coerce"),
            "Stage": df["Stage"].map(normalize_stage),
            "Survival_Months": pd.to_numeric(df["Survival_Months"], errors="coerce"),
            "Vital_Status": pd.to_numeric(df["Vital_Status"], errors="coerce"),
        })
        return out

    b = to_common(b, "TCGA-BRCA")
    m = to_common(m, "METABRIC")
    df = pd.concat([b, m], ignore_index=True)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    return df


# ===================================================================
# 2. Domain-separability probe (same "fingerprint meter" as Exp D)
# ===================================================================
def probe_domain_sep(df):
    feats = pd.DataFrame(index=df.index)
    feats["Age"] = df["Age"].fillna(50.0).astype(float)
    for c in ["Sex", "Stage"]:
        codes, _ = pd.factorize(df[c].fillna("__MISSING__").astype(str))
        feats[f"{c}_code"] = codes
        feats[f"{c}_known"] = df[c].notna().astype(int)
    X = feats.values.astype(float)
    y = (df["Source"] == "TCGA-BRCA").astype(int).values
    rng = np.random.RandomState(7)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[:int(0.7 * n)], idx[int(0.7 * n):]
    clf = GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.1, random_state=7, subsample=0.8,
    )
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    return {
        "auc": round(float(roc_auc_score(y[te], proba)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y[te], (proba >= 0.5).astype(int))), 4),
    }


# ===================================================================
# 3. Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--domain_weight", type=float, default=0.3)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT E; Clean-Data Control (TCGA-BRCA vs METABRIC)")
    print("=" * 70)

    df = load_breast()
    print(f"\nSamples: {len(df)}  |  {df['Source'].value_counts().to_dict()}")
    print(f"Events:   {df.groupby('Source')['Vital_Status'].mean().round(3).to_dict()}")
    print("\nMissingness (features used):")
    for c in ["Age", "Sex", "Stage"]:
        print(f"  {c:6s}: " + ", ".join(
            f"{s}={df[df['Source']==s][c].isna().mean()*100:.1f}%" for s in df['Source'].unique()))

    probe = probe_domain_sep(df)
    print(f"\nDomain-separability probe: {probe}")
    with open(RESULTS_DIR / "E_clean_probe.json", "w") as f:
        json.dump({"probe": probe, "samples": len(df), "features": CONT_FEATURES + CAT_FEATURES},
                  f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # E0: baseline data QC; survival curves / event balance
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for s, color in [("TCGA-BRCA", "#4c72b0"), ("METABRIC", "#55a868")]:
        sub = df[df["Source"] == s]
        ax.hist(sub["Survival_Months"], bins=40, alpha=0.55, label=s, color=color)
    ax.set_xlabel("Survival_Months")
    ax.set_ylabel("patients")
    ax.set_title("Experiment E; breast-cancer cohorts (both near-complete)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "E0_survival_dist.png", dpi=150)
    plt.close(fig)

    if args.skip_train:
        print("\n--skip_train: probe only. Done.")
        return

    t02 = _load_train_02()
    train_model = t02.train_model

    base_config = {
        "surv_type": "deephit",
        "d_model": 128, "n_layers": 4, "dropout": 0.15,
        "lr": 5e-4, "n_bins": 32,
        "epochs": args.epochs, "patience": 40,
        "per_cohort_loss": True, "domain_weight_max": args.domain_weight,
    }

    data = build_data(df, CONT_FEATURES, CAT_FEATURES, surv_type="deephit", n_bins=32)
    print(f"\nDomains: {data['cohort_map']}")

    results_all = {}
    for mode, dw in [("dann", args.domain_weight), ("baseline", 0.0)]:
        cfg = {**base_config, "domain_weight_max": dw}
        out_dir = EXP_DIR / mode
        print(f"\n{'─'*60}\nTRAINING: {mode.upper()}\n{'─'*60}")
        model, res, _ = train_model(cfg, data, mode=mode, output_dir=out_dir)
        results_all[mode] = res

    comp = {
        "experiment": "E_brca_metabric",
        "cohorts": list(data["cohort_map"].keys()),
        "cohort_map": data["cohort_map"],
        "features_used": CONT_FEATURES + CAT_FEATURES,
        "grade_excluded_reason": "Grade is 100% missing in TCGA-BRCA; including it "
                                 "would re-inject a missingness fingerprint",
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

    # E1: domain classifier trajectory (compare vs Exp A 97% baseline)
    hist_d = results_all["dann"]["history"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(hist_d["epoch"], hist_d["domain_acc"], "s-", ms=4, color="#4c72b0",
            label="Exp E (clean breast data)")
    try:
        hist_a = json.load(open(RESULTS_DIR / "lihc_experiments" / "A_tcga_vs_seer" / "dann" / "results.json"))["history"]
        ax.plot(hist_a["epoch"], hist_a["domain_acc"], "o-", ms=4, color="#c44e52",
                label="Exp A (LIHC, extreme missingness)")
    except FileNotFoundError:
        pass
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("domain classifier accuracy")
    ax.set_title("DANN domain classifier on CLEAN data vs missing data")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "E1_domain_acc_clean.png", dpi=150)
    plt.close(fig)

    # E2: per-cohort C-index DANN vs Baseline
    rev = {v: k for k, v in data["cohort_map"].items()}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    names = list(rev.values())
    dann = [comp["dann_per_cohort"].get(str(k), np.nan) for k in rev]
    base = [comp["baseline_per_cohort"].get(str(k), np.nan) for k in rev]
    x = np.arange(len(names))
    ax.bar(x - 0.18, dann, 0.36, label="DANN (GRL)", color="#c44e52")
    ax.bar(x + 0.18, base, 0.36, label="Baseline", color="#4c72b0")
    for xi, da, ba in zip(x, dann, base):
        if not np.isnan(da): ax.text(xi - 0.18, da + 0.01, f"{da:.3f}", ha="center", fontsize=8)
        if not np.isnan(ba): ax.text(xi + 0.18, ba + 0.01, f"{ba:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("C-index")
    ax.set_ylim(0.45, 0.85)
    ax.legend(fontsize=8)
    ax.set_title("Experiment E; per-cohort C-index (clean data)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "E2_per_cohort_cindex.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures: {FIG_DIR / 'E1_domain_acc_clean.png'}, {FIG_DIR / 'E2_per_cohort_cindex.png'}")
    print(f"\nComparison saved → {EXP_DIR / 'comparison.json'}")
    print(json.dumps({k: v for k, v in comp.items() if k != "probe"}, indent=2))
    print("\n✅ Experiment E complete!")


if __name__ == "__main__":
    main()
