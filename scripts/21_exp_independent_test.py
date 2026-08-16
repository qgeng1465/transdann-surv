#!/usr/bin/env python3
"""
21_exp_independent_test.py; True independent-test-set 3-fold retrain (DANN vs Baseline)
========================================================================================

Reviewer risk: the existing A/B/C C-index is a "best-validation-set" metric; both
early stopping and model selection are done on the 15% validation set, so strictly
speaking the validation set participates in model selection and is not a truly
independent test set.

This script performs a **3-fold split stratified by (domain, event)** for each of
A/B/C:

  - each fold: 2/3 forms the development set (with a further per-domain 15%
    early-stopping validation split inside it), and 1/3 is a **test set that never
    participates in training**;
  - preprocessing (median imputation / LabelEncoder / StandardScaler / DeepHit
    time windows) is **fitted only on the development set**, then applied to the
    test set; eliminating any test-set information leakage;
  - each fold × 3 training seeds {42,43,44} × {DANN, Baseline} is trained with
    early stopping on the validation set, and the final best-val model is evaluated
    on the **true test set** (DANN also records test-set domain-classification accuracy).

  DANN uses the default α 0→1 schedule (the fixed-α variant is covered in script 19).

Outputs (results/experiments/exp_independent_test/):
  - runs/…/{mode}/results.json     standard train_model output (incl. best_val_cindex)
  - runs/…/{mode}/test_results.json  true test-set evaluation results
  - runs.json                       per (exp, fold, seed, mode) summary
  - summary.txt                     human-readable summary
  - logs/21_exp_independent_test.log

Usage:
  python3 scripts/21_exp_independent_test.py                          # A/B/C × 3 folds × 3 seeds × 2 modes
  python3 scripts/21_exp_independent_test.py --experiments A          # run A only
  python3 scripts/21_exp_independent_test.py --folds 0 --seeds 42 --epochs 20 --patience 5   # smoke test
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
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"

OUT_ROOT = RESULTS_DIR / "experiments" / "exp_independent_test"
OUT_LOG_DIR = OUT_ROOT / "logs"
OUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_LOG_DIR / "21_exp_independent_test.log", mode="w"),
    ],
)
log = logging.getLogger("indep_test")

# ---------------------------------------------------------------------------
# Import the training engine (02); filename starts with a digit, use importlib
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "train_engine", SCRIPTS_DIR / "02_train_dann_lihc.py"
)
train_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_engine)
EXPERIMENTS = train_engine.EXPERIMENTS
CONT_FEATURES = train_engine.CONT_FEATURES
CAT_FEATURES = train_engine.CAT_FEATURES
train_model = train_engine.train_model
evaluate = train_engine.evaluate
full_evaluation = train_engine.full_evaluation
make_loss_fn = train_engine.make_loss_fn
LiverCancerDataset = train_engine.LiverCancerDataset
compute_time_bins = train_engine.compute_time_bins
DEVICE = train_engine.DEVICE

DEFAULT_SEEDS = [42, 43, 44]
SUBSAMPLE_SEER = 10000  # mirrors 02's load_and_preprocess default


# ===================================================================
# 1. Stratified 3-fold split (per (Source, Vital_Status) round-robin)
# ===================================================================
def make_folds(df, n_splits=3, seed=42):
    """Stratified 3-fold split by (domain, event).

    Rows in each (Source, Vital_Status) group are shuffled first, then assigned
    round-robin across folds, so every fold receives roughly 1/3 of each domain
    (and its event mix). Never fails because of small strata.
    Returns folds: list[np.ndarray], each fold is a set of global df indices.
    """
    folds = [[] for _ in range(n_splits)]
    rng = np.random.RandomState(seed)
    for cohort in sorted(df["Source"].unique()):
        cdf = df[df["Source"] == cohort]
        for ev in sorted(df["Vital_Status"].unique()):
            gi = cdf[cdf["Vital_Status"] == ev].index.tolist()
            rng.shuffle(gi)
            for j, i in enumerate(gi):
                folds[j % n_splits].append(i)
    return [np.asarray(f, dtype=np.int64) for f in folds]


# ===================================================================
# 2. Fold-level data prep; dev/train/val/test, leakage-free
# ===================================================================
def prepare_fold(exp_df, dev_idx, test_idx, fold, n_bins=32):
    """Build one fold's data on (dev, test).

    dev  = development set (train + early-stopping val); test = true independent test set.
    All fitted quantities (imputation medians / LabelEncoder / StandardScaler /
    time windows) are taken from dev only; the test set is only transformed and
    never enters any fit.
    """
    dev_df = exp_df.loc[dev_idx].copy()
    test_df = exp_df.loc[test_idx].copy()

    # ---- Continuous features: median imputation (medians only from dev) ----
    for col in CONT_FEATURES:
        med = dev_df[col].median()
        if pd.isna(med):
            med = dev_df[col].dropna().median() if dev_df[col].notna().any() else 50.0
        dev_df[col] = dev_df[col].fillna(med)
        test_df[col] = test_df[col].fillna(med)

    # ---- Categorical features: LabelEncoder fit on dev only; unknown categories fall back to -1 ----
    # Explicitly include "-1" in classes to avoid an out-of-bounds when dev has no
    # missing values but test does.
    label_encoders, cat_cardinalities = {}, []
    for col in CAT_FEATURES:
        dev_df[col] = dev_df[col].fillna(-1).astype(str)
        test_df[col] = test_df[col].fillna(-1).astype(str)
        le = LabelEncoder()
        le.fit(np.concatenate([dev_df[col].values, ["-1"]]))
        dev_df[col] = le.transform(dev_df[col].values).astype(int)
        placeholder_idx = int(np.where(le.classes_ == "-1")[0][0])
        test_codes = [
            int(le.transform([v])[0]) if v in le.classes_ else placeholder_idx
            for v in test_df[col].values
        ]
        test_df[col] = test_codes
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))

    # ---- Continuous feature standardisation (scaler fit on dev only) ----
    scaler = StandardScaler()
    dev_df[CONT_FEATURES] = scaler.fit_transform(dev_df[CONT_FEATURES])
    test_df[CONT_FEATURES] = scaler.transform(test_df[CONT_FEATURES])

    # ---- Domain labels: the experiment's domain set is fixed (fold-independent) ----
    cohorts = sorted(exp_df["Source"].unique())
    cohort_map = {c: i for i, c in enumerate(cohorts)}
    dev_df["Domain_Label"] = dev_df["Source"].map(cohort_map).astype(int)
    test_df["Domain_Label"] = test_df["Source"].map(cohort_map).astype(int)
    num_domains = len(cohorts)

    # ---- 15% early-stopping validation split inside dev (stratified by domain; seed varies per fold, independent of training seeds) ----
    rng = np.random.RandomState(1000 + fold)
    val_idx, train_idx = [], []
    for cohort in sorted(dev_df["Source"].unique()):
        cidx = dev_df[dev_df["Source"] == cohort].index.tolist()
        rng.shuffle(cidx)
        n_val = max(1, int(len(cidx) * 0.15))
        val_idx.extend(cidx[:n_val])
        train_idx.extend(cidx[n_val:])
    train_df = dev_df.loc[train_idx].reset_index(drop=True)
    val_df = dev_df.loc[val_idx].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # ---- DeepHit time windows (event times from dev-train only) ----
    bin_edges, bin_centers = compute_time_bins(
        torch.tensor(train_df["Survival_Months"].values),
        torch.tensor(train_df["Vital_Status"].values),
        n_bins=n_bins,
    )

    train_ds = LiverCancerDataset(train_df, CONT_FEATURES, CAT_FEATURES, "Domain_Label")
    val_ds = LiverCancerDataset(val_df, CONT_FEATURES, CAT_FEATURES, "Domain_Label")
    test_ds = LiverCancerDataset(test_df, CONT_FEATURES, CAT_FEATURES, "Domain_Label")

    domain_counts = train_df["Domain_Label"].value_counts().sort_index()
    w = 1.0 / np.sqrt(domain_counts.values + 1)
    sw = w[train_df["Domain_Label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sw), num_samples=len(sw), replacement=True
    )
    batch_size = min(512, max(64, len(train_df) // 4))
    batch_size = min(batch_size, len(train_df))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    data = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_df": train_df,
        "val_df": val_df,
        "cont_features": CONT_FEATURES,
        "cat_features": CAT_FEATURES,
        "cat_cardinalities": cat_cardinalities,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "num_domains": num_domains,
        "cohort_map": cohort_map,
    }
    sizes = {"train_n": len(train_df), "val_n": len(val_df), "test_n": len(test_df)}
    return data, test_loader, sizes


# ===================================================================
# 3. Test-set domain accuracy (DANN mechanism side metric)
# ===================================================================
def domain_acc(model, loader, device):
    """Test-set domain-classification accuracy (evaluated at α=0: domain separability from features alone)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in loader:
            _, dom_pred, _ = model(x_cont.to(device), x_cat.to(device), alpha=0.0)
            pred = dom_pred.argmax(dim=1).cpu()
            correct += (pred == d).sum().item()
            total += len(d)
    return correct / total if total else None


# ===================================================================
# 4. Per-experiment runner
# ===================================================================
def run_experiment_indep(exp_key, df, fold_ids, seeds, epochs, patience, out_root):
    """Run 3-fold true-independent-test-set retraining for one experiment; return per-run records."""
    exp_cfg = EXPERIMENTS[exp_key]
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    log.info(f"\n{'#'*72}")
    log.info(f"EXPERIMENT: {exp_cfg['name']}  (cohorts={exp_cfg['cohorts']})")
    log.info(f"Samples (pre-subsample): {len(exp_df)}")
    log.info(f"{'#'*72}")

    # SEER subsampling: global, deterministic (RandomState(42)), fold-independent; consistent with 02
    if "US_SEER" in exp_df["Source"].values:
        seer_idx = exp_df[exp_df["Source"] == "US_SEER"].index
        if len(seer_idx) > SUBSAMPLE_SEER:
            keep = np.random.RandomState(42).choice(seer_idx, SUBSAMPLE_SEER, replace=False)
            drop = seer_idx.difference(pd.Index(keep))
            exp_df = exp_df.drop(drop).copy()
            log.info(f"Subsampled SEER: {len(seer_idx)} → {SUBSAMPLE_SEER}")

    all_idx = exp_df.index
    folds = make_folds(exp_df, n_splits=3, seed=42)
    log.info(f"Fold sizes (test fold): "
             + ", ".join(f"fold{k}={len(folds[k])}" for k in fold_ids))

    records = []
    for fold in fold_ids:
        test_idx = folds[fold]
        dev_idx = all_idx.difference(pd.Index(test_idx)).values
        data, test_loader, sizes = prepare_fold(exp_df, dev_idx, test_idx, fold,
                                                n_bins=32)
        loss_fn, risk_fn = make_loss_fn("deephit", data["bin_edges"])
        log.info(f"\n  fold {fold}: train={sizes['train_n']}, "
                 f"val={sizes['val_n']}, test={sizes['test_n']} "
                 f"(cohort_map={data['cohort_map']})")

        for seed in seeds:
            for mode in ["baseline", "dann"]:
                cfg = {
                    "surv_type": "deephit", "d_model": 128, "n_layers": 4,
                    "dropout": 0.15, "lr": 5e-4, "n_bins": 32,
                    "epochs": epochs, "patience": patience,
                    "per_cohort_loss": True, "domain_weight_max": 0.3,
                    "seed": seed, "experiment": exp_key, "fold": fold,
                }
                if mode == "baseline":
                    cfg["domain_weight_max"] = 0.0
                out_dir = out_root / "runs" / exp_key / f"fold_{fold}" \
                          / f"seed_{seed}" / mode
                os.makedirs(out_dir, exist_ok=True)

                log.info(f"\n>>> {exp_key} fold={fold} seed={seed} mode={mode}")
                try:
                    model, res, _ = train_model(cfg, data, mode=mode, output_dir=out_dir)
                except Exception as e:
                    log.error(f"Training failed: {e}")
                    import traceback; traceback.print_exc()
                    records.append({
                        "experiment": exp_key, "fold": fold, "seed": seed,
                        "mode": mode, "test_cindex": None, "error": str(e),
                    })
                    continue

                # True test-set evaluation (best-val model; test never took part in training/early-stopping/selection)
                test_c = evaluate(model, test_loader, DEVICE, risk_fn,
                                  data["bin_centers"])
                per_cohort_test, td_auc_test, _, _ = full_evaluation(
                    model, test_loader, DEVICE, risk_fn, data["bin_centers"]
                )
                test_dom_acc = domain_acc(model, test_loader, DEVICE) \
                    if mode == "dann" else None

                record = {
                    "experiment": exp_key, "fold": fold, "seed": seed, "mode": mode,
                    "test_cindex": round(float(test_c), 4),
                    "test_per_cohort_cindex": per_cohort_test,
                    "test_domain_acc": (round(float(test_dom_acc), 4)
                                        if test_dom_acc is not None else None),
                    "best_val_cindex": res.get("best_val_cindex"),
                    "best_epoch": res.get("best_epoch"),
                    "sizes": sizes,
                }
                records.append(record)
                with open(out_dir / "test_results.json", "w") as f:
                    json.dump(record, f, indent=2, default=str)
                log.info(
                    f"  TEST C-index={record['test_cindex']:.4f} "
                    f"(best-val={record['best_val_cindex']}, ep={record['best_epoch']})"
                    + (f"  test-domacc={record['test_domain_acc']}"
                       if record["test_domain_acc"] is not None else "")
                )

    os.makedirs(out_root / exp_key, exist_ok=True)
    with open(out_root / exp_key / "runs.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    return records


def main():
    parser = argparse.ArgumentParser(description="True independent-test-set 3-fold retrain")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C"],
                        help="experiments to run (A/B/C)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2],
                        help="fold ids (default: 0 1 2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="training seeds (default: 42 43 44)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        log.error(f"Data not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    log.info(f"Loaded {len(df)} rows, {df['Source'].nunique()} cohorts")
    log.info(f"Device: {DEVICE}")

    exp_keys = [
        f"{e}_" + {"A": "tcga_vs_seer", "B": "tcga_vs_external", "C": "all_cohorts"}[e]
        for e in args.experiments
    ]

    all_records = []
    for ek in exp_keys:
        all_records.extend(run_experiment_indep(
            ek, df, args.folds, args.seeds, args.epochs, args.patience, OUT_ROOT
        ))

    out_json = {
        "folds": args.folds, "seeds": args.seeds,
        "epochs": args.epochs, "patience": args.patience,
        "n_splits": 3, "subsample_seer": SUBSAMPLE_SEER,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs": len(all_records),
        "all_runs": all_records,
    }
    with open(OUT_ROOT / "runs.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)

    # ---- Human-readable summary ----
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("TRUE INDEPENDENT TEST-SET 3-FOLD RETAIN (DANN vs Baseline)")
    lines.append(f"  folds={args.folds}, seeds={args.seeds}, "
                 f"epochs={args.epochs}, patience={args.patience}, "
                 f"folds_seed=42, subsample_seer={SUBSAMPLE_SEER}")
    lines.append("=" * 100)

    by_exp = {}
    for r in all_records:
        by_exp.setdefault(r["experiment"], []).append(r)
    for ek in exp_keys:
        rs = by_exp.get(ek, [])
        lines.append(f"\n--- {ek} ---")
        for fold in sorted(set(r["fold"] for r in rs)):
            for mode in ["baseline", "dann"]:
                cids = [r["test_cindex"] for r in rs
                        if r["fold"] == fold and r["mode"] == mode
                        and r["test_cindex"] is not None]
                if cids:
                    lines.append(
                        f"  fold {fold} {mode:<8}: test C-index "
                        f"mean={np.mean(cids):.4f} ± {np.std(cids, ddof=1) if len(cids) > 1 else 0.0:.4f} "
                        f"[{', '.join(f'{c:.4f}' for c in cids)}]"
                    )
            d = [r["test_cindex"] for r in rs
                 if r["fold"] == fold and r["mode"] == "dann"
                 and r["test_cindex"] is not None]
            b = [r["test_cindex"] for r in rs
                 if r["fold"] == fold and r["mode"] == "baseline"
                 and r["test_cindex"] is not None]
            if d and b and len(d) == len(b):
                dd = np.mean(d) - np.mean(b)
                lines.append(f"         Δ(DANN−Base) per-seed deltas: "
                             f"[{', '.join(f'{x - y:+.4f}' for x, y in zip(d, b))}] "
                             f"→ fold mean Δ={dd:+.4f}")
    lines.append("\n" + "=" * 100)
    txt = "\n".join(lines)
    log.info(txt)
    with open(OUT_ROOT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {OUT_ROOT}/runs.json, summary.txt")
    log.info("\n✅ Independent-test retrain complete!")


if __name__ == "__main__":
    main()
