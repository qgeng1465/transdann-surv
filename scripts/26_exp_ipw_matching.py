#!/usr/bin/env python3
"""
26_exp_ipw_matching.py; IPW × propensity-matching joint experiment
===================================================================

Background
----------
Point 3 of the honest conclusions of the IPW-ERM prototype (24_exp_ipw_erm.py,
table11) notes that IPW alone is not strong enough: "the full path needs IPW ×
standardisation/matching". This script completes that constructive path; on top of
IPW it adds **propensity-score matching (PSM)** to enforce common support, forming
an IPW × matching joint estimate. It answers reviewer P2-8 "whether the
constructive path actually works".

Design (changes stacked layer by layer, rigorous logic)
-------------------------------------------------------
1. Propensity model identical to the IPW prototype: multinomial logistic
   regression (encoded on the training set, no leakage).
   Each sample has π_own = P(own cohort|x), weight w(x) = clip(1/π_own, [1.05, 20]).
   Additionally take the target-cohort propensity π_TCGA = P(TCGA_LIHC|x),
   logit-transformed for matching distance.
2. Matching: for every TCGA training sample, match its k=3 nearest (with
   replacement) non-TCGA samples on the logit π_TCGA; training set = all TCGA +
   matched non-TCGA (multi-matched samples appear repeatedly). Matching forces
   "TCGA-like" source samples into training; equivalent to imposing common
   support before training.
3. Modes (A/B/C × 3 seeds × 5 modes):
     erm          : engine baseline + per_cohort_loss=True (paper ERM protocol)
     erm_pooled   : engine baseline + per_cohort_loss=False (unweighted pooled, direct IPW control)
     ipw_erm      : full training set + IPW-weighted pooled loss (= prototype table11, same recipe)
     matching     : matched subset + unweighted pooled loss
     ipw_matching : matched subset + IPW-weighted pooled loss (IPW × matching joint)
   All modes share the same loss function/optimizer/early stopping (best-val
   selection)/validation every 5 epochs; the matching modes use the same
   cohort-balanced WeightedRandomSampler on the matched subset (recipe identical
   to the engine).
4. Evaluation: overall best-val C-index + per-cohort C-index on the validation set
   (the target cohort TCGA is key for cross-population prediction).
5. Honest expectation (hard-coded as a diagnostic, no predetermined conclusion):
   structurally unobservable missingness (SEER Grade 100% missing, etc.) cannot be
   repaired by any weighting/matching; matching + weighting can at most improve the
   distribution alignment of the target cohort, but both the overall
   (SEER-dominated) performance and the real benefit in the target-cohort noise band
   must be reported honestly.

Outputs (results/experiments/exp_ipw_matching/):
  - runs.json / matching_summary.json / summary.txt
  - results/tables/table13_ipw_matching.csv
  - logs/26_exp_ipw_matching.log

Usage:
  python3 scripts/26_exp_ipw_matching.py                  # A/B/C × 3 seeds × 5 modes
  python3 scripts/26_exp_ipw_matching.py --experiments A  # run A only
  python3 scripts/26_exp_ipw_matching.py --seeds 42       # smoke
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_ipw_matching"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "26_exp_ipw_matching.log", mode="w"),
    ],
)
log = logging.getLogger("ipw_matching")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "train_engine", SCRIPTS_DIR / "02_train_dann_lihc.py"
)
train_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_engine)
EXPERIMENTS = train_engine.EXPERIMENTS
train_model = train_engine.train_model

spec24 = importlib.util.spec_from_file_location(
    "exp_ipw_24", SCRIPTS_DIR / "24_exp_ipw_erm.py"
)
ipw24 = importlib.util.module_from_spec(spec24)
spec24.loader.exec_module(ipw24)
WeightedSurvDataset = ipw24.WeightedSurvDataset
weighted_deep_hit_loss = ipw24.weighted_deep_hit_loss
eval_val_per_cohort = ipw24.eval_val_per_cohort
fit_propensity_and_weights = ipw24.fit_propensity_and_weights

from transdann_utils import (  # noqa: E402
    DEVICE, TransDANNSurvV3, deep_hit_risk,
)
from lihc_recon import build_data, CONT_FEATURES, CAT_FEATURES  # noqa: E402

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_EXPS = ["A", "B", "C"]
EXP_KEY = {"A": "A_tcga_vs_seer", "B": "B_tcga_vs_external", "C": "C_all_cohorts"}
TARGET_COHORT = "TCGA_LIHC"
K_NEIGHBORS = 3  # k:1 nearest-neighbour matching (with replacement)
PROPENSITY_MIN = 0.05
PROPENSITY_MAX = 0.95


# ===================================================================
# 1. Propensity model (same as IPW prototype) + π_TCGA for matching
# ===================================================================
def fit_propensity_with_target(data, seed=7):
    """Multinomial logistic propensity on train-only encoded features.

    Returns:
      w        : IPW weights aligned to train_df row order (1/π_own truncated)
      pi_tcga  : P(TCGA_LIHC|x) per train row
      logit_tcga : logit(pi_tcga)
      diag     : diagnostics
    """
    tr = data["train_df"].reset_index(drop=True)
    feats = pd.DataFrame(index=tr.index)
    feats["Age"] = tr["Age"].astype(float)
    for c in ["Sex", "Stage", "Grade"]:
        le = data["label_encoders"][c]
        mi = ipw24._missing_index(le)
        code = tr[c].astype(int)
        feats[f"{c}_code"] = code
        feats[f"{c}_known"] = (code != mi).astype(int)

    X = feats.values.astype(float)
    y = tr["Domain_Label"].values
    clf = LogisticRegression(max_iter=2000, C=0.5, multi_class="multinomial")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    own = np.array([proba[i, y[i]] for i in range(len(y))])

    own_clip = np.clip(own, PROPENSITY_MIN, PROPENSITY_MAX)
    w = 1.0 / own_clip
    bal = balanced_accuracy_score(y, clf.predict(X))

    target_id = int(data["cohort_map"][TARGET_COHORT])
    pi_tcga = proba[:, target_id]
    logit_tcga = np.log(pi_tcga + 1e-8) - np.log(1.0 - pi_tcga + 1e-8)

    cohorts = sorted(tr["Source"].unique())
    diag = {
        "propensity_balanced_acc": round(float(bal), 4),
        "n_train": int(len(tr)),
        "target_cohort": TARGET_COHORT,
        "pi_tcga": {
            "min": round(float(pi_tcga.min()), 4),
            "median": round(float(np.median(pi_tcga)), 4),
            "max": round(float(pi_tcga.max()), 4),
        },
        "per_cohort_mean_pi_tcga": {
            c: round(float(pi_tcga[tr["Source"].values == c].mean()), 4) for c in cohorts
        },
    }
    return w, pi_tcga, logit_tcga, diag


# ===================================================================
# 2. k:1 nearest-neighbour propensity matching (with replacement)
# ===================================================================
def build_matched_rows(train_df, logit_tcga, k=3, seed=7):
    """Return row indices (multi-set, with replacement) of the matched training
    subset: all target-cohort rows + the k nearest non-target rows on |logit|."""
    rng = np.random.RandomState(seed)
    src = train_df["Source"].values
    tcga_pos = np.where(src == TARGET_COHORT)[0]
    non_pos = np.where(src != TARGET_COHORT)[0]
    if len(non_pos) == 0:
        return tcga_pos.tolist(), {}
    d = np.abs(logit_tcga[non_pos, None] - logit_tcga[tcga_pos][None, :])  # (n_non, n_tcga)
    matched = list(tcga_pos)
    # k nearest non-target rows per target row (with replacement → multi-set)
    best_non = np.argsort(d, axis=0)[:k, :].ravel()  # (k·n_tcga,) row ids into non_pos
    matched.extend(non_pos[best_non].tolist())
    matched_arr = np.asarray(matched, dtype=np.int64)

    # common-support coverage: fraction of non-target within [min,max] of target logits
    lo, hi = float(logit_tcga[tcga_pos].min()), float(logit_tcga[tcga_pos].max())
    cov = float(np.mean((logit_tcga[non_pos] >= lo) & (logit_tcga[non_pos] <= hi)))
    # mean |Δlogit| of the actual matched pairs
    pair_dists = d[best_non, np.repeat(np.arange(len(tcga_pos)), k)].ravel()

    diag = {
        "k_neighbors": k,
        "n_target_train": int(len(tcga_pos)),
        "n_non_target_train": int(len(non_pos)),
        "n_matched_rows": int(len(matched_arr)),
        "n_unique_non_target_used": int(len(np.unique(best_non))),
        "common_support_coverage": round(cov, 4),
        "mean_abs_logit_dist_matched": round(float(np.mean(pair_dists)), 4),
        "max_abs_logit_dist_matched": round(float(np.max(pair_dists)), 4),
    }
    return matched_arr.tolist(), diag


# ===================================================================
# 3. Loader over a (multi-set) row-index subset with optional weights
# ===================================================================
def build_subset_loader(data, row_idx, weights=None):
    """Same cohort-balanced sampler + batch-size recipe as the engine, over the
    row subset.  weights aligned to row_idx (None → all-ones)."""
    train_df = data["train_df"].reset_index(drop=True)
    sub = train_df.iloc[row_idx].reset_index(drop=True)
    if weights is None:
        w = np.ones(len(sub), dtype=np.float64)
    else:
        w = np.asarray(weights)[row_idx].astype(np.float64)

    domain_counts = sub["Domain_Label"].value_counts().sort_index()
    sw = 1.0 / np.sqrt(domain_counts.values + 1)
    sampler_w = sw[sub["Domain_Label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sampler_w),
        num_samples=len(sampler_w), replacement=True,
    )
    ds = WeightedSurvDataset(sub, data["cont_features"], data["cat_features"],
                             "Domain_Label", w)
    bs = min(512, max(64, len(sub) // 4))
    bs = min(bs, len(sub))
    return DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True)


# ===================================================================
# 4. Weighted pooled training loop (generalises train_ipw_erm to a subset
#    loader; weights=None → unweighted pooled = plain DeepHit mean)
# ===================================================================
def train_weighted_pooled(config, data, train_loader, weights, output_dir, mode_label):
    EPOCHS = config.get("epochs", 200)
    PATIENCE = config.get("patience", 40)
    LR = config.get("lr", 5e-4)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = DEVICE
    os.makedirs(output_dir, exist_ok=True)
    val_loader = data["val_loader"]
    bin_edges = data["bin_edges"]
    bin_centers = data["bin_centers"]

    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=config.get("d_model", 128), n_heads=8,
        n_layers=config.get("n_layers", 4),
        dropout=config.get("dropout", 0.15),
        num_domains=data["num_domains"],
        surv_head_type="deephit", n_bins=config.get("n_bins", 32),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    best_val_c = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"epoch": [], "surv_loss": [], "val_cindex": [], "lr": []}

    def risk_fn(x, bc):
        return deep_hit_risk(x, bc)

    w_is_none = weights is None
    for epoch in range(EPOCHS):
        model.train()
        total_surv = 0.0
        n_batches = 0
        for batch in train_loader:
            if w_is_none:
                x_cont, x_cat, t, e, d, _ = batch
                w_batch = None
            else:
                x_cont, x_cat, t, e, d, w_batch = batch
                w_batch = w_batch.to(device)
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            t, e, d = t.to(device), e.to(device), d.to(device)

            optimizer.zero_grad()
            surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
            if w_batch is None:
                ones = torch.ones_like(t, dtype=torch.float32)
                loss_surv = weighted_deep_hit_loss(surv_out, t, e, bin_edges, ones)
            else:
                loss_surv = weighted_deep_hit_loss(surv_out, t, e, bin_edges, w_batch)
            loss_surv.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_surv += loss_surv.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_c = train_engine.evaluate(model, val_loader, device, risk_fn, bin_centers)
            history["epoch"].append(epoch + 1)
            history["surv_loss"].append(float(f"{total_surv / max(n_batches, 1):.4f}"))
            history["val_cindex"].append(float(f"{val_c:.4f}"))
            history["lr"].append(float(f"{scheduler.get_last_lr()[0]:.2e}"))
            log.info(f"  E{epoch + 1:3d} | {mode_label} SurvL={history['surv_loss'][-1]:.3f} "
                     f"| ValC={val_c:.4f} | LR={history['lr'][-1]:.2e}")

            if val_c > best_val_c:
                best_val_c = val_c
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                no_improve += 5
            if no_improve >= PATIENCE:
                log.info(f"  Early stop at E{epoch + 1}, best={best_epoch}, "
                         f"Val C={best_val_c:.4f}")
                break

    model.load_state_dict(torch.load(output_dir / "best_model.pth",
                                     map_location=device, weights_only=True))
    model.eval()
    val_per_cohort = eval_val_per_cohort(model, val_loader, device, risk_fn,
                                         bin_centers, data["cohort_map"])
    log.info(f"  Val per-cohort C-index: {val_per_cohort}")

    return {
        "mode": mode_label,
        "best_epoch": best_epoch,
        "best_val_cindex": round(float(best_val_c), 4),
        "val_per_cohort_cindex": val_per_cohort,
    }


# ===================================================================
# 5. Orchestrator
# ===================================================================
def run_split(exp_letter, df, seeds, epochs, patience, out_root, k=K_NEIGHBORS):
    exp_key = EXP_KEY[exp_letter]
    exp_cfg = EXPERIMENTS[exp_key]
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    log.info(f"\n{'#' * 72}")
    log.info(f"EXPERIMENT {exp_letter}: {exp_cfg['name']}  (cohorts={exp_cfg['cohorts']})")
    log.info(f"Samples: {len(exp_df)}")
    log.info(f"{'#' * 72}")

    data = build_data(exp_df, CONT_FEATURES, CAT_FEATURES,
                      surv_type="deephit", n_bins=32, subsample_seer=10000)
    log.info(f"Domains: {data['cohort_map']}  | Train {len(data['train_df'])} / "
             f"Val {len(data['val_df'])}")

    w, pi_tcga, logit_tcga, prop_diag = fit_propensity_with_target(data)
    matched_rows, match_diag = build_matched_rows(
        data["train_df"].reset_index(drop=True), logit_tcga, k=k)
    log.info(f"Propensity bal-acc={prop_diag['propensity_balanced_acc']} | "
             f"π_TCGA per-cohort mean: {prop_diag['per_cohort_mean_pi_tcga']}")
    log.info(f"Matching: k={k}, rows={match_diag['n_matched_rows']} "
             f"(TCGA {match_diag['n_target_train']}), unique non-target used "
             f"{match_diag['n_unique_non_target_used']}, common-support cov "
             f"{match_diag['common_support_coverage']}, mean|Δlogit|="
             f"{match_diag['mean_abs_logit_dist_matched']}")

    base_cfg = {
        "surv_type": "deephit", "d_model": 128, "n_layers": 4,
        "dropout": 0.15, "lr": 5e-4, "n_bins": 32,
        "epochs": epochs, "patience": patience,
        "per_cohort_loss": True, "domain_weight_max": 0.3,
    }

    exp_out = out_root / exp_key
    records = []
    for seed in seeds:
        # engine ERM (per-cohort) & pooled unweighted ERM; paper/control protocols
        for label, per_coh in [("erm", True), ("erm_pooled", False)]:
            out_dir = exp_out / label / f"seed_{seed}"
            cfg = {**base_cfg, "per_cohort_loss": per_coh, "seed": seed}
            log.info(f"\n>>> {label.upper()} seed={seed}")
            _, res, _ = train_model(cfg, data, mode="baseline", output_dir=out_dir)
            vpc = eval_val_per_cohort_for_model(out_dir, data)
            records.append({
                "experiment": exp_key, "mode": label, "seed": seed,
                "best_val_cindex": res.get("best_val_cindex"),
                "val_per_cohort_cindex": vpc,
            })

        # IPW-ERM on the full training set (weighted pooled)
        out_dir = exp_out / "ipw_erm" / f"seed_{seed}"
        cfg = {**base_cfg, "per_cohort_loss": False, "seed": seed}
        log.info(f"\n>>> IPW-ERM seed={seed}")
        res = train_weighted_pooled(cfg, data, build_subset_loader(
            data, np.arange(len(data["train_df"])), w), w, out_dir, "IPW")
        records.append({
            "experiment": exp_key, "mode": "ipw_erm", "seed": seed,
            "best_val_cindex": res.get("best_val_cindex"),
            "val_per_cohort_cindex": res.get("val_per_cohort_cindex", {}),
        })

        # matching (matched subset, unweighted pooled)
        out_dir = exp_out / "matching" / f"seed_{seed}"
        cfg = {**base_cfg, "per_cohort_loss": False, "seed": seed}
        log.info(f"\n>>> MATCHING seed={seed}")
        res = train_weighted_pooled(cfg, data, build_subset_loader(
            data, matched_rows, None), None, out_dir, "MATCH")
        records.append({
            "experiment": exp_key, "mode": "matching", "seed": seed,
            "best_val_cindex": res.get("best_val_cindex"),
            "val_per_cohort_cindex": res.get("val_per_cohort_cindex", {}),
        })

        # IPW × matching (matched subset, IPW-weighted pooled)
        out_dir = exp_out / "ipw_matching" / f"seed_{seed}"
        cfg = {**base_cfg, "per_cohort_loss": False, "seed": seed}
        log.info(f"\n>>> IPW-MATCHING seed={seed}")
        res = train_weighted_pooled(cfg, data, build_subset_loader(
            data, matched_rows, w), w, out_dir, "IPWxMATCH")
        records.append({
            "experiment": exp_key, "mode": "ipw_matching", "seed": seed,
            "best_val_cindex": res.get("best_val_cindex"),
            "val_per_cohort_cindex": res.get("val_per_cohort_cindex", {}),
        })

    with open(exp_out / "runs.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    return records, prop_diag, match_diag


def eval_val_per_cohort_for_model(out_dir, data):
    """Load the engine-saved best model and compute val-only per-cohort C-index."""
    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=128, n_heads=8, n_layers=4, dropout=0.15,
        num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32,
    ).to(DEVICE)
    model.load_state_dict(torch.load(out_dir / "best_model.pth",
                                     map_location=DEVICE, weights_only=True))
    return eval_val_per_cohort(model, data["val_loader"], DEVICE,
                               lambda x, bc: deep_hit_risk(x, bc),
                               data["bin_centers"], data["cohort_map"])


def aggregate(records):
    rows = {}
    for r in records:
        rows.setdefault((r["experiment"], r["mode"]), []).append(r)
    out = []
    for (exp, mode), rs in sorted(rows.items()):
        cids = [x["best_val_cindex"] for x in rs if x["best_val_cindex"] is not None]
        per_cohort = {}
        for cname in sorted({k for r in rs for k in r["val_per_cohort_cindex"]}):
            vals = [r["val_per_cohort_cindex"].get(cname)
                    for r in rs if r["val_per_cohort_cindex"].get(cname) is not None]
            per_cohort[cname] = round(float(np.mean(vals)), 4) if vals else None
        out.append({
            "experiment": exp, "mode": mode,
            "cindex_mean": round(float(np.mean(cids)), 4) if cids else None,
            "cindex_std": round(float(np.std(cids, ddof=1)), 4) if len(cids) > 1 else 0.0,
            "cindex_each": [round(x, 4) for x in cids],
            "val_per_cohort_mean": per_cohort,
        })
    return out


def write_table_csv(summary, all_diag):
    """Write results/tables/table13_ipw_matching.csv."""
    rows = []
    for s in summary:
        row = {
            "experiment": s["experiment"], "mode": s["mode"],
            "cindex_mean": s["cindex_mean"], "cindex_std": s["cindex_std"],
            "cindex_each": "[" + ", ".join(f"{x:.4f}" for x in s["cindex_each"]) + "]",
        }
        for cname, v in s["val_per_cohort_mean"].items():
            row[f"val_{cname}"] = v
        rows.append(row)
    for ek, (pd_, md) in all_diag.items():
        rows.append({"experiment": ek, "mode": "matching_diag",
                     "cindex_mean": md["n_matched_rows"], "cindex_std": None,
                     "cindex_each": str({"k": md["k_neighbors"],
                                         "cov": md["common_support_coverage"],
                                         "mean_dlogit": md["mean_abs_logit_dist_matched"],
                                         "prop_bal_acc": pd_["propensity_balanced_acc"]})})
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "tables" / "table13_ipw_matching.csv", index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table13_ipw_matching.csv'} "
             f"({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="IPW × propensity-matching joint")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPS,
                        choices=list(EXP_KEY))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        log.error(f"Data not found: {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    log.info(f"Loaded {len(df)} rows, {df['Source'].nunique()} cohorts")

    all_records, all_diag = [], {}
    for letter in args.experiments:
        recs, prop_d, match_d = run_split(letter, df, args.seeds, args.epochs,
                                          args.patience, EXP_OUT, k=args.k)
        all_records.extend(recs)
        all_diag[EXP_KEY[letter]] = (prop_d, match_d)

    summary = aggregate(all_records)
    out_json = {
        "experiments": args.experiments,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "k_neighbors": args.k,
        "diagnostics": {k: {"propensity": v[0], "matching": v[1]}
                        for k, v in all_diag.items()},
        "summary": summary,
        "all_runs": all_records,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXP_OUT / "matching_summary.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    with open(EXP_OUT / "runs.json", "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    write_table_csv(summary, all_diag)

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("IPW × MATCHING JOINT SUMMARY  (seeds %s, k=%d)"
                 % (args.seeds, args.k))
    lines.append("=" * 100)
    for s in summary:
        rows = "[" + ", ".join(f"{x:.4f}" for x in s["cindex_each"]) + "]"
        lines.append(f"  {s['experiment']:<18}{s['mode']:<13}"
                     f"best-val C={s['cindex_mean']:.4f}±{s['cindex_std']:.4f}  "
                     f"{rows}")
        for cname, v in s["val_per_cohort_mean"].items():
            if v is not None:
                lines.append(f"      val-{cname:<12}C={v:.4f}")
    lines.append("=" * 100)
    for ek, (pd_, md) in all_diag.items():
        lines.append(f"  {ek}: prop bal-acc={pd_['propensity_balanced_acc']} | "
                     f"matched rows={md['n_matched_rows']} (TCGA {md['n_target_train']}), "
                     f"unique non-target used={md['n_unique_non_target_used']}, "
                     f"common-support cov={md['common_support_coverage']}, "
                     f"mean|Δlogit|={md['mean_abs_logit_dist_matched']}")
    lines.append("=" * 100)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {EXP_OUT}/matching_summary.json, runs.json, summary.txt")
    log.info("\n✅ IPW × matching joint complete!")


if __name__ == "__main__":
    main()
