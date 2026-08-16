#!/usr/bin/env python3
"""
24_exp_ipw_erm.py; IPW-weighted ERM prototype (inverse-propensity reweighting)
===============================================================================

Responds to reviewer P2-8 ("constructive path for the negative result"): the first
step of the operative proposal in Discussion §10.4; "match the population
distribution first, then talk about domain adaptation"; reweight an ERM baseline
with propensity scores, testing whether **reweighting alone** improves
cross-population survival prediction.

Design (rigorous logic, single change)
--------------------------------------
1. Data identical to ERM: build_data (same as Exp A/B/C: SEER subsample 10000,
   15% stratified validation set, DeepHit 32 bins); the three modes share the same
   train_loader/val_loader (including the same **cohort-balanced
   WeightedRandomSampler**); the only difference is the **per-sample weight at
   the loss level**.
2. Propensity model: multinomial logistic regression (same input encoding as Exp I:
   Age + factorized category codes + observed indicators), **fitted on the training
   set only** (no leakage).
   π_own(x) = P(sample's own cohort | x). Weight w(x) = 1/π_own(x),
   π_own truncated to [0.05, 0.95] (truncated-IPW, w ∈ [1.05, 20], guarding against
   near-zero propensity instability).
   In the binary A split (SEER ≈ 96%), this is equivalent to heavily upweighting
   samples that "look like TCGA" (≈10–20×), the literal implementation of
   REVISION_TEXT F.3's w = 1/π(x), used to remove the loss imbalance caused by
   SEER dominance.
3. Modes (per split × per seed):
     erm        : engine baseline mode, per_cohort_loss=True (paper ERM protocol)
     dann       : engine dann mode, per_cohort_loss=True (paper DANN protocol)
     erm_pooled : engine baseline mode, per_cohort_loss=False (pooled loss, no weights), the direct IPW control (differs from ipw_erm only by weights)
     ipw_erm    : this script's weighted training loop, per_cohort_loss=False,
                   per-sample weighted pooled loss
                   L = Σ w(xₙ)·ℓ(yₙ, f(xₙ)) / Σ w(xₙ)
   Training dynamics identical to the engine (seed, AdamW 5e-4, CosineAnnealing,
   early-stop on best-val, validate every 5 epochs).
4. Evaluation: overall best-val C-index + **per-cohort C-index on the validation
   set** (the key cross-population metric, especially the target cohort TCGA).

Outputs (results/experiments/exp_ipw_erm/):
  - runs.json          per (exp, mode, seed) records (best_val_cindex, val per-cohort)
  - ipw_summary.json   aggregate (mean±std over seeds)
  - summary.txt        human-readable summary
  - logs/24_exp_ipw_erm.log

Usage:
  python3 scripts/24_exp_ipw_erm.py                  # A/B/C × 3 seeds × 4 modes
  python3 scripts/24_exp_ipw_erm.py --experiments A  # run A only
  python3 scripts/24_exp_ipw_erm.py --seeds 42       # smoke
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_ipw_erm"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "24_exp_ipw_erm.log", mode="w"),
    ],
)
log = logging.getLogger("ipw_erm")

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
evaluate = train_engine.evaluate

from transdann_utils import (  # noqa: E402
    DEVICE, TransDANNSurvV3, deep_hit_risk, compute_time_bins,
)
from lihc_recon import build_data, CONT_FEATURES, CAT_FEATURES  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_EXPS = ["A", "B", "C"]
EXP_KEY = {"A": "A_tcga_vs_seer", "B": "B_tcga_vs_external", "C": "C_all_cohorts"}
PROPENSITY_MIN = 0.05
PROPENSITY_MAX = 0.95


# ===================================================================
# 1. Propensity scores + importance weights (train-only, leakage-free)
# ===================================================================
def _missing_index(le):
    """Index of the '-' missing placeholder inside a LabelEncoder."""
    classes = list(le.classes_)
    return classes.index("-1") if "-1" in classes else -1


def fit_propensity_and_weights(data, seed=7):
    """Fit multinomial logistic propensity on train-only encoded features.

    Returns (weights aligned to train_df row order, diagnostics).
    """
    tr = data["train_df"].reset_index(drop=True)
    feats = pd.DataFrame(index=tr.index)
    feats["Age"] = tr["Age"].astype(float)  # already standardised
    for c in ["Sex", "Stage", "Grade"]:
        le = data["label_encoders"][c]
        mi = _missing_index(le)
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

    cohorts = sorted(tr["Source"].unique())
    diag = {
        "propensity_balanced_acc": round(float(bal), 4),
        "n_train": int(len(tr)),
        "weight": {
            "min": round(float(w.min()), 3),
            "median": round(float(np.median(w)), 3),
            "max": round(float(w.max()), 3),
            "p99": round(float(np.quantile(w, 0.99)), 3),
        },
        "per_cohort_mean_weight": {
            c: round(float(w[tr["Source"].values == c].mean()), 2) for c in cohorts
        },
        "per_cohort_weight_ratio": {
            c: round(float(w[tr["Source"].values == c].mean() /
                           w.mean()), 2) for c in cohorts
        },
    }
    return w, diag


# ===================================================================
# 2. Weighted DeepHit pooled loss
# ===================================================================
def weighted_deep_hit_loss(hazard_logits, times, events, bin_edges, weights):
    """Per-sample DeepHit NLL, averaged with importance weights:
    L = Σ wₙ·ℓₙ / Σ wₙ.  (Same per-sample NLL as transdann_utils.deep_hit_loss.)"""
    hazards = F.softmax(hazard_logits, dim=1)
    n_bins = hazards.size(1)
    bin_edges = bin_edges.to(times.device)
    time_bins = torch.bucketize(times, bin_edges) - 1
    time_bins = time_bins.clamp(0, n_bins - 1)
    batch = len(times)
    idx = torch.arange(batch, device=times.device)
    h_t = hazards[idx, time_bins]
    cum_haz = torch.cumsum(hazards, dim=1)
    F_t = cum_haz[idx, time_bins]
    S_t = (1.0 - F_t).clamp(min=1e-8)
    nll = torch.where(events == 1, -torch.log(h_t + 1e-8), -torch.log(S_t))
    wsum = weights.sum().clamp(min=1e-8)
    return (weights * nll).sum() / wsum


# ===================================================================
# 3. Weighted dataset + loader (same sampler / batch size as the engine)
# ===================================================================
class WeightedSurvDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, cont_features, cat_features, domain_label_col,
                 weights):
        self.x_cont = torch.tensor(dataframe[cont_features].values, dtype=torch.float32)
        self.x_cat = torch.tensor(dataframe[cat_features].values, dtype=torch.long)
        self.t = torch.tensor(dataframe["Survival_Months"].values, dtype=torch.float32)
        self.e = torch.tensor(dataframe["Vital_Status"].values, dtype=torch.float32)
        self.d = torch.tensor(dataframe[domain_label_col].values, dtype=torch.long)
        self.w = torch.tensor(np.asarray(weights), dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return (self.x_cont[idx], self.x_cat[idx], self.t[idx],
                self.e[idx], self.d[idx], self.w[idx])


def build_weighted_loader(data, weights):
    """Loader over the same train_df with the SAME cohort-balanced sampler and
    batch size as data['train_loader']; each batch additionally yields w."""
    train_df = data["train_df"]
    domain_counts = train_df["Domain_Label"].value_counts().sort_index()
    sw = 1.0 / np.sqrt(domain_counts.values + 1)
    sampler_w = sw[train_df["Domain_Label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sampler_w),
        num_samples=len(sampler_w), replacement=True,
    )
    ds = WeightedSurvDataset(train_df, data["cont_features"], data["cat_features"],
                             "Domain_Label", weights)
    return DataLoader(
        ds, batch_size=data["train_loader"].batch_size, sampler=sampler,
        drop_last=data["train_loader"].drop_last,
    )


# ===================================================================
# 4. IPW-ERM training loop (mirrors engine's baseline branch verbatim,
#    only the survival loss is importance-weighted)
# ===================================================================
def train_ipw_erm(config, data, weights, output_dir):
    EPOCHS = config.get("epochs", 200)
    PATIENCE = config.get("patience", 40)
    LR = config.get("lr", 5e-4)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = DEVICE
    os.makedirs(output_dir, exist_ok=True)
    train_loader = build_weighted_loader(data, weights)
    val_loader = data["val_loader"]
    num_domains = data["num_domains"]
    bin_edges = data["bin_edges"]
    bin_centers = data["bin_centers"]

    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=config.get("d_model", 128), n_heads=8,
        n_layers=config.get("n_layers", 4),
        dropout=config.get("dropout", 0.15),
        num_domains=num_domains,
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

    for epoch in range(EPOCHS):
        model.train()
        total_surv = 0.0
        n_batches = 0
        for batch in train_loader:
            x_cont, x_cat, t, e, d, w = batch
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            t, e, d = t.to(device), e.to(device), d.to(device)
            w = w.to(device)

            optimizer.zero_grad()
            surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
            loss_surv = weighted_deep_hit_loss(surv_out, t, e, bin_edges, w)
            loss_surv.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_surv += loss_surv.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_c = evaluate(model, val_loader, device, risk_fn, bin_centers)
            history["epoch"].append(epoch + 1)
            history["surv_loss"].append(float(f"{total_surv / max(n_batches, 1):.4f}"))
            history["val_cindex"].append(float(f"{val_c:.4f}"))
            history["lr"].append(float(f"{scheduler.get_last_lr()[0]:.2e}"))
            log.info(f"  E{epoch + 1:3d} | IPW SurvL={history['surv_loss'][-1]:.3f} "
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

    # ---- val per-cohort C-index (held-out metric for cross-population story) ----
    val_per_cohort = eval_val_per_cohort(model, val_loader, device, risk_fn,
                                         bin_centers, data["cohort_map"])
    log.info(f"  Val per-cohort C-index: {val_per_cohort}")

    results = {
        "mode": "ipw_erm",
        "config": config,
        "best_epoch": best_epoch,
        "best_val_cindex": round(float(best_val_c), 4),
        "val_per_cohort_cindex": val_per_cohort,
        "history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def eval_val_per_cohort(model, loader, device, risk_fn, bin_centers, cohort_map):
    """Overall + per-cohort C-index on a val/held-out loader."""
    model.eval()
    all_risks, all_times, all_events, all_domains = [], [], [], []
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in loader:
            surv_out, _, _ = model(x_cont.to(device), x_cat.to(device), alpha=0.0)
            risk = risk_fn(surv_out, bin_centers)
            all_risks.append(risk.cpu().numpy())
            all_times.append(t.numpy())
            all_events.append(e.numpy())
            all_domains.append(d.numpy())
    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    all_domains = np.concatenate(all_domains)
    per_coh = {}
    for d in sorted(np.unique(all_domains)):
        mask = all_domains == d
        if mask.sum() < 5 or all_events[mask].sum() < 2:
            continue
        ci = concordance_index(all_times[mask], -all_risks[mask], all_events[mask])
        name = {v: k for k, v in cohort_map.items()}[int(d)]
        per_coh[name] = round(float(ci), 4)
    return per_coh


# ===================================================================
# 5. Orchestrator
# ===================================================================
def run_split(exp_letter, df, seeds, epochs, patience, out_root):
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

    weights, prop_diag = fit_propensity_and_weights(data)
    log.info(f"Propensity bal-acc (train): {prop_diag['propensity_balanced_acc']} | "
             f"w: min={prop_diag['weight']['min']}, med={prop_diag['weight']['median']}, "
             f"max={prop_diag['weight']['max']} | per-cohort mean w: "
             f"{prop_diag['per_cohort_mean_weight']}")

    base_cfg = {
        "surv_type": "deephit", "d_model": 128, "n_layers": 4,
        "dropout": 0.15, "lr": 5e-4, "n_bins": 32,
        "epochs": epochs, "patience": patience,
        "per_cohort_loss": True, "domain_weight_max": 0.3,
    }

    exp_out = out_root / exp_key
    records = []
    for seed in seeds:
        # paper-protocol ERM & DANN (per-cohort loss, engine)
        # NOTE: the engine's train_model dispatches on mode == 'baseline' (no GRL)
        # vs 'dann' (GRL + domain loss); 'erm'/'erm_pooled' are ERM and therefore
        # run through the 'baseline' branch (per_cohort_loss toggled below).
        for label, engine_mode, dw in [("erm", "baseline", 0.0), ("dann", "dann", 0.3)]:
            out_dir = exp_out / label / f"seed_{seed}"
            cfg = {**base_cfg, "domain_weight_max": dw, "seed": seed}
            log.info(f"\n>>> {label.upper()} seed={seed}")
            _, res, _ = train_model(cfg, data, mode=engine_mode, output_dir=out_dir)
            # val per-cohort on the held-out val (same metric for all modes)
            risk_fn, _ = train_engine.make_loss_fn("deephit", data["bin_edges"])
            model = TransDANNSurvV3(
                num_continuous=len(data["cont_features"]),
                num_categorical=len(data["cat_features"]),
                cat_cardinalities=data["cat_cardinalities"],
                d_model=128, n_heads=8, n_layers=4, dropout=0.15,
                num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32,
            ).to(DEVICE)
            model.load_state_dict(torch.load(out_dir / "best_model.pth",
                                             map_location=DEVICE, weights_only=True))
            vpc = eval_val_per_cohort(model, data["val_loader"], DEVICE,
                                      lambda x, bc: deep_hit_risk(x, bc),
                                      data["bin_centers"], data["cohort_map"])
            records.append({
                "experiment": exp_key, "mode": label, "seed": seed,
                "best_val_cindex": res.get("best_val_cindex"),
                "val_per_cohort_cindex": vpc,
            })

        # pooled unweighted ERM (per_cohort_loss=False); direct IPW control
        out_dir = exp_out / "erm_pooled" / f"seed_{seed}"
        cfg = {**base_cfg, "per_cohort_loss": False, "seed": seed}
        log.info(f"\n>>> ERM_POOLED (no per-cohort, no weights) seed={seed}")
        _, res, _ = train_model(cfg, data, mode="baseline", output_dir=out_dir)
        model = TransDANNSurvV3(
            num_continuous=len(data["cont_features"]),
            num_categorical=len(data["cat_features"]),
            cat_cardinalities=data["cat_cardinalities"],
            d_model=128, n_heads=8, n_layers=4, dropout=0.15,
            num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32,
        ).to(DEVICE)
        model.load_state_dict(torch.load(out_dir / "best_model.pth",
                                         map_location=DEVICE, weights_only=True))
        vpc = eval_val_per_cohort(model, data["val_loader"], DEVICE,
                                  lambda x, bc: deep_hit_risk(x, bc),
                                  data["bin_centers"], data["cohort_map"])
        records.append({
            "experiment": exp_key, "mode": "erm_pooled", "seed": seed,
            "best_val_cindex": res.get("best_val_cindex"),
            "val_per_cohort_cindex": vpc,
        })

        # IPW-ERM (weighted pooled loss)
        out_dir = exp_out / "ipw_erm" / f"seed_{seed}"
        cfg = {**base_cfg, "per_cohort_loss": False, "seed": seed}
        log.info(f"\n>>> IPW-ERM seed={seed}")
        res = train_ipw_erm(cfg, data, weights, out_dir)
        records.append({
            "experiment": exp_key, "mode": "ipw_erm", "seed": seed,
            "best_val_cindex": res.get("best_val_cindex"),
            "val_per_cohort_cindex": res.get("val_per_cohort_cindex", {}),
        })

    with open(exp_out / "runs.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    return records, prop_diag


def aggregate(records):
    rows = {}
    for r in records:
        k = (r["experiment"], r["mode"])
        rows.setdefault(k, []).append(r)
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


def write_table_csv(summary, all_prop):
    """Write results/tables/table11_ipw_erm.csv (per experiment × mode)."""
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
    for ek, p in all_prop.items():
        rows.append({"experiment": ek, "mode": "propensity_bal_acc",
                     "cindex_mean": p["propensity_balanced_acc"], "cindex_std": None,
                     "cindex_each": str(p["weight"])})
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "tables" / "table11_ipw_erm.csv", index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table11_ipw_erm.csv'} "
             f"({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="IPW-ERM prototype")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPS,
                        choices=list(EXP_KEY))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
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

    all_records, all_prop = [], {}
    for letter in args.experiments:
        recs, prop = run_split(letter, df, args.seeds, args.epochs, args.patience, EXP_OUT)
        all_records.extend(recs)
        all_prop[EXP_KEY[letter]] = prop

    summary = aggregate(all_records)
    out_json = {
        "experiments": args.experiments,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "propensity": all_prop,
        "summary": summary,
        "all_runs": all_records,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXP_OUT / "ipw_summary.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    with open(EXP_OUT / "runs.json", "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    write_table_csv(summary, all_prop)

    # ---- human-readable ----
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("IPW-ERM PROTOTYPE SUMMARY  (seeds %s)" % (args.seeds,))
    lines.append("=" * 100)
    for s in summary:
        rows = "[" + ", ".join(f"{x:.4f}" for x in s["cindex_each"]) + "]"
        lines.append(f"  {s['experiment']:<18}{s['mode']:<11}"
                     f"best-val C={s['cindex_mean']:.4f}±{s['cindex_std']:.4f}  "
                     f"{rows}")
        for cname, v in s["val_per_cohort_mean"].items():
            if v is not None:
                lines.append(f"      val-{cname:<12}C={v:.4f}")
    lines.append("=" * 100)
    for ek, p in all_prop.items():
        lines.append(f"  {ek}: prop bal-acc={p['propensity_balanced_acc']}, "
                     f"w[min/med/max]={p['weight']['min']}/{p['weight']['median']}"
                     f"/{p['weight']['max']}, per-cohort mean w={p['per_cohort_mean_weight']}")
    lines.append("=" * 100)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {EXP_OUT}/ipw_summary.json, runs.json, summary.txt")
    log.info("\n✅ IPW-ERM prototype complete!")


if __name__ == "__main__":
    main()
