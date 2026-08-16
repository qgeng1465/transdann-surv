#!/usr/bin/env python3
"""
30_exp_ipw_mixup.py; IPW × Mixup joint experiment
==================================================

Background: after Exp J's BH-FDR correction, **Mixup is the only non-adversarial
method significantly positive** (C: ΔC-index=+0.019, p<0.05); IPW-ERM (script 24)
shows a paired improvement of +0.016/+0.031/+0.055 on the target-cohort TCGA
validation set but it is unstable. This experiment **stacks** the two:
on top of Mixup's cross-domain interpolation regularisation, it applies the
propensity IPW weights to the per-sample loss, testing whether "input-level Mixup +
reweighting" can obtain the benefits of both mechanisms at once.

Design (rigorous logic, single change)
--------------------------------------
1. Data/optimizer/early stopping/validation all identical to script 24 (same
   build_data, same cohort-balanced WeightedRandomSampler, same DeepHit 32 bins,
   same AdamW/Cosine/patience=40).
2. Four modes share the same config; the only difference is **how the training loss
   is constructed**:
     erm        : engine baseline mode, per_cohort_loss=True (paper ERM protocol)
     mixup      : script 11's Mixup step reproduced verbatim (mixup_pair_perm
                   cross-domain pairing, λ~Beta(2,2), continuous-feature
                   interpolation only, weighted loss over two forward passes), no IPW;
                   intended as a control for Exp J's Mixup (should reproduce its
                   best-val magnitude).
     ipw_erm    : script 24's per-sample IPW-weighted pooled DeepHit loss (reused).
     ipw_mixup  : IPW stacked on the Mixup step; weight the mixup pair's two label
                   loss terms by each sample's own w:
                   L = Σ_i [λ·w_i·ℓ_a,i + (1−λ)·w_{perm,i}·ℓ_b,i]
                       / Σ_i [λ·w_i + (1−λ)·w_{perm,i}]
3. Propensity scores identical to script 24 (multinomial logistic, fitted on the
   training set only, π truncated to [0.05,0.95], w=1/π, w∈[1.05,20]).
4. Evaluation: overall best-val C-index + per-cohort C-index on the validation set
   (target TCGA is key).

Outputs (results/experiments/exp_ipw_mixup/):
  - runs.json                per (exp, mode, seed) records
  - ipw_mixup_summary.json   aggregate
  - summary.txt
  - results/tables/table17_ipw_mixup.csv
  - logs/30_exp_ipw_mixup.log

Usage:
  python3 scripts/30_exp_ipw_mixup.py                # A/B/C × 3 seeds × 4 modes
  python3 scripts/30_exp_ipw_mixup.py --experiments A --seeds 42   # smoke
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
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_ipw_mixup"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "30_exp_ipw_mixup.log", mode="w"),
    ],
)
log = logging.getLogger("ipw_mixup")

# ---------------------------------------------------------------------------
# Reuse script 24 (IPW machinery) + script 11 (Mixup step) + engine
# ---------------------------------------------------------------------------
spec24 = importlib.util.spec_from_file_location(
    "exp_ipw_24", SCRIPTS_DIR / "24_exp_ipw_erm.py"
)
exp24 = importlib.util.module_from_spec(spec24)
spec24.loader.exec_module(exp24)
fit_propensity_and_weights = exp24.fit_propensity_and_weights
build_weighted_loader = exp24.build_weighted_loader
eval_val_per_cohort = exp24.eval_val_per_cohort
WeightedSurvDataset = exp24.WeightedSurvDataset
EXP_KEY = exp24.EXP_KEY

spec02 = importlib.util.spec_from_file_location(
    "train_engine", SCRIPTS_DIR / "02_train_dann_lihc.py"
)
train_engine = importlib.util.module_from_spec(spec02)
spec02.loader.exec_module(train_engine)
EXPERIMENTS = train_engine.EXPERIMENTS
train_model = train_engine.train_model
evaluate = train_engine.evaluate

from transdann_utils import (  # noqa: E402
    DEVICE, TransDANNSurvV3, deep_hit_loss, deep_hit_risk, compute_time_bins,
)
from lihc_recon import build_data, CONT_FEATURES, CAT_FEATURES  # noqa: E402
from method_utils import mixup_pair_perm  # noqa: E402

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_EXPS = ["A", "B", "C"]


# ===================================================================
# 1. Per-sample DeepHit NLL (for IPW-weighted mixup)
# ===================================================================
def per_sample_deep_hit_nll(hazard_logits, times, events, bin_edges):
    """(batch,) per-sample NLL; same formula as deep_hit_loss, no mean()."""
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
    return torch.where(events == 1, -torch.log(h_t + 1e-8), -torch.log(S_t))


# ===================================================================
# 2. Training loops
# ===================================================================
def _make_model(data, config):
    return TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=config.get("d_model", 128), n_heads=8,
        n_layers=config.get("n_layers", 4),
        dropout=config.get("dropout", 0.15),
        num_domains=data["num_domains"],
        surv_head_type="deephit", n_bins=config.get("n_bins", 32),
    ).to(DEVICE)


def _setup(config):
    EPOCHS = config.get("epochs", 200)
    PATIENCE = config.get("patience", 40)
    LR = config.get("lr", 5e-4)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return EPOCHS, PATIENCE, LR


def _save_best(model, best_val_c, best_epoch, no_improve, val_c, epoch,
               output_dir, history):
    if val_c > best_val_c:
        best_val_c = val_c
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(), output_dir / "best_model.pth")
    else:
        no_improve += 5
    return best_val_c, best_epoch, no_improve


def train_mixup(config, data, output_dir):
    """Unweighted input-level Mixup; replicates script 11's _mixup_step verbatim
    (pooled loss, β(2,2), cross-domain pairing).  No IPW."""
    EPOCHS, PATIENCE, LR = _setup(config)
    os.makedirs(output_dir, exist_ok=True)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    bin_edges = data["bin_edges"]
    bin_centers = data["bin_centers"]

    model = _make_model(data, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )
    best_val_c, best_epoch, no_improve = 0.0, 0, 0
    history = {"epoch": [], "surv_loss": [], "val_cindex": [], "lr": []}

    def risk_fn(x, bc):
        return deep_hit_risk(x, bc)

    for epoch in range(EPOCHS):
        model.train()
        total_surv, n_batches = 0.0, 0
        for x_cont, x_cat, t, e, d in train_loader:
            x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
            t, e = t.to(DEVICE), e.to(DEVICE)
            perm = mixup_pair_perm(d, np.random)
            lam = float(np.random.beta(2.0, 2.0))
            x_mix = lam * x_cont + (1 - lam) * x_cont[perm]
            optimizer.zero_grad()
            surv_a, _, _ = model(x_mix, x_cat, alpha=0.0)
            surv_b, _, _ = model(x_mix, x_cat[perm], alpha=0.0)
            loss_a = deep_hit_loss(surv_a, t, e, bin_edges)
            loss_b = deep_hit_loss(surv_b, t[perm], e[perm], bin_edges)
            loss_total = lam * loss_a + (1 - lam) * loss_b
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_surv += loss_total.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_c = evaluate(model, val_loader, DEVICE, risk_fn, bin_centers)
            history["epoch"].append(epoch + 1)
            history["surv_loss"].append(float(f"{total_surv / max(n_batches, 1):.4f}"))
            history["val_cindex"].append(float(f"{val_c:.4f}"))
            history["lr"].append(float(f"{scheduler.get_last_lr()[0]:.2e}"))
            best_val_c, best_epoch, no_improve = _save_best(
                model, best_val_c, best_epoch, no_improve, val_c, epoch,
                output_dir, history)
            log.info(f"  E{epoch + 1:3d} | Mixup SurvL={history['surv_loss'][-1]:.3f} "
                     f"| ValC={val_c:.4f} | LR={history['lr'][-1]:.2e}")
            if no_improve >= PATIENCE:
                log.info(f"  Early stop at E{epoch + 1}, best={best_epoch}, "
                         f"Val C={best_val_c:.4f}")
                break

    model.load_state_dict(torch.load(output_dir / "best_model.pth",
                                     map_location=DEVICE, weights_only=True))
    model.eval()
    val_per_cohort = eval_val_per_cohort(model, val_loader, DEVICE, risk_fn,
                                         bin_centers, data["cohort_map"])
    return {"mode": "mixup", "best_epoch": best_epoch,
            "best_val_cindex": round(float(best_val_c), 4),
            "val_per_cohort_cindex": val_per_cohort, "history": history}


def train_ipw_mixup(config, data, weights, output_dir):
    """IPW × Mixup: the Mixup step, with each label-loss term weighted by the
    owning sample's propensity weight w."""
    EPOCHS, PATIENCE, LR = _setup(config)
    os.makedirs(output_dir, exist_ok=True)
    train_loader = build_weighted_loader(data, weights)
    val_loader = data["val_loader"]
    bin_edges = data["bin_edges"]
    bin_centers = data["bin_centers"]

    model = _make_model(data, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )
    best_val_c, best_epoch, no_improve = 0.0, 0, 0
    history = {"epoch": [], "surv_loss": [], "val_cindex": [], "lr": []}

    def risk_fn(x, bc):
        return deep_hit_risk(x, bc)

    for epoch in range(EPOCHS):
        model.train()
        total_surv, n_batches = 0.0, 0
        for x_cont, x_cat, t, e, d, w in train_loader:
            x_cont, x_cat = x_cont.to(DEVICE), x_cat.to(DEVICE)
            t, e, w = t.to(DEVICE), e.to(DEVICE), w.to(DEVICE)
            perm = mixup_pair_perm(d, np.random)
            lam = float(np.random.beta(2.0, 2.0))
            x_mix = lam * x_cont + (1 - lam) * x_cont[perm]
            optimizer.zero_grad()
            surv_a, _, _ = model(x_mix, x_cat, alpha=0.0)
            surv_b, _, _ = model(x_mix, x_cat[perm], alpha=0.0)
            nll_a = per_sample_deep_hit_nll(surv_a, t, e, bin_edges)
            nll_b = per_sample_deep_hit_nll(surv_b, t[perm], e[perm], bin_edges)
            wa = lam * w
            wb = (1 - lam) * w[perm]
            loss_total = (wa * nll_a + wb * nll_b).sum() / (wa + wb).sum().clamp(min=1e-8)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_surv += loss_total.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_c = evaluate(model, val_loader, DEVICE, risk_fn, bin_centers)
            history["epoch"].append(epoch + 1)
            history["surv_loss"].append(float(f"{total_surv / max(n_batches, 1):.4f}"))
            history["val_cindex"].append(float(f"{val_c:.4f}"))
            history["lr"].append(float(f"{scheduler.get_last_lr()[0]:.2e}"))
            best_val_c, best_epoch, no_improve = _save_best(
                model, best_val_c, best_epoch, no_improve, val_c, epoch,
                output_dir, history)
            log.info(f"  E{epoch + 1:3d} | IPW-Mixup SurvL={history['surv_loss'][-1]:.3f} "
                     f"| ValC={val_c:.4f} | LR={history['lr'][-1]:.2e}")
            if no_improve >= PATIENCE:
                log.info(f"  Early stop at E{epoch + 1}, best={best_epoch}, "
                         f"Val C={best_val_c:.4f}")
                break

    model.load_state_dict(torch.load(output_dir / "best_model.pth",
                                     map_location=DEVICE, weights_only=True))
    model.eval()
    val_per_cohort = eval_val_per_cohort(model, val_loader, DEVICE, risk_fn,
                                         bin_centers, data["cohort_map"])
    return {"mode": "ipw_mixup", "best_epoch": best_epoch,
            "best_val_cindex": round(float(best_val_c), 4),
            "val_per_cohort_cindex": val_per_cohort, "history": history}


# ===================================================================
# 3. Orchestrator
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
             f"w[min/med/max]={prop_diag['weight']['min']}/"
             f"{prop_diag['weight']['median']}/{prop_diag['weight']['max']}")

    base_cfg = {
        "surv_type": "deephit", "d_model": 128, "n_layers": 4,
        "dropout": 0.15, "lr": 5e-4, "n_bins": 32,
        "epochs": epochs, "patience": patience,
        "per_cohort_loss": True, "domain_weight_max": 0.3,
    }

    exp_out = out_root / exp_key
    records = []
    for seed in seeds:
        # erm (paper protocol, engine)
        out_dir = exp_out / "erm" / f"seed_{seed}"
        cfg = {**base_cfg, "seed": seed}
        log.info(f"\n>>> ERM seed={seed}")
        _, res, _ = train_model(cfg, data, mode="baseline", output_dir=out_dir)
        model = _make_model(data, cfg)
        model.load_state_dict(torch.load(out_dir / "best_model.pth",
                                         map_location=DEVICE, weights_only=True))
        vpc = eval_val_per_cohort(model, data["val_loader"], DEVICE,
                                  lambda x, bc: deep_hit_risk(x, bc),
                                  data["bin_centers"], data["cohort_map"])
        records.append({
            "experiment": exp_key, "mode": "erm", "seed": seed,
            "best_val_cindex": res.get("best_val_cindex"),
            "val_per_cohort_cindex": vpc,
        })

        # mixup (unweighted, script-11 recipe)
        out_dir = exp_out / "mixup" / f"seed_{seed}"
        cfg = {**base_cfg, "seed": seed}
        log.info(f"\n>>> MIXUP seed={seed}")
        r = train_mixup(cfg, data, out_dir)
        records.append({
            "experiment": exp_key, "mode": "mixup", "seed": seed,
            "best_val_cindex": r["best_val_cindex"],
            "val_per_cohort_cindex": r["val_per_cohort_cindex"],
        })

        # ipw_erm (script 24 recipe)
        out_dir = exp_out / "ipw_erm" / f"seed_{seed}"
        cfg = {**base_cfg, "seed": seed}
        log.info(f"\n>>> IPW-ERM seed={seed}")
        r = exp24.train_ipw_erm(cfg, data, weights, out_dir)
        records.append({
            "experiment": exp_key, "mode": "ipw_erm", "seed": seed,
            "best_val_cindex": r["best_val_cindex"],
            "val_per_cohort_cindex": r["val_per_cohort_cindex"],
        })

        # ipw_mixup
        out_dir = exp_out / "ipw_mixup" / f"seed_{seed}"
        cfg = {**base_cfg, "seed": seed}
        log.info(f"\n>>> IPW-MIXUP seed={seed}")
        r = train_ipw_mixup(cfg, data, weights, out_dir)
        records.append({
            "experiment": exp_key, "mode": "ipw_mixup", "seed": seed,
            "best_val_cindex": r["best_val_cindex"],
            "val_per_cohort_cindex": r["val_per_cohort_cindex"],
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


def write_table_csv(summary):
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
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "table17_ipw_mixup.csv",
                              index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table17_ipw_mixup.csv'} "
             f"({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="IPW × Mixup joint")
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
        recs, prop = run_split(letter, df, args.seeds, args.epochs, args.patience,
                               EXP_OUT)
        all_records.extend(recs)
        all_prop[EXP_KEY[letter]] = prop

    summary = aggregate(all_records)
    out_json = {
        "experiments": args.experiments, "seeds": args.seeds,
        "epochs": args.epochs, "patience": args.patience,
        "propensity": all_prop, "summary": summary, "all_runs": all_records,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXP_OUT / "ipw_mixup_summary.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    with open(EXP_OUT / "runs.json", "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    write_table_csv(summary)

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("IPW × MIXUP JOINT SUMMARY  (seeds %s)" % (args.seeds,))
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
                     f"/{p['weight']['max']}")
    lines.append("=" * 100)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    log.info(f"\nSaved: {EXP_OUT}/ipw_mixup_summary.json, runs.json, summary.txt")
    log.info("\n✅ IPW × Mixup joint complete!")


if __name__ == "__main__":
    main()
