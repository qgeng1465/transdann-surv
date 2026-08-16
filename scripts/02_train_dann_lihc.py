#!/usr/bin/env python3
"""
02_train_dann_lihc.py; TransDANN Training Engine for LIHC
============================================================
Trains TransDANNSurvV3 on LIHC cohorts with two modes:
  - DANN mode (with GRL): Full domain-adversarial training
  - Baseline mode (no GRL): Same architecture, no domain loss

Experiments:
  A) TCGA_LIHC vs US_SEER; extreme missingness contrast (Level 4)
  B) TCGA_LIHC vs External HCC; different databases, same cancer (Level 2)
  C) All LIHC cohorts combined; multi-domain (2-6 domains)

For each experiment × mode, we:
  1. Preprocess data (impute, encode, scale)
  2. Train with early stopping + cosine annealing
  3. Save best model + training curves + evaluation metrics
  4. Record domain classification accuracy during training

Usage:
  python scripts/02_train_dann_lihc.py                   # train all experiments
  python scripts/02_train_dann_lihc.py --experiment A     # single experiment
  python scripts/02_train_dann_lihc.py --quick            # faster, fewer epochs
"""

import argparse
import json
import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add parent to path for transdann_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transdann_utils import (
    TransDANNSurvV3, GradientReversal,
    cox_loss, deep_hit_loss, deep_hit_risk,
    compute_time_bins, time_dependent_auc,
    LiverCancerDataset, prepare_v3_data, DEVICE
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("train_lihc")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
RESULTS_DIR = BASE_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "A_tcga_vs_seer": {
        "name": "TCGA-LIHC vs SEER-LIHC (Level 4; Extreme Missingness)",
        "cohorts": ["TCGA_LIHC", "US_SEER"],
        "description": "Extreme missingness contrast: genomic vs registry data",
    },
    "B_tcga_vs_external": {
        "name": "TCGA-LIHC vs External HCC (Level 2; Different Databases)",
        "cohorts": ["TCGA_LIHC", "hcc_msk_2024", "lihc_amc_prv", "hcc_meric_2021"],
        "description": "Same cancer, different independent clinical cohorts",
    },
    "C_all_cohorts": {
        "name": "All LIHC Cohorts (Multi-Domain)",
        "cohorts": ["TCGA_LIHC", "US_SEER", "hcc_msk_2024", "lihc_amc_prv", "hcc_meric_2021"],
        "description": "All liver cancer cohorts combined",
    },
}

CONT_FEATURES = ["Age"]
CAT_FEATURES = ["Sex", "Stage", "Grade"]


# ===================================================================
# 1. Data Loading & Preprocessing
# ===================================================================
def load_and_preprocess(df, surv_type="deephit", n_bins=32, subsample_seer=10000):
    """Load LIHC data and prepare for training.

    Parameters
    ----------
    df : pd.DataFrame; combined LIHC data with 'Source' column
    surv_type : str; 'deephit' or 'cox'
    n_bins : int; number of time bins for DeepHit
    subsample_seer : int or None; randomly subsample SEER to this size

    Returns dict of preprocessed tensors, loaders, metadata.
    """
    log.info(f"Total samples: {len(df)}")
    log.info(f"Cohorts:\n{df['Source'].value_counts().to_string()}")
    log.info(f"Event rates:\n{df.groupby('Source')['Vital_Status'].mean().to_string()}")

    # Optional SEER subsampling to balance
    if subsample_seer and "US_SEER" in df["Source"].values:
        seer_idx = df[df["Source"] == "US_SEER"].index
        seer_n = len(seer_idx)
        if seer_n > subsample_seer:
            keep = np.random.RandomState(42).choice(
                seer_idx, subsample_seer, replace=False
            )
            drop = seer_idx.difference(pd.Index(keep))
            df = df.drop(drop).copy()
            log.info(f"Subsampled SEER: {seer_n} → {subsample_seer}")

    # ---- Impute continuous missing with median ----
    for col in CONT_FEATURES:
        col_median = df[col].median()
        if pd.isna(col_median):
            # Entire column is NaN (e.g., hcc_msk_2024 has 100% missing Age)
            # Use a safe default (global median of non-NaN, or 50)
            col_median = df[col].dropna().median() if df[col].notna().any() else 50.0
        df[col] = df[col].fillna(col_median)

    # ---- Encode categorical ----
    label_encoders = {}
    cat_cardinalities = []
    for col in CAT_FEATURES:
        # Convert to string, fill NaN with placeholder for unknown
        df[col] = df[col].fillna(-1).astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))

    # ---- Standardise continuous ----
    scaler = StandardScaler()
    df[CONT_FEATURES] = scaler.fit_transform(df[CONT_FEATURES])

    # ---- Create domain labels ----
    cohorts = sorted(df["Source"].unique())
    cohort_map = {c: i for i, c in enumerate(cohorts)}
    df["Domain_Label"] = df["Source"].map(cohort_map)
    num_domains = len(cohorts)

    log.info(f"Continuous: {len(CONT_FEATURES)}, Categorical: {len(CAT_FEATURES)}")
    log.info(f"Domains: {cohort_map}")
    log.info(f"Cat cardinalities: {cat_cardinalities}")

    # ---- Train / Val split (stratified by cohort) ----
    np.random.seed(42)
    val_idx, train_idx = [], []
    for cohort in df["Source"].unique():
        cidx = df[df["Source"] == cohort].index.tolist()
        np.random.shuffle(cidx)
        n_val = max(1, int(len(cidx) * 0.15))
        val_idx.extend(cidx[:n_val])
        train_idx.extend(cidx[n_val:])

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    log.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # ---- Time bins (from training event times) ----
    bin_edges, bin_centers = None, None
    if surv_type == "deephit":
        bin_edges, bin_centers = compute_time_bins(
            torch.tensor(train_df["Survival_Months"].values),
            torch.tensor(train_df["Vital_Status"].values),
            n_bins=n_bins,
        )
        log.info(f"Time bins: {n_bins} bins, "
                 f"range [{bin_edges[0]:.1f}–{bin_edges[-1]:.1f}] months")

    # ---- DataLoaders ----
    train_ds = LiverCancerDataset(
        train_df, CONT_FEATURES, CAT_FEATURES,
        domain_label_col="Domain_Label"
    )
    val_ds = LiverCancerDataset(
        val_df, CONT_FEATURES, CAT_FEATURES,
        domain_label_col="Domain_Label"
    )

    # Weighted sampler for cohort balance
    domain_counts = train_df["Domain_Label"].value_counts().sort_index()
    w = 1.0 / np.sqrt(domain_counts.values + 1)
    sw = w[train_df["Domain_Label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sw),
        num_samples=len(sw), replacement=True
    )

    batch_size = min(512, max(64, len(train_df) // 4))
    # Ensure batch_size doesn't exceed dataset size
    batch_size = min(batch_size, len(train_df))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    return {
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


# ===================================================================
# 2. Loss helpers
# ===================================================================
def make_loss_fn(surv_type, bin_edges=None):
    """Return (loss_fn, risk_fn) pair."""
    if surv_type == "cox":
        def loss_fn(pred, times, events, cohorts=None):
            return cox_loss(pred, events, times)
        return loss_fn, lambda x, bc: x.squeeze(-1)
    elif surv_type == "deephit":
        assert bin_edges is not None, "DeepHit requires bin_edges"
        def loss_fn(pred, times, events, cohorts=None):
            return deep_hit_loss(pred, times, events, bin_edges)
        return loss_fn, lambda x, bc: deep_hit_risk(x, bc)
    raise ValueError(f"Unknown surv_type: {surv_type}")


def per_cohort_loss(loss_fn, predictions, times, events, domain_labels, n_domains):
    """Per-cohort unweighted mean loss."""
    losses = []
    for d in range(n_domains):
        mask = domain_labels == d
        if mask.sum() < 5:
            continue
        loss_c = loss_fn(predictions[mask], times[mask], events[mask])
        losses.append(loss_c)
    if not losses:
        return torch.tensor(0.0, device=times.device)
    return torch.stack(losses).mean()


# ===================================================================
# 3. Evaluation
# ===================================================================
def evaluate(model, val_loader, device, risk_fn, bin_centers=None):
    """Compute C-index on validation set."""
    model.eval()
    all_risks, all_times, all_events, all_domains = [], [], [], []
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in val_loader:
            surv_out, _, _ = model(x_cont.to(device), x_cat.to(device), alpha=0.0)
            risk = risk_fn(surv_out, bin_centers if bin_centers is not None and not (isinstance(bin_centers, torch.Tensor) and bin_centers.numel() == 0) else None)
            all_risks.append(risk.cpu().numpy())
            all_times.append(t.numpy())
            all_events.append(e.numpy())
            all_domains.append(d.numpy())

    from lifelines.utils import concordance_index
    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    all_domains = np.concatenate(all_domains)

    # Overall
    try:
        overall_c = concordance_index(all_times, -all_risks, all_events)
    except Exception:
        overall_c = 0.5

    return overall_c


def full_evaluation(model, full_loader, device, risk_fn, bin_centers=None):
    """Full evaluation with per-cohort C-index and time-dependent AUC."""
    from lifelines.utils import concordance_index
    model.eval()
    all_risks, all_times, all_events, all_domains, all_feats = [], [], [], [], []
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in full_loader:
            surv_out, _, feat = model(x_cont.to(device), x_cat.to(device), alpha=0.0)
            risk = risk_fn(surv_out, bin_centers if bin_centers is not None else None)
            all_risks.append(risk.cpu().numpy())
            all_times.append(t.numpy())
            all_events.append(e.numpy())
            all_domains.append(d.numpy())
            all_feats.append(feat.cpu().numpy())

    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    all_domains = np.concatenate(all_domains)
    all_feats = np.vstack(all_feats) if all_feats else np.array([])

    # Per-cohort C-index
    per_cohort = {}
    for d in sorted(np.unique(all_domains)):
        mask = all_domains == d
        if mask.sum() < 5 or all_events[mask].sum() < 2:
            continue
        ci = concordance_index(all_times[mask], -all_risks[mask], all_events[mask])
        per_cohort[int(d)] = float(f"{ci:.4f}")

    # Time-dependent AUC
    td_auc = time_dependent_auc(all_times, all_events, all_risks, eval_times=(12, 36, 60))

    return per_cohort, td_auc, all_feats, all_risks


# ===================================================================
# 4. Training
# ===================================================================
def train_model(config, data, mode="dann", output_dir=None):
    """Train TransDANNSurvV3 in DANN or Baseline mode.

    Parameters
    ----------
    config : dict; training hyperparameters
    data : dict; from load_and_preprocess()
    mode : str; 'dann' (with GRL, domain loss) or 'baseline' (no domain loss)
    output_dir : Path; where to save models + logs
    """
    SurvType = config.get("surv_type", "deephit")
    D_MODEL = config.get("d_model", 128)
    N_LAYERS = config.get("n_layers", 4)
    DROPOUT = config.get("dropout", 0.15)
    LR = config.get("lr", 5e-4)
    N_BINS = config.get("n_bins", 32)
    EPOCHS = config.get("epochs", 200)
    PATIENCE = config.get("patience", 40)
    PER_COHORT_LOSS = config.get("per_cohort_loss", True)
    DOMAIN_WEIGHT_MAX = config.get("domain_weight_max", 0.3)
    # If set, fix the GRL reversal scale to a constant instead of the default
    # 0→1 schedule. Used by 19_exp_alpha_sensitivity.py for the α-sweep;
    # None preserves the original DANN schedule for all saved results.
    ALPHA_FIXED = config.get("alpha_fixed", None)

    # Fix torch RNG so training is reproducible (advisory: the numbers already
    # saved in results/ were produced before this seed was added; a re-run may
    # shift C-index by ~±0.001 without changing any conclusion).
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = DEVICE

    if output_dir is None:
        output_dir = RESULTS_DIR / f"train_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup
    loss_fn, risk_fn = make_loss_fn(SurvType, data["bin_edges"])
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    num_domains = data["num_domains"]
    n_params_logged = False

    # Model
    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=D_MODEL, n_heads=8, n_layers=N_LAYERS,
        dropout=DROPOUT, num_domains=num_domains,
        surv_head_type=SurvType, n_bins=N_BINS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    domain_criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_params:,}")
    log.info(f"Mode: {'DANN (with GRL)' if mode == 'dann' else 'Baseline (no GRL)'}")

    # Training loop
    best_val_c = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"epoch": [], "surv_loss": [], "domain_loss": [], "val_cindex": [],
               "domain_acc": [], "lr": [], "alpha": []}

    for epoch in range(EPOCHS):
        model.train()
        total_surv = 0.0
        total_dom = 0.0
        total_dom_acc = 0.0
        n_batches = 0

        p = float(epoch) / EPOCHS
        if ALPHA_FIXED is not None:
            alpha = float(ALPHA_FIXED)
        else:
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        domain_weight = DOMAIN_WEIGHT_MAX * p  # gradually increase

        for x_cont, x_cat, t, e, d in train_loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            t, e, d = t.to(device), e.to(device), d.to(device)

            optimizer.zero_grad()

            if mode == "baseline":
                # No GRL; train survival only
                surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
                if PER_COHORT_LOSS:
                    loss_surv = per_cohort_loss(
                        loss_fn, surv_out, t, e, d, num_domains
                    )
                else:
                    loss_surv = loss_fn(surv_out, t, e)
                loss_total = loss_surv
                loss_domain = torch.tensor(0.0)
                dom_acc = 0.0
            else:
                # DANN mode: with GRL and domain loss
                surv_out, domain_pred, _ = model(x_cont, x_cat, alpha=alpha)

                if PER_COHORT_LOSS:
                    loss_surv = per_cohort_loss(
                        loss_fn, surv_out, t, e, d, num_domains
                    )
                else:
                    loss_surv = loss_fn(surv_out, t, e)

                loss_domain = domain_criterion(domain_pred, d)
                loss_total = loss_surv + domain_weight * loss_domain

                # Domain accuracy
                dom_pred_class = domain_pred.argmax(dim=1)
                dom_acc = (dom_pred_class == d).float().mean().item()

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_surv += loss_surv.item()
            total_dom += loss_domain.item() if isinstance(loss_domain, torch.Tensor) and loss_domain.requires_grad else loss_domain
            total_dom_acc += dom_acc if mode == "dann" else 0.0
            n_batches += 1

        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_c = evaluate(model, val_loader, device, risk_fn, data["bin_centers"])
            avg_s = total_surv / max(n_batches, 1)
            avg_d = total_dom / max(n_batches, 1)
            avg_da = total_dom_acc / max(n_batches, 1) if mode == "dann" else 0.0
            lr_now = scheduler.get_last_lr()[0]

            history["epoch"].append(epoch + 1)
            history["surv_loss"].append(float(f"{avg_s:.4f}"))
            history["domain_loss"].append(float(f"{avg_d:.4f}"))
            history["val_cindex"].append(float(f"{val_c:.4f}"))
            history["domain_acc"].append(float(f"{avg_da:.4f}"))
            history["lr"].append(float(f"{lr_now:.2e}"))
            history["alpha"].append(float(f"{alpha:.2f}"))

            log.info(
                f"  E{epoch+1:3d} | α={alpha:.2f} | dw={domain_weight:.2f} | "
                f"SurvL={avg_s:.3f} | DomL={avg_d:.3f} | "
                f"DomAcc={avg_da:.3f} | ValC={val_c:.4f} | LR={lr_now:.2e}"
            )

            if val_c > best_val_c:
                best_val_c = val_c
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                no_improve += 5

            if no_improve >= PATIENCE:
                log.info(f"  Early stop at E{epoch+1}, best={best_epoch}, "
                         f"Val C={best_val_c:.4f}")
                break

    # Load best model
    log.info(f"\nBest: epoch {best_epoch}, Val C-index = {best_val_c:.4f}")
    model.load_state_dict(
        torch.load(output_dir / "best_model.pth", map_location=device, weights_only=True)
    )
    model.eval()

    # Full evaluation
    full_df = pd.concat([data["train_df"], data["val_df"]], ignore_index=True)
    full_ds = LiverCancerDataset(
        full_df, data["cont_features"], data["cat_features"],
        domain_label_col="Domain_Label"
    )
    full_loader = DataLoader(full_ds, batch_size=4096, shuffle=False)
    per_cohort, td_auc, all_feats, all_risks = full_evaluation(
        model, full_loader, device, risk_fn, data["bin_centers"]
    )

    log.info(f"Per-cohort C-index: {per_cohort}")
    log.info(f"Time-dependent AUC: {td_auc}")

    # Save results
    results = {
        "mode": mode,
        "config": config,
        "cohort_map": data["cohort_map"],
        "best_epoch": best_epoch,
        "best_val_cindex": round(float(best_val_c), 4),
        "per_cohort_cindex": per_cohort,
        "time_dependent_auc": {str(k): round(float(v), 4) if not np.isnan(v) else None
                               for k, v in td_auc.items()},
        "history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save feature embeddings for visualization
    np.save(output_dir / "features.npy", all_feats)
    np.save(output_dir / "risks.npy", all_risks)
    full_df[["Source", "Domain_Label"]].to_csv(output_dir / "domain_labels.csv", index=False)

    return model, results, all_feats


# ===================================================================
# 5. Main runner
# ===================================================================
def run_experiment(exp_key, exp_cfg, base_config, df, quick=False):
    """Run one experiment (DANN + Baseline) for a given cohort set."""
    log.info(f"\n{'#'*60}")
    log.info(f"EXPERIMENT: {exp_cfg['name']}")
    log.info(f"Cohorts: {exp_cfg['cohorts']}")
    log.info(f"{'#'*60}")

    # Filter to relevant cohorts
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    log.info(f"Samples: {len(exp_df)}")

    # Preprocess
    data = load_and_preprocess(
        exp_df,
        surv_type=base_config["surv_type"],
        n_bins=base_config["n_bins"],
        subsample_seer=5000 if quick else 10000,
    )

    # Create output dir
    exp_dir = RESULTS_DIR / "lihc_experiments" / exp_key
    os.makedirs(exp_dir, exist_ok=True)

    # Save data info
    cohort_map = data["cohort_map"]
    log.info(f"Domain map: {cohort_map}")

    # Train DANN
    log.info(f"\n{'─'*50}")
    log.info("TRAINING: DANN (with GRL)")
    log.info(f"{'─'*50}")
    dann_dir = exp_dir / "dann"
    try:
        _, dann_results, dann_feats = train_model(base_config, data, mode="dann", output_dir=dann_dir)
    except Exception as e:
        log.error(f"DANN training failed: {e}")
        import traceback; traceback.print_exc()
        dann_results = {"mode": "dann", "error": str(e)}
        dann_feats = None

    # Train Baseline
    log.info(f"\n{'─'*50}")
    log.info("TRAINING: BASELINE (no GRL)")
    log.info(f"{'─'*50}")
    base_dir = exp_dir / "baseline"
    try:
        base_config_no_domain = {**base_config, "domain_weight_max": 0.0}
        _, base_results, base_feats = train_model(
            base_config_no_domain, data, mode="baseline", output_dir=base_dir
        )
    except Exception as e:
        log.error(f"Baseline training failed: {e}")
        import traceback; traceback.print_exc()
        base_results = {"mode": "baseline", "error": str(e)}
        base_feats = None

    # Compare
    log.info(f"\n{'='*50}")
    log.info("COMPARISON: DANN vs Baseline")
    log.info(f"{'='*50}")
    dann_c = dann_results.get("best_val_cindex", "ERR")
    base_c = base_results.get("best_val_cindex", "ERR")
    delta = f"{float(dann_c) - float(base_c):+.4f}" if dann_c != "ERR" and base_c != "ERR" else "ERR"
    log.info(f"  DANN     Val C-index: {dann_c}")
    log.info(f"  Baseline Val C-index: {base_c}")
    log.info(f"  Δ = {delta}")

    # Per-cohort comparison
    dann_pc = dann_results.get("per_cohort_cindex", {})
    base_pc = base_results.get("per_cohort_cindex", {})
    rev_map = {v: k for k, v in cohort_map.items()}
    for dom_id in sorted(rev_map.keys()):
        cname = rev_map[dom_id]
        dc = dann_pc.get(str(dom_id), "N/A")
        bc = base_pc.get(str(dom_id), "N/A")
        log.info(f"  {cname:20s}: DANN={dc} | Baseline={bc}")

    # Save comparison
    comparison = {
        "experiment": exp_key,
        "cohorts": exp_cfg["cohorts"],
        "dann_val_cindex": dann_results.get("best_val_cindex"),
        "baseline_val_cindex": base_results.get("best_val_cindex"),
        "delta": delta,
        "cohort_map": cohort_map,
        "dann_per_cohort": dann_pc,
        "baseline_per_cohort": base_pc,
    }
    with open(exp_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="TransDANN LIHC Training")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()) + ["ALL"],
                       default="ALL", help="Which experiment to run")
    parser.add_argument("--quick", action="store_true",
                       help="Quick run (fewer epochs, fewer SEER samples)")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Training epochs (default: 200)")
    parser.add_argument("--surv_type", choices=["deephit", "cox"], default="deephit")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--domain_weight", type=float, default=0.3,
                       help="Max domain loss weight (default: 0.3)")
    args = parser.parse_args()

    log.info(f"PyTorch {torch.__version__}, Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                 f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Load data
    log.info("\nLoading LIHC data ...")
    if not DATA_PATH.exists():
        log.error(f"Data not found: {DATA_PATH}")
        log.error("Run 01_prepare_lihc_data.py first!")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    log.info(f"Loaded {len(df)} rows, {df['Source'].nunique()} cohorts")

    # Base config
    base_config = {
        "surv_type": args.surv_type,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "n_bins": 32,
        "epochs": 100 if args.quick else args.epochs,
        "patience": 20 if args.quick else 40,
        "per_cohort_loss": True,
        "domain_weight_max": args.domain_weight,
    }

    # Run experiments
    experiments_to_run = (
        [args.experiment] if args.experiment != "ALL" else list(EXPERIMENTS.keys())
    )

    all_comparisons = {}
    for exp_key in experiments_to_run:
        if exp_key not in EXPERIMENTS:
            log.warning(f"Unknown experiment: {exp_key}, skipping")
            continue
        try:
            comp = run_experiment(exp_key, EXPERIMENTS[exp_key], base_config, df,
                                 quick=args.quick)
            all_comparisons[exp_key] = comp
        except Exception as e:
            log.error(f"Experiment {exp_key} failed: {e}")
            import traceback; traceback.print_exc()

    # Summary
    log.info(f"\n{'='*60}")
    log.info("OVERALL SUMMARY")
    log.info(f"{'='*60}")
    for exp_key, comp in all_comparisons.items():
        name = EXPERIMENTS[exp_key]["name"]
        dann_c = comp.get("dann_val_cindex", "?")
        base_c = comp.get("baseline_val_cindex", "?")
        delta = comp.get("delta", "?")
        log.info(f"  {name:60s}")
        log.info(f"    DANN={dann_c}  Baseline={base_c}  Δ={delta}")

    # Save global summary
    summary_path = RESULTS_DIR / "lihc_experiments" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_comparisons, f, indent=2, default=str)
    log.info(f"\nSummary saved to {summary_path}")
    log.info("\n✅ Training complete!")


if __name__ == "__main__":
    main()
