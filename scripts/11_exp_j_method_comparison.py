#!/usr/bin/env python3
"""
11_exp_j_method_comparison.py; Experiment J: comprehensive methodological comparison (Method Comparison)
=================================================================================
Core evidence for the paper: it is NOT that "DANN was poorly tuned"; the whole
domain-adaptation / domain-generalisation family fails on medical survival data.

Runs 9 methods on the SAME data splits as Experiments A/B/C:
    ERM, DANN, CORAL, IRM, GroupDRO, V-REx, Fish, Mixup, MLDG
Each method shares the identical backbone (TransDANNSurvV3, 4×8-head, d_model=128,
DeepHit 32-bin survival head) and identical training config (AdamW 5e-4, cosine
annealing, early-stop patience=40, per-cohort loss).  The only difference is the
regularisation/adversarial/alignment term added on top of the survival loss.

Reported per method: C-index (overall + per cohort), Integrated Brier Score,
time-dependent AUC @ 12/36/60, domain-classifier accuracy (DANN only), wall-clock
time, and a 1000-sample stratified bootstrap of Δ vs ERM (95% CI + p-value).

Outputs
-------
  results/experiments/exp_j/method_comparison.json
  results/tables/table3_method_comparison.csv
  results/logs/exp_j_training.log
  results/figures/extended/J1_method_comparison_cindex.png
  results/figures/extended/J2_method_comparison_ibs.png
  results/figures/extended/J3_method_comparison_bootstrap_forest.png
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index

# IRM and Fish penalties require *second-order* gradients. PyTorch's fused
# SDPA kernels (flash / memory-efficient) do not implement double backward, so
# force the plain math attention kernel for the whole run. This keeps every
# method on the identical attention implementation (fairness).
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from transdann_utils import (
    TransDANNSurvV3, deep_hit_loss, deep_hit_risk,
    compute_time_bins, time_dependent_auc, LiverCancerDataset, DEVICE,
)
from lihc_recon import BASE_DIR, build_data, EXPERIMENTS
from method_utils import (
    per_domain_losses, coral_penalty, irmv1_penalty, fish_penalty,
    vrex_penalty, groupdro_loss, mixup_pair_perm,
    deephit_survival_probabilities, survival_at_times,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
EXP_OUT = RESULTS_DIR / "experiments" / "exp_j"
TABLE_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
for d in (EXP_OUT, TABLE_DIR, LOG_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

METHODS = ["ERM", "DANN", "CORAL", "IRM", "GroupDRO", "V-REx", "Fish", "Mixup", "MLDG"]

# Method hyper-parameters (λ of the regularisation term).  Values follow the
# common practice in DomainBed / the original papers.
METHOD_HPARAMS = {
    "CORAL":    {"lambda": 1.0},
    "IRM":      {"lambda": 1.0},
    "V-REx":    {"lambda": 1.0},
    "Fish":     {"lambda": 1.0},
    "Mixup":    {"lambda": 1.0, "alpha": 2.0},
    "MLDG":     {"beta": 1.0, "lr_inner": 0.05},
}


# ===================================================================
# Logging helper (structured epoch-level log lines)
# ===================================================================
def setup_logger(exp_name):
    import logging
    logger = logging.getLogger(exp_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    fh = logging.FileHandler(LOG_DIR / f"{exp_name}_training.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger("exp_j")


# ===================================================================
# Training loop; one method
# ===================================================================
def train_method(method, config, data, seed, output_dir):
    """Train one method on the given data dict. Returns (model, results)."""
    os.makedirs(output_dir, exist_ok=True)
    device = DEVICE
    loss_fn, risk_fn = _make_loss_fn(config["surv_type"], data["bin_edges"])
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    n_domains = data["num_domains"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=config["d_model"], n_heads=8, n_layers=config["n_layers"],
        dropout=config["dropout"], num_domains=n_domains,
        surv_head_type=config["surv_type"], n_bins=config["n_bins"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    domain_criterion = nn.CrossEntropyLoss()

    hparams = METHOD_HPARAMS.get(method, {})
    alpha_max = config.get("domain_weight_max", 0.3)
    EPOCHS, PATIENCE = config["epochs"], config["patience"]

    best_val_c = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"epoch": [], "surv_loss": [], "domain_loss": [], "val_cindex": [],
               "domain_acc": [], "lr": [], "alpha": []}
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        p = float(epoch) / EPOCHS
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        domain_weight = alpha_max * p
        total_surv = total_dom = total_acc = 0.0
        n_batches = 0

        for x_cont, x_cat, t, e, d in train_loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            t, e, d = t.to(device), e.to(device), d.to(device)
            optimizer.zero_grad()

            if method == "ERM":
                surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
                losses = per_domain_losses(loss_fn, surv_out, t, e, d, n_domains)
                loss_total = torch.stack(losses).mean()
                loss_dom = torch.tensor(0.0); dom_acc = 0.0

            elif method == "DANN":
                surv_out, domain_pred, _ = model(x_cont, x_cat, alpha=alpha)
                losses = per_domain_losses(loss_fn, surv_out, t, e, d, n_domains)
                loss_total = torch.stack(losses).mean() + domain_weight * domain_criterion(domain_pred, d)
                loss_dom = domain_criterion(domain_pred, d)
                dom_acc = (domain_pred.argmax(dim=1) == d).float().mean().item()

            elif method == "CORAL":
                surv_out, _, cls = model(x_cont, x_cat, alpha=0.0)
                losses = per_domain_losses(loss_fn, surv_out, t, e, d, n_domains)
                loss_total = torch.stack(losses).mean() + hparams["lambda"] * coral_penalty(cls, d, n_domains)
                loss_dom = torch.tensor(0.0); dom_acc = 0.0

            elif method == "IRM":
                surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
                losses = per_domain_losses(loss_fn, surv_out, t, e, d, n_domains)
                pen = irmv1_penalty(loss_fn, surv_out, t, e, d, n_domains)
                loss_total = torch.stack(losses).mean() + hparams["lambda"] * pen
                loss_dom = torch.tensor(0.0); dom_acc = 0.0

            elif method == "GroupDRO":
                surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
                losses = per_domain_losses(loss_fn, surv_out, t, e, d, n_domains)
                loss_total = groupdro_loss(losses)
                loss_dom = torch.tensor(0.0); dom_acc = 0.0

            elif method == "V-REx":
                surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
                losses = per_domain_losses(loss_fn, surv_out, t, e, d, n_domains)
                loss_total = torch.stack(losses).mean() + hparams["lambda"] * vrex_penalty(losses)
                loss_dom = torch.tensor(0.0); dom_acc = 0.0

            elif method == "Fish":
                surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
                losses = per_domain_losses(loss_fn, surv_out, t, e, d, n_domains)
                loss_total = torch.stack(losses).mean() + hparams["lambda"] * fish_penalty(list(model.parameters()), losses)
                loss_dom = torch.tensor(0.0); dom_acc = 0.0

            elif method == "Mixup":
                loss_total, loss_dom, dom_acc = _mixup_step(
                    model, loss_fn, x_cont, x_cat, t, e, d, n_domains, hparams
                )

            elif method == "MLDG":
                loss_total, loss_dom, dom_acc = _mldg_step(
                    model, x_cont, x_cat, t, e, d, n_domains, loss_fn, hparams
                )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_surv += loss_total.item()
            total_dom += loss_dom.item()
            total_acc += dom_acc
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_c = _evaluate_cindex(model, val_loader, device, risk_fn, data["bin_centers"])
            history["epoch"].append(epoch + 1)
            history["surv_loss"].append(round(total_surv / max(n_batches, 1), 4))
            history["domain_loss"].append(round(total_dom / max(n_batches, 1), 4))
            history["val_cindex"].append(round(val_c, 4))
            history["domain_acc"].append(round(total_acc / max(n_batches, 1), 4) if method == "DANN" else 0.0)
            history["lr"].append(float(f"{scheduler.get_last_lr()[0]:.2e}"))
            history["alpha"].append(round(alpha, 2))
            logger.info(
                f"[{method} seed={seed}] E{epoch+1:3d} α={alpha:.2f} dw={domain_weight:.2f} "
                f"SurvL={history['surv_loss'][-1]:.3f} ValC={val_c:.4f}"
            )
            if val_c > best_val_c:
                best_val_c = val_c
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                no_improve += 5
            if no_improve >= PATIENCE:
                logger.info(f"[{method} seed={seed}] early stop @ E{epoch+1}, best={best_epoch}")
                break

    model.load_state_dict(
        torch.load(output_dir / "best_model.pth", map_location=device, weights_only=True)
    )
    model.eval()
    elapsed = time.time() - t_start

    # Full evaluation on val split (same population the reported Δ uses)
    eval_metrics = _full_val_evaluation(model, data, device, config)
    results = {
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_cindex": round(float(best_val_c), 4),
        "per_cohort_cindex": eval_metrics["per_cohort"],
        "time_dependent_auc": eval_metrics["td_auc"],
        "ibs": eval_metrics["ibs"],
        "domain_acc_peak": round(max(history["domain_acc"]), 4) if method == "DANN" else None,
        "elapsed_sec": round(elapsed, 1),
        "history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    np.save(output_dir / "risks.npy", eval_metrics["risks"])
    return model, results


# ===================================================================
# Mixup step; cross-domain interpolation with two-label loss
# ===================================================================
def _mixup_step(model, loss_fn, x_cont, x_cat, t, e, d, n_domains, hparams):
    device = x_cont.device
    # Use the *global* seeded NumPy RNG (np.random.seed(seed) is set at the top
    # of each run).  A fresh unseeded RandomState() here made the cross-domain
    # pairing non-reproducible; the numbers in results/experiments/exp_j were
    # produced before this fix and may shift slightly on a re-run.
    perm = mixup_pair_perm(d, np.random)
    lam = float(np.random.beta(hparams.get("alpha", 2.0), hparams.get("alpha", 2.0)))

    x_cont_mix = lam * x_cont + (1 - lam) * x_cont[perm]
    # Two forward passes: mixed continuous features + each sample's categoricals
    surv_a, _, _ = model(x_cont_mix, x_cat, alpha=0.0)
    surv_b, _, _ = model(x_cont_mix, x_cat[perm], alpha=0.0)
    loss_a = loss_fn(surv_a, t, e)
    loss_b = loss_fn(surv_b, t[perm], e[perm])
    loss_total = lam * loss_a + (1 - lam) * loss_b
    return loss_total, torch.tensor(0.0), 0.0


# ===================================================================
# MLDG step; leave-one-domain-out meta-learning (first-order)
# ===================================================================
def _mldg_step(model, x_cont, x_cat, t, e, d, n_domains, loss_fn, hparams):
    device = x_cont.device
    beta = hparams.get("beta", 1.0)
    lr_inner = hparams.get("lr_inner", 0.05)
    params = list(model.parameters())

    surv, _, _ = model(x_cont, x_cat, alpha=0.0)
    doms = [dom for dom in range(n_domains) if int((d == dom).sum()) >= 5]
    if len(doms) < 2:
        losses = per_domain_losses(loss_fn, surv, t, e, d, n_domains)
        return torch.stack(losses).mean(), torch.tensor(0.0), 0.0
    held = int(np.random.choice(doms))
    train_doms = [x for x in doms if x != held]

    loss_inner = torch.stack([loss_fn(surv[d == dom], t[d == dom], e[d == dom]) for dom in train_doms]).mean()
    # retain_graph: the inner-loss graph must survive until the outer backward
    grads = torch.autograd.grad(loss_inner, params, allow_unused=True,
                                create_graph=False, retain_graph=True)
    # inner parameter state (first-order: inner grads treated as constants)
    state = {}
    g_iter = iter(grads)
    for name, p in model.named_parameters():
        g = next(g_iter)
        state[name] = p - lr_inner * g if g is not None else p.detach()

    mask = d == held
    if int(mask.sum()) < 5:
        return loss_inner, torch.tensor(0.0), 0.0
    surv_outer, _, _ = torch.func.functional_call(
        model, state, (x_cont[mask], x_cat[mask], torch.tensor(0.0, device=device))
    )
    loss_outer = loss_fn(surv_outer, t[mask], e[mask])
    loss_total = loss_inner + beta * loss_outer
    return loss_total, torch.tensor(0.0), 0.0


# ===================================================================
# Loss / risk helpers
# ===================================================================
def _make_loss_fn(surv_type, bin_edges):
    if surv_type == "cox":
        from transdann_utils import cox_loss
        def loss_fn(pred, t, e):
            return cox_loss(pred, e, t)
        return loss_fn, lambda x, bc: x.squeeze(-1)
    def loss_fn(pred, t, e):
        return deep_hit_loss(pred, t, e, bin_edges)
    return loss_fn, lambda x, bc: deep_hit_risk(x, bc)


def _evaluate_cindex(model, val_loader, device, risk_fn, bin_centers):
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for x_cont, x_cat, t, e, _d in val_loader:
            surv_out, _, _ = model(x_cont.to(device), x_cat.to(device), alpha=0.0)
            risks.append(risk_fn(surv_out, bin_centers).cpu().numpy())
            times.append(t.numpy()); events.append(e.numpy())
    risks = np.concatenate(risks); times = np.concatenate(times); events = np.concatenate(events)
    try:
        return concordance_index(times, -risks, events)
    except Exception:
        return 0.5


def _full_val_evaluation(model, data, device, config):
    """C-index, per-cohort C-index, time-AUC and IBS on the validation split."""
    loss_fn, risk_fn = _make_loss_fn(config["surv_type"], data["bin_edges"])
    val_loader = data["val_loader"]
    model.eval()
    risks, times, events, domains = [], [], [], []
    with torch.no_grad():
        for x_cont, x_cat, t, e, d in val_loader:
            surv_out, _, _ = model(x_cont.to(device), x_cat.to(device), alpha=0.0)
            risks.append(risk_fn(surv_out, data["bin_centers"]).cpu().numpy())
            times.append(t.numpy()); events.append(e.numpy()); domains.append(d.numpy())
    risks = np.concatenate(risks); times = np.concatenate(times)
    events = np.concatenate(events); domains = np.concatenate(domains)

    try:
        overall = concordance_index(times, -risks, events)
    except Exception:
        overall = 0.5
    per_cohort = {}
    for dom in sorted(np.unique(domains)):
        m = domains == dom
        if m.sum() < 5 or events[m].sum() < 2:
            continue
        per_cohort[int(dom)] = round(float(concordance_index(times[m], -risks[m], events[m])), 4)
    td_auc = time_dependent_auc(times, events, risks, eval_times=(12, 36, 60))

    ibs = _compute_ibs(model, data, device)
    return {"overall": round(float(overall), 4), "per_cohort": per_cohort,
            "td_auc": {str(k): round(float(v), 4) if not np.isnan(v) else None for k, v in td_auc.items()},
            "ibs": ibs, "risks": risks}


def _compute_ibs(model, data, device):
    """Integrated Brier Score on the validation split (scikit-survival)."""
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv
    train_times = data["train_df"]["Survival_Months"].values
    train_events = data["train_df"]["Vital_Status"].values.astype(bool)
    val_times = data["val_df"]["Survival_Months"].values
    val_events = data["val_df"]["Vital_Status"].values.astype(bool)

    # scikit-survival requires every evaluation time to lie strictly inside the
    # test follow-up window [min(test times), max(test times)).
    lo = max(float(np.min(train_times)), float(np.min(val_times)), 0.1)
    hi = min(float(np.quantile(train_times, 0.9)), float(np.max(val_times)) * 0.999)
    if hi <= lo:
        return None
    times_grid = np.linspace(lo, hi, 40)

    bin_edges = data["bin_edges"].to(device)
    model.eval()
    probs = []
    with torch.no_grad():
        for x_cont, x_cat, t, e, _d in data["val_loader"]:
            surv_out, _, _ = model(x_cont.to(device), x_cat.to(device), alpha=0.0)
            surv, centers = deephit_survival_probabilities(surv_out, bin_edges)
            S = survival_at_times(surv, centers, torch.tensor(times_grid, device=device))
            probs.append(S.cpu().numpy())
    probs = np.concatenate(probs)
    if probs.shape[0] == 0:
        return None
    y_train = Surv.from_arrays(train_events, train_times)
    y_test = Surv.from_arrays(val_events, val_times)
    try:
        ibs = float(integrated_brier_score(y_train, y_test, probs, times_grid))
        return round(ibs, 4)
    except Exception as e:
        logger.warning(f"IBS failed: {e}")
        return None


# ===================================================================
# Bootstrap Δ vs ERM (stratified by cohort, on val rows)
# ===================================================================
def _stratified_bootstrap_delta(times, events, r_method, r_erm, domains,
                                B=1000, seed=777):
    """Bootstrap Δ = C-index(method) − C-index(ERM) on the val population,
    resampling with replacement *within each cohort* (same recipe as Exp F)."""
    from fast_cindex import fast_lifelines_cindex
    rng = np.random.RandomState(seed)
    def cidx(t, e, r):
        if len(t) < 2 or e.sum() < 1:
            return np.nan
        return fast_lifelines_cindex(t, -r, e)
    uniq = np.unique(domains)
    deltas = np.zeros(B)
    for i in range(B):
        idx = []
        for u in uniq:
            m = np.where(domains == u)[0]
            idx.append(m[rng.randint(0, len(m), size=len(m))])
        idx = np.concatenate(idx)
        deltas[i] = cidx(times[idx], events[idx], r_method[idx]) - cidx(times[idx], events[idx], r_erm[idx])
    return deltas


def compute_bootstraps(letter, methods, val_df, risks_by_method_seed, B=1000):
    """Pool the 3 seeds' bootstrap draws for each method vs ERM."""
    times = val_df["Survival_Months"].values.astype(float)
    events = val_df["Vital_Status"].values.astype(float)
    domains = val_df["Domain_Label"].values.astype(int)
    out = {}
    if "ERM" not in risks_by_method_seed:
        return out
    for method in methods:
        if method == "ERM":
            continue
        draws = []
        for seed in risks_by_method_seed["ERM"].keys():
            r_erm = risks_by_method_seed["ERM"][seed]
            r_m = risks_by_method_seed[method].get(seed)
            if r_m is None or r_erm is None or len(r_m) != len(times):
                continue
            draws.append(_stratified_bootstrap_delta(
                times, events, r_m, r_erm, domains, B=B))
        if not draws:
            continue
        pooled = np.concatenate(draws)
        out[method] = {
            "delta_mean": round(float(np.mean(pooled)), 4),
            "ci_low": round(float(np.percentile(pooled, 2.5)), 4),
            "ci_high": round(float(np.percentile(pooled, 97.5)), 4),
            "p_dann_superior": round(float(np.mean(pooled > 0)), 4),
            "p_two_sided": round(min(2 * min(float(np.mean(pooled > 0)),
                                             float(np.mean(pooled < 0))), 1.0), 4),
            "n_draws": int(len(pooled)),
        }
    return out


# ===================================================================
# Main
# ===================================================================
def _run_split(letter, exp_key, exp_cfg, df, methods, seeds, config,
               subsample_seer=10000, out_root=None):
    """Train all requested methods on one data split and aggregate.

    Returns (exp_results, data) where exp_results[method] holds the 3-seed
    aggregates and exp_results["__bootstrap__"] the Δ-vs-ERM bootstrap.
    `out_root` allows routing saved models to a different directory (used by
    the --seer_subsample sweep so runs don't clobber the canonical outputs).
    """
    out_root = out_root or EXP_OUT
    logger.info(f"\n{'#'*70}\nEXPERIMENT {letter}: {exp_cfg['name']} "
                f"(SEER subsample={subsample_seer})\n{'#'*70}")
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    data = build_data(exp_df, ["Age"], ["Sex", "Stage", "Grade"],
                      surv_type="deephit", n_bins=32, subsample_seer=subsample_seer)
    val_df = data["val_df"]
    n_seer_used = int((val_df["Source"].values == "US_SEER").sum()
                      + (data["train_df"]["Source"].values == "US_SEER").sum())

    exp_results = {}
    risks_by_seed = {}
    for method in methods:
        logger.info(f"\n--- Method {method} (seeds {seeds}) ---")
        seed_results = []
        risks_by_seed[method] = {}
        for seed in seeds:
            out_dir = out_root / f"exp_{letter}_{method}" / f"seed_{seed}"
            logger.info(f"[{letter}][{method}] seed={seed} start")
            t0 = time.time()
            try:
                _, res = train_method(method, config, data, seed, out_dir)
                seed_results.append(res)
                rfile = out_dir / "risks.npy"
                if rfile.exists():
                    risks_by_seed[method][seed] = np.load(rfile)
                logger.info(f"[{letter}][{method}] seed={seed} done "
                            f"C={res['best_val_cindex']} IBS={res['ibs']} "
                            f"({time.time()-t0:.0f}s)")
            except Exception as ex:
                import traceback
                traceback.print_exc()
                logger.error(f"[{letter}][{method}] seed={seed} FAILED: {ex}")
                seed_results.append(None)
        # Aggregate across seeds
        ok = [r for r in seed_results if r is not None]
        if ok:
            cinds = [r["best_val_cindex"] for r in ok]
            ibss = [r["ibs"] for r in ok if r["ibs"] is not None]
            exp_results[method] = {
                "cindex_mean": round(float(np.mean(cinds)), 4),
                "cindex_std": round(float(np.std(cinds)), 4),
                "cindex_per_seed": cinds,
                "ibs_mean": round(float(np.mean(ibss)), 4) if ibss else None,
                "ibs_per_seed": ibss,
                "elapsed_sec": round(float(np.mean([r["elapsed_sec"] for r in ok])), 1),
                "n_seeds_ok": len(ok),
            }
        else:
            exp_results[method] = None

    # 1000-sample stratified bootstrap of Δ vs ERM (pooled across seeds)
    boot = compute_bootstraps(letter, methods, val_df, risks_by_seed, B=1000)
    exp_results["__bootstrap__"] = boot
    return exp_results, data, n_seer_used


def _run_subsample_sweep(df, methods, seeds, config, sizes):
    """Reviewer point #3: is the 'SEER dominates' multi-domain result Δ≈0 an
    artefact of SEER being ~96% of the samples?  Retrain ERM vs DANN on the
    Exp-C (5-domain) split with SEER downsampled to 10000/5000/2000/1000/500/309
    and check whether Δ remains ≈ 0 and insignificant."""
    exp_key, exp_cfg = "C_all_cohorts", EXPERIMENTS["C_all_cohorts"]
    summary = {}
    for size in sizes:
        letter = f"C_s{size}"
        out_root = EXP_OUT / f"subsample_C_seer{size}"
        logger.info(f"\n{'='*70}\nSEER SUBSAMPLE sweep → {size} "
                    f"(Exp C, {methods} × seeds {seeds})\n{'='*70}")
        exp_results, data, n_seer = _run_split(
            letter, exp_key, exp_cfg, df, methods, seeds, config,
            subsample_seer=size, out_root=out_root)
        boot = exp_results.get("__bootstrap__", {}).get("DANN", {})
        entry = {
            "seer_target": size,
            "n_seer_used": n_seer,
            "n_total": int(len(data["train_df"]) + len(data["val_df"])),
            "n_val": int(len(data["val_df"])),
            "seer_share_of_full": round(n_seer / (len(data["train_df"]) + len(data["val_df"])), 4),
            "erm_cindex": exp_results["ERM"]["cindex_mean"],
            "dann_cindex": exp_results["DANN"]["cindex_mean"],
            "delta": round(exp_results["DANN"]["cindex_mean"]
                           - exp_results["ERM"]["cindex_mean"], 4),
            "bootstrap_delta_mean": boot.get("delta_mean"),
            "ci_low": boot.get("ci_low"), "ci_high": boot.get("ci_high"),
            "p_two_sided": boot.get("p_two_sided"),
        }
        summary[str(size)] = entry
        with open(out_root / "results.json", "w") as f:
            json.dump({"seer_target": size, "entry": entry,
                       "exp_results": {k: v for k, v in exp_results.items()
                                       if k != "__bootstrap__"},
                       "bootstrap": exp_results.get("__bootstrap__")}, f, indent=2)
        logger.info(f"  [{size}] Δ = {entry['delta']:+.4f}  "
                    f"p = {entry['p_two_sided']}  (SEER {n_seer}/{entry['n_total']})")

    # Combined summary JSON + CSV
    with open(EXP_OUT / "seer_subsample_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    rows = [{**{"size": int(k)}, **{kk: vv for kk, vv in v.items() if kk != "seer_target"}}
            for k, v in summary.items()]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table3b_seer_subsample.csv", index=False)
    logger.info(f"Summary → {EXP_OUT / 'seer_subsample_summary.json'}")
    logger.info(f"Table   → {TABLE_DIR / 'table3b_seer_subsample.csv'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None, choices=METHODS,
                        help="Methods to run (default: all 9)")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seer_subsample", nargs="+", type=int, default=None,
                        choices=[10000, 5000, 2000, 1000, 500, 309],
                        help="SEER downsampling sensitivity analysis (Exp C, "
                             "ERM vs DANN only). Give one or more sizes, e.g. "
                             "--seer_subsample 10000 2000 500.")
    args = parser.parse_args()

    methods = args.methods or METHODS
    exp_keys = {"A": "A_tcga_vs_seer", "B": "B_tcga_vs_external", "C": "C_all_cohorts"}

    # Map experiment letter → cohorts for data prep
    config = {
        "surv_type": "deephit", "d_model": 128, "n_layers": 4, "dropout": 0.15,
        "lr": 5e-4, "n_bins": 32,
        "epochs": 60 if args.quick else args.epochs,
        "patience": 15 if args.quick else args.patience,
        "per_cohort_loss": True, "domain_weight_max": 0.3,
    }

    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()

    # ---- SEER downsampling sensitivity analysis (reviewer point #3) ----
    if args.seer_subsample:
        sweep_methods = [m for m in ("ERM", "DANN") if m in methods] or ["ERM", "DANN"]
        _run_subsample_sweep(df, sweep_methods, args.seeds, config,
                             sorted(set(args.seer_subsample)))
        print(f"\n✅ SEER subsample sweep done. See "
              f"{EXP_OUT / 'seer_subsample_summary.json'}")
        return

    all_comp = {}
    for letter in args.experiments:
        exp_key = exp_keys[letter]
        exp_cfg = EXPERIMENTS[exp_key]
        exp_results, _, _ = _run_split(letter, exp_key, exp_cfg, df, methods,
                                       args.seeds, config, subsample_seer=10000)
        all_comp[letter] = exp_results

    # Save
    def _clean(v):
        if v is None:
            return None
        return {kk: vv for kk, vv in v.items() if kk != "best_seed"}
    with open(EXP_OUT / "method_comparison.json", "w") as f:
        json.dump({"config": config, "methods": methods,
                   "results": {k: {m: (_clean(v) if m != "__bootstrap__" else v)
                                   for m, v in mv.items()}
                               for k, mv in all_comp.items()}},
                  f, indent=2, default=str)

    # CSV table
    rows = []
    for letter in all_comp:
        boot = all_comp[letter].get("__bootstrap__", {})
        for method in METHODS:
            r = all_comp[letter].get(method)
            b = boot.get(method, {})
            rows.append({
                "experiment": letter, "method": method,
                "cindex_mean": r["cindex_mean"] if r else None,
                "cindex_std": r["cindex_std"] if r else None,
                "ibs": r["ibs_mean"] if r else None,
                "delta_vs_erm": b.get("delta_mean"),
                "ci_low": b.get("ci_low"), "ci_high": b.get("ci_high"),
                "p_two_sided": b.get("p_two_sided"),
                "elapsed_sec": r["elapsed_sec"] if r else None,
            })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table3_method_comparison.csv", index=False)

    # Figures
    _make_figures(all_comp)

    logger.info("\n✅ Experiment J complete.")
    print("\nSUMMARY (C-index mean ± std):")
    for letter, mv in all_comp.items():
        print(f"  [{letter}]")
        for method in METHODS:
            r = mv.get(method)
            if r:
                print(f"    {method:8s}: {r['cindex_mean']:.4f} ± {r['cindex_std']:.4f}  "
                      f"IBS={r['ibs_mean']}")
    print(f"\nTable → {TABLE_DIR / 'table3_method_comparison.csv'}")
    print(f"JSON  → {EXP_OUT / 'method_comparison.json'}")


def _num_or_nan(v):
    return v if v is not None else np.nan


def _method_entry(all_comp, letter, method):
    """Safely fetch a method's aggregate entry (may be None for failed runs)."""
    r = all_comp.get(letter, {}).get(method)
    return r if r is not None else {}


def _make_figures(all_comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(METHODS)))
    letters = list(all_comp.keys())

    # J1: C-index grouped bar chart
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(letters))
    w = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        means = [_num_or_nan(_method_entry(all_comp, l, method).get("cindex_mean")) for l in letters]
        ax.bar(x + (i - len(METHODS) / 2 + 0.5) * w, means, w, label=method,
               color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(letters)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_ylabel("C-index (val)"); ax.set_xlabel("experiment (A/B/C split)")
    ax.set_title("Experiment J; C-index across 9 DA/DG methods")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.set_ylim(0.5, 0.75)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "J1_method_comparison_cindex.png", dpi=200)
    plt.close(fig)

    # J2: IBS grouped bar chart
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, method in enumerate(METHODS):
        ibss = [_num_or_nan(_method_entry(all_comp, l, method).get("ibs_mean")) for l in letters]
        ax.bar(x + (i - len(METHODS) / 2 + 0.5) * w, ibss, w, label=method, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(letters)
    ax.set_ylabel("Integrated Brier Score (lower better)")
    ax.set_title("Experiment J; IBS across 9 DA/DG methods")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "J2_method_comparison_ibs.png", dpi=200)
    plt.close(fig)

    # J3: Δ vs ERM bootstrap forest plot
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rows = []
    for letter in letters:
        boot = all_comp.get(letter, {}).get("__bootstrap__", {})
        for method in METHODS:
            if method == "ERM":
                continue
            b = boot.get(method)
            if not b:
                continue
            sig = "✱" if b["p_two_sided"] < 0.05 else ""
            rows.append((letter, method, b["delta_mean"], b["ci_low"], b["ci_high"], sig))
    if rows:
        rows.sort(key=lambda x: (x[0], x[2]))
        y = np.arange(len(rows))
        for i, (letter, method, mean, lo, hi, sig) in enumerate(rows):
            ax.plot([lo, hi], [y[i], y[i]], color="#4c72b0", lw=1.6)
            ax.plot(mean, y[i], "o", ms=6, color="#c44e52")
            ax.text(lo - 0.0004, y[i], f"{mean:+.4f}{sig}", va="center", ha="right", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{l}:{m}" for l, m, *_ in rows], fontsize=8)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set_xlabel("Δ C-index vs ERM (95% bootstrap CI; ✱ = p<0.05)")
        ax.set_title("Experiment J; 1000× stratified bootstrap of Δ vs ERM")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "J3_method_comparison_bootstrap_forest.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
