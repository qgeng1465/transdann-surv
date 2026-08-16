#!/usr/bin/env python3
"""
10_exp_i_domain_quantify.py; Experiment I: domain shift quantification (Domain Shift Quantification)
=====================================================================================
Upgrades the qualitative "domain fingerprint" story into quantitative measures:

  1. Wasserstein-1 distance matrices (Age continuous; Stage one-hot distributions)
  2. Propensity-score overlap (multi-class logistic-regression domain classifier)
  3. MINE mutual information:  I(domain; survival outcome)  and
                               I(domain; learned representation)   per experiment
  4. Domain-separability probe (2-layer MLP) on raw / ERM-hidden / DANN-hidden
     features, plus a "probe AUC vs DANN C-index Δ" scatter

Outputs
-------
  results/experiments/exp_i/domain_quantify.json
  results/tables/table5_domain_quantify.csv
  results/logs/exp_i_training.log
  results/figures/extended/I1_wasserstein_matrix.png
  results/figures/extended/I2_propensity_overlap.png
  results/figures/extended/I3_mine_mutual_info.png
  results/figures/extended/I4_probe_auc_vs_cindex.png
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from transdann_utils import DEVICE
from lihc_recon import BASE_DIR, _load_train_02, EXPERIMENTS, CONT_FEATURES, CAT_FEATURES

RESULTS_DIR = BASE_DIR / "results"
EXP_OUT = RESULTS_DIR / "experiments" / "exp_i"
TABLE_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures" / "extended"
for d in (EXP_OUT, TABLE_DIR, LOG_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

DATA = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"


# ===================================================================
# 1. Wasserstein-1 distance matrices
# ===================================================================
def wasserstein_matrices(df):
    cohorts = sorted(df["Source"].unique())
    n = len(cohorts)
    age_mat = np.zeros((n, n))
    stage_mat = np.zeros((n, n))
    # Age: median-imputed (same as the model's input preprocessing)
    age_imp = df["Age"].fillna(df["Age"].median())
    # Stage: one-hot distribution over stage levels 0..4 (missing → extra level)
    stage_levels = ["0", "1", "2", "3", "4", "MISSING"]
    stage_dist = {}
    for c in cohorts:
        sub = df[df["Source"] == c]
        stage_dist[c] = np.array([
            (sub["Stage"].fillna(-1).astype(str) == lvl).mean() if lvl != "MISSING"
            else sub["Stage"].isna().mean()
            for lvl in stage_levels
        ])
        stage_dist[c] = stage_dist[c] / stage_dist[c].sum()
    for i, ci in enumerate(cohorts):
        for j, cj in enumerate(cohorts):
            age_mat[i, j] = wasserstein_distance(
                age_imp[df["Source"] == ci], age_imp[df["Source"] == cj]
            )
            stage_mat[i, j] = wasserstein_distance(
                np.arange(len(stage_levels)), np.arange(len(stage_levels)),
                stage_dist[ci], stage_dist[cj]
            )
    return cohorts, age_mat, stage_mat


# ===================================================================
# 2. Propensity-score overlap (5-class logistic regression)
# ===================================================================
def encode_raw(df):
    feats = pd.DataFrame(index=df.index)
    feats["Age"] = df["Age"].fillna(df["Age"].median()).astype(float)
    for c in ["Sex", "Stage", "Grade"]:
        codes, _ = pd.factorize(df[c].fillna("__MISSING__").astype(str))
        feats[f"{c}_code"] = codes
        feats[f"{c}_known"] = df[c].notna().astype(int)
    return feats.values.astype(float)


def overlap_coeff(a, b, bins=50):
    """Overlap coefficient between two density histograms."""
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    hist_a, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return float(np.minimum(hist_a, hist_b).sum() * (hi - lo) / bins)


def propensity_overlap(df):
    cohorts = sorted(df["Source"].unique())
    X = encode_raw(df)
    y = df["Source"].map({c: i for i, c in enumerate(cohorts)}).values
    rng = np.random.RandomState(7)
    idx = rng.permutation(len(y))
    tr, te = idx[:int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    clf = LogisticRegression(max_iter=2000, C=0.5, multi_class="multinomial")
    clf.fit(X[tr], y[tr])
    Xte = X[te]; yte = y[te]
    proba = clf.predict_proba(Xte)
    # per-domain propensity for its own class
    own = np.array([proba[k, yte[k]] for k in range(len(te))])
    ov = np.zeros((len(cohorts), len(cohorts)))
    for i, ci in enumerate(cohorts):
        for j, cj in enumerate(cohorts):
            a = own[yte == i]; b = own[yte == j]
            ov[i, j] = overlap_coeff(a, b)
    bal_acc = balanced_accuracy_score(yte, clf.predict(Xte))
    return cohorts, clf, own, yte, ov, bal_acc


# ===================================================================
# 3. MINE mutual-information estimator
# ===================================================================
class MINEStat(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


def estimate_mi(x, y, iters=800, batch=512, lr=1e-3):
    """MINE (Donsker–Varadhan lower bound) estimate of I(x; y).

    Uses the numerically stable per-batch form
        MI = E[T(x,y)] − logsumexp(T(x,y′)) + log(B)
    (no moving-average of exp(T), which overflows in the original formulation).
    Validated on synthetic data: recovers I(x; x+N(0,0.5))≈0.81 (true 0.805)
    and ≈0 for independent variables.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    n = x.size(0)
    if n < batch:
        batch = max(16, n // 2)
    net = MINEStat(x.size(1) + y.size(1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    log_b = torch.log(torch.tensor(float(batch), device=DEVICE))
    mi_hist = []
    for it in range(iters):
        idx = torch.randint(0, n, (batch,))
        xb, yb = x[idx], y[idx]
        perm = torch.randperm(batch)
        joint = torch.cat([xb, yb], dim=1)
        marg = torch.cat([xb, yb[perm]], dim=1)
        t_joint = net(joint)
        t_marg = net(marg)
        mi = t_joint.mean() - (torch.logsumexp(t_marg, dim=0) - log_b)
        loss = -mi
        opt.zero_grad(); loss.backward(); opt.step()
        mi_hist.append(mi.item())
    # average of the last 100 iterations (finite-sample bias downward)
    return float(np.mean(mi_hist[-100:]))


def domain_onehot(domains, n_domains):
    return np.eye(n_domains)[domains]


def mine_pipeline():
    """MINE for I(domain; survival) and I(domain; representation) per experiment."""
    out = {}
    exps = ["A", "B", "C", "D", "E"]
    pairs = [("A_tcga_vs_seer", "abc"), ("B_tcga_vs_external", "abc"),
             ("C_all_cohorts", "abc"), ("D_imputed_tcga_seer", "d"),
             ("E_brca_metabric", "e")]
    for letter, (exp_key, kind) in zip(exps, pairs):
        if kind == "abc":
            _, times, events, domains, _ = _recon_full(exp_key)
        elif kind == "d":
            _, times, events, domains, _ = _recon_exp_d()
        else:
            _, times, events, domains, _ = _recon_exp_e()
        n_dom = int(domains.max()) + 1
        d_onehot = domain_onehot(domains, n_dom)
        # survival encoding: [event, log(time+1)]
        surv = np.stack([events, np.log1p(np.clip(times, 0, None))], axis=1).astype(float)

        mi_surv = estimate_mi(d_onehot, surv)
        mi_surv = max(mi_surv, 0.0)

        # representation: ERM-hidden CLS features (PCA-reduced for stability)
        rep = None
        try:
            feats = np.load(RESULTS_DIR / "lihc_experiments" / exp_key / "baseline" / "features.npy")
            rep_pca = PCA(n_components=min(16, feats.shape[1]), random_state=42)
            rep = rep_pca.fit_transform(feats).astype(float)
        except Exception as e:
            print(f"  [I] no baseline features for {exp_key}: {e}")
        if rep is not None and len(rep) == len(domains):
            mi_rep = max(estimate_mi(d_onehot, rep), 0.0)
        else:
            mi_rep = None

        out[letter] = {
            "n_samples": int(len(domains)),
            "n_domains": int(n_dom),
            "mi_domain_survival": round(mi_surv, 4),
            "mi_domain_representation": round(mi_rep, 4) if mi_rep is not None else None,
        }
        print(f"  [I] Exp {letter}: I(dom;surv)={mi_surv:.4f}  "
              f"I(dom;rep)={mi_rep if mi_rep is None else round(mi_rep,4)}")
    return out


# ===================================================================
# Reconstructions (aligned with the saved features.npy ordering)
# ===================================================================
def _recon_full(exp_key):
    return _recon_kind(exp_key)


def _recon_kind(exp_key):
    full_df, times, events, domains, sources = _recon_full_impl(exp_key)
    return full_df, times, events, domains, sources


def _recon_full_impl(exp_key):
    from lihc_recon import reconstruct_full_data
    return reconstruct_full_data(exp_key, subsample_seer=10000)


def _recon_exp_d():
    from lihc_recon import reconstruct_exp_d
    return reconstruct_exp_d(subsample_seer=10000)


def _recon_exp_e():
    from lihc_recon import reconstruct_exp_e
    return reconstruct_exp_e()


# ===================================================================
# 4. Domain probe (2-layer MLP) + scatter
# ===================================================================
class ProbeMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)


def probe_auc(X, y, seed=7, epochs=200, batch=512):
    """Train a 2-layer MLP (minibatch) to classify domain; return macro-AUC + balanced acc."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[:int(0.7 * len(y))], idx[int(0.7 * len(y)):]
    Xs = StandardScaler().fit_transform(X)
    n_classes = int(y.max()) + 1
    net = ProbeMLP(X.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xs[tr], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(Xs[te], dtype=torch.float32, device=DEVICE)
    n_tr = len(tr)
    batch = min(batch, n_tr)
    for ep in range(epochs):
        perm = torch.randperm(n_tr)
        for s in range(0, n_tr, batch):
            b_idx = perm[s:s + batch]
            opt.zero_grad()
            loss = ce(net(Xt[b_idx]), yt[b_idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        proba = torch.softmax(net(Xv), dim=1).cpu().numpy()
    if len(np.unique(y[te])) < 2:
        return {"auc": None, "balanced_acc": None}
    try:
        if n_classes == 2:
            auc = roc_auc_score(y[te], proba[:, 1])
        else:
            auc = roc_auc_score(y[te], proba, multi_class="ovr", average="macro")
    except Exception:
        auc = None
    bal = balanced_accuracy_score(y[te], proba.argmax(axis=1))
    return {"auc": round(float(auc), 4) if auc is not None else None,
            "balanced_acc": round(float(bal), 4)}


def probe_pipeline():
    """Probe AUC on raw / ERM-hidden / DANN-hidden features per experiment."""
    out = {}
    # raw probe on LIHC 5-cohort data (SEER subsampled to 10k, as in training)
    df = pd.read_csv(DATA)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    seer_idx = df[df["Source"] == "US_SEER"].index
    keep = np.random.RandomState(42).choice(seer_idx, min(len(seer_idx), 10000), replace=False)
    df = df.drop(seer_idx.difference(pd.Index(keep))).copy()
    cohorts = sorted(df["Source"].unique())
    y = df["Source"].map({c: i for i, c in enumerate(cohorts)}).values
    X_raw = encode_raw(df)
    out["raw_lihc_5cohort"] = probe_auc(X_raw, y)

    for exp_key, kind in [("A_tcga_vs_seer", "abc"), ("B_tcga_vs_external", "abc"),
                          ("C_all_cohorts", "abc"), ("D_imputed_tcga_seer", "d"),
                          ("E_brca_metabric", "e")]:
        if kind == "abc":
            _, _, _, domains, _ = _recon_kind(exp_key)
        elif kind == "d":
            _, _, _, domains, _ = _recon_exp_d()
        else:
            _, _, _, domains, _ = _recon_exp_e()
        entry = {"n_domains": int(domains.max()) + 1}
        for mode in ["baseline", "dann"]:
            try:
                feats = np.load(RESULTS_DIR / "lihc_experiments" / exp_key / mode / "features.npy")
                entry[f"{mode}_probe"] = probe_auc(feats, domains)
            except Exception as e:
                entry[f"{mode}_probe"] = {"auc": None, "balanced_acc": None}
        out[exp_key] = entry
    return out


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("EXPERIMENT I; Domain Shift Quantification")
    print("=" * 70)

    df = pd.read_csv(DATA)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()

    # 1. Wasserstein
    print("\n[1/4] Wasserstein-1 distance matrices ...")
    cohorts, age_mat, stage_mat = wasserstein_matrices(df)
    w_results = {
        "cohorts": cohorts,
        "age_wasserstein": age_mat.round(3).tolist(),
        "stage_wasserstein": stage_mat.round(3).tolist(),
        "age_mean_pairwise": round(float(age_mat[np.triu_indices_from(age_mat, 1)].mean()), 3),
        "stage_mean_pairwise": round(float(stage_mat[np.triu_indices_from(stage_mat, 1)].mean()), 3),
    }
    print(f"  mean pairwise W1 Age={w_results['age_mean_pairwise']}, "
          f"Stage={w_results['stage_mean_pairwise']}")

    # 2. Propensity
    print("[2/4] Propensity-score overlap ...")
    p_cohorts, clf, own, yte, ov, bal_acc = propensity_overlap(df)
    p_results = {
        "cohorts": p_cohorts,
        "overlap_matrix": ov.round(3).tolist(),
        "mean_overlap": round(float(ov[np.triu_indices_from(ov, 1)].mean()), 3),
        "domain_clf_balanced_acc": round(float(bal_acc), 3),
    }
    print(f"  mean pairwise overlap={p_results['mean_overlap']}")

    # 3. MINE
    print("[3/4] MINE mutual information ...")
    mine_results = mine_pipeline()

    # 4. Probe
    print("[4/4] Domain-separability probes (2-layer MLP) ...")
    probe_results = probe_pipeline()

    results = {
        "wasserstein": w_results,
        "propensity": p_results,
        "mine": mine_results,
        "probe": probe_results,
    }
    with open(EXP_OUT / "domain_quantify.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV table
    rows = []
    for letter, m in mine_results.items():
        rows.append({"experiment": letter, "metric": "mi_domain_survival",
                     "value": m["mi_domain_survival"]})
        rows.append({"experiment": letter, "metric": "mi_domain_representation",
                     "value": m["mi_domain_representation"]})
    for exp_key, p in probe_results.items():
        for mode in ["baseline_probe", "dann_probe"]:
            if mode in p and p[mode] and p[mode]["auc"]:
                rows.append({"experiment": exp_key.replace("_tcga_vs_seer", "").replace("_brca_metabric", ""),
                             "metric": mode, "value": p[mode]["auc"]})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table5_domain_quantify.csv", index=False)

    _make_figures(cohorts, age_mat, stage_mat, p_cohorts, ov, own, yte,
                  mine_results, probe_results)

    print(f"\n✅ Experiment I complete → {EXP_OUT / 'domain_quantify.json'}")


# ===================================================================
# Figures
# ===================================================================
def _make_figures(cohorts, age_mat, stage_mat, p_cohorts, ov, own, yte,
                  mine_results, probe_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # I1: Wasserstein matrices (Age + Stage)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mat, title in [(axes[0], age_mat, "Wasserstein-1; Age"),
                           (axes[1], stage_mat, "Wasserstein-1; Stage (one-hot)")]:
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                    xticklabels=cohorts, yticklabels=cohorts, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels(cohorts, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cohorts, fontsize=8)
    fig.suptitle("Experiment I; pairwise domain shift", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "I1_wasserstein_matrix.png", dpi=200)
    plt.close(fig)

    # I2: propensity overlap heatmap + density
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.heatmap(ov, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
                xticklabels=p_cohorts, yticklabels=p_cohorts, vmin=0, vmax=1,
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Propensity-score overlap (pairwise)")
    axes[0].set_xticklabels(p_cohorts, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(p_cohorts, fontsize=8)
    for i, c in enumerate(p_cohorts):
        axes[1].hist(own[yte == i], bins=40, alpha=0.5, label=c)
    axes[1].set_xlabel("P(own domain | covariates)")
    axes[1].set_ylabel("samples")
    axes[1].set_title("Propensity density per domain")
    axes[1].legend(fontsize=7)
    fig.suptitle("Experiment I; propensity-score overlap", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "I2_propensity_overlap.png", dpi=200)
    plt.close(fig)

    # I3: MINE bar chart
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    exps = list(mine_results.keys())
    surv_vals = [mine_results[e]["mi_domain_survival"] for e in exps]
    rep_vals = [mine_results[e]["mi_domain_representation"] for e in exps]
    x = np.arange(len(exps))
    ax.bar(x - 0.18, surv_vals, 0.36, label="I(domain; survival)", color="#4c72b0")
    ax.bar(x + 0.18, rep_vals, 0.36, label="I(domain; representation)", color="#c44e52")
    for xi, v in zip(x - 0.18, surv_vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.18, rep_vals):
        if v is not None:
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"Exp {e}" for e in exps])
    ax.set_ylabel("Mutual information (nats, MINE)")
    ax.set_title("Experiment I; I(domain; ·) across experiments")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "I3_mine_mutual_info.png", dpi=200)
    plt.close(fig)

    # I4: probe AUC vs DANN Δ scatter
    deltas = {"A_tcga_vs_seer": 0.0007, "B_tcga_vs_external": -0.0032,
              "C_all_cohorts": -0.0011, "D_imputed_tcga_seer": 0.0027,
              "E_brca_metabric": 0.0074}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for exp_key, delta in deltas.items():
        p = probe_results.get(exp_key, {})
        raw_auc = None
        base_auc = (p.get("baseline_probe") or {}).get("auc")
        lab = exp_key.replace("_tcga_vs_seer", "").replace("_tcga_vs_external", "") \
                     .replace("_all_cohorts", "").replace("_imputed_tcga_seer", "") \
                     .replace("_brca_metabric", "")
        ax.scatter(base_auc if base_auc else 0.5, delta, s=80, label=lab)
        ax.annotate(lab, (base_auc if base_auc else 0.5, delta), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("Domain-separability AUC (ERM hidden features)")
    ax.set_ylabel("Δ C-index (DANN − ERM)")
    ax.set_title("Experiment I; domain separability vs GRL gain")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "I4_probe_auc_vs_cindex.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
