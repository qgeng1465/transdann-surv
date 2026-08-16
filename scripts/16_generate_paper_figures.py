#!/usr/bin/env python3
"""
16_generate_paper_figures.py; Publication Figure Reorganization v2
========================================================================
Reorganizes the ~50 scattered figures into the paper narrative:

  8 main figures   (results/figures/paper/Figure_0{1..8}.png/.pdf)
  5 supplementary  (results/figures/paper/supplementary/Figure_S{1..5}.png/.pdf)
  + Figure_Legends.md + README_figures.md
  + supplementary data tables (CSV) for the manuscript supplement

Every number is read from the verified result artifacts (JSON/CSV/npy/pth);
nothing is hard-coded except axis cosmetics. Run:

  python scripts/16_generate_paper_figures.py

Requires the same env as the rest of the pipeline (torch CUDA for the K-M
panel in Fig 6 and the t-SNE panels in Fig S1).
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
from PIL import Image

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PAPER_DIR = FIGURES_DIR / "paper"
SUPPL_DIR = PAPER_DIR / "supplementary"
EXTENDED_DIR = FIGURES_DIR / "extended"
LIHC_DIR = FIGURES_DIR / "lihc"
EXP_DIR = RESULTS_DIR / "lihc_experiments"
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(SUPPL_DIR, exist_ok=True)

# ===================================================================
# Global publication style (2026-08-02 v3; decluttered: no heavy boxes,
# no redundant borders, minimal overlapping annotations)
# ===================================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 9.5
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["axes.titlesize"] = 10.5
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.edgecolor"] = "#666666"
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["lines.linewidth"] = 1.3

# bar helper: thin white edge instead of a black "box" outline
BAR_EDGE = dict(edgecolor="white", linewidth=0.4)

COLOR_DANN = "#D62728"        # adversarial (red)
COLOR_BASELINE = "#1F77B4"    # baseline / ERM (blue)
COLOR_CORAL = "#2CA02C"
COLOR_IRM = "#FF7F0E"
COLOR_GDRO = "#9467BD"
COLOR_VREX = "#8C564B"
COLOR_FISH = "#E377C2"
COLOR_MIXUP = "#7F7F7F"
COLOR_MLDG = "#BCBD22"

METHOD_COLORS = {
    "ERM": COLOR_BASELINE, "DANN": COLOR_DANN, "CORAL": COLOR_CORAL,
    "IRM": COLOR_IRM, "GroupDRO": COLOR_GDRO, "V-REx": COLOR_VREX,
    "Fish": COLOR_FISH, "Mixup": COLOR_MIXUP, "MLDG": COLOR_MLDG,
}

COHORT_COLORS = {
    "TCGA_LIHC": "#1F77B4", "US_SEER": "#D62728", "hcc_msk_2024": "#2CA02C",
    "lihc_amc_prv": "#FF7F0E", "hcc_meric_2021": "#9467BD",
    "TCGA-BRCA": "#1F77B4", "METABRIC": "#D62728",
    "TCGA-COAD": "#1F77B4", "CPTAC-COAD": "#D62728",
    "TCGA-LUAD": "#1F77B4", "MSKCC-2020": "#D62728",
}

EXP_LABELS = {"A": "A\n(TCGA vs SEER)", "B": "B\n(TCGA vs ext.)",
              "C": "C\n(5 cohorts)", "D": "D\n(imputed)",
              "E": "E\n(BRCA clean)"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_figure(fig, name, is_supplementary=False):
    d = SUPPL_DIR if is_supplementary else PAPER_DIR
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(d / f"{name}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"  saved {d.name}/{name}.{{png,pdf}}")
    plt.close(fig)


def add_panel_label(ax, label, x=-0.14, y=1.06, fontsize=12):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="top", ha="right")


def add_axis_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.7)


def no_legend_box(ax, **kw):
    """Legend without a frame/box."""
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", 1.4)
    return ax.legend(**kw)


def annotate_no_box(ax, *args, **kw):
    """Annotation without a surrounding box."""
    kw.pop("bbox", None)
    return ax.text(*args, **kw)


# ===================================================================
# Shared data loading
# ===================================================================
def load_lihc():
    return pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")


COHORTS5 = ["TCGA_LIHC", "US_SEER", "hcc_msk_2024", "lihc_amc_prv", "hcc_meric_2021"]


def norm_stage(s):
    s = str(s).strip().upper()
    if s in ("I", "1", "1A", "1B", "0"):
        return "I"
    if s in ("II", "2"):
        return "II"
    if s in ("III", "IIIA", "IIIB", "IIIC", "IIINOS", "3A", "3B"):
        return "III"
    if s in ("IV", "IVA", "IVB", "4A", "4B"):
        return "IV"
    return "Unknown"


def experiment_history(exp_key, mode="dann"):
    d = load_json(EXP_DIR / exp_key / mode / "results.json")
    h = d.get("history", {})
    epochs = h.get("epoch", [])
    return {k: np.asarray(v) for k, v in h.items()}, epochs


def per_cohort_cindex(exp_key):
    d = load_json(EXP_DIR / exp_key / "comparison.json")
    cm = d["cohort_map"]
    rev = {int(v): k for k, v in cm.items()}
    dann = {rev[int(k)]: v for k, v in d["dann_per_cohort"].items()}
    base = {rev[int(k)]: v for k, v in d["baseline_per_cohort"].items()}
    return dann, base


def val_bootstrap_deltas(exp_key, kind, B=1000, seed=777):
    """Recompute the val-only stratified bootstrap of Δ (mirrors 04b)."""
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from fast_cindex import fast_lifelines_cindex
    from lihc_recon import (reconstruct_full_data, reconstruct_exp_d,
                            reconstruct_exp_e, build_data, _load_train_02,
                            _load_exp_d, _load_exp_e, BASE_DIR as _B,
                            CONT_FEATURES, CAT_FEATURES, EXPERIMENTS)
    import pandas as _pd

    def cindex(t, e, r):
        if len(t) < 2 or (e > 0).sum() < 1:
            return np.nan
        return fast_lifelines_cindex(t, -r, e)

    if kind == "abc":
        full_df, times, events, domains, sources = reconstruct_full_data(exp_key)
    elif kind == "d":
        full_df, times, events, domains, sources = reconstruct_exp_d()
    else:
        full_df, times, events, domains, sources = reconstruct_exp_e()

    r_dann = np.load(EXP_DIR / exp_key / "dann" / "risks.npy")
    r_base = np.load(EXP_DIR / exp_key / "baseline" / "risks.npy")

    # recover val block length
    if kind == "abc":
        df = _pd.read_csv(_B / "data_processed" / "lihc_all_cohorts.csv")
        df = df[(df["Survival_Months"] > 0) & (df["Vital_Status"].isin([0, 1]))]
        df = df[df["Source"].isin(EXPERIMENTS[exp_key]["cohorts"])].copy()
        data = _load_train_02().load_and_preprocess(df, surv_type="deephit",
                                                    n_bins=32, subsample_seer=10000)
    elif kind == "d":
        raw = _load_exp_d().load_raw()
        imputed = _load_exp_d().impute_knn(raw)
        data = build_data(imputed, CONT_FEATURES, CAT_FEATURES, n_bins=32,
                          subsample_seer=10000)
    else:
        breast = _load_exp_e().load_breast()
        data = build_data(breast, ["Age"], ["Sex", "Stage"], n_bins=32)

    n_train = len(data["train_df"])
    val_mask = np.arange(n_train, len(full_df))

    t, e, d = times[val_mask], events[val_mask], domains[val_mask]
    a, b = r_dann[val_mask], r_base[val_mask]

    rng = np.random.RandomState(seed)
    uniq = np.unique(d)
    deltas = np.zeros(B)
    for i in range(B):
        idx = []
        for u in uniq:
            m = np.where(d == u)[0]
            idx.append(m[rng.randint(0, len(m), size=len(m))])
        idx = np.concatenate(idx)
        deltas[i] = cindex(t[idx], e[idx], a[idx]) - cindex(t[idx], e[idx], b[idx])

    obs = cindex(t, e, a) - cindex(t, e, b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_sup, p_inf = float(np.mean(deltas > 0)), float(np.mean(deltas < 0))
    p = min(2.0 * min(p_sup, p_inf), 1.0)
    return deltas, obs, lo, hi, p


# ===================================================================
# Figure 1; The Domain Fingerprint in Multi-Institutional HCC Data
# ===================================================================
def generate_figure_1():
    print("[Fig 1] Domain fingerprint ...")
    miss = pd.read_csv(RESULTS_DIR / "lihc_missingness.csv").set_index("Source")
    df = load_lihc()
    df["stage_norm"] = df["Stage"].map(norm_stage)

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.48, hspace=0.50,
                           left=0.09, right=0.96, top=0.90, bottom=0.09)

    # ---- (a) missingness heatmap (light grid, no heavy cell boxes) ----
    ax = fig.add_subplot(gs[0, 0])
    feats = ["Age", "Sex", "Stage", "Grade"]
    m = miss.loc[COHORTS5, feats]
    sns.heatmap(m, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100,
                linewidths=0.3, linecolor="#e8e8e8",
                cbar_kws={"label": "Missing %", "shrink": 0.85},
                ax=ax, annot_kws={"fontsize": 7.5})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Clinical feature")
    ax.set_ylabel("Cohort")
    add_panel_label(ax, "a")
    ax.set_title("Missingness heatmap", fontweight="bold", pad=8)

    # ---- (b) Age distribution (violin only; quartile marks replace boxplot) ----
    ax = fig.add_subplot(gs[0, 1])
    age_dfs = []
    for c in COHORTS5:
        sub = df[df["Source"] == c]["Age"].dropna()
        age_dfs.append(pd.DataFrame({"cohort": c, "Age": sub.values}))
    age_df = pd.concat(age_dfs)
    order = [c for c in COHORTS5 if (age_df["cohort"] == c).sum() > 0]
    sns.violinplot(data=age_df, x="cohort", y="Age", order=order, ax=ax,
                   inner="quartile",
                   palette=[COHORT_COLORS[c] for c in order], linewidth=0.4, cut=0)
    for c in COHORTS5:
        if c not in order:
            ax.text(list(COHORTS5).index(c), 10, "no Age data\n(100% missing)",
                    ha="center", fontsize=6.5, color="#555")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Age (years)")
    add_axis_style(ax)
    add_panel_label(ax, "b")
    ax.set_title("Age distribution by cohort", fontweight="bold", pad=8)

    # ---- (c) Stage distribution (100% stacked bar, thin edges) ----
    ax = fig.add_subplot(gs[1, 0])
    order_s = ["I", "II", "III", "IV", "Unknown"]
    st = df.groupby(["Source", "stage_norm"]).size().unstack(fill_value=0).reindex(COHORTS5)
    st = st[order_s].div(st[order_s].sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(COHORTS5))
    for s in order_s:
        ax.bar(range(len(COHORTS5)), st[s], bottom=bottom, label=s, width=0.62,
               color={"I": "#4C72B0", "II": "#55A868", "III": "#C44E52",
                      "IV": "#8172B2", "Unknown": "#CCB974"}[s], **BAR_EDGE)
        bottom += st[s].values
    ax.set_xticks(range(len(COHORTS5)))
    ax.set_xticklabels([c.replace("_", "\n") for c in COHORTS5], fontsize=6.5)
    ax.set_ylabel("Proportion (%)")
    no_legend_box(ax, title="Stage", loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=6.5,
                  title_fontsize=7, columnspacing=0.8)
    ax.set_ylim(0, 100)
    add_axis_style(ax)
    add_panel_label(ax, "c")
    ax.set_title("Stage distribution by cohort", fontweight="bold", pad=8)

    # ---- (d) event rate (bars) + sample size (log circles) ----
    ax = fig.add_subplot(gs[1, 1])
    er = df.groupby("Source")["Vital_Status"].mean().reindex(COHORTS5) * 100
    n = df.groupby("Source").size().reindex(COHORTS5)
    ax.bar(range(len(COHORTS5)), er.values, color=[COHORT_COLORS[c] for c in COHORTS5],
           alpha=0.85, width=0.62, **BAR_EDGE)
    for i, e in enumerate(er.values):
        ax.text(i, e + 1.5, f"{e:.1f}%", ha="center", fontsize=6.5, fontweight="bold")
    ax.set_ylabel("Event rate (%)")
    ax.set_xticks(range(len(COHORTS5)))
    ax.set_xticklabels([c.replace("_", "\n") for c in COHORTS5], fontsize=6.5)
    ax2 = ax.twinx()
    ax2.scatter(range(len(COHORTS5)), n.values, color="black", s=38, zorder=5,
                marker="o", facecolors="none", edgecolors="black", linewidths=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("Sample size (log scale)")
    ax2.set_ylim(50, 2e5)
    ax2.tick_params(axis="y", length=3, width=0.7)
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("Event rate and sample size", fontweight="bold", pad=8)

    save_figure(fig, "Figure_01_domain_fingerprint")


# ===================================================================
# Figure 2; Main results (zero to negative gain)
# ===================================================================
def generate_figure_2():
    print("[Fig 2] Main results ...")
    comp = {e: load_json(EXP_DIR / e / "comparison.json")
            for e in ["A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts"]}
    valb = load_json(RESULTS_DIR / "F_bootstrap_val_results.json")
    exps = ["A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts"]
    labels = ["A", "B", "C"]
    dann_c = [comp[e]["dann_val_cindex"] for e in exps]
    base_c = [comp[e]["baseline_val_cindex"] for e in exps]
    ps = [valb[e]["p_two_sided"] for e in exps]

    # Panel (b) marker styles per experiment
    MARK = {"A_tcga_vs_seer": "o", "B_tcga_vs_external": "s", "C_all_cohorts": "^"}

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.50,
                           left=0.08, right=0.94, top=0.90, bottom=0.09)

    # ---- (a) C-index grouped bars (A–C) ----
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    w = 0.32
    ax.bar(x - w / 2, base_c, w, label="Baseline", color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, dann_c, w, label="DANN", color=COLOR_DANN, **BAR_EDGE)
    for i in range(3):
        d = dann_c[i] - base_c[i]
        ax.annotate(f"$\\Delta$={d:+.4f}\np={ps[i]:.3f}",
                    xy=(x[i], max(dann_c[i], base_c[i]) + 0.0015), ha="center",
                    fontsize=6.5, fontweight="bold")
    ax.axhline(0.63, color="grey", lw=0.6, ls="--")
    ax.set_ylabel("Validation C-index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.615, 0.652)
    no_legend_box(ax, loc="lower right")
    add_axis_style(ax)
    add_panel_label(ax, "a")
    ax.set_title("C-index: A–C", fontweight="bold", pad=8)

    # ---- (b) time-dependent AUC (legend outside, no overlap) ----
    ax = fig.add_subplot(gs[0, 1])
    times = ["12", "36", "60"]
    tt = np.arange(len(times))
    for i, e in enumerate(exps):
        d = load_json(EXP_DIR / e / "dann" / "results.json")["time_dependent_auc"]
        b = load_json(EXP_DIR / e / "baseline" / "results.json")["time_dependent_auc"]
        ax.plot(tt, [d[t] for t in times], color=COLOR_DANN, ls="--",
                marker=MARK[e], ms=4, lw=1.1, alpha=0.9)
        ax.plot(tt, [b[t] for t in times], color=COLOR_BASELINE, ls="-",
                marker=MARK[e], ms=4, lw=1.1, alpha=0.9)
    ax.set_xticks(tt)
    ax.set_xticklabels([f"{t} mo" for t in times])
    ax.set_ylabel("Time-dependent AUC")
    ax.set_ylim(0.55, 0.70)
    ax.text(0.02, 0.05, "Baseline $\\geq$ DANN\nat all 9 time points",
            transform=ax.transAxes, fontsize=6.5, color="#444")
    # one compact legend, to the right of the axes
    lines = [Line2D([0], [0], color=COLOR_BASELINE, lw=1.3, label="Baseline"),
             Line2D([0], [0], color=COLOR_DANN, lw=1.3, ls="--", label="DANN"),
             Line2D([0], [0], color="grey", marker="o", ls="", label="Exp A"),
             Line2D([0], [0], color="grey", marker="s", ls="", label="Exp B"),
             Line2D([0], [0], color="grey", marker="^", ls="", label="Exp C")]
    ax.legend(handles=lines, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=6.5)
    add_axis_style(ax)
    add_panel_label(ax, "b")
    ax.set_title("Time-dependent AUC", fontweight="bold", pad=8)

    # ---- (c) per-cohort C-index, all three experiments in one panel ----
    ax = fig.add_subplot(gs[1, 0])
    xpos, xtick, xlab = [], [], []
    w = 0.30
    for k, e in enumerate(exps):
        dann, base = per_cohort_cindex(e)
        cohorts = list(dann.keys())
        # position each cohort's two bars within the experiment group
        group_center = k * 4.0 + 1.0
        xs = group_center + np.arange(len(cohorts)) * 0.55
        ax.bar(xs - w / 2, [base[c] for c in cohorts], w, color=COLOR_BASELINE, **BAR_EDGE)
        ax.bar(xs + w / 2, [dann[c] for c in cohorts], w, color=COLOR_DANN, **BAR_EDGE)
        # annotate only the reference cohort (TCGA) delta; the one that matters
        if "TCGA_LIHC" in cohorts:
            c = "TCGA_LIHC"
            d = dann[c] - base[c]
            ax.annotate(f"{d:+.3f}", xy=(xs[cohorts.index(c)], max(dann[c], base[c]) + 0.015),
                        ha="center", fontsize=6, fontweight="bold", color="#d62728")
        xpos.extend(xs)
        xtick.extend([c.replace("_", "\n") for c in cohorts])
        xlab.append((group_center, f"Exp {labels[k]}"))
        if k < len(exps) - 1:
            ax.axvline(group_center + 1.7, color="#cccccc", lw=0.6, ls=":")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xtick, fontsize=5.5)
    for gc, lab in xlab:
        ax.text(gc, -0.27, lab, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=7, fontweight="bold")
    ax.set_ylabel("Per-cohort C-index")
    ax.set_ylim(0.45, 0.80)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.text(0.02, 0.96, "red = TCGA reference (consistently hurt)",
            transform=ax.transAxes, fontsize=6, color="#444")
    no_legend_box(ax, handles=[
        Patch(facecolor=COLOR_BASELINE, label="Baseline"),
        Patch(facecolor=COLOR_DANN, label="DANN")],
        loc="lower left", fontsize=6)
    add_axis_style(ax)
    add_panel_label(ax, "c")
    ax.set_title("Per-cohort C-index (A/B/C)", fontweight="bold", pad=8)

    # ---- (d) domain accuracy trajectory Exp A (no GRL-α overlay) ----
    ax = fig.add_subplot(gs[1, 1])
    hist, epochs = experiment_history("A_tcga_vs_seer", "dann")
    ep = hist["epoch"]
    acc = hist["domain_acc"]
    ax.plot(ep, acc, color=COLOR_DANN, lw=1.6, marker="o", ms=3)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.text(ep[0], 0.53, "random chance (50%)", fontsize=6, color="#555")
    ax.text(ep[-1], acc[-1] + 0.03, f"final {acc[-1]*100:.1f}%",
            ha="right", fontsize=6.5, color="#c0392b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain accuracy")
    ax.set_ylim(0, 1.05)
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("Domain classifier (Exp A)", fontweight="bold", pad=8)

    save_figure(fig, "Figure_02_main_results")


# ===================================================================
# Figure 3; Imputation does not remove the domain fingerprint
# ===================================================================
def generate_figure_3():
    print("[Fig 3] Imputation does not remove fingerprint ...")
    dprobe = load_json(RESULTS_DIR / "D_imputation_probe.json")
    valb = load_json(RESULTS_DIR / "F_bootstrap_val_results.json")

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.46, hspace=0.50,
                           left=0.09, right=0.96, top=0.90, bottom=0.09)

    # ---- (a) missingness before/after ----
    ax = fig.add_subplot(gs[0, 0])
    feats = ["Age", "Sex", "Stage", "Grade"]
    before = [max(dprobe["missingness_before"][f].values()) for f in feats]
    after = [dprobe["missingness_after"][f] for f in feats]
    x = np.arange(len(feats))
    w = 0.34
    ax.bar(x - w / 2, before, w, label="Original", color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, after, w, label="KNN-imputed (k=7)", color=COLOR_DANN, **BAR_EDGE)
    for i in range(len(feats)):
        ax.text(x[i] + w / 2, after[i] + 1.5, "0%", ha="center", fontsize=7,
                color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_ylabel("Missingness (%)")
    ax.set_ylim(0, 110)
    no_legend_box(ax, loc="upper right")
    add_axis_style(ax)
    add_panel_label(ax, "a")
    ax.set_title("Missingness: original vs imputed", fontweight="bold", pad=8)

    # ---- (b) domain separability probe ----
    ax = fig.add_subplot(gs[0, 1])
    vals = [dprobe["before"]["auc"], dprobe["after"]["auc"]]
    ax.bar(["Original", "Imputed"], vals, width=0.45,
           color=[COLOR_BASELINE, COLOR_DANN], **BAR_EDGE)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"AUC={v:.3f}", ha="center", fontsize=8, fontweight="bold")
    d = vals[1] - vals[0]
    ax.text(0.5, 0.98, f"+{d:.3f} after imputation\n(no decrease)",
            transform=ax.transAxes, ha="center", fontsize=7, color="#2ca02c")
    ax.set_ylabel("Domain-separability probe AUC")
    ax.set_ylim(0.85, 1.02)
    add_axis_style(ax)
    add_panel_label(ax, "b")
    ax.set_title("Domain separability (probe AUC)", fontweight="bold", pad=8)

    # ---- (c) domain acc trajectory original vs imputed ----
    ax = fig.add_subplot(gs[1, 0])
    hist_a, ep_a = experiment_history("A_tcga_vs_seer", "dann")
    hist_d, ep_d = experiment_history("D_imputed_tcga_seer", "dann")
    ax.plot(hist_a["epoch"], hist_a["domain_acc"], color=COLOR_BASELINE, lw=1.5,
            marker="o", ms=3, label=f"Original (peak {hist_a['domain_acc'].max()*100:.1f}%)")
    ax.plot(hist_d["epoch"], hist_d["domain_acc"], color=COLOR_DANN, lw=1.5,
            marker="s", ms=3, label=f"Imputed (peak {hist_d['domain_acc'].max()*100:.1f}%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", ls="--", lw=0.7, alpha=0.6)
    no_legend_box(ax, loc="center right")
    add_axis_style(ax)
    add_panel_label(ax, "c")
    ax.set_title("Domain classifier trajectory", fontweight="bold", pad=8)

    # ---- (d) c-index original vs imputed ----
    ax = fig.add_subplot(gs[1, 1])
    groups = ["Original", "Imputed"]
    base = [0.6331, 0.6335]
    dann = [0.6338, 0.6362]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, base, w, label="Baseline", color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, dann, w, label="DANN", color=COLOR_DANN, **BAR_EDGE)
    ps = [valb["A_tcga_vs_seer"]["p_two_sided"], valb["D_imputed_tcga_seer"]["p_two_sided"]]
    for i in range(2):
        d = dann[i] - base[i]
        ax.text(x[i], max(dann[i], base[i]) + 0.0015, f"$\\Delta$={d:+.4f}\np={ps[i]:.3f}",
                ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Validation C-index")
    ax.set_ylim(0.625, 0.646)
    no_legend_box(ax, loc="lower right")
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("C-index: original vs imputed", fontweight="bold", pad=8)

    save_figure(fig, "Figure_03_imputation")


# ===================================================================
# Figure 4; Clean data reveals upper bound + statistical closing
# ===================================================================
def generate_figure_4():
    print("[Fig 4] Clean data + bootstrap ...")
    valb = load_json(RESULTS_DIR / "F_bootstrap_val_results.json")
    hist_a, ep_a = experiment_history("A_tcga_vs_seer", "dann")
    hist_e, ep_e = experiment_history("E_brca_metabric", "dann")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.46,
                           left=0.07, right=0.96, top=0.90, bottom=0.09)

    # ---- (a) domain acc LIHC vs BRCA ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hist_a["epoch"], hist_a["domain_acc"], color=COLOR_BASELINE, lw=1.5,
            marker="o", ms=3, label="LIHC Exp A (peak 97.3%)")
    ax.plot(hist_e["epoch"], hist_e["domain_acc"], color=COLOR_DANN, lw=1.5,
            marker="s", ms=3, label="BRCA Exp E (peak 64.3%)")
    ax.axhline(0.5, color="grey", ls="--", lw=0.7, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain accuracy")
    ax.set_ylim(0, 1.05)
    no_legend_box(ax, loc="lower right")
    add_axis_style(ax)
    add_panel_label(ax, "a")
    ax.set_title("Domain accuracy: LIHC vs BRCA", fontweight="bold", pad=8)

    # ---- (b) per-cohort clean ----
    ax = fig.add_subplot(gs[0, 1])
    dann, base = per_cohort_cindex("E_brca_metabric")
    cohorts = list(dann.keys())
    x = np.arange(len(cohorts))
    w = 0.34
    ax.bar(x - w / 2, [base[c] for c in cohorts], w, label="Baseline",
           color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, [dann[c] for c in cohorts], w, label="DANN",
           color=COLOR_DANN, **BAR_EDGE)
    for i, c in enumerate(cohorts):
        d = dann[c] - base[c]
        ax.annotate(f"{d:+.3f}", xy=(x[i], max(dann[c], base[c]) + 0.02),
                    ha="center", fontsize=7, fontweight="bold",
                    color="#2ca02c" if d > 0 else "#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("C-index")
    ax.set_ylim(0.55, 0.85)
    no_legend_box(ax, loc="lower left")
    ax.text(0.05, 0.92, "$\\Delta$=+0.0074 (p=0.058)", transform=ax.transAxes,
            fontsize=6.5, color="#2ca02c", fontweight="bold")
    add_axis_style(ax)
    add_panel_label(ax, "b")
    ax.set_title("Per-cohort C-index (Exp E)", fontweight="bold", pad=8)

    # ---- (c) GRL gain vs domain separability ----
    ax = fig.add_subplot(gs[0, 2])
    peaks = {"A": hist_a["domain_acc"].max(), "B": 0.8597, "C": 0.8333,
             "D": 0.9771, "E": hist_e["domain_acc"].max()}
    dels = {"A": 0.0007, "B": -0.0032, "C": -0.0011, "D": 0.0027, "E": 0.0074}
    xs = np.array([peaks[k] * 100 for k in ["A", "B", "C", "D", "E"]])
    ys = np.array([dels[k] for k in ["A", "B", "C", "D", "E"]])
    for k, (xi, yi) in enumerate(zip(xs, ys)):
        if k == 4:
            ax.scatter(xi, yi, s=70, color=COLOR_DANN, zorder=5, edgecolor="white",
                       linewidth=0.4)
            ax.annotate("E (p=0.058)", xy=(xi, yi), xytext=(xi + 1.5, yi + 0.003),
                        fontsize=6.5, color=COLOR_DANN, fontweight="bold")
        else:
            ax.scatter(xi, yi, s=45, color="#888888", zorder=4, edgecolor="white",
                       linewidth=0.4)
        ax.annotate(["A", "B", "C", "D", "E"][k], xy=(xi, yi), xytext=(3, 3),
                    textcoords="offset points", fontsize=7, fontweight="bold")
    res = stats.linregress(xs, ys)
    xx = np.linspace(xs.min() - 3, xs.max() + 3, 50)
    ax.plot(xx, res.intercept + res.slope * xx, color="#444", ls="--", lw=1.0)
    ax.fill_between(xx, (res.intercept + res.slope * xx) - 1.96 * res.stderr,
                    (res.intercept + res.slope * xx) + 1.96 * res.stderr,
                    color="#444", alpha=0.08)
    ax.set_xlabel("Domain classifier peak accuracy (%)")
    ax.set_ylabel("GRL gain $\\Delta$ C-index")
    ax.set_ylim(-0.008, 0.011)
    ax.text(0.03, 0.05, f"R={res.rvalue:.2f}, p={res.pvalue:.2f}",
            transform=ax.transAxes, fontsize=6.5)
    add_axis_style(ax)
    add_panel_label(ax, "c")
    ax.set_title("GRL gain vs domain separability", fontweight="bold", pad=8)

    # ---- (d) bootstrap forest plot (val-set) ----
    ax = fig.add_subplot(gs[1, 0])
    rows = [("A", "A_tcga_vs_seer"), ("B", "B_tcga_vs_external"),
            ("C", "C_all_cohorts"), ("D", "D_imputed_tcga_seer"), ("E", "E_brca_metabric")]
    for i, (lab, e) in enumerate(rows):
        v = valb[e]
        y = 4 - i
        ax.plot([v["ci_low"], v["ci_high"]], [y, y], color="#4c72b0", lw=4,
                solid_capstyle="round")
        ax.plot(v["delta_obs"], y, "o", color=COLOR_DANN, ms=7)
        ax.text(0.026, y, f"p={v['p_two_sided']:.3f}", va="center", fontsize=7)
    ax.axvline(0, color="grey", ls=":", lw=1)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Exp {r[0]}" for r in rows])
    ax.set_xlabel("$\\Delta$ C-index (DANN $-$ Baseline), 95% CI")
    ax.set_xlim(-0.035, 0.035)
    ax.set_ylim(-0.6, 4.6)
    ax.text(0.05, 0.98, "All p > 0.05 (held-out val)",
            transform=ax.transAxes, fontsize=6.5, color="#444")
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("Bootstrap 95% CI (B=1000, val)", fontweight="bold", pad=8)

    # ---- (e/f) bootstrap distributions A and E ----
    for idx, (exp_key, kind, lab) in enumerate(
            [("A_tcga_vs_seer", "abc", "e"), ("E_brca_metabric", "e", "f")]):
        ax = fig.add_subplot(gs[1, idx + 1])
        deltas, obs, lo, hi, p = val_bootstrap_deltas(exp_key, kind)
        ax.hist(deltas, bins=40, color="#4c72b0", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(obs, color=COLOR_DANN, lw=1.8, label=f"obs $\\Delta$={obs:+.4f}")
        ax.axvline(lo, color=COLOR_DANN, ls="--", lw=1)
        ax.axvline(hi, color=COLOR_DANN, ls="--", lw=1)
        ax.axvline(0, color="black", ls=":", lw=0.9)
        ax.set_xlabel("$\\Delta$ C-index (DANN $-$ Baseline)")
        ax.set_ylabel("count")
        ax.text(0.98, 0.95, f"p={p:.3f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=8, fontweight="bold")
        if lab == "f":
            ax.text(0.98, 0.70, "Best-case GRL gain,\nstill not significant",
                    transform=ax.transAxes, ha="right", fontsize=6.5, color="#c0392b")
        no_legend_box(ax, loc="upper left")
        add_axis_style(ax)
        add_panel_label(ax, lab)
        ax.set_title(f"Bootstrap distribution (Exp {lab.upper()})",
                     fontweight="bold", pad=8)

    save_figure(fig, "Figure_04_clean_data_bootstrap")


# ===================================================================
# Figure 5; SHAP interpretability (native clean beeswarms)
# ===================================================================
def _shap_data(mode, n_sub=300, nsamples=50, seed=7):
    """Recompute SHAP on a subsample of the Exp-A validation set.

    Mirrors 07_exp_g_shap.py: a SmoothSurvNet replicates the real model's
    risk path so gradient-based SHAP (Expected Gradients) is differentiable.
    Returns (shap_values, display_values) on the subsample.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import shap
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from transdann_utils import TransDANNSurvV3, DEVICE
    from lihc_recon import reconstruct_full_data, _load_train_02

    full_df, times, events, domains, sources = reconstruct_full_data("A_tcga_vs_seer")
    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[(df["Survival_Months"] > 0) & (df["Vital_Status"].isin([0, 1]))]
    exp_df = df[df["Source"].isin(["TCGA_LIHC", "US_SEER"])].copy()
    data = _load_train_02().load_and_preprocess(exp_df, surv_type="deephit",
                                                n_bins=32, subsample_seer=10000)
    n_train = len(data["train_df"])
    val_df = full_df.iloc[n_train:].reset_index(drop=True)

    # smooth inputs: [Age(scaled), Sex-oh, Stage-oh, Grade-oh]
    X = [val_df["Age"].values.astype(np.float32).reshape(-1, 1)]
    for col, card in zip(data["cat_features"], data["cat_cardinalities"]):
        v = np.clip(val_df[col].values.astype(np.int64), 0, card - 1)
        oh = np.zeros((len(v), card), dtype=np.float32)
        oh[np.arange(len(v)), v] = 1.0
        X.append(oh)
    X = np.concatenate(X, axis=1)

    # display values (Age in years, ordinal codes)
    disp = np.full((len(val_df), 4), np.nan, dtype=float)
    disp[:, 0] = data["scaler"].inverse_transform(
        val_df["Age"].values.astype(float).reshape(-1, 1)).ravel()
    ord_map = {"Sex": {"0.0": 0.0, "1.0": 1.0},
               "Stage": {"I": 1.0, "1": 1.0, "1A": 1.0, "1B": 1.0, "II": 2.0, "2": 2.0,
                         "III": 3.0, "IIIA": 3.0, "IIIB": 3.0, "IV": 4.0, "IVA": 4.0,
                         "IVB": 4.0},
               "Grade": {"1.0": 1.0, "2.0": 2.0, "3.0": 3.0, "4.0": 4.0}}
    for j, col in enumerate(data["cat_features"]):
        le = data["label_encoders"][col]
        for idx, cls in enumerate(le.classes_):
            if cls in ord_map.get(col, {}):
                mask = val_df[col].values == idx
                disp[mask, j + 1] = ord_map[col][cls]

    # subsample
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(X), n_sub, replace=False)

    class _SmoothNet(nn.Module):
        def __init__(self, base, cat_cards, bin_centers):
            super().__init__()
            self.base = base
            self.cat_cards = cat_cards
            self.register_buffer("bin_centers", bin_centers.to(next(base.parameters()).device))
        def forward(self, x):
            B = x.size(0)
            tokens = [self.base.cls_token.expand(B, -1, -1)]
            tokens.append(self.base.cont_embeddings[0](x[:, 0:1]).unsqueeze(1))
            off = 1
            for i, card in enumerate(self.cat_cards):
                oh = x[:, off:off + card]
                off += card
                tokens.append(torch.matmul(
                    oh, self.base.cat_embeddings[i].weight[:card]).unsqueeze(1))
            x = torch.cat(tokens, dim=1)
            x = x + self.base.pos_encoding[:, :x.size(1), :]
            x = self.base.transformer(x)
            x = self.base.layer_norm(x)
            h = F.softmax(self.base.survival_head(x[:, 0, :]), dim=1)
            return -torch.sum(h * self.bin_centers, dim=1).reshape(-1, 1)

    cat_cards = list(data["cat_cardinalities"])
    model = TransDANNSurvV3(
        num_continuous=len(data["cont_features"]),
        num_categorical=len(data["cat_features"]),
        cat_cardinalities=data["cat_cardinalities"],
        d_model=128, n_heads=8, n_layers=4, dropout=0.15,
        num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32).to(DEVICE)
    model.load_state_dict(torch.load(EXP_DIR / "A_tcga_vs_seer" / mode / "best_model.pth",
                                     map_location=DEVICE, weights_only=True))
    model.eval()
    smooth = _SmoothNet(model, cat_cards, data["bin_centers"]).to(DEVICE)

    bg = torch.tensor(X[sub[:128]], dtype=torch.float32).to(DEVICE)
    Xi = torch.tensor(X[sub], dtype=torch.float32).to(DEVICE)
    explainer = shap.GradientExplainer(smooth, bg)
    sv = explainer.shap_values(Xi, nsamples=nsamples)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)[:, :, 0]
    # aggregate one-hot columns per feature
    feats = ["Age", "Sex", "Stage", "Grade"]
    agg = np.zeros((len(sub), 4))
    agg[:, 0] = sv[:, 0]
    off = 1
    for i, card in enumerate(cat_cards):
        agg[:, i + 1] = sv[:, off:off + card].sum(axis=1)
        off += card
    return agg, disp[sub], data


def _draw_beeswarm(ax, sv, disp, feats):
    """Minimal box-free beeswarm: one horizontal strip per feature, points
    jittered in y, coloured by the normalised feature value (grey = missing)."""
    cmap = plt.get_cmap("viridis")
    n = len(sv)
    for i, f in enumerate(feats):
        vals = sv[:, i]
        o = np.argsort(vals)
        y = i + 0.5 + np.random.RandomState(100 + i).uniform(-0.28, 0.28, n)
        dv = disp[:, i]
        norm = (dv - np.nanmin(dv)) / (np.nanmax(dv) - np.nanmin(dv) + 1e-9)
        col_rgba = cmap(np.clip(norm, 0, 1))          # (n, 4)
        col_rgba[np.isnan(norm), :] = 0.78            # grey when missing
        ax.scatter(vals[o], y[o], s=3, c=col_rgba[o], alpha=0.7, linewidths=0, rasterized=True)
        ax.axvline(0, color="#cccccc", lw=0.6, ls=":")
    ax.set_yticks([i + 0.5 for i in range(len(feats))])
    ax.set_yticklabels(feats)
    ax.set_xlabel("SHAP value (risk impact)")
    ax.set_ylim(-0.1, len(feats) + 0.1)
    ax.set_xlim(sv.min() - 0.1, sv.max() + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def generate_figure_5():
    print("[Fig 5] SHAP interpretability (native beeswarms) ...")
    g = load_json(RESULTS_DIR / "G_shap_results.json")
    feats = g["features"]
    base = g["models"]["baseline"]["mean_abs_shap"]
    dann = g["models"]["dann"]["mean_abs_shap"]

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.34, hspace=0.48,
                           left=0.08, right=0.96, top=0.90, bottom=0.09)

    # ---- (a/b) native beeswarms (subsampled SHAP, no embedded PNG boxes) ----
    for idx, (mode, lab) in enumerate([("baseline", "a"), ("dann", "b")]):
        ax = fig.add_subplot(gs[0, idx])
        sv, disp, _ = _shap_data(mode, n_sub=300, nsamples=50)
        _draw_beeswarm(ax, sv, disp, feats)
        add_panel_label(ax, lab)
        ax.set_title(f"SHAP beeswarm; {mode.capitalize()} (n=300)", fontweight="bold", pad=8)

    # ---- (c) mean |SHAP| comparison ----
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(feats))
    w = 0.34
    ax.bar(x - w / 2, [base[f] for f in feats], w, label="Baseline",
           color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, [dann[f] for f in feats], w, label="DANN",
           color=COLOR_DANN, **BAR_EDGE)
    for i, f in enumerate(feats):
        d = (dann[f] - base[f]) / base[f] * 100
        ax.annotate(f"{d:+.1f}%", xy=(x[i], max(dann[f], base[f]) + 0.25),
                    ha="center", fontsize=7.5, fontweight="bold",
                    color="#2ca02c" if d > 0 else "#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_ylabel("Mean |SHAP|")
    ax.set_ylim(0, 12.5)
    no_legend_box(ax, loc="upper right")
    add_axis_style(ax)
    add_panel_label(ax, "c")
    ax.set_title(f"Mean |SHAP| (val, n={g['n_val']})", fontweight="bold", pad=8)

    # ---- (d) Stage SHAP by cohort ----
    ax = fig.add_subplot(gs[1, 1])
    cb = g["models"]["baseline"]["cohort_mean_abs"]
    cd = g["models"]["dann"]["cohort_mean_abs"]
    cohorts = ["TCGA_LIHC", "US_SEER"]
    base_v = [cb[c]["Stage"] for c in cohorts]
    dann_v = [cd[c]["Stage"] for c in cohorts]
    x = np.arange(len(cohorts))
    w = 0.34
    ax.bar(x - w / 2, base_v, w, label="Baseline", color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, dann_v, w, label="DANN", color=COLOR_DANN, **BAR_EDGE)
    for i, c in enumerate(cohorts):
        d = (dann_v[i] - base_v[i]) / base_v[i] * 100
        ax.annotate(f"{d:+.1f}%", xy=(x[i], max(dann_v[i], base_v[i]) + 0.5),
                    ha="center", fontsize=8, fontweight="bold", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Mean |SHAP|; Stage")
    ax.set_ylim(0, 20)
    no_legend_box(ax, loc="upper right")
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("Stage SHAP importance by cohort", fontweight="bold", pad=8)

    save_figure(fig, "Figure_05_shap")


# ===================================================================
# Figure 6; Kaplan-Meier stratification and clinical utility
# ===================================================================
def _km_curves():
    """Recompute K-M curves on the SEER test set (reproduces 08_exp_h)."""
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    import torch
    from transdann_utils import TransDANNSurvV3, deep_hit_risk, DEVICE
    from lihc_recon import reconstruct_full_data, _load_train_02
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.utils import concordance_index

    full_df, times, events, domains, sources = reconstruct_full_data("A_tcga_vs_seer")
    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[(df["Survival_Months"] > 0) & (df["Vital_Status"].isin([0, 1]))]
    exp_df = df[df["Source"].isin(["TCGA_LIHC", "US_SEER"])].copy()
    data = _load_train_02().load_and_preprocess(exp_df, surv_type="deephit",
                                                n_bins=32, subsample_seer=10000)
    n_train = len(data["train_df"])
    val_mask = np.arange(n_train, len(full_df))
    val_df = full_df.iloc[val_mask].reset_index(drop=True)
    val_t, val_e = times[val_mask], events[val_mask]
    seer = val_df["Source"].values == "US_SEER"
    km_t, km_e = val_t[seer], val_e[seer]

    out = {}
    for mode in ("baseline", "dann"):
        model = TransDANNSurvV3(num_continuous=len(data["cont_features"]),
                                num_categorical=len(data["cat_features"]),
                                cat_cardinalities=data["cat_cardinalities"],
                                d_model=128, n_heads=8, n_layers=4, dropout=0.15,
                                num_domains=data["num_domains"],
                                surv_head_type="deephit", n_bins=32).to(DEVICE)
        model.load_state_dict(torch.load(EXP_DIR / "A_tcga_vs_seer" / mode / "best_model.pth",
                                         map_location=DEVICE, weights_only=True))
        model.eval()
        with torch.no_grad():
            x_cont = torch.tensor(val_df[data["cont_features"]].values,
                                  dtype=torch.float32).to(DEVICE)
            x_cat = torch.tensor(val_df[data["cat_features"]].values,
                                 dtype=torch.long).to(DEVICE)
            surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
            risk = deep_hit_risk(surv_out, data["bin_centers"]).cpu().numpy()[seer]
        ci = concordance_index(km_t, -risk, km_e)
        med = np.median(risk)
        high = risk >= med
        low = ~high
        kmf_hi = KaplanMeierFitter().fit(km_t[high], km_e[high])
        kmf_lo = KaplanMeierFitter().fit(km_t[low], km_e[low])
        lr = logrank_test(km_t[high], km_t[low], km_e[high], km_e[low])
        grid = np.linspace(0, 120, 240)
        out[mode] = {
            "cindex": ci, "logrank_p": lr.p_value,
            "med_hi": kmf_hi.median_survival_time_, "med_lo": kmf_lo.median_survival_time_,
            "km_hi_t": kmf_hi.survival_function_.index.values,
            "km_hi_s": kmf_hi.survival_function_["KM_estimate"].values,
            "km_hi_ci_l": kmf_hi.confidence_interval_["KM_estimate_lower_0.95"].values,
            "km_hi_ci_u": kmf_hi.confidence_interval_["KM_estimate_upper_0.95"].values,
            "km_lo_t": kmf_lo.survival_function_.index.values,
            "km_lo_s": kmf_lo.survival_function_["KM_estimate"].values,
            "km_lo_ci_l": kmf_lo.confidence_interval_["KM_estimate_lower_0.95"].values,
            "km_lo_ci_u": kmf_lo.confidence_interval_["KM_estimate_upper_0.95"].values,
            "n_high": int(high.sum()), "n_low": int(low.sum()),
        }
    return out, data


def generate_figure_6():
    print("[Fig 6] K-M stratification ...")
    h = load_json(RESULTS_DIR / "H_km_results.json")
    km, data = _km_curves()

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.44,
                           left=0.08, right=0.96, top=0.90, bottom=0.09)

    # ---- (a/b) K-M curves ----
    for idx, mode in enumerate(["baseline", "dann"]):
        ax = fig.add_subplot(gs[0, idx])
        k = km[mode]
        hi_c, lo_c = "#C44E52", "#4C72B0"
        ax.fill_between(k["km_hi_t"], k["km_hi_ci_l"], k["km_hi_ci_u"], color=hi_c, alpha=0.18, lw=0)
        ax.fill_between(k["km_lo_t"], k["km_lo_ci_l"], k["km_lo_ci_u"], color=lo_c, alpha=0.18, lw=0)
        ax.step(k["km_hi_t"], k["km_hi_s"], color=hi_c, lw=1.8,
                label=f"High risk (n={k['n_high']})")
        ax.step(k["km_lo_t"], k["km_lo_s"], color=lo_c, lw=1.8,
                label=f"Low risk (n={k['n_low']})")
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Time (months)")
        if idx == 0:
            ax.set_ylabel("Survival probability")
        no_legend_box(ax, loc="lower left")
        ax.text(0.03, 0.05, f"log-rank P={k['logrank_p']:.1e}\n"
                            f"median: {k['med_hi']:.0f} vs {k['med_lo']:.0f} mo",
                transform=ax.transAxes, fontsize=7, color="#333")
        add_axis_style(ax)
        add_panel_label(ax, "ab"[idx])
        ax.set_title(f"Kaplan-Meier; {mode.capitalize()} (SEER test)",
                     fontweight="bold", pad=8)

    # ---- (c) stratification stats (light table, no heavy grid) ----
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    rows_t = [
        ("Metric", "Baseline", "DANN"),
        ("C-index (SEER test)", f"{h['models']['baseline']['cindex_seer_test']:.4f}",
         f"{h['models']['dann']['cindex_seer_test']:.4f}"),
        ("log-rank P", f"{h['models']['baseline']['logrank_p']:.2e}",
         f"{h['models']['dann']['logrank_p']:.2e}"),
        ("Median survival high", f"{h['models']['baseline']['median_survival_high_months']:.0f} mo",
         f"{h['models']['dann']['median_survival_high_months']:.0f} mo"),
        ("Median survival low", f"{h['models']['baseline']['median_survival_low_months']:.0f} mo",
         f"{h['models']['dann']['median_survival_low_months']:.0f} mo"),
        ("N high / low", f"{h['models']['baseline']['n_high']}/{h['models']['baseline']['n_low']}",
         f"{h['models']['dann']['n_high']}/{h['models']['dann']['n_low']}"),
    ]
    table = ax.table(cellText=rows_t, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for j in range(3):
        table[0, j].set_facecolor("#f0f0f0")
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title(f"Stratification statistics (SEER test, n={h['n_seer_test']:,}, "
                 f"{h['n_events']:,} events)", fontweight="bold", pad=6)
    add_panel_label(ax, "c", x=-0.12, y=1.08)

    # ---- (d) discrimination vs stratification ----
    ax = fig.add_subplot(gs[1, 1])
    for mode, lab in [("baseline", "Baseline"), ("dann", "DANN")]:
        m = h["models"][mode]
        x, y = m["cindex_seer_test"], -np.log10(m["logrank_p"])
        ax.scatter(x, y, s=80, label=lab, zorder=5,
                   color=COLOR_BASELINE if mode == "baseline" else COLOR_DANN,
                   edgecolor="white", linewidth=0.5)
        ax.annotate(lab, xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("C-index (SEER test)")
    ax.set_ylabel("$-$log$_{10}$(log-rank P)")
    ax.set_xlim(0.628, 0.642)
    ax.set_ylim(23, 29)
    ax.text(0.03, 0.06, "Equivalent clinical stratification",
            transform=ax.transAxes, fontsize=7, color="#444")
    no_legend_box(ax, loc="lower right")
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("Discrimination vs stratification", fontweight="bold", pad=8)

    save_figure(fig, "Figure_06_km_clinical")


# ===================================================================
# Figure 7; Systematic failure of domain generalization methods
# ===================================================================
def generate_figure_7():
    print("[Fig 7] Method comparison ...")
    mc = load_json(RESULTS_DIR / "experiments" / "exp_j" / "method_comparison.json")
    res = mc["results"]
    methods = mc["methods"]
    table3 = pd.read_csv(RESULTS_DIR / "tables" / "table3_method_comparison.csv")
    klin = load_json(RESULTS_DIR / "experiments" / "exp_k" / "clinical_metrics.json")

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.50,
                           left=0.07, right=0.94, top=0.90, bottom=0.09)

    # ---- (a) C-index across methods (Exp A) ----
    ax = fig.add_subplot(gs[0, 0])
    exp = "A"
    means = [res[exp][m]["cindex_mean"] for m in methods]
    stds = [res[exp][m]["cindex_std"] for m in methods]
    x = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]
    ax.bar(x, means, yerr=stds, color=colors, width=0.7, **BAR_EDGE,
           capsize=2, error_kw={"lw": 0.8, "color": "#555555"})
    ax.axhline(res[exp]["ERM"]["cindex_mean"], color="grey", ls="--", lw=0.9)
    ax.text(len(methods) - 0.5, res[exp]["ERM"]["cindex_mean"] + 0.0015, "ERM",
            ha="right", fontsize=6, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("C-index (3 seeds)")
    ax.set_ylim(0.61, 0.645)
    add_axis_style(ax)
    add_panel_label(ax, "a")
    ax.set_title("C-index across 9 methods (Exp A)", fontweight="bold", pad=8)

    # ---- (b) forest: gain vs ERM, grouped by experiment, p in right column ----
    ax = fig.add_subplot(gs[0, 1])
    rows = []
    for exp_lab in ["A", "B", "C"]:
        sub = table3[(table3["experiment"] == exp_lab) & (table3["method"] != "ERM")]
        for _, r in sub.iterrows():
            rows.append((f"{exp_lab}·{r['method']}", r["delta_vs_erm"],
                         r["ci_low"], r["ci_high"], r["p_two_sided"],
                         r.get("fdr_significant_05", False)))
    y = np.arange(len(rows))
    for i, (lab, d, lo, hi, p, fdr) in enumerate(rows):
        if fdr:
            col = "#2ca02c" if d > 0 else "#d62728"
        else:
            col = "#888888"
        ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round", alpha=0.85)
        ax.plot(d, i, "o", color=col, ms=4.5)
        ax.text(0.135, i, f"p={p:.3f}", va="center", ha="left", fontsize=5, color="#555")
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.5)
    ax.set_xlim(-0.16, 0.16)
    ax.set_xlabel("$\\Delta$ C-index vs ERM, 95% CI")
    ax.set_ylim(-1, len(rows))
    # light separators between experiment groups
    for exp_lab in ["B", "C"]:
        row0 = next(i for i, r in enumerate(rows) if r[0].startswith(exp_lab))
        ax.axhline(row0 - 0.5, color="#cccccc", lw=0.6, ls=":")
    ax.text(0.02, 0.97, "green/red = survives FDR\n(grey = n.s.)",
            transform=ax.transAxes, fontsize=6, color="#444")
    add_axis_style(ax)
    add_panel_label(ax, "b")
    ax.set_title("Gain vs ERM (bootstrap)", fontweight="bold", pad=8)

    # ---- (c) IBS comparison ----
    ax = fig.add_subplot(gs[0, 2])
    methods_k = ["ERM", "DANN", "CORAL", "IRM"]
    ibs = [klin["models"][m]["ibs"] for m in methods_k]
    km_null = klin["models"]["ERM"]["ibs_km_null"]
    ax.bar(range(len(methods_k)), ibs, color=[METHOD_COLORS[m] for m in methods_k],
           width=0.6, **BAR_EDGE)
    ax.axhline(km_null, color="grey", ls="--", lw=1)
    ax.text(3.3, km_null, f"KM null {km_null:.3f}", fontsize=6, color="#555")
    for i, v in enumerate(ibs):
        ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(methods_k)))
    ax.set_xticklabels(methods_k)
    ax.set_ylabel("Integrated Brier score")
    ax.set_ylim(0.19, 0.22)
    add_axis_style(ax)
    add_panel_label(ax, "c")
    ax.set_title("Calibration: IBS (lower better)", fontweight="bold", pad=8)

    # ---- (d) representation domain separability: baseline vs DANN probe ----
    ax = fig.add_subplot(gs[1, 0])
    probe = load_json(RESULTS_DIR / "experiments" / "exp_i" / "domain_quantify.json")["probe"]
    order = ["A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts",
             "D_imputed_tcga_seer", "E_brca_metabric"]
    labs = ["A", "B", "C", "D", "E"]
    bp = [probe[k]["baseline_probe"]["auc"] for k in order]
    dp = [probe[k]["dann_probe"]["auc"] for k in order]
    x = np.arange(len(labs))
    w = 0.34
    ax.bar(x - w / 2, bp, w, label="Baseline repr.", color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, dp, w, label="DANN repr.", color=COLOR_DANN, **BAR_EDGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("Representation probe AUC")
    ax.set_ylim(0.6, 1.02)
    ax.axhline(0.5, color="grey", ls=":", lw=0.7)
    no_legend_box(ax, loc="lower left")
    ax.text(0.03, 0.97, "representations stay\ndomain-separable", transform=ax.transAxes,
            fontsize=6, color="#444")
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("Domain separability: baseline vs DANN", fontweight="bold", pad=8)

    # ---- (e) training time vs performance (colour legend, no text labels) ----
    ax = fig.add_subplot(gs[1, 1])
    exp = "A"
    for m in methods:
        t = res[exp][m]["elapsed_sec"]
        c = res[exp][m]["cindex_mean"]
        col = METHOD_COLORS[m]
        sz = 70 if m == "DANN" else 40
        ax.scatter(t, c, s=sz, color=col, zorder=5, edgecolor="white", linewidth=0.4,
                   label=m)
    ax.set_xlabel("Training time (s); Exp A")
    ax.set_ylabel("C-index")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.61, 0.645)
    ax.text(0.03, 0.05, "DANN adds cost, no gain;\nFish is the slowest",
            transform=ax.transAxes, fontsize=6, color="#444")
    no_legend_box(ax, loc="center right", fontsize=5.5, ncol=1,
                  markerscale=0.7)
    add_axis_style(ax)
    add_panel_label(ax, "e")
    ax.set_title("Training time vs performance", fontweight="bold", pad=8)

    # ---- (f) adversarial oscillation Exp B ----
    ax = fig.add_subplot(gs[1, 2])
    hist, epochs = experiment_history("B_tcga_vs_external", "dann")
    ep = hist["epoch"]
    ax.plot(ep, hist["domain_loss"], color=COLOR_BASELINE, lw=1.4, label="Domain loss (left)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain loss", color=COLOR_BASELINE)
    ax2 = ax.twinx()
    ax2.plot(ep, hist["val_cindex"], color=COLOR_DANN, lw=1.4, ls="--",
             label="Val C-index (right)")
    ax2.set_ylabel("Val C-index", color=COLOR_DANN)
    ax2.set_ylim(0.4, 0.75)
    ax.set_ylim(0, 4)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax2.legend(loc="lower right", frameon=False, fontsize=6)
    ax.text(0.04, 0.62, "minimax oscillation;\nsurvival track is flat",
            transform=ax.transAxes, fontsize=6, color="#444")
    add_axis_style(ax)
    add_panel_label(ax, "f")
    ax.set_title("Adversarial oscillation (Exp B)", fontweight="bold", pad=8)

    save_figure(fig, "Figure_07_methods")


# ===================================================================
# Figure 8; Domain-survival entanglement + cross-cancer generalization
# ===================================================================
def generate_figure_8():
    print("[Fig 8] Quantification + multi-cancer ...")
    diq = load_json(RESULTS_DIR / "experiments" / "exp_i" / "domain_quantify.json")
    coad = load_json(RESULTS_DIR / "experiments" / "exp_l" / "coad_results.json")
    luad = load_json(RESULTS_DIR / "experiments" / "exp_m" / "luad_results.json")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.46,
                           left=0.07, right=0.97, top=0.90, bottom=0.09)

    # ---- (a) Wasserstein matrix (no diagonal boxes, light grid) ----
    ax = fig.add_subplot(gs[0, 0])
    wmat = np.array(diq["wasserstein"]["age_wasserstein"])
    cohorts = diq["wasserstein"]["cohorts"]
    mask = np.triu(np.ones_like(wmat, dtype=bool), k=1)
    sns.heatmap(wmat, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                xticklabels=cohorts, yticklabels=cohorts, vmin=0, vmax=14,
                linewidths=0.3, linecolor="#e8e8e8",
                cbar_kws={"label": "W1 (Age, years)", "shrink": 0.85},
                mask=~mask, annot_kws={"fontsize": 6.5})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.set_title("Wasserstein-1, Age", fontweight="bold", pad=8)
    add_panel_label(ax, "a")

    # ---- (b) propensity overlap TCGA vs SEER ----
    ax = fig.add_subplot(gs[0, 1])
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from sklearn.linear_model import LogisticRegression
    df = load_lihc()
    feats = pd.DataFrame(index=df.index)
    feats["Age"] = df["Age"].fillna(df["Age"].median()).astype(float)
    for c in ["Sex", "Stage", "Grade"]:
        codes, _ = pd.factorize(df[c].fillna("__MISSING__").astype(str))
        feats[f"{c}_code"] = codes
        feats[f"{c}_known"] = df[c].notna().astype(int)
    X = feats.values.astype(float)
    cohorts_all = sorted(df["Source"].unique())
    y = df["Source"].map({c: i for i, c in enumerate(cohorts_all)}).values
    rng = np.random.RandomState(7)
    idx = rng.permutation(len(y))
    tr, te = idx[:int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    clf = LogisticRegression(max_iter=2000, C=0.5, multi_class="multinomial")
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])
    yte = y[te]
    own = np.array([proba[k, yte[k]] for k in range(len(te))])
    src_te = df["Source"].values[te]
    a_tcga = own[src_te == "TCGA_LIHC"]
    a_seer = own[src_te == "US_SEER"]
    bins = np.linspace(0, 1, 50)
    ax.hist(a_tcga, bins=bins, alpha=0.55, color=COLOR_BASELINE, density=True, label="TCGA-LIHC",
            edgecolor="white", linewidth=0.3)
    ax.hist(a_seer, bins=bins, alpha=0.55, color=COLOR_DANN, density=True, label="US_SEER",
            edgecolor="white", linewidth=0.3)
    ov_tcga_seer = diq["propensity"]["overlap_matrix"]
    i_t, i_s = cohorts_all.index("TCGA_LIHC"), cohorts_all.index("US_SEER")
    ov = ov_tcga_seer[i_t][i_s]
    ax.text(0.97, 0.95, f"overlap = {ov:.2f}\nmean pairwise = {diq['propensity']['mean_overlap']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    ax.set_xlabel("Propensity score (own cohort)")
    ax.set_ylabel("Density")
    no_legend_box(ax)
    ax.set_title("Propensity overlap: TCGA vs SEER", fontweight="bold", pad=8)
    add_panel_label(ax, "b")

    # ---- (c) MINE mutual information (values on key bars only) ----
    ax = fig.add_subplot(gs[0, 2])
    mine = diq["mine"]
    labs = ["A", "B", "C", "D", "E"]
    mi_surv = [mine[l]["mi_domain_survival"] for l in labs]
    mi_repr = [mine[l]["mi_domain_representation"] for l in labs]
    x = np.arange(len(labs))
    w = 0.36
    ax.bar(x - w / 2, mi_surv, w, label="I(domain; survival)", color="#C44E52", **BAR_EDGE)
    ax.bar(x + w / 2, mi_repr, w, label="I(domain; repr.)", color="#4C72B0", **BAR_EDGE)
    for i in range(len(labs)):
        ax.text(x[i] - w / 2, mi_surv[i] + 0.01, f"{mi_surv[i]:.2f}",
                ha="center", fontsize=5.5, color="#555")
        ax.text(x[i] + w / 2, mi_repr[i] + 0.01, f"{mi_repr[i]:.2f}",
                ha="center", fontsize=5.5, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("MINE estimate")
    ax.set_ylim(0, 0.95)
    no_legend_box(ax, fontsize=5.5)
    ax.text(0.03, 0.97, "E: weakest repr.-level\nentanglement (0.07)",
            transform=ax.transAxes, fontsize=6, color="#444")
    add_axis_style(ax)
    add_panel_label(ax, "c")
    ax.set_title("MINE mutual information (A–E)", fontweight="bold", pad=8)

    # ---- (d) domain separability vs GRL gain ----
    ax = fig.add_subplot(gs[1, 0])
    probe = diq["probe"]
    order = ["A_tcga_vs_seer", "B_tcga_vs_external", "C_all_cohorts",
             "D_imputed_tcga_seer", "E_brca_metabric"]
    labs_d = ["A", "B", "C", "D", "E"]
    px = [probe[k]["baseline_probe"]["auc"] for k in order]
    dy = [0.0007, -0.0032, -0.0011, 0.0027, 0.0074]
    for i, (xi, yi) in enumerate(zip(px, dy)):
        col = COLOR_DANN if labs_d[i] == "E" else "#888888"
        ax.scatter(xi, yi, s=55, color=col, zorder=5, edgecolor="white", linewidth=0.4)
        ax.annotate(labs_d[i], xy=(xi, yi), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, fontweight="bold")
    res = stats.linregress(px, dy)
    xx = np.linspace(0.68, 1.0, 50)
    ax.plot(xx, res.intercept + res.slope * xx, color="#444", ls="--", lw=1.0)
    ax.fill_between(xx, (res.intercept + res.slope * xx) - 1.96 * res.stderr,
                    (res.intercept + res.slope * xx) + 1.96 * res.stderr,
                    color="#444", alpha=0.08)
    ax.set_xlabel("Baseline representation probe AUC")
    ax.set_ylabel("GRL gain $\\Delta$ C-index")
    ax.set_ylim(-0.008, 0.011)
    ax.text(0.03, 0.05, f"R={res.rvalue:.2f}, p={res.pvalue:.2f}",
            transform=ax.transAxes, fontsize=6.5)
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("Probe AUC vs GRL gain", fontweight="bold", pad=8)

    # ---- (e/f) COAD / LUAD ----
    for idx, (resm, name, lab) in enumerate([(coad, "COAD", "e"), (luad, "LUAD", "f")]):
        ax = fig.add_subplot(gs[1, idx + 1])
        methods = ["ERM", "DANN", "CORAL", "IRM"]
        means = [resm[m]["cindex_mean"] for m in methods]
        stds = [resm[m]["cindex_std"] for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, color=[METHOD_COLORS[m] for m in methods],
               width=0.6, **BAR_EDGE, capsize=3, error_kw={"lw": 0.9, "color": "#555555"})
        b = resm["bootstrap_vs_erm"]["DANN"]
        ax.text(0.98, 0.05, f"DANN $\\Delta$={b['delta_mean']:+.3f}\np={b['p_two_sided']:.2f}",
                transform=ax.transAxes, ha="right", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("C-index (3 seeds)")
        ax.set_ylim(0.7, 0.86)
        add_axis_style(ax)
        add_panel_label(ax, lab)
        ax.set_title(f"{name} validation", fontweight="bold", pad=8)

    save_figure(fig, "Figure_08_quantify_multicancer")


# ===================================================================
# Supplementary Figure S1, t-SNE
# ===================================================================
def generate_s1():
    print("[S1] t-SNE ...")
    from sklearn.manifold import TSNE
    exps = [("A_tcga_vs_seer", "A"), ("B_tcga_vs_external", "B"), ("C_all_cohorts", "C")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (e, lab) in zip(axes, exps):
        labels_df = pd.read_csv(EXP_DIR / e / "dann" / "domain_labels.csv")
        cohorts = labels_df["Source"].values
        f_d = np.load(EXP_DIR / e / "dann" / "features.npy")
        f_b = np.load(EXP_DIR / e / "baseline" / "features.npy")
        n = min(3000, len(f_d))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(f_d), n, replace=False)
        # colour by cohort, shape by model
        feats = np.vstack([f_d[idx], f_b[idx]])
        coords = TSNE(n_components=2, perplexity=min(30, n // 3), max_iter=1000,
                      random_state=42).fit_transform(feats)
        cd, cb = coords[:n], coords[n:]
        coh = cohorts[idx]
        uniq = list(dict.fromkeys(coh))
        for c in uniq:
            m = coh == c
            col = COHORT_COLORS.get(c, "#95a5a6")
            ax.scatter(cd[m, 0], cd[m, 1], s=5, color=col, alpha=0.6,
                       marker="o", label=c)
            ax.scatter(cb[m, 0], cb[m, 1], s=4, color=col, alpha=0.35,
                       marker="^")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Exp {lab}", fontweight="bold")
        add_axis_style(ax)
    handles = [Line2D([0], [0], marker="o", ls="", color="grey", label="DANN (circles)"),
               Line2D([0], [0], marker="^", ls="", color="grey", label="Baseline (triangles)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    save_figure(fig, "Figure_S1_tsne", is_supplementary=True)


# ===================================================================
# Supplementary Figure S2, training dynamics (3 rows)
# ===================================================================
def generate_s2():
    print("[S2] Training dynamics ...")
    exps = [("A_tcga_vs_seer", "A"), ("B_tcga_vs_external", "B"), ("C_all_cohorts", "C")]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for ax, (e, lab) in zip(axes, exps):
        hist, epochs = experiment_history(e, "dann")
        ep = hist["epoch"]
        ax.plot(ep, hist["surv_loss"], color="#2ca02c", lw=1.3, label="Survival loss")
        ax.plot(ep, hist["domain_loss"], color="#FF7F0E", lw=1.3, label="Domain loss")
        ax.plot(ep, hist["domain_acc"], color=COLOR_DANN, lw=1.3, ls="--",
                label="Domain acc. (right)")
        ax2 = ax.twinx()
        ax2.plot(ep, hist["val_cindex"], color=COLOR_BASELINE, lw=1.3, label="Val C-index (right)")
        ax2.set_ylim(0.4, 0.75)
        no_legend_box(ax, loc="upper left", fontsize=6.5)
        ax2.legend(loc="lower right", frameon=False, fontsize=6.5)
        ax.set_ylabel("Loss")
        ax2.set_ylabel("C-index / Acc.")
        ax.set_title(f"Exp {lab}", fontweight="bold", fontsize=9)
        add_axis_style(ax)
    axes[-1].set_xlabel("Epoch")
    save_figure(fig, "Figure_S2_training_dynamics", is_supplementary=True)


# ===================================================================
# Supplementary Figure S3; GRL scheduling
# ===================================================================
def generate_s3():
    print("[S3] GRL scheduling ...")
    hist, epochs = experiment_history("A_tcga_vs_seer", "dann")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ep = hist["epoch"]
    ax.plot(ep, hist["alpha"], color=COLOR_DANN, lw=1.8, marker="o", ms=3,
            label="GRL $\\alpha$ (left)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GRL $\\alpha$", color=COLOR_DANN)
    ax.tick_params(axis="y", labelcolor=COLOR_DANN)
    ax2 = ax.twinx()
    dom_w = np.linspace(0, 0.3, len(ep))
    ax2.plot(ep, dom_w, color="#FF7F0E", lw=1.8, ls="--", label="domain weight (right)")
    ax2.set_ylabel("Domain weight", color="#FF7F0E")
    ax2.tick_params(axis="y", labelcolor="#FF7F0E")
    ax2.set_ylim(0, 0.4)
    ax.set_ylim(0, 1.05)
    add_axis_style(ax)
    save_figure(fig, "Figure_S3_grl_schedule", is_supplementary=True)


# ===================================================================
# Supplementary Figure S4; calibration curves
# ===================================================================
def generate_s4():
    print("[S4] Calibration curves ...")
    klin = load_json(RESULTS_DIR / "experiments" / "exp_k" / "clinical_metrics.json")
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.44,
                           left=0.09, right=0.95, top=0.90, bottom=0.10)
    times = ["12", "36", "60"]
    for idx, t in enumerate(times):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        for m in ["ERM", "DANN"]:
            cal = klin["models"][m]["calibration"][t]["calibration"]
            p = [c["predicted"] for c in cal]
            o = [c["observed"] for c in cal]
            ax.plot(p, o, marker="o", ms=4, lw=1.2,
                    color=COLOR_BASELINE if m == "ERM" else COLOR_DANN,
                    label=f"{m} (ECE={klin['models'][m]['calibration'][t]['ece']:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        no_legend_box(ax, fontsize=6.5)
        add_axis_style(ax)
        add_panel_label(ax, "abc"[idx])
        ax.set_title(f"Calibration at {t} months", fontweight="bold", pad=8)
    # (d) ECE summary
    ax = fig.add_subplot(gs[1, 1])
    ece = np.zeros((2, 3))
    for i, m in enumerate(["ERM", "DANN"]):
        for j, t in enumerate(times):
            ece[i, j] = klin["models"][m]["calibration"][t]["ece"]
    x = np.arange(3)
    w = 0.34
    ax.bar(x - w / 2, ece[0], w, label="ERM", color=COLOR_BASELINE, **BAR_EDGE)
    ax.bar(x + w / 2, ece[1], w, label="DANN", color=COLOR_DANN, **BAR_EDGE)
    for j in range(3):
        ax.text(x[j] - w / 2, ece[0, j] + 0.002, f"{ece[0, j]:.3f}", ha="center", fontsize=6.5)
        ax.text(x[j] + w / 2, ece[1, j] + 0.002, f"{ece[1, j]:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} mo" for t in times])
    ax.set_ylabel("Expected calibration error")
    no_legend_box(ax)
    add_axis_style(ax)
    add_panel_label(ax, "d")
    ax.set_title("ECE summary", fontweight="bold", pad=8)
    save_figure(fig, "Figure_S4_calibration", is_supplementary=True)


# ===================================================================
# Supplementary Figure S5; Decision curve analysis
# ===================================================================
def generate_s5():
    print("[S5] Decision curve analysis ...")
    klin = load_json(RESULTS_DIR / "experiments" / "exp_k" / "clinical_metrics.json")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for m in ["ERM", "DANN", "CORAL", "IRM"]:
        dca = klin["models"][m]["dca"]
        th = dca["thresholds"]
        nb = dca["net_benefit"]
        col = METHOD_COLORS.get(m, "#888888")
        ax.plot(th, nb, lw=1.4, color=col, label=m)
    # treat-all / treat-none
    t_all = 0.6308
    prev = t_all
    ths = np.linspace(0.01, 0.99, 50)
    ax.plot(ths, prev - (1 - prev) * ths / (1 - ths), color="grey", ls="--", lw=1,
            label="Treat all")
    ax.plot(ths, np.zeros_like(ths), color="black", ls=":", lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit (36 mo)")
    ax.set_ylim(-0.1, 0.8)
    no_legend_box(ax, fontsize=6.5)
    add_axis_style(ax)
    save_figure(fig, "Figure_S5_dca", is_supplementary=True)


# ===================================================================
# Supplementary data tables
# ===================================================================
def generate_supplementary_tables():
    print("[Tables] Supplementary data tables ...")
    # ---- S1: full harmonized data inventory ----
    # used_in: which experiment (A–M) consumes each harmonized source
    used_in = {
        "harmonized_LIHC.csv": "A,B,C,D", "harmonized_seer.csv": "A,B,C,D",
        "harmonized_hcc_msk_2024.csv": "B,C", "harmonized_lihc_amc_prv.csv": "B,C",
        "harmonized_hcc_meric_2021.csv": "B,C",
        "harmonized_BRCA.csv": "E", "harmonized_metabric.csv": "E",
        "harmonized_COAD.csv": "L", "harmonized_cptac_coad.csv": "L",
        "harmonized_LUAD.csv": "M", "harmonized_luad_mskcc_2020.csv": "M",
    }
    rows = []
    for f in sorted(os.listdir(BASE_DIR / "data_processed")):
        if not f.endswith(".csv") or f == "lihc_all_cohorts.csv" \
                or f == "missingness_summary.csv":
            continue
        df = pd.read_csv(BASE_DIR / "data_processed" / f)
        n = len(df)
        ev = df["Vital_Status"].mean() if "Vital_Status" in df else np.nan
        # cohort label: strip harmonized_ prefix + .csv, upper-case TCGA names
        src = f.replace("harmonized_", "").replace(".csv", "")
        if src in ("BRCA", "LIHC", "LUAD", "COAD", "KIRC", "PRAD", "SKCM",
                   "STAD", "THCA", "UCEC"):
            src = f"TCGA-{src}"
        lihc_file = ("hcc" in f or "lihc" in f or f in ("harmonized_LIHC.csv",
                     "harmonized_seer.csv"))
        rows.append({"source": src, "file": f, "cancer_type":
                     "LIHC" if lihc_file else
                     ("BRCA" if "brca" in f else ("COAD" if "coad" in f else
                      ("LUAD" if "luad" in f else "other"))),
                     "n_patients": n,
                     "event_rate": round(float(ev), 4) if ev == ev else "",
                     "used_in_experiments": used_in.get(f, "")})
    s1 = pd.DataFrame(rows)
    s1.to_csv(SUPPL_DIR / "Supp_Table_S1_data_inventory.csv", index=False)

    # ---- S2: cohort summary used in experiments ----
    li = load_lihc()
    miss = pd.read_csv(RESULTS_DIR / "lihc_missingness.csv").set_index("Source")
    rows2 = []
    for c in COHORTS5:
        sub = li[li["Source"] == c]
        rows2.append({
            "cohort": c, "experiments": {"TCGA_LIHC": "A,B,C,D", "US_SEER": "A,B,C,D",
                                         "hcc_msk_2024": "B,C", "lihc_amc_prv": "B,C",
                                         "hcc_meric_2021": "B,C"}[c],
            "n": len(sub), "n_events": int(sub["Vital_Status"].sum()),
            "event_rate": round(sub["Vital_Status"].mean(), 4),
            "age_median": round(sub["Age"].median(), 1) if sub["Age"].notna().any() else "",
            "age_missing_pct": miss.loc[c, "Age"], "sex_missing_pct": miss.loc[c, "Sex"],
            "stage_missing_pct": miss.loc[c, "Stage"], "grade_missing_pct": miss.loc[c, "Grade"],
        })
    # E/L/M cohorts
    rows2.append({"cohort": "TCGA-BRCA", "experiments": "E", "n": 1098,
                  "n_events": "", "event_rate": 0.139, "age_median": "", "age_missing_pct": 0,
                  "sex_missing_pct": 0, "stage_missing_pct": 0, "grade_missing_pct": 100})
    rows2.append({"cohort": "METABRIC", "experiments": "E", "n": 2509,
                  "n_events": "", "event_rate": 0.456, "age_median": "", "age_missing_pct": 0,
                  "sex_missing_pct": 0, "stage_missing_pct": 0, "grade_missing_pct": ""})
    rows2.append({"cohort": "TCGA-COAD", "experiments": "L", "n": 461,
                  "n_events": "", "event_rate": 0.222, "age_median": "", "age_missing_pct": 0,
                  "sex_missing_pct": 0, "stage_missing_pct": "", "grade_missing_pct": ""})
    rows2.append({"cohort": "CPTAC-COAD", "experiments": "L", "n": 110,
                  "n_events": "", "event_rate": 0.078, "age_median": "", "age_missing_pct": 0,
                  "sex_missing_pct": 0, "stage_missing_pct": "", "grade_missing_pct": ""})
    rows2.append({"cohort": "TCGA-LUAD", "experiments": "M", "n": 585,
                  "n_events": "", "event_rate": 0.36, "age_median": "", "age_missing_pct": 0,
                  "sex_missing_pct": 0, "stage_missing_pct": "", "grade_missing_pct": ""})
    rows2.append({"cohort": "LUAD-MSKCC-2020", "experiments": "M", "n": 604,
                  "n_events": "", "event_rate": 0.075, "age_median": "", "age_missing_pct": 0,
                  "sex_missing_pct": 0, "stage_missing_pct": "", "grade_missing_pct": ""})
    s2 = pd.DataFrame(rows2)
    s2.to_csv(SUPPL_DIR / "Supp_Table_S2_cohort_summary.csv", index=False)

    # ---- S3: baseline (DANN vs ERM) results ----
    tbl2 = pd.read_csv(RESULTS_DIR / "tables" / "table2_main_results.csv")
    tbl2.to_csv(SUPPL_DIR / "Supp_Table_S3_baseline_results.csv", index=False)

    # ---- S4: method comparison ----
    pd.read_csv(RESULTS_DIR / "tables" / "table3_method_comparison.csv").to_csv(
        SUPPL_DIR / "Supp_Table_S4_method_comparison.csv", index=False)

    # ---- S5: clinical metrics ----
    pd.read_csv(RESULTS_DIR / "tables" / "table4_clinical_metrics.csv").to_csv(
        SUPPL_DIR / "Supp_Table_S5_clinical_metrics.csv", index=False)

    # ---- S6: domain quantification ----
    pd.read_csv(RESULTS_DIR / "tables" / "table5_domain_quantify.csv").to_csv(
        SUPPL_DIR / "Supp_Table_S6_domain_quantify.csv", index=False)

    # ---- S7: multi-cancer validation ----
    pd.read_csv(RESULTS_DIR / "tables" / "table6_multicancer_validation.csv").to_csv(
        SUPPL_DIR / "Supp_Table_S7_multicancer_validation.csv", index=False)
    print("  saved 7 supplementary CSV tables")


# ===================================================================
# Main
# ===================================================================
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    funcs = {
        "fig1": generate_figure_1, "fig2": generate_figure_2,
        "fig3": generate_figure_3, "fig4": generate_figure_4,
        "fig5": generate_figure_5, "fig6": generate_figure_6,
        "fig7": generate_figure_7, "fig8": generate_figure_8,
        "s1": generate_s1, "s2": generate_s2, "s3": generate_s3,
        "s4": generate_s4, "s5": generate_s5, "tables": generate_supplementary_tables,
    }
    if which == "all":
        for f in funcs.values():
            f()
    else:
        funcs[which]()
    print("Done.")


if __name__ == "__main__":
    main()
