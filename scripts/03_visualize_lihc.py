#!/usr/bin/env python3
"""
03_visualize_lihc.py; Comprehensive Visualization for LIHC DANN Experiments
==============================================================================
Generates publication-ready figures:

  1. Missingness Heatmap; Per-cohort missingness matrix (domain fingerprint)
  2. t-SNE Comparison; DANN vs Baseline latent space (cohort separation)
  3. C-index Bar Chart; DANN vs Baseline per-experiment comparison
  4. Domain Loss Curves; Domain classifier accuracy during DANN training
  5. Training Convergence; Survival + Domain loss curves
  6. Per-cohort C-index; Detailed per-cohort comparison table + bar chart
  7. Time-dependent AUC; AUC at 12/36/60 months comparison

Usage:
  python scripts/03_visualize_lihc.py                        # full visualization
  python scripts/03_visualize_lihc.py --figures tsne,cindex   # specific figures
  python scripts/03_visualize_lihc.py --output paper_figures  # custom output
"""

import argparse
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
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
LIHC_EXP_DIR = RESULTS_DIR / "lihc_experiments"
MISSINGNESS_PATH = RESULTS_DIR / "lihc_missingness.csv"
DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"

sns.set_theme(style="whitegrid")
COLOR_DANN = "#e74c3c"
COLOR_BASE = "#3498db"
COLOR_PALETTE = {
    "TCGA_LIHC": "#2ecc71",
    "US_SEER": "#3498db",
    "hcc_msk_2024": "#e74c3c",
    "hcc_clca_2024": "#f39c12",
    "lihc_amc_prv": "#9b59b6",
    "hcc_meric_2021": "#1abc9c",
    "hcc_inserm_fr_2015": "#e67e22",
    "hcc_mskimpact_2018": "#e91e63",
}

EXPERIMENT_LABELS = {
    "A_tcga_vs_seer": "TCGA-LIHC vs SEER\n(Extreme Missingness)",
    "B_tcga_vs_external": "TCGA-LIHC vs External HCC\n(Different Databases)",
    "C_all_cohorts": "All LIHC Cohorts\n(Multi-Domain)",
}

OUTPUT_DIR = BASE_DIR / "results" / "figures" / "lihc"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# 1. Missingness Heatmap
# ===================================================================
def plot_missingness_heatmap():
    """Plot per-cohort missingness matrix as a heatmap."""
    print("\n[1/7] Missingness heatmap ...")

    if not MISSINGNESS_PATH.exists():
        print("  ⚠ Missingness data not found, computing from harmonized data ...")
        # Compute from harmonized files
        try:
            sys.path.insert(0, str(BASE_DIR / "scripts"))
            from analyze_expanded_data import load_all_harmonized, compute_missingness_matrix
            df = load_all_harmonized()
            missing_df = compute_missingness_matrix(df)
        except Exception as e:
            print(f"  ✗ Cannot compute: {e}")
            return
    else:
        missing_df = pd.read_csv(MISSINGNESS_PATH)

    # Filter to LIHC cohorts
    lihc_sources = [
        "TCGA_LIHC", "US_SEER", "hcc_msk_2024", "hcc_clca_2024",
        "lihc_amc_prv", "hcc_meric_2021", "hcc_inserm_fr_2015",
        "hcc_mskimpact_2018"
    ]
    missing_df = missing_df[missing_df["Source"].isin(lihc_sources)].copy()

    # Features for heatmap
    feat_cols = [c for c in ["Age", "Sex", "Stage", "Grade", "Survival_Months", "Vital_Status"]
                 if c in missing_df.columns]
    plot_df = missing_df.set_index("Source")[feat_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.color_palette("Reds", as_cmap=True)

    sns.heatmap(
        plot_df, annot=True, fmt=".1f", cmap=cmap,
        vmin=0, vmax=100, linewidths=1, linecolor="white",
        cbar_kws={"label": "Missing Rate (%)", "shrink": 0.8},
        ax=ax
    )
    ax.set_title("Missingness Fingerprint; LIHC Cohorts\n"
                 "(Higher = More Missing, Darker Red = Stronger Domain Signal)",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("Clinical Feature", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add N annotation
    for i, src in enumerate(plot_df.index):
        n = missing_df[missing_df["Source"] == src]["N"].values
        ax.text(len(feat_cols) + 0.3, i + 0.5, f"N={n[0]}",
                va="center", fontsize=8, color="#555")

    plt.tight_layout()
    path = OUTPUT_DIR / "01_missingness_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {path}")


# ===================================================================
# 2. t-SNE Comparison
# ===================================================================
def plot_tsne_comparison():
    """t-SNE visualization: DANN vs Baseline latent space."""
    print("\n[2/7] t-SNE latent space comparison ...")

    from sklearn.manifold import TSNE

    experiments = sorted([
        d for d in os.listdir(LIHC_EXP_DIR)
        if (LIHC_EXP_DIR / d / "dann" / "features.npy").exists()
        and (LIHC_EXP_DIR / d / "baseline" / "features.npy").exists()
    ])

    for exp_key in experiments:
        exp_dir = LIHC_EXP_DIR / exp_key
        labels_path = exp_dir / "dann" / "domain_labels.csv"

        if not labels_path.exists():
            continue

        labels_df = pd.read_csv(labels_path)
        cohorts = labels_df["Source"].values
        domain_labels = labels_df["Domain_Label"].values

        # ---- DANN features ----
        dann_feats = np.load(exp_dir / "dann" / "features.npy")
        base_feats = np.load(exp_dir / "baseline" / "features.npy")

        # Subsample for t-SNE (max 3000)
        n_samples = min(3000, len(dann_feats))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(dann_feats), n_samples, replace=False)

        dann_sample = dann_feats[idx]
        base_sample = base_feats[idx]
        cohort_labels = [cohorts[i] for i in idx]

        # Map cohort names to palette colors
        cohort_colors = [COLOR_PALETTE.get(c, "#95a5a6") for c in cohort_labels]
        unique_cohorts = list(dict.fromkeys(cohort_labels))  # preserve order

        # Compute t-SNE for both
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, feats, title, colors in [
            (axes[0], dann_sample, "TransDANN (with GRL)", cohort_colors),
            (axes[1], base_sample, "Baseline Transformer (no GRL)", cohort_colors),
        ]:
            tsne = TSNE(n_components=2, perplexity=min(30, n_samples // 3),
                       max_iter=1000, random_state=42, verbose=0)
            coords = tsne.fit_transform(feats)

            for i, cohort in enumerate(unique_cohorts):
                mask = [c == cohort for c in cohort_labels]
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=COLOR_PALETTE.get(cohort, "#95a5a6"),
                    label=cohort, alpha=0.6, s=15, edgecolors="none"
                )

            ax.set_title(title, fontweight="bold", fontsize=12)
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.legend(markerscale=2, frameon=True, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle(
            f"Latent Space Visualization: {EXPERIMENT_LABELS.get(exp_key, exp_key)}",
            fontweight="bold", fontsize=13, y=1.02
        )
        plt.tight_layout()
        path = OUTPUT_DIR / f"02_tsne_{exp_key}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved: {path}")


# ===================================================================
# 3. C-index Bar Chart
# ===================================================================
def plot_cindex_comparison():
    """C-index bar chart: DANN vs Baseline across all experiments."""
    print("\n[3/7] C-index comparison bar chart ...")

    experiments = sorted([
        d for d in os.listdir(LIHC_EXP_DIR)
        if (LIHC_EXP_DIR / d / "dann" / "results.json").exists()
        and (LIHC_EXP_DIR / d / "baseline" / "results.json").exists()
    ])

    if not experiments:
        print("  ⚠ No experiment results found")
        return

    # Collect data
    labels = []
    dann_scores = []
    base_scores = []
    dann_per_cohort = {}
    base_per_cohort = {}

    for exp_key in experiments:
        label = EXPERIMENT_LABELS.get(exp_key, exp_key)
        labels.append(label)

        with open(LIHC_EXP_DIR / exp_key / "dann" / "results.json") as f:
            dann_r = json.load(f)
        with open(LIHC_EXP_DIR / exp_key / "baseline" / "results.json") as f:
            base_r = json.load(f)

        dann_scores.append(dann_r.get("best_val_cindex", 0.5))
        base_scores.append(base_r.get("best_val_cindex", 0.5))
        dann_per_cohort[exp_key] = dann_r.get("per_cohort_cindex", {})
        base_per_cohort[exp_key] = base_r.get("per_cohort_cindex", {})

    # Main comparison bar chart
    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, dann_scores, width, label="DANN (with GRL)",
                   color=COLOR_DANN, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, base_scores, width, label="Baseline (no GRL)",
                   color=COLOR_BASE, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                color=COLOR_DANN, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                color=COLOR_BASE, fontweight="bold")

    # Add delta labels
    for i in range(len(labels)):
        delta = dann_scores[i] - base_scores[i]
        color = "#e74c3c" if delta < 0 else "#2ecc71"
        ax.annotate(
            f"Δ={delta:+.4f}",
            xy=(x[i], max(dann_scores[i], base_scores[i]) + 0.03),
            ha="center", fontsize=9, color=color, fontweight="bold"
        )

    ax.set_ylabel("Validation C-index", fontsize=12)
    ax.set_title("DANN vs Baseline: Cross-Domain Survival Prediction",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.legend(frameon=True, fontsize=11, loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "03_cindex_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {path}")

    # ---- Per-cohort C-index detail chart ----
    for exp_key in experiments:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Load cohort map
        with open(LIHC_EXP_DIR / exp_key / "dann" / "results.json") as f:
            dann_r = json.load(f)
        cohort_map = dann_r.get("cohort_map", {})
        rev_map = {v: k for k, v in cohort_map.items()}

        dpc = dann_per_cohort[exp_key]
        bpc = base_per_cohort[exp_key]

        cohorts_plot = sorted(rev_map.keys(), key=lambda x: int(x))
        cohort_names = [rev_map[int(k)] for k in cohorts_plot]
        d_vals = [dpc.get(str(k), 0) for k in cohorts_plot]
        b_vals = [bpc.get(str(k), 0) for k in cohorts_plot]

        x = np.arange(len(cohort_names))
        width = 0.3

        ax.bar(x - width / 2, d_vals, width, label="DANN (with GRL)",
               color=COLOR_DANN, edgecolor="white")
        ax.bar(x + width / 2, b_vals, width, label="Baseline (no GRL)",
               color=COLOR_BASE, edgecolor="white")

        for i in range(len(cohort_names)):
            delta = d_vals[i] - b_vals[i]
            color = "#e74c3c" if delta < 0 else "#2ecc71"
            ax.annotate(f"Δ={delta:+.3f}", xy=(x[i], max(d_vals[i], b_vals[i]) + 0.02),
                       ha="center", fontsize=8, color=color, fontweight="bold")

        ax.set_ylabel("C-index", fontsize=11)
        ax.set_title(f"Per-Cohort C-index: {EXPERIMENT_LABELS.get(exp_key, exp_key)}",
                     fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_names, fontsize=10)
        ax.legend(frameon=True, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = OUTPUT_DIR / f"03_per_cohort_cindex_{exp_key}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved: {path}")


# ===================================================================
# 4. Domain Loss Curves
# ===================================================================
def plot_domain_loss_curves():
    """Domain classifier accuracy and loss during DANN training."""
    print("\n[4/7] Domain loss curves ...")

    experiments = sorted([
        d for d in os.listdir(LIHC_EXP_DIR)
        if (LIHC_EXP_DIR / d / "dann" / "results.json").exists()
    ])

    for exp_key in experiments:
        with open(LIHC_EXP_DIR / exp_key / "dann" / "results.json") as f:
            results = json.load(f)

        history = results.get("history", {})
        if not history or not history.get("epoch"):
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Domain accuracy
        ax = axes[0]
        epochs = history["epoch"]
        dom_acc = history.get("domain_acc", [])
        if dom_acc and any(d > 0 for d in dom_acc):
            ax.plot(epochs[:len(dom_acc)], dom_acc, color=COLOR_DANN,
                    linewidth=1.5, marker=".", markersize=3)
            ax.set_title("Domain Classifier Accuracy\n(Higher = More Domain Info)",
                        fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        # Domain loss
        ax = axes[1]
        dom_loss = history.get("domain_loss", [])
        if dom_loss:
            ax.plot(epochs[:len(dom_loss)], dom_loss, color="#f39c12",
                    linewidth=1.5, marker=".", markersize=3)
            ax.set_title("Domain Classification Loss\n(Should Rise if GRL Works)",
                        fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy Loss")

        # Survival loss
        ax = axes[2]
        surv_loss = history.get("surv_loss", [])
        val_c = history.get("val_cindex", [])
        if surv_loss:
            ax_twin = ax.twinx()
            ax.plot(epochs[:len(surv_loss)], surv_loss, color="#2ecc71",
                    linewidth=1.5, marker=".", markersize=3, label="Surv Loss")
            ax_twin.plot(epochs[:len(val_c)], val_c, color=COLOR_DANN,
                        linewidth=1.5, marker=".", markersize=3, label="Val C-index")
            ax.set_title("Survival Loss & Validation C-index", fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Survival Loss", color="#2ecc71")
            ax_twin.set_ylabel("Val C-index", color=COLOR_DANN)
            ax_twin.set_ylim(0, 1.0)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, frameon=True, fontsize=8)

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle(f"Training Dynamics: {EXPERIMENT_LABELS.get(exp_key, exp_key)}",
                    fontweight="bold", fontsize=13)
        plt.tight_layout()
        path = OUTPUT_DIR / f"04_training_dynamics_{exp_key}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved: {path}")

    # Multi-experiment domain accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, exp_key in enumerate(experiments[:3]):
        with open(LIHC_EXP_DIR / exp_key / "dann" / "results.json") as f:
            results = json.load(f)
        history = results.get("history", {})
        dom_acc = history.get("domain_acc", [])
        epochs = history.get("epoch", [])
        if dom_acc and any(d > 0 for d in dom_acc):
            label = EXPERIMENT_LABELS.get(exp_key, exp_key).replace("\n", " ")
            ax.plot(epochs[:len(dom_acc)], dom_acc, color=colors[i % len(colors)],
                   linewidth=1.5, label=label)

    ax.set_title("Domain Classifier Accuracy Across Experiments\n"
                 "(GRL Should Increase Domain Loss → Decrease Accuracy)",
                 fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain Classification Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend(frameon=True, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "04_domain_accuracy_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {path}")


# ===================================================================
# 5. GRL Alpha & Domain Weight Schedule
# ===================================================================
def plot_training_schedule():
    """Visualize GRL alpha ramp and domain weight schedule."""
    print("\n[5/7] Training schedule ...")

    epochs = 200
    alphas = [2.0 / (1.0 + np.exp(-10.0 * e / epochs)) - 1.0 for e in range(epochs)]
    weights = [0.3 * e / epochs for e in range(epochs)]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(range(epochs), alphas, color=COLOR_DANN, linewidth=2, label="GRL α")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("GRL Alpha (α)", color=COLOR_DANN, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=COLOR_DANN)

    ax2 = ax1.twinx()
    ax2.plot(range(epochs), weights, color="#f39c12", linewidth=2,
             linestyle="--", label="Domain Weight")
    ax2.set_ylabel("Domain Loss Weight", color="#f39c12", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#f39c12")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, fontsize=10)

    ax1.set_title("GRL Schedule: Alpha Ramp & Domain Weight",
                  fontweight="bold")
    ax1.spines["top"].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "05_training_schedule.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {path}")


# ===================================================================
# 6. Time-dependent AUC
# ===================================================================
def plot_time_auc():
    """Time-dependent AUC bar chart."""
    print("\n[6/7] Time-dependent AUC ...")

    experiments = sorted([
        d for d in os.listdir(LIHC_EXP_DIR)
        if (LIHC_EXP_DIR / d / "dann" / "results.json").exists()
        and (LIHC_EXP_DIR / d / "baseline" / "results.json").exists()
    ])

    if not experiments:
        return

    time_points = ["12", "36", "60"]
    n_exp = len(experiments)

    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5))
    if n_exp == 1:
        axes = [axes]

    for ax, exp_key in zip(axes, experiments):
        with open(LIHC_EXP_DIR / exp_key / "dann" / "results.json") as f:
            dann_r = json.load(f)
        with open(LIHC_EXP_DIR / exp_key / "baseline" / "results.json") as f:
            base_r = json.load(f)

        dann_auc = dann_r.get("time_dependent_auc", {})
        base_auc = base_r.get("time_dependent_auc", {})

        x = np.arange(len(time_points))
        width = 0.3

        d_vals = [dann_auc.get(t, None) or 0 for t in time_points]
        b_vals = [base_auc.get(t, None) or 0 for t in time_points]

        bars1 = ax.bar(x - width / 2, d_vals, width, label="DANN", color=COLOR_DANN)
        bars2 = ax.bar(x + width / 2, b_vals, width, label="Baseline", color=COLOR_BASE)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                   f"{bar.get_height():.3f}", ha="center", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                   f"{bar.get_height():.3f}", ha="center", fontsize=7)

        ax.set_title(EXPERIMENT_LABELS.get(exp_key, exp_key), fontweight="bold", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}-month" for t in time_points], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Time-dependent AUC", fontsize=11)
            ax.legend(frameon=True, fontsize=9)

    plt.suptitle("Time-Dependent AUC: DANN vs Baseline", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "06_time_auc_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {path}")


# ===================================================================
# 7. Table Summary
# ===================================================================
def print_summary_table():
    """Print a text summary table of all results."""
    print("\n[7/7] Summary table ...")

    experiments = sorted([
        d for d in os.listdir(LIHC_EXP_DIR)
        if (LIHC_EXP_DIR / d / "comparison.json").exists()
    ])

    if not experiments:
        print("  ⚠ No comparison results found")
        return

    rows = []
    for exp_key in experiments:
        with open(LIHC_EXP_DIR / exp_key / "comparison.json") as f:
            comp = json.load(f)

        name = EXPERIMENT_LABELS.get(exp_key, exp_key).replace("\n", " ")
        dann_c = comp.get("dann_val_cindex", "?")
        base_c = comp.get("baseline_val_cindex", "?")
        delta = comp.get("delta", "?")

        rows.append({
            "Experiment": name,
            "DANN (C-index)": dann_c,
            "Baseline (C-index)": base_c,
            "Δ (DANN - Baseline)": delta,
        })

        # Per-cohort detail
        dann_pc = comp.get("dann_per_cohort", {})
        base_pc = comp.get("baseline_per_cohort", {})
        cohort_map = comp.get("cohort_map", {})
        rev_map = {v: k for k, v in cohort_map.items()}

        for dom_id in sorted(rev_map.keys()):
            cname = rev_map[dom_id]
            dc = dann_pc.get(str(dom_id), "N/A")
            bc = base_pc.get(str(dom_id), "N/A")
            rows.append({
                "Experiment": f"  ├─ {cname}",
                "DANN (C-index)": dc,
                "Baseline (C-index)": bc,
                "Δ (DANN - Baseline)": (
                    f"{float(dc) - float(bc):+.4f}"
                    if dc != "N/A" and bc != "N/A" else "N/A"
                ),
            })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)

    summary_df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    print(f"  → Saved: {OUTPUT_DIR / 'results_summary.csv'}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="LIHC DANN Visualization")
    parser.add_argument("--figures", type=str, default="all",
                       help="Comma-separated: heatmap,tsne,cindex,domain,auc,schedule,table")
    parser.add_argument("--output", type=str, default=None,
                       help="Custom output directory")
    args = parser.parse_args()

    if args.output:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    figures = args.figures.split(",") if args.figures != "all" else [
        "heatmap", "tsne", "cindex", "domain", "schedule", "auc", "table"
    ]

    print("=" * 60)
    print("LIHC DANN Visualization Pipeline")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    figure_map = {
        "heatmap": plot_missingness_heatmap,
        "tsne": plot_tsne_comparison,
        "cindex": plot_cindex_comparison,
        "domain": plot_domain_loss_curves,
        "schedule": plot_training_schedule,
        "auc": plot_time_auc,
        "table": print_summary_table,
    }

    for fig_name in figures:
        fig_name = fig_name.strip()
        if fig_name in figure_map:
            try:
                figure_map[fig_name]()
            except Exception as e:
                print(f"  ✗ {fig_name} failed: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"  ⚠ Unknown figure: {fig_name}")

    print(f"\n✅ All visualizations saved to {OUTPUT_DIR}")
    print("Figures:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
