#!/usr/bin/env python3
"""
07_exp_g_shap.py; SHAP Interpretability for Experiment A (Baseline vs DANN)
==============================================================================
Goal: does the adversarial head (GRL) destroy the model's attention on key
clinical features (Stage, Age, Grade, Sex)?

Approach
--------
  - Reconstruct Exp A (TCGA_LIHC vs US_SEER) exactly as the training pipeline.
  - Reimplement the survival path of TransDANNSurvV3 with SMOOTH inputs
    (Age continuous + one-hot encoded categoricals) so gradient-based SHAP
    (Expected Gradients) is well-defined.
  - Validate the smooth net reproduces the real model's risk.
  - Compute SHAP on the held-out val/test set for both Baseline and DANN.
  - Aggregate per-feature SHAP by summing one-hot columns; draw beeswarm
    plots + a grouped feature-importance comparison.

Outputs (results/figures/extended/):
  G1_shap_beeswarm_baseline.png
  G2_shap_beeswarm_dann.png
  G3_shap_importance_compare.png
  results/G_shap_results.json
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lihc_recon import reconstruct_full_data, _load_train_02
from transdann_utils import TransDANNSurvV3, deep_hit_risk, DEVICE

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = BASE_DIR / "results" / "lihc_experiments" / "A_tcga_vs_seer"
FIG_DIR = BASE_DIR / "results" / "figures" / "extended"
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = ["Age", "Sex", "Stage", "Grade"]

# Semantic / display ordinals (raw values are LabelEncoder indices)
STAGE_ORD = {
    "0": 0.0,
    "1": 1.0, "1A": 1.0, "1B": 1.0, "I": 1.0,
    "2": 2.0, "II": 2.0,
    "3A": 3.0, "3B": 3.0, "III": 3.0, "IIIA": 3.0,
    "IIIB": 3.0, "IIIC": 3.0, "IIINOS": 3.0,
    "4A": 4.0, "4B": 4.0, "IV": 4.0, "IVA": 4.0, "IVB": 4.0,
    # "-1" / "99" / "UNK Stage"  -> NaN (missing)
}
GRADE_ORD = {"1.0": 1.0, "2.0": 2.0, "3.0": 3.0, "4.0": 4.0}
SEX_ORD = {"0.0": 0.0, "1.0": 1.0}


class SmoothSurvNet(nn.Module):
    """One-hot / continuous inputs -> scalar risk, using the trained weights.

    Faithfully mirrors TransDANNSurvV3.forward's survival path but accepts
    smooth one-hot categorical vectors instead of discrete token indices, so
    gradient-based SHAP (Expected Gradients) is differentiable.
    """

    def __init__(self, base, cat_cards, bin_centers):
        super().__init__()
        self.base = base
        self.cat_cards = cat_cards
        self.register_buffer(
            "bin_centers", bin_centers.to(next(base.parameters()).device))

    def forward(self, x):
        B = x.size(0)
        tokens = [self.base.cls_token.expand(B, -1, -1)]

        # Continuous feature(s): Age (scaled)  -- 1 column
        ce = self.base.cont_embeddings[0](x[:, 0:1])          # (B, d)
        tokens.append(ce.unsqueeze(1))

        # Categorical features: one-hot blocks -> weighted sum of embeddings
        off = 1
        for i, card in enumerate(self.cat_cards):
            oh = x[:, off:off + card]
            off += card
            w = self.base.cat_embeddings[i].weight[:card]     # (card, d)
            tokens.append(torch.matmul(oh, w).unsqueeze(1))

        x = torch.cat(tokens, dim=1)
        x = x + self.base.pos_encoding[:, :x.size(1), :]
        x = self.base.transformer(x)
        x = self.base.layer_norm(x)
        cls = x[:, 0, :]
        surv = self.base.survival_head(cls)
        h = F.softmax(surv, dim=1)
        risk = -torch.sum(h * self.bin_centers, dim=1)        # risk = -E[lifetime]
        return risk.reshape(-1, 1)                            # SHAP expects (B, out)


def load_model_and_data():
    """Reconstruct Exp A data, load both trained models. Returns (models, data, val_df)."""
    # Reconstruct the exact full pipeline instead.
    full_df, times, events, domains, sources = reconstruct_full_data("A_tcga_vs_seer")

    # Rebuild data dict (encoders / scaler / bin_centers / split) identically.
    df = pd.read_csv(BASE_DIR / "data_processed" / "lihc_all_cohorts.csv")
    df = df[(df["Survival_Months"] > 0) & (df["Vital_Status"].isin([0, 1]))]
    exp_df = df[df["Source"].isin(["TCGA_LIHC", "US_SEER"])].copy()
    data = _load_train_02().load_and_preprocess(
        exp_df, surv_type="deephit", n_bins=32, subsample_seer=10000)

    n_train = len(data["train_df"])
    val_df = full_df.iloc[n_train:].reset_index(drop=True)
    assert len(val_df) == len(data["val_df"]), "val block mismatch"

    # Instantiate both models with the same architecture.
    cfg = {
        "d_model": 128, "n_layers": 4, "dropout": 0.15, "n_bins": 32,
    }
    models = {}
    for mode in ("baseline", "dann"):
        m = TransDANNSurvV3(
            num_continuous=len(data["cont_features"]),
            num_categorical=len(data["cat_features"]),
            cat_cardinalities=data["cat_cardinalities"],
            d_model=cfg["d_model"], n_heads=8, n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], num_domains=data["num_domains"],
            surv_head_type="deephit", n_bins=cfg["n_bins"],
        ).to(DEVICE)
        m.load_state_dict(torch.load(
            EXP_DIR / mode / "best_model.pth", map_location=DEVICE, weights_only=True))
        m.eval()
        models[mode] = m
    return models, data, val_df, times, events, sources


def build_smooth_inputs(val_df, data):
    """Concatenate [Age (scaled), Sex-oh, Stage-oh, Grade-oh]."""
    X = [val_df["Age"].values.astype(np.float32).reshape(-1, 1)]
    for col, card in zip(data["cat_features"], data["cat_cardinalities"]):
        v = val_df[col].values.astype(np.int64)
        v = np.clip(v, 0, card - 1)
        oh = np.zeros((len(v), card), dtype=np.float32)
        oh[np.arange(len(v)), v] = 1.0
        X.append(oh)
    return np.concatenate(X, axis=1)


def display_values(val_df, data):
    """Semantic feature values for beeswarm colouring (Age=years, Stage/grade ordinal)."""
    n = len(val_df)
    disp = np.full((n, 4), np.nan, dtype=float)
    # Age: inverse-scale back to years
    age_scaled = val_df["Age"].values.astype(float).reshape(-1, 1)
    age_raw = data["scaler"].inverse_transform(age_scaled).ravel()
    disp[:, 0] = age_raw
    for j, col in enumerate(data["cat_features"]):
        le = data["label_encoders"][col]
        ord_map = {"Sex": SEX_ORD, "Stage": STAGE_ORD, "Grade": GRADE_ORD}[col]
        for idx, cls in enumerate(le.classes_):
            if cls in ord_map:
                mask = val_df[col].values == idx
                disp[mask, j + 1] = ord_map[cls]
    return disp


def main():
    models, data, val_df, times, events, sources = load_model_and_data()
    X = build_smooth_inputs(val_df, data)
    disp = display_values(val_df, data)
    cat_cards = list(data["cat_cardinalities"])
    print(f"Val set: {len(val_df)} samples | smooth dim = {X.shape[1]} | cat_cards={cat_cards}")

    # Mean SHAP over features & expected value (for validation of smooth net)
    results = {"n_val": int(len(val_df)), "features": FEATURES,
               "cat_cardinalities": cat_cards, "models": {}}

    shap_by_mode = {}
    for mode in ("baseline", "dann"):
        model = models[mode]
        smooth = SmoothSurvNet(model, cat_cards, data["bin_centers"]).to(DEVICE)

        # ---- sanity: smooth net must reproduce the real model's risk ----
        with torch.no_grad():
            x_cat = torch.tensor(
                val_df[data["cat_features"]].values, dtype=torch.long).to(DEVICE)
            x_cont = torch.tensor(
                val_df[data["cont_features"]].values, dtype=torch.float32).to(DEVICE)
            surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
            risk_real = deep_hit_risk(surv_out, data["bin_centers"]).cpu().numpy()
            risk_smooth = smooth(torch.tensor(X).to(DEVICE)).cpu().numpy().ravel()
        diff = float(np.max(np.abs(risk_real - risk_smooth)))
        print(f"[{mode}] max|risk_real - risk_smooth| = {diff:.6f}")
        assert diff < 1e-3, f"Smooth net mismatch for {mode}"

        # ---- SHAP: Expected Gradients on the val/test set ----
        X_t = torch.tensor(X).to(DEVICE)
        bg_idx = np.random.RandomState(0).choice(len(X), size=min(200, len(X)), replace=False)
        bg_t = torch.tensor(X[bg_idx]).to(DEVICE)
        with torch.no_grad():
            base_value = float(smooth(bg_t).mean().item())
        explainer = shap.GradientExplainer(smooth, bg_t)
        sv = explainer.shap_values(X_t, nsamples=100)      # (n, dim, out) | (n, dim)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 0]                                 # take the single output

        # ---- aggregate one-hot columns -> per-feature SHAP ----
        agg = np.zeros((len(X), 4))
        agg[:, 0] = sv[:, 0]
        off = 1
        for j, card in enumerate(cat_cards):
            agg[:, j + 1] = sv[:, off:off + card].sum(axis=1)
            off += card
        shap_by_mode[mode] = agg

        # ---- summary statistics ----
        mean_abs = np.abs(agg).mean(axis=0)
        feats_sorted = [f for _, f in sorted(zip(mean_abs, FEATURES), reverse=True)]
        # mean SHAP of Age with respect to risk direction
        results["models"][mode] = {
            "mean_abs_shap": {f: round(float(v), 5) for f, v in zip(FEATURES, mean_abs)},
            "feature_rank": feats_sorted,
            "base_value": round(base_value, 5),
            "risk_mean": round(float(risk_real.mean()), 5),
            "shap_sum_additivity": round(
                float(np.mean(np.abs(agg.sum(axis=1) - (risk_real - base_value)))), 5,
            ),
            # per-cohort mean|SHAP| (SEER vs TCGA)
            "cohort_mean_abs": {},
        }
        for src in ("TCGA_LIHC", "US_SEER"):
            mask = (val_df["Source"] == src).values
            if mask.sum() < 5:
                continue
            results["models"][mode]["cohort_mean_abs"][src] = {
                f: round(float(np.abs(agg[mask, j]).mean()), 5)
                for j, f in enumerate(FEATURES)
            }
        print(f"[{mode}] mean|SHAP|: " +
              ", ".join(f"{f}={v:.4f}" for f, v in zip(FEATURES, mean_abs)))

    # ---- Beeswarm plots ----
    disp_filled = disp.copy()
    for j in range(disp_filled.shape[1]):
        col = disp_filled[:, j]
        col[np.isnan(col)] = np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
    for mode in ("baseline", "dann"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        shap.summary_plot(
            shap_by_mode[mode], disp_filled, feature_names=FEATURES,
            show=False, max_display=4)
        plt.title(f"SHAP Beeswarm; {mode.upper()} (Exp A val set)\n"
                  f"risk = negative expected lifetime; higher SHAP => higher risk")
        plt.tight_layout()
        out = FIG_DIR / f"G{'1' if mode=='baseline' else '2'}_shap_beeswarm_{mode}.png"
        plt.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved -> {out}")

    # ---- Grouped feature-importance bar (Baseline vs DANN) ----
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    y = np.arange(len(FEATURES))[::-1]
    h = 0.35
    b_base = [results["models"]["baseline"]["mean_abs_shap"][f] for f in FEATURES]
    b_dann = [results["models"]["dann"]["mean_abs_shap"][f] for f in FEATURES]
    ax.barh(y - h / 2, b_base, height=h, color="#4C72B0", label="Baseline")
    ax.barh(y + h / 2, b_dann, height=h, color="#C44E52", label="DANN")
    ax.set_yticks(y)
    ax.set_yticklabels(FEATURES)
    ax.set_xlabel("Mean |SHAP| (feature importance)")
    ax.set_title("Feature importance on the Exp A val set:\nBaseline vs DANN")
    for yy, bb, dd in zip(y, b_base, b_dann):
        ax.text(bb, yy - h / 2, f"{bb:.4f}", va="center", fontsize=8)
        ax.text(dd, yy + h / 2, f"{dd:.4f}", va="center", fontsize=8)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "G3_shap_importance_compare.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")

    with open(BASE_DIR / "results" / "G_shap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {BASE_DIR / 'results' / 'G_shap_results.json'}")


if __name__ == "__main__":
    main()
