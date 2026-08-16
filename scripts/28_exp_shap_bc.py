#!/usr/bin/env python3
"""
28_exp_shap_bc.py; SHAP attributions for experiments B and C
=============================================================

Exp G (07_exp_g_shap.py) showed on experiment A (TCGA vs SEER) that the GRL
domain-adversarial head does not break the model's attention on the key clinical
features (Stage/Age/Grade/Sex; Stage attention +26%). This script applies the same
smooth-inputs + Expected-Gradients SHAP protocol to the validation sets of
experiment B (4 domains) and C (5 domains), testing whether the "adversarial
decoupling does not break clinical-feature attention" conclusion is robust in the
**multi-domain** setting.

Outputs (results/experiments/exp_shap_bc/):
  - B_tcga_vs_external_shap.json / C_all_cohorts_shap.json
  - summary.txt
  - results/figures/extended/G4_shap_importance_compare_B.png
  - results/figures/extended/G5_shap_importance_compare_C.png
  - results/tables/table15_shap_bc.csv
  - logs/28_exp_shap_bc.log

Usage:
  python3 scripts/28_exp_shap_bc.py            # B + C
  python3 scripts/28_exp_shap_bc.py --exps B   # run B only
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"

EXP_OUT = RESULTS_DIR / "experiments" / "exp_shap_bc"
LOG_DIR = EXP_OUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(EXP_OUT, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "28_exp_shap_bc.log", mode="w"),
    ],
)
log = logging.getLogger("shap_bc")

# ---------------------------------------------------------------------------
# Reuse the smooth SHAP machinery from Exp G (07)
# ---------------------------------------------------------------------------
spec07 = importlib.util.spec_from_file_location(
    "exp_g_07", SCRIPTS_DIR / "07_exp_g_shap.py"
)
exp_g07 = importlib.util.module_from_spec(spec07)
spec07.loader.exec_module(exp_g07)
SmoothSurvNet = exp_g07.SmoothSurvNet
build_smooth_inputs = exp_g07.build_smooth_inputs
display_values = exp_g07.display_values
FEATURES = exp_g07.FEATURES

from transdann_utils import (  # noqa: E402
    DEVICE, TransDANNSurvV3, deep_hit_risk,
)
from lihc_recon import _load_train_02  # noqa: E402
import shap  # noqa: E402

DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"
EXP_KEYS = {
    "B": "B_tcga_vs_external",
    "C": "C_all_cohorts",
}
FIG_DIR = RESULTS_DIR / "figures" / "extended"
os.makedirs(FIG_DIR, exist_ok=True)


def run_shap(exp_key, exp_df):
    data = _load_train_02().load_and_preprocess(
        exp_df, surv_type="deephit", n_bins=32, subsample_seer=10000)
    val_df = data["val_df"].copy()
    n_val = len(val_df)
    log.info(f"[{exp_key}] val set: {n_val} samples | "
             f"cohorts={sorted(set(val_df['Source']))}")

    X = build_smooth_inputs(val_df, data)
    disp = display_values(val_df, data)
    cat_cards = list(data["cat_cardinalities"])
    log.info(f"[{exp_key}] smooth dim = {X.shape[1]} | cat_cards={cat_cards}")

    exp_dir = RESULTS_DIR / "lihc_experiments" / exp_key
    results = {"experiment": exp_key, "n_val": n_val, "features": FEATURES,
               "cat_cardinalities": cat_cards, "models": {}}

    shap_by_mode = {}
    for mode in ("baseline", "dann"):
        model = TransDANNSurvV3(
            num_continuous=len(data["cont_features"]),
            num_categorical=len(data["cat_features"]),
            cat_cardinalities=data["cat_cardinalities"],
            d_model=128, n_heads=8, n_layers=4, dropout=0.15,
            num_domains=data["num_domains"], surv_head_type="deephit", n_bins=32,
        ).to(DEVICE)
        ckpt = exp_dir / mode / "best_model.pth"
        if not ckpt.exists():
            log.error(f"  [{exp_key}] missing checkpoint {ckpt}; skipping {mode}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()

        smooth = SmoothSurvNet(model, cat_cards, data["bin_centers"]).to(DEVICE)

        # sanity: smooth net reproduces the real model's risk
        with torch.no_grad():
            x_cat = torch.tensor(val_df[data["cat_features"]].values,
                                 dtype=torch.long).to(DEVICE)
            x_cont = torch.tensor(val_df[data["cont_features"]].values,
                                  dtype=torch.float32).to(DEVICE)
            surv_out, _, _ = model(x_cont, x_cat, alpha=0.0)
            risk_real = deep_hit_risk(surv_out, data["bin_centers"]).cpu().numpy()
            risk_smooth = smooth(torch.tensor(X).to(DEVICE)).cpu().numpy().ravel()
        diff = float(np.max(np.abs(risk_real - risk_smooth)))
        assert diff < 1e-3, f"smooth-net mismatch for {exp_key}/{mode}: {diff}"
        log.info(f"[{exp_key}] [{mode}] max|risk_real − risk_smooth| = {diff:.6f}")

        X_t = torch.tensor(X).to(DEVICE)
        bg_idx = np.random.RandomState(0).choice(
            len(X), size=min(200, len(X)), replace=False)
        bg_t = torch.tensor(X[bg_idx]).to(DEVICE)
        with torch.no_grad():
            base_value = float(smooth(bg_t).mean().item())
        explainer = shap.GradientExplainer(smooth, bg_t)
        sv = explainer.shap_values(X_t, nsamples=100)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 0]

        agg = np.zeros((len(X), len(FEATURES)))
        agg[:, 0] = sv[:, 0]
        off = 1
        for j, card in enumerate(cat_cards):
            agg[:, j + 1] = sv[:, off:off + card].sum(axis=1)
            off += card
        shap_by_mode[mode] = agg
        # per-sample SHAP (n_val × n_features); enables bootstrap CIs later
        np.save(EXP_OUT / f"{exp_key}_agg_{mode}.npy", agg)

        mean_abs = np.abs(agg).mean(axis=0)
        feats_sorted = [f for _, f in sorted(zip(mean_abs, FEATURES), reverse=True)]
        results["models"][mode] = {
            "mean_abs_shap": {f: round(float(v), 5)
                              for f, v in zip(FEATURES, mean_abs)},
            "feature_rank": feats_sorted,
            "base_value": round(base_value, 5),
            "risk_mean": round(float(risk_real.mean()), 5),
            "shap_sum_additivity": round(float(np.mean(
                np.abs(agg.sum(axis=1) - (risk_real - base_value)))), 5),
            "cohort_mean_abs": {},
        }
        for src in sorted(set(val_df["Source"])):
            mask = (val_df["Source"] == src).values
            if mask.sum() < 5:
                continue
            results["models"][mode]["cohort_mean_abs"][src] = {
                f: round(float(np.abs(agg[mask, j]).mean()), 5)
                for j, f in enumerate(FEATURES)
            }
        log.info(f"[{exp_key}] [{mode}] mean|SHAP|: "
                 + ", ".join(f"{f}={v:.4f}"
                             for f, v in zip(FEATURES, mean_abs)))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "baseline" in results["models"] and "dann" in results["models"]:
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
        ax.set_title(f"Feature importance on {exp_key} val set:\nBaseline vs DANN")
        for yy, bb, dd in zip(y, b_base, b_dann):
            ax.text(bb, yy - h / 2, f"{bb:.4f}", va="center", fontsize=8)
            ax.text(dd, yy + h / 2, f"{dd:.4f}", va="center", fontsize=8)
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        letter = "B" if exp_key == "B_tcga_vs_external" else "C"
        out = FIG_DIR / f"G{4 if letter == 'B' else 5}_shap_importance_compare_{letter}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info(f"Figure → {out}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SHAP for experiments B and C")
    parser.add_argument("--exps", nargs="+", default=list(EXP_KEYS), choices=list(EXP_KEYS))
    parser.add_argument("--summarize-only", action="store_true",
                        help="re-aggregate summary.txt + table15 from saved per-exp "
                             "JSONs (no SHAP recompute)")
    args = parser.parse_args()

    if args.summarize_only:
        all_results = {}
        for letter in args.exps:
            ek = EXP_KEYS[letter]
            p = EXP_OUT / f"{ek}_shap.json"
            if not p.exists():
                log.error(f"missing {p}; run without --summarize-only first")
                sys.exit(1)
            all_results[ek] = json.load(open(p))
        summarize(all_results)
        log.info("\n✅ SHAP B/C re-aggregation complete (no recompute)!")
        return

    df = pd.read_csv(DATA_PATH)
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    t02 = _load_train_02()

    all_results = {}
    for letter in args.exps:
        ek = EXP_KEYS[letter]
        exp_cfg = t02.EXPERIMENTS[ek]
        exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
        log.info(f"EXPERIMENT {letter}: {ek}; {len(exp_df)} samples, "
                 f"cohorts={exp_cfg['cohorts']}")
        all_results[ek] = run_shap(ek, exp_df)
        with open(EXP_OUT / f"{ek}_shap.json", "w") as f:
            json.dump(all_results[ek], f, indent=2)

    summarize(all_results)
    log.info("\n✅ SHAP B/C complete!")


def summarize(all_results):
    """Aggregate summary.txt + table15 from per-experiment SHAP results."""
    rows = []
    lines = ["\n" + "=" * 90,
             "SHAP INTERPRETABILITY; EXPERIMENTS B & C (Baseline vs DANN)",
             "=" * 90]
    for ek, res in all_results.items():
        lines.append(f"\n{ek} (n_val={res['n_val']}):")
        for mode in ("baseline", "dann"):
            if mode not in res["models"]:
                continue
            m = res["models"][mode]
            lines.append(f"  {mode:<8} rank={m['feature_rank']}")
            lines.append(f"          mean|SHAP| " + ", ".join(
                f"{f}={m['mean_abs_shap'][f]:.4f}" for f in FEATURES))
            for cname, cm in m["cohort_mean_abs"].items():
                lines.append(f"          {cname:<18}" + ", ".join(
                    f"{f}={cm[f]:.3f}" for f in FEATURES))
            rows.append({"experiment": ek, "mode": mode,
                         "feature_rank": str(m["feature_rank"]),
                         **{f"mean_abs_{f}": m["mean_abs_shap"][f]
                            for f in FEATURES}})
        if "baseline" in res["models"] and "dann" in res["models"]:
            d = {f: round(res["models"]["dann"]["mean_abs_shap"][f]
                          - res["models"]["baseline"]["mean_abs_shap"][f], 5)
                 for f in FEATURES}
            lines.append(f"  Δ(DANN−Base) " + ", ".join(
                f"{f}={d[f]:+.4f}" for f in FEATURES))
            rows.append({"experiment": ek, "mode": "delta_dann_minus_baseline",
                         **{f"mean_abs_{f}": d[f] for f in FEATURES}})
    lines.append("=" * 90)
    txt = "\n".join(lines)
    log.info(txt)
    with open(EXP_OUT / "summary.txt", "w") as f:
        f.write(txt + "\n")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "table15_shap_bc.csv",
                              index=False)
    log.info(f"Table → {RESULTS_DIR / 'tables' / 'table15_shap_bc.csv'} "
             f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
