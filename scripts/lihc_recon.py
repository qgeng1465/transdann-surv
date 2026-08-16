#!/usr/bin/env python3
"""
lihc_recon.py; Shared data reconstruction / preparation for experiments D, E, F.
==================================================================================
  - reconstruct_full_data(exp_key) : reproduces the EXACT train+val ordering the
    original pipeline used, so saved risks.npy line up with times/events/domains.
  - build_data(...)                : generic version of 02's load_and_preprocess
    with a configurable feature set (used by Exp D imputation & Exp E clean data).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transdann_utils import (
    compute_time_bins, LiverCancerDataset,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data_processed" / "lihc_all_cohorts.csv"

# Mirror of the experiment definitions in 02_train_dann_lihc.py
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


def _base_filter(df):
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    return df


def _load_module(filename, alias):
    """Load a sibling script via importlib (filenames start with digits)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        alias, str(Path(__file__).resolve().parent / filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_train_02():
    return _load_module("02_train_dann_lihc.py", "train_02")


def _load_exp_d():
    return _load_module("05_exp_d_imputation.py", "exp_d_05")


def _load_exp_e():
    return _load_module("06_exp_e_clean_control.py", "exp_e_06")


def reconstruct_full_data(exp_key, subsample_seer=10000):
    """Reproduce the exact full_df (train+val, in evaluation order) for exp_key.

    Returns (full_df, times, events, domains, sources).
    """
    load_and_preprocess = _load_train_02().load_and_preprocess
    df = pd.read_csv(DATA_PATH)
    df = _base_filter(df)
    exp_cfg = EXPERIMENTS[exp_key]
    exp_df = df[df["Source"].isin(exp_cfg["cohorts"])].copy()
    data = load_and_preprocess(
        exp_df, surv_type="deephit", n_bins=32, subsample_seer=subsample_seer,
    )
    full_df = pd.concat([data["train_df"], data["val_df"]], ignore_index=True)
    times = full_df["Survival_Months"].values.astype(float)
    events = full_df["Vital_Status"].values.astype(float)
    domains = full_df["Domain_Label"].values.astype(int)
    sources = full_df["Source"].values
    return full_df, times, events, domains, sources


def reconstruct_exp_d(subsample_seer=10000):
    """Reproduce Exp D's full_df (imputed TCGA+SEER, train+val order)."""
    mod = _load_exp_d()
    raw = mod.load_raw()
    imputed = mod.impute_knn(raw)
    data = build_data(imputed, CONT_FEATURES, CAT_FEATURES, n_bins=32,
                      subsample_seer=subsample_seer)
    full_df = pd.concat([data["train_df"], data["val_df"]], ignore_index=True)
    times = full_df["Survival_Months"].values.astype(float)
    events = full_df["Vital_Status"].values.astype(float)
    domains = full_df["Domain_Label"].values.astype(int)
    sources = full_df["Source"].values
    return full_df, times, events, domains, sources


def reconstruct_exp_e():
    """Reproduce Exp E's full_df (harmonized TCGA-BRCA + METABRIC, train+val order)."""
    mod = _load_exp_e()
    df = mod.load_breast()
    data = build_data(df, ["Age"], ["Sex", "Stage"], n_bins=32)
    full_df = pd.concat([data["train_df"], data["val_df"]], ignore_index=True)
    times = full_df["Survival_Months"].values.astype(float)
    events = full_df["Vital_Status"].values.astype(float)
    domains = full_df["Domain_Label"].values.astype(int)
    sources = full_df["Source"].values
    return full_df, times, events, domains, sources


def build_data(df, cont_features, cat_features, surv_type="deephit", n_bins=32,
               subsample_seer=None, seed=42, val_frac=0.15, cohort_col="Source"):
    """Generic data-prep: returns the same dict shape train_model() expects.

    `df` must already be imputed/corrected for missingness by the caller.
    """
    df = df.copy()

    # Optional cohort subsampling (mirrors 02's SEER downsampling)
    if subsample_seer and "US_SEER" in df[cohort_col].values:
        seer_idx = df[df[cohort_col] == "US_SEER"].index
        if len(seer_idx) > subsample_seer:
            keep = np.random.RandomState(42).choice(seer_idx, subsample_seer, replace=False)
            df = df.drop(seer_idx.difference(pd.Index(keep))).copy()

    # Continuous: fill any residual NaN with median (defensive)
    for col in cont_features:
        med = df[col].median()
        if pd.isna(med):
            med = df[col].dropna().median() if df[col].notna().any() else 50.0
        df[col] = df[col].fillna(med)

    # Categorical: encode, NaN → placeholder (handled by Embedding padding_idx)
    label_encoders, cat_cardinalities = {}, []
    for col in cat_features:
        df[col] = df[col].fillna(-1).astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))

    scaler = StandardScaler()
    df[cont_features] = scaler.fit_transform(df[cont_features])

    cohorts = sorted(df[cohort_col].unique())
    cohort_map = {c: i for i, c in enumerate(cohorts)}
    df["Domain_Label"] = df[cohort_col].map(cohort_map)
    num_domains = len(cohorts)

    # Stratified train/val split (same recipe as 02)
    np.random.seed(seed)
    val_idx, train_idx = [], []
    for cohort in df[cohort_col].unique():
        cidx = df[df[cohort_col] == cohort].index.tolist()
        np.random.shuffle(cidx)
        n_val = max(1, int(len(cidx) * val_frac))
        val_idx.extend(cidx[:n_val])
        train_idx.extend(cidx[n_val:])
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)

    bin_edges = bin_centers = None
    if surv_type == "deephit":
        bin_edges, bin_centers = compute_time_bins(
            torch.tensor(train_df["Survival_Months"].values),
            torch.tensor(train_df["Vital_Status"].values),
            n_bins=n_bins,
        )

    train_ds = LiverCancerDataset(train_df, cont_features, cat_features, "Domain_Label")
    val_ds = LiverCancerDataset(val_df, cont_features, cat_features, "Domain_Label")

    domain_counts = train_df["Domain_Label"].value_counts().sort_index()
    w = 1.0 / np.sqrt(domain_counts.values + 1)
    sw = w[train_df["Domain_Label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sw), num_samples=len(sw), replacement=True
    )
    batch_size = min(512, max(64, len(train_df) // 4))
    batch_size = min(batch_size, len(train_df))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_df": train_df,
        "val_df": val_df,
        "cont_features": cont_features,
        "cat_features": cat_features,
        "cat_cardinalities": cat_cardinalities,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "num_domains": num_domains,
        "cohort_map": cohort_map,
    }
