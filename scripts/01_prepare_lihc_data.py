#!/usr/bin/env python3
"""
01_prepare_lihc_data.py; Prepare LIHC (Liver Cancer) datasets for DANN training
===============================================================================
Merges all LIHC-relevant harmonized files into unified training datasets.

Outputs (in data_processed/):
  - lihc_all_cohorts.csv; All LIHC cohorts with survival data
  - lihc_meta.json; Cohort metadata (sample counts, event rates)
  - lihc_missingness.csv; Per-cohort missingness matrix
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_processed"
OUTPUT_DIR = BASE_DIR / "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# 1. Define LIHC-relevant files
# ===================================================================
LIHC_COHORTS = {
    "TCGA_LIHC": {
        "file": "harmonized_LIHC.csv",
        "source_col": None,  # TCGA files don't have Source column
    },
    "US_SEER": {
        "file": "harmonized_seer.csv",
        "source_col": "Source",
    },
    "hcc_msk_2024": {
        "file": "harmonized_hcc_msk_2024.csv",
        "source_col": "Source",
    },
    "hcc_clca_2024": {
        "file": "harmonized_hcc_clca_2024.csv",
        "source_col": "Source",
    },
    "lihc_amc_prv": {
        "file": "harmonized_lihc_amc_prv.csv",
        "source_col": "Source",
    },
    "hcc_meric_2021": {
        "file": "harmonized_hcc_meric_2021.csv",
        "source_col": "Source",
    },
}

# Base schema: features available in ALL cohorts (incl. SEER)
# We use minimal features for maximum coverage.
# The V3 model handles unknown categorical values via embedding padding_idx.
BASE_CONT_FEATURES = ["Age"]
BASE_CAT_FEATURES = ["Sex", "Stage", "Grade"]

# ===================================================================
# 2. Load & harmonize
# ===================================================================
def load_lihc_cohort(config, cohort_name):
    """Load one cohort file, add Source column, filter to LIHC."""
    filepath = DATA_DIR / config["file"]
    if not filepath.exists():
        print(f"  ⚠ WARNING: {filepath} not found, skipping")
        return None

    df = pd.read_csv(filepath)

    # Add Source column if missing
    if config["source_col"] is None:
        df["Source"] = cohort_name
    else:
        # Some files already have Source; trust it
        if config["source_col"] in df.columns:
            df["Source"] = df[config["source_col"]]
        else:
            df["Source"] = cohort_name

    # Filter to LIHC
    if "Cancer_Type" in df.columns:
        df = df[df["Cancer_Type"] == "LIHC"].copy()

    # Ensure consistent column naming
    df.attrs["cohort"] = cohort_name
    return df


def harmonize_schema(df_list):
    """Harmonize all dataframes to a common minimal schema."""
    common_cols = {"Source", "Patient_ID", "Age", "Sex", "Stage", "Grade",
                   "Survival_Months", "Vital_Status"}
    # Add _missing columns
    for base in ["Age", "Sex", "Stage", "Grade"]:
        common_cols.add(f"{base}_missing")

    harmonized = []
    for df in df_list:
        cohort = df.attrs["cohort"]
        # Keep only common columns that exist
        keep = [c for c in common_cols if c in df.columns]
        sub = df[keep].copy()

        # Ensure all _missing columns exist (fill with 1 if missing)
        for base in ["Age", "Sex", "Stage", "Grade"]:
            col = f"{base}_missing"
            if col not in sub.columns:
                sub[col] = 1  # 1 = missing if column absent entirely

        # Fill NaN in actual feature columns appropriately
        # For categoricals: keep NaN (model handles via embedding)
        # For continuous: keep NaN (will be imputed later)

        harmonized.append(sub)
        print(f"  ✓ {cohort}: {len(sub)} rows, "
              f"event_rate={sub['Vital_Status'].dropna().mean():.3f}" if sub['Vital_Status'].notna().any() else f"  ✓ {cohort}: {len(sub)} rows")

    return pd.concat(harmonized, ignore_index=True)


def filter_valid_survival(df):
    """Remove rows without valid survival data."""
    before = len(df)
    df = df[df["Survival_Months"].notna()].copy()
    df = df[df["Survival_Months"] > 0].copy()
    df = df[df["Vital_Status"].notna()].copy()
    df = df[df["Vital_Status"].isin([0, 1])].copy()
    print(f"  After survival filter: {len(df)} / {before} rows kept")
    return df


def compute_missingness(df):
    """Compute per-source missingness matrix."""
    features = ["Age", "Sex", "Stage", "Grade", "Survival_Months", "Vital_Status"]
    rows = []
    for source in sorted(df["Source"].unique()):
        sub = df[df["Source"] == source]
        row = {"Source": source, "N": len(sub)}
        for feat in features:
            if feat in sub.columns:
                pct = sub[feat].isna().mean() * 100
            else:
                pct = 100.0
            row[feat] = round(pct, 1)
        row["Event_Rate"] = round(sub["Vital_Status"].mean() * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================
# 3. Main
# ===================================================================
def main():
    print("=" * 60)
    print("STEP 1: Prepare LIHC Training Data")
    print("=" * 60)

    # Load all cohorts
    print("\n[1/4] Loading LIHC cohorts ...")
    loaded = []
    for name, cfg in LIHC_COHORTS.items():
        df = load_lihc_cohort(cfg, name)
        if df is not None and len(df) > 0:
            loaded.append(df)
            print(f"  ✓ {name}: {len(df)} rows loaded")
        else:
            print(f"  ✗ {name}: skipped (empty or missing)")

    if not loaded:
        print("ERROR: No LIHC data loaded!")
        sys.exit(1)

    # Harmonize
    print(f"\n[2/4] Harmonizing {len(loaded)} cohorts to common schema ...")
    combined = harmonize_schema(loaded)

    # Filter valid survival
    print(f"\n[3/4] Filtering valid survival data ...")
    combined = filter_valid_survival(combined)

    # Report
    print(f"\n[4/4] Final dataset summary:")
    counts = combined["Source"].value_counts()
    for src, n in counts.items():
        sub = combined[combined["Source"] == src]
        print(f"  {src:20s}: {n:>6d} rows, "
              f"events={int(sub['Vital_Status'].sum()):>5d}, "
              f"Age={sub['Age'].mean():.1f}±{sub['Age'].std():.1f}")

    print(f"\n  TOTAL: {len(combined)} rows, {combined['Source'].nunique()} cohorts")

    # Compute missingness
    missing_df = compute_missingness(combined)
    print("\nMissingness matrix:")
    print(missing_df.to_string(index=False))

    # Save
    out_path = DATA_DIR / "lihc_all_cohorts.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  → Saved: {out_path}")

    miss_path = OUTPUT_DIR / "lihc_missingness.csv"
    missing_df.to_csv(miss_path, index=False)
    print(f"  → Saved: {miss_path}")

    # Save metadata
    meta = {
        "total_samples": len(combined),
        "n_cohorts": combined["Source"].nunique(),
        "cohorts": counts.to_dict(),
        "features": {
            "continuous": BASE_CONT_FEATURES,
            "categorical": BASE_CAT_FEATURES,
        },
    }
    meta_path = OUTPUT_DIR / "lihc_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  → Saved: {meta_path}")

    print("\n✅ Data preparation complete!")
    return meta


if __name__ == "__main__":
    main()
