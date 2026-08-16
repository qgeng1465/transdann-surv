#!/usr/bin/env python3
"""
SEER Clinical Data Processor
=============================
Processes SEER (Surveillance, Epidemiology, and End Results) data
from the archived cohort into the unified schema format with missing-value
indicators, matching the harmonized TCGA/cBioPortal output format.

Input:  archive/data/old_seer/seer_cohort.csv
Output: data_processed/harmonized_seer.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / 'archive' / 'data' / 'old_seer' / 'seer_cohort.csv'
OUTPUT_PATH = BASE_DIR / 'data_processed' / 'harmonized_seer.csv'

UNIFIED_FEATURE_COLS = [
    'Age', 'Sex', 'Stage', 'Grade', 'Survival_Months', 'Vital_Status'
]


def parse_age(raw):
    """Parse '77 years' → 77.0, '80 years' → 80.0."""
    if pd.isna(raw) or str(raw).strip() in ('', 'Blank(s)'):
        return np.nan
    m = re.search(r'(\d+)', str(raw))
    return float(m.group(1)) if m else np.nan


def parse_sex(raw):
    """Female → 1, Male → 0."""
    if pd.isna(raw) or str(raw).strip() in ('', 'Blank(s)'):
        return np.nan
    s = str(raw).strip().lower()
    return 1 if s.startswith('f') else (0 if s.startswith('m') else np.nan)


def parse_stage(raw_ajcc, raw_eod):
    """Try AJCC 7th first, fall back to EOD 2018."""
    for val in [raw_ajcc, raw_eod]:
        if pd.notna(val) and str(val).strip() not in ('', 'Blank(s)'):
            return str(val).strip()
    return np.nan


def parse_survival_months(raw):
    """Parse zero-padded 4-digit string → int."""
    if pd.isna(raw) or str(raw).strip() in ('', 'Blank(s)'):
        return np.nan
    try:
        return float(str(raw).strip())
    except (ValueError, TypeError):
        return np.nan


def parse_vital_status(raw):
    """Dead → 1, Alive → 0."""
    if pd.isna(raw) or str(raw).strip() in ('', 'Blank(s)'):
        return np.nan
    s = str(raw).strip().lower()
    return 1 if s == 'dead' else (0 if s == 'alive' else np.nan)


def main():
    print(f'Reading SEER data from: {INPUT_PATH}')
    df = pd.read_csv(INPUT_PATH, dtype=str)
    print(f'  Rows: {len(df)}')
    print(f'  Columns: {list(df.columns)}')

    # Map column names
    col_map = {
        'Age recode with single ages and 90+': 'Age_raw',
        'Sex': 'Sex_raw',
        'Derived AJCC Stage Group, 7th ed (2010-2015)': 'Stage_AJCC',
        'Derived EOD 2018 Stage Group Recode (2018+)': 'Stage_EOD',
        'Survival months': 'Survival_Months_raw',
        'Vital status recode (study cutoff used)': 'Vital_Status_raw',
    }

    df_work = df.rename(columns=col_map)
    records = []

    for _, row in df_work.iterrows():
        record = {
            'Source': 'US_SEER',
            'Cancer_Type': 'LIHC',
            'Patient_ID': row.get('Patient_ID', np.nan),
        }
        record['Age'] = parse_age(row.get('Age_raw', np.nan))
        record['Sex'] = parse_sex(row.get('Sex_raw', np.nan))
        record['Stage'] = parse_stage(
            row.get('Stage_AJCC', np.nan),
            row.get('Stage_EOD', np.nan)
        )
        # SEER does not have Grade information
        record['Grade'] = np.nan
        record['Survival_Months'] = parse_survival_months(
            row.get('Survival_Months_raw', np.nan)
        )
        record['Vital_Status'] = parse_vital_status(
            row.get('Vital_Status_raw', np.nan)
        )
        records.append(record)

    out = pd.DataFrame(records)

    # Add _missing indicators for each feature
    for col in UNIFIED_FEATURE_COLS:
        missing_col = f'{col}_missing'
        if col in out.columns:
            out[missing_col] = out[col].isna().astype(int)

    # Reorder columns: features first, then their _missing counterparts
    feature_cols = [c for c in UNIFIED_FEATURE_COLS if c in out.columns]
    missing_cols = [f'{c}_missing' for c in feature_cols]
    id_cols = ['Source', 'Cancer_Type', 'Patient_ID']
    out = out[id_cols + feature_cols + missing_cols]

    # Summary
    print(f'\nOutput shape: {out.shape}')
    print(f'\nMissingness summary:')
    for col in feature_cols:
        n_miss = out[col].isna().sum()
        print(f'  {col:20s}: {n_miss:6d} / {len(out):6d} ({n_miss/len(out)*100:5.1f}%)')

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved to: {OUTPUT_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()
