#!/usr/bin/env python3
"""
Multi-Source Clinical Data Downloader (v2)
===========================================
Downloads clinical data from multiple independent sources:
1. METABRIC (via cBioPortal API) - ~2509 breast cancer patients
2. CPTAC (via cBioPortal API) - Independent cancer cohorts (5 types)
3. SEER (requires manual download, provides processing guide)

All data is aligned to the unified schema with missing-value indicators.

Usage:
    python scripts/download_additional_data.py
"""

import requests
import pandas as pd
import numpy as np
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

CBIOPORTAL_BASE = 'https://www.cbioportal.org/api'
PAGE_SIZE = 10000

# ============================================================================
# Source Definitions
# ============================================================================

ADDITIONAL_SOURCES = {
    'metabric': {
        'study_id': 'brca_metabric',
        'name': 'METABRIC Breast Cancer',
        'cancer_type': 'BRCA',
    },
    'cptac_brca': {
        'study_id': 'brca_cptac_2020',
        'name': 'CPTAC Breast Cancer',
        'cancer_type': 'BRCA',
    },
    'cptac_coad': {
        'study_id': 'coad_cptac_2019',
        'name': 'CPTAC Colon Cancer',
        'cancer_type': 'COAD',
    },
    'cptac_ucec': {
        'study_id': 'uec_cptac_gdc',
        'name': 'CPTAC Uterine Cancer',
        'cancer_type': 'UCEC',
    },
    'cptac_luad': {
        'study_id': 'luad_cptac_gdc',
        'name': 'CPTAC Lung Cancer',
        'cancer_type': 'LUAD',
    },
    'cptac_kirc': {
        'study_id': 'rcc_cptac_gdc',
        'name': 'CPTAC Renal Cell Carcinoma',
        'cancer_type': 'KIRC',
    },
    'coadread_china': {
        'study_id': 'coadread_cass_2020',
        'name': 'CAS Shanghai Colorectal Cancer',
        'cancer_type': 'COAD',
    },

    # ============================================================================
    # NEW: HCC / Liver Cancer Independent Cohorts (multi-institution)
    # ============================================================================
    'hcc_msk_2024': {
        'study_id': 'hcc_msk_2024',
        'name': 'MSK Hepatocellular Carcinoma',
        'cancer_type': 'LIHC',
    },
    'hcc_clca_2024': {
        'study_id': 'hcc_clca_2024',
        'name': 'CLCA Hepatocellular Carcinoma',
        'cancer_type': 'LIHC',
    },
    'hcc_inserm_fr_2015': {
        'study_id': 'hcc_inserm_fr_2015',
        'name': 'INSERM French Hepatocellular Carcinoma',
        'cancer_type': 'LIHC',
    },
    'lihc_amc_prv': {
        'study_id': 'lihc_amc_prv',
        'name': 'AMC Korean Liver Cancer',
        'cancer_type': 'LIHC',
    },
    'hcc_meric_2021': {
        'study_id': 'hcc_meric_2021',
        'name': 'MERiC Basel Hepatocellular Carcinoma',
        'cancer_type': 'LIHC',
    },
    'hcc_mskimpact_2018': {
        'study_id': 'hcc_mskimpact_2018',
        'name': 'MSK-IMPACT Hepatocellular Carcinoma',
        'cancer_type': 'LIHC',
    },

    # ============================================================================
    # NEW: Large Pan-Cancer from China
    # ============================================================================
    'pan_origimed_2020': {
        'study_id': 'pan_origimed_2020',
        'name': 'OrigMed China Pan-cancer',
        'cancer_type': 'PANCAN',
    },

    # ============================================================================
    # NEW: Colorectal Cancer (independent from TCGA-COAD)
    # ============================================================================
    'coadread_dfci_2016': {
        'study_id': 'coadread_dfci_2016',
        'name': 'DFCI Colorectal Adenocarcinoma',
        'cancer_type': 'COAD',
    },
    'crc_msk_2017': {
        'study_id': 'crc_msk_2017',
        'name': 'MSK Metastatic Colorectal Cancer',
        'cancer_type': 'COAD',
    },
    'crc_sysucc_2022': {
        'study_id': 'crc_sysucc_2022',
        'name': 'SYSUCC Colorectal Cancer China',
        'cancer_type': 'COAD',
    },

    # ============================================================================
    # NEW: Lung Cancer (independent from TCGA-LUAD)
    # ============================================================================
    'luad_mskcc_2020': {
        'study_id': 'luad_mskcc_2020',
        'name': 'MSK Lung Adenocarcinoma',
        'cancer_type': 'LUAD',
    },
    'nsclc_tracerx_2017': {
        'study_id': 'nsclc_tracerx_2017',
        'name': 'TRACERx Lung Cancer UK',
        'cancer_type': 'LUAD',
    },

    # ============================================================================
    # NEW: Esophagogastric Cancer
    # ============================================================================
    'egc_msk_2023': {
        'study_id': 'egc_msk_2023',
        'name': 'MSK Esophagogastric Cancer',
        'cancer_type': 'STAD',
    },

    # ============================================================================
    # NEW: ICGC International Cohort
    # ============================================================================
    'chol_icgc_2017': {
        'study_id': 'chol_icgc_2017',
        'name': 'ICGC Cholangiocarcinoma',
        'cancer_type': 'CHOL',
    },

    # ============================================================================
    # NEW: TARGET Pediatric Cohorts (completely different domain)
    # ============================================================================
    'all_phase2_target_2018_pub': {
        'study_id': 'all_phase2_target_2018_pub',
        'name': 'Pediatric Acute Lymphoid Leukemia - Phase II (TARGET)',
        'cancer_type': 'PEDIATRIC',
    },
    'nbl_target_2018_pub': {
        'study_id': 'nbl_target_2018_pub',
        'name': 'Pediatric Neuroblastoma (TARGET)',
        'cancer_type': 'PEDIATRIC',
    },
    'wt_target_2018_pub': {
        'study_id': 'wt_target_2018_pub',
        'name': 'Pediatric Wilms Tumor (TARGET)',
        'cancer_type': 'PEDIATRIC',
    },
    'aml_target_2018_pub': {
        'study_id': 'aml_target_2018_pub',
        'name': 'Pediatric Acute Myeloid Leukemia (TARGET)',
        'cancer_type': 'PEDIATRIC',
    },
}

# Multi-database attribute → unified column mapping
# Supports different naming conventions across (METABRIC, CPTAC, etc.)
# Each entry: (attribute_id_in_source, unified_column, transform_function)
# First match wins (sources with more specific names take priority)
FIELD_MAP_ENTRIES = [
    # --- Age ---
    ('AGE_AT_DIAGNOSIS', 'Age', float),
    ('AGE', 'Age', float),

    # --- Sex ---
    ('SEX', 'Sex', lambda x: 1 if str(x).lower() in ('female', 'f') else (0 if str(x).lower() in ('male', 'm') else np.nan)),

    # --- Vital Status (multiple source formats) ---
    ('VITAL_STATUS', 'Vital_Status', lambda x: 1 if str(x).upper() in ('DECEASED', 'DEAD', 'DIED OF DISEASE', 'DIED OF OTHER CAUSES')
                                       else (0 if str(x).upper() in ('ALIVE', 'LIVING', 'LIVING') else np.nan)),
    ('OS_STATUS', 'Vital_Status', lambda x: 1 if str(x).upper() in ('DECEASED', 'DEAD', '1:DECEASED', '1:DEAD')
                                   else (0 if str(x).upper() in ('ALIVE', 'LIVING', '0:LIVING', '0:ALIVE') else np.nan)),

    # --- Survival Time ---
    ('OS_MONTHS', 'Survival_Months', float),
    ('DFS_MONTHS', 'Survival_Months', float),
    ('RFS_MONTHS', 'RFS_Months', float),

    # --- Stage (TNM + Overall) ---
    ('TUMOR_STAGE', 'Stage', lambda x: str(x).upper().replace('STAGE ', '').strip() if x and x != 'NA' else np.nan),
    ('STAGE', 'Stage', lambda x: str(x).upper().replace('STAGE ', '').strip() if x and x != 'NA' else np.nan),
    ('PATHOLOGY_T_STAGE', 'T_Stage', str),
    ('PATHOLOGY_N_STAGE', 'N_Stage', str),

    # --- Grade ---
    ('GRADE', 'Grade', lambda x: int(str(x).replace('G', '').replace('g', '').strip()) if x and str(x).strip() in ['1','2','3','4','G1','G2','G3','G4','g1','g2','g3','g4'] else np.nan),

    # --- Tumor Size ---
    ('TUMOR_SIZE', 'Tumor_Size', float),

    # --- Lymph Nodes ---
    ('LYMPH_NODES_EXAMINED_POSITIVE', 'Lymph_Nodes', float),

    # --- Molecular Markers (breast cancer) ---
    ('ER_STATUS', 'ER_Status', str),
    ('PR_STATUS', 'PR_Status', str),
    ('HER2_STATUS', 'HER2_Status', str),

    # --- Treatment ---
    ('CHEMOTHERAPY', 'Chemotherapy', lambda x: 1 if str(x).lower() in ('yes', 'y', 'received') else (0 if str(x).lower() in ('no', 'n', 'not received') else np.nan)),
    ('RADIO_THERAPY', 'Radiotherapy', lambda x: 1 if str(x).lower() in ('yes', 'y', 'received') else (0 if str(x).lower() in ('no', 'n', 'not received') else np.nan)),
    ('HORMONE_THERAPY', 'Hormone_Therapy', lambda x: 1 if str(x).lower() in ('yes', 'y') else (0 if str(x).lower() in ('no', 'n') else np.nan)),
]

UNIFIED_SCHEMA = [
    'Source', 'Cancer_Type', 'Patient_ID',
    'Age', 'Sex', 'Stage', 'Grade',
    'Survival_Months', 'Vital_Status',
    'Tumor_Size', 'Lymph_Nodes',
    'ER_Status', 'PR_Status', 'HER2_Status',
    'Chemotherapy', 'Radiotherapy',
]

FEATURE_COLS = [c for c in UNIFIED_SCHEMA if c not in ('Source', 'Cancer_Type', 'Patient_ID')]

DIRS = {
    'raw': Path('data_raw'),
    'processed': Path('data_processed'),
}


# ============================================================================
# cBioPortal API Functions (with pagination)
# ============================================================================

def fetch_all_clinical_data(study_id: str, clinical_data_type: str = 'PATIENT') -> List[dict]:
    """
    Fetch ALL clinical data records for a study, handling pagination.

    Args:
        study_id: cBioPortal study ID
        clinical_data_type: 'PATIENT' or 'SAMPLE'

    Returns:
        List of clinical data records (dicts with patientId, clinicalAttributeId, value)
    """
    all_data = []
    page = 0

    while True:
        resp = requests.get(
            f'{CBIOPORTAL_BASE}/studies/{study_id}/clinical-data',
            params={
                'clinicalDataType': clinical_data_type,
                'projection': 'SUMMARY',
                'pageSize': PAGE_SIZE,
                'pageNumber': page,
            },
            timeout=60
        )

        if resp.status_code != 200:
            logger.warning(f'    Page {page}: HTTP {resp.status_code}')
            break

        data = resp.json()
        if not data:
            break

        all_data.extend(data)
        logger.info(f'    Page {page}: +{len(data)} records (total: {len(all_data)})')

        # If fewer than pageSize returned, we've reached the end
        if len(data) < PAGE_SIZE:
            break

        page += 1
        time.sleep(0.3)

    return all_data


def pivot_clinical_data(data: List[dict]) -> pd.DataFrame:
    """Convert long-format clinical data list to wide DataFrame."""
    records = []
    for d in data:
        records.append({
            'patientId': d['patientId'],
            'attr': d['clinicalAttributeId'],
            'value': d.get('value', np.nan),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df_wide = df.pivot_table(
        index='patientId', columns='attr',
        values='value', aggfunc='first'
    ).reset_index()

    return df_wide


# ============================================================================
# Harmonization
# ============================================================================

def harmonize_data(df: pd.DataFrame, source_key: str, cancer_type: str) -> pd.DataFrame:
    """Map cBioPortal attributes to unified schema (flexible multi-source)."""
    records = []

    for _, row in df.iterrows():
        record = {
            'Source': source_key,
            'Cancer_Type': cancer_type,
            'Patient_ID': row.get('patientId', np.nan),
        }

        for attr_id, col_name, transform in FIELD_MAP_ENTRIES:
            if attr_id in df.columns:
                raw_val = row.get(attr_id, np.nan)
                # Skip if column already filled (first match wins)
                if col_name in record and record[col_name] is not np.nan and not (isinstance(record[col_name], float) and np.isnan(record[col_name])):
                    continue
                try:
                    if raw_val is not np.nan and str(raw_val).strip() not in ('', 'NA', 'N/A', 'nan'):
                        record[col_name] = transform(raw_val)
                except (ValueError, TypeError):
                    pass

        records.append(record)

    df_out = pd.DataFrame(records)

    # Ensure all schema columns exist
    for col in UNIFIED_SCHEMA:
        if col not in df_out.columns:
            df_out[col] = np.nan

    df_out = df_out[UNIFIED_SCHEMA]

    # Add missing-value indicators
    for col in FEATURE_COLS:
        missing_col = f'{col}_missing'
        df_out[missing_col] = df_out[col].isna().astype(int)

    return df_out


# ============================================================================
# Per-Source Processing
# ============================================================================

def process_cbioportal_source(source_key: str, config: dict) -> Optional[pd.DataFrame]:
    """Download, harmonize, and save a single cBioPortal source."""
    study_id = config['study_id']
    name = config['name']
    cancer_type = config['cancer_type']

    logger.info(f'\n{"=" * 60}')
    logger.info(f'{name} ({study_id})')
    logger.info(f'{"=" * 60}')

    # Fetch patient-level clinical data
    logger.info('  Fetching patient-level clinical data...')
    patient_data = fetch_all_clinical_data(study_id, 'PATIENT')
    if not patient_data:
        logger.warning('  No patient data found, skipping')
        return None

    df_patient = pivot_clinical_data(patient_data)
    logger.info(f'  Patient data: {df_patient.shape[0]} patients, {df_patient.shape[1] - 1} attributes')

    # Try to fetch sample-level data (GRADE, TUMOR_STAGE, etc.)
    logger.info('  Fetching sample-level clinical data...')
    sample_data = fetch_all_clinical_data(study_id, 'SAMPLE')
    if sample_data:
        df_sample = pivot_clinical_data(sample_data)
        logger.info(f'  Sample data: {df_sample.shape[0]} patients, {df_sample.shape[1] - 1} attributes')

        # Merge sample data into patient data
        overlap_cols = [c for c in df_sample.columns if c not in df_patient.columns or c == 'patientId']
        df_patient = df_patient.merge(
            df_sample[['patientId'] + [c for c in overlap_cols if c != 'patientId']],
            on='patientId', how='left'
        )
        logger.info(f'  Merged shape: {df_patient.shape}')

    # Show available mapped attributes
    mapped_attr_ids = set(e[0] for e in FIELD_MAP_ENTRIES)
    available = [c for c in df_patient.columns if c in mapped_attr_ids]
    logger.info(f'  Mapped attributes: {len(available)}/29')
    for attr in sorted(available):
        n_non_null = df_patient[attr].notna().sum()
        logger.info(f'    {attr:35s} {n_non_null:5d} values')

    # Harmonize
    df_harm = harmonize_data(df_patient, source_key, cancer_type)
    logger.info(f'  Harmonized: {df_harm.shape[0]} rows, {df_harm.shape[1]} columns')

    # Source-specific fixes
    if source_key == 'metabric':
        # METABRIC uses OS_STATUS + OS_MONTHS for survival
        if 'Vital_Status' in df_harm.columns:
            df_harm['Vital_Status'] = pd.to_numeric(df_harm['Vital_Status'], errors='coerce').fillna(0).astype(int)
        if 'Survival_Months' in df_harm.columns:
            df_harm['Survival_Months'] = pd.to_numeric(df_harm['Survival_Months'], errors='coerce')

    # Save
    out_path = DIRS['processed'] / f'harmonized_{source_key}.csv'
    df_harm.to_csv(out_path, index=False)
    logger.info(f'  Saved → {out_path}')

    return df_harm


# ============================================================================
# SEER Guide
# ============================================================================

def write_seer_guide():
    """Write SEER processing instructions."""
    guide = """# SEER Data Download & Processing Guide

## Why SEER?

SEER (Surveillance, Epidemiology, and End Results) is the gold-standard US
cancer registry with ~28% of the US population. Its missingness patterns are
**dramatically different** from TCGA:

- **No genomic data** (no ER/PR/HER2, no sequencing)
- **Limited treatment fields** (~80% missing radiation/surgery details)
- **Different staging system** (SEER historic stage A vs AJCC)
- **Very large sample size** (millions of patients)

This makes SEER the perfect "different domain" for our paper; a classifier
can easily tell SEER from TCGA just by which fields are empty.

## Download Instructions

1. **Register**: Go to https://seer.cancer.gov/data/ and click
   "SEER Research Data" (or "SEER Research Plus Data")
2. **Complete request**: Fill in the form (free, usually approved 1-5 days)
3. **Download**: Get the SEER data file (.tar.gz, ~500 MB)
4. **Extract**: Place the `.txt` data files in `data_raw/seer/`
5. **Process**: Run (once files are placed):
   ```bash
   python scripts/process_seer.py
   ```

## Data Processing

The script `process_seer.py` (in scripts/) will:
- Parse SEER fixed-width format (.txt)
- Extract: Age, Sex, Race, Stage, Grade, Survival_Months, Vital_Status
- Map to unified schema
- Add missing-value indicators
- Save as `data_processed/harmonized_seer.csv`

## Expected Missingness

| Field | TCGA (BRCA) | SEER |
|-------|-------------|------|
| Age | ~0% | ~0% |
| Sex | ~0% | ~0% |
| Stage | ~9% | ~5% |
| Grade | 100% (BRCA) | ~40% |
| Treatment | ~30% | ~80% |
| Molecular markers | ~10% | 100% |
"""
    Path('docs').mkdir(exist_ok=True)
    with open('docs/SEER_Data_Guide.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    logger.info('SEER guide → docs/SEER_Data_Guide.md')


# ============================================================================
# Report
# ============================================================================

def compute_report(df: pd.DataFrame, source_key: str, config: dict) -> dict:
    """Compute per-source missingness report."""
    existing = [c for c in FEATURE_COLS if c in df.columns]
    missing_pct = (df[existing].isna().sum() / len(df) * 100).round(1)

    return {
        'source': source_key,
        'name': config['name'],
        'cancer_type': config['cancer_type'],
        'n_samples': len(df),
        'n_events': int(df['Vital_Status'].sum()) if 'Vital_Status' in df.columns else 0,
        'event_rate': round(df['Vital_Status'].mean() * 100, 1) if 'Vital_Status' in df.columns else 0,
        'missing_pct': missing_pct.to_dict(),
    }


def write_report(reports: List[dict]):
    """Write combined data inventory report."""
    lines = ['# Report: Multi-Source Data Inventory\n']
    lines.append(f'*Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}*\n')

    lines.append('## Data Sources\n')
    lines.append('| Key | Source | Cancer | N | Events | Event Rate |\n')
    lines.append('|---|---|---|---|---|---|\n')
    for r in reports:
        lines.append(f'| {r["source"]:15s} | {r["name"]:35s} | {r["cancer_type"]:6s} | '
                      f'{r["n_samples"]:5d} | {r["n_events"]:4d} | {r["event_rate"]:5.1f}% |\n')

    total = sum(r['n_samples'] for r in reports)
    ev = sum(r['n_events'] for r in reports)
    lines.append(f'\n**Total additional: {total} patients, {ev} events**\n')

    # Missingness matrix
    lines.append('\n## Missingness Matrix (% missing)\n')
    feat_header = ' | '.join(f'{c:>15s}' for c in FEATURE_COLS)
    lines.append(f'| Source | N | {feat_header} |\n')
    lines.append('|' + '|'.join('---' for _ in range(len(FEATURE_COLS) + 2)) + '|\n')

    for r in reports:
        row = f'| {r["source"]:15s} | {r["n_samples"]:5d} |'
        for feat in FEATURE_COLS:
            v = r['missing_pct'].get(feat, 100.0)
            row += f' {v:>14.1f}% |'
        lines.append(row + '\n')

    Path('docs').mkdir(exist_ok=True)
    with open('docs/02_Multi_Source_Inventory.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info('Report → docs/02_Multi_Source_Inventory.md')


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info('=' * 60)
    logger.info('Multi-Source Clinical Data Downloader')
    logger.info(f'Target: {len(ADDITIONAL_SOURCES)} independent cohorts')
    logger.info('=' * 60)

    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    results = {}
    reports = []
    failed = []

    for key, config in ADDITIONAL_SOURCES.items():
        try:
            df = process_cbioportal_source(key, config)
            if df is not None:
                results[key] = df
                r = compute_report(df, key, config)
                reports.append(r)
                logger.info(f'  >> {key:15s} | {r["n_samples"]:5d} samples | '
                            f'{r["n_events"]:4d} events | {r["event_rate"]:5.1f}%')
            else:
                failed.append(key)
        except Exception as e:
            logger.error(f'  FAILED {key}: {e}')
            failed.append(key)
            import traceback
            traceback.print_exc()

        time.sleep(1)

    # Summary
    logger.info('\n' + '=' * 60)
    logger.info('SUMMARY')
    logger.info('=' * 60)
    logger.info(f'Successful: {len(results)}/{len(ADDITIONAL_SOURCES)}')
    if failed:
        logger.warning(f'Failed: {failed}')

    if results:
        write_report(reports)
        logger.info(f'Total additional patients: {sum(df.shape[0] for df in results.values())}')

    write_seer_guide()
    logger.info('\nOutputs:')
    for k in results:
        logger.info(f'  data_processed/harmonized_{k}.csv')
    logger.info(f'  docs/02_Multi_Source_Inventory.md')
    logger.info(f'  docs/SEER_Data_Guide.md')


if __name__ == '__main__':
    main()
