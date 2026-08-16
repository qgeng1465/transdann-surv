#!/usr/bin/env python3
"""
TCGA Multi-Cancer Clinical Data Downloader
============================================
Downloads clinical data from NCI GDC API for multiple TCGA cancer types.
Aligns all data to a unified schema and adds missing-value indicators.

Target Cancer Types: BRCA, LUAD, LIHC, COAD, KIRC, STAD, PRAD, THCA, UCEC, SKCM

Data Source: NCI Genomic Data Commons (GDC) - public API, no authentication required.

Output: data_raw/tcga/{cancer}.csv (raw) → data_processed/harmonized_{cancer}.csv (aligned)
"""

import requests
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Configuration
# ============================================================================

# Cancer types to download (TCGA project IDs)
CANCER_TYPES = {
    'BRCA': 'Breast Invasive Carcinoma',
    'LUAD': 'Lung Adenocarcinoma',
    'LIHC': 'Liver Hepatocellular Carcinoma',
    'COAD': 'Colon Adenocarcinoma',
    'KIRC': 'Kidney Renal Clear Cell Carcinoma',
    'STAD': 'Stomach Adenocarcinoma',
    'PRAD': 'Prostate Adenocarcinoma',
    'THCA': 'Thyroid Carcinoma',
    'UCEC': 'Uterine Corpus Endometrial Carcinoma',
    'SKCM': 'Skin Cutaneous Melanoma',
}

# Unified schema mapping:
# Maps GDC API fields → our unified column names
# Based on actual GDC API response structure (verified 2026-07-30):
#   demographic = dict with: vital_status, days_to_death, sex_at_birth, age_at_index, race, ethnicity
#   diagnoses   = list of dicts with: ajcc_pathologic_stage, ajcc_pathologic_t/n/m, tumor_grade,
#                  days_to_last_follow_up, primary_diagnosis, morphology
#   exposures   = list of dicts, mostly empty for most cancer types
CLINICAL_FIELDS_MAP = {
    # Demographic fields (dict)
    'demographic.sex_at_birth': 'Sex',
    'demographic.race': 'Race',
    'demographic.ethnicity': 'Ethnicity',
    'demographic.age_at_index': 'Age',
    'demographic.vital_status': 'Vital_Status',
    'demographic.days_to_death': 'Days_to_Death',

    # Diagnosis fields (list → first element)
    'diagnoses.ajcc_pathologic_stage': 'Stage',
    'diagnoses.primary_diagnosis': 'Primary_Diagnosis',
    'diagnoses.days_to_last_follow_up': 'Days_to_Last_Follow_Up',
    'diagnoses.tumor_grade': 'Grade',
    'diagnoses.ajcc_pathologic_t': 'T',
    'diagnoses.ajcc_pathologic_n': 'N',
    'diagnoses.ajcc_pathologic_m': 'M',
    'diagnoses.morphology': 'Morphology',

    # Exposures (rarely available but kept for completeness)
    'exposures.alcohol_history': 'Alcohol_History',
    'exposures.cigarettes_per_day': 'Smoking',
}

# These columns we want in our final unified schema
UNIFIED_SCHEMA = [
    'Cancer_Type', 'Patient_ID',
    'Age', 'Sex', 'Race', 'Stage', 'T', 'N', 'M', 'Grade',
    'Survival_Months', 'Vital_Status',
    'Alcohol_History', 'Smoking',
]

DIRS = {
    'raw': Path('data_raw/tcga'),
    'processed': Path('data_processed'),
}

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# GDC API Helpers
# ============================================================================

GDC_API_BASE = 'https://api.gdc.cancer.gov'


def query_gdc_cases(project_id: str, fields: List[str],
                    size: int = 2000, max_retries: int = 3) -> List[dict]:
    """
    Query GDC Cases API for all cases in a TCGA project.

    Args:
        project_id: TCGA project ID, e.g., 'TCGA-BRCA'
        fields: List of clinical fields to retrieve
        size: Page size for results
        max_retries: Number of retries on failure

    Returns:
        List of case records as dicts
    """
    params = {
        'filters': json.dumps({
            'op': '=',
            'content': {
                'field': 'project.project_id',
                'value': project_id,
            }
        }),
        'fields': ','.join(['case_id', 'project.project_id'] + fields),
        'format': 'JSON',
        'size': size,
        'expand': ','.join(set(f.split('.')[0] for f in fields)),
    }

    url = f'{GDC_API_BASE}/cases'

    for attempt in range(max_retries):
        try:
            logger.info(f'  Querying GDC for {project_id} (attempt {attempt + 1})...')
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            total = data.get('data', {}).get('pagination', {}).get('total', 0)
            cases = data.get('data', {}).get('hits', [])
            logger.info(f'  Retrieved {len(cases)} / {total} cases for {project_id}')

            return cases

        except requests.exceptions.RequestException as e:
            logger.warning(f'  Attempt {attempt + 1} failed: {e}')
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                logger.info(f'  Retrying in {wait}s...')
                time.sleep(wait)
            else:
                logger.error(f'  Failed after {max_retries} attempts for {project_id}')
                raise

    return []


def extract_nested_value(case: dict, field_path: str) -> any:
    """
    Extract a value from nested GDC response structure.

    GDC returns nested JSON like:
    case['demographic'][0]['gender'] → 'female'
    case['diagnoses'][0]['tumor_stage'] → 'stage i'
    """
    parts = field_path.split('.')

    # First part is the expand block, second is the field
    block = parts[0]
    field = parts[1] if len(parts) > 1 else None

    # Get the block (may be a list or a dict)
    value = case.get(block, None)

    # If block is a list, take the first element
    if isinstance(value, list):
        # Skip empty lists
        if not value:
            return np.nan
        value = value[0]

    # If value is a dict, extract the field
    if isinstance(value, dict) and field:
        value = value.get(field, np.nan)

    # Handle None
    if value is None:
        return np.nan

    return value


# ============================================================================
# Data Transformation
# ============================================================================

def parse_clinical_data(cases: List[dict], cancer_type: str) -> pd.DataFrame:
    """
    Parse GDC case records into a flat DataFrame with unified columns.
    """
    records = []

    for case in cases:
        record = {
            'Cancer_Type': cancer_type,
            'Patient_ID': case.get('case_id', np.nan),
        }

        # Map each GDC field to our schema
        for gdc_path, our_col in CLINICAL_FIELDS_MAP.items():
            record[our_col] = extract_nested_value(case, gdc_path)

        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f'  Parsed {len(df)} records')
    return df


def convert_days_to_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert days to months for survival time.

    GDC data structure:
    - Vital_Status comes from demographic.vital_status: 'Alive' | 'Dead'
    - Days_to_Death comes from demographic.days_to_death (days, for dead patients)
    - Days_to_Last_Follow_Up comes from diagnoses.days_to_last_follow_up (days, for censored)
    """
    # Convert days columns to numeric
    for col in ['Days_to_Death', 'Days_to_Last_Follow_Up']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Vital status: 'Dead' → 1 (event), 'Alive' → 0 (censored)
    if 'Vital_Status' in df.columns:
        # Convert to string first to handle mixed types (e.g., NaN float)
        vs_raw = df['Vital_Status'].astype(str).str.lower().str.strip()
        df['Vital_Status'] = vs_raw.map({
            'dead': 1,
            'alive': 0,
            'nan': np.nan,
            '': np.nan,
            'not reported': np.nan,
            'unknown': np.nan,
        })

    # Survival_Months = days / 30.44
    # Priority: Days_to_Death for dead, Days_to_Last_Follow_Up for alive
    death = pd.to_numeric(df.get('Days_to_Death', pd.Series([np.nan] * len(df))), errors='coerce')
    follow_up = pd.to_numeric(df.get('Days_to_Last_Follow_Up', pd.Series([np.nan] * len(df))), errors='coerce')

    # Use death time if dead AND death days available
    df['Survival_Months'] = np.where(
        (df['Vital_Status'] == 1) & death.notna(),
        death / 30.44,
        np.where(
            follow_up.notna(),
            follow_up / 30.44,
            np.nan
        )
    )

    return df


def clean_stage(stage_series: pd.Series) -> pd.Series:
    """
    Clean and standardize AJCC stage values.
    'stage i' → 'I', 'stage iiia' → 'IIIA', etc.
    """
    def _clean(s):
        if pd.isna(s):
            return np.nan
        s = str(s).lower().strip()
        # Remove 'stage ' prefix
        if s.startswith('stage '):
            s = s[6:]
        # Remove ' or ' suffixes
        if ' or ' in s:
            s = s.split(' or ')[0]
        s = s.strip()
        return s.upper() if s else np.nan

    return stage_series.apply(_clean)


def clean_tnm(value_series: pd.Series) -> pd.Series:
    """
    Clean TNM values.
    'p1' → '1', 'p2a' → '2a', 'n0' → '0', etc.
    """
    def _clean(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        # Remove leading letters: pN0 → 0, N0 → 0, pT2 → 2
        s = s.lstrip('pPTNMc ')
        if s == '' or s == 'x' or s == 'X':
            return np.nan
        return s

    return value_series.apply(_clean)


def process_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process age: convert from days (GDC format) to years.
    GDC age_at_index is in days.
    """
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        # Convert days to years (if values are large, > 1000)
        if df['Age'].mean() > 100:
            df['Age'] = df['Age'] / 365.25
        # Round to integer
        df['Age'] = df['Age'].round(0)
    return df


def align_to_schema(df: pd.DataFrame, cancer_type: str) -> pd.DataFrame:
    """
    Align the parsed data to the unified schema.
    Add missing-value indicators for each feature.
    """
    # Derive Survival_Months
    df = convert_days_to_months(df)

    # Process age
    df = process_age(df)

    # Clean staging
    if 'Stage' in df.columns:
        df['Stage'] = clean_stage(df['Stage'])
    if 'T' in df.columns:
        df['T'] = clean_tnm(df['T'])
    if 'N' in df.columns:
        df['N'] = clean_tnm(df['N'])
    if 'M' in df.columns:
        df['M'] = clean_tnm(df['M'])

    # Sex mapping
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].str.lower().map({
            'male': 0, 'female': 1, 'm': 0, 'f': 1,
        })

    # Grade mapping
    if 'Grade' in df.columns:
        grade_map = {
            'g1': 1, 'g2': 2, 'g3': 3, 'g4': 4,
            'grade i': 1, 'grade ii': 2, 'grade iii': 3, 'grade iv': 4,
            'low grade': 1, 'high grade': 3,
        }
        df['Grade'] = df['Grade'].astype(str).str.lower().map(
            lambda x: grade_map.get(x, np.nan) if x != 'nan' else np.nan
        )

    # Select and order columns to match unified schema
    result_columns = []
    for col in UNIFIED_SCHEMA:
        if col in df.columns:
            result_columns.append(col)
        else:
            df[col] = np.nan
            result_columns.append(col)

    df_out = df[result_columns].copy()

    # Add missing-value indicators for clinical features
    feature_cols = ['Age', 'Sex', 'Stage', 'T', 'N', 'M', 'Grade']
    for col in feature_cols:
        missing_col = f'{col}_missing'
        if col in df_out.columns:
            df_out[missing_col] = df_out[col].isna().astype(int)

    # Add Cancer_Type label if not present
    df_out['Cancer_Type'] = cancer_type

    logger.info(f'  Aligned to schema: {df_out.shape[1]} columns, {df_out.shape[0]} rows')

    return df_out


# ============================================================================
# Missingness Analysis
# ============================================================================

def compute_missingness_report(df: pd.DataFrame, cancer_type: str) -> dict:
    """
    Compute missingness statistics for the dataset.
    """
    feature_cols = ['Age', 'Sex', 'Stage', 'T', 'N', 'M', 'Grade', 'Survival_Months', 'Vital_Status']
    existing = [c for c in feature_cols if c in df.columns]

    missing_pct = (df[existing].isna().sum() / len(df) * 100).round(1)

    report = {
        'cancer_type': cancer_type,
        'n_samples': len(df),
        'n_events': int(df['Vital_Status'].sum()) if 'Vital_Status' in df.columns else 0,
        'event_rate': round(df['Vital_Status'].mean() * 100, 1) if 'Vital_Status' in df.columns else 0,
        'missing_pct': missing_pct.to_dict(),
    }

    return report


# ============================================================================
# Main Pipeline
# ============================================================================

def download_and_process_cancer(cancer_code: str, cancer_name: str) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    Download and process data for a single cancer type.

    Returns:
        Tuple of (harmonized_df, missing_report)
    """
    project_id = f'TCGA-{cancer_code}'
    logger.info(f'\n{"="*60}')
    logger.info(f'Processing {cancer_code}: {cancer_name} ({project_id})')
    logger.info(f'{"="*60}')

    # Step 1: Define GDC field paths to query
    # Note: GDC uses 'expand' parameter - demographic is dict, diagnoses is list
    gdc_fields = [
        # Demographic (dict)
        'demographic.sex_at_birth',
        'demographic.race',
        'demographic.ethnicity',
        'demographic.age_at_index',
        'demographic.vital_status',
        'demographic.days_to_death',

        # Diagnoses (list)
        'diagnoses.ajcc_pathologic_stage',
        'diagnoses.primary_diagnosis',
        'diagnoses.days_to_last_follow_up',
        'diagnoses.tumor_grade',
        'diagnoses.ajcc_pathologic_t',
        'diagnoses.ajcc_pathologic_n',
        'diagnoses.ajcc_pathologic_m',
        'diagnoses.morphology',

        # Exposures (rarely available)
        'exposures.alcohol_history',
        'exposures.cigarettes_per_day',
    ]

    # Step 2: Query GDC API
    try:
        cases = query_gdc_cases(project_id, gdc_fields)
    except Exception as e:
        logger.error(f'GDC query failed for {cancer_code}: {e}')
        return None, None

    if not cases:
        logger.warning(f'No cases found for {cancer_code}')
        return None, None

    # Step 3: Parse clinical data
    df = parse_clinical_data(cases, cancer_code)

    # Step 4: Save raw data
    raw_path = DIRS['raw'] / f'{cancer_code}_raw.csv'
    df.to_csv(raw_path, index=False)
    logger.info(f'  Saved raw data to {raw_path}')

    # Step 5: Align to unified schema
    df_harmonized = align_to_schema(df, cancer_code)

    # Step 6: Save harmonized data
    processed_path = DIRS['processed'] / f'harmonized_{cancer_code}.csv'
    df_harmonized.to_csv(processed_path, index=False)
    logger.info(f'  Saved harmonized data to {processed_path}')

    # Step 7: Compute missingness report
    report = compute_missingness_report(df_harmonized, cancer_code)

    return df_harmonized, report


def create_missingness_summary(reports: List[dict]) -> pd.DataFrame:
    """
    Create a missingness heatmap data frame from all reports.
    """
    feature_cols = ['Age', 'Sex', 'Stage', 'T', 'N', 'M', 'Grade', 'Survival_Months', 'Vital_Status']

    rows = []
    for report in reports:
        row = {'Cancer_Type': report['cancer_type'],
               'N': report['n_samples'],
               'Event_Rate': report['event_rate']}
        for col in feature_cols:
            row[f'{col}_missing_pct'] = report['missing_pct'].get(col, 100.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_data_inventory_report(all_dfs: Dict[str, pd.DataFrame],
                                    reports: List[dict],
                                    missingness_df: pd.DataFrame):
    """
    Generate a comprehensive data inventory markdown report.
    """
    report_lines = []
    report_lines.append('# Report 1: Data Inventory & Missingness Analysis\n')
    report_lines.append(f'*Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}*\n')

    # Summary table
    report_lines.append('## Summary\n')
    report_lines.append('| Cancer | Code | N Patients | N Events | Event Rate | Features |\n')
    report_lines.append('|--------|------|-----------|----------|------------|----------|\n')

    for r in reports:
        report_lines.append(
            f'| {CANCER_TYPES[r["cancer_type"]]} | {r["cancer_type"]} | '
            f'{r["n_samples"]} | {r["n_events"]} | {r["event_rate"]}% | 15+ |\n'
        )

    total_patients = sum(r['n_samples'] for r in reports)
    total_events = sum(r['n_events'] for r in reports)
    report_lines.append(f'\n**Total: {total_patients} patients, {total_events} events across {len(reports)} cancer types**\n')

    # Missingness table
    report_lines.append('\n## Missingness Analysis\n')
    report_lines.append('Percentage of missing values per feature per cancer type:\n\n')

    feature_cols = ['Age', 'Sex', 'Stage', 'T', 'N', 'M', 'Grade', 'Survival_Months', 'Vital_Status']

    # Table header
    header = '| Cancer | N | ' + ' | '.join(f'{c}' for c in feature_cols) + ' |\n'
    report_lines.append(header)
    sep = '|' + '|'.join('---' for _ in range(len(feature_cols) + 2)) + '|\n'
    report_lines.append(sep)

    for r in reports:
        row = f'| {r["cancer_type"]} | {r["n_samples"]} |'
        for col in feature_cols:
            val = r['missing_pct'].get(col, 'N/A')
            if val != 'N/A':
                # Color code: >50% red, 20-50% yellow, <20% green
                row += f' {val:>5}% |'
            else:
                row += '  N/A |'
        row += '\n'
        report_lines.append(row)

    # Survival curve summary
    report_lines.append('\n## Survival Distribution\n')
    report_lines.append('| Cancer | Median Survival (months) | Min | Max |\n')
    report_lines.append('|--------|-------------------------|-----|-----|\n')

    for code, df in all_dfs.items():
        surv = df['Survival_Months'].dropna()
        if len(surv) > 0:
            report_lines.append(
                f'| {code} | {surv.median():.1f} | {surv.min():.0f} | {surv.max():.0f} |\n'
            )
        else:
            report_lines.append(f'| {code} | N/A | N/A | N/A |\n')

    report_lines.append('\n---\n')
    report_lines.append('*Report generated by TCGA Clinical Data Downloader*\n')

    Path('docs').mkdir(exist_ok=True)
    with open('docs/01_Data_Inventory.md', 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    logger.info('Report saved to docs/01_Data_Inventory.md')


def main():
    """Main pipeline."""
    logger.info('=' * 60)
    logger.info('TCGA Multi-Cancer Clinical Data Downloader')
    logger.info(f'Target: {len(CANCER_TYPES)} cancer types')
    logger.info(f'Data source: NCI Genomic Data Commons (GDC)')
    logger.info('=' * 60)

    # Create directories
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Process each cancer type
    all_dfs = {}
    all_reports = []
    failed = []

    for code, name in CANCER_TYPES.items():
        df_harm, report = download_and_process_cancer(code, name)

        if df_harm is not None and report is not None:
            all_dfs[code] = df_harm
            all_reports.append(report)
        else:
            failed.append(code)

        # Be polite to GDC API - rate limiting
        time.sleep(1)

    # Summary
    logger.info('\n' + '=' * 60)
    logger.info('DOWNLOAD SUMMARY')
    logger.info('=' * 60)
    logger.info(f'Successful: {len(all_dfs)}/{len(CANCER_TYPES)} cancer types')
    if failed:
        logger.warning(f'Failed: {failed}')

    if all_dfs:
        total = sum(df.shape[0] for df in all_dfs.values())
        logger.info(f'Total patients downloaded: {total}')

        # Create missingness summary
        missingness_df = create_missingness_summary(all_reports)
        missingness_df.to_csv(DIRS['processed'] / 'missingness_summary.csv', index=False)
        logger.info(f'Missingness summary saved to {DIRS["processed"] / "missingness_summary.csv"}')

        # Generate data inventory report
        generate_data_inventory_report(all_dfs, all_reports, missingness_df)

        # Print per-cancer stats
        logger.info('\nPer-Cancer Summary:')
        logger.info(f'{"Cancer":<8} {"Patients":<10} {"Events":<8} {"Rate":<8}')
        logger.info('-' * 36)
        for r in all_reports:
            logger.info(f'{r["cancer_type"]:<8} {r["n_samples"]:<10} {r["n_events"]:<8} {r["event_rate"]:<7}%')

    logger.info('\nDone!')

    # Print paths to output files
    logger.info('\nOutput files:')
    for code in all_dfs:
        logger.info(f'  data_processed/harmonized_{code}.csv')
    logger.info(f'  data_processed/missingness_summary.csv')
    logger.info(f'  docs/01_Data_Inventory.md')

    return all_dfs, all_reports


if __name__ == '__main__':
    main()
