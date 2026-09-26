"""
Exploratory Data Analysis (EDA) and Dataset Profiler.
====================================================
Computes key distribution statistics, missing value profiles, singleton ratios,
country splits, and match cardinalities across train and test datasets.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def profile_dataframe(name: str, df: pd.DataFrame):
    """Prints a detailed profile of a DataFrame."""
    print(f"\n{'=' * 60}")
    print(f"Profiling {name}: {len(df):,} rows, {len(df.columns)} columns")
    print(f"{'=' * 60}")
    print(f"Columns: {list(df.columns)}")
    
    # Missing values
    print("\nMissing values:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        empty_count = (df[col].astype(str).str.strip() == "").sum()
        total_missing = null_count + empty_count
        pct = total_missing / len(df) * 100
        print(f"  - {col:20s}: {total_missing:10,d} missing ({pct:5.2f}%)")
        
    # Country distribution if present
    if "country" in df.columns:
        print("\nCountry distribution:")
        country_counts = df["country"].value_counts(dropna=False).head(10)
        for c, count in country_counts.items():
            pct = count / len(df) * 100
            print(f"  - {str(c):20s}: {count:10,d} ({pct:5.2f}%)")

    # Name and address string lengths
    if "business_name" in df.columns:
        name_lens = df["business_name"].fillna("").astype(str).str.len()
        print(f"\nBusiness name length: Mean = {name_lens.mean():.1f}, Median = {name_lens.median():.0f}, Max = {name_lens.max()}")
        
    if "business_address" in df.columns:
        addr_lens = df["business_address"].fillna("").astype(str).str.len()
        print(f"Business address length: Mean = {addr_lens.mean():.1f}, Median = {addr_lens.median():.0f}, Max = {addr_lens.max()}")


def run_eda():
    t0 = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Load Training Source Datasets
    print("\n[1/4] Loading training sources ...")
    s1_train = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str)
    profile_dataframe("Train Source 1 (Anchors)", s1_train)

    s2_train = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str)
    profile_dataframe("Train Source 2 (Target DB 1)", s2_train)

    s3_train = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str)
    profile_dataframe("Train Source 3 (Target DB 2)", s3_train)

    # 2. Ground Truth & Singleton Analysis
    print("\n[2/4] Analyzing Ground Truth & Match Cardinality ...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str)
    print(f"\nGround Truth Records: {len(gt):,}")
    print(f"Ground Truth Columns: {list(gt.columns)}")

    s1_col = "source1_entity_id" if "source1_entity_id" in gt.columns else gt.columns[0]
    match_col = "matched_entity_ids" if "matched_entity_ids" in gt.columns else gt.columns[1]

    valid_matches = gt[gt[match_col].notna() & (gt[match_col].astype(str).str.strip() != "")]
    s1_matched_ids = set(valid_matches[s1_col].unique())
    s1_total_ids = set(s1_train[s1_train.columns[0]].unique()) | set(gt[s1_col].unique())
    singletons = s1_total_ids - s1_matched_ids

    print(f"\n--- SINGLETON ENTITY BREAKDOWN ---")
    print(f"Total Source 1 Entities : {len(s1_total_ids):,}")
    print(f"Entities with True Match: {len(s1_matched_ids):,} ({len(s1_matched_ids)/len(s1_total_ids)*100:.2f}%)")
    print(f"Singleton Entities (0 m): {len(singletons):,} ({len(singletons)/len(s1_total_ids)*100:.2f}%)")

    # Match cardinality distribution (parsing comma-separated IDs)
    match_counts = valid_matches[match_col].apply(lambda x: len([i for i in str(x).split(",") if i.strip()]))
    print(f"\n--- MATCH CARDINALITY PER ENTITY ---")
    print(f"Average matches per non-singleton entity: {match_counts.mean():.2f}")
    print(f"Max matches for a single entity         : {match_counts.max():,}")
    cardinality_counts = match_counts.value_counts().sort_index().head(10)
    for num_matches, count in cardinality_counts.items():
        print(f"  - Entities with exactly {num_matches} match(es): {count:10,d}")

    # 3. Test Set Overview
    print("\n[3/4] Profiling Test Datasets ...")
    s1_test = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str)
    profile_dataframe("Test Source 1", s1_test)

    s2_test = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str)
    profile_dataframe("Test Source 2", s2_test)

    s3_test = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str)
    profile_dataframe("Test Source 3", s3_test)

    print("\n" + "=" * 70)
    print(f"EDA Completed in {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
