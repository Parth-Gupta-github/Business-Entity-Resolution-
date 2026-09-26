"""
Benchmark Script: LightGBM vs CatBoost vs XGBoost vs Ensemble
============================================================

This script benchmarks model architectures on pairwise business entity
resolution features extracted from real challenge data:
1. LightGBM Classifier
2. CatBoost Classifier
3. XGBoost Classifier
4. Dual Ensemble (0.6 LightGBM + 0.4 CatBoost)
5. Tri-Ensemble (0.5 LightGBM + 0.3 CatBoost + 0.2 XGBoost)

It evaluates the models using the competition metric: Macro F0.5 (with singleton handling)
and outputs a comparative performance table.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.config import config
from src.preprocess import preprocess_dataframe
from src.blocking import MultiPassBlocker
from src.features import build_feature_matrix, FEATURE_COLUMNS
from src.models import (
    train_lightgbm,
    train_catboost,
    train_xgboost,
    EnsembleClassifier,
    optimize_threshold,
    benchmark_models,
)
from src.metrics import evaluate_predictions


def load_benchmark_sample(
    data_dir: Path = None,
    target_records_count: int = 15000,
    target_distractors_count: int = 15000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """
    Extracts a representative benchmark subset containing matched entities,
    negative distractors, and singletons directly from local train TSVs.
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "dataset" / "train"

    print(f"Loading benchmark slice from {data_dir} ...")

    # 1. Read early S2 targets
    print("  Reading early S2 records ...")
    s2_early = pd.read_csv(data_dir / "train_source2.tsv", sep="\t", nrows=target_records_count, dtype=str)
    s2_early_ids = set(s2_early["entity_id"])

    # 2. Find matching S1 IDs in ground truth
    print("  Scanning ground truth for matching S1 anchors ...")
    matching_s1_ids = []
    gt_map: Dict[str, Set[str]] = {}

    gt_file = data_dir / "train_ground_truth.tsv"
    for chunk in pd.read_csv(gt_file, sep="\t", chunksize=50000, dtype=str):
        s1_col = "source1_entity_id" if "source1_entity_id" in chunk.columns else chunk.columns[0]
        match_col = "matched_entity_ids" if "matched_entity_ids" in chunk.columns else chunk.columns[1]

        for s1_id, m in zip(chunk[s1_col], chunk[match_col]):
            m_str = str(m)
            if m_str and m_str != "nan":
                tids = set(x.strip() for x in m_str.split(",") if x.strip())
                matched_in_s2 = tids & s2_early_ids
                if matched_in_s2:
                    matching_s1_ids.append(s1_id)
                    gt_map[s1_id] = matched_in_s2
        if len(matching_s1_ids) >= 400:
            break

    print(f"  Found {len(matching_s1_ids)} S1 entities with true matches in S2 early pool.")

    # 3. Read matching S1 records + singletons
    print("  Reading S1 records ...")
    matching_set = set(matching_s1_ids)
    s1_matches = []
    singleton_rows = []
    n_singles = 0

    s1_file = data_dir / "train_source1.tsv"
    for chunk in pd.read_csv(s1_file, sep="\t", chunksize=100000, dtype=str):
        matched_in_chunk = chunk[chunk["entity_id"].isin(matching_set)]
        if not matched_in_chunk.empty:
            s1_matches.append(matched_in_chunk)
        if n_singles < 150:
            non_matched = chunk[~chunk["entity_id"].isin(matching_set)]
            take_count = min(150 - n_singles, len(non_matched))
            sample_singles = non_matched.head(take_count)
            singleton_rows.append(sample_singles)
            for eid in sample_singles["entity_id"]:
                gt_map[eid] = set()
            n_singles += take_count

        total_found = sum(len(x) for x in s1_matches)
        if total_found >= len(matching_s1_ids):
            break

    s1_df = pd.concat(s1_matches + singleton_rows, ignore_index=True)

    # 4. Read distractors from S3
    print("  Reading S3 distractor targets ...")
    s3_distractors = pd.read_csv(data_dir / "train_source3.tsv", sep="\t", nrows=target_distractors_count, dtype=str)

    targets_df = pd.concat([s2_early, s3_distractors], ignore_index=True)

    print(f"  Total S1 anchors: {len(s1_df):,} ({len(matching_s1_ids):,} with true matches, {n_singles} singletons)")
    print(f"  Total Target pool: {len(targets_df):,} records (S2: {len(s2_early):,}, S3: {len(s3_distractors):,})")

    return s1_df, targets_df, gt_map


def main():
    start_time = time.time()
    config.create_dirs()

    # Step 1: Load sample
    s1_df, targets_df, gt_map = load_benchmark_sample()

    # Step 2: Preprocess
    print("\nPreprocessing data ...")
    s1_df = preprocess_dataframe(s1_df)
    targets_df = preprocess_dataframe(targets_df)

    # Step 3: Train / Val split (70 / 30)
    np.random.seed(config.RANDOM_SEED)
    all_s1_ids = list(s1_df["entity_id"].values)
    np.random.shuffle(all_s1_ids)

    split_idx = int(len(all_s1_ids) * 0.70)
    train_ids = set(all_s1_ids[:split_idx])
    val_ids = set(all_s1_ids[split_idx:])

    train_s1 = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_s1 = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    train_gt = {k: v for k, v in gt_map.items() if k in train_ids}
    val_gt = {k: v for k, v in gt_map.items() if k in val_ids}

    # Step 4: Blocking
    print("\nRunning multi-pass blocking ...")
    blocker = MultiPassBlocker(
        top_k=25,
        max_features=35_000,
        ngram_range=(3, 5),
        chunk_size=10_000,
        min_score=0.04,
    )
    blocker.fit(targets_df)
    train_cands = blocker.generate_candidates(train_s1, targets_df)
    val_cands = blocker.generate_candidates(val_s1, targets_df)

    # Step 5: Blocking recall on val
    n_true = 0
    n_found = 0
    for s1_id, true_set in val_gt.items():
        if not true_set:
            continue
        cands = set(c for c, _ in val_cands.get(s1_id, []))
        n_true += len(true_set)
        n_found += len(true_set & cands)
    recall = n_found / max(n_true, 1)
    print(f"\nValidation Blocking Recall: {recall:.4f} ({n_found}/{n_true} true pairs captured in candidates)")

    # Step 6: Feature extraction
    def extract_features(s1_data, cands, gt):
        pairs = []
        for s1_id, cand_list in cands.items():
            for cand_id, score in cand_list:
                pairs.append((s1_id, cand_id, score))
        f_df = build_feature_matrix(s1_data, targets_df, pairs, batch_size=25000)
        f_df["label"] = [
            1 if c in gt.get(s, set()) else 0
            for s, c in zip(f_df["source1_entity_id"], f_df["candidate_entity_id"])
        ]
        return f_df

    print("\nExtracting features for training fold ...")
    train_feat = extract_features(train_s1, train_cands, train_gt)
    print("\nExtracting features for validation fold ...")
    val_feat = extract_features(val_s1, val_cands, val_gt)

    pos_tr = int((train_feat["label"] == 1).sum())
    neg_tr = int((train_feat["label"] == 0).sum())
    pos_val = int((val_feat["label"] == 1).sum())
    neg_val = int((val_feat["label"] == 0).sum())

    print(f"\nTrain pairs: {len(train_feat):,} (Pos: {pos_tr:,}, Neg: {neg_tr:,}, Ratio: 1:{neg_tr/max(pos_tr, 1):.1f})")
    print(f"Val pairs:   {len(val_feat):,} (Pos: {pos_val:,}, Neg: {neg_val:,}, Ratio: 1:{neg_val/max(pos_val, 1):.1f})")

    # Step 7: Benchmark models!
    all_val_ids_set = set(val_s1["entity_id"])
    results = benchmark_models(train_feat, val_feat, val_gt, all_val_ids_set)

    print(f"\nAll models trained & evaluated successfully in {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()
