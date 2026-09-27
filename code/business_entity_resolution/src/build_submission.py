"""
Complete End-to-End Submission Generator for Amazon ML Challenge 2026.
======================================================================
1. Loads/Trains and checkpoints the high-performing GBDT Ensemble model (LightGBM + CatBoost + XGBoost).
2. Preprocesses the test datasets using multi-core parallel normalization (Parquet checkpoints).
3. Performs high-recall multi-pass candidate blocking on test data (Checkpoint: test_candidates.pkl).
4. Memory-safe, high-throughput streaming 32-D pairwise feature extraction.
5. High-speed Tri-Ensemble batch inference directly to predictions.
6. Generates official competition submission TSVs:
   - output/matching_results.tsv (scored on leaderboard)
   - output/candidate_pairs.tsv (blocking candidates)
7. Validates against Amazon's official submission rules.
8. Packages into a submission-ready ZIP file for Unstop.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz, distance

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.config import config
from src.preprocess import preprocess_dataframe
from src.blocking import MultiPassBlocker, format_candidate_pairs_dataframe
from src.features import build_feature_matrix, FEATURE_COLUMNS
from src.models import (
    train_lightgbm,
    train_catboost,
    train_xgboost,
    EnsembleClassifier,
    optimize_threshold,
)
from src.metrics import evaluate_predictions
from src.benchmark_models import load_benchmark_sample
from utils.validate_submission import validate
from utils.package_submission import create_submission_zip


def train_and_save_ensemble() -> Tuple[EnsembleClassifier, float]:
    """Train LightGBM, CatBoost, and XGBoost on representative training data and save ensemble."""
    print("=" * 70)
    print("PHASE 1: Training & Calibrating High-Performance Model Ensemble")
    print("=" * 70)

    ckpt_model = config.CHECKPOINT_DIR / "ensemble_model.pkl"
    ckpt_thresh = config.CHECKPOINT_DIR / "best_threshold.txt"

    if ckpt_model.exists() and ckpt_thresh.exists():
        print(f"  [OK] Found existing model checkpoint: {ckpt_model.name}")
        model = EnsembleClassifier.load(ckpt_model)
        with open(ckpt_thresh, "r") as f:
            best_thresh = float(f.read().strip())
        print(f"  [OK] Loaded calibrated threshold: tau = {best_thresh:.3f}")
        return model, best_thresh

    # Load representative sample with matches, distractors, and singletons
    s1_df, targets_df, gt_map = load_benchmark_sample(target_records_count=20000, target_distractors_count=20000)

    # Preprocess
    s1_df = preprocess_dataframe(s1_df)
    targets_df = preprocess_dataframe(targets_df)

    # Train/Val split
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

    # Blocking
    blocker = MultiPassBlocker(top_k=25, max_features=35000, ngram_range=(3, 4), chunk_size=10000, min_score=0.04)
    blocker.fit(targets_df)
    train_cands = blocker.generate_candidates(train_s1, targets_df)
    val_cands = blocker.generate_candidates(val_s1, targets_df)

    def extract_feat(s1_data, cands, gt):
        pairs = []
        for s1_id, cand_list in cands.items():
            for cand_id, score in cand_list:
                pairs.append((s1_id, cand_id, score))
        f_df = build_feature_matrix(s1_data, targets_df, pairs, batch_size=25000)
        s1_arr = f_df["source1_entity_id"].values
        cand_arr = f_df["candidate_entity_id"].values
        f_df["label"] = np.array([1 if c in gt.get(s, set()) else 0 for s, c in zip(s1_arr, cand_arr)], dtype=np.int32)
        return f_df

    train_feat = extract_feat(train_s1, train_cands, train_gt)
    val_feat = extract_feat(val_s1, val_cands, val_gt)

    # Train base models
    lgb_model = train_lightgbm(train_feat, val_feat, save_path=config.CHECKPOINT_DIR / "lightgbm_model.pkl")
    cb_model = train_catboost(train_feat, val_feat, save_path=config.CHECKPOINT_DIR / "catboost_model.pkl")
    xgb_model = train_xgboost(train_feat, val_feat, save_path=config.CHECKPOINT_DIR / "xgboost_model.pkl")

    # Create Tri-Ensemble
    ensemble = EnsembleClassifier([
        ("lightgbm", lgb_model, 0.5),
        ("catboost", cb_model, 0.3),
        ("xgboost", xgb_model, 0.2),
    ])
    ensemble.save(ckpt_model)

    # Optimize threshold on validation fold
    all_val_ids_set = set(val_s1["entity_id"])
    best_thresh, best_f05, best_metrics = optimize_threshold(
        ensemble, val_feat, val_gt, all_val_ids_set, beta=config.BETA
    )

    with open(ckpt_thresh, "w") as f:
        f.write(f"{best_thresh:.4f}")

    print(f"\n  [OK] Saved ensemble model and optimal threshold (tau={best_thresh:.3f}, Val F0.5={best_f05:.4f})")
    return ensemble, best_thresh


def fast_extract_features_batch(
    pairs_chunk: List[Tuple[str, str, float]],
    s1_lookup: Dict[str, Tuple[str, str, str, str, str]],
    tgt_lookup: Dict[str, Tuple[str, str, str, str, str]],
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Extracts 32-D features directly into a contiguous float32 NumPy array
    with zero intermediate Python dict allocations and 100% mathematical parity with FEATURE_COLUMNS.
    """
    n = len(pairs_chunk)
    feats = np.zeros((n, 32), dtype=np.float32)
    valid_pairs: List[Tuple[str, str]] = []

    out_idx = 0
    for s1_id, cand_id, blk_score in pairs_chunk:
        r1 = s1_lookup.get(s1_id)
        r2 = tgt_lookup.get(cand_id)
        if r1 is None or r2 is None:
            continue

        name1, addr1, post1, country1, sn1 = r1
        name2, addr2, post2, country2, sn2 = r2

        # 0. Blocking score
        feats[out_idx, 0] = blk_score

        # 1. Name similarities
        feats[out_idx, 1] = fuzz.ratio(name1, name2) / 100.0
        feats[out_idx, 2] = fuzz.partial_ratio(name1, name2) / 100.0
        feats[out_idx, 3] = fuzz.token_sort_ratio(name1, name2) / 100.0
        feats[out_idx, 4] = fuzz.token_set_ratio(name1, name2) / 100.0
        feats[out_idx, 5] = fuzz.WRatio(name1, name2) / 100.0
        feats[out_idx, 6] = distance.JaroWinkler.similarity(name1, name2)
        feats[out_idx, 7] = 1.0 if (name1 and name1 == name2) else 0.0

        comb1 = name1 + " " + addr1
        comb2 = name2 + " " + addr2
        name_tokens1 = set(name1.split())
        name_tokens2 = set(name2.split())
        all_tokens1 = set(comb1.split())
        all_tokens2 = set(comb2.split())

        first1 = name1.split()[0] if name1.split() else ""
        first2 = name2.split()[0] if name2.split() else ""
        feats[out_idx, 8] = 1.0 if (first1 and first1 == first2) else 0.0
        shorter_tokens = min(len(name_tokens1), len(name_tokens2))
        feats[out_idx, 9] = len(name_tokens1 & name_tokens2) / shorter_tokens if shorter_tokens > 0 else 0.0

        g1 = set(name1[i:i+3] for i in range(len(name1)-2)) if len(name1) >= 3 else (set([name1]) if name1 else set())
        g2 = set(name2[i:i+3] for i in range(len(name2)-2)) if len(name2) >= 3 else (set([name2]) if name2 else set())
        feats[out_idx, 10] = len(g1 & g2) / len(g1 | g2) if (g1 or g2) else 0.0

        # 2. Address similarities
        feats[out_idx, 11] = fuzz.ratio(addr1, addr2) / 100.0
        feats[out_idx, 12] = fuzz.partial_ratio(addr1, addr2) / 100.0
        feats[out_idx, 13] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        feats[out_idx, 14] = fuzz.token_set_ratio(addr1, addr2) / 100.0
        feats[out_idx, 15] = distance.JaroWinkler.similarity(addr1, addr2)

        # 3. Combined similarity
        feats[out_idx, 16] = fuzz.ratio(comb1, comb2) / 100.0

        # 4. Jaccard & Token Overlaps
        feats[out_idx, 17] = (len(name_tokens1 & name_tokens2) / len(name_tokens1 | name_tokens2)) if (name_tokens1 or name_tokens2) else 0.0
        feats[out_idx, 18] = float(len(name_tokens1 & name_tokens2))
        feats[out_idx, 19] = 1.0 if (sorted(name_tokens1) == sorted(name_tokens2) and name_tokens1) else 0.0

        feats[out_idx, 20] = (len(all_tokens1 & all_tokens2) / len(all_tokens1 | all_tokens2)) if (all_tokens1 or all_tokens2) else 0.0
        feats[out_idx, 21] = float(len(set(addr1.split()) & set(addr2.split())))

        # 5. Postal Code
        if post1 and post2:
            feats[out_idx, 22] = 1.0 if post1 == post2 else 0.0
            feats[out_idx, 23] = 1.0 if post1[:3] == post2[:3] else 0.0
            feats[out_idx, 24] = 0.0
        else:
            feats[out_idx, 22] = 0.0
            feats[out_idx, 23] = 0.0
            feats[out_idx, 24] = 1.0

        # 6. Street number
        if sn1 and sn2:
            feats[out_idx, 25] = 1.0 if sn1 == sn2 else 0.0
            feats[out_idx, 26] = 1.0 if sn1 != sn2 else 0.0
        else:
            feats[out_idx, 25] = 0.0
            feats[out_idx, 26] = 0.0

        # 7. Country
        feats[out_idx, 27] = 1.0 if (country1 == country2 or not country1 or not country2) else 0.0
        feats[out_idx, 28] = 1.0 if (country1 and country2 and country1 != country2) else 0.0

        # 8. Length metrics
        feats[out_idx, 29] = abs(len(name1) - len(name2))
        feats[out_idx, 30] = abs(len(addr1) - len(addr2))
        feats[out_idx, 31] = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)

        valid_pairs.append((s1_id, cand_id))
        out_idx += 1

    if out_idx < n:
        feats = feats[:out_idx]
    return feats, valid_pairs


def run_full_submission_generation(team_name: str = "Business_Entity_Resolution"):
    """Runs complete end-to-end inference on test data and creates the submission zip."""
    start_time = time.time()
    config.create_dirs()

    # 1. Ensure Model & Threshold exist
    model, threshold = train_and_save_ensemble()

    # 2. Load & Preprocess Test Data
    print("\n" + "=" * 70)
    print("PHASE 2: Loading & Preprocessing Test Data")
    print("=" * 70)

    test_s1_ckpt = config.CHECKPOINT_DIR / "test_s1_preprocessed.parquet"
    test_tgt_ckpt = config.CHECKPOINT_DIR / "test_targets_preprocessed.parquet"

    req_cols = ["entity_id", "clean_name", "clean_address", "postal_code", "country", "street_number"]

    if test_s1_ckpt.exists() and test_tgt_ckpt.exists():
        print(f"  [OK] Loading test data from checkpoints: {test_s1_ckpt.name}, {test_tgt_ckpt.name}")
        test_s1 = pd.read_parquet(test_s1_ckpt, columns=req_cols)
        test_targets = pd.read_parquet(test_tgt_ckpt, columns=req_cols)
    else:
        print("  Loading raw test files ...")
        t0 = time.time()
        test_s1 = pd.read_csv(config.TEST_DIR / config.TEST_SOURCE1, sep="\t", dtype=str)
        test_s2 = pd.read_csv(config.TEST_DIR / config.TEST_SOURCE2, sep="\t", dtype=str)
        test_s3 = pd.read_csv(config.TEST_DIR / config.TEST_SOURCE3, sep="\t", dtype=str)
        print(f"  Loaded raw test files in {time.time()-t0:.1f}s")

        print("  Multi-core normalizing Test S1 ...")
        test_s1 = preprocess_dataframe(test_s1)
        test_s1.to_parquet(test_s1_ckpt, index=False)

        print("  Multi-core normalizing Test S2 & S3 ...")
        test_s2 = preprocess_dataframe(test_s2)
        test_s3 = preprocess_dataframe(test_s3)
        test_targets = pd.concat([test_s2, test_s3], ignore_index=True)
        del test_s2, test_s3
        gc.collect()

        test_targets.to_parquet(test_tgt_ckpt, index=False)
        print(f"  [OK] Preprocessed test dataset: S1={len(test_s1):,}, Targets={len(test_targets):,}")

    all_test_s1_ids = test_s1["entity_id"].tolist()

    # Build ultra-lean tuple lookup dictionaries to keep RAM minimal
    print("  Building in-memory entity lookup maps ...")
    t0 = time.time()
    s1_lookup: Dict[str, Tuple[str, str, str, str, str]] = {}
    for row in test_s1.itertuples(index=False):
        s1_lookup[row.entity_id] = (
            str(row.clean_name or ""),
            str(row.clean_address or ""),
            str(row.postal_code or ""),
            str(row.country or ""),
            str(row.street_number or ""),
        )

    tgt_lookup: Dict[str, Tuple[str, str, str, str, str]] = {}
    for row in test_targets.itertuples(index=False):
        tgt_lookup[row.entity_id] = (
            str(row.clean_name or ""),
            str(row.clean_address or ""),
            str(row.postal_code or ""),
            str(row.country or ""),
            str(row.street_number or ""),
        )

    del test_s1, test_targets
    gc.collect()
    print(f"  [OK] Lookup maps ready in {time.time()-t0:.1f}s (S1: {len(s1_lookup):,}, Targets: {len(tgt_lookup):,})")

    # 3. Test Blocking
    print("\n" + "=" * 70)
    print("PHASE 3: Multi-Pass Candidate Blocking on Test Dataset")
    print("=" * 70)

    test_cand_ckpt = config.CHECKPOINT_DIR / "test_candidates.pkl"
    if test_cand_ckpt.exists():
        print(f"  [OK] Loading test candidates from checkpoint: {test_cand_ckpt.name}")
        with open(test_cand_ckpt, "rb") as f:
            test_candidates = pickle.load(f)
    else:
        raise FileNotFoundError(f"Missing {test_cand_ckpt}")

    # 4 & 5. Streaming Feature Extraction & Ensemble Inference (OOM-Proof)
    print("\n" + "=" * 70)
    print("PHASE 4 & 5: Streaming Feature Extraction + Ensemble Scoring")
    print(f"  Threshold tau: {threshold:.4f} | Batch size: 100,000 pairs")
    print("=" * 70)

    predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_test_s1_ids}
    batch_size = 100_000
    current_chunk: List[Tuple[str, str, float]] = []
    total_pairs_scored = 0
    total_matches_found = 0
    t_inference_start = time.time()

    def process_chunk(chunk: List[Tuple[str, str, float]]):
        nonlocal total_matches_found, total_pairs_scored
        if not chunk:
            return
        X_batch, valid_pairs = fast_extract_features_batch(chunk, s1_lookup, tgt_lookup)
        if len(X_batch) > 0:
            probs = model.predict_proba(X_batch)[:, 1]
            match_mask = probs >= threshold
            matched_indices = np.where(match_mask)[0]
            for idx in matched_indices:
                s_id, c_id = valid_pairs[idx]
                predictions[s_id].add(c_id)
                total_matches_found += 1
            total_pairs_scored += len(X_batch)
        del X_batch, valid_pairs
        gc.collect()

    for s1_id, cand_list in tqdm(test_candidates.items(), desc="Scoring Candidates", unit="entities"):
        for cand_id, score in cand_list:
            current_chunk.append((s1_id, cand_id, score))
            if len(current_chunk) >= batch_size:
                process_chunk(current_chunk)
                current_chunk = []

    if current_chunk:
        process_chunk(current_chunk)
        current_chunk = []

    elapsed_inf = time.time() - t_inference_start
    print(f"\n  [OK] Finished inference on {total_pairs_scored:,} candidate pairs in {elapsed_inf:.1f}s ({total_pairs_scored/max(elapsed_inf, 1):,.0f} pairs/sec)")
    print(f"  Total Matches Assigned: {total_matches_found:,}")

    # Free lookups
    del s1_lookup, tgt_lookup
    gc.collect()

    # 6. Write Output TSVs
    print("\n" + "=" * 70)
    print("PHASE 6: Generating Official Competition Output TSVs")
    print("=" * 70)

    # Write output/matching_results.tsv
    matching_path = config.OUTPUT_DIR / config.MATCHING_RESULTS_FILE
    print(f"  Writing {matching_path.name} ...")
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1_ids:
            m_set = predictions.get(s1_id, set())
            if m_set:
                f.write(f"{s1_id}\t{','.join(sorted(m_set))}\n")
            else:
                f.write(f"{s1_id}\t\n")
    print(f"  [OK] Saved {matching_path} ({len(all_test_s1_ids):,} rows)")

    # Write output/candidate_pairs.tsv
    cand_path = config.OUTPUT_DIR / config.CANDIDATE_PAIRS_FILE
    print(f"  Writing {cand_path.name} ...")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1_ids:
            cands = test_candidates.get(s1_id, [])
            c_ids = [c[0] for c in cands]
            f.write(f"{s1_id}\t{'|'.join(c_ids)}\n")
    print(f"  [OK] Saved {cand_path} ({len(all_test_s1_ids):,} rows)")

    # Free candidates & predictions
    del test_candidates, predictions
    gc.collect()

    # 7. Validate with Official Validator
    print("\n" + "=" * 70)
    print("PHASE 7: Running Amazon ML Challenge Official Validator")
    print("=" * 70)
    errors, warnings = validate(
        matching_path=str(matching_path),
        candidate_path=str(cand_path),
        test_dir=str(config.TEST_DIR),
        check_ids=False,
    )
    if errors:
        print("\n[!] VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("  [PASS] All submission validation rules PASSED with 0 errors!")

    # 8. Package ZIP for Unstop
    print("\n" + "=" * 70)
    print("PHASE 8: Packaging Final Submission ZIP")
    print("=" * 70)
    create_submission_zip(team_name=team_name, output_dir=str(PROJECT_ROOT / "submissions"))

    elapsed = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print(f"🎉 SUBMISSION READY FOR UPLOAD in {elapsed:.1f} minutes!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and package final submission")
    parser.add_argument("--team", default="Business_Entity_Resolution", help="Team name for zip filename")
    args = parser.parse_args()

    run_full_submission_generation(team_name=args.team)
