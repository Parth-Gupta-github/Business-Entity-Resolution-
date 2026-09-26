"""
Complete End-to-End Submission Generator for Amazon ML Challenge 2026.
======================================================================
1. Trains and checkpoints the high-performing GBDT Ensemble model (LightGBM + CatBoost + XGBoost).
2. Preprocesses the test datasets using multi-core parallel normalization.
3. Performs high-recall multi-pass candidate blocking on test data.
4. Extracts 32-dimensional pairwise similarity features.
5. Generates official competition submission TSVs:
   - output/matching_results.tsv (scored on leaderboard)
   - output/candidate_pairs.tsv (blocking candidates)
6. Validates against Amazon's official submission rules.
7. Packages into a submission-ready ZIP file for Unstop.
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


def run_full_submission_generation(team_name: str = "Business_Entity_Resolution"):
    """Runs complete end-to-end inference on test data and creates the submission zip."""
    start_time = time.time()
    config.create_dirs()

    # 1. Ensure Model & Threshold exist
    model, threshold = train_and_save_ensemble()

    # 2. Load & Preprocess Test Data
    print("\n" + "=" * 70)
    print("PHASE 2: Loading & Preprocessing Test Data (Multi-Core)")
    print("=" * 70)

    test_s1_ckpt = config.CHECKPOINT_DIR / "test_s1_preprocessed.parquet"
    test_tgt_ckpt = config.CHECKPOINT_DIR / "test_targets_preprocessed.parquet"

    if test_s1_ckpt.exists() and test_tgt_ckpt.exists():
        print(f"  [OK] Loading test data from checkpoints: {test_s1_ckpt.name}, {test_tgt_ckpt.name}")
        test_s1 = pd.read_parquet(test_s1_ckpt)
        test_targets = pd.read_parquet(test_tgt_ckpt)
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
        blocker = MultiPassBlocker(
            top_k=config.TOP_K_CANDIDATES,
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.CHAR_NGRAM_RANGE,
            chunk_size=config.BLOCKING_CHUNK_SIZE,
            min_score=config.MIN_TFIDF_SCORE,
        )
        print("  Fitting TF-IDF index on test targets (10M records) ...")
        blocker.fit(test_targets)

        print("  Retrieving candidates for 1.73M Test S1 entities ...")
        test_candidates = blocker.generate_candidates(test_s1, test_targets)

        with open(test_cand_ckpt, "wb") as f:
            pickle.dump(test_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [OK] Saved test candidate checkpoint: {test_cand_ckpt.name}")
        del blocker
        gc.collect()

    # 4. Feature Extraction
    print("\n" + "=" * 70)
    print("PHASE 4: 32-Dimensional Pairwise Feature Extraction on Test Candidates")
    print("=" * 70)

    test_feat_ckpt = config.CHECKPOINT_DIR / "test_features.parquet"
    if test_feat_ckpt.exists():
        print(f"  [OK] Loading test features from checkpoint: {test_feat_ckpt.name}")
        test_features = pd.read_parquet(test_feat_ckpt)
    else:
        pairs = []
        for s1_id, cand_list in test_candidates.items():
            for cand_id, score in cand_list:
                pairs.append((s1_id, cand_id, score))

        print(f"  Extracting features for {len(pairs):,} candidate pairs ...")
        test_features = build_feature_matrix(
            test_s1, test_targets, pairs, batch_size=config.FEATURE_BATCH_SIZE
        )
        test_features.to_parquet(test_feat_ckpt, index=False)
        print(f"  [OK] Saved test features checkpoint: {test_feat_ckpt.name}")

    # 5. Model Inference & Prediction Generation
    print("\n" + "=" * 70)
    print("PHASE 5: Generating Predictions & Writing Output TSVs")
    print("=" * 70)

    X_test = test_features[FEATURE_COLUMNS].values
    probs = model.predict_proba(X_test)[:, 1]
    test_features = test_features.copy()
    test_features["prob"] = probs

    all_test_s1_ids = test_s1["entity_id"].tolist()
    predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_test_s1_ids}

    matched_pairs = test_features[test_features["prob"] >= threshold]
    for s1_id, cand_id in zip(matched_pairs["source1_entity_id"], matched_pairs["candidate_entity_id"]):
        predictions[s1_id].add(cand_id)

    n_with_matches = sum(1 for v in predictions.values() if v)
    n_singletons = sum(1 for v in predictions.values() if not v)
    total_matches = sum(len(v) for v in predictions.values())

    print(f"  Applied Calibrated Threshold: tau = {threshold:.3f}")
    print(f"  Entities with Matches       : {n_with_matches:,} ({n_with_matches/len(all_test_s1_ids)*100:.2f}%)")
    print(f"  Predicted Singletons (Empty): {n_singletons:,} ({n_singletons/len(all_test_s1_ids)*100:.2f}%)")
    print(f"  Total Match Pairs Assigned  : {total_matches:,}")

    # Write output/matching_results.tsv
    matching_path = config.OUTPUT_DIR / config.MATCHING_RESULTS_FILE
    rows = []
    for s1_id in all_test_s1_ids:
        m_ids = sorted(predictions.get(s1_id, set()))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(m_ids) if m_ids else "",
        })
    matching_df = pd.DataFrame(rows)
    matching_df.to_csv(matching_path, sep="\t", index=False)
    print(f"  [OK] Wrote {matching_path} ({len(matching_df):,} rows)")

    # Write output/candidate_pairs.tsv
    cand_df = format_candidate_pairs_dataframe(test_candidates)
    cand_path = config.OUTPUT_DIR / config.CANDIDATE_PAIRS_FILE
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f"  [OK] Wrote {cand_path} ({len(cand_df):,} rows)")

    # 6. Validate with Official Validator
    print("\n" + "=" * 70)
    print("PHASE 6: Running Amazon ML Challenge Official Validator")
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

    # 7. Package ZIP for Unstop
    print("\n" + "=" * 70)
    print("PHASE 7: Packaging Final Submission ZIP")
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
