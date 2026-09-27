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
from src.postprocess import resolve_matches_for_entity
# NOTE: load_benchmark_sample (src.benchmark_models) is intentionally no
# longer imported/used here -- it produced the biased debug-slice training
# set that caused the 0.111 score. It remains available for its original
# purpose: quick architecture comparisons in benchmark_models.py's own
# __main__, which is not part of the production submission path.
from utils.validate_submission import validate
from utils.package_submission import create_submission_zip


def train_and_save_ensemble() -> Tuple[EnsembleClassifier, float]:
    """Train LightGBM, CatBoost, and XGBoost on a full-scale, randomly
    stratified training sample and save the calibrated ensemble.

    IMPORTANT: earlier versions of this function trained on
    ``load_benchmark_sample()`` -- a ~550-anchor / 40k-target debug slice
    (first N rows of train_source2/3, first 400 matched anchors found)
    built for the quick model-architecture comparisons in
    benchmark_models.py. That slice is not remotely representative of the
    real ~2.2M x 10.3M distribution the model sees at test time, which
    silently produced a model + decision threshold badly miscalibrated for
    production use (this was the direct cause of the 0.111 leaderboard
    score: with the threshold tuned against the biased slice, almost every
    real test candidate scored below it, so ~93% of entities fell back to
    "no match").

    This version reuses the exact same full-scale, randomly-sampled
    training procedure already implemented in pipeline.py: a stratified
    random sample of S1 anchors drawn from the *entire* train_source1 pool,
    blocked against the *entire* S2+S3 target pool.
    """
    # Local import avoids a module-load-time circular import (pipeline.py
    # itself imports several names from src.models that this module also
    # imports at top-level).
    from src.pipeline import (
        load_and_preprocess, load_ground_truth, create_validation_split,
        run_blocking, extract_features_with_labels, measure_blocking_recall,
    )

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

    # 1. Load & preprocess the FULL training data (every row of S1/S2/S3),
    #    not a debug slice. Cached to parquet checkpoints by load_and_preprocess.
    train_s1_full, train_targets = load_and_preprocess(
        config.TRAIN_DIR,
        config.TRAIN_SOURCE1, config.TRAIN_SOURCE2, config.TRAIN_SOURCE3,
        label="train",
    )
    ground_truth = load_ground_truth()

    # 2. Stratified RANDOM train/val split drawn from the FULL S1 pool
    #    (config.SAMPLE_TRAIN_ENTITIES / SAMPLE_VAL_ENTITIES control size).
    train_s1, val_s1, train_gt, val_gt = create_validation_split(train_s1_full, ground_truth)
    del train_s1_full
    gc.collect()

    # 3. Multi-pass blocking against the FULL target pool (all of S2+S3),
    #    exactly as it will be run at test time.
    print("\n-- Blocking on TRAINING fold (full target pool) --")
    train_cands = run_blocking(train_s1, train_targets, label="train_fold")
    print("\n-- Blocking on VALIDATION fold (full target pool) --")
    val_cands = run_blocking(val_s1, train_targets, label="val_fold")

    measure_blocking_recall(val_cands, val_gt)

    # 4. Feature extraction with real ground-truth labels
    train_feat = extract_features_with_labels(
        train_s1, train_targets, train_cands, train_gt, label="train_fold"
    )
    val_feat = extract_features_with_labels(
        val_s1, train_targets, val_cands, val_gt, label="val_fold"
    )

    pos_tr = int((train_feat["label"] == 1).sum())
    neg_tr = int((train_feat["label"] == 0).sum())
    print(f"  Train pairs: {len(train_feat):,} (Pos: {pos_tr:,}, Neg: {neg_tr:,}, "
          f"Ratio 1:{neg_tr / max(pos_tr, 1):.1f})")

    # 5. Train base models
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

    # Only pairs scoring within POST_MARGIN of the threshold are kept around
    # (with their probability) so we can apply per-entity postprocessing
    # (score-gap filtering + a match-count cap) after scoring everything --
    # this is the F0.5-precision boost from postprocess.py that was written
    # but never actually wired into the submission before.
    POST_MARGIN = 0.15
    POST_SCORE_GAP = 0.08
    POST_MAX_MATCHES = 3
    near_threshold: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in all_test_s1_ids}
    batch_size = 100_000
    current_chunk: List[Tuple[str, str, float]] = []
    total_pairs_scored = 0
    t_inference_start = time.time()

    def process_chunk(chunk: List[Tuple[str, str, float]]):
        nonlocal total_pairs_scored
        if not chunk:
            return
        X_batch, valid_pairs = fast_extract_features_batch(chunk, s1_lookup, tgt_lookup)
        if len(X_batch) > 0:
            probs = model.predict_proba(X_batch)[:, 1]
            keep_mask = probs >= (threshold - POST_MARGIN)
            keep_indices = np.where(keep_mask)[0]
            for idx in keep_indices:
                s_id, c_id = valid_pairs[idx]
                near_threshold[s_id].append((c_id, float(probs[idx])))
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

    # Resolve final matches per entity: threshold + score-gap filtering +
    # a small cap on matches per entity (precision-heavy metric = don't
    # grab every borderline candidate just because it crossed tau).
    predictions: Dict[str, Set[str]] = {}
    total_matches_found = 0
    for s1_id in all_test_s1_ids:
        selected = resolve_matches_for_entity(
            near_threshold.get(s1_id, []),
            threshold=threshold,
            score_gap=POST_SCORE_GAP,
            max_matches_per_entity=POST_MAX_MATCHES,
        )
        predictions[s1_id] = set(selected)
        total_matches_found += len(selected)

    elapsed_inf = time.time() - t_inference_start
    print(f"\n  [OK] Finished inference on {total_pairs_scored:,} candidate pairs in {elapsed_inf:.1f}s ({total_pairs_scored/max(elapsed_inf, 1):,.0f} pairs/sec)")
    print(f"  Total Matches Assigned: {total_matches_found:,}")

    # Free lookups
    del s1_lookup, tgt_lookup, near_threshold
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
            f.write(f"{s1_id}\t{','.join(c_ids)}\n")
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
    print(f"[SUCCESS] SUBMISSION READY FOR UPLOAD in {elapsed:.1f} minutes!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and package final submission")
    parser.add_argument("--team", default="Business_Entity_Resolution", help="Team name for zip filename")
    args = parser.parse_args()

    run_full_submission_generation(team_name=args.team)
