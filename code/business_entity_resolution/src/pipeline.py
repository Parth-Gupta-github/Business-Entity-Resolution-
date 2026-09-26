#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 -- End-to-End Entity Resolution Pipeline
================================================================

This script runs the complete pipeline from raw data to a submission-ready
``matching_results.tsv``.

Stages
------
1. Load & preprocess all source data
2. Create a stratified validation split from training ground truth
3. Run multi-pass blocking (TF-IDF + postal code + name prefix)
4. Measure blocking recall on the validation fold
5. Extract pairwise features for all candidate pairs
6. Train a LightGBM binary classifier
7. Sweep decision threshold to maximise Macro F₀.₅
8. Run the full pipeline on test data → generate output files

Usage
-----
Run from the repository root::

    python code/business_entity_resolution/src/pipeline.py [--mode full|val_only|test_only]

Intermediate checkpoints are saved under ``checkpoints/`` so you can resume
after a crash or iterate on later stages without re-running expensive earlier
stages.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.config import config
from src.preprocess import preprocess_dataframe
from src.blocking import MultiPassBlocker, format_candidate_pairs_dataframe
from src.features import build_feature_matrix, FEATURE_COLUMNS
from src.metrics import evaluate_predictions
from src.models import (
    train_lightgbm,
    train_catboost,
    train_xgboost,
    EnsembleClassifier,
    benchmark_models,
)


# ===========================================================================
# STEP 1 -- Load & Preprocess
# ===========================================================================
def load_and_preprocess(
    source_dir: Path, source1_file: str, source2_file: str, source3_file: str,
    label: str = "train",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load source TSVs, preprocess, and return (s1_df, targets_df)."""
    print(f"\n{'='*70}")
    print(f"STEP 1: Loading & preprocessing {label} data ...")
    print(f"{'='*70}")

    ckpt_s1 = config.CHECKPOINT_DIR / f"{label}_s1_preprocessed.parquet"
    ckpt_tgt = config.CHECKPOINT_DIR / f"{label}_targets_preprocessed.parquet"

    if ckpt_s1.exists() and ckpt_tgt.exists():
        print(f"  [OK] Loading from checkpoint: {ckpt_s1.name}, {ckpt_tgt.name}")
        s1 = pd.read_parquet(ckpt_s1)
        targets = pd.read_parquet(ckpt_tgt)
        print(f"    S1: {len(s1):,}  |  Targets (S2+S3): {len(targets):,}")
        return s1, targets

    config.create_dirs()

    t0 = time.time()
    s1 = pd.read_csv(source_dir / source1_file, sep="\t")
    print(f"  Loaded {source1_file}: {len(s1):,} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    s2 = pd.read_csv(source_dir / source2_file, sep="\t")
    print(f"  Loaded {source2_file}: {len(s2):,} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    s3 = pd.read_csv(source_dir / source3_file, sep="\t")
    print(f"  Loaded {source3_file}: {len(s3):,} rows ({time.time()-t0:.1f}s)")

    # Preprocess
    print("  Preprocessing S1 ...")
    s1 = preprocess_dataframe(s1)
    print("  Preprocessing S2 ...")
    s2 = preprocess_dataframe(s2)
    print("  Preprocessing S3 ...")
    s3 = preprocess_dataframe(s3)

    targets = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    print(f"  [OK] S1: {len(s1):,}  |  Targets (S2+S3): {len(targets):,}")

    # Save checkpoints
    s1.to_parquet(ckpt_s1, index=False)
    targets.to_parquet(ckpt_tgt, index=False)
    print(f"  [OK] Saved checkpoints")

    return s1, targets


# ===========================================================================
# STEP 2 -- Validation Split
# ===========================================================================
def load_ground_truth() -> Dict[str, Set[str]]:
    """Load training ground truth into a {s1_id: set(matched_ids)} dict."""
    gt_path = config.TRAIN_DIR / config.TRAIN_GROUND_TRUTH
    gt = pd.read_csv(gt_path, sep="\t")
    truth: Dict[str, Set[str]] = {}
    for s1_id, matches_str in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        m = str(matches_str)
        if m and m != "nan":
            truth[s1_id] = set(m.split(","))
        else:
            truth[s1_id] = set()
    return truth


def create_validation_split(
    s1_df: pd.DataFrame, ground_truth: Dict[str, Set[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split S1 entities into train/val folds. Returns
    (train_s1_df, val_s1_df, train_gt, val_gt).
    """
    print(f"\n{'='*70}")
    print("STEP 2: Creating validation split ...")
    print(f"{'='*70}")

    np.random.seed(config.RANDOM_SEED)
    all_s1_ids = s1_df["entity_id"].values
    np.random.shuffle(all_s1_ids)

    split_idx = int(len(all_s1_ids) * (1 - config.VAL_SPLIT_RATIO))
    train_ids = set(all_s1_ids[:split_idx])
    val_ids = set(all_s1_ids[split_idx:])

    train_s1 = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_s1 = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)

    train_gt = {k: v for k, v in ground_truth.items() if k in train_ids}
    val_gt = {k: v for k, v in ground_truth.items() if k in val_ids}

    n_train_matches = sum(1 for v in train_gt.values() if v)
    n_val_matches = sum(1 for v in val_gt.values() if v)
    print(f"  Train: {len(train_s1):,} entities ({n_train_matches:,} with matches)")
    print(f"  Val:   {len(val_s1):,} entities ({n_val_matches:,} with matches)")

    return train_s1, val_s1, train_gt, val_gt


# ===========================================================================
# STEP 3 -- Blocking
# ===========================================================================
def run_blocking(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    label: str = "val",
) -> Dict[str, List[Tuple[str, float]]]:
    """Run multi-pass blocking and return candidates with scores."""
    print(f"\n{'='*70}")
    print(f"STEP 3: Multi-pass blocking ({label}) ...")
    print(f"{'='*70}")

    ckpt = config.CHECKPOINT_DIR / f"{label}_candidates.pkl"
    if ckpt.exists():
        print(f"  [OK] Loading from checkpoint: {ckpt.name}")
        with open(ckpt, "rb") as f:
            candidates = pickle.load(f)
        total = sum(len(v) for v in candidates.values())
        print(f"    {len(candidates):,} entities, {total:,} total pairs")
        return candidates

    blocker = MultiPassBlocker(
        top_k=config.TOP_K_CANDIDATES,
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.CHAR_NGRAM_RANGE,
        chunk_size=config.BLOCKING_CHUNK_SIZE,
        min_score=config.MIN_TFIDF_SCORE,
    )

    print("  Fitting TF-IDF index on targets ...")
    blocker.fit(target_df)

    print("  Generating candidates ...")
    candidates = blocker.generate_candidates(query_df, target_df)

    # Save checkpoint
    with open(ckpt, "wb") as f:
        pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [OK] Saved checkpoint: {ckpt.name}")

    # Free the blocker's large matrices
    del blocker
    gc.collect()

    return candidates


# ===========================================================================
# STEP 4 -- Blocking Recall
# ===========================================================================
def measure_blocking_recall(
    candidates: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """Compute blocking recall = |true ∩ candidates| / |true|."""
    print(f"\n{'='*70}")
    print("STEP 4: Measuring blocking recall ...")
    print(f"{'='*70}")

    total_true = 0
    total_found = 0

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cand_ids = set(cid for cid, _ in candidates.get(s1_id, []))
        total_true += len(true_matches)
        total_found += len(true_matches & cand_ids)

    recall = total_found / max(total_true, 1)
    print(f"  True match pairs: {total_true:,}")
    print(f"  Found in candidates: {total_found:,}")
    print(f"  * Blocking Recall: {recall:.4f} ({recall*100:.2f}%)")

    if recall < 0.95:
        print("  [!] WARNING: Blocking recall < 95% -- increase TOP_K or add more passes!")
    elif recall < 0.97:
        print("  [!] Blocking recall < 97% -- consider increasing TOP_K for better results")
    else:
        print("  [OK] Blocking recall ≥ 97% -- excellent!")

    return recall


# ===========================================================================
# STEP 5 -- Feature Extraction
# ===========================================================================
def extract_features_with_labels(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidates: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    label: str = "train",
) -> pd.DataFrame:
    """Build feature matrix for candidate pairs with ground-truth labels."""
    print(f"\n{'='*70}")
    print(f"STEP 5: Feature extraction ({label}) ...")
    print(f"{'='*70}")

    ckpt = config.CHECKPOINT_DIR / f"{label}_features.parquet"
    if ckpt.exists():
        print(f"  [OK] Loading from checkpoint: {ckpt.name}")
        return pd.read_parquet(ckpt)

    # Build flat list of (s1_id, cand_id, blocking_score)
    pairs: List[Tuple[str, str, float]] = []
    for s1_id, cand_list in candidates.items():
        for cand_id, score in cand_list:
            pairs.append((s1_id, cand_id, score))

    print(f"  Total pairs to featurize: {len(pairs):,}")

    feature_df = build_feature_matrix(
        query_df, target_df, pairs,
        batch_size=config.FEATURE_BATCH_SIZE,
    )

    # Add labels from ground truth
    def _get_label(row):
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        true_matches = ground_truth.get(s1_id, set())
        return 1 if cand_id in true_matches else 0

    feature_df["label"] = feature_df.apply(_get_label, axis=1)

    n_pos = (feature_df["label"] == 1).sum()
    n_neg = (feature_df["label"] == 0).sum()
    print(f"  Positives: {n_pos:,}  |  Negatives: {n_neg:,}  |  "
          f"Ratio: 1:{n_neg / max(n_pos, 1):.1f}")

    feature_df.to_parquet(ckpt, index=False)
    print(f"  [OK] Saved checkpoint: {ckpt.name}")

    return feature_df


def extract_features_unlabeled(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidates: Dict[str, List[Tuple[str, float]]],
    label: str = "test",
) -> pd.DataFrame:
    """Build feature matrix for candidate pairs WITHOUT labels (test set)."""
    print(f"\n{'='*70}")
    print(f"STEP 5: Feature extraction ({label}, unlabeled) ...")
    print(f"{'='*70}")

    ckpt = config.CHECKPOINT_DIR / f"{label}_features.parquet"
    if ckpt.exists():
        print(f"  [OK] Loading from checkpoint: {ckpt.name}")
        return pd.read_parquet(ckpt)

    pairs: List[Tuple[str, str, float]] = []
    for s1_id, cand_list in candidates.items():
        for cand_id, score in cand_list:
            pairs.append((s1_id, cand_id, score))

    print(f"  Total pairs to featurize: {len(pairs):,}")

    feature_df = build_feature_matrix(
        query_df, target_df, pairs,
        batch_size=config.FEATURE_BATCH_SIZE,
    )

    feature_df.to_parquet(ckpt, index=False)
    print(f"  [OK] Saved checkpoint: {ckpt.name}")

    return feature_df


# ===========================================================================
# STEP 6 -- Train Classifier (LightGBM / CatBoost / XGBoost / Ensemble)
# ===========================================================================
def train_model(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    model_type: str = "lightgbm",
):
    """Train the specified model (lightgbm, catboost, xgboost, or ensemble)."""
    print(f"\n{'='*70}")
    print(f"STEP 6: Training {model_type.upper()} classifier ...")
    print(f"{'='*70}")

    ckpt = config.CHECKPOINT_DIR / f"{model_type}_model.pkl"
    if ckpt.exists():
        print(f"  [OK] Loading model from checkpoint: {ckpt.name}")
        with open(ckpt, "rb") as f:
            model = pickle.load(f)
        return model

    if model_type == "lightgbm":
        model = train_lightgbm(train_features, val_features, save_path=ckpt)
    elif model_type == "catboost":
        model = train_catboost(train_features, val_features, save_path=ckpt)
    elif model_type == "xgboost":
        model = train_xgboost(train_features, val_features, save_path=ckpt)
    elif model_type == "ensemble":
        print("  Training base models for ensemble blend (0.5 LGB + 0.3 CB + 0.2 XGB) ...")
        lgb_model = train_lightgbm(train_features, val_features)
        cb_model = train_catboost(train_features, val_features)
        xgb_model = train_xgboost(train_features, val_features)
        model = EnsembleClassifier([
            ("lightgbm", lgb_model, 0.5),
            ("catboost", cb_model, 0.3),
            ("xgboost", xgb_model, 0.2),
        ])
        model.save(ckpt)
        print(f"  [OK] Saved ensemble model checkpoint: {ckpt.name}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


# ===========================================================================
# STEP 7 -- Threshold Optimization
# ===========================================================================
def optimize_threshold(
    model,
    val_features: pd.DataFrame,
    val_ground_truth: Dict[str, Set[str]],
    all_val_s1_ids: Set[str],
) -> float:
    """Sweep threshold to maximise Macro F0.5 on the validation set."""
    print(f"\n{'='*70}")
    print("STEP 7: Optimizing decision threshold ...")
    print(f"{'='*70}")

    X_val = val_features[FEATURE_COLUMNS].values
    probs = model.predict_proba(X_val)[:, 1]

    # Pre-extract arrays for vectorized threshold sweep
    s1_ids_arr = val_features["source1_entity_id"].values
    cand_ids_arr = val_features["candidate_entity_id"].values

    best_score = -1.0
    best_threshold = 0.5
    results = []

    thresholds = np.arange(
        config.THRESHOLD_SCAN_MIN,
        config.THRESHOLD_SCAN_MAX + config.THRESHOLD_SCAN_STEP,
        config.THRESHOLD_SCAN_STEP,
    )

    for T in tqdm(thresholds, desc="Threshold sweep"):
        # Vectorized: find all pairs above threshold
        mask = probs >= T
        matched_s1 = s1_ids_arr[mask]
        matched_cand = cand_ids_arr[mask]

        # Build predictions dict
        predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_val_s1_ids}
        for s1_id, cand_id in zip(matched_s1, matched_cand):
            predictions[s1_id].add(cand_id)

        metrics = evaluate_predictions(val_ground_truth, predictions, beta=config.BETA)
        f05 = metrics["macro_f_beta"]
        results.append((T, f05, metrics["macro_precision"], metrics["macro_recall"]))

        if f05 > best_score:
            best_score = f05
            best_threshold = T

    # Print top-5 thresholds
    results.sort(key=lambda x: -x[1])
    print(f"\n  Top-5 thresholds:")
    print(f"  {'Threshold':>10s} {'F0.5':>8s} {'Precision':>10s} {'Recall':>8s}")
    print(f"  {'-'*40}")
    for T, f05, prec, rec in results[:5]:
        marker = " <-- BEST" if T == best_threshold else ""
        print(f"  {T:10.3f} {f05:8.4f} {prec:10.4f} {rec:8.4f}{marker}")

    print(f"\n  * Optimal Threshold: {best_threshold:.3f}  ->  Macro F0.5 = {best_score:.4f}")

    return best_threshold


# ===========================================================================
# STEP 8 -- Generate Predictions
# ===========================================================================
def generate_predictions(
    model,
    feature_df: pd.DataFrame,
    all_s1_ids: List[str],
    threshold: float,
    output_dir: Path,
    candidates_map: Dict[str, List[Tuple[str, float]]],
):
    """Apply the trained model to generate final predictions and write
    ``matching_results.tsv`` and ``candidate_pairs.tsv``.
    """
    print(f"\n{'='*70}")
    print("STEP 8: Generating predictions & output files ...")
    print(f"{'='*70}")

    config.create_dirs()

    X = feature_df[FEATURE_COLUMNS].values
    probs = model.predict_proba(X)[:, 1]
    feature_df = feature_df.copy()
    feature_df["prob"] = probs

    # Build predictions
    predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}
    matched = feature_df[feature_df["prob"] >= threshold]
    for _, row in matched.iterrows():
        predictions[row["source1_entity_id"]].add(row["candidate_entity_id"])

    n_with_matches = sum(1 for v in predictions.values() if v)
    n_singletons = sum(1 for v in predictions.values() if not v)
    total_matches = sum(len(v) for v in predictions.values())
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Entities with matches: {n_with_matches:,}")
    print(f"  Predicted singletons: {n_singletons:,}")
    print(f"  Total match pairs: {total_matches:,}")

    # Write matching_results.tsv
    matching_path = output_dir / config.MATCHING_RESULTS_FILE
    rows = []
    for s1_id in all_s1_ids:
        match_ids = sorted(predictions.get(s1_id, set()))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(match_ids) if match_ids else "",
        })
    matching_df = pd.DataFrame(rows)
    matching_df.to_csv(matching_path, sep="\t", index=False)
    print(f"  [OK] Wrote {matching_path} ({len(matching_df):,} rows)")

    # Write candidate_pairs.tsv
    cand_df = format_candidate_pairs_dataframe(candidates_map)
    cand_path = output_dir / config.CANDIDATE_PAIRS_FILE
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f"  [OK] Wrote {cand_path} ({len(cand_df):,} rows)")

    return predictions


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument(
        "--mode", choices=["full", "val_only", "test_only"], default="full",
        help="full = train+val+test, val_only = train+val only, test_only = test inference only",
    )
    parser.add_argument(
        "--model", choices=["lightgbm", "catboost", "xgboost", "ensemble"], default="lightgbm",
        help="Model architecture: lightgbm (default), catboost, xgboost, or ensemble",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run comparative benchmark across LightGBM, CatBoost, XGBoost, and Ensemble on validation fold",
    )
    parser.add_argument("--clear-checkpoints", action="store_true",
                        help="Delete all checkpoints before running")
    args = parser.parse_args()

    config.create_dirs()

    if args.clear_checkpoints and config.CHECKPOINT_DIR.exists():
        import shutil
        shutil.rmtree(config.CHECKPOINT_DIR)
        print("  [OK] Cleared checkpoints")
        config.create_dirs()

    start_time = time.time()

    # -- Training Phase ------------------------------------------------
    if args.mode in ("full", "val_only"):
        # Step 1: Load & preprocess training data
        train_s1, train_targets = load_and_preprocess(
            config.TRAIN_DIR,
            config.TRAIN_SOURCE1, config.TRAIN_SOURCE2, config.TRAIN_SOURCE3,
            label="train",
        )

        # Step 2: Validation split
        ground_truth = load_ground_truth()
        train_s1_split, val_s1, train_gt, val_gt = create_validation_split(
            train_s1, ground_truth
        )
        del train_s1  # free full train s1 as we only need the splits
        gc.collect()

        # Step 3: Blocking on training fold
        print("\n-- Blocking on TRAINING fold --")
        train_candidates = run_blocking(train_s1_split, train_targets, label="train_fold")

        # Step 3b: Blocking on validation fold
        print("\n-- Blocking on VALIDATION fold --")
        val_candidates = run_blocking(val_s1, train_targets, label="val_fold")

        # Step 4: Blocking recall on validation
        val_recall = measure_blocking_recall(val_candidates, val_gt)

        # Step 5: Feature extraction
        train_features = extract_features_with_labels(
            train_s1_split, train_targets, train_candidates, train_gt, label="train_fold"
        )
        val_features = extract_features_with_labels(
            val_s1, train_targets, val_candidates, val_gt, label="val_fold"
        )

        all_val_ids = set(val_s1["entity_id"])

        # Optional: Run comparative benchmark
        if args.benchmark:
            print("\n-- Running Comparative Model Benchmark --")
            benchmark_models(train_features, val_features, val_gt, all_val_ids)

        # Step 6: Train model
        model = train_model(train_features, val_features, model_type=args.model)

        # Step 7: Threshold optimization
        best_threshold = optimize_threshold(model, val_features, val_gt, all_val_ids)

        # Save threshold
        with open(config.CHECKPOINT_DIR / "best_threshold.txt", "w") as f:
            f.write(f"{best_threshold:.4f}")

        print(f"\n  Training phase complete in {(time.time()-start_time)/60:.1f} minutes")

        if args.mode == "val_only":
            print("\n✅ Validation-only mode complete. Run with --mode test_only for inference.")
            return

        # Free training data
        del train_s1_split, val_s1, train_targets, train_candidates, val_candidates
        del train_features, val_features
        gc.collect()

    # -- Test Inference Phase ------------------------------------------
    if args.mode in ("full", "test_only"):
        if args.mode == "test_only":
            # Load model and threshold from checkpoints
            model_path = config.CHECKPOINT_DIR / f"{args.model}_model.pkl"
            if not model_path.exists():
                model_path = config.CHECKPOINT_DIR / "lgb_model.pkl"
            thresh_path = config.CHECKPOINT_DIR / "best_threshold.txt"
            if not model_path.exists() or not thresh_path.exists():
                print(f"ERROR: Model {model_path.name} not found. Run training first.")
                sys.exit(1)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(thresh_path, "r") as f:
                best_threshold = float(f.read().strip())
            print(f"  Loaded model ({model_path.name}) and threshold ({best_threshold:.3f}) from checkpoints")

        # Step 1: Load & preprocess test data
        test_s1, test_targets = load_and_preprocess(
            config.TEST_DIR,
            config.TEST_SOURCE1, config.TEST_SOURCE2, config.TEST_SOURCE3,
            label="test",
        )

        # Step 3: Blocking on test
        test_candidates = run_blocking(test_s1, test_targets, label="test")

        # Step 5: Feature extraction (unlabeled)
        test_features = extract_features_unlabeled(
            test_s1, test_targets, test_candidates, label="test"
        )

        # Step 8: Generate predictions
        all_test_s1_ids = test_s1["entity_id"].tolist()
        generate_predictions(
            model, test_features, all_test_s1_ids, best_threshold,
            config.OUTPUT_DIR, test_candidates,
        )

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"✅ Pipeline complete in {elapsed:.1f} minutes")
    print(f"{'='*70}")
    print(f"  Output files:")
    print(f"    {config.OUTPUT_DIR / config.MATCHING_RESULTS_FILE}")
    print(f"    {config.OUTPUT_DIR / config.CANDIDATE_PAIRS_FILE}")
    print(f"\n  Next: validate with:")
    print(f"    python utils/validate_submission.py "
          f"--matching output/matching_results.tsv "
          f"--candidate output/candidate_pairs.tsv "
          f"--test-dir dataset/test")


if __name__ == "__main__":
    main()
