"""
Official Macro F0.5 Accuracy & Evaluation Tool.
===============================================
Calculates the exact competition evaluation metric (Macro-averaged F0.5)
between a predicted matching_results.tsv and the ground truth TSV.

Competition Evaluation Metric Definition:
-----------------------------------------
For each Source 1 query entity 's':
  - Let G_s be the set of true target IDs.
  - Let P_s be the set of predicted target IDs.
  
  Precision(G_s, P_s) = |G_s ∩ P_s| / |P_s| if |P_s| > 0 else (1.0 if |G_s| == 0 else 0.0)
  Recall(G_s, P_s)    = |G_s ∩ P_s| / |G_s| if |G_s| > 0 else (1.0 if |P_s| == 0 else 0.0)
  
  F_0.5(G_s, P_s)     = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
                        with special case: if G_s == ∅ and P_s == ∅ -> F_0.5 = 1.0 (Correct Singleton)
                        if G_s == ∅ and P_s != ∅ -> F_0.5 = 0.0 (False Positive on Singleton)

Macro F_0.5 = (1 / |S_1|) * Σ_{s ∈ S_1} F_0.5(G_s, P_s)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Set
import pandas as pd
import numpy as np

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.metrics import evaluate_predictions, calculate_f_beta


def load_ground_truth_map(gt_path: Path) -> Dict[str, Set[str]]:
    """Loads ground truth mapping from TSV file."""
    df = pd.read_csv(gt_path, sep="\t", dtype=str)
    s1_col = "source1_entity_id" if "source1_entity_id" in df.columns else df.columns[0]
    match_col = "matched_entity_ids" if "matched_entity_ids" in df.columns else df.columns[1]

    gt_map: Dict[str, Set[str]] = {}
    for s1_id, match_str in zip(df[s1_col], df[match_col]):
        if pd.isna(match_str) or not str(match_str).strip():
            gt_map[str(s1_id)] = set()
        else:
            gt_map[str(s1_id)] = set(str(match_str).strip().split())
    return gt_map


def load_predictions_map(pred_path: Path) -> Dict[str, Set[str]]:
    """Loads predictions mapping from TSV file."""
    df = pd.read_csv(pred_path, sep="\t", dtype=str)
    s1_col = "source1_entity_id" if "source1_entity_id" in df.columns else df.columns[0]
    match_col = "matched_entity_ids" if "matched_entity_ids" in df.columns else df.columns[1]

    pred_map: Dict[str, Set[str]] = {}
    for s1_id, match_str in zip(df[s1_col], df[match_col]):
        if pd.isna(match_str) or not str(match_str).strip():
            pred_map[str(s1_id)] = set()
        else:
            pred_map[str(s1_id)] = set(str(match_str).strip().split())
    return pred_map


def evaluate_file(ground_truth_tsv: Path, predictions_tsv: Path, subset_only: bool = False):
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — MACRO F0.5 ACCURACY EVALUATION")
    print("=" * 70)
    print(f"Ground Truth File: {ground_truth_tsv}")
    print(f"Predictions File : {predictions_tsv}")

    gt_map = load_ground_truth_map(ground_truth_tsv)
    pred_map = load_predictions_map(predictions_tsv)

    print(f"\nTotal Ground Truth Entities: {len(gt_map):,}")
    print(f"Total Predicted Entities   : {len(pred_map):,}")

    if subset_only or len(pred_map) < len(gt_map):
        eval_ids = set(pred_map.keys())
        print(f"Evaluating across predicted subset: {len(eval_ids):,} entities")
    else:
        eval_ids = set(gt_map.keys())

    aligned_gt = {s1: gt_map.get(s1, set()) for s1 in eval_ids}
    aligned_pred = {s1: pred_map.get(s1, set()) for s1 in eval_ids}

    # Evaluate
    metrics = evaluate_predictions(aligned_gt, aligned_pred, beta=0.5)

    print("\n" + "=" * 70)
    print(f"  OFFICIAL SCORE (Macro F0.5): {metrics['macro_f_beta']:.4f} ({metrics['macro_f_beta']*100:.2f}%)")
    print("=" * 70)
    print(f"  - Macro Precision    : {metrics['macro_precision']:.4f} ({metrics['macro_precision']*100:.2f}%)")
    print(f"  - Macro Recall       : {metrics['macro_recall']:.4f} ({metrics['macro_recall']*100:.2f}%)")
    print(f"  - Singleton Accuracy : {metrics['singleton_accuracy']:.4f} ({metrics['singleton_accuracy']*100:.2f}%)")
    print(f"  - Evaluated Entities : {metrics['total_entities']:,}")
    print("=" * 70)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction TSV against ground truth")
    parser.add_argument("--gt", type=str, default=str(PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"))
    parser.add_argument("--pred", type=str, required=False, default=str(PROJECT_ROOT / "output" / "verify_test" / "matching_results.tsv"))
    parser.add_argument("--subset", action="store_true", help="Evaluate strictly on predicted entities subset")
    args = parser.parse_args()

    evaluate_file(Path(args.gt), Path(args.pred), subset_only=args.subset)
