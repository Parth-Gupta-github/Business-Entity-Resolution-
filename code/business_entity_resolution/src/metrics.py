"""
Evaluation metrics matching the Amazon ML Challenge 2026 specifications.
Computes Macro-averaged F_0.5 score across all Source 1 entities,
properly accounting for singletons (entities with no true matches).
"""

from typing import Dict, Set, Union, List


def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculates F_beta score given precision and recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes Macro-averaged F_beta score for Business Entity Resolution.

    Parameters:
    -----------
    ground_truth : Dict[str, Set[str]]
        Mapping of source1_entity_id -> set of true matching entity IDs (S2/S3).
        Singletons have empty sets.
    predictions : Dict[str, Set[str]]
        Mapping of source1_entity_id -> set of predicted matching entity IDs (S2/S3).
        Empty set signifies predicted singleton.
    beta : float
        Default is 0.5 (Precision-heavy).

    Returns:
    --------
    Dict[str, float] containing:
        - macro_f_beta
        - macro_precision
        - macro_recall
        - total_entities
        - singleton_accuracy
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {
            "macro_f_beta": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "total_entities": 0,
            "singleton_accuracy": 0.0,
        }

    f_beta_scores = []
    precisions = []
    recalls = []
    singleton_correct = 0
    total_singletons = 0

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())

        # Check singleton case
        if len(true_matches) == 0:
            total_singletons += 1
            if len(pred_matches) == 0:
                # Correct singleton prediction earns full 1.0
                f_beta_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
                singleton_correct += 1
            else:
                # False positive merge on a singleton scores 0.0
                f_beta_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            continue

        # Non-singleton case
        if len(pred_matches) == 0:
            # False negative (missed all matches)
            f_beta_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        # True positives
        tp = len(true_matches.intersection(pred_matches))
        precision = tp / len(pred_matches)
        recall = tp / len(true_matches)

        fb = calculate_f_beta(precision, recall, beta=beta)
        f_beta_scores.append(fb)
        precisions.append(precision)
        recalls.append(recall)

    macro_f_beta = sum(f_beta_scores) / total_entities
    macro_precision = sum(precisions) / total_entities
    macro_recall = sum(recalls) / total_entities
    singleton_acc = (singleton_correct / total_singletons) if total_singletons > 0 else 1.0

    return {
        "macro_f_beta": macro_f_beta,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "total_entities": total_entities,
        "singleton_accuracy": singleton_acc,
    }
