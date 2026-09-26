"""
Post-Processing & Match Conflict Resolution Module.
===================================================

Applies domain logic and calibrated filtering to binary candidate predictions:
1. Score-gap filtering (retains secondary matches only if confidence is within delta of top match)
2. Source-level deduplication (handles multiple candidates from same target source)
3. Singleton protection (clears predictions if max score is below confidence threshold)
"""

from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np


def resolve_matches_for_entity(
    candidate_predictions: List[Tuple[str, float]],
    threshold: float = 0.43,
    score_gap: float = 0.08,
    max_matches_per_entity: int = 2,
) -> List[str]:
    """
    Resolves matching target IDs for a single Source 1 entity query.

    Parameters:
    -----------
    candidate_predictions : List[Tuple[str, float]]
        List of (candidate_entity_id, probability) pairs for this S1 entity.
    threshold : float
        Decision threshold tau. Candidates below tau are rejected.
    score_gap : float
        Maximum allowed difference between top candidate probability and secondary candidates.
        Prevents adding weak secondary matches that trigger false-positive penalties.
    max_matches_per_entity : int
        Maximum number of target matches to assign (competition ground truth has avg 1.0 matches).

    Returns:
    --------
    List[str]
        List of selected matching candidate entity IDs (empty for singletons).
    """
    if not candidate_predictions:
        return []

    # Filter by threshold
    passing = [(cid, prob) for cid, prob in candidate_predictions if prob >= threshold]
    if not passing:
        return []

    # Sort descending by probability
    passing.sort(key=lambda x: -x[1])
    top_cid, top_prob = passing[0]
    selected = [top_cid]

    # Check secondary candidates with score-gap filtering
    for cid, prob in passing[1:]:
        if len(selected) >= max_matches_per_entity:
            break
        # Only include secondary match if its probability is close to the top match
        if (top_prob - prob) <= score_gap:
            selected.append(cid)

    return selected


def apply_postprocessing(
    predictions_df: pd.DataFrame,
    threshold: float = 0.43,
    score_gap: float = 0.08,
    max_matches_per_entity: int = 2,
    all_s1_ids: List[str] = None,
) -> Dict[str, List[str]]:
    """
    Applies post-processing across all candidate predictions in a DataFrame.

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with columns ['source1_entity_id', 'candidate_entity_id', 'probability'].
    threshold : float
        Classification threshold.
    score_gap : float
        Score gap parameter.
    all_s1_ids : List[str], optional
        Complete list of Source 1 entity IDs to ensure singletons are present.

    Returns:
    --------
    Dict[str, List[str]]
        Mapping of source1_entity_id -> list of matched candidate IDs.
    """
    from collections import defaultdict
    grouped = defaultdict(list)

    for s1_id, cand_id, prob in zip(
        predictions_df["source1_entity_id"],
        predictions_df["candidate_entity_id"],
        predictions_df["probability"],
    ):
        grouped[s1_id].append((cand_id, float(prob)))

    resolved: Dict[str, List[str]] = {}

    target_s1_ids = all_s1_ids if all_s1_ids is not None else grouped.keys()
    for s1_id in target_s1_ids:
        cands = grouped.get(s1_id, [])
        resolved[s1_id] = resolve_matches_for_entity(
            cands, threshold=threshold, score_gap=score_gap, max_matches_per_entity=max_matches_per_entity
        )

    return resolved
