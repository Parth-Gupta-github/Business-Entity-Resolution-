"""
Pairwise Feature Extraction Module.
Computes string similarity, token overlap, address metrics,
and statistical indicators between query and candidate pairs.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance


def extract_pair_features(row1: pd.Series, row2: pd.Series) -> Dict[str, float]:
    """
    Extracts numerical similarity features for a single (Source 1, Candidate) pair.
    """
    name1 = str(row1.get("clean_name", ""))
    name2 = str(row2.get("clean_name", ""))
    addr1 = str(row1.get("clean_address", ""))
    addr2 = str(row2.get("clean_address", ""))
    post1 = str(row1.get("postal_code", ""))
    post2 = str(row2.get("postal_code", ""))
    country1 = str(row1.get("country", ""))
    country2 = str(row2.get("country", ""))

    features = {}

    # 1. Name Similarities
    features["name_fuzz_ratio"] = fuzz.ratio(name1, name2) / 100.0
    features["name_token_sort_ratio"] = fuzz.token_sort_ratio(name1, name2) / 100.0
    features["name_token_set_ratio"] = fuzz.token_set_ratio(name1, name2) / 100.0
    features["name_w_ratio"] = fuzz.WRatio(name1, name2) / 100.0
    features["name_jaro_winkler"] = distance.JaroWinkler.similarity(name1, name2)
    features["name_exact_match"] = 1.0 if (name1 and name1 == name2) else 0.0

    # 2. Address Similarities
    features["addr_fuzz_ratio"] = fuzz.ratio(addr1, addr2) / 100.0
    features["addr_token_sort_ratio"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    features["addr_token_set_ratio"] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    features["addr_jaro_winkler"] = distance.JaroWinkler.similarity(addr1, addr2)

    # 3. Token Overlap / Jaccard
    tokens1 = set(name1.split() + addr1.split())
    tokens2 = set(name2.split() + addr2.split())
    if tokens1 or tokens2:
        features["token_jaccard"] = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    else:
        features["token_jaccard"] = 0.0

    # 4. Postal Code & Location Checks
    if post1 and post2:
        features["postal_exact_match"] = 1.0 if post1 == post2 else 0.0
        features["postal_missing"] = 0.0
    else:
        features["postal_exact_match"] = 0.0
        features["postal_missing"] = 1.0

    # 5. Country check (1 if matching or unknown)
    features["country_match"] = 1.0 if (country1 == country2 or not country1 or not country2) else 0.0

    # 6. Length and Difference features
    features["name_len_diff"] = abs(len(name1) - len(name2))
    features["addr_len_diff"] = abs(len(addr1) - len(addr2))
    features["name_len_ratio"] = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)

    return features


def build_feature_matrix(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_pairs: List[tuple]
) -> pd.DataFrame:
    """
    Builds a complete feature DataFrame for a list of (source1_id, candidate_id) pairs.
    """
    q_map = query_df.set_index("entity_id").to_dict(orient="index")
    t_map = target_df.set_index("entity_id").to_dict(orient="index")

    rows = []
    for s1_id, cand_id in candidate_pairs:
        if s1_id in q_map and cand_id in t_map:
            feat = extract_pair_features(q_map[s1_id], t_map[cand_id])
            feat["source1_entity_id"] = s1_id
            feat["candidate_entity_id"] = cand_id
            rows.append(feat)

    return pd.DataFrame(rows)
