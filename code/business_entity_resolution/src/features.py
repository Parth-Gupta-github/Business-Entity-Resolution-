"""
Pairwise Feature Extraction Module.
Computes string similarity, token overlap, address metrics,
and statistical indicators between query and candidate pairs.

Feature count: ~25 numerical features per pair.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

LEGAL_SUFFIX_TOKENS = frozenset({
    "corporation", "incorporated", "limited", "private", "company",
    "liability", "partnership", "gmbh", "department", "manufacturing",
    "international", "technology", "solutions", "services", "sarl",
    "sa", "sas", "sasu", "eurl", "sei", "sci", "snc",
})
ADDRESS_UNIT_MARKERS = frozenset({"apartment", "suite", "unit", "floor", "building"})


def _unit_identifier(address: str) -> str:
    tokens = address.split()
    for index, token in enumerate(tokens[:-1]):
        if token in ADDRESS_UNIT_MARKERS:
            return tokens[index + 1]
    return ""


def extract_pair_features(
    row1: Dict[str, Any],
    row2: Dict[str, Any],
    blocking_score: float = 0.0,
) -> Dict[str, float]:
    """
    Extracts numerical similarity features for a single (Source 1, Candidate) pair.
    ``blocking_score`` is the TF-IDF cosine similarity from the blocking stage.
    """
    name1 = str(row1.get("clean_name", ""))
    name2 = str(row2.get("clean_name", ""))
    addr1 = str(row1.get("clean_address", ""))
    addr2 = str(row2.get("clean_address", ""))
    post1 = str(row1.get("postal_code", ""))
    post2 = str(row2.get("postal_code", ""))
    country1 = str(row1.get("country", ""))
    country2 = str(row2.get("country", ""))
    sn1 = str(row1.get("street_number", ""))
    sn2 = str(row2.get("street_number", ""))

    features: Dict[str, float] = {}

    # ── 0. Blocking score (free, very informative) ────────────────────
    features["blocking_cosine"] = blocking_score

    # ── 1. Name similarities ─────────────────────────────────────────
    features["name_fuzz_ratio"] = fuzz.ratio(name1, name2) / 100.0
    features["name_partial_ratio"] = fuzz.partial_ratio(name1, name2) / 100.0
    features["name_token_sort_ratio"] = fuzz.token_sort_ratio(name1, name2) / 100.0
    features["name_token_set_ratio"] = fuzz.token_set_ratio(name1, name2) / 100.0
    features["name_w_ratio"] = fuzz.WRatio(name1, name2) / 100.0
    features["name_jaro_winkler"] = distance.JaroWinkler.similarity(name1, name2)
    features["name_exact_match"] = 1.0 if (name1 and name1 == name2) else 0.0

    # ── 2. Address similarities ──────────────────────────────────────
    features["addr_fuzz_ratio"] = fuzz.ratio(addr1, addr2) / 100.0
    features["addr_partial_ratio"] = fuzz.partial_ratio(addr1, addr2) / 100.0
    features["addr_token_sort_ratio"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    features["addr_token_set_ratio"] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    features["addr_jaro_winkler"] = distance.JaroWinkler.similarity(addr1, addr2)

    # ── 3. Combined text similarity ──────────────────────────────────
    comb1 = name1 + " " + addr1
    comb2 = name2 + " " + addr2
    features["combined_fuzz_ratio"] = fuzz.ratio(comb1, comb2) / 100.0

    # ── 4. Token overlap / Jaccard ───────────────────────────────────
    name_tokens1 = set(name1.split())
    name_tokens2 = set(name2.split())
    all_tokens1 = set(comb1.split())
    all_tokens2 = set(comb2.split())

    # Name-level token metrics
    if name_tokens1 or name_tokens2:
        features["name_token_jaccard"] = (
            len(name_tokens1 & name_tokens2) / len(name_tokens1 | name_tokens2)
        )
    else:
        features["name_token_jaccard"] = 0.0
    features["name_common_token_count"] = float(len(name_tokens1 & name_tokens2))
    features["name_sorted_equal"] = 1.0 if (
        sorted(name_tokens1) == sorted(name_tokens2) and name_tokens1
    ) else 0.0

    # First token / anchor match & token containment ratio
    first1 = name1.split()[0] if name1.split() else ""
    first2 = name2.split()[0] if name2.split() else ""
    features["name_first_token_match"] = 1.0 if (first1 and first1 == first2) else 0.0
    shorter_tokens = min(len(name_tokens1), len(name_tokens2))
    features["name_token_overlap_ratio"] = (
        len(name_tokens1 & name_tokens2) / shorter_tokens if shorter_tokens > 0 else 0.0
    )

    core_name_tokens1 = name_tokens1 - LEGAL_SUFFIX_TOKENS
    core_name_tokens2 = name_tokens2 - LEGAL_SUFFIX_TOKENS
    core_name_union = core_name_tokens1 | core_name_tokens2
    core_name_common = core_name_tokens1 & core_name_tokens2
    features["name_core_token_jaccard"] = (
        len(core_name_common) / len(core_name_union) if core_name_union else 0.0
    )
    core_name_shorter = min(len(core_name_tokens1), len(core_name_tokens2))
    features["name_core_token_overlap_ratio"] = (
        len(core_name_common) / core_name_shorter if core_name_shorter else 0.0
    )

    # Character 3-gram Jaccard
    g1 = set(name1[i:i+3] for i in range(len(name1)-2)) if len(name1) >= 3 else (set([name1]) if name1 else set())
    g2 = set(name2[i:i+3] for i in range(len(name2)-2)) if len(name2) >= 3 else (set([name2]) if name2 else set())
    features["name_char_3gram_jaccard"] = (
        len(g1 & g2) / len(g1 | g2) if (g1 or g2) else 0.0
    )

    # Combined-level Jaccard
    if all_tokens1 or all_tokens2:
        features["token_jaccard"] = (
            len(all_tokens1 & all_tokens2) / len(all_tokens1 | all_tokens2)
        )
    else:
        features["token_jaccard"] = 0.0
    features["addr_common_token_count"] = float(
        len(set(addr1.split()) & set(addr2.split()))
    )

    address_numbers1 = {
        token for token in addr1.split()
        if token.isdigit() and token != post1
    }
    address_numbers2 = {
        token for token in addr2.split()
        if token.isdigit() and token != post2
    }
    address_number_union = address_numbers1 | address_numbers2
    features["addr_number_token_jaccard"] = (
        len(address_numbers1 & address_numbers2) / len(address_number_union)
        if address_number_union else 0.0
    )

    unit1 = _unit_identifier(addr1)
    unit2 = _unit_identifier(addr2)
    features["address_unit_match"] = float(bool(unit1 and unit2 and unit1 == unit2))
    features["address_unit_mismatch"] = float(bool(unit1 and unit2 and unit1 != unit2))

    # ── 5. Postal code & location ────────────────────────────────────
    if post1 and post2:
        features["postal_exact_match"] = 1.0 if post1 == post2 else 0.0
        features["postal_prefix3_match"] = 1.0 if post1[:3] == post2[:3] else 0.0
        features["postal_missing"] = 0.0
    else:
        features["postal_exact_match"] = 0.0
        features["postal_prefix3_match"] = 0.0
        features["postal_missing"] = 1.0

    # ── 6. Street number match & mismatch penalty ────────────────────
    if sn1 and sn2:
        features["street_number_match"] = 1.0 if sn1 == sn2 else 0.0
        features["addr_num_mismatch_penalty"] = 1.0 if sn1 != sn2 else 0.0
    else:
        features["street_number_match"] = 0.0
        features["addr_num_mismatch_penalty"] = 0.0

    # ── 7. Country check & mismatch penalty ──────────────────────────
    features["country_match"] = 1.0 if (
        country1 == country2 or not country1 or not country2
    ) else 0.0
    features["country_mismatch_penalty"] = 1.0 if (
        country1 and country2 and country1 != country2
    ) else 0.0

    # ── 8. Length / ratio features ────────────────────────────────────
    features["name_len_diff"] = abs(len(name1) - len(name2))
    features["addr_len_diff"] = abs(len(addr1) - len(addr2))
    features["name_len_ratio"] = (
        min(len(name1), len(name2)) / max(len(name1), len(name2), 1)
    )

    return features


# ── Feature column names (in deterministic order) ────────────────────────
FEATURE_COLUMNS = [
    "blocking_cosine",
    "name_fuzz_ratio", "name_partial_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_w_ratio", "name_jaro_winkler",
    "name_exact_match", "name_first_token_match", "name_token_overlap_ratio",
    "name_core_token_jaccard", "name_core_token_overlap_ratio",
    "name_char_3gram_jaccard",
    "addr_fuzz_ratio", "addr_partial_ratio", "addr_token_sort_ratio",
    "addr_token_set_ratio", "addr_jaro_winkler",
    "combined_fuzz_ratio",
    "name_token_jaccard", "name_common_token_count", "name_sorted_equal",
    "token_jaccard", "addr_common_token_count", "addr_number_token_jaccard",
    "address_unit_match", "address_unit_mismatch",
    "postal_exact_match", "postal_prefix3_match", "postal_missing",
    "street_number_match", "addr_num_mismatch_penalty",
    "country_match", "country_mismatch_penalty",
    "name_len_diff", "addr_len_diff", "name_len_ratio",
]


def build_feature_matrix(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_pairs: List[Tuple[str, str, float]],
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """
    Builds a complete feature DataFrame for a list of
    ``(source1_id, candidate_id, blocking_score)`` triples.

    Processes in batches for memory efficiency on large candidate lists.
    """
    from tqdm import tqdm

    q_map = query_df.set_index("entity_id").to_dict(orient="index")
    t_map = target_df.set_index("entity_id").to_dict(orient="index")

    all_rows = []
    n = len(candidate_pairs)

    for start in tqdm(range(0, n, batch_size), desc="Feature extraction", unit="batch"):
        end = min(start + batch_size, n)
        batch = candidate_pairs[start:end]

        for s1_id, cand_id, blk_score in batch:
            if s1_id in q_map and cand_id in t_map:
                feat = extract_pair_features(
                    q_map[s1_id], t_map[cand_id], blocking_score=blk_score
                )
                feat["source1_entity_id"] = s1_id
                feat["candidate_entity_id"] = cand_id
                all_rows.append(feat)

    return pd.DataFrame(all_rows)
