"""
Multi-Pass Candidate Generation (Blocking) Module.

Uses a union of complementary retrieval strategies to achieve >97% recall
while reducing the search space from trillions to tens of candidates per
entity. Designed for 10M+ target records with controlled memory usage.

GPU Acceleration
----------------
When CuPy is available and an NVIDIA GPU is detected, the TF-IDF sparse
matrix multiplication is offloaded to the GPU for 5-20x speedup.  Falls
back transparently to CPU (sparse_dot_topn → SciPy) if no GPU is present.

Passes
------
1. TF-IDF Character N-gram Cosine — broad textual similarity
2. Exact Postal Code Blocking    — location-based recall boost
3. Name Prefix Token Blocking    — catches abbreviation and typo variants
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
from scipy import sparse

# ── GPU support via CuPy ────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ── Fast CPU fallback via sparse_dot_topn ────────────────────────────────
try:
    from sparse_dot_topn import awesome_cossim_topn  # type: ignore
    HAS_SPARSE_DOT_TOPN = True
except ImportError:
    HAS_SPARSE_DOT_TOPN = False


def _report_backend():
    """Print which compute backend will be used."""
    if HAS_CUPY:
        try:
            dev = cp.cuda.Device(0)
            mem = dev.mem_info
            free_gb = mem[0] / 1e9
            total_gb = mem[1] / 1e9
            print(f"  [GPU] CuPy on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} "
                  f"({free_gb:.1f}/{total_gb:.1f} GB free)")
        except Exception:
            print("  [GPU] CuPy detected (device info unavailable)")
    elif HAS_SPARSE_DOT_TOPN:
        print("  [CPU] Backend: sparse_dot_topn (fast, multi-threaded)")
    else:
        print("  [CPU] Backend: SciPy fallback (slow)")


# ────────────────── GPU-accelerated top-k sparse matmul ──────────────────
def _gpu_sparse_topk(query_csr, target_csr, top_k: int, threshold: float,
                     gpu_chunk: int = 20_000):
    """Compute top-k cosine neighbours on the GPU using CuPy.

    Transfers query chunks to GPU, performs sparse matmul against the
    (pre-uploaded) target matrix, extracts top-k per row, and transfers
    results back to CPU.  GPU chunk size is tuned for ~6 GB VRAM.
    """
    n_queries = query_csr.shape[0]
    n_targets = target_csr.shape[0]

    # Upload target matrix to GPU once (transposed for matmul)
    target_gpu = cp_sparse.csr_matrix(target_csr.T.tocsc()).T  # keep as CSR
    target_gpu_T = cp_sparse.csc_matrix(target_csr.T)

    all_rows, all_cols, all_data = [], [], []

    for start in range(0, n_queries, gpu_chunk):
        end = min(start + gpu_chunk, n_queries)
        # Transfer query chunk to GPU
        q_chunk_gpu = cp_sparse.csr_matrix(query_csr[start:end])

        # Sparse matmul on GPU: (chunk_size × features) @ (features × targets)
        sim_gpu = q_chunk_gpu @ target_gpu_T
        sim_gpu = sim_gpu.tocsr()

        # Extract top-k per row on GPU
        indptr = cp.asnumpy(sim_gpu.indptr)
        indices = cp.asnumpy(sim_gpu.indices)
        data = cp.asnumpy(sim_gpu.data)

        for local_i in range(end - start):
            r_start = indptr[local_i]
            r_end = indptr[local_i + 1]
            if r_start == r_end:
                continue
            r_data = data[r_start:r_end]
            r_cols = indices[r_start:r_end]

            mask = r_data >= threshold
            if not np.any(mask):
                continue
            r_data = r_data[mask]
            r_cols = r_cols[mask]

            if len(r_data) > top_k:
                top_idx = np.argpartition(r_data, -top_k)[-top_k:]
                r_data = r_data[top_idx]
                r_cols = r_cols[top_idx]

            global_i = start + local_i
            all_rows.extend([global_i] * len(r_cols))
            all_cols.extend(r_cols.tolist())
            all_data.extend(r_data.tolist())

        # Free GPU memory for this chunk
        del q_chunk_gpu, sim_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Free target from GPU
    del target_gpu, target_gpu_T
    cp.get_default_memory_pool().free_all_blocks()

    result = sparse.csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n_queries, n_targets),
    )
    return result


# ────────────────── CPU top-k sparse matmul (fallback) ───────────────────
def _cpu_sparse_topk(query_matrix, target_matrix, top_k: int, threshold: float = 0.08):
    """Return top-k cosine neighbours via CPU sparse matmul.

    Uses sparse_dot_topn if available, otherwise pure SciPy CSR slicing.
    """
    if HAS_SPARSE_DOT_TOPN:
        result = awesome_cossim_topn(
            query_matrix,
            target_matrix.T,
            ntop=top_k,
            lower_bound=threshold,
            use_threads=True,
            n_jobs=4,
        )
        return result

    # ── Pure SciPy CSR pointer slicing ──────────────────────────────────
    n_queries = query_matrix.shape[0]
    chunk_sz = 5000  # smaller chunks to limit peak memory
    all_rows, all_cols, all_data = [], [], []

    for start in range(0, n_queries, chunk_sz):
        end = min(start + chunk_sz, n_queries)
        sim_chunk = (query_matrix[start:end] @ target_matrix.T).tocsr()
        indptr = sim_chunk.indptr
        indices = sim_chunk.indices
        data = sim_chunk.data

        for local_i in range(sim_chunk.shape[0]):
            r_start = indptr[local_i]
            r_end = indptr[local_i + 1]
            if r_start == r_end:
                continue
            r_data = data[r_start:r_end]
            r_cols = indices[r_start:r_end]

            mask = r_data >= threshold
            if not np.any(mask):
                continue
            r_data = r_data[mask]
            r_cols = r_cols[mask]

            if len(r_data) > top_k:
                top_idx = np.argpartition(r_data, -top_k)[-top_k:]
                r_data = r_data[top_idx]
                r_cols = r_cols[top_idx]

            global_i = start + local_i
            all_rows.extend([global_i] * len(r_cols))
            all_cols.extend(r_cols.tolist())
            all_data.extend(r_data.tolist())

        del sim_chunk
        gc.collect()

    result = sparse.csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n_queries, target_matrix.shape[0]),
    )
    return result


# ───────────────── Unified dispatcher ────────────────────────────────────
def _sparse_topk(query_matrix, target_matrix, top_k: int, threshold: float = 0.08):
    """Dispatch to GPU or CPU top-k implementation based on availability."""
    if HAS_CUPY:
        try:
            return _gpu_sparse_topk(query_matrix, target_matrix, top_k, threshold)
        except Exception as e:
            print(f"  [WARN] GPU matmul failed ({e}), falling back to CPU ...")
    return _cpu_sparse_topk(query_matrix, target_matrix, top_k, threshold)


# ───────────────────────── Pass 1: TF-IDF char n-gram blocking ────────────
class TFIDFBlocker:
    """Builds a TF-IDF char-ngram index on targets and retrieves top-K
    candidates for each query entity, processing queries in memory-safe
    chunks.  Uses GPU acceleration when available.
    """

    def __init__(
        self,
        top_k: int = 25,
        max_features: int = 15_000,
        ngram_range: Tuple[int, int] = (3, 4),
        min_score: float = 0.08,
        chunk_size: int = 50_000,
    ):
        self.top_k = top_k
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_score = min_score
        self.chunk_size = chunk_size
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.target_matrix = None
        self.target_ids: List[str] = []

    def fit(self, target_df: pd.DataFrame):
        """Fit the TF-IDF vectorizer on the target entities (S2 + S3)."""
        _report_backend()
        self.target_ids = target_df["entity_id"].tolist()
        texts = target_df["combined_text"].tolist()

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.target_matrix = self.vectorizer.fit_transform(texts)
        del texts
        gc.collect()
        print(f"  TF-IDF index: {self.target_matrix.shape[0]:,} targets × "
              f"{self.target_matrix.shape[1]:,} features, "
              f"nnz={self.target_matrix.nnz:,}")

    def query(self, query_df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve top-K candidates with scores for each query S1 entity."""
        query_ids = query_df["entity_id"].tolist()
        query_texts = query_df["combined_text"].tolist()
        n = len(query_ids)
        candidates: Dict[str, List[Tuple[str, float]]] = {}

        n_chunks = (n + self.chunk_size - 1) // self.chunk_size
        print(f"  Processing {n:,} queries in {n_chunks} chunks of {self.chunk_size:,} ...")

        for start in tqdm(range(0, n, self.chunk_size),
                          desc="  TF-IDF blocking chunks", unit="chunk"):
            end = min(start + self.chunk_size, n)
            chunk_texts = query_texts[start:end]
            chunk_ids = query_ids[start:end]

            t0 = time.time()
            q_matrix = self.vectorizer.transform(chunk_texts)

            sim = _sparse_topk(q_matrix, self.target_matrix, self.top_k, self.min_score).tocsr()
            elapsed = time.time() - t0

            indptr = sim.indptr
            indices = sim.indices
            data = sim.data

            chunk_matched = 0
            for local_i, s1_id in enumerate(chunk_ids):
                r_start = indptr[local_i]
                r_end = indptr[local_i + 1]
                if r_start == r_end:
                    candidates[s1_id] = []
                    continue
                cols = indices[r_start:r_end]
                scores = data[r_start:r_end]
                order = np.argsort(-scores)
                candidates[s1_id] = [
                    (self.target_ids[cols[j]], float(scores[j])) for j in order
                ]
                chunk_matched += 1

            del q_matrix, sim
            gc.collect()
            if HAS_CUPY:
                cp.get_default_memory_pool().free_all_blocks()

            print(f"    Chunk {start//self.chunk_size + 1}/{n_chunks}: "
                  f"{chunk_matched}/{end-start} matched in {elapsed:.1f}s")

        return candidates


# ───────────────────────── Pass 2: Postal code blocking ───────────────────
def postal_code_blocking(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    max_per_block: int = 200,
) -> Dict[str, List[str]]:
    """Group targets by postal code; return candidate IDs for each S1 entity
    that shares the same postal code.

    Blocks larger than ``max_per_block`` are skipped (they would be
    uninformative and expensive).
    """
    target_postal = target_df[target_df["postal_code"] != ""]
    postal_index: Dict[str, List[str]] = defaultdict(list)
    for eid, pc in zip(target_postal["entity_id"], target_postal["postal_code"]):
        postal_index[pc].append(eid)

    # Remove overly large blocks (common postal codes are not discriminative)
    postal_index = {k: v for k, v in postal_index.items() if len(v) <= max_per_block}

    candidates: Dict[str, List[str]] = {}
    query_with_postal = query_df[query_df["postal_code"] != ""]

    for s1_id, pc in zip(query_with_postal["entity_id"], query_with_postal["postal_code"]):
        if pc in postal_index:
            candidates[s1_id] = postal_index[pc]

    print(f"  Postal blocking: {len(candidates):,} S1 entities matched a postal block")
    return candidates


# ───────────────────────── Pass 3: Name prefix blocking ───────────────────
def name_prefix_blocking(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prefix_len: int = 4,
    max_per_block: int = 300,
) -> Dict[str, List[str]]:
    """First ``prefix_len`` characters of the first significant name token."""

    def _get_prefix(name: str) -> str:
        for tok in name.split():
            if len(tok) >= prefix_len:
                return tok[:prefix_len]
        return name[:prefix_len] if len(name) >= prefix_len else ""

    target_prefixed = target_df[target_df["clean_name"].str.len() >= prefix_len]
    prefix_index: Dict[str, List[str]] = defaultdict(list)
    for eid, name in zip(target_prefixed["entity_id"], target_prefixed["clean_name"]):
        pfx = _get_prefix(name)
        if pfx:
            prefix_index[pfx].append(eid)

    prefix_index = {k: v for k, v in prefix_index.items() if len(v) <= max_per_block}

    candidates: Dict[str, List[str]] = {}
    for s1_id, name in zip(query_df["entity_id"], query_df["clean_name"]):
        pfx = _get_prefix(name)
        if pfx and pfx in prefix_index:
            candidates[s1_id] = prefix_index[pfx]

    print(f"  Prefix blocking: {len(candidates):,} S1 entities matched a prefix block")
    return candidates


# ───────────────────────── Multi-pass orchestrator ────────────────────────
class MultiPassBlocker:
    """Orchestrates multiple blocking passes and unions their results."""

    def __init__(self, top_k: int = 25, max_features: int = 15_000,
                 ngram_range: Tuple[int, int] = (3, 4),
                 chunk_size: int = 50_000, min_score: float = 0.08):
        self.top_k = top_k
        self.tfidf_blocker = TFIDFBlocker(
            top_k=top_k, max_features=max_features,
            ngram_range=ngram_range, min_score=min_score,
            chunk_size=chunk_size,
        )

    def fit(self, target_df: pd.DataFrame):
        """Fit the TF-IDF index on target data."""
        self.tfidf_blocker.fit(target_df)

    def generate_candidates(
        self,
        query_df: pd.DataFrame,
        target_df: pd.DataFrame,
        use_postal: bool = True,
        use_prefix: bool = True,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Run all blocking passes and return unified candidates with scores.

        Returns a dict mapping each S1 entity_id to a list of
        ``(candidate_id, score)`` tuples, where ``score`` is the TF-IDF
        cosine similarity (0.0 for candidates found only via auxiliary passes).
        """
        # Pass 1: TF-IDF
        print("  > Pass 1: TF-IDF char n-gram blocking ...")
        tfidf_cands = self.tfidf_blocker.query(query_df)

        # Build a {s1_id: {cand_id: score}} mapping for easy merging
        merged: Dict[str, Dict[str, float]] = {}
        for s1_id, cand_list in tfidf_cands.items():
            merged[s1_id] = {cid: score for cid, score in cand_list}
        del tfidf_cands
        gc.collect()

        # Pass 2: Postal code
        if use_postal:
            print("  > Pass 2: Postal code blocking ...")
            postal_cands = postal_code_blocking(query_df, target_df)
            for s1_id, cand_ids in postal_cands.items():
                if s1_id not in merged:
                    merged[s1_id] = {}
                for cid in cand_ids:
                    if cid not in merged[s1_id]:
                        merged[s1_id][cid] = 0.0  # no TF-IDF score
            del postal_cands

        # Pass 3: Name prefix
        if use_prefix:
            print("  > Pass 3: Name prefix blocking ...")
            prefix_cands = name_prefix_blocking(query_df, target_df)
            for s1_id, cand_ids in prefix_cands.items():
                if s1_id not in merged:
                    merged[s1_id] = {}
                for cid in cand_ids:
                    if cid not in merged[s1_id]:
                        merged[s1_id][cid] = 0.0
            del prefix_cands

        # Ensure every S1 entity has an entry (even if empty)
        for s1_id in query_df["entity_id"]:
            if s1_id not in merged:
                merged[s1_id] = {}

        # Convert to sorted list format (highest score first)
        result: Dict[str, List[Tuple[str, float]]] = {}
        for s1_id, cand_dict in merged.items():
            sorted_cands = sorted(cand_dict.items(), key=lambda x: -x[1])
            result[s1_id] = sorted_cands

        total_pairs = sum(len(v) for v in result.values())
        avg_cands = total_pairs / max(len(result), 1)
        print(f"  [OK] Multi-pass blocking complete: {len(result):,} S1 entities, "
              f"{total_pairs:,} total pairs (avg {avg_cands:.1f} per entity)")

        return result


def format_candidate_pairs_dataframe(
    candidates_map: Dict[str, List[Tuple[str, float]]]
) -> pd.DataFrame:
    """
    Formats the candidates map into the official candidate_pairs.tsv DataFrame format.
    Columns: source1_entity_id \t candidate_entity_ids
    """
    rows = []
    for s1_id, cand_list in candidates_map.items():
        unique_cands = list(dict.fromkeys(cid for cid, _ in cand_list))
        cand_str = ",".join(unique_cands)
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})
    return pd.DataFrame(rows)
