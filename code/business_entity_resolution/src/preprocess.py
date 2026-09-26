"""
Data Preprocessing and Text Normalization Module (High-Performance Multi-Core).
Handles noisy business names, address variations, legal suffixes,
abbreviations, and international variations (US, India, France).
"""

import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd


# Suffix standardization dictionary
LEGAL_SUFFIXES = {
    r"\bcorp(\.|oration)?\b": "corporation",
    r"\binc(\.|orporated)?\b": "incorporated",
    r"\bltd(\.)?\b": "limited",
    r"\bpvt(\.)?\b": "private",
    r"\bpte(\.)?\b": "private",
    r"\bllc(\.)?\b": "limited liability company",
    r"\bllp(\.)?\b": "limited liability partnership",
    r"\bgmbh(\.)?\b": "gmbh",
    r"\bco(\.)?\b": "company",
    r"\bdept(\.)?\b": "department",
    r"\bmfg(\.)?\b": "manufacturing",
    r"\bintl(\.)?\b": "international",
    r"\btech(\.)?\b": "technology",
    r"\bsols(\.)?\b": "solutions",
    r"\bserv(\.)?\b": "services",
    # French legal forms (keep as canonical tokens)
    r"\bsarl(\.)?\b": "sarl",
    r"\bsa(\.)?\b": "sa",
    r"\bsas(\.)?\b": "sas",
    r"\bsasu(\.)?\b": "sasu",
    r"\beurl(\.)?\b": "eurl",
    r"\bsei(\.)?\b": "sei",
    r"\bsci(\.)?\b": "sci",
    r"\bsnc(\.)?\b": "snc",
}

# Address abbreviations standardization dictionary
ADDRESS_ABBREVIATIONS = {
    r"\brd(\.)?\b": "road",
    r"\bst(\.)?\b": "street",
    r"\bave(\.)?\b": "avenue",
    r"\bblvd(\.)?\b": "boulevard",
    r"\bdr(\.)?\b": "drive",
    r"\bln(\.)?\b": "lane",
    r"\bct(\.)?\b": "court",
    r"\bpl(\.)?\b": "place",
    r"\bsq(\.)?\b": "square",
    r"\bhwy(\.)?\b": "highway",
    r"\bflr(\.)?\b": "floor",
    r"\bfl(\.)?\b": "floor",
    r"\bbldg(\.)?\b": "building",
    r"\bapt(\.)?\b": "apartment",
    r"\bste(\.)?\b": "suite",
    r"\bno(\.)?\b": "number",
    r"\bopp(\.)?\b": "opposite",
    r"\bnr(\.)?\b": "near",
    r"\bbhd(\.)?\b": "behind",
    r"\bext(\.)?\b": "extension",
    r"\bsec(\.)?\b": "sector",
    r"\bdist(\.)?\b": "district",
    r"\bav(\.)?\b": "avenue",
    r"\brue(\.)?\b": "rue",
    # French address terms (keep as-is after accent folding)
    r"\bchemin(\.)?\b": "chemin",
    r"\bimpasse(\.)?\b": "impasse",
    r"\ballee(\.)?\b": "allee",
    r"\bpassage(\.)?\b": "passage",
}

# Precompile all regex patterns for high throughput
COMPILED_LEGAL_SUFFIXES = [(re.compile(p, re.IGNORECASE), repl) for p, repl in LEGAL_SUFFIXES.items()]
COMPILED_ADDRESS_ABBREVIATIONS = [(re.compile(p, re.IGNORECASE), repl) for p, repl in ADDRESS_ABBREVIATIONS.items()]
POSTAL_RE = re.compile(r"\b\d{5,6}\b")
STREET_NUM_RE = re.compile(r"^\s*(\d{1,5})\b")
PUNCT_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")

NOISE_TOKENS = {"cedex", "bp", "cs"}


def strip_accents(text: str) -> str:
    """Normalizes unicode characters and removes diacritics (e.g., French accents)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Basic clean: lowercase, accent removal, punctuation stripping."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text.lower())
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _deduplicate_tokens(text: str) -> str:
    """Removes consecutive duplicate tokens: 'road road' -> 'road'."""
    tokens = text.split()
    if not tokens:
        return ""
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)
    return " ".join(deduped)


def _remove_noise_tokens(text: str) -> str:
    """Removes tokens that are noise for matching (e.g., 'cedex', 'bp')."""
    tokens = text.split()
    return " ".join(t for t in tokens if t not in NOISE_TOKENS)


def normalize_business_name(name: str) -> str:
    """Standardizes business names by normalizing abbreviations and legal suffixes."""
    text = clean_text(name)
    for pattern, replacement in COMPILED_LEGAL_SUFFIXES:
        text = pattern.sub(replacement, text)
    text = _deduplicate_tokens(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_address(address: str) -> str:
    """Standardizes address strings by expanding abbreviations."""
    text = clean_text(address)
    for pattern, replacement in COMPILED_ADDRESS_ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = _remove_noise_tokens(text)
    text = _deduplicate_tokens(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_postal_code(address: str) -> str:
    """Extracts 5 or 6-digit postal/PIN codes."""
    if not isinstance(address, str):
        return ""
    matches = POSTAL_RE.findall(address)
    return matches[0] if matches else ""


def extract_street_number(address: str) -> str:
    """Extracts the leading street/building number from an address."""
    if not isinstance(address, str):
        return ""
    m = STREET_NUM_RE.match(address)
    return m.group(1) if m else ""


def _process_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """Worker function to process a slice of records in a single pass."""
    names = df_chunk["business_name"].tolist()
    addrs = df_chunk["business_address"].tolist()

    clean_names = [normalize_business_name(n) for n in names]
    clean_addrs = [normalize_address(a) for a in addrs]
    postals = [extract_postal_code(a) for a in addrs]
    street_nums = [extract_street_number(a) for a in addrs]
    comb_texts = [f"{cn} {ca}" for cn, ca in zip(clean_names, clean_addrs)]

    df_chunk = df_chunk.copy()
    df_chunk["clean_name"] = clean_names
    df_chunk["clean_address"] = clean_addrs
    df_chunk["postal_code"] = postals
    df_chunk["street_number"] = street_nums
    df_chunk["combined_text"] = comb_texts
    return df_chunk


def preprocess_dataframe(df: pd.DataFrame, n_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Applies end-to-end normalization to a business entity DataFrame using
    all CPU cores for high parallelism.
    """
    df = df.copy()

    # Fill missing values
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str).str.strip().str.upper()

    num_rows = len(df)
    if num_rows < 10_000:
        return _process_chunk(df)

    # Multi-core parallel chunking
    workers = n_jobs or max(1, (os.cpu_count() or 4) - 1)
    chunk_size = int(np.ceil(num_rows / workers))
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, num_rows, chunk_size)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process_chunk, chunks))

    return pd.concat(results, ignore_index=True)
