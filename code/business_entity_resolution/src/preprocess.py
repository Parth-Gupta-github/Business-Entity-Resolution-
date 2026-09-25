"""
Data Preprocessing and Text Normalization Module.
Handles noisy business names, address variations, legal suffixes,
abbreviations, and international variations (US, India, France).
"""

import re
import unicodedata
from typing import Optional, Dict, Any
import pandas as pd


# Suffix standardization dictionary
LEGAL_SUFFIXES = {
    r"\bcorp(\.|\b)": "corporation",
    r"\binc(\.|\b)": "incorporated",
    r"\bltd(\.|\b)": "limited",
    r"\bpvt(\.|\b)": "private",
    r"\bpte(\.|\b)": "private",
    r"\bllc(\.|\b)": "limited liability company",
    r"\bllp(\.|\b)": "limited liability partnership",
    r"\bgmbh(\.|\b)": "gmbh",
    r"\bsarl(\.|\b)": "sarl",
    r"\bsa(\.|\b)": "sa",
    r"\bco(\.|\b)": "company",
    r"\bdept(\.|\b)": "department",
    r"\bmfg(\.|\b)": "manufacturing",
    r"\bintl(\.|\b)": "international",
    r"\btech(\.|\b)": "technology",
    r"\bsols(\.|\b)": "solutions",
    r"\bserv(\.|\b)": "services",
}

# Address abbreviations standardization dictionary
ADDRESS_ABBREVIATIONS = {
    r"\brd(\.|\b)": "road",
    r"\bst(\.|\b)": "street",
    r"\bave(\.|\b)": "avenue",
    r"\bblvd(\.|\b)": "boulevard",
    r"\bdr(\.|\b)": "drive",
    r"\bln(\.|\b)": "lane",
    r"\bct(\.|\b)": "court",
    r"\bpl(\.|\b)": "place",
    r"\bsq(\.|\b)": "square",
    r"\bhwy(\.|\b)": "highway",
    r"\bflr(\.|\b)": "floor",
    r"\bfl(\.|\b)": "floor",
    r"\bbldg(\.|\b)": "building",
    r"\bapt(\.|\b)": "apartment",
    r"\bste(\.|\b)": "suite",
    r"\bno(\.|\b)": "number",
    r"\bopp(\.|\b)": "opposite",
    r"\bnr(\.|\b)": "near",
    r"\bbhd(\.|\b)": "behind",
    r"\bext(\.|\b)": "extension",
    r"\bsec(\.|\b)": "sector",
    r"\bdist(\.|\b)": "district",
    r"\bav(\.|\b)": "avenue",
    r"\brue(\.|\b)": "rue",
}


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
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: str) -> str:
    """Standardizes business names by normalizing abbreviations and legal suffixes."""
    text = clean_text(name)
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(address: str) -> str:
    """Standardizes address strings by expanding abbreviations."""
    text = clean_text(address)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_postal_code(address: str) -> str:
    """
    Extracts 5 or 6-digit postal/PIN codes (handles US 5-digit, India 6-digit, France 5-digit).
    """
    if not isinstance(address, str):
        return ""
    # Look for 5 or 6 consecutive digits
    matches = re.findall(r"\b\d{5,6}\b", address)
    return matches[0] if matches else ""


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies end-to-end normalization to a business entity DataFrame."""
    df = df.copy()
    
    # Fill missing values
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str).str.strip().str.upper()

    # Preprocessed columns
    df["clean_name"] = df["business_name"].apply(normalize_business_name)
    df["clean_address"] = df["business_address"].apply(normalize_address)
    df["postal_code"] = df["business_address"].apply(extract_postal_code)
    df["combined_text"] = df["clean_name"] + " " + df["clean_address"]

    return df
