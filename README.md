# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-green.svg)](https://opensource.org/licenses/)

This repository contains the solution architecture, environment setup, and baseline pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**.

---

## 📌 Problem Overview

In commercial platforms, business records arrive from multiple disparate sources with noisy, incomplete, or varied fields (abbreviations, formatting differences, missing postal codes, typos). 

The goal is to resolve which records across sources belong to the same real-world business entity:
- **Source 1 (`S1-*`)**: Deduplicated reference anchor.
- **Source 2 (`S2-*`) & Source 3 (`S3-*`)**: Target sources to match against Source 1.
- **Countries**: Training covers **US** and **India**. The test set additionally introduces a third country, **France** (open set string labels — country filtering must never be hardcoded).

### 🎯 Evaluation Metric: Macro $F_{0.5}$ Score
Submissions are evaluated on **Macro-averaged $F_{\beta}$ Score ($\beta = 0.5$)**:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Precision-Heavy**: False merges (merging two distinct entities) are penalized $2\times$ more heavily than missed matches.
- **Singletons**: Entities in Source 1 with zero true matches score **1.0** when correctly predicted as empty, and **0.0** if false matches are predicted.

---

## 👥 Peer Setup & Quickstart Guide

Follow these steps to set up the repository locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/Parth-Gupta-github/Business-Entity-Resolution-.git
cd Business-Entity-Resolution-
```

### 2. Set Up Python Environment
We recommend using Python 3.10+ and a virtual environment:

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗂️ Dataset Placement (Do NOT Push Data to GitHub)

> ⚠️ **IMPORTANT**: The dataset files are large and strictly ignored by `.gitignore`. **Do not upload data files or zip archives to GitHub.**

Place the challenge dataset files in the following local directory structure:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 📁 Repository Structure

```text
Business-Entity-Resolution-/
├── .gitignore                          # Strict gitignore for datasets, models, & outputs
├── requirements.txt                    # Project dependencies
├── README.md                           # Main documentation & onboarding guide
├── Documentation_template.md           # Submission methodology report template
├── utils/
│   └── validate_submission.py          # Official submission validator
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt            # Package dependencies
│       ├── README.md                   # Execution & reproduction guide
│       └── src/
│           ├── __init__.py
│           ├── config.py               # Global paths & hyperparameters
│           ├── preprocess.py           # Text/address cleaner & normalizer
│           ├── metrics.py              # Exact competition Macro F_0.5 evaluator
│           ├── blocking.py             # Candidate generation (TF-IDF / Inverted Index)
│           └── features.py             # Pairwise similarity feature engineering
└── output/                             # Generated predictions (ignored by git)
    ├── matching_results.tsv            # Leaderboard submission
    └── candidate_pairs.tsv             # Blocking candidate set
```

---

## 🧪 Validating Outputs Locally

Before submitting to the portal, verify your output format with the official validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 📦 Final Submission Format

The final deliverable submitted to the organizers will be a single zip archive formatted as:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
