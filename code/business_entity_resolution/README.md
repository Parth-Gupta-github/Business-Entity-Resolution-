# Business Entity Resolution Pipeline

This folder contains the self-contained, reproducible source code for candidate generation (blocking), feature extraction, matching model training, and inference.

---

## 🛠️ Requirements & Environment

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏗️ Architecture Modules

- `src/config.py`: Filepaths, validation ratios, and pipeline configurations.
- `src/preprocess.py`: International address and business name normalization (abbreviations, legal suffixes, accents).
- `src/metrics.py`: Evaluator computing Macro $F_{0.5}$ score including singletons.
- `src/blocking.py`: Multi-pass candidate generation engine (TF-IDF character n-grams + nearest neighbors).
- `src/features.py`: Pairwise similarity features (Fuzzy token ratios, Jaro-Winkler, Levenshtein, address & postal code matches).

---

## 🚀 Execution Workflow

1. Place dataset files in `dataset/train/` and `dataset/test/`.
2. Run preprocessing, blocking, and matching to generate:
   - `output/candidate_pairs.tsv`
   - `output/matching_results.tsv`
3. Validate output files using `utils/validate_submission.py`.
