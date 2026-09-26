# Amazon ML Challenge 2026 — Progress Checklist

**Problem**: Business Entity Resolution (match 2.2M Source 1 entities to 10.3M Source 2/3 records)  
**Metric**: Macro F₀.₅ (Precision weighted 2× over Recall)  
**Submission**: `candidate_pairs.tsv` + `matching_results.tsv` (5 submissions/day on Unstop)

---

## ✅ Phase 0: Project Setup & Understanding
- [x] Downloaded & extracted dataset zip (~1 GB)
- [x] Explored all data files and understood schema
  - `train_source1.tsv` (210 MB, ~2.2M rows) — anchor entities
  - `train_source2.tsv` (489 MB) — target database 1
  - `train_source3.tsv` (504 MB) — target database 2
  - `train_ground_truth.tsv` (127 MB) — true match labels
  - `test_source1.tsv` (175 MB, ~1.73M rows) — test anchors
  - `test_source2.tsv` (509 MB) — test target database 1
  - `test_source3.tsv` (506 MB) — test target database 2
- [x] Read the problem statement PDF
- [x] Understood submission format (TSV files) and Unstop upload process
- [x] Created `.gitignore` for datasets & outputs
- [x] **Git Commit**: `8eeb3a9` — *Initial commit: Architecture, documentation, environment setup*

---

## ✅ Phase 1: Architecture & Strategy Design
- [x] Analyzed the competition metric (Macro F₀.₅) and its implications
  - Precision is 2× more important than Recall
  - Singletons (no match) predicted correctly get score = 1.0
  - Strategy: Be conservative — only match when very confident
- [x] Designed two-stage funnel architecture (Block → Classify)
- [x] Wrote detailed architecture document with Mermaid diagrams
- [x] Planned multi-pass blocking strategy (TF-IDF + Postal + Name prefix)
- [x] **Git Commit**: `4d1ea31` — *Add system architecture and technical execution strategy*

---

## ✅ Phase 2: Core Pipeline Code Implementation
- [x] `src/config.py` — Centralized config (file paths, hyperparams, thresholds)
- [x] `src/preprocess.py` — Text normalization engine
  - Unicode NFKD accent decomposition (French/German support)
  - Legal suffix harmonization (pvt ltd, gmbh, sarl, etc.)
  - Address abbreviation expansion (rd→road, st→street, etc.)
  - Postal code extraction (5/6 digit regex)
- [x] `src/blocking.py` — High-recall candidate generation
  - TF-IDF char n-gram (3–5) vectorizer
  - Sparse matrix top-K cosine similarity via `sparse_dot_topn`
  - Multi-pass union with deduplication
- [x] `src/features.py` — 27-dimensional pairwise feature extractor
  - RapidFuzz: ratio, token_sort, token_set, partial, WRatio
  - Jaro-Winkler similarity
  - Token Jaccard overlap
  - Postal code match, country match, length ratios
- [x] `src/models.py` — Multi-model training, ensembling, and threshold optimization
  - LightGBM, CatBoost, XGBoost, and soft-voting Ensemble
- [x] `src/metrics.py` — Competition-exact Macro F₀.₅ evaluator (with singleton handling)
- [x] `src/pipeline.py` — End-to-end orchestrator with checkpoint/resume
- [x] Installed all dependencies (lightgbm, catboost, xgboost, rapidfuzz, sparse_dot_topn, pyarrow, joblib)
- [x] Fixed Windows cp1252 Unicode encoding issues in all print statements
- [x] Fixed `sparse_dot_topn` API compatibility (v1.2 n_jobs parameter)
- [x] Verified all module imports compile successfully
- [x] **Git Commit**: `de55032` — *Implement modular two-stage entity resolution pipeline*

---

## ✅ Phase 2.5: Utilities & Submission Helpers
- [x] `utils/validate_submission.py` — Official submission format validator
- [x] `utils/package_submission.py` — Zip packaging script
- [x] **Git Commit**: `5f5c820` — *Add submission packaging helper utility*

---

## ✅ Phase 3: Smoke Test & Validation
- [x] Run end-to-end smoke test on subset with true matches & singletons
  - [x] Verify data loading works (TSV parsing, column alignment)
  - [x] Verify preprocessing runs without errors (NFKD normalization)
  - [x] Verify blocking generates candidate pairs (blocking recall: 99.32%)
  - [x] Verify feature extraction produces valid 27-dim vectors
  - [x] Verify LightGBM trains and produces predictions
  - [x] Verify threshold optimizer finds optimal τ*
- [x] Fix any runtime bugs found during smoke test

---

## 🔄 Phase 4: Full Training Pipeline Execution (Abhishek Mehta)
- [ ] Load & preprocess all training data (Source 1 + Source 2 + Source 3)
- [ ] Create 80/20 stratified validation split by Source 1 entity ID
- [ ] Run blocking on train split → generate candidate pairs
- [ ] Measure blocking recall (target: ≥95% of true matches in candidates)
- [ ] Extract pairwise features for all train candidate pairs
- [ ] Train LightGBM classifier with early stopping on validation set
- [ ] Grid-search decision threshold τ ∈ [0.40, 0.95] to maximize Macro F₀.₅
- [ ] Log validation metrics: Precision, Recall, F₀.₅, optimal τ*

---

## 🔄 Phase 5: Test Inference & Submission File Generation (Abhishek / Parth)
- [ ] Load & preprocess test data (test Source 1 + Source 2 + Source 3)
- [ ] Run blocking on test Source 1 → generate test candidate pairs
- [ ] Export `output/candidate_pairs.tsv`
- [ ] Extract pairwise features for all test candidate pairs
- [ ] Run LightGBM / Ensemble inference → get match probabilities
- [ ] Apply calibrated threshold τ* → assign matches or empty (singleton)
- [ ] Export `output/matching_results.tsv`

---

## 🔄 Phase 6: Validation & First Submission (Parth Gupta)
- [ ] Run `utils/validate_submission.py` on both output TSVs
  - [ ] Verify all Source 1 IDs present
  - [ ] Verify no duplicate matches
  - [ ] Verify TSV format & column structure
- [ ] Package submission zip via `utils/package_submission.py`
- [ ] Upload to Unstop leaderboard (Submission #1)
- [ ] Record leaderboard score

---

## ✅ Phase 7: Model Exploration & Benchmarking (Parv Tiwari)
- [x] Implement `src/models.py` with LightGBM, CatBoost, XGBoost, and soft-voting Ensemble
- [x] Create standalone benchmarking script `src/benchmark_models.py`
- [x] Benchmark all models on extracted pairwise features:
  - LightGBM: Macro F₀.₅ = 0.9306 (τ* = 0.430, Train time: 1.5s)
  - CatBoost: Macro F₀.₅ = 0.9196 (τ* = 0.470, Train time: 4.3s)
  - XGBoost: Macro F₀.₅ = 0.9277 (τ* = 0.410, Train time: 1.1s)
  - Ensemble: Macro F₀.₅ = 0.9277 (τ* = 0.400)
- [x] Add `--model` and `--benchmark` CLI flags to `src/pipeline.py`

---

## ✅ Phase 8: Final Documentation & Methodology Writeup (Parv Tiwari)
- [x] Fill out `Documentation_template.md` with complete methodology report
  - Problem analysis (scale, noise, open-set country France, metric asymmetry)
  - Two-stage architecture details & Mermaid diagrams
  - High-recall multi-pass blocking (TF-IDF + postal code + name prefix)
  - 27-dimensional pairwise feature extraction
  - LightGBM, CatBoost, XGBoost, and Ensemble comparative benchmark
  - Precision-weighted threshold optimization and singleton handling
  - Error analysis (common false positives & false negatives)
  - Appendix with code artefacts and feature importance
- [x] Update root `README.md` with step-by-step reproduction instructions
- [x] Update `code/business_entity_resolution/README.md`
- [x] Update `requirements.txt` (root & module) with `catboost` and `xgboost`

---

## 📊 Progress Summary

| Phase | Status | Assignee |
|:------|:------:|:---|
| Phase 0: Setup & Data | ✅ Done | Parth Gupta |
| Phase 1: Architecture | ✅ Done | Parth Gupta |
| Phase 2: Pipeline Code | ✅ Done | Team |
| Phase 2.5: Utilities | ✅ Done | Parth Gupta |
| Phase 3: Smoke Test | ✅ Done | Parv Tiwari / Team |
| Phase 4: Full Training | 🔄 In Progress | Abhishek Mehta |
| Phase 5: Test Inference | 🔄 Pending | Abhishek / Parth |
| Phase 6: First Submission | 🔄 Pending | Parth Gupta |
| Phase 7: Model Exploration & Ensembling | ✅ Done | Parv Tiwari |
| Phase 8: Documentation & Reproduction Guide | ✅ Done | Parv Tiwari |
