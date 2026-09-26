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
- [x] `src/features.py` — 18-dimensional pairwise feature extractor
  - RapidFuzz: ratio, token_sort, token_set, partial, WRatio
  - Jaro-Winkler similarity
  - Token Jaccard overlap
  - Postal code match, country match, length ratios
- [x] `src/metrics.py` — Competition-exact Macro F₀.₅ evaluator (with singleton handling)
- [x] `src/pipeline.py` — End-to-end orchestrator with checkpoint/resume
- [x] Installed all dependencies (lightgbm, rapidfuzz, sparse_dot_topn, pyarrow, joblib)
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

## ❌ Phase 3: Smoke Test & Validation  ← **WE ARE HERE**
- [ ] Run end-to-end smoke test on small subset (~1000 entities)
  - [ ] Verify data loading works (TSV parsing, column alignment)
  - [ ] Verify preprocessing runs without errors
  - [ ] Verify blocking generates candidate pairs
  - [ ] Verify feature extraction produces valid 18-dim vectors
  - [ ] Verify LightGBM trains and produces predictions
  - [ ] Verify threshold optimizer finds optimal τ*
- [ ] Fix any runtime bugs found during smoke test
- [ ] **Git Commit**: *Checkpoint — Smoke test passing*

---

## ❌ Phase 4: Full Training Pipeline Execution
- [ ] Load & preprocess all training data (Source 1 + Source 2 + Source 3)
- [ ] Create 80/20 stratified validation split by Source 1 entity ID
- [ ] Run blocking on train split → generate candidate pairs
- [ ] Measure blocking recall (target: ≥95% of true matches in candidates)
- [ ] Extract pairwise features for all train candidate pairs
- [ ] Train LightGBM classifier with early stopping on validation set
- [ ] Grid-search decision threshold τ ∈ [0.50, 0.90] to maximize Macro F₀.₅
- [ ] Log validation metrics: Precision, Recall, F₀.₅, optimal τ*
- [ ] **Git Commit**: *Checkpoint — Trained model with validation F₀.₅ score*

---

## ❌ Phase 5: Test Inference & Submission File Generation
- [ ] Load & preprocess test data (test Source 1 + Source 2 + Source 3)
- [ ] Run blocking on test Source 1 → generate test candidate pairs
- [ ] Export `output/candidate_pairs.tsv`
- [ ] Extract pairwise features for all test candidate pairs
- [ ] Run LightGBM inference → get match probabilities
- [ ] Apply calibrated threshold τ* → assign matches or empty (singleton)
- [ ] Export `output/matching_results.tsv`
- [ ] **Git Commit**: *Checkpoint — Test inference complete, submission files generated*

---

## ❌ Phase 6: Validation & First Submission
- [ ] Run `utils/validate_submission.py` on both output TSVs
  - [ ] Verify all Source 1 IDs present
  - [ ] Verify no duplicate matches
  - [ ] Verify TSV format & column structure
- [ ] Package submission zip via `utils/package_submission.py`
- [ ] Upload to Unstop leaderboard (Submission #1)
- [ ] Record leaderboard score
- [ ] **Git Commit**: *Checkpoint — First submission uploaded*

---

## ❌ Phase 7: Iteration & Score Improvement (If Time Permits)
- [ ] Analyze error cases from validation split
  - [ ] False positives (wrong matches) — tighten threshold?
  - [ ] False negatives (missed matches) — improve blocking recall?
- [ ] Tune blocking parameters (top-K, TF-IDF threshold, n-gram range)
- [ ] Add additional features (e.g., street number exact match, word overlap count)
- [ ] Try CatBoost as alternative/ensemble with LightGBM
- [ ] Consider adding sentence-transformer embeddings (MiniLM/BGE-small) for semantic similarity
- [ ] Re-optimize threshold on improved model
- [ ] Upload improved submission to Unstop (Submissions #2–5)
- [ ] **Git Commit**: *Checkpoint — Improved model iteration*

---

## ❌ Phase 8: Final Documentation & Cleanup
- [ ] Fill out `Documentation_template.md` with methodology writeup
- [ ] Update `README.md` with final results and reproduction instructions
- [ ] Clean up code, remove scratch files
- [ ] Final `git push` to remote
- [ ] **Git Commit**: *Final submission — Documentation complete*

---

## 📊 Progress Summary

| Phase | Status | Git Commit |
|:------|:------:|:-----------|
| Phase 0: Setup & Data | ✅ Done | `8eeb3a9` |
| Phase 1: Architecture | ✅ Done | `4d1ea31` |
| Phase 2: Pipeline Code | ✅ Done | `de55032` |
| Phase 2.5: Utilities | ✅ Done | `5f5c820` |
| Phase 3: Smoke Test | ❌ Next | — |
| Phase 4: Full Training | ❌ Pending | — |
| Phase 5: Test Inference | ❌ Pending | — |
| Phase 6: First Submission | ❌ Pending | — |
| Phase 7: Iterations | ❌ Pending | — |
| Phase 8: Documentation | ❌ Pending | — |

> **Current Position**: All code is written. Next step is running and validating it end-to-end.
