import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Base Directories ──────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "dataset"
    TRAIN_DIR: Path = DATA_DIR / "train"
    TEST_DIR: Path = DATA_DIR / "test"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"

    # ── Raw File Names ────────────────────────────────────────────────
    TRAIN_SOURCE1: str = "train_source1.tsv"
    TRAIN_SOURCE2: str = "train_source2.tsv"
    TRAIN_SOURCE3: str = "train_source3.tsv"
    TRAIN_GROUND_TRUTH: str = "train_ground_truth.tsv"

    TEST_SOURCE1: str = "test_source1.tsv"
    TEST_SOURCE2: str = "test_source2.tsv"
    TEST_SOURCE3: str = "test_source3.tsv"

    # ── Submission Output File Names ─────────────────────────────────
    MATCHING_RESULTS_FILE: str = "matching_results.tsv"
    CANDIDATE_PAIRS_FILE: str = "candidate_pairs.tsv"

    # ── Validation Split ─────────────────────────────────────────────
    VAL_SPLIT_RATIO: float = 0.20
    RANDOM_SEED: int = 42

    # ── Blocking Hyperparameters ─────────────────────────────────────
    TOP_K_CANDIDATES: int = 25          # per source per S1 entity
    TFIDF_MAX_FEATURES: int = 35_000
    CHAR_NGRAM_RANGE: tuple = (3, 4)
    BLOCKING_CHUNK_SIZE: int = 100_000  # S1 queries processed per chunk
    MIN_TFIDF_SCORE: float = 0.05       # ignore negligible cosine scores
    SAMPLE_TRAIN_ENTITIES: int = 50_000 # Stratified S1 anchors for fast model fitting (~300k pairs)
    SAMPLE_VAL_ENTITIES: int = 20_000   # Fast validation fold for threshold search

    # ── Feature Engineering ──────────────────────────────────────────
    FEATURE_BATCH_SIZE: int = 100_000   # pairs per feature extraction batch

    # ── LightGBM Training ────────────────────────────────────────────
    LGB_PARAMS: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "n_estimators": 800,
        "verbose": -1,
    })
    LGB_EARLY_STOPPING: int = 30

    # ── Threshold Optimization ───────────────────────────────────────
    BETA: float = 0.5
    THRESHOLD_SCAN_MIN: float = 0.40
    THRESHOLD_SCAN_MAX: float = 0.95
    THRESHOLD_SCAN_STEP: float = 0.01

    def create_dirs(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
