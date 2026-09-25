import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "dataset"
    TRAIN_DIR: Path = DATA_DIR / "train"
    TEST_DIR: Path = DATA_DIR / "test"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"

    # Raw File Names
    TRAIN_SOURCE1: str = "train_source1.tsv"
    TRAIN_SOURCE2: str = "train_source2.tsv"
    TRAIN_SOURCE3: str = "train_source3.tsv"
    TRAIN_GROUND_TRUTH: str = "train_ground_truth.tsv"

    TEST_SOURCE1: str = "test_source1.tsv"
    TEST_SOURCE2: str = "test_source2.tsv"
    TEST_SOURCE3: str = "test_source3.tsv"

    # Submission Output File Names
    MATCHING_RESULTS_FILE: str = "matching_results.tsv"
    CANDIDATE_PAIRS_FILE: str = "candidate_pairs.tsv"

    # Validation Split
    VAL_SPLIT_RATIO: float = 0.20
    RANDOM_SEED: int = 42

    # Blocking Hyperparameters
    TOP_K_CANDIDATES_PER_SOURCE: int = 20
    MIN_TOKEN_CHAR_LEN: int = 2
    TFIDF_MAX_FEATURES: int = 50000
    CHAR_NGRAM_RANGE: tuple = (2, 4)

    # Classification & Threshold
    BETA: float = 0.5  # F_0.5 macro score metric weight
    INITIAL_THRESHOLD: float = 0.65

    def create_dirs(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
