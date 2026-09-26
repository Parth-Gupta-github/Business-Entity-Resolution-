"""
Model Training, Ensembling, and Benchmarking Module.
===================================================

This module provides model training and evaluation interfaces for:
1. LightGBM Classifier (baseline workhorse)
2. CatBoost Classifier (robust handling of non-linear interactions & numerical splits)
3. XGBoost Classifier (alternative tree-boosting architecture)
4. Ensemble Classifier (weighted soft-voting probability blend)

Also includes threshold optimization directly aligned with Macro F0.5.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import config
from src.features import FEATURE_COLUMNS
from src.metrics import evaluate_predictions


# ===========================================================================
# 1. Ensemble Wrapper
# ===========================================================================
class EnsembleClassifier:
    """
    Weighted probability ensemble of multiple binary classifiers.
    Supports LightGBM, CatBoost, XGBoost, and other scikit-learn compatible models.
    """

    def __init__(self, models: List[Tuple[str, Any, float]]):
        """
        Parameters
        ----------
        models : List[Tuple[str, Any, float]]
            List of (name, model_instance, weight).
            Example: [("lgb", lgb_model, 0.6), ("catboost", cb_model, 0.4)]
        """
        self.models = models
        # Normalize weights so they sum to 1.0
        total_weight = sum(w for _, _, w in models)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive.")
        self.weights = [w / total_weight for _, _, w in models]
        self.names = [name for name, _, _ in models]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted average of positive class probabilities."""
        n_samples = X.shape[0]
        blended_prob = np.zeros(n_samples, dtype=np.float64)

        for (name, model, _), norm_w in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
            elif hasattr(model, "predict"):
                probs = model.predict(X)
            else:
                raise AttributeError(f"Model {name} has neither predict_proba nor predict.")
            blended_prob += norm_w * probs

        # Return (N, 2) array matching scikit-learn convention
        return np.column_stack([1.0 - blended_prob, blended_prob])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at given threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, file_path: Union[str, Path]):
        """Save ensemble bundle to pickle."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> EnsembleClassifier:
        """Load ensemble bundle from pickle."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# 2. LightGBM Training
# ===========================================================================
def train_lightgbm(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    params: Optional[dict] = None,
    save_path: Optional[Path] = None,
):
    """
    Train a LightGBM binary classifier with early stopping.
    """
    import lightgbm as lgb

    print("\n" + "=" * 60)
    print("Training LightGBM Classifier ...")
    print("=" * 60)

    lgb_params = params or config.LGB_PARAMS.copy()

    X_train = train_features[FEATURE_COLUMNS].values
    y_train = train_features["label"].values
    X_val = val_features[FEATURE_COLUMNS].values
    y_val = val_features["label"].values

    print(f"  Train samples: {len(X_train):,} ({y_train.sum():,} pos, {len(y_train) - y_train.sum():,} neg)")
    print(f"  Val samples:   {len(X_val):,} ({y_val.sum():,} pos, {len(y_val) - y_val.sum():,} neg)")

    model = lgb.LGBMClassifier(**lgb_params)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(config.LGB_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - t0
    print(f"  [OK] LightGBM trained in {elapsed:.1f}s | Best iteration: {model.best_iteration_}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [OK] Saved LightGBM to {save_path.name}")

    return model


# ===========================================================================
# 3. CatBoost Training
# ===========================================================================
def train_catboost(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    params: Optional[dict] = None,
    save_path: Optional[Path] = None,
):
    """
    Train a CatBoost binary classifier with early stopping.
    CatBoost provides strong regularization and handles deep feature interactions.
    """
    from catboost import CatBoostClassifier

    print("\n" + "=" * 60)
    print("Training CatBoost Classifier ...")
    print("=" * 60)

    cb_params = params or {
        "iterations": 800,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": config.RANDOM_SEED,
        "early_stopping_rounds": 30,
        "verbose": 50,
    }

    X_train = train_features[FEATURE_COLUMNS].values
    y_train = train_features["label"].values
    X_val = val_features[FEATURE_COLUMNS].values
    y_val = val_features["label"].values

    model = CatBoostClassifier(**cb_params)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    elapsed = time.time() - t0
    print(f"  [OK] CatBoost trained in {elapsed:.1f}s | Best iteration: {model.get_best_iteration()}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [OK] Saved CatBoost to {save_path.name}")

    return model


# ===========================================================================
# 4. XGBoost Training
# ===========================================================================
def train_xgboost(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    params: Optional[dict] = None,
    save_path: Optional[Path] = None,
):
    """
    Train an XGBoost binary classifier with early stopping.
    """
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("Training XGBoost Classifier ...")
    print("=" * 60)

    xgb_params = params or {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": config.RANDOM_SEED,
        "early_stopping_rounds": 30,
        "tree_method": "hist",
    }

    X_train = train_features[FEATURE_COLUMNS].values
    y_train = train_features["label"].values
    X_val = val_features[FEATURE_COLUMNS].values
    y_val = val_features["label"].values

    model = xgb.XGBClassifier(**xgb_params)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0
    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"  [OK] XGBoost trained in {elapsed:.1f}s | Best iteration: {best_iter}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [OK] Saved XGBoost to {save_path.name}")

    return model


# ===========================================================================
# 5. Threshold Optimization for Macro F0.5
# ===========================================================================
def optimize_threshold(
    model: Any,
    val_features: pd.DataFrame,
    val_ground_truth: Dict[str, Set[str]],
    all_val_s1_ids: Set[str],
    scan_min: float = 0.40,
    scan_max: float = 0.95,
    scan_step: float = 0.01,
    beta: float = 0.5,
    verbose: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Sweeps decision threshold tau to maximize Macro F_beta (default beta=0.5).
    Correctly accounts for true singletons (Source 1 anchors with no matches).

    Returns
    -------
    best_threshold : float
    best_f_beta : float
    best_metrics : Dict[str, Any]
    """
    X_val = val_features[FEATURE_COLUMNS].values
    probs = model.predict_proba(X_val)[:, 1]

    s1_ids_arr = val_features["source1_entity_id"].values
    cand_ids_arr = val_features["candidate_entity_id"].values

    best_score = -1.0
    best_threshold = 0.5
    best_metrics = {}
    sweep_history = []

    thresholds = np.arange(scan_min, scan_max + scan_step, scan_step)

    iterator = tqdm(thresholds, desc="Optimizing Threshold") if verbose else thresholds

    for T in iterator:
        mask = probs >= T
        matched_s1 = s1_ids_arr[mask]
        matched_cand = cand_ids_arr[mask]

        # Populate prediction mapping
        predictions: Dict[str, Set[str]] = {s1: set() for s1 in all_val_s1_ids}
        for s1, cand in zip(matched_s1, matched_cand):
            predictions[s1].add(cand)

        metrics = evaluate_predictions(val_ground_truth, predictions, beta=beta)
        f_val = metrics["macro_f_beta"]
        sweep_history.append((T, f_val, metrics["macro_precision"], metrics["macro_recall"]))

        if f_val > best_score:
            best_score = f_val
            best_threshold = float(T)
            best_metrics = metrics

    if verbose:
        sweep_history.sort(key=lambda x: -x[1])
        print(f"\n  Optimal Threshold: {best_threshold:.3f} | Macro F{beta}: {best_score:.4f} | "
              f"Macro Precision: {best_metrics.get('macro_precision', 0):.4f} | "
              f"Macro Recall: {best_metrics.get('macro_recall', 0):.4f}")

    return best_threshold, best_score, best_metrics


# ===========================================================================
# 6. Comparative Model Benchmark
# ===========================================================================
def benchmark_models(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    val_ground_truth: Dict[str, Set[str]],
    all_val_s1_ids: Set[str],
) -> Dict[str, Dict[str, float]]:
    """
    Trains LightGBM, CatBoost, XGBoost, and an Ensemble blend.
    Evaluates each on the validation set to produce a comparative benchmark report.
    """
    results: Dict[str, Dict[str, float]] = {}

    # 1. LightGBM
    lgb_model = train_lightgbm(train_features, val_features)
    t_lgb, f_lgb, m_lgb = optimize_threshold(
        lgb_model, val_features, val_ground_truth, all_val_s1_ids
    )
    results["LightGBM"] = {
        "Threshold": t_lgb,
        "Macro_F05": f_lgb,
        "Macro_Precision": m_lgb.get("macro_precision", 0.0),
        "Macro_Recall": m_lgb.get("macro_recall", 0.0),
    }

    # 2. CatBoost
    cb_model = train_catboost(train_features, val_features)
    t_cb, f_cb, m_cb = optimize_threshold(
        cb_model, val_features, val_ground_truth, all_val_s1_ids
    )
    results["CatBoost"] = {
        "Threshold": t_cb,
        "Macro_F05": f_cb,
        "Macro_Precision": m_cb.get("macro_precision", 0.0),
        "Macro_Recall": m_cb.get("macro_recall", 0.0),
    }

    # 3. XGBoost
    xgb_model = train_xgboost(train_features, val_features)
    t_xgb, f_xgb, m_xgb = optimize_threshold(
        xgb_model, val_features, val_ground_truth, all_val_s1_ids
    )
    results["XGBoost"] = {
        "Threshold": t_xgb,
        "Macro_F05": f_xgb,
        "Macro_Precision": m_xgb.get("macro_precision", 0.0),
        "Macro_Recall": m_xgb.get("macro_recall", 0.0),
    }

    # 4. Ensemble (0.6 LightGBM + 0.4 CatBoost)
    ens_lgb_cb = EnsembleClassifier([
        ("lightgbm", lgb_model, 0.6),
        ("catboost", cb_model, 0.4),
    ])
    t_ens1, f_ens1, m_ens1 = optimize_threshold(
        ens_lgb_cb, val_features, val_ground_truth, all_val_s1_ids
    )
    results["Ensemble (0.6 LGB + 0.4 CB)"] = {
        "Threshold": t_ens1,
        "Macro_F05": f_ens1,
        "Macro_Precision": m_ens1.get("macro_precision", 0.0),
        "Macro_Recall": m_ens1.get("macro_recall", 0.0),
    }

    # 5. Tri-Ensemble (0.5 LightGBM + 0.3 CatBoost + 0.2 XGBoost)
    ens_tri = EnsembleClassifier([
        ("lightgbm", lgb_model, 0.5),
        ("catboost", cb_model, 0.3),
        ("xgboost", xgb_model, 0.2),
    ])
    t_ens2, f_ens2, m_ens2 = optimize_threshold(
        ens_tri, val_features, val_ground_truth, all_val_s1_ids
    )
    results["Ensemble (0.5 LGB + 0.3 CB + 0.2 XGB)"] = {
        "Threshold": t_ens2,
        "Macro_F05": f_ens2,
        "Macro_Precision": m_ens2.get("macro_precision", 0.0),
        "Macro_Recall": m_ens2.get("macro_recall", 0.0),
    }

    # Print summary table
    print("\n" + "=" * 78)
    print("MODEL COMPARATIVE BENCHMARK SUMMARY (VALIDATION SET)")
    print("=" * 78)
    print(f"{'Model Architecture':<36} | {'Threshold':<10} | {'Macro F0.5':<10} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 78)
    for model_name, res in results.items():
        print(f"{model_name:<36} | {res['Threshold']:<10.3f} | {res['Macro_F05']:<10.4f} | "
              f"{res['Macro_Precision']:<10.4f} | {res['Macro_Recall']:<8.4f}")
    print("=" * 78)

    return results
