"""
Model training for employee attrition prediction.
Trains Logistic Regression, Random Forest, and XGBoost with stratified CV.
Saves the best model by ROC AUC.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except Exception as e:
    xgb = None
    _XGBOOST_AVAILABLE = False

from src.config.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Metrics to compute during CV
CV_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _get_model(name: str, config: dict) -> Any:
    """Instantiate model from config."""
    if name == "logistic_regression":
        params = config["models"]["logistic_regression"]
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            random_state=params["random_state"],
        )
    if name == "random_forest":
        params = config["models"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
    if name == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not available (e.g. missing libomp on macOS). "
                "Install with: brew install libomp"
            )
        params = config["models"]["xgboost"]
        return xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"],
            use_label_encoder=params.get("use_label_encoder", False),
            eval_metric=params.get("eval_metric", "logloss"),
        )
    raise ValueError(f"Unknown model: {name}")


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_name: str = "xgboost",
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a single model with stratified K-fold CV and return fitted model and mean metrics.

    Args:
        X: Feature matrix (already preprocessed).
        y: Target vector (0/1).
        feature_names: Optional list of feature names (for tree models / logging).
        model_name: One of 'logistic_regression', 'random_forest', 'xgboost'.

    Returns:
        (fitted model, dict of metric name -> mean CV score).
    """
    config = load_config()
    train_config = config["training"]
    cv_folds = train_config.get("cv_folds", 5)
    random_state = train_config.get("random_state", 42)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    model = _get_model(model_name, config)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    start = time.time()
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )
    elapsed = time.time() - start

    mean_metrics = {}
    for sc in scoring:
        key = f"test_{sc}"
        if key in scores:
            mean_metrics[sc] = float(np.mean(scores[key]))
    mean_metrics["cv_folds"] = cv_folds
    logger.info("Model %s CV results: %s (%.2fs)", model_name, mean_metrics, elapsed)

    # Refit on full training data for final model
    model.fit(X, y)
    return model, mean_metrics


def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Any, str, Dict[str, Dict[str, float]]]:
    """
    Train Logistic Regression, Random Forest, and XGBoost (if available); return best model by ROC AUC.

    Args:
        X: Preprocessed feature matrix.
        y: Target vector.
        feature_names: Optional feature names.

    Returns:
        (best_model, best_model_name, all_metrics dict model_name -> metrics).
    """
    config = load_config()
    all_metrics: Dict[str, Dict[str, float]] = {}
    best_model = None
    best_name = None
    best_auc = -1.0

    model_names = ["logistic_regression", "random_forest"]
    if _XGBOOST_AVAILABLE:
        model_names.append("xgboost")
    else:
        logger.warning(
            "XGBoost not available (e.g. libomp missing on macOS). "
            "Training with Logistic Regression and Random Forest only. "
            "To fix: brew install libomp"
        )
    for name in model_names:
        model, metrics = train_and_evaluate(X, y, feature_names=feature_names, model_name=name)
        all_metrics[name] = metrics
        auc = metrics.get("roc_auc", 0)
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    logger.info("Best model: %s (ROC AUC=%.4f)", best_name, best_auc)
    return best_model, best_name, all_metrics


def save_model(model: Any, path: Optional[Path] = None) -> Path:
    """Save fitted model to disk. Path from config if not provided."""
    config = load_config()
    if path is None:
        model_dir = Path(config["paths"]["model_dir"])
        path = model_dir / config["paths"]["best_model_filename"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: Optional[Path] = None) -> Any:
    """Load model from disk."""
    config = load_config()
    if path is None:
        path = Path(config["paths"]["model_dir"]) / config["paths"]["best_model_filename"]
    return joblib.load(path)
