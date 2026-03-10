"""
Prediction module: load model and preprocessor, return probability and class.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.config.config_loader import load_config
from src.data.preprocessing import get_feature_names_after_preprocessing
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_artifacts(
    model_path: Optional[Path] = None,
    preprocessor_path: Optional[Path] = None,
    feature_names_path: Optional[Path] = None,
) -> Tuple[Any, Any, List[str]]:
    """
    Load fitted model, preprocessor, and feature names from disk.

    Returns:
        (model, preprocessor, feature_names).
    """
    config = load_config()
    paths = config["paths"]
    project_root = Path(__file__).resolve().parent.parent.parent

    model_path = Path(model_path) if model_path else project_root / paths["model_dir"] / paths["best_model_filename"]
    preprocessor_path = Path(preprocessor_path) if preprocessor_path else project_root / paths["model_dir"] / paths["preprocessor_filename"]

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    if feature_names_path:
        import json
        with open(feature_names_path) as f:
            feature_names = json.load(f)
    else:
        fn_path = project_root / paths["model_dir"] / paths.get("feature_names_filename", "feature_names.json")
        if fn_path.exists():
            import json
            with open(fn_path) as f:
                feature_names = json.load(f)
        else:
            feature_names = get_feature_names_after_preprocessing(preprocessor)

    return model, preprocessor, feature_names


def predict(
    X: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    return_proba: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Transform X with preprocessor and predict with model.

    Args:
        X: Raw feature DataFrame (same columns as training, including engineered).
        model: Fitted classifier.
        preprocessor: Fitted ColumnTransformer.
        return_proba: If True, also return probability of positive class.

    Returns:
        (predictions 0/1, probabilities or None).
    """
    X_t = preprocessor.transform(X)
    preds = model.predict(X_t)
    if return_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_t)[:, 1]
        return preds, proba
    return preds, None


def get_top_feature_drivers(
    model: Any,
    feature_names: List[str],
    X_transformed: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Union[str, float]]]:
    """
    Return top drivers for a single sample using feature importance and values.
    For tree models uses feature_importances_; for linear uses coefficients * value.

    X_transformed: single row (1, n_features) or multiple; we use first row.
    """
    if X_transformed.ndim == 2:
        row = X_transformed[0]
    else:
        row = X_transformed

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).flatten()
    else:
        return []

    if len(imp) != len(row) or len(imp) != len(feature_names):
        return []
    # Contribution ≈ importance * (value - 0) for tree; for linear we use coef * value
    if hasattr(model, "coef_"):
        contrib = model.coef_.flatten() * row
    else:
        contrib = imp * row
    indices = np.argsort(np.abs(contrib))[::-1][:top_k]
    return [
        {"feature": feature_names[i], "contribution": float(contrib[i])}
        for i in indices
    ]
