"""
Preprocessing pipeline for employee attrition data.
Handles missing values, categorical encoding (OneHotEncoder), and scaling (StandardScaler).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_preprocessing_pipeline(
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    numerical_imputer: str = "median",
    handle_unknown: str = "ignore",
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for preprocessing.

    - Numerical: SimpleImputer (median) -> StandardScaler
    - Categorical: SimpleImputer (most_frequent) -> OneHotEncoder (handle_unknown=ignore)

    Args:
        categorical_features: List of categorical column names.
        numerical_features: List of numerical column names.
        numerical_imputer: Strategy for numerical imputation ('median', 'mean').
        handle_unknown: How to handle unknown categories in OneHotEncoder.

    Returns:
        ColumnTransformer instance (not fitted).
    """
    config = load_config()
    preprocess_config = config.get("preprocessing", {})

    if categorical_features is None:
        categorical_features = config["data"].get("categorical_features", [])
    if numerical_features is None:
        numerical_features = config["data"].get("numerical_features", [])

    num_imputer = preprocess_config.get("numerical_imputer", numerical_imputer)
    cat_imputer = preprocess_config.get("categorical_imputer", "most_frequent")
    unknown = preprocess_config.get("handle_unknown", handle_unknown)

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=num_imputer)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=cat_imputer)),
            ("onehot", OneHotEncoder(handle_unknown=unknown, sparse_output=False)),
        ]
    )

    transformers = []
    if numerical_features:
        transformers.append(("num", numerical_pipeline, numerical_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names_after_preprocessing(transformer: ColumnTransformer) -> List[str]:
    """
    Get ordered feature names after ColumnTransformer (one-hot names for categoricals).

    Args:
        transformer: Fitted ColumnTransformer.

    Returns:
        List of feature names in the same order as the transformed matrix columns.
    """
    names = []
    for name, pipe, columns in transformer.transformers_:
        if name == "num":
            names.extend(columns)
        elif name == "cat":
            enc = pipe.named_steps["onehot"]
            names.extend(enc.get_feature_names_out(columns))
    return names


def fit_preprocessor(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Fit the preprocessing pipeline on X and return the fitted transformer and feature names.

    Only columns present in X are used; lists are filtered to existing columns.

    Args:
        X: Training feature DataFrame (may include engineered features).
        categorical_features: Categorical columns. None = use config.
        numerical_features: Numerical columns. None = use config.

    Returns:
        (fitted ColumnTransformer, list of output feature names).
    """
    config = load_config()
    cat = categorical_features or config["data"].get("categorical_features", [])
    num = numerical_features or config["data"].get("numerical_features", [])

    cat = [c for c in cat if c in X.columns]
    num = [c for c in num if c in X.columns]

    pipeline = build_preprocessing_pipeline(
        categorical_features=cat,
        numerical_features=num,
    )
    X_trans = pipeline.fit_transform(X)
    feature_names = get_feature_names_after_preprocessing(pipeline)
    logger.info("Preprocessor fitted: %d features, shape %s", len(feature_names), X_trans.shape)
    return pipeline, feature_names


def save_preprocessor(transformer: ColumnTransformer, path: Union[str, Path]) -> None:
    """Save fitted preprocessor to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, path)
    logger.info("Saved preprocessor to %s", path)


def load_preprocessor(path: Union[str, Path]) -> ColumnTransformer:
    """Load preprocessor from disk."""
    return joblib.load(path)
