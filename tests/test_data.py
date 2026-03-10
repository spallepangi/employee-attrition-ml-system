"""Tests for data loading, validation, and preprocessing."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_raw_data
from src.data.data_validation import validate_schema, validate_data
from src.data.preprocessing import build_preprocessing_pipeline, fit_preprocessor


def test_load_raw_data():
    df = load_raw_data(project_root=PROJECT_ROOT)
    assert isinstance(df, pd.DataFrame)
    assert "Attrition" in df.columns
    assert set(df["Attrition"].dropna().unique()).issubset({0, 1})
    assert len(df) > 0


def test_validate_schema():
    df = load_raw_data(project_root=PROJECT_ROOT)
    ok, errors = validate_schema(df)
    assert ok, errors
    assert len(errors) == 0


def test_validate_data():
    df = load_raw_data(project_root=PROJECT_ROOT)
    validate_data(df)


def test_preprocessing_pipeline():
    df = load_raw_data(project_root=PROJECT_ROOT)
    from src.config.config_loader import load_config
    config = load_config()
    cat = [c for c in config["data"]["categorical_features"] if c in df.columns]
    num = [c for c in config["data"]["numerical_features"] if c in df.columns]
    pipe = build_preprocessing_pipeline(categorical_features=cat, numerical_features=num)
    assert pipe is not None
    X = df.drop(columns=["Attrition"])
    from src.features.feature_engineering import apply_feature_engineering
    X = apply_feature_engineering(X)
    cat = [c for c in cat if c in X.columns]
    num = [c for c in num if c in X.columns]
    num += [e for e in ["income_per_year_of_experience", "tenure_ratio", "promotion_gap"] if e in X.columns]
    trans, names = fit_preprocessor(X, categorical_features=cat, numerical_features=num)
    X_t = trans.transform(X.head(10))
    assert X_t.shape[0] == 10
    assert X_t.shape[1] == len(names)
