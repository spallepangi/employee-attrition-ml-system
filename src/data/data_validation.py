"""
Data validation for the Employee Attrition dataset.
Checks schema, missing values, and basic constraints.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.config.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that required columns exist and have expected dtypes where applicable.

    Returns:
        (is_valid, list of error messages).
    """
    config = load_config()
    errors: List[str] = []

    target_col = config["project"]["target_column"]
    if target_col not in df.columns:
        errors.append(f"Missing target column: {target_col}")

    cat_features = config["data"].get("categorical_features", [])
    num_features = config["data"].get("numerical_features", [])
    required = set(cat_features + num_features) | {target_col}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if df[target_col].dtype not in ("int64", "float64", "bool") and target_col in df.columns:
        errors.append(f"Target column '{target_col}' should be numeric (0/1).")

    return (len(errors) == 0, errors)


def validate_missing(df: pd.DataFrame, max_missing_ratio: float = 0.5) -> Tuple[bool, List[str]]:
    """
    Check for excessive missing values.

    Args:
        df: Input DataFrame.
        max_missing_ratio: Maximum allowed fraction of missing values per column.

    Returns:
        (is_valid, list of warning/error messages).
    """
    messages: List[str] = []
    for col in df.columns:
        missing = df[col].isna().sum()
        ratio = missing / len(df) if len(df) > 0 else 0
        if ratio > max_missing_ratio:
            messages.append(f"Column '{col}' has {ratio:.1%} missing (max allowed {max_missing_ratio:.1%}).")
        elif ratio > 0:
            logger.info("Column '%s' has %d missing values (%.2f%%)", col, missing, ratio * 100)
    return (len([m for m in messages if "max allowed" in m]) == 0, messages)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Run full validation: schema and missing values.
    Logs warnings and raises ValueError if schema is invalid.

    Returns:
        True if validation passed.
    """
    ok_schema, schema_errors = validate_schema(df)
    if not ok_schema:
        for e in schema_errors:
            logger.error("Validation: %s", e)
        raise ValueError("Schema validation failed: " + "; ".join(schema_errors))

    ok_missing, missing_messages = validate_missing(df)
    for m in missing_messages:
        if "max allowed" in m:
            logger.error("Validation: %s", m)
        else:
            logger.warning("Validation: %s", m)
    if not ok_missing:
        raise ValueError("Missing value validation failed.")

    logger.info("Data validation passed.")
    return True
