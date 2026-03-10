"""
Data loader for the IBM HR Analytics Employee Attrition dataset.
Loads raw CSV and returns DataFrame with target encoding.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(
    data_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load raw employee attrition CSV from data/raw.

    Args:
        data_path: Full path to CSV. If None, path is built from config and project_root.
        project_root: Project root directory. If None, inferred from config file location.

    Returns:
        DataFrame with raw data. Target column 'Attrition' is converted to binary (Yes->1, No->0).
    """
    config = load_config()
    if project_root is None:
        # Assume project root is two levels up from src/config
        project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(project_root)

    if data_path is None:
        raw_dir = project_root / config["data"]["raw_dir"]
        filename = config["data"]["raw_filename"]
        data_path = raw_dir / filename

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {data_path}")

    logger.info("Loading raw data from %s", data_path)
    df = pd.read_csv(data_path)

    # Encode target: Yes -> 1, No -> 0
    target_col = config["project"]["target_column"]
    target_mapping = config["project"]["target_mapping"]
    if target_col in df.columns:
        df[target_col] = df[target_col].map(target_mapping)
        if df[target_col].isna().any():
            raise ValueError(
                f"Target column '{target_col}' contains values not in {list(target_mapping.keys())}"
            )

    # Drop columns that are not used in modeling
    drop_cols = config["data"].get("drop_columns", [])
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        logger.info("Dropped columns: %s", existing_drop)

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df
