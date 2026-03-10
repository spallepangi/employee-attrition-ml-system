"""
Feature engineering for employee attrition prediction.
Adds domain-informed derived features to improve model signal.
"""

from typing import List, Optional

import pandas as pd

from src.config.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_income_per_year_of_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature: income_per_year_of_experience = MonthlyIncome / TotalWorkingYears.

    Reasoning: Employees with high income relative to tenure may be more satisfied
    (fast growth) or more likely to leave if underpaid for experience. Captures
    compensation efficiency and career trajectory.
    """
    if "TotalWorkingYears" not in df.columns or "MonthlyIncome" not in df.columns:
        return df
    # Avoid division by zero; use 1 where TotalWorkingYears is 0
    denom = df["TotalWorkingYears"].replace(0, 1)
    df = df.copy()
    df["income_per_year_of_experience"] = (df["MonthlyIncome"] / denom).astype(float)
    return df


def add_tenure_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature: tenure_ratio = YearsAtCompany / TotalWorkingYears.

    Reasoning: Ratio of time at current company vs total career. Low ratio with
    high TotalWorkingYears suggests job-hopping; high ratio suggests loyalty.
    Helps capture stability and commitment patterns linked to attrition.
    """
    if "YearsAtCompany" not in df.columns or "TotalWorkingYears" not in df.columns:
        return df
    denom = df["TotalWorkingYears"].replace(0, 1)
    df = df.copy()
    df["tenure_ratio"] = (df["YearsAtCompany"] / denom).astype(float)
    return df


def add_promotion_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature: promotion_gap = YearsInCurrentRole - YearsSinceLastPromotion.

    Reasoning: Large gap (long time in role without recent promotion) can indicate
    stagnation and dissatisfaction. Negative values imply recent promotion.
    Strong driver of attrition in HR literature.
    """
    if "YearsInCurrentRole" not in df.columns or "YearsSinceLastPromotion" not in df.columns:
        return df
    df = df.copy()
    df["promotion_gap"] = (
        df["YearsInCurrentRole"].astype(float) - df["YearsSinceLastPromotion"].astype(float)
    )
    return df


def apply_feature_engineering(
    df: pd.DataFrame,
    income_per_year: bool = True,
    tenure_ratio: bool = True,
    promotion_gap: bool = True,
) -> pd.DataFrame:
    """
    Apply all configured engineered features to the DataFrame.

    Args:
        df: Input DataFrame with raw and/or preprocessed columns.
        income_per_year: Whether to add income_per_year_of_experience.
        tenure_ratio: Whether to add tenure_ratio.
        promotion_gap: Whether to add promotion_gap.

    Returns:
        New DataFrame with additional columns. Original columns preserved.
    """
    config = load_config()
    fe_config = config.get("feature_engineering", {})
    income_per_year = fe_config.get("income_per_year", income_per_year)
    tenure_ratio = fe_config.get("tenure_ratio", tenure_ratio)
    promotion_gap = fe_config.get("promotion_gap", promotion_gap)

    out = df.copy()
    if income_per_year:
        out = add_income_per_year_of_experience(out)
    if tenure_ratio:
        out = add_tenure_ratio(out)
    if promotion_gap:
        out = add_promotion_gap(out)

    added = []
    if income_per_year and "income_per_year_of_experience" in out.columns:
        added.append("income_per_year_of_experience")
    if tenure_ratio and "tenure_ratio" in out.columns:
        added.append("tenure_ratio")
    if promotion_gap and "promotion_gap" in out.columns:
        added.append("promotion_gap")
    if added:
        logger.info("Applied feature engineering: %s", added)
    return out


def get_engineered_feature_names() -> List[str]:
    """Return list of engineered feature names (for config/numerical list)."""
    return ["income_per_year_of_experience", "tenure_ratio", "promotion_gap"]
