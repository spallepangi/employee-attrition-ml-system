"""Smoke test for training pipeline (short run)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_training_pipeline_short_run():
    """Run pipeline with Optuna disabled and 1 CV fold for speed."""
    from src.pipelines.training_pipeline import run_training_pipeline

    result = run_training_pipeline(
        project_root=PROJECT_ROOT,
        use_optuna=False,
        use_mlflow=False,
        run_shap=False,
    )
    assert "best_model_name" in result
    assert "test_metrics" in result
    assert result["best_model_name"] in ("logistic_regression", "random_forest", "xgboost")
    assert "roc_auc" in result["test_metrics"]
