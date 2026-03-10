"""
FastAPI prediction service for Employee Attrition.
POST /predict returns attrition probability, prediction, and top feature drivers.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.config_loader import load_config
from src.features.feature_engineering import apply_feature_engineering
from src.models.predict_model import load_artifacts, predict, get_top_feature_drivers
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict employee attrition and get explainability.",
    version="1.0.0",
)

# Load artifacts at startup (paths relative to project root)
_model = None
_preprocessor = None
_feature_names = None


def _get_artifacts() -> tuple:
    global _model, _preprocessor, _feature_names
    if _model is None:
        try:
            _model, _preprocessor, _feature_names = load_artifacts(
                model_path=PROJECT_ROOT / "models" / "best_model.pkl",
                preprocessor_path=PROJECT_ROOT / "models" / "preprocessor.pkl",
            )
        except Exception as e:
            logger.error("Failed to load model artifacts: %s", e)
            raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")
    return _model, _preprocessor, _feature_names


# Pydantic schema for request body: all raw features used in training
class EmployeeFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    BusinessTravel: str = Field(..., description="Travel_Rarely | Travel_Frequently | Non-Travel")
    DailyRate: float = Field(..., ge=0)
    Department: str = Field(...)
    DistanceFromHome: int = Field(..., ge=0, le=30)
    Education: int = Field(..., ge=1, le=5)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: str = Field(...)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: str = Field(...)
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MonthlyIncome: float = Field(..., ge=0)
    NumCompaniesWorked: int = Field(..., ge=0, le=10)
    OverTime: str = Field(..., description="Yes | No")
    PercentSalaryHike: float = Field(..., ge=0, le=100)
    PerformanceRating: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0, le=50)
    TrainingTimesLastYear: int = Field(..., ge=0, le=10)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=50)
    YearsInCurrentRole: int = Field(..., ge=0, le=25)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=25)
    YearsWithCurrManager: int = Field(..., ge=0, le=25)

    class Config:
        extra = "ignore"


class PredictResponse(BaseModel):
    attrition_probability: float
    prediction: int
    top_feature_drivers: List[Dict[str, Any]]


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_attrition(features: EmployeeFeatures) -> PredictResponse:
    """
    Predict employee attrition. Returns probability, binary prediction, and top drivers.
    """
    model, preprocessor, feature_names = _get_artifacts()

    # Convert to DataFrame (single row)
    row = features.model_dump()
    X = pd.DataFrame([row])

    # Feature engineering (same as training)
    X = apply_feature_engineering(X)

    # Ensure column order matches what preprocessor expects (only existing columns)
    preds, proba = predict(X, model, preprocessor, return_proba=True)
    probability = float(proba[0]) if proba is not None else float(preds[0])
    prediction = int(preds[0])

    # Top drivers: need transformed row
    X_t = preprocessor.transform(X)
    top_drivers = get_top_feature_drivers(model, feature_names, X_t, top_k=5)

    logger.info("Prediction: prediction=%s probability=%.4f", prediction, probability)
    return PredictResponse(
        attrition_probability=round(probability, 4),
        prediction=prediction,
        top_feature_drivers=top_drivers,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
