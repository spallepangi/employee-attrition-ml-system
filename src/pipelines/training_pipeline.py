"""
End-to-end training pipeline: data ingestion -> validation -> preprocessing ->
feature engineering -> model training -> evaluation -> saving.
Optionally runs Optuna tuning for XGBoost and logs to MLflow.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config.config_loader import load_config
from src.data.data_loader import load_raw_data
from src.data.data_validation import validate_data
from src.data.preprocessing import (
    fit_preprocessor,
    save_preprocessor,
    get_feature_names_after_preprocessing,
)
from src.features.feature_engineering import apply_feature_engineering, get_engineered_feature_names
from src.models.train_model import train_all_models, save_model, _XGBOOST_AVAILABLE
from src.models.evaluate_model import evaluate_model
from src.explainability.shap_analysis import run_shap_analysis
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline(
    project_root: Optional[Path] = None,
    use_optuna: bool = True,
    use_mlflow: bool = True,
    optuna_trials: Optional[int] = None,
    run_shap: bool = True,
) -> dict:
    """
    Execute the full training pipeline.

    Steps:
    1. Data ingestion
    2. Data validation
    3. Feature engineering
    4. Train/test split (stratified)
    5. Preprocessing fit on train
    6. Model training (LR, RF, XGBoost) with stratified CV
    7. Optional: Optuna tuning for XGBoost
    8. Model evaluation on test set
    9. Save best model, preprocessor, feature names

    Args:
        project_root: Project root directory. Default: inferred.
        use_optuna: If True, run Optuna tuning for XGBoost and use best params.
        use_mlflow: If True, log experiments to MLflow.
        optuna_trials: Number of Optuna trials (default from config).

    Returns:
        Dict with keys: best_model_name, metrics, test_metrics, paths.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(project_root)

    config = load_config()
    target_col = config["project"]["target_column"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    # 1. Data ingestion
    logger.info("Step 1: Data ingestion")
    df = load_raw_data(project_root=project_root)

    # 2. Data validation
    logger.info("Step 2: Data validation")
    validate_data(df)

    # 3. Feature engineering
    logger.info("Step 3: Feature engineering")
    df = apply_feature_engineering(df)
    engineered = get_engineered_feature_names()
    numerical_features = config["data"]["numerical_features"] + [
        e for e in engineered if e in df.columns
    ]
    categorical_features = [c for c in config["data"]["categorical_features"] if c in df.columns]

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # 4. Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    # 5. Preprocessing
    logger.info("Step 4: Preprocessing")
    preprocessor, feature_names = fit_preprocessor(
        X_train,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Optional: Optuna + MLflow for XGBoost (skip if XGBoost unavailable, e.g. libomp missing on macOS)
    if use_optuna and not _XGBOOST_AVAILABLE:
        logger.warning("Optuna requested but XGBoost not available; training all models without tuning.")
        use_optuna = False
    if use_optuna:
        best_model, best_name, all_metrics = _run_optuna_xgboost(
            X_train_t, np.array(y_train),
            X_test_t, np.array(y_test),
            feature_names,
            use_mlflow=use_mlflow,
            n_trials=optuna_trials,
        )
    else:
        # 6. Model training
        logger.info("Step 5: Model training")
        best_model, best_name, all_metrics = train_all_models(
            X_train_t, np.array(y_train), feature_names=feature_names
        )

    if use_mlflow and not use_optuna:
        _log_to_mlflow(best_name, best_model, all_metrics, None)

    # 7. Evaluation on test set
    logger.info("Step 6: Model evaluation")
    test_metrics = evaluate_model(
        best_model,
        X_test_t,
        np.array(y_test),
        feature_names,
        output_dir=project_root / "models" / "eval_plots",
    )

    # 7b. SHAP explainability (optional)
    if run_shap:
        logger.info("Step 6b: SHAP analysis")
        run_shap_analysis(
            best_model,
            X_train_t,
            X_explain=X_test_t,
            feature_names=feature_names,
            output_dir=project_root / "models" / "shap_plots",
            top_dependence=3,
        )

    # 8. Save artifacts
    logger.info("Step 7: Saving artifacts")
    model_dir = project_root / config["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(best_model, path=model_dir / config["paths"]["best_model_filename"])
    save_preprocessor(preprocessor, path=model_dir / config["paths"]["preprocessor_filename"])
    with open(model_dir / config["paths"]["feature_names_filename"], "w") as f:
        json.dump(feature_names, f, indent=2)

    return {
        "best_model_name": best_name,
        "cv_metrics": all_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_dir / config["paths"]["best_model_filename"]),
    }


def _run_optuna_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    use_mlflow: bool = True,
    n_trials: Optional[int] = None,
) -> tuple:
    """Run Optuna study for XGBoost; return best model, name, and all_metrics dict."""
    import optuna
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import xgboost as xgb

    config = load_config()
    optuna_config = config.get("optuna", {})
    n_trials = n_trials or optuna_config.get("n_trials", 50)
    timeout = optuna_config.get("timeout", 300)
    cv_folds = config["training"].get("cv_folds", 5)
    random_state = config["training"].get("random_state", 42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        return float(score)

    study = optuna.create_study(direction="maximize", study_name=optuna_config.get("study_name", "xgboost_attrition"))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(
        **best_params,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    best_model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    all_metrics = {
        "xgboost": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
    }

    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "mlruns"))
        mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "employee_attrition"))
        with mlflow.start_run():
            mlflow.log_params(best_params)
            for k, v in all_metrics["xgboost"].items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(best_model, "model")

    return best_model, "xgboost", all_metrics


def _log_to_mlflow(
    model_name: str,
    model: Any,
    all_metrics: dict,
    params: Optional[dict],
) -> None:
    """Log run to MLflow when not using Optuna."""
    try:
        import mlflow
        import mlflow.sklearn
        config = load_config()
        mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "mlruns"))
        mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "employee_attrition"))
        with mlflow.start_run():
            if params:
                mlflow.log_params(params)
            for name, metrics in all_metrics.items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{name}_{k}", v)
            mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


if __name__ == "__main__":
    run_training_pipeline(use_optuna=True, use_mlflow=True, run_shap=True)
