# Employee Attrition Prediction System

A production-grade end-to-end machine learning system for predicting employee attrition using the IBM HR Analytics Employee Attrition dataset.

## Project Overview

**Business problem:** Predict whether an employee will leave the company (binary classification: Attrition Yes → 1, No → 0) to support retention and HR planning.

**Target column:** `Attrition` (Yes / No, encoded as 1 / 0).

## Dataset Description

- **Source:** IBM HR Analytics Employee Attrition dataset (CSV).
- **Location:** Place the file `WA_Fn-UseC_-HR-Employee-Attrition.csv` in `data/raw/` (or it is loaded from there after copying from project root).
- **Features used:** Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EnvironmentSatisfaction, Gender, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MonthlyIncome, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager.
- **Engineered features:** `income_per_year_of_experience`, `tenure_ratio`, `promotion_gap`.

## System Architecture

```
Employee Attrition Prediction System
├── data/raw          → Raw CSV
├── data/processed    → Processed artifacts (optional)
├── notebooks/       → EDA (exploratory_analysis.ipynb)
├── src/
│   ├── config/       → config.yaml, config_loader
│   ├── data/         → data_loader, data_validation, preprocessing
│   ├── features/     → feature_engineering
│   ├── models/       → train_model, evaluate_model, predict_model
│   ├── pipelines/    → training_pipeline (end-to-end)
│   ├── explainability/ → SHAP analysis
│   └── utils/        → logger
├── api/              → FastAPI prediction service (app.py)
├── dashboard/        → Streamlit app (streamlit_app.py)
├── models/           → Saved model, preprocessor, feature_names, eval + SHAP plots
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## ML Pipeline Explanation

1. **Data ingestion:** Load CSV from `data/raw/`, encode target (Yes→1, No→0), drop unused columns.
2. **Data validation:** Schema check (required columns, dtypes), missing-value checks.
3. **Feature engineering:** Add `income_per_year_of_experience`, `tenure_ratio`, `promotion_gap`.
4. **Train/test split:** Stratified 80/20.
5. **Preprocessing:** ColumnTransformer — numerical: median impute + StandardScaler; categorical: most_frequent impute + OneHotEncoder.
6. **Model training:** Logistic Regression, Random Forest, XGBoost with stratified 5-fold CV; best model by ROC AUC.
7. **Hyperparameter tuning (optional):** Optuna for XGBoost; MLflow logs parameters, metrics, and model.
8. **Evaluation:** Confusion matrix, ROC curve, precision-recall curve, feature importance on test set.
9. **Explainability:** SHAP summary and dependence plots.
10. **Saving:** Best model, preprocessor, and feature names under `models/`.

## How to Run

### 1. Environment

```bash
cd "/Users/santhosh/Documents/Projects/Employee Attrition Prediction System"
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**macOS (Apple Silicon / Intel):** XGBoost needs the OpenMP library. If you see `Library not loaded: libomp.dylib`, install it with Homebrew:

```bash
brew install libomp
```

If you cannot install `libomp`, the pipeline still runs using **Logistic Regression and Random Forest** only (XGBoost is skipped automatically).

### 2. Dataset

Ensure `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` exists. If the CSV is in the project root, copy it:

```bash
cp WA_Fn-UseC_-HR-Employee-Attrition.csv data/raw/
```

### 3. Training (full pipeline)

From project root:

```bash
python -m src.pipelines.training_pipeline
```

Options (edit in code or extend with argparse): `use_optuna=True`, `use_mlflow=True`, `run_shap=True`. Outputs:

- `models/best_model.pkl`
- `models/preprocessor.pkl`
- `models/feature_names.json`
- `models/eval_plots/` (confusion matrix, ROC, PR curve, feature importance)
- `models/shap_plots/` (SHAP summary and dependence plots)

### 4. Start API (FastAPI)

From project root:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

- **Health:** `GET http://localhost:8000/health`
- **Predict:** `POST http://localhost:8000/predict` with JSON body of employee features (see `EmployeeFeatures` in `api/app.py`).

Response: `attrition_probability`, `prediction` (0/1), `top_feature_drivers`.

### 5. Run dashboard (Streamlit)

From project root:

```bash
streamlit run dashboard/streamlit_app.py
```

Tabs: Single prediction, Feature importance, Employee risk scoring, SHAP explanations.

### 6. Docker (API only)

```bash
docker build -t attrition-api .
docker run -p 8000:8000 attrition-api
```

Ensure `models/` contains `best_model.pkl` and `preprocessor.pkl` (e.g. mount or COPY after training).

## Model Interpretation

- **Feature importance:** From tree-based models (or logistic coefficients); saved in `models/eval_plots/feature_importance.png`.
- **SHAP:** Global (summary plot) and local (dependence plots) in `models/shap_plots/`.
- **API:** Each prediction returns `top_feature_drivers` (contribution-style).

## Business Insights

- **Attrition drivers** often include: overtime, job satisfaction, work-life balance, income relative to tenure, tenure ratio, and promotion gap (see EDA notebook and SHAP).
- **Use cases:** Identify high-risk employees, target retention programs, and analyze impact of satisfaction and compensation.

## Configuration

Edit `src/config/config.yaml` for:

- Data paths, train/test split, random seed
- Categorical and numerical feature lists
- Preprocessing (imputer, scaler, encoder)
- Training (CV folds, model params)
- Optuna (n_trials, timeout)
- MLflow (tracking_uri, experiment_name)
- Paths for model, preprocessor, feature names

## Coding Standards

- PEP 8, type hints, docstrings
- Modular layout under `src/` with clear separation of data, features, models, pipelines, and explainability

## License

Use as needed for the project.

---

**Repository:** [github.com/spallepangi/employee-attrition-ml-system](https://github.com/spallepangi/employee-attrition-ml-system)
