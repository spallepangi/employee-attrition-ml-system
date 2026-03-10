"""
Streamlit dashboard: Employee attrition prediction, feature importance, risk scoring, SHAP plots.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Optional

import pandas as pd
import streamlit as st

from src.config.config_loader import load_config
from src.data.data_loader import load_raw_data
from src.features.feature_engineering import apply_feature_engineering
from src.models.predict_model import load_artifacts, predict, get_top_feature_drivers

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Employee Attrition Prediction Dashboard")
st.markdown("Predict attrition risk and explore feature importance and SHAP explanations.")

# Sidebar: data source and model info
st.sidebar.header("Settings")
use_sample = st.sidebar.checkbox("Use sample employee from dataset", value=True)

@st.cache_resource
def load_model_artifacts():
    try:
        return load_artifacts(
            model_path=PROJECT_ROOT / "models" / "best_model.pkl",
            preprocessor_path=PROJECT_ROOT / "models" / "preprocessor.pkl",
        )
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}. Run the training pipeline first.")
        return None


def load_sample_employee():
    config = load_config()
    df = load_raw_data(project_root=PROJECT_ROOT)
    df = apply_feature_engineering(df)
    # Drop target for display/prediction
    target_col = config["project"]["target_column"]
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    return df.sample(1, random_state=42).iloc[0].to_dict(), df


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single prediction",
    "Feature importance",
    "Who might leave?",
    "Why the model thinks so",
])

with tab1:
    st.header("Employee attrition prediction")
    _artifacts = load_model_artifacts()
    model, preprocessor, feature_names = (_artifacts if _artifacts else (None, None, None))

    if model is not None:
        if use_sample:
            sample_dict, full_df = load_sample_employee()
            st.subheader("Sample employee (from dataset)")
            st.json(sample_dict)
            X = pd.DataFrame([sample_dict])
        else:
            st.info("Enter employee features below or use 'Use sample employee from dataset' in the sidebar.")
            # Minimal form for key fields; for full form we could use expanders
            col1, col2, col3 = st.columns(3)
            with col1:
                Age = st.number_input("Age", min_value=18, max_value=100, value=35)
                Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
                JobRole = st.selectbox("Job Role", [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources",
                ])
            with col2:
                MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=5000)
                YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=50, value=5)
                OverTime = st.selectbox("OverTime", ["Yes", "No"])
            with col3:
                JobSatisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                WorkLifeBalance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
                EnvironmentSatisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)

            # Build minimal row with defaults for required fields
            sample_dict = {
                "Age": Age,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 800,
                "Department": Department,
                "DistanceFromHome": 10,
                "Education": 3,
                "EnvironmentSatisfaction": EnvironmentSatisfaction,
                "Gender": "Male",
                "JobInvolvement": 3,
                "JobLevel": 2,
                "JobRole": JobRole,
                "JobSatisfaction": JobSatisfaction,
                "MonthlyIncome": float(MonthlyIncome),
                "NumCompaniesWorked": 2,
                "OverTime": OverTime,
                "PercentSalaryHike": 12,
                "PerformanceRating": 3,
                "StockOptionLevel": 1,
                "TotalWorkingYears": 10,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": WorkLifeBalance,
                "YearsAtCompany": YearsAtCompany,
                "YearsInCurrentRole": 3,
                "YearsSinceLastPromotion": 2,
                "YearsWithCurrManager": 3,
            }
            X = pd.DataFrame([sample_dict])

        X = apply_feature_engineering(X)
        preds, proba = predict(X, model, preprocessor, return_proba=True)
        prob = float(proba[0]) if proba is not None else float(preds[0])
        pred = int(preds[0])

        st.metric("Attrition probability", f"{prob:.2%}")
        st.metric("Prediction", "Yes (leave)" if pred == 1 else "No (stay)")

        X_t = preprocessor.transform(X)
        drivers = get_top_feature_drivers(model, feature_names, X_t, top_k=5)
        st.subheader("Top feature drivers")
        st.dataframe(pd.DataFrame(drivers))

with tab2:
    st.header("Feature importance")
    try:
        model_dir = PROJECT_ROOT / "models"
        imp_path = model_dir / "eval_plots" / "feature_importance.png"
        if imp_path.exists():
            st.image(str(imp_path), use_container_width=True)
        else:
            st.info("Run the training pipeline to generate feature importance plot.")
    except Exception as e:
        st.warning(str(e))

with tab3:
    st.header("Who might leave?")
    st.markdown(
        "See which employees are **most likely to leave** so you can prioritize retention efforts. "
        "Risk is shown as **Low**, **Medium**, or **High** based on the model’s prediction."
    )
    _artifacts = load_model_artifacts()
    model, preprocessor, _ = (_artifacts if _artifacts else (None, None, None))
    if model is not None:
        n_sample = st.slider("How many employees to review?", 10, 200, 50, help="More people = longer load time.")
        df = load_raw_data(project_root=PROJECT_ROOT)
        target_col = load_config()["project"]["target_column"]
        df = apply_feature_engineering(df).drop(columns=[target_col])
        df = df.head(n_sample)
        preds, proba = predict(df, model, preprocessor, return_proba=True)
        proba = proba if proba is not None else preds.astype(float)
        # Risk bands for executives (interpretable)
        def risk_band(p):
            if p < 0.25:
                return "Low"
            if p < 0.50:
                return "Medium"
            return "High"
        df["Likelihood to leave"] = [f"{p:.0%}" for p in proba]
        df["Risk level"] = [risk_band(p) for p in proba]
        df["Outlook"] = ["At risk — consider outreach" if pr == 1 else "Likely to stay" for pr in preds]
        # Summary metrics
        high_risk = sum(1 for p in proba if risk_band(p) == "High")
        medium_risk = sum(1 for p in proba if risk_band(p) == "Medium")
        low_risk = sum(1 for p in proba if risk_band(p) == "Low")
        c1, c2, c3 = st.columns(3)
        c1.metric("High risk (consider action)", f"{high_risk} employees", help="Likelihood to leave ≥ 50%")
        c2.metric("Medium risk (monitor)", f"{medium_risk} employees", help="Likelihood 25–50%")
        c3.metric("Low risk (likely to stay)", f"{low_risk} employees", help="Likelihood < 25%")
        st.markdown("#### Employee list (by risk)")
        display_df = df[["Risk level", "Likelihood to leave", "Outlook"]].head(25)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("#### How risk is distributed")
        risk_counts = df["Risk level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        st.bar_chart(risk_counts)
    else:
        st.info("Load model first (see Single prediction tab).")

with tab4:
    st.header("Why the model thinks so")
    st.markdown(
        "This view shows **which factors** drive the model’s predictions. "
        "Features higher on the chart have more impact on whether someone is predicted to leave."
    )
    try:
        model_dir = PROJECT_ROOT / "models" / "shap_plots"
        summary_path = model_dir / "shap_summary.png"
        if summary_path.exists():
            st.image(
                str(summary_path),
                use_container_width=True,
                caption="Feature impact on attrition prediction (red = pushes toward leave, blue = pushes toward stay)",
            )
            st.caption("Re-run the training pipeline to regenerate this plot with different settings.")
        else:
            st.info("Run the training pipeline with SHAP enabled to generate this explanation plot.")
    except Exception as e:
        st.warning(str(e))
