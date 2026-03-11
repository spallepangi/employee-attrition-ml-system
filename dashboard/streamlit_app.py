"""
Employee Attrition Prediction — Executive dashboard.
No code displayed; narrative, visuals, and model documentation only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    initial_sidebar_state="expanded",
)

st.title("Employee Attrition Prediction Dashboard")
st.markdown("Understand who might leave and why — so you can act in time.")

# Sidebar
st.sidebar.header("Settings")
use_sample = st.sidebar.checkbox("Use sample employee for prediction", value=True)


@st.cache_resource
def load_model_artifacts():
    try:
        return load_artifacts(
            model_path=PROJECT_ROOT / "models" / "best_model.pkl",
            preprocessor_path=PROJECT_ROOT / "models" / "preprocessor.pkl",
        )
    except Exception:
        return None


@st.cache_data
def get_cached_data():
    df = load_raw_data(project_root=PROJECT_ROOT)
    return apply_feature_engineering(df)


@st.cache_data
def get_evaluation_metrics():
    """Compute model metrics for display (no code shown)."""
    try:
        model, preprocessor, _ = load_artifacts()
        if model is None:
            return None
        config = load_config()
        target_col = config["project"]["target_column"]
        df = get_cached_data()
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_t = preprocessor.transform(X_test)
        y_pred = model.predict(X_test_t)
        y_prob = model.predict_proba(X_test_t)[:, 1]
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "n_test": len(y_test),
        }
    except Exception:
        return None


def _image(path: Path, caption: str = ""):
    """Display image; avoid use_container_width for older Streamlit."""
    if not path.exists():
        return False
    st.image(str(path), caption=caption if caption else None)
    return True


# ----- Tabs -----
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Explanation of the project",
    "Dataset overview",
    "EDA visuals",
    "Actions you can take",
    "Model evaluation",
    "Model comparison",
    "Feature importance",
    "Model explanation",
    "Who might leave?",
    "System design",
])

# --- Tab 1: Explanation of the project ---
with tab1:
    st.header("Explanation of the project")
    st.markdown("""
    **What is this project?**  
    This system predicts whether an employee is **likely to leave** the company (attrition).  
    It uses historical HR data and a machine learning model to score each employee’s risk so that managers and HR can focus retention efforts where they matter most.

    **Why does it matter?**  
    Losing key people is costly (recruitment, knowledge loss, team morale).  
    By spotting at-risk employees early, you can have conversations, adjust workload, improve satisfaction, or plan succession — instead of reacting after someone quits.

    **How does it work?**  
    The model uses factors such as job satisfaction, overtime, income, tenure, department, and work–life balance.  
    It outputs a **likelihood to leave** (e.g. 15% or 72%) and a **risk level** (Low / Medium / High).  
    The dashboard also shows **which factors** drive each prediction, so decisions are easier to explain and act on.

    **Who is it for?**  
    HR, people managers, and leadership — no technical background required.  
    Use the **“Who might leave?”** tab to see a list of employees by risk, and the **“Model explanation”** tab to understand why the model says what it says.
    """)

# --- Tab 2: Dataset overview ---
with tab2:
    st.header("Dataset overview")
    try:
        df = get_cached_data()
        config = load_config()
        target_col = config["project"]["target_column"]
        n_rows, n_cols = df.shape
        n_leave = (df[target_col] == 1).sum()
        n_stay = (df[target_col] == 0).sum()
        pct_leave = 100 * n_leave / n_rows if n_rows else 0
    except Exception:
        n_rows = n_cols = n_leave = n_stay = pct_leave = 0

    st.markdown(f"""
    The dataset used to build and evaluate the model has **{n_rows:,} employees** and **{n_cols}** attributes (e.g. age, department, job role, satisfaction, income, tenure).

    **Attrition (target):**  
    - **{n_leave}** employees left (**{pct_leave:.1f}%**)  
    - **{n_stay}** employees stayed (**{100 - pct_leave:.1f}%**)

    The data is **imbalanced**: many more people stayed than left.  
    The model is tuned to account for this so that we can still identify those at risk of leaving reliably.
    """)
    st.markdown("""
    **Main groups of features:**  
    - **Demographics & role:** Age, department, job role, job level.  
    - **Satisfaction & engagement:** Job satisfaction, environment satisfaction, work–life balance, job involvement.  
    - **Compensation & growth:** Monthly income, salary hike %, stock options, years since last promotion.  
    - **Tenure & workload:** Years at company, years with manager, overtime, distance from home.
    """)

# --- Tab 3: EDA visuals ---
with tab3:
    st.header("EDA visuals")
    st.markdown("Exploratory visuals focused on **attrition**: who leaves, and how they differ from those who stay.")
    try:
        df = get_cached_data()
        config = load_config()
        target_col = config["project"]["target_column"]
    except Exception:
        df = None

    if df is not None and target_col in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attrition by department")
            dept_att = df.groupby("Department")[target_col].mean().multiply(100).sort_values(ascending=True)
            st.bar_chart(dept_att)
        with col2:
            st.subheader("Attrition by overtime")
            ot_att = df.groupby("OverTime")[target_col].mean().multiply(100)
            st.bar_chart(ot_att)

        st.subheader("Attrition rate by job satisfaction")
        sat_att = df.groupby("JobSatisfaction")[target_col].mean().multiply(100)
        st.bar_chart(sat_att)

        st.subheader("Attrition rate by work–life balance")
        wlb_att = df.groupby("WorkLifeBalance")[target_col].mean().multiply(100)
        st.bar_chart(wlb_att)

        st.markdown("**Takeaway:** Higher satisfaction and better work–life balance are associated with lower attrition; overtime is associated with higher attrition.")
    else:
        st.info("Load data to see EDA visuals. Run the training pipeline first if needed.")

# --- Tab 4: Actions you can take ---
with tab4:
    st.header("Actions you can take")
    st.markdown("""
    Based on the **project goal** (predict and reduce attrition), here are concrete actions:

    1. **Use the “Who might leave?” tab**  
       Review the list of employees by risk (High / Medium / Low).  
       Prioritize **High risk** for 1:1 conversations or exit interviews.

    2. **Address drivers the model highlights**  
       The model explanation and feature importance show what matters most (e.g. overtime, satisfaction, income).  
       Where possible: reduce unnecessary overtime, improve recognition and growth, or adjust workload.

    3. **Segment by department and role**  
       EDA shows attrition varies by department and job role.  
       Target retention programs (e.g. development, flexibility) to high-attrition segments.

    4. **Track over time**  
       Re-run predictions periodically (e.g. quarterly) and compare risk levels.  
       Use improving or worsening risk as a signal for whether actions are working.

    5. **Combine with HR processes**  
       Feed risk scores into stay interviews, succession planning, and compensation reviews — not as the only factor, but as one input among others.
    """)

# --- Tab 5: Model evaluation ---
with tab5:
    st.header("Model evaluation")
    metrics = get_evaluation_metrics()
    if metrics:
        st.subheader("Which metrics matter for this project?")
        st.markdown("""
        - **ROC AUC (primary):** We use it to **rank** employees by risk. High ROC AUC means at-risk people tend to get higher scores than those who stay, so when you act on the top 20% by risk, you’re really focusing on leavers. This is the main metric we use to choose the model.
        - **Recall:** Important so we **don’t miss** people who actually leave. Higher recall means we flag more of the real leavers (even if we also flag some stayers). For retention, missing a leaver is often costlier than an extra conversation with a stayer.
        - **Precision:** Keeps **false alarms** in check. High precision means when we say “at risk,” we’re right more often, so HR and managers can trust the list.
        - **F1:** Balances precision and recall when you care about both.
        - **Accuracy:** Less informative here because most people stay; a “predict everyone stays” model can look accurate but is useless. We report it for context only.
        """)
        st.subheader("Test-set performance (numbers)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1 score", f"{metrics['f1']:.2%}")
        c5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        st.caption(f"Test set: {metrics['n_test']} employees.")
        st.subheader("Business metrics (interpretation)")
        n_test = metrics["n_test"]
        recall = metrics["recall"]
        precision = metrics["precision"]
        st.markdown(f"""
        - **Catch rate:** Of all employees who actually leave in the test set, the model flags about **{recall:.0%}** as at risk. So we “catch” that share of leavers when using the model’s risk score.
        - **Action list quality:** When we act on everyone the model labels “will leave,” about **{precision:.0%}** actually leave; the rest are false alarms. That’s the expected hit rate on a risk-based action list.
        - **Use in practice:** Rank employees by predicted risk and focus retention efforts on the top 10–20%. ROC AUC tells us how well that ranking separates real leavers from stayers.
        """)
        st.subheader("Confusion matrix")
        _image(PROJECT_ROOT / "models" / "eval_plots" / "confusion_matrix.png", "Rows = actual, columns = predicted.")
        st.subheader("ROC curve")
        _image(PROJECT_ROOT / "models" / "eval_plots" / "roc_curve.png", "Curve above the diagonal = better than random.")
        st.subheader("Precision–recall curve")
        _image(PROJECT_ROOT / "models" / "eval_plots" / "precision_recall_curve.png", "Trade-off between precision and recall.")
    else:
        st.info("Run the training pipeline and ensure the model is saved to see evaluation numbers and plots.")

# --- Tab 6: Model comparison ---
with tab6:
    st.header("Model comparison")
    st.markdown("We trained three models and compared them on the same test set. Below: their performance and which one was chosen.")
    comp_path = PROJECT_ROOT / "models" / "model_comparison.json"
    if comp_path.exists():
        import json
        with open(comp_path) as f:
            comp = json.load(f)
        chosen = comp.get("chosen_model", "")
        reason = comp.get("chosen_reason", "")
        for k in ["chosen_model", "chosen_reason"]:
            comp.pop(k, None)
        if comp:
            roc_data = {name: m.get("roc_auc", 0) for name, m in comp.items() if isinstance(m, dict)}
            if roc_data:
                st.subheader("ROC AUC by model (test set)")
                st.bar_chart(pd.DataFrame({"ROC AUC": roc_data}))
            st.subheader("All metrics by model")
            rows = []
            for name, m in comp.items():
                if isinstance(m, dict):
                    rows.append({"Model": name.replace("_", " ").title(), **m})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Model"))
        st.subheader("Chosen model and reason")
        st.markdown(f"**Chosen model:** {chosen.replace('_', ' ').title()}.")
        st.markdown(reason if reason else "Chosen by highest ROC AUC on the test set so we rank at-risk employees well.")
    else:
        st.info("Run the training pipeline to generate model comparison (saved as model_comparison.json).")

# --- Tab 7: Feature importance ---
with tab7:
    st.header("Feature importance")
    st.markdown("Which inputs matter most for the model’s predictions (overall). The chart is **horizontal**: most important features at the **top**; bars show relative impact.")
    if _image(PROJECT_ROOT / "models" / "eval_plots" / "feature_importance.png"):
        st.subheader("How these features matter for prediction")
        st.markdown("""
        The model uses many features; the chart shows their **relative importance** (how much each drives the prediction).

        - **Age, tenure, income:** Often among the top drivers. Younger or shorter-tenure employees, or those with lower relative income, may show higher attrition risk. Income and tenure also feed into engineered features (e.g. income per year of experience, tenure ratio), which can matter a lot.
        - **Daily rate / compensation:** Reflects how the model uses pay-related signals; low or out-of-line compensation can push risk up.
        - **Job and environment satisfaction, work–life balance:** Lower satisfaction and poorer work–life balance usually **increase** predicted risk; improving these is a direct lever for retention.
        - **Overtime:** Working overtime is often associated with higher attrition risk (burnout, imbalance).
        - **Department and job role:** Different roles and departments have different typical attrition; the model uses these to adjust risk.
        - **Years at company, years with manager, years since last promotion:** Tenure and growth signals. Long time without promotion or with the same manager can lift risk.
        - **Engineered features (e.g. income per year of experience, tenure ratio, promotion gap):** Summarize compensation vs experience, loyalty vs job-hopping, and career stagnation; they often rank high in importance.

        Use this view to **prioritize** which levers (satisfaction, overtime, compensation, growth) to focus on in retention and to **explain** to stakeholders why the model emphasizes certain factors.
        """)
    else:
        st.info("Run the training pipeline to generate the feature importance plot (horizontal chart with many features).")

# --- Tab 8: Model explanation (SHAP) ---
with tab8:
    st.header("Model explanation")
    st.markdown("This chart shows **how each feature pushes** the prediction toward “leave” or “stay.” Red = pushes toward leave; blue = pushes toward stay. Features are ordered by impact.")
    summary_path = PROJECT_ROOT / "models" / "shap_plots" / "shap_summary.png"
    if summary_path.exists():
        st.image(str(summary_path), caption="Feature impact on attrition prediction (red = toward leave, blue = toward stay).")
    else:
        st.info("Run the training pipeline with SHAP enabled to generate this explanation plot.")

# --- Tab 9: Who might leave? ---
with tab9:
    st.header("Who might leave?")
    st.markdown("Employees ranked by **risk level** so you can prioritize who to reach out to. Risk is Low (likely to stay), Medium (monitor), or High (consider action).")
    _artifacts = load_model_artifacts()
    model, preprocessor, _ = (_artifacts if _artifacts else (None, None, None))
    if model is not None:
        n_sample = st.slider("How many employees to review?", 10, 200, 50)
        try:
            df = get_cached_data()
            config = load_config()
            target_col = config["project"]["target_column"]
            df = df.drop(columns=[target_col]).head(n_sample)
            preds, proba = predict(df, model, preprocessor, return_proba=True)
            proba = proba if proba is not None else preds.astype(float)
            def risk_band(p):
                if p < 0.25: return "Low"
                if p < 0.50: return "Medium"
                return "High"
            df["Likelihood to leave"] = [f"{p:.0%}" for p in proba]
            df["Risk level"] = [risk_band(p) for p in proba]
            df["Outlook"] = ["At risk — consider outreach" if pr == 1 else "Likely to stay" for pr in preds]
            high_risk = sum(1 for p in proba if risk_band(p) == "High")
            medium_risk = sum(1 for p in proba if risk_band(p) == "Medium")
            low_risk = sum(1 for p in proba if risk_band(p) == "Low")
            c1, c2, c3 = st.columns(3)
            c1.metric("High risk (consider action)", f"{high_risk} employees")
            c2.metric("Medium risk (monitor)", f"{medium_risk} employees")
            c3.metric("Low risk (likely to stay)", f"{low_risk} employees")
            st.subheader("Employee list (by risk)")
            display_df = df[["Risk level", "Likelihood to leave", "Outlook"]].head(25)
            st.dataframe(display_df)
            risk_counts = df["Risk level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
            st.subheader("How risk is distributed")
            st.bar_chart(risk_counts)
        except Exception as e:
            st.warning("Could not load data or run prediction. Check that the pipeline has been run.")
    else:
        st.info("Run the training pipeline first so the model is available.")

# --- Tab 10: System design ---
with tab10:
    st.header("System design & pipeline documentation")
    st.markdown("End-to-end architecture of the Employee Attrition Prediction system: from raw data to trained model and how it is served.")

    design_path = PROJECT_ROOT / "models" / "system_design.png"
    if design_path.exists():
        st.subheader("Architecture diagram")
        st.image(str(design_path), caption="Data → Validation → Feature engineering → Encode & scale → Train/Test split → LR / RF / XGBoost → Best model (ROC AUC) → FastAPI & Streamlit.")
    else:
        st.info("Run `python dashboard/generate_system_design.py` from the project root to generate the diagram.")

    st.subheader("Pipeline documentation")
    st.markdown("""
    **1. Data ingestion**  
    Raw IBM HR Analytics CSV is loaded from `data/raw/`. The target column (Attrition) is encoded as Yes → 1, No → 0. Identifiers and constants (e.g. EmployeeNumber, Over18) are dropped.

    **2. Data validation**  
    Schema is checked: required columns (demographics, satisfaction, tenure, compensation, etc.) must exist. Missing-value checks ensure no column has excessive missing data before training.

    **3. Feature engineering**  
    Three derived features are added:  
    - **Income per year of experience** = MonthlyIncome / TotalWorkingYears (compensation vs experience).  
    - **Tenure ratio** = YearsAtCompany / TotalWorkingYears (loyalty vs job-hopping).  
    - **Promotion gap** = YearsInCurrentRole − YearsSinceLastPromotion (career stagnation).

    **4. Train/test split**  
    Data is split 80% train / 20% test with stratification so the proportion of leavers is similar in both sets.

    **5. Preprocessing**  
    - **Numerical features:** Missing values filled with median; then standardized (zero mean, unit variance).  
    - **Categorical features:** Missing values filled with most frequent category; then one-hot encoded.  
    The same fitted preprocessor is saved and used at prediction time.

    **6. Model training**  
    Three models are trained and evaluated on the test set:  
    - **Logistic Regression** (baseline, interpretable).  
    - **Random Forest** (non-linear, robust).  
    - **XGBoost** (tuned with Optuna for 50 trials; typically best ROC AUC).  
    The **best model by ROC AUC** is saved (e.g. XGBoost), along with the preprocessor and feature names. Model comparison metrics are written to `models/model_comparison.json`.

    **7. Model evaluation**  
    Confusion matrix, ROC curve, precision–recall curve, and feature importance plot are generated and saved under `models/eval_plots/`. Optional SHAP summary is saved under `models/shap_plots/`.

    **8. Serving**  
    - **FastAPI** (`api/app.py`): Loads the saved model and preprocessor; exposes `POST /predict` with employee features and returns attrition probability, prediction, and top feature drivers.  
    - **Streamlit** (this app): Loads the same artifacts; provides risk lists, EDA, model comparison, feature importance, SHAP explanation, and system design. No code is shown in the UI.

    **How to run the pipeline**  
    From the project root with the virtual environment activated, run:  
    `python -m src.pipelines.training_pipeline`  
    This runs data load → validation → feature engineering → split → preprocessing → training (with Optuna for XGBoost) → evaluation → SHAP (if enabled) → save model, preprocessor, feature names, and model_comparison.json.  

    **How to regenerate the diagram**  
    Run: `python dashboard/generate_system_design.py`. The image is saved as `models/system_design.png` and displayed in this tab.
    """)
