import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import statsmodels.api as sm  # noqa: F401  statsmodels is a Python library focused on statistical analysis.
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small CSS polish for a cleaner, more "product" look.
st.markdown(
    """
    <style>
        .main > div {padding-top: 1.2rem;}
        div[data-testid="stMetric"] {
            background: rgba(120, 120, 120, 0.08);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .app-subtitle {color: #7a7a7a; font-size: 0.95rem;}
        .footer-note {color: #9a9a9a; font-size: 0.8rem; text-align:center; margin-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True, #Allow CSS and HTML in this
)



MODEL_PATH = "Models/StudentMarksPredictor.pkl"
SCALER_PATH = "Models/ModelsScaler.pkl"
ENCODERS_PATH = "Models/LabelEncoders.pkl"

CLEANED_DATASET_PATH = "Dataset/CleanedDataset.csv"
RAW_DATASET_PATH = "Dataset/StudentPerformanceFactors.csv"

FEATURE_IMPORTANCE_PATH = "Reports/Feature_Importance.csv"
PREDICTION_REPORT_PATH = "Reports/Prediction_Report.csv"

ACTUAL_VS_PREDICTED_IMG = "Images/Actual_vs_Predicted.png"
RESIDUAL_PLOT_IMG = "Images/Residual_Plot.png"

# Column order MUST match the order used when the model/scaler were fit
# (i.e. df.drop("Exam_Score", axis=1) from FinalModel.py).
FEATURE_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

NUMERIC_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

CATEGORICAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]
# Above line only gives us Categorical Columns

# Sensible fallback ranges (used only if the raw dataset isn't available,
# and to flag "extrapolation" for the confidence estimate).
NUMERIC_DEFAULT_RANGES = {
    "Hours_Studied": (0, 50),
    "Attendance": (60, 100),
    "Sleep_Hours": (4, 10),
    "Previous_Scores": (50, 100),
    "Tutoring_Sessions": (0, 20),
    "Physical_Activity": (0, 6),
} # It decides the min and maximum range for the input


# @st.cache_resource , @st.cache_data are used for fast response -  Data/Resource is already avialabe in the cache --
# No need to load everytime

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load the trained model, scaler and label encoders from disk."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, encoders


@st.cache_data(show_spinner=False)
def load_cleaned_dataset():
    if not os.path.exists(CLEANED_DATASET_PATH):
        return None
    return pd.read_csv(CLEANED_DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_raw_dataset():
    if not os.path.exists(RAW_DATASET_PATH):
        return None
    return pd.read_csv(RAW_DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_feature_importance():
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        return None
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data(show_spinner=False)
def load_prediction_report():
    if not os.path.exists(PREDICTION_REPORT_PATH):
        return None
    return pd.read_csv(PREDICTION_REPORT_PATH)

# Python Logic
def calculate_grade(score: float) -> str:
    if score >= 95:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C+"
    elif score >= 50:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def calculate_risk_level(score: float, attendance: float, hours_studied: float):
    """Simple rule-based risk classification."""
    if score < 50 or attendance < 65:
        return "High Risk", "🔴"
    elif score < 65 or hours_studied < 8:
        return "Moderate Risk", "🟠"
    else:
        return "Low Risk", "🟢"


def generate_recommendations(inputs: dict, predicted_score: float) -> list:
    """Rule-based expert system producing personalized study tips."""
    tips = [] # All tips based on condition are appened here

    if inputs["Hours_Studied"] < 10:
        tips.append("📖 Increase weekly study hours — you're currently below the level associated with higher scores.")
    if inputs["Attendance"] < 75:
        tips.append("🏫 Improve class attendance; attendance shows a strong link to exam performance in this dataset.")
    if inputs["Sleep_Hours"] < 6:
        tips.append("😴 Aim for at least 7 hours of sleep — rest supports memory consolidation and focus.")
    if inputs["Tutoring_Sessions"] == 0:
        tips.append("👩‍🏫 Consider a few tutoring sessions, especially in topics you find difficult.")
    if inputs["Physical_Activity"] < 1:
        tips.append("🏃 Add light physical activity to your routine — it's linked to better concentration.")
    if inputs["Motivation_Level"] == "Low":
        tips.append("🎯 Set small, achievable weekly goals to help rebuild motivation.")
    if inputs["Peer_Influence"] == "Negative":
        tips.append("🤝 Try to study with peers who have positive academic habits.")
    if inputs["Access_to_Resources"] == "Low":
        tips.append("📚 Seek free/library resources or online material to compensate for limited access.")

    if predicted_score >= 85 and not tips:
        tips.append("🌟 Great habits! Keep maintaining your current study routine and consistency.")
    if not tips:
        tips.append("✅ Your current habits look balanced — keep it consistent and monitor progress regularly.")

    return tips


def make_prediction(inputs: dict, model, scaler, encoders):
    """Encode + scale a single student's inputs and return the prediction."""
    row = {col: inputs[col] for col in FEATURE_COLUMNS}
    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # Categorical encoding using the SAME encoders used at training time.
    for col in CATEGORICAL_COLUMNS:
        encoder = encoders.get(col)
        if encoder is None:
            raise ValueError(f"No saved LabelEncoder found for column '{col}'.")
        value = input_df.at[0, col]
        if value not in list(encoder.classes_):
            raise ValueError(
                f"Value '{value}' for '{col}' was not seen during training. "
                f"Expected one of: {list(encoder.classes_)}"
            )
        input_df[col] = encoder.transform(input_df[col])

    # Feature scaling — same StandardScaler used at training time.
    scaled = scaler.transform(input_df[FEATURE_COLUMNS])

    prediction = model.predict(scaled)[0]
    prediction = float(np.clip(prediction, 0, 100))
    return prediction


def score_gauge(score: float):
    """Circular gauge chart -> Plotly."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score, 1),
            number={"suffix": " / 100"},
            title={"text": "Predicted Exam Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4C78A8"},
                "steps": [
                    {"range": [0, 40], "color": "#f8d7da"},
                    {"range": [40, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#d1ecf1"},
                    {"range": [80, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
    return fig



def home_page():
    st.title("🎓 Student Exam Score Predictor")
    st.markdown(
        '<p class="app-subtitle">Predict exam performance, understand risk factors, '
        "and explore the data behind a trained Linear Regression model.</p>",
        unsafe_allow_html=True,
    )

    now = datetime.now()
    st.caption(f"📅 {now.strftime('%A, %d %B %Y')} · 🕒 {now.strftime('%I:%M %p')}")

    st.divider()

    raw_df = load_raw_dataset()
    cleaned_df = load_cleaned_dataset()
    report_df = load_prediction_report()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Students in Dataset", f"{len(raw_df):,}" if raw_df is not None else "N/A")
    with col2:
        st.metric("Features Used", len(FEATURE_COLUMNS))
    with col3:
        if report_df is not None and "Difference" in report_df.columns:
            mae = report_df["Difference"].abs().mean()
            st.metric("Avg. Error (MAE)", f"{mae:.2f} pts")
        else:
            st.metric("Avg. Error (MAE)", "N/A")
    with col4:
        st.metric("Model Type", "Linear Regression")

    st.divider()

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("What this app does")
        st.markdown(
            """
            - Loads an **already-trained** model, scaler, and encoders (no retraining here).
            - Lets you enter a student's profile and predicts their **exam score**.
            - Shows a **grade**, **risk level**, and **personalized recommendations**.
            - Explores the training data with interactive charts.
            - Lets you download a **CSV report** of predictions.

            Use the sidebar to navigate between **Prediction**, **Analytics**,
            **Reports**, and **About** pages.
            """
        )
    with right:
        st.subheader("Quick Start")
        st.info(
            "Go to **🎯 Predict Score** in the sidebar, fill in the student's "
            "details, and click **Predict** to see the result."
        )
        if raw_df is None:
            st.warning(
                f"Raw dataset not found at `{RAW_DATASET_PATH}`. "
                "Some charts on the Analytics page may be limited."
            )


def prediction_page(model, scaler, encoders, raw_df):
    st.title("🎯 Student Score Prediction")
    st.caption("Fill in the student's profile below and click **Predict**.")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            hours_studied = st.number_input("Hours Studied / week", min_value=0, max_value=60, value=20)
            attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=90)
            sleep_hours = st.number_input("Sleep Hours / night", min_value=0, max_value=14, value=7)
            previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=80)
            tutoring_sessions = st.number_input("Tutoring Sessions / month", min_value=0, max_value=15, value=3)
            physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0, max_value=15, value=3)

        with c2:
            parental_involvement = st.selectbox("Parental Involvement", list(encoders["Parental_Involvement"].classes_))
            access_to_resources = st.selectbox("Access to Resources", list(encoders["Access_to_Resources"].classes_))
            extracurricular = st.selectbox("Extracurricular Activities", list(encoders["Extracurricular_Activities"].classes_))
            motivation_level = st.selectbox("Motivation Level", list(encoders["Motivation_Level"].classes_))
            internet_access = st.selectbox("Internet Access", list(encoders["Internet_Access"].classes_))
            family_income = st.selectbox("Family Income", list(encoders["Family_Income"].classes_))

        with c3:
            teacher_quality = st.selectbox("Teacher Quality", list(encoders["Teacher_Quality"].classes_))
            school_type = st.selectbox("School Type", list(encoders["School_Type"].classes_))
            peer_influence = st.selectbox("Peer Influence", list(encoders["Peer_Influence"].classes_))
            learning_disabilities = st.selectbox("Learning Disabilities", list(encoders["Learning_Disabilities"].classes_))
            parental_education = st.selectbox("Parental Education Level", list(encoders["Parental_Education_Level"].classes_))
            distance_from_home = st.selectbox("Distance from Home", list(encoders["Distance_from_Home"].classes_))
            gender = st.selectbox("Gender", list(encoders["Gender"].classes_))

        submitted = st.form_submit_button("🔮 Predict", width='stretch')

    if not submitted:
        return

    # ---- Input Validation -> Python Validation -----------------------
    errors = []
    if attendance > 100 or attendance < 0:
        errors.append("Attendance must be between 0 and 100.")
    if hours_studied < 0:
        errors.append("Hours studied cannot be negative.")
    if previous_scores < 0 or previous_scores > 100:
        errors.append("Previous scores must be between 0 and 100.")

    if errors:
        for e in errors:
            st.error(e)
        return

    inputs = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_from_home,
        "Gender": gender,
    }

    # ---- Prediction (with spinner + error handling) -------------------
    try:
        with st.spinner("Running the model..."):
            predicted_score = make_prediction(inputs, model, scaler, encoders)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    grade = calculate_grade(predicted_score)
    risk_label, risk_icon = calculate_risk_level(predicted_score, attendance, hours_studied)
    recommendations = generate_recommendations(inputs, predicted_score)

    st.session_state["history"].append(
        {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **inputs,
            "Predicted_Score": round(predicted_score, 2),
            "Grade": grade,
            "Risk_Level": risk_label,
        }
    )

    st.divider()
    st.subheader("📊 Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Exam Score", f"{predicted_score:.2f}")
    m2.metric("Grade", grade)
    m3.metric("Risk Level", f"{risk_icon} {risk_label}")

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(score_gauge(predicted_score), width='stretch')
    with g2:
        st.markdown("**Score Progress**")
        st.progress(int(predicted_score))
        st.caption(f"{predicted_score:.1f} / 100 points")
        st.markdown("**💡 Personalized Recommendations**")
        for tip in recommendations:
            st.write(f"- {tip}")


def analytics_page():
    st.title("📈 Interactive Analytics Dashboard")

    raw_df = load_raw_dataset()
    cleaned_df = load_cleaned_dataset()

    if cleaned_df is None and raw_df is None:
        st.error(
            "No dataset found. Expected "
            f"`{RAW_DATASET_PATH}` or `{CLEANED_DATASET_PATH}`."
        )
        return

    chart_df = raw_df if raw_df is not None else cleaned_df

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔥 Correlation", "🧮 Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if "Exam_Score" in chart_df.columns:
                fig = px.histogram(chart_df, x="Exam_Score", nbins=25, title="Exam Score Distribution")
                st.plotly_chart(fig, width='stretch')
        with c2:
            if {"Hours_Studied", "Exam_Score"}.issubset(chart_df.columns):
                fig = px.scatter(
                    chart_df, x="Hours_Studied", y="Exam_Score",
                    trendline="ols" if HAS_STATSMODELS else None,
                    title="Hours Studied vs Exam Score",
                )
                st.plotly_chart(fig, width='stretch')

        if {"Attendance", "Exam_Score"}.issubset(chart_df.columns):
            fig = px.scatter(
                chart_df, x="Attendance", y="Exam_Score",
                trendline="ols" if HAS_STATSMODELS else None,
                title="Attendance vs Exam Score",
            )
            st.plotly_chart(fig, width='stretch')

        if not HAS_STATSMODELS:
            st.caption(
                "ℹ️ Trendlines are hidden because `statsmodels` isn't installed. "
                "Run `pip install statsmodels` to enable OLS trendlines on the scatter plots."
            )

    with tab2:
        if cleaned_df is not None:
            st.markdown("Correlation heatmap uses the fully-encoded dataset so categorical columns are included.")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cleaned_df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
            st.pyplot(fig)
        else:
            st.info(f"Cleaned dataset not found at `{CLEANED_DATASET_PATH}`.")

    with tab3:
        importance_df = load_feature_importance()
        if importance_df is not None:
            importance_df = importance_df.sort_values("Coefficient")
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ["#d9534f" if v < 0 else "#5cb85c" for v in importance_df["Coefficient"]]
            ax.barh(importance_df["Feature"], importance_df["Coefficient"], color=colors)
            ax.set_xlabel("Coefficient (impact on Exam Score)")
            ax.set_title("Feature Importance (Linear Regression Coefficients)")
            st.pyplot(fig)
        else:
            st.info(f"Feature importance file not found at `{FEATURE_IMPORTANCE_PATH}`.")

    st.divider()
    st.subheader("🖼 Model Diagnostic Plots")
    i1, i2 = st.columns(2)
    with i1:
        if os.path.exists(ACTUAL_VS_PREDICTED_IMG):
            st.image(ACTUAL_VS_PREDICTED_IMG, caption="Actual vs Predicted Exam Score", width='stretch')
        else:
            st.info(f"Image not found at `{ACTUAL_VS_PREDICTED_IMG}`.")
    with i2:
        if os.path.exists(RESIDUAL_PLOT_IMG):
            st.image(RESIDUAL_PLOT_IMG, caption="Residual Plot", width='stretch')
        else:
            st.info(f"Image not found at `{RESIDUAL_PLOT_IMG}`.")


def reports_page():
    st.title("📄 Prediction Reports")

    tab1, tab2 = st.tabs(["🧪 Test-Set Prediction Report", "📝 Your Session Predictions"])

    with tab1:
        report_df = load_prediction_report()
        if report_df is None:
            st.info(f"Report not found at `{PREDICTION_REPORT_PATH}`.")
        else:
            s1, s2, s3 = st.columns(3)
            s1.metric("Rows", len(report_df))
            if "Difference" in report_df.columns:
                s2.metric("Mean Abs. Error", f"{report_df['Difference'].abs().mean():.2f}")
                s3.metric("Max Abs. Error", f"{report_df['Difference'].abs().max():.2f}")

            search = st.text_input("🔍 Search / filter rows (e.g. type a score)", key="search_report")
            filtered = report_df
            if search:
                mask = report_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False))
                filtered = report_df[mask.any(axis=1)]
            st.dataframe(filtered, width='stretch')

            st.download_button(
                "📥 Download Test-Set Report (CSV)",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="prediction_report.csv",
                mime="text/csv",
            )

    with tab2:
        history = st.session_state.get("history", [])
        if not history:
            st.info("No predictions made yet in this session. Go to **🎯 Predict Score** to generate one.")
        else:
            hist_df = pd.DataFrame(history)
            search2 = st.text_input("🔍 Search / filter your predictions", key="search_history")
            filtered2 = hist_df
            if search2:
                mask = hist_df.astype(str).apply(lambda col: col.str.contains(search2, case=False, na=False))
                filtered2 = hist_df[mask.any(axis=1)]
            st.dataframe(filtered2, width='stretch')

            st.download_button(
                "📥 Download Your Predictions (CSV)",
                data=filtered2.to_csv(index=False).encode("utf-8"),
                file_name="my_predictions.csv",
                mime="text/csv",
            )


def about_model_page():
    st.title("🤖 About the Model")
    st.markdown(
        """
        ### Model
        A **Linear Regression** model trained with scikit-learn on the
        *Student Performance Factors* dataset.

        ### Pipeline
        1. Missing values in `Teacher_Quality`, `Parental_Education_Level`,
           and `Distance_from_Home` are filled with the column mode.
        2. All categorical (text) columns are converted to numbers with
           `LabelEncoder`, and each encoder is saved so the app can encode
           new inputs consistently (`Models/LabelEncoders.pkl`).
        3. Features are standardized with `StandardScaler`
           (`Models/ModelsScaler.pkl`) before training/prediction.
        4. A `LinearRegression` model is fit on the scaled training split
           and saved to `Models/StudentMarksPredictor.pkl`.

        ### Comparison
        During development, Linear Regression, Decision Tree, and Random
        Forest regressors were all trained and evaluated with MAE, MSE,
        RMSE, and R². Linear Regression was selected for this app because
        of its strong interpretability (coefficients double as a feature
        importance chart) and competitive accuracy.

        ### Inputs (19 features)
        Study habits (hours studied, attendance, sleep, tutoring sessions,
        physical activity), prior performance, and contextual/demographic
        factors (parental involvement & education, access to resources,
        motivation, peer influence, school type, distance from home,
        family income, teacher quality, internet access, learning
        disabilities, gender, extracurricular activities).
        """
    )


def about_developer_page():
    st.title("👩‍💻 About the Backend Tech")
    with st.container(border=True):
        st.subheader("Project: Student Exam Score Predictor")
        st.write(
            "An end-to-end machine learning project — from data cleaning "
            "and model training to an interactive Streamlit dashboard."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Stage", "Data → Model → App")
        c2.metric("Framework", "Streamlit")
        c3.metric("ML Library", "scikit-learn")

    with st.container(border=True):
        st.markdown(
            """
            **Tech stack:** Python, pandas, scikit-learn, joblib, Streamlit,
            Plotly, Matplotlib, Seaborn.

            **Files:**
            - `ReadDataSet.py` — cleaning + label encoding
            - `FinalModel.py` — training, evaluation, artifact saving
            - `App.py` — this dashboard
            """
        )

def main():
    st.sidebar.title("🎓 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "🎯 Predict Score",
            "📈 Analytics",
            "📄 Reports",
            "🤖 About Model",
            "👩‍💻 About Developer",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(f"🕒 {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

    # Load model artifacts once (cached) with error handling.
    try:
        with st.spinner("Loading model, scaler, and encoders..."):
            model, scaler, encoders = load_model_artifacts()
    except FileNotFoundError as exc:
        st.error(
            "Could not load model artifacts. Make sure you've run "
            "`ReadDataSet.py` and `FinalModel.py` first so the following "
            f"files exist: `{MODEL_PATH}`, `{SCALER_PATH}`, `{ENCODERS_PATH}`.\n\n"
            f"Details: {exc}"
        )
        model = scaler = encoders = None
    except Exception as exc:
        st.error(f"Unexpected error while loading model artifacts: {exc}")
        model = scaler = encoders = None

    raw_df = load_raw_dataset()

    if page == "🏠 Home":
        home_page()
    elif page == "🎯 Predict Score":
        if model is None:
            st.warning("Prediction is unavailable until model artifacts load successfully.")
        else:
            prediction_page(model, scaler, encoders, raw_df)
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "📄 Reports":
        reports_page()
    elif page == "🤖 About Model":
        about_model_page()
    elif page == "👩‍💻 About Developer":
        about_developer_page()

    st.markdown(
        '<p class="footer-note">Built with Streamlit · scikit-learn · Plotly · Matplotlib · Seaborn</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
