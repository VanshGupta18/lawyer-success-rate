# Streamlit app: Legal Case Outcome & Top Lawyer Predictor (with card UI)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained pipeline models
@st.cache_resource
def load_models():
    dt_model = joblib.load("models/decision_tree_model.joblib")
    rf_model = joblib.load("models/random_forest_model.joblib")
    return dt_model, rf_model

# Load and preprocess merged case-lawyer dataset
@st.cache_data
def load_data():
    lawyers_df = pd.read_csv('data/lawyers.csv')
    cases_df = pd.read_csv('data/cases.csv')
    df = pd.merge(cases_df, lawyers_df, on='lawyer_id')

    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['start_practice'] = pd.to_datetime(df['start_practice'])
    df['case_month'] = df['start_date'].dt.month
    df['case_year'] = df['start_date'].dt.year
    df['case_day_of_week'] = df['start_date'].dt.dayofweek
    df['lawyer_experience_at_case'] = (df['start_date'] - df['start_practice']).dt.days / 365.25

    return df

# Top Lawyer Prediction Function
def get_top_lawyers(df, model, case_type, court_level, duration_days, top_n=10):
    features = ['case_type', 'court_level', 'duration_days', 'school_tier',
                'lawyer_experience_at_case', 'case_month', 'case_year', 'case_day_of_week']

    similar_cases = df[
        (df['case_type'] == case_type) &
        (df['court_level'] == court_level) &
        (df['duration_days'].between(duration_days - 15, duration_days + 15))
    ].copy()

    if similar_cases.empty:
        return pd.DataFrame()

    X_similar = similar_cases[features]
    similar_cases['predicted_result'] = model.predict(X_similar)

    if hasattr(model, 'predict_proba'):
        win_index = list(model.classes_).index('Won')
        similar_cases['win_confidence'] = model.predict_proba(X_similar)[:, win_index]
    else:
        similar_cases['win_confidence'] = np.nan

    top_lawyers = (
        similar_cases.groupby('lawyer_id')
        .agg(avg_confidence=('win_confidence', 'mean'))
        .sort_values(by='avg_confidence', ascending=False)
        .head(top_n)
        .reset_index()
    )

    lawyer_columns = ['lawyer_id', 'name', 'law_school', 'school_tier', 'start_practice', 'years_experience']
    enriched = pd.merge(top_lawyers, df[lawyer_columns].drop_duplicates(), on='lawyer_id', how='left')

    return enriched[['lawyer_id', 'name', 'law_school', 'school_tier',
                     'start_practice', 'years_experience', 'avg_confidence']]

def render_lawyer_cards(results):
    st.markdown("### 👨‍⚖️ Top Lawyer Recommendations")
    for idx, row in results.iterrows():
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color: #ffffff;
                    padding: 16px;
                    margin-bottom: 10px;
                    border-radius: 12px;
                    border: 1px solid #e0e0e0;
                ">
                    <h4 style="margin-bottom:5px; color: #333;">👤 {row['name']}</h4>
                    <p style="margin:0; color: #555;"><strong>🎓 Law School:</strong> {row['law_school']} ({row['school_tier']})</p>
                    <p style="margin:0; color: #555;"><strong>📅 Started Practice:</strong> {pd.to_datetime(row['start_practice']).date()}</p>
                    <p style="margin:0; color: #555;"><strong>🧠 Years of Experience:</strong> {row['years_experience']}</p>
                    <p style="margin:0; color: #555;"><strong>🏆 Win Confidence:</strong> <span style="color:#1a7f37; font-weight:bold;">{row['avg_confidence']*100:.2f}%</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Get evaluation metrics
def get_model_metrics(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    matrix = confusion_matrix(y, y_pred)
    return accuracy, report, matrix

# App Configuration
st.set_page_config(page_title="Lawyer Predictor", layout="wide")
st.title("🏛️ Legal Case Outcome & Top Lawyer Predictor")

# Load data and models
df = load_data()
dt_model, rf_model = load_models()

# Feature and Target Columns
features = ['case_type', 'court_level', 'duration_days', 'school_tier',
            'lawyer_experience_at_case', 'case_month', 'case_year', 'case_day_of_week']
target = 'result'
X = df[features]
y = df[target]

# UI Tabs
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Model Metrics"])

# --- Tab 1: Prediction ---
with tab1:
    st.header("Find Top Lawyers Based on Case Details")

    col1, col2 = st.columns(2)
    with col1:
        case_type = st.selectbox("Select Case Type", df['case_type'].unique())
        court_level = st.selectbox("Select Court Level", df['court_level'].unique())

    with col2:
        duration_days = st.slider("Expected Case Duration (days)", 30, 365, 120)
        model_choice = st.radio("Choose Prediction Model", ("Decision Tree", "Random Forest"))
        top_n = st.slider("Number of Top Lawyers to Show", 1, 15, 5)

    selected_model = dt_model if model_choice == "Decision Tree" else rf_model

    if st.button("Predict Top Lawyers"):
        results = get_top_lawyers(df, selected_model, case_type, court_level, duration_days, top_n)
        if results.empty:
            st.warning("⚠️ No similar cases found. Try changing inputs.")
        else:
            st.success("✅ Top Lawyers Based on Prediction Confidence")
            render_lawyer_cards(results)

# --- Tab 2: Metrics ---
with tab2:
    st.header("📈 Evaluation Metrics for Models")

    dt_acc, dt_report, dt_cm = get_model_metrics(dt_model, X, y)
    rf_acc, rf_report, rf_cm = get_model_metrics(rf_model, X, y)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Decision Tree Metrics")
        st.metric("Accuracy", f"{dt_acc:.2f}")
        st.text("Classification Report")
        st.json(dt_report)
        st.text("Confusion Matrix")
        fig1, ax1 = plt.subplots()
        sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)

    with col2:
        st.subheader("Random Forest Metrics")
        st.metric("Accuracy", f"{rf_acc:.2f}")
        st.text("Classification Report")
        st.json(rf_report)
        st.text("Confusion Matrix")
        fig2, ax2 = plt.subplots()
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)