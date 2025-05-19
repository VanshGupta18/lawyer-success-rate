# Streamlit app: Legal Case Outcome & Top Lawyer Predictor (with card UI)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from time import time
from plotly.subplots import make_subplots


# App Configuration
st.set_page_config(page_title="Lawyer Predictor", layout="wide")
st.title("🏛️ Legal Case Outcome & Top Lawyer Predictor")

# Define color palettes
COLORS = {
    'primary': ['#2E86C1', '#2874A6', '#1F618D'],  # Professional blues
    'secondary': ['#27AE60', '#219653', '#1E8449'],  # Professional greens
    'accent': ['#E67E22', '#D35400', '#A04000'],  # Warm oranges
    'neutral': ['#95A5A6', '#7F8C8D', '#616A6B'],  # Professional grays
    'gradient': ['#2E86C1', '#27AE60', '#E67E22'],  # Multi-color gradient
    'heatmap': ['#F8F9F9', '#D6EAF8', '#2E86C1', '#1A5276'],  # Blue heatmap
    'heatmap_green': ['#F8F9F9', '#D5F5E3', '#27AE60', '#196F3D'],  # Green heatmap
}

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
    st.markdown("""
    <style>
    .lawyer-grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 20px;
        margin-top: 10px;
    }
    .lawyer-card {
        flex: 0 0 48%;
        background: linear-gradient(to bottom right, #e3f2fd, #ffffff);
        border: 1px solid #bbdefb;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', sans-serif;
        color: #0d1b2a;
        margin-bottom: 20px;
    }
    .lawyer-card h4 {
        font-size: 1.1em;
        margin-bottom: 8px;
        color: #1a237e;
    }
    .lawyer-card p {
        margin: 4px 0;
        font-size: 0.9em;
        line-height: 1.4;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background-color: #d1e7dd;
        overflow: hidden;
        margin-top: 4px;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(to right, #2d6a4f, #40916c);
    }
    </style>
    <div class="lawyer-grid">
    """, unsafe_allow_html=True)

    card_html = ""
    for _, row in results.iterrows():
        card_html += f"""
        <div class="lawyer-card">
            <h4>👤 {row['name']}</h4>
            <p><strong>🎓 Law School:</strong> {row['law_school']} ({row['school_tier']})</p>
            <p><strong> 📅 Started Practice:</strong> {pd.to_datetime(row['start_practice']).date()}</p>
            <p><strong>🧠 Experience:</strong> {row['years_experience']} years</p>
            <p><strong>🏆 Win Confidence:</strong> {row['avg_confidence']*100:.2f}%</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {row['avg_confidence']*100:.2f}%"></div>
            </div>
        </div>
        """

    card_html += "</div>"
    st.markdown(card_html, unsafe_allow_html=True)

# Get evaluation metrics
def get_model_metrics(model, X, y):
    start_time = time()
    y_pred = model.predict(X)
    prediction_time = time() - start_time
    
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    matrix = confusion_matrix(y, y_pred)
    return accuracy, report, matrix, prediction_time

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'named_steps'):
        if 'classifier' in model.named_steps and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            
            if hasattr(preprocessor, 'get_feature_names_out'):
                processed_feature_names = preprocessor.get_feature_names_out()
            else:
                processed_feature_names = preprocessor.get_feature_names()
            
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importances = model.named_steps['classifier'].feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': processed_feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale=COLORS['heatmap']
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=False,
                    template='plotly_white',
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title=dict(
                        font=dict(size=20, color='#2C3E50')
                    )
                )
                
                return fig
    return None

def plot_classification_report(report, model_name):
    # Convert report to DataFrame
    report_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Skip 'accuracy' and other non-class metrics
            metrics['class'] = label
            report_data.append(metrics)
    
    df_report = pd.DataFrame(report_data)
    
    # Create subplots for each metric
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Precision', 'Recall', 'F1-Score'),
        shared_yaxes=True
    )
    
    # Add bars for each metric
    metrics = ['precision', 'recall', 'f1-score']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                y=df_report['class'],
                x=df_report[metric],
                name=metric.capitalize(),
                orientation='h',
                marker_color=COLORS['gradient'][i],
                marker=dict(
                    line=dict(width=1, color='rgba(0,0,0,0.1)')
                )
            ),
            row=1, col=i+1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Classification Report - {model_name}',
            font=dict(size=20, color='#2C3E50')
        ),
        height=400,
        showlegend=False,
        template='plotly_white',
        margin=dict(t=100),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update axes
    for i in range(3):
        fig.update_xaxes(
            title_text='Score',
            range=[0, 1],
            row=1,
            col=i+1,
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.1)'
        )
        if i == 0:
            fig.update_yaxes(
                title_text='Class',
                row=1,
                col=i+1,
                gridcolor='rgba(0,0,0,0.1)'
            )
    
    return fig

# Load data and models
df = load_data()
dt_model, rf_model = load_models()

# Feature and Target Columns
features = ['case_type', 'court_level', 'duration_days', 'school_tier',
            'lawyer_experience_at_case', 'case_month', 'case_year', 'case_day_of_week']
target = 'result'
X = df[features]
y = df[target]

# Get metrics and timing for both models
dt_acc, dt_report, dt_cm, dt_time = get_model_metrics(dt_model, X, y)
rf_acc, rf_report, rf_cm, rf_time = get_model_metrics(rf_model, X, y)

# Automatically select best model based on accuracy
best_model = None
if dt_acc >= rf_acc:
    best_model = dt_model
    best_model_name = "Decision Tree"
else:
    best_model = rf_model
    best_model_name = "Random Forest"

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
        top_n = st.slider("Number of Top Lawyers to Show", 1, 15, 5)

    st.markdown(f"**Automatically Selected Best Model:** {best_model_name}")

    if st.button("Predict Top Lawyers"):
        results = get_top_lawyers(df, best_model, case_type, court_level, duration_days, top_n)
        if results.empty:
            st.warning("⚠️ No similar cases found. Try changing inputs.")
        else:
            st.success("✅ Top Lawyers Based on Prediction Confidence")
            render_lawyer_cards(results)

# --- Tab 2: Metrics ---
with tab2:
    st.header("📈 Evaluation Metrics for Models")

    # Model Performance Comparison
    performance_data = {
        'Model': ['Decision Tree', 'Random Forest'],
        'Accuracy': [dt_acc, rf_acc],
        'Prediction Time (s)': [dt_time, rf_time]
    }
    performance_df = pd.DataFrame(performance_data)
    
    # Plot performance comparison
    fig_perf = px.bar(
        performance_df,
        x='Model',
        y=['Accuracy', 'Prediction Time (s)'],
        barmode='group',
        title='Model Performance Comparison',
        labels={'value': 'Score', 'variable': 'Metric'},
        color_discrete_sequence=COLORS['gradient']
    )
    fig_perf.update_layout(
        template='plotly_white',
        height=400,
        showlegend=True,
        legend_title='Metric',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            font=dict(size=20, color='#2C3E50')
        )
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # Feature Importance for Random Forest
    st.subheader("Feature Importance (Random Forest)")
    fi_fig = plot_feature_importance(rf_model, features)
    if fi_fig:
        st.plotly_chart(fi_fig, use_container_width=True)

    # Model-specific metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Tree Metrics")
        st.metric("Accuracy", f"{dt_acc:.2f}")
        st.metric("Prediction Time", f"{dt_time:.4f}s")
        
        # Plot classification report
        st.text("Classification Report")
        dt_report_fig = plot_classification_report(dt_report, "Decision Tree")
        st.plotly_chart(dt_report_fig, use_container_width=True)
        
        st.text("Confusion Matrix")
        fig1 = px.imshow(
            dt_cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=np.unique(y),
            y=np.unique(y),
            color_continuous_scale=COLORS['heatmap'],
            title="Decision Tree Confusion Matrix"
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                font=dict(size=20, color='#2C3E50')
            )
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Random Forest Metrics")
        st.metric("Accuracy", f"{rf_acc:.2f}")
        st.metric("Prediction Time", f"{rf_time:.4f}s")
        
        # Plot classification report
        st.text("Classification Report")
        rf_report_fig = plot_classification_report(rf_report, "Random Forest")
        st.plotly_chart(rf_report_fig, use_container_width=True)
        
        st.text("Confusion Matrix")
        fig2 = px.imshow(
            rf_cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=np.unique(y),
            y=np.unique(y),
            color_continuous_scale=COLORS['heatmap_green'],
            title="Random Forest Confusion Matrix"
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                font=dict(size=20, color='#2C3E50')
            )
        )
        st.plotly_chart(fig2, use_container_width=True)