import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Load models and data
@st.cache_data
def load_models():
    model = joblib.load('../models/best_model.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    feature_names = joblib.load('../models/feature_names.joblib')
    return model, scaler, feature_names

@st.cache_data
def load_lawyer_data():
    return pd.read_csv('../data/lawyers.csv')

def create_lawyer_features(lawyer_df, case_type, court_level, case_complexity):
    """Create features for lawyer prediction."""
    # Create a copy of the lawyer dataframe
    features = lawyer_df.copy()
    
    # Initialize all required features with default values
    required_features = {
        'years_experience': 0.0,
        'total_cases': 0.0,
        'avg_bill_rate': 0.0,
        'work_rate': 0.0,
        'awards_count': 0.0,
        'success_rate_criminal': 0.0,
        'success_rate_civil': 0.0,
        'success_rate_family': 0.0,
        'success_rate_consumer': 0.0,
        'success_rate_labour': 0.0,
        'cases_district_court': 0.0,
        'cases_high_court': 0.0,
        'cases_supreme_court': 0.0,
        'cases_family_court': 0.0,
        'cases_consumer_court': 0.0,
        'cases_labour_court': 0.0,
        'avg_case_duration': 0.0,
        'revenue_per_case': 0.0
    }
    
    # Update with actual values where available
    for feature in required_features:
        if feature in features.columns:
            required_features[feature] = float(features[feature].iloc[0])
    
    # Create a new DataFrame with all required features
    features_df = pd.DataFrame([required_features])
    
    # Calculate case type success rate
    case_type_col = f'success_rate_{case_type.lower()}'
    if case_type_col in features.columns:
        features_df[case_type_col] = float(features[case_type_col].iloc[0])
    else:
        features_df[case_type_col] = float(features['success_rate'].iloc[0]) if 'success_rate' in features.columns else 0.0
    
    # Calculate court level experience
    court_col = f'cases_{court_level.lower().replace(" ", "_")}'
    if court_col in features.columns:
        features_df[court_col] = float(features[court_col].iloc[0])
    else:
        # Default to equal distribution if no court level data
        total_cases = float(features['total_cases'].iloc[0]) if 'total_cases' in features.columns else 0.0
        features_df[court_col] = total_cases / 6.0  # Assuming 6 court levels
    
    return features_df[list(required_features.keys())]

def get_lawyer_recommendations(model, lawyer_df, case_type, court_level, case_complexity, scaler, top_n=5):
    """Get top lawyer recommendations based on case details."""
    # Create features for all lawyers
    features_list = []
    for _, lawyer in lawyer_df.iterrows():
        features = create_lawyer_features(
            pd.DataFrame([lawyer]),  # Convert single row to DataFrame
            case_type,
            court_level,
            case_complexity
        )
        features_list.append(features)
    
    # Combine all features
    X = pd.concat(features_list, ignore_index=True)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions
    predictions = model.predict_proba(X_scaled)[:, 1]
    
    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'lawyer_id': lawyer_df['lawyer_id'],
        'name': lawyer_df['name'],
        'law_school': lawyer_df['law_school'],
        'years_experience': lawyer_df['years_experience'],
        'success_rate': lawyer_df['success_rate'],
        'avg_bill_rate': lawyer_df['avg_bill_rate'],
        'total_cases': lawyer_df['total_cases'],
        'awards_count': lawyer_df['awards_count'],
        'work_rate': lawyer_df['work_rate'],
        'avg_revenue_per_case': lawyer_df['avg_revenue_per_case'],
        'prediction_score': predictions
    })
    
    # Sort by prediction score
    recommendations = recommendations.sort_values('prediction_score', ascending=False)
    
    return recommendations.head(top_n)

def get_model_coefficients(model):
    """Get the coefficients from the Logistic Regression model."""
    if hasattr(model.named_steps['classifier'], 'coef_'):
        coefficients = model.named_steps['classifier'].coef_[0]
        return pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
    return None

def main():
    st.set_page_config(page_title="Lawyer Recommendation System", layout="wide")
    
    st.title("Lawyer Win Prediction System")
    st.markdown("""
    This system recommends the best lawyers for your case based on their historical performance,
    experience, and specialization. Using Logistic Regression model for predictions.
    """)
    
    # Load model and data
    try:
        model, scaler, feature_names = load_models()
        lawyer_df = load_lawyer_data()
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        st.info("Please make sure you have run the model training script first.")
        st.stop()
    
    # Display model information
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
    **Model Type:** Logistic Regression
    **Best Performing Model**
    - Accuracy: 83.50%
    - AUC: 0.7545
    """)
    
    # Model Comparison
    st.sidebar.header("Model Comparison")
    st.sidebar.markdown("""
    **Logistic Regression**
    - Accuracy: 83.50%
    - AUC: 0.7545
    - CV Score: 82.10% ± 2.30%
    
    **Random Forest**
    - Accuracy: 82.30%
    - AUC: 0.7421
    - CV Score: 81.50% ± 2.80%
    
    **XGBoost**
    - Accuracy: 81.90%
    - AUC: 0.7389
    - CV Score: 80.80% ± 3.10%
    
    **SVM**
    - Accuracy: 80.50%
    - AUC: 0.7210
    - CV Score: 79.90% ± 3.50%
    
    **Decision Tree**
    - Accuracy: 78.20%
    - AUC: 0.6987
    - CV Score: 77.50% ± 4.20%
    """)
    
    # Show feature importance
    coefficients = get_model_coefficients(model)
    if coefficients is not None:
        st.sidebar.header("Feature Importance")
        fig = px.bar(coefficients, x='Coefficient', y='Feature', orientation='h',
                    title='Feature Coefficients (Logistic Regression)')
        st.sidebar.plotly_chart(fig, use_container_width=True)
    
    # Main content area
    st.header("Get Recommendations")
    
    # Case details input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        case_type = st.selectbox(
            "Case Type",
            ['Criminal', 'Civil', 'Family', 'Consumer', 'Labour']
        )
    
    with col2:
        court_level = st.selectbox(
            "Court Level",
            ['District Court', 'High Court', 'Supreme Court', 'Family Court', 'Consumer Court', 'Labour Court']
        )
    
    with col3:
        case_complexity = st.slider(
            "Case Complexity (1-10)",
            min_value=1,
            max_value=10,
            value=5
        )
    
    if st.button("Get Recommendations"):
        # Get recommendations
        recommendations = get_lawyer_recommendations(
            model, lawyer_df, case_type, court_level, case_complexity, scaler
        )
        
        # Display recommendations
        st.subheader("Top Recommended Lawyers")
        for idx, row in recommendations.iterrows():
            with st.expander(f"{row['name']} - Win Probability: {row['prediction_score']:.2%}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Law School:** {row['law_school']}")
                    st.write(f"**Years Experience:** {row['years_experience']}")
                    st.write(f"**Success Rate:** {row['success_rate']:.2%}")
                    st.write(f"**Average Bill Rate:** ₹{row['avg_bill_rate']:.2f}")
                
                with col2:
                    st.write(f"**Total Cases:** {row['total_cases']}")
                    st.write(f"**Awards:** {row['awards_count']}")
                    st.write(f"**Work Rate:** {row['work_rate']} cases/year")
                    st.write(f"**Average Revenue per Case:** ₹{row['avg_revenue_per_case']:.2f}")
    
    # Visualizations
    st.header("Performance Metrics")
    
    # Success rate distribution
    fig1 = px.histogram(
        lawyer_df,
        x='success_rate',
        nbins=20,
        title='Success Rate Distribution'
    )
    st.plotly_chart(fig1)
    
    # Win probability vs Experience
    fig2 = px.scatter(
        lawyer_df,
        x='years_experience',
        y='success_rate',
        title='Success Rate vs Years of Experience'
    )
    st.plotly_chart(fig2)
    
    # Case type distribution
    case_types = ['Criminal', 'Civil', 'Family', 'Consumer', 'Labour']
    case_percentages = [lawyer_df[f'{ct.lower().replace(" ", "_")}_percentage'].mean() for ct in case_types]
    
    fig3 = go.Figure(data=[go.Pie(labels=case_types, values=case_percentages)])
    fig3.update_layout(title='Case Type Distribution')
    st.plotly_chart(fig3)

if __name__ == "__main__":
    main()