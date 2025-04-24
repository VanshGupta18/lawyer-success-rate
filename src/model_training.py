import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the generated lawyer and case data."""
    lawyer_df = pd.read_csv('data/lawyers.csv')
    case_df = pd.read_csv('data/cases.csv')
    return lawyer_df, case_df

def engineer_features(lawyer_df, case_df):
    """Engineer features for the prediction model."""
    # Calculate case type success rates for each lawyer
    case_type_success = case_df.groupby(['lawyer_id', 'case_type', 'result']).size().unstack(fill_value=0)
    case_type_success['total_cases'] = case_type_success.sum(axis=1)
    case_type_success['success_rate'] = (case_type_success['Won'] + case_type_success['Settled']) / case_type_success['total_cases']
    
    # Pivot to get success rates by case type
    case_type_rates = case_type_success['success_rate'].unstack(fill_value=0)
    case_type_rates.columns = [f'success_rate_{col.lower()}' for col in case_type_rates.columns]
    
    # Calculate court level experience
    court_level_exp = case_df.groupby(['lawyer_id', 'court_level']).size().unstack(fill_value=0)
    court_level_exp.columns = [f'cases_{col.lower().replace(" ", "_")}' for col in court_level_exp.columns]
    
    # Calculate average case duration and revenue
    case_stats = case_df.groupby('lawyer_id').agg({
        'duration_days': 'mean',
        'revenue': ['mean', 'sum']
    }).reset_index()
    case_stats.columns = ['lawyer_id', 'avg_case_duration', 'revenue_per_case', 'total_revenue']
    
    # Merge all features
    features = lawyer_df.merge(case_type_rates, on='lawyer_id', how='left')
    features = features.merge(court_level_exp, on='lawyer_id', how='left')
    features = features.merge(case_stats, on='lawyer_id', how='left')
    
    # Fill NaN values
    features = features.fillna(0)
    
    return features

def prepare_data(features):
    """Prepare data for model training."""
    # Define features and target
    X = features[[
        'years_experience',
        'total_cases',
        'avg_bill_rate',
        'work_rate',
        'awards_count',
        'success_rate_criminal',
        'success_rate_civil',
        'success_rate_family',
        'success_rate_consumer',
        'success_rate_labour',
        'cases_district_court',
        'cases_high_court',
        'cases_supreme_court',
        'cases_family_court',
        'cases_consumer_court',
        'cases_labour_court',
        'avg_case_duration',
        'revenue_per_case'
    ]]
    
    # Create target variable (overall success rate > 0.6)
    y = (features['success_rate'] > 0.6).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def create_model_pipelines():
    """Create pipelines for different models."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), list(range(18)))  # All 18 features are numeric
        ])
    
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ]),
        'SVM': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=42))
        ])
    }
    
    return models

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    models = create_model_pipelines()
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'report': report
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("Classification Report:")
        print(report)
    
    return results

def plot_model_comparison(results):
    """Plot comparison of model performance."""
    # Prepare data for plotting
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    auc_scores = [results[name]['auc'] for name in model_names]
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy and AUC
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, auc_scores, width, label='AUC')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/model_comparison.png')
    plt.close()
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
    plt.xlabel('Models')
    plt.ylabel('Cross-validation Score')
    plt.title('Cross-validation Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('models/cv_comparison.png')
    plt.close()

def save_best_model(results):
    """Save the best performing model."""
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.joblib')
    
    # Save the scaler
    scaler = best_model.named_steps['preprocessor'].named_transformers_['num']
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save feature names
    feature_names = [
        'years_experience',
        'total_cases',
        'avg_bill_rate',
        'work_rate',
        'awards_count',
        'success_rate_criminal',
        'success_rate_civil',
        'success_rate_family',
        'success_rate_consumer',
        'success_rate_labour',
        'cases_district_court',
        'cases_high_court',
        'cases_supreme_court',
        'cases_family_court',
        'cases_consumer_court',
        'cases_labour_court',
        'avg_case_duration',
        'revenue_per_case'
    ]
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    print(f"\nBest model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"AUC: {results[best_model_name]['auc']:.4f}")

def main():
    # Load data
    print("Loading data...")
    lawyer_df, case_df = load_data()
    
    # Engineer features
    print("Engineering features...")
    features = engineer_features(lawyer_df, case_df)
    
    # Prepare data
    print("Preparing data for training...")
    X_train, X_test, y_train, y_test = prepare_data(features)
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot comparison
    print("Generating comparison plots...")
    plot_model_comparison(results)
    
    # Save best model
    print("Saving best model...")
    save_best_model(results)
    
    print("Model training complete!")

if __name__ == "__main__":
    main() 