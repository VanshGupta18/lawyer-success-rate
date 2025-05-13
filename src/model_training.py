# All required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json

# Load or generate data
if not os.path.exists('data/lawyers.csv') or not os.path.exists('data/cases.csv'):
    import data_generation
    data_generation.main()

lawyers_df = pd.read_csv('data/lawyers.csv')
cases_df = pd.read_csv('data/cases.csv')

# Merge and preprocess
df = pd.merge(cases_df, lawyers_df, on='lawyer_id')
for col in ['start_date', 'end_date', 'start_practice']:
    df[col] = pd.to_datetime(df[col])
df['case_month'] = df['start_date'].dt.month
df['case_year'] = df['start_date'].dt.year
df['case_day_of_week'] = df['start_date'].dt.dayofweek
df['lawyer_experience_at_case'] = (df['start_date'] - df['start_practice']).dt.days / 365.25

# Features and target
features = ['case_type', 'court_level', 'duration_days', 'school_tier',
            'lawyer_experience_at_case', 'case_month', 'case_year', 'case_day_of_week']
target = 'result'
categorical_features = ['case_type', 'court_level', 'school_tier']
numerical_features = ['duration_days', 'lawyer_experience_at_case', 'case_month', 'case_year', 'case_day_of_week']

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Create models folder
os.makedirs('models', exist_ok=True)

# Decision Tree Model
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
dt_param_grid = {
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}
dt_grid = GridSearchCV(dt_pipeline, dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)
dt_best_model = dt_grid.best_estimator_
joblib.dump(dt_best_model, 'models/decision_tree_model.joblib')

# Evaluate Decision Tree
dt_y_pred = dt_best_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_report = classification_report(y_test, dt_y_pred, output_dict=True)
print("Decision Tree Accuracy:", dt_accuracy)

# Random Forest Model
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
rf_param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None, 20],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1]
}
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best_model = rf_grid.best_estimator_
joblib.dump(rf_best_model, 'models/random_forest_model.joblib')

# Evaluate Random Forest
rf_y_pred = rf_best_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred, output_dict=True)
print("Random Forest Accuracy:", rf_accuracy)

# Get Top Lawyers Function
def get_top_lawyers(case_type, court_level, duration_days, model, top_n=10):
    similar_cases = df[
        (df['case_type'] == case_type) &
        (df['court_level'] == court_level) &
        (df['duration_days'].between(duration_days - 15, duration_days + 15))
    ].copy()

    if similar_cases.empty:
        print("No similar cases found.")
        return pd.DataFrame()

    X_similar = similar_cases[features]
    similar_cases.loc[:, 'predicted_result'] = model.predict(X_similar)
    proba = model.predict_proba(X_similar)
    win_index = list(model.classes_).index('Won')
    similar_cases.loc[:, 'win_confidence'] = proba[:, win_index]

    # Aggregate top lawyers by average win confidence
    top_lawyers = (
        similar_cases.groupby('lawyer_id')
        .agg(avg_confidence=('win_confidence', 'mean'))
        .sort_values(by='avg_confidence', ascending=False)
        .head(top_n)
        .reset_index()
    )

    # Merge with original lawyer details
    enriched = pd.merge(top_lawyers, lawyers_df, on='lawyer_id', how='left')
    return enriched[['name', 'law_school', 'start_practice', 'years_experience', 'avg_confidence']]

# Example usage
example_input = {
    'case_type': 'Criminal',
    'court_level': 'High Court',
    'duration_days': 100
}
top_lawyers = get_top_lawyers(
    case_type=example_input['case_type'],
    court_level=example_input['court_level'],
    duration_days=example_input['duration_days'],
    model=rf_best_model,
    top_n=5
)
print("\nTop Lawyers Likely to Win (Random Forest):")
print(top_lawyers)

# Save Evaluation Reports
with open("models/dt_classification_report.json", "w") as f:
    json.dump(dt_report, f, indent=4)
with open("models/rf_classification_report.json", "w") as f:
    json.dump(rf_report, f, indent=4)