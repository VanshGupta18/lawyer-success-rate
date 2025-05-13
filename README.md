# 🏛️ Legal Case Outcome & Top Lawyer Predictor

This project predicts the likely outcome of legal cases and recommends the top lawyers most likely to win a case based on historical data. It uses machine learning models (Decision Tree and Random Forest) and provides an interactive interface built with Streamlit.

---

## 🚀 Features

- Predicts legal case outcomes using ML models
- Recommends top lawyers based on:
  - Case type
  - Court level
  - Duration and contextual similarity
- Interactive and user-friendly Streamlit UI
- Detailed model metrics and visualizations
- Dynamic lawyer cards with experience and confidence level

---

## 📁 Project Structure

legal-predictor/
│
├── data/
│   ├── lawyers.csv               # Lawyer profile data
│   └── cases.csv                 # Historical case data
│
├── models/
│   ├── decision_tree_model.joblib
│   └── random_forest_model.joblib
│
├── app.py                        # Streamlit frontend app
├── model_training.py             # Model training + pipeline setup
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── .gitignore

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lawyer_win_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

1. Generate synthetic data:
```bash
python src/data_generation.py
```

2. Train the models:
```bash
python src/model_training.py
```

3. Run the web dashboard:
```bash
streamlit run webapp/app.py
```

## Features

- Synthetic data generation for lawyers and cases
- Multiple machine learning models for prediction
- Interactive web dashboard with:
  - Lawyer filtering by case type and experience
  - Win probability predictions
  - Performance visualizations
  - Detailed lawyer profiles

## Data Schema

### Lawyer Information
- Basic info: ID, law school, graduation year
- Career info: Starting practice date, years of experience
- Case statistics: Total cases handled


### Case Information
- Case identifiers: Case ID, lawyer ID
- Case details: Type of case, court level
- Timeline: Start date, end date, duration
- Outcomes: Case result (won, lost, settled)


## Model Performance

The system trains and evaluates multiple models:
- Decision Trees
- Random Forest


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 