# Lawyer Win Prediction System

This project implements a comprehensive system for predicting lawyer success rates and recommending lawyers based on various performance metrics. The system consists of four main phases:

1. Data Generation: Creates synthetic lawyer and case data
2. Feature Engineering: Processes and prepares data for modeling
3. Model Training: Trains multiple machine learning models
4. Web Dashboard: Provides an interactive interface for lawyer recommendations

## Project Structure

```
lawyer_win_prediction/
├── data/               # Generated data files
├── models/            # Trained models and feature information
├── src/               # Source code
│   ├── data_generation.py
│   └── model_training.py
├── webapp/            # Streamlit web application
│   └── app.py
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

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
- Performance metrics: Average bill rate, work rate
- Professional achievements: Awards count
- Case statistics: Total cases handled
- Success metrics: Overall success rate
- Case type frequencies: Percentage breakdown by legal area
- Financial performance: Total revenue, average revenue per case

### Case Information
- Case identifiers: Case ID, lawyer ID
- Case details: Type of case, court level
- Timeline: Start date, end date, duration
- Outcomes: Case result (won, lost, settled)
- Financial metrics: Billable hours, revenue generated

## Model Performance

The system trains and evaluates multiple models:
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- XGBoost

The best performing model is automatically selected and used in the web dashboard.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 