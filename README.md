# E-Commerce Customer Behaviour Analysis & Churn Prediction

A complete end-to-end Data Analytics, Machine Learning, and Generative AI project.

## Project Description

This application analyses e-commerce customer transaction data to uncover behavioural patterns and predict customer churn using multiple machine learning models, enhanced with Google Gemini AI for automated business intelligence and natural language insights.

## Features

- **Data Preprocessing** — Deduplication, missing value handling, encoding, and scaling
- **Feature Engineering** — CLV approximation, recency, return rate, purchase diversity
- **Exploratory Data Analysis** — 14+ interactive visualisations across demographics, behaviour, and churn
- **5 ML Models** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC with visual comparisons
- **Explainability** — Confusion matrices, ROC curves, feature importance, SHAP values
- **Gemini AI Integration** — EDA summaries, churn explanations, retention recommendations, Q&A
- **Interactive Prediction** — Real-time churn risk gauge for custom customer profiles

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
streamlit run app.py --server.port 5000
```

## Model Details

| Model | Type |
|---|---|
| Logistic Regression | Linear classifier |
| Decision Tree | Tree-based |
| Random Forest | Bagging ensemble |
| Gradient Boosting | Boosting ensemble |
| XGBoost | Optimised boosting |

The best model is selected by ROC-AUC score and saved to `models/best_model.joblib`.

## GenAI Integration

Uses Google Gemini (`gemini-2.0-flash`) to:
- Generate automated EDA summaries
- Explain customer churn reasons from feature importances
- Produce actionable retention recommendations
- Answer natural language questions about the dataset

Requires `GEMINI_API_KEY` environment variable.

## Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   └── ecommerce_data.csv    # Dataset
├── models/
│   └── best_model.joblib     # Trained best model
├── utils/
│   ├── preprocessing.py      # Data cleaning & encoding
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py     # Model training & saving
│   ├── evaluation.py         # Metrics & visualisation
│   ├── visualization.py      # EDA plots
│   └── gemini_utils.py       # Gemini AI integration
└── README.md
```
