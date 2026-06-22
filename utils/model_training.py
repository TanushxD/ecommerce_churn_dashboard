import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from utils.evaluation import evaluate_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
    }


def train_all_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    classifiers = get_classifiers()
    results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test)
        results[name] = {
            "model": clf,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
        }

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)

    scaler_info_path = os.path.join(MODELS_DIR, "best_model_name.txt")
    with open(scaler_info_path, "w") as f:
        f.write(best_name)

    return results, best_name, X_train, X_test, y_train, y_test


def load_best_model():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def predict_churn(model, X: pd.DataFrame):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pred, proba
