import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churned", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


def plot_roc_curves(results: dict):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, info in results.items():
        model = info["model"]
        X_test = info["X_test"]
        y_test = info["y_test"]
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = info["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    fig, ax = plt.subplots(figsize=(8, 6))
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        ax.barh(
            [feature_names[i] for i in indices][::-1],
            importances[indices][::-1],
            color="steelblue",
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {model_name}")
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_[0])
        indices = np.argsort(coef)[::-1][:top_n]
        ax.barh(
            [feature_names[i] for i in indices][::-1],
            coef[indices][::-1],
            color="steelblue",
        )
        ax.set_xlabel("|Coefficient|")
        ax.set_title(f"Feature Coefficients — {model_name}")
    else:
        ax.text(0.5, 0.5, "Feature importance not available for this model.",
                ha="center", va="center")
    plt.tight_layout()
    return fig


def plot_shap_summary(model, X_test, feature_names: list, model_name: str):
    try:
        import shap
        fig, ax = plt.subplots(figsize=(8, 6))
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values
            shap.summary_plot(sv, X_test, feature_names=feature_names, show=False, plot_size=None)
            fig = plt.gcf()
            fig.suptitle(f"SHAP Summary — {model_name}", y=1.01)
        else:
            explainer = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, plot_size=None)
            fig = plt.gcf()
        plt.tight_layout()
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "SHAP not available for this model configuration.",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig


def metrics_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for name, info in results.items():
        m = info["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1"],
            "ROC-AUC": m["roc_auc"],
        })
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return df
