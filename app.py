import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import load_data, preprocess_data, encode_and_scale
from utils.feature_engineering import build_features
from utils.model_training import train_all_models, load_best_model, predict_churn
from utils.evaluation import (
    plot_confusion_matrix, plot_roc_curves, plot_feature_importance,
    plot_shap_summary, metrics_dataframe
)
from utils.visualization import (
    plot_age_distribution, plot_gender_distribution, plot_category_analysis,
    plot_payment_method, plot_purchase_trend, plot_returns_analysis,
    plot_spending_distribution, plot_churn_distribution, plot_churn_by_gender,
    plot_churn_by_age_group, plot_churn_by_payment, plot_churn_by_category,
    plot_returns_vs_churn, plot_spending_vs_churn
)
from utils.gemini_utils import (
    generate_eda_insights, explain_churn_reasons,
    generate_retention_recommendations, answer_question
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ecommerce_data.csv")

st.set_page_config(
    page_title="E-Commerce Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_raw_data():
    return load_data(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_preprocessed(_df):
    return preprocess_data(_df)


@st.cache_data(show_spinner=False)
def get_features(_df_clean):
    return build_features(_df_clean)


@st.cache_data(show_spinner=False)
def get_eda_summary(_df):
    age_col = "Age" if "Age" in _df.columns else "Customer Age"
    churn_base = _df.drop_duplicates("Customer ID") if "Customer ID" in _df.columns else _df
    churn_rate = churn_base["Churn"].mean() * 100 if "Churn" in _df.columns else 0
    return_rate = _df["Returns"].mean() * 100 if "Returns" in _df.columns else 0
    gender_dist = churn_base["Gender"].value_counts().to_dict() if "Gender" in _df.columns else {}
    date_range = "N/A"
    if "Purchase Date" in _df.columns:
        dates = pd.to_datetime(_df["Purchase Date"], errors="coerce")
        date_range = f"{dates.min().date()} to {dates.max().date()}"
    return {
        "total_transactions": len(_df),
        "unique_customers": _df["Customer ID"].nunique() if "Customer ID" in _df.columns else len(_df),
        "churn_rate": round(churn_rate, 2),
        "avg_purchase": _df["Total Purchase Amount"].mean() if "Total Purchase Amount" in _df.columns else 0,
        "return_rate": round(return_rate, 2),
        "top_category": _df["Product Category"].mode()[0] if "Product Category" in _df.columns else "N/A",
        "top_payment": _df["Payment Method"].mode()[0] if "Payment Method" in _df.columns else "N/A",
        "avg_age": round(_df[age_col].mean(), 1) if age_col in _df.columns else "N/A",
        "date_range": date_range,
        "gender_dist": str(gender_dist),
    }


@st.cache_resource(show_spinner=False)
def run_training(_feat_df):
    target = "Churn"
    if target not in _feat_df.columns:
        return None, None, None, None, None, None
    X_sc, y, scaler, encoders, feature_cols = encode_and_scale(_feat_df, target_col=target)
    results, best_name, X_train, X_test, y_train, y_test = train_all_models(X_sc, y)
    return results, best_name, scaler, encoders, feature_cols, X_sc


df_raw = load_raw_data()
df_clean = get_preprocessed(df_raw)
feat_df = get_features(df_clean)
eda_summary = get_eda_summary(df_raw)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("E-Commerce Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📋 Dataset Overview",
            "📊 Exploratory Data Analysis",
            "🤖 Churn Prediction",
            "📈 Model Performance",
            "💡 AI Business Insights",
            "🎯 Recommendations",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Powered by Gemini AI & Scikit-Learn")

if page == "🏠 Home":
    st.title("E-Commerce Customer Behaviour Analysis")
    st.subheader("Customer Churn Prediction using Machine Learning & Generative AI")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{eda_summary['total_transactions']:,}")
    with col2:
        st.metric("Unique Customers", f"{eda_summary['unique_customers']:,}")
    with col3:
        st.metric("Churn Rate", f"{eda_summary['churn_rate']}%")
    with col4:
        st.metric("Avg Purchase Value", f"${eda_summary['avg_purchase']:.2f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Project Overview")
        st.markdown("""
This end-to-end analytics platform analyses e-commerce customer behaviour and predicts
customer churn using state-of-the-art machine learning models enhanced with
Google Gemini AI for automated business intelligence.

**Key Features:**
- Complete data preprocessing pipeline
- 14+ interactive EDA visualisations
- Feature engineering (CLV, recency, diversity)
- 5 ML models compared head-to-head
- SHAP-based model explainability
- Gemini-powered business insights
- Interactive churn prediction tool
        """)
    with col_r:
        st.markdown("### ML Models Used")
        models_info = {
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest",
                       "Gradient Boosting", "XGBoost"],
            "Type": ["Linear", "Tree-based", "Ensemble", "Ensemble", "Boosting"],
        }
        st.dataframe(pd.DataFrame(models_info), use_container_width=True, hide_index=True)

        st.markdown("### Dataset Info")
        st.info(f"""
- **Date Range:** {eda_summary['date_range']}
- **Top Category:** {eda_summary['top_category']}
- **Top Payment:** {eda_summary['top_payment']}
- **Return Rate:** {eda_summary['return_rate']}%
        """)

elif page == "📋 Dataset Overview":
    st.title("Dataset Overview")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df_raw):,}")
    with col2:
        st.metric("Columns", len(df_raw.columns))
    with col3:
        st.metric("Missing Values", int(df_raw.isnull().sum().sum()))

    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head(100), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Column Types")
        dtype_df = pd.DataFrame({
            "Column": df_raw.dtypes.index,
            "Type": df_raw.dtypes.values.astype(str),
            "Missing": df_raw.isnull().sum().values,
            "Missing %": (df_raw.isnull().mean() * 100).round(2).values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with col_r:
        st.subheader("Descriptive Statistics")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

    st.subheader("Engineered Features Preview")
    st.dataframe(feat_df.head(50), use_container_width=True)

elif page == "📊 Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Customer Demographics", "Purchase Behaviour", "Churn Analysis"])

    with tab1:
        st.subheader("Customer Demographics")
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_age_distribution(df_raw)
            if fig:
                st.plotly_chart(fig)
        with col2:
            fig = plot_gender_distribution(df_raw)
            if fig:
                st.plotly_chart(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig = plot_category_analysis(df_raw)
            if fig:
                st.plotly_chart(fig)
        with col4:
            fig = plot_payment_method(df_raw)
            if fig:
                st.plotly_chart(fig)

    with tab2:
        st.subheader("Purchase Behaviour")
        fig = plot_purchase_trend(df_raw)
        if fig:
            st.plotly_chart(fig)

        col1, col2 = st.columns(2)
        with col1:
            fig = plot_returns_analysis(df_raw)
            if fig:
                st.plotly_chart(fig)
        with col2:
            fig = plot_spending_distribution(df_raw)
            if fig:
                st.plotly_chart(fig)

    with tab3:
        st.subheader("Churn Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_churn_distribution(df_raw)
            if fig:
                st.plotly_chart(fig)
        with col2:
            fig = plot_churn_by_gender(df_raw)
            if fig:
                st.plotly_chart(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig = plot_churn_by_age_group(df_raw)
            if fig:
                st.plotly_chart(fig)
        with col4:
            fig = plot_churn_by_payment(df_raw)
            if fig:
                st.plotly_chart(fig)

        col5, col6 = st.columns(2)
        with col5:
            fig = plot_churn_by_category(df_raw)
            if fig:
                st.plotly_chart(fig)
        with col6:
            fig = plot_returns_vs_churn(df_raw)
            if fig:
                st.plotly_chart(fig)

        fig = plot_spending_vs_churn(df_raw)
        if fig:
            st.plotly_chart(fig)

elif page == "🤖 Churn Prediction":
    st.title("Customer Churn Prediction")
    st.markdown("---")

    with st.spinner("Training models (this may take a moment on first run)..."):
        results, best_name, scaler, encoders, feature_cols, X_scaled = run_training(feat_df)

    if results is None:
        st.error("Training failed — Churn column not found in engineered features.")
        st.stop()

    st.success(f"Models trained! Best model: **{best_name}** (ROC-AUC: {results[best_name]['metrics']['roc_auc']:.4f})")

    st.subheader("Predict Churn for a New Customer")
    st.markdown("Adjust the sliders below to simulate a customer profile:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Customer Age", 18, 75, 35)
        total_spend = st.slider("Total Spend ($)", 100, 50000, 5000)
        purchase_freq = st.slider("Number of Purchases", 1, 30, 5)
    with col2:
        avg_purchase = st.slider("Avg Purchase Value ($)", 50, 5000, 1000)
        return_rate = st.slider("Return Rate", 0.0, 1.0, 0.1, step=0.05)
        recency = st.slider("Recency (days since last purchase)", 1, 365, 60)
    with col3:
        clv = st.slider("Customer Lifetime Value ($)", 100, 100000, 10000)
        diversity = st.slider("Purchase Category Diversity", 1, 5, 2)
        tenure = st.slider("Customer Tenure (days)", 30, 1500, 365)

    input_data = {col: 0 for col in feature_cols}
    mapping = {
        "Total_Spend": total_spend,
        "Avg_Purchase_Value": avg_purchase,
        "Purchase_Frequency": purchase_freq,
        "Total_Returns": return_rate * purchase_freq,
        "Return_Rate": return_rate,
        "Recency_Days": recency,
        "CLV_Approx": clv,
        "Purchase_Diversity": diversity,
        "Customer_Tenure_Days": tenure,
        "Age": age,
        "Customer Age": age,
    }
    for k, v in mapping.items():
        if k in input_data:
            input_data[k] = v

    input_df = pd.DataFrame([input_data])
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)

    best_model = results[best_name]["model"]
    pred, proba = predict_churn(best_model, input_scaled)

    st.markdown("---")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        label = "Likely to Churn" if pred[0] == 1 else "Not Likely to Churn"
        color = "error" if pred[0] == 1 else "success"
        if pred[0] == 1:
            st.error(f"**Prediction: {label}**")
        else:
            st.success(f"**Prediction: {label}**")
    with col_res2:
        st.metric("Churn Probability", f"{proba[0]*100:.1f}%")

    import plotly.graph_objects as go_gauge
    gauge_fig = go_gauge.Figure(go_gauge.Indicator(
        mode="gauge+number",
        value=proba[0] * 100,
        number={"suffix": "%"},
        title={"text": "Churn Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#EF553B" if proba[0] > 0.5 else "#00CC96"},
            "steps": [
                {"range": [0, 30], "color": "#d4edda"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 100], "color": "#f8d7da"},
            ],
        },
    ))
    gauge_fig.update_layout(height=300)
    st.plotly_chart(gauge_fig)

elif page == "📈 Model Performance":
    st.title("Model Performance Comparison")
    st.markdown("---")

    with st.spinner("Loading model results..."):
        results, best_name, scaler, encoders, feature_cols, X_scaled = run_training(feat_df)

    if results is None:
        st.error("No model results available.")
        st.stop()

    st.subheader("Metrics Summary")
    metrics_df = metrics_dataframe(results)
    st.dataframe(
        metrics_df.style.highlight_max(subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
                                        color="#d4edda"),
        use_container_width=True, hide_index=True
    )

    st.subheader("ROC Curves")
    roc_fig = plot_roc_curves(results)
    st.pyplot(roc_fig)

    st.subheader("Confusion Matrices")
    model_names = list(results.keys())
    cols = st.columns(min(3, len(model_names)))
    for i, name in enumerate(model_names):
        with cols[i % 3]:
            cm_fig = plot_confusion_matrix(
                results[name]["model"],
                results[name]["X_test"],
                results[name]["y_test"],
                name
            )
            st.pyplot(cm_fig)

    st.subheader(f"Feature Importance — {best_name}")
    fi_fig = plot_feature_importance(results[best_name]["model"], feature_cols, best_name)
    st.pyplot(fi_fig)

    st.subheader(f"SHAP Explanations — {best_name}")
    with st.spinner("Computing SHAP values..."):
        shap_fig = plot_shap_summary(
            results[best_name]["model"],
            results[best_name]["X_test"],
            feature_cols, best_name
        )
    st.pyplot(shap_fig)

elif page == "💡 AI Business Insights":
    st.title("AI-Generated Business Insights")
    st.subheader("Powered by Google Gemini")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["EDA Insights", "Churn Explanation", "Ask a Question"])

    with tab1:
        st.subheader("Automated EDA Insights")
        if st.button("Generate EDA Insights", type="primary"):
            with st.spinner("Gemini is analysing your data..."):
                insights = generate_eda_insights(eda_summary)
            st.markdown(insights)
        else:
            st.info("Click the button above to generate AI-powered EDA insights from your dataset.")

    with tab2:
        st.subheader("Why Are Customers Churning?")
        with st.spinner("Loading model results..."):
            results, best_name, scaler, encoders, feature_cols, X_scaled = run_training(feat_df)

        if results and st.button("Explain Churn Reasons", type="primary"):
            best_model = results[best_name]["model"]
            feature_importances = []
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
                feature_importances = sorted(
                    zip(feature_cols, importances), key=lambda x: x[1], reverse=True
                )
            model_metrics = {
                "best_model": best_name,
                "roc_auc": results[best_name]["metrics"]["roc_auc"],
                "churn_rate": eda_summary["churn_rate"],
            }
            with st.spinner("Gemini is explaining churn patterns..."):
                explanation = explain_churn_reasons(model_metrics, feature_importances)
            st.markdown(explanation)
        elif not results:
            st.warning("Train models first on the Model Performance page.")

    with tab3:
        st.subheader("Ask a Question About the Dataset")
        question = st.text_input(
            "Your question",
            placeholder="e.g. Which age group has the highest churn rate?"
        )
        if st.button("Ask Gemini", type="primary") and question:
            with st.spinner("Gemini is thinking..."):
                answer = answer_question(question, eda_summary)
            st.markdown("**Answer:**")
            st.markdown(answer)

elif page == "🎯 Recommendations":
    st.title("Customer Retention Recommendations")
    st.subheader("AI-Powered Strategies to Reduce Churn")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Churn Rate", f"{eda_summary['churn_rate']}%")
    with col2:
        target_churn = max(5.0, eda_summary['churn_rate'] * 0.7)
        st.metric("Target Churn Rate", f"{target_churn:.1f}%", delta=f"-{eda_summary['churn_rate'] - target_churn:.1f}%")
    with col3:
        st.metric("Return Rate", f"{eda_summary['return_rate']}%")

    if st.button("Generate Retention Recommendations", type="primary"):
        with st.spinner("Gemini is crafting personalised recommendations..."):
            recommendations = generate_retention_recommendations(
                eda_summary, eda_summary["churn_rate"]
            )
        st.markdown(recommendations)
    else:
        st.info("Click the button above to generate AI-powered retention strategies.")

    st.markdown("---")
    st.subheader("Quick Win Strategies")
    quick_wins = [
        ("Loyalty Programme", "Reward repeat purchases to increase retention"),
        ("Re-engagement Emails", "Target customers with high recency scores"),
        ("Return Policy Review", "Improve satisfaction to reduce return rates"),
        ("Payment Flexibility", "Offer more payment methods to reduce friction"),
        ("Personalised Offers", "Use purchase history to send targeted promotions"),
        ("Customer Feedback Loop", "Survey churned customers to identify root causes"),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(quick_wins):
        with cols[i % 3]:
            st.info(f"**{title}**\n\n{desc}")
