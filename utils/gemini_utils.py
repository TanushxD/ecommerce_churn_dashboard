import os
try:
    from google import genai
except Exception:
    genai = None

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    if genai is None:
        return None

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        _client = genai.Client(api_key=api_key)
    except Exception:
        return None
    return _client


def _generate(prompt: str) -> str:
    if genai is None:
        return "⚠️ Gemini SDK not available. Install `google-genai` to enable AI features."
    client = get_client()
    if client is None:
        if not os.environ.get("GEMINI_API_KEY", ""):
            return "⚠️ GEMINI_API_KEY not configured. Please set your API key."
        return "⚠️ Unable to initialize Gemini client. Check your API key and SDK installation."
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text or "⚠️ Gemini returned an empty response."


def generate_eda_insights(eda_summary: dict) -> str:
    prompt = f"""You are a senior e-commerce data analyst. Based on the following EDA statistics from a customer dataset, provide a professional and insightful business summary (5-8 bullet points):

Dataset Summary:
- Total transactions: {eda_summary.get('total_transactions', 'N/A')}
- Unique customers: {eda_summary.get('unique_customers', 'N/A')}
- Churn rate: {eda_summary.get('churn_rate', 'N/A')}%
- Average purchase value: ${eda_summary.get('avg_purchase', 0):.2f}
- Return rate: {eda_summary.get('return_rate', 'N/A')}%
- Top product category: {eda_summary.get('top_category', 'N/A')}
- Most used payment method: {eda_summary.get('top_payment', 'N/A')}
- Average customer age: {eda_summary.get('avg_age', 'N/A')}
- Date range: {eda_summary.get('date_range', 'N/A')}

Generate 5-8 concise, actionable business insights from this data. Format as bullet points starting with •."""
    try:
        return _generate(prompt)
    except Exception as e:
        return f"⚠️ Error generating insights: {str(e)}"


def explain_churn_reasons(model_metrics: dict, feature_importances: list) -> str:
    top_features = ", ".join([f[0] for f in feature_importances[:10]]) if feature_importances else "N/A"
    prompt = f"""You are a machine learning expert and customer retention specialist. Based on the following churn prediction model results, explain why customers are likely churning.

Best Model: {model_metrics.get('best_model', 'N/A')}
ROC-AUC Score: {model_metrics.get('roc_auc', 'N/A')}
Churn Rate: {model_metrics.get('churn_rate', 'N/A')}%
Top Predictive Features: {top_features}

Provide a clear explanation (5-7 bullet points) of:
1. Why customers are churning based on the key features
2. Which customer segments are most at risk
3. What behavioural patterns indicate churn risk

Format as bullet points starting with •."""
    try:
        return _generate(prompt)
    except Exception as e:
        return f"⚠️ Error generating churn explanation: {str(e)}"


def generate_retention_recommendations(eda_summary: dict, churn_rate: float) -> str:
    prompt = f"""You are a customer retention strategist for an e-commerce platform. Given:
- Current churn rate: {churn_rate:.1f}%
- Average purchase value: ${eda_summary.get('avg_purchase', 0):.2f}
- Top product category: {eda_summary.get('top_category', 'N/A')}
- Most used payment method: {eda_summary.get('top_payment', 'N/A')}
- Return rate: {eda_summary.get('return_rate', 0):.1f}%

Generate 8-10 specific, actionable customer retention recommendations. Be concrete and data-driven. Format each recommendation with:
• [Recommendation]: [Brief explanation of expected impact]"""
    try:
        return _generate(prompt)
    except Exception as e:
        return f"⚠️ Error generating recommendations: {str(e)}"


def answer_question(question: str, eda_summary: dict) -> str:
    prompt = f"""You are a data analyst assistant for an e-commerce company. Answer the following question based on the dataset context below.

Dataset Context:
- Total transactions: {eda_summary.get('total_transactions', 'N/A')}
- Unique customers: {eda_summary.get('unique_customers', 'N/A')}
- Churn rate: {eda_summary.get('churn_rate', 'N/A')}%
- Average purchase value: ${eda_summary.get('avg_purchase', 0):.2f}
- Return rate: {eda_summary.get('return_rate', 0):.1f}%
- Top product category: {eda_summary.get('top_category', 'N/A')}
- Most used payment method: {eda_summary.get('top_payment', 'N/A')}
- Date range: {eda_summary.get('date_range', 'N/A')}
- Gender distribution: {eda_summary.get('gender_dist', 'N/A')}

User Question: {question}

Provide a clear, concise, data-informed answer in 3-5 sentences."""
    try:
        return _generate(prompt)
    except Exception as e:
        return f"⚠️ Error answering question: {str(e)}"
