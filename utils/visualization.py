import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def plot_age_distribution(df: pd.DataFrame):
    age_col = "Age" if "Age" in df.columns else "Customer Age"
    fig = px.histogram(df, x=age_col, nbins=30, title="Customer Age Distribution",
                       color_discrete_sequence=["#636EFA"])
    fig.update_layout(xaxis_title="Age", yaxis_title="Count")
    return fig


def plot_gender_distribution(df: pd.DataFrame):
    if "Gender" not in df.columns:
        return None
    counts = df.drop_duplicates("Customer ID")["Gender"].value_counts() if "Customer ID" in df.columns else df["Gender"].value_counts()
    fig = px.pie(names=counts.index, values=counts.values, title="Gender Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    return fig


def plot_category_analysis(df: pd.DataFrame):
    if "Product Category" not in df.columns:
        return None
    counts = df["Product Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig = px.bar(counts, x="Category", y="Count", title="Purchases by Product Category",
                 color="Category", color_discrete_sequence=px.colors.qualitative.Pastel)
    return fig


def plot_payment_method(df: pd.DataFrame):
    if "Payment Method" not in df.columns:
        return None
    counts = df["Payment Method"].value_counts().reset_index()
    counts.columns = ["Payment Method", "Count"]
    fig = px.bar(counts, x="Payment Method", y="Count", title="Purchases by Payment Method",
                 color="Payment Method", color_discrete_sequence=px.colors.qualitative.Set1)
    return fig


def plot_purchase_trend(df: pd.DataFrame):
    if "Purchase Date" not in df.columns:
        return None
    df2 = df.copy()
    df2["Purchase Date"] = pd.to_datetime(df2["Purchase Date"], errors="coerce")
    df2["Month"] = df2["Purchase Date"].dt.to_period("M").dt.to_timestamp()
    trend = df2.groupby("Month")["Total Purchase Amount"].sum().reset_index()
    fig = px.line(trend, x="Month", y="Total Purchase Amount",
                  title="Monthly Purchase Revenue Trend",
                  color_discrete_sequence=["#EF553B"])
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Revenue")
    return fig


def plot_returns_analysis(df: pd.DataFrame):
    if "Returns" not in df.columns:
        return None
    df2 = df.copy()
    df2["Returns"] = df2["Returns"].fillna(0)
    ret_counts = df2["Returns"].value_counts().rename({0.0: "No Return", 1.0: "Returned"})
    fig = px.pie(names=ret_counts.index.astype(str), values=ret_counts.values,
                 title="Returns Distribution", color_discrete_sequence=["#00CC96", "#EF553B"])
    return fig


def plot_spending_distribution(df: pd.DataFrame):
    if "Total Purchase Amount" not in df.columns:
        return None
    fig = px.box(df, y="Total Purchase Amount", x="Product Category" if "Product Category" in df.columns else None,
                 title="Spending Distribution by Category",
                 color="Product Category" if "Product Category" in df.columns else None,
                 color_discrete_sequence=px.colors.qualitative.Bold)
    return fig


def plot_churn_distribution(df: pd.DataFrame):
    if "Churn" not in df.columns:
        return None
    counts = df.drop_duplicates("Customer ID")["Churn"].value_counts() if "Customer ID" in df.columns else df["Churn"].value_counts()
    counts.index = ["Not Churned" if i == 0 else "Churned" for i in counts.index]
    fig = px.pie(names=counts.index, values=counts.values, title="Churn Distribution",
                 color_discrete_sequence=["#00CC96", "#EF553B"])
    return fig


def plot_churn_by_gender(df: pd.DataFrame):
    if "Gender" not in df.columns or "Churn" not in df.columns:
        return None
    base = df.drop_duplicates("Customer ID") if "Customer ID" in df.columns else df
    ct = base.groupby(["Gender", "Churn"]).size().reset_index(name="Count")
    ct["Churn"] = ct["Churn"].map({0: "Not Churned", 1: "Churned"})
    fig = px.bar(ct, x="Gender", y="Count", color="Churn", barmode="group",
                 title="Churn by Gender", color_discrete_map={"Not Churned": "#00CC96", "Churned": "#EF553B"})
    return fig


def plot_churn_by_age_group(df: pd.DataFrame):
    age_col = "Age" if "Age" in df.columns else ("Customer Age" if "Customer Age" in df.columns else None)
    if age_col is None or "Churn" not in df.columns:
        return None
    base = df.drop_duplicates("Customer ID") if "Customer ID" in df.columns else df
    base = base.copy()
    base["Age Group"] = pd.cut(base[age_col], bins=[0, 25, 35, 45, 55, 65, 100],
                                labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"])
    ct = base.groupby(["Age Group", "Churn"]).size().reset_index(name="Count")
    ct["Churn"] = ct["Churn"].map({0: "Not Churned", 1: "Churned"})
    fig = px.bar(ct, x="Age Group", y="Count", color="Churn", barmode="group",
                 title="Churn by Age Group", color_discrete_map={"Not Churned": "#00CC96", "Churned": "#EF553B"})
    return fig


def plot_churn_by_payment(df: pd.DataFrame):
    if "Payment Method" not in df.columns or "Churn" not in df.columns:
        return None
    ct = df.groupby(["Payment Method", "Churn"]).size().reset_index(name="Count")
    ct["Churn"] = ct["Churn"].map({0: "Not Churned", 1: "Churned"})
    fig = px.bar(ct, x="Payment Method", y="Count", color="Churn", barmode="group",
                 title="Churn by Payment Method",
                 color_discrete_map={"Not Churned": "#00CC96", "Churned": "#EF553B"})
    return fig


def plot_churn_by_category(df: pd.DataFrame):
    if "Product Category" not in df.columns or "Churn" not in df.columns:
        return None
    ct = df.groupby(["Product Category", "Churn"]).size().reset_index(name="Count")
    ct["Churn"] = ct["Churn"].map({0: "Not Churned", 1: "Churned"})
    fig = px.bar(ct, x="Product Category", y="Count", color="Churn", barmode="group",
                 title="Churn by Product Category",
                 color_discrete_map={"Not Churned": "#00CC96", "Churned": "#EF553B"})
    return fig


def plot_returns_vs_churn(df: pd.DataFrame):
    if "Returns" not in df.columns or "Churn" not in df.columns:
        return None
    base = df.drop_duplicates("Customer ID") if "Customer ID" in df.columns else df
    ct = base.groupby(["Returns", "Churn"]).size().reset_index(name="Count")
    ct["Churn"] = ct["Churn"].map({0: "Not Churned", 1: "Churned"})
    ct["Returns"] = ct["Returns"].map({0.0: "No Returns", 1.0: "Has Returns"}).fillna(ct["Returns"].astype(str))
    fig = px.bar(ct, x="Returns", y="Count", color="Churn", barmode="group",
                 title="Returns vs Churn",
                 color_discrete_map={"Not Churned": "#00CC96", "Churned": "#EF553B"})
    return fig


def plot_spending_vs_churn(df: pd.DataFrame):
    if "Total Purchase Amount" not in df.columns or "Churn" not in df.columns:
        return None
    df2 = df.copy()
    df2["Churn"] = df2["Churn"].map({0: "Not Churned", 1: "Churned"})
    fig = px.box(df2, x="Churn", y="Total Purchase Amount", color="Churn",
                 title="Spending vs Churn",
                 color_discrete_map={"Not Churned": "#00CC96", "Churned": "#EF553B"})
    return fig
