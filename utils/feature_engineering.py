import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level feature matrix from transaction data."""
    df = df.copy()

    if "Purchase Date" in df.columns:
        df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")

    id_col = "Customer ID" if "Customer ID" in df.columns else None
    if id_col is None:
        return df

    reference_date = df["Purchase Date"].max() if "Purchase Date" in df.columns else pd.Timestamp.now()

    agg_dict = {
        "Total Purchase Amount": ["sum", "mean", "count"],
        "Returns": ["sum", "mean"],
    }

    if "Product Price" in df.columns:
        agg_dict["Product Price"] = ["mean"]
    if "Quantity" in df.columns:
        agg_dict["Quantity"] = ["sum", "mean"]

    features = df.groupby(id_col).agg(agg_dict)
    features.columns = ["_".join(c) for c in features.columns]
    features = features.reset_index()

    features.rename(columns={
        "Total Purchase Amount_sum": "Total_Spend",
        "Total Purchase Amount_mean": "Avg_Purchase_Value",
        "Total Purchase Amount_count": "Purchase_Frequency",
        "Returns_sum": "Total_Returns",
        "Returns_mean": "Return_Rate",
    }, inplace=True)

    if "Purchase Date" in df.columns:
        recency = df.groupby(id_col)["Purchase Date"].max().reset_index()
        recency["Recency_Days"] = (reference_date - recency["Purchase Date"]).dt.days
        features = features.merge(recency[[id_col, "Recency_Days"]], on=id_col, how="left")

        first_purchase = df.groupby(id_col)["Purchase Date"].min().reset_index()
        first_purchase.rename(columns={"Purchase Date": "First_Purchase"}, inplace=True)
        features = features.merge(first_purchase, on=id_col, how="left")
        features["Customer_Tenure_Days"] = (reference_date - features["First_Purchase"]).dt.days
        features = features.drop(columns=["First_Purchase"], errors="ignore")

    features["CLV_Approx"] = features["Avg_Purchase_Value"] * features["Purchase_Frequency"]

    if "Product Category" in df.columns:
        diversity = df.groupby(id_col)["Product Category"].nunique().reset_index()
        diversity.rename(columns={"Product Category": "Purchase_Diversity"}, inplace=True)
        features = features.merge(diversity, on=id_col, how="left")

    meta_cols = ["Churn", "Gender", "Age", "Customer Age"]
    for col in meta_cols:
        if col in df.columns:
            meta = df.groupby(id_col)[col].first().reset_index()
            features = features.merge(meta, on=id_col, how="left")

    features = features.drop(columns=[id_col], errors="ignore")

    return features
