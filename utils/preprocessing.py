import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    df = df.drop_duplicates()

    if "Returns" in df.columns:
        df["Returns"] = df["Returns"].fillna(0)

    for col in df.select_dtypes(include="object").columns:
        if col not in ["Customer Name", "Customer ID", "Purchase Date"]:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    if "Purchase Date" in df.columns:
        df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")
        df["Purchase Year"] = df["Purchase Date"].dt.year
        df["Purchase Month"] = df["Purchase Date"].dt.month
        df["Purchase DayOfWeek"] = df["Purchase Date"].dt.dayofweek

    drop_cols = [c for c in ["Customer Name"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def encode_and_scale(df: pd.DataFrame, target_col: str = "Churn"):
    df = df.copy()

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["Purchase Date"]]

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    drop_date = [c for c in ["Purchase Date"] if c in df.columns]
    df = df.drop(columns=drop_date, errors="ignore")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, scaler, label_encoders, feature_cols


def get_customer_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to customer level for EDA."""
    df = df.copy()
    if "Purchase Date" in df.columns:
        df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")

    group_key = None
    if "Customer ID" in df.columns:
        group_key = "Customer ID"
    else:
        return df

    agg = df.groupby(group_key).agg(
        Total_Spend=("Total Purchase Amount", "sum"),
        Num_Purchases=("Total Purchase Amount", "count"),
        Avg_Purchase=("Total Purchase Amount", "mean"),
        Total_Returns=("Returns", "sum"),
        Churn=("Churn", "first"),
        Gender=("Gender", "first"),
        Age=("Age", "first") if "Age" in df.columns else ("Customer Age", "first"),
    ).reset_index()

    if "Product Category" in df.columns:
        category_counts = df.groupby([group_key, "Product Category"]).size().unstack(fill_value=0)
        agg = agg.merge(category_counts, on=group_key, how="left")

    if "Payment Method" in df.columns:
        payment_counts = df.groupby([group_key, "Payment Method"]).size().unstack(fill_value=0)
        agg = agg.merge(payment_counts, on=group_key, how="left")

    return agg
