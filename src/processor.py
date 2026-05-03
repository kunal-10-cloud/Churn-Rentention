import pandas as pd
import numpy as np
import streamlit as st

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

def validate_input(df):
    errors = []

    if df.empty:
        errors.append("Uploaded file has no data (0 rows).")
        return errors

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    if "tenure" in df.columns and not pd.api.types.is_numeric_dtype(df["tenure"]):
        errors.append("Column 'tenure' must be numeric.")

    if "MonthlyCharges" in df.columns and not pd.api.types.is_numeric_dtype(df["MonthlyCharges"]):
        errors.append("Column 'MonthlyCharges' must be numeric.")

    return errors


def _detect_outliers_iqr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def analyze_data_quality(df):
    """Return per-column missing counts, outlier counts, and a 0-100 quality score."""
    total_rows = len(df)
    working = df.copy()

    if "TotalCharges" in working.columns:
        working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")

    missing = {col: int(working[col].isna().sum()) for col in working.columns if working[col].isna().any()}

    outliers = {}
    for col in NUMERIC_COLUMNS:
        if col in working.columns:
            count = _detect_outliers_iqr(working[col])
            if count > 0:
                outliers[col] = count

    total_cells = total_rows * max(len(working.columns), 1)
    missing_ratio = sum(missing.values()) / total_cells if total_cells else 0
    outlier_ratio = sum(outliers.values()) / (total_rows * max(len(NUMERIC_COLUMNS), 1)) if total_rows else 0
    score = max(0, round(100 * (1 - missing_ratio - 0.5 * outlier_ratio), 1))

    return {
        "total_rows": total_rows,
        "missing": missing,
        "outliers": outliers,
        "quality_score": score,
    }


@st.cache_data
def preprocess_data(df, _feature_names):
    """
    Cleans and aligns uploaded dataframe with training features.
    Imputes numeric columns with median and categorical with mode.
    """
    df_processed = df.copy()

    if "customerID" in df_processed.columns:
        df_processed = df_processed.drop("customerID", axis=1)

    if "TotalCharges" in df_processed.columns:
        df_processed["TotalCharges"] = pd.to_numeric(
            df_processed["TotalCharges"], errors="coerce"
        )

    for col in df_processed.columns:
        if df_processed[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                fill = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(fill if pd.notna(fill) else 0)
            else:
                mode = df_processed[col].mode(dropna=True)
                df_processed[col] = df_processed[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    df_processed = pd.get_dummies(df_processed)

    for col in _feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0

    return df_processed[_feature_names]

def get_risk_label(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"
