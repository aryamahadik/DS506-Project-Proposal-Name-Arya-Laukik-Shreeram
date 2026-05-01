from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IDENTIFIER_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"
CATEGORICAL_FEATURES = ["Geography", "Gender"]
ENGINEERED_FEATURES = [
    "Balance_Income_Ratio",
    "Products_Per_Tenure",
    "Balance_Product_Interaction",
    "Senior_Citizen_Age",
]

REMOVED_CORRELATED_FEATURES = [
    "Is_Active",
    "Has_Credit_Card",
    "Age_Squared",
    "Salary_Per_Product",
    "Zero_Balance",
    "High_Balance",
]


def load_raw_data(path: str | Path = "Churn_Modelling.csv") -> pd.DataFrame:
    """Load the collected bank churn CSV from disk."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, normalize categories, and handle missing values."""
    cleaned = df.copy()
    missing = {TARGET_COLUMN, *CATEGORICAL_FEATURES} - set(cleaned.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = cleaned.drop(columns=[col for col in IDENTIFIER_COLUMNS if col in cleaned.columns])
    cleaned = cleaned.drop_duplicates()

    numeric_cols = cleaned.select_dtypes(include="number").columns
    for col in numeric_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in CATEGORICAL_FEATURES:
        mode = cleaned[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        cleaned[col] = cleaned[col].fillna(fill_value).astype(str).str.strip()

    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived modeling features used in the notebook."""
    features = df.copy()
    features["Balance_Income_Ratio"] = features["Balance"] / (features["EstimatedSalary"] + 1)
    features["Products_Per_Tenure"] = features["NumOfProducts"] / (features["Tenure"] + 1)
    features["Balance_Product_Interaction"] = features["Balance"] * features["NumOfProducts"]
    features["Senior_Citizen_Age"] = (features["Age"] > 60).astype(int)
    return features


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        col for col in df.columns if col not in CATEGORICAL_FEATURES + [TARGET_COLUMN]
    ]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipeline(df: pd.DataFrame, model: Any | None = None) -> Pipeline:
    if model is None:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=1,
        )
    return Pipeline([("preprocessor", build_preprocessor(df)), ("model", model)])


def train_evaluate_model(
    df: pd.DataFrame,
    model: Any | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    cleaned = engineer_features(clean_data(df))
    x, y = split_features_target(cleaned)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = build_model_pipeline(cleaned, model=model)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }


def train_fast_baseline(df: pd.DataFrame) -> dict[str, float]:
    model = LogisticRegression(max_iter=500, random_state=42)
    return train_evaluate_model(df, model=model)
