from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.models.failure_model import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)

TARGET_COLUMN = "rul_mileage"


def build_rul_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    labeled_rows = df[~df["rul_censored"]].copy()
    if labeled_rows.empty:
        raise ValueError("No non-censored RUL rows found — nothing to train on.")

    missing = [
        c
        for c in NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
        if c not in labeled_rows.columns
    ]
    if missing:
        raise ValueError(f"Expected feature columns missing from dataframe: {missing}")

    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    X = labeled_rows[feature_cols].copy()
    y = labeled_rows[TARGET_COLUMN].astype(float)
    return X, y


def build_linear_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest_rul_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=8,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_xgboost_rul_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=1,  
                ),
            ),
        ]
    )