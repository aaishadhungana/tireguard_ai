from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.failure_model import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)

TARGET_COLUMN = "failure_type"

FAILURE_TYPE_CLASSES = [
    "structural_degradation",
    "underinflation",
    "overheating",
    "overloading",
    "puncture",
    "sensor_malfunction",
]


def build_failure_type_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filters to failure rows only, then reuses Milestone 3's feature
    exclusion logic. Raises if any expected column is missing, same
    contract as build_feature_matrix in failure_model.py."""
    failure_rows = df[df["failure"] == 1].copy()
    if failure_rows.empty:
        raise ValueError("No failure rows found — nothing to classify.")

    missing = [
        c
        for c in NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
        if c not in failure_rows.columns
    ]
    if missing:
        raise ValueError(f"Expected feature columns missing from dataframe: {missing}")

    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    X = failure_rows[feature_cols].copy()
    y = failure_rows[TARGET_COLUMN].astype(str)
    return X, y


def build_random_forest_type_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    class_weight="balanced",  
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost_type_pipeline(num_classes: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=num_classes,
                    n_estimators=200,
                    max_depth=3,  
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )