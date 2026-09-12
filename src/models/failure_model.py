"""
Failure prediction models: feature selection, preprocessing, and model
definitions.

FEATURE EXCLUSIONS (each one is a deliberate decision, not an oversight):
- tire_id, vehicle_id: including these would let a model with only 60
  tires memorize "which specific tire fails" rather than learning
  generalizable physical patterns. Excluded entirely.
- pressure_true, temperature_true, tread_depth_true: these are
  simulator ground-truth values a real system never observes (only the
  sensor's possibly-faulted reading is observable). Using them would
  make the "prediction" trivial and meaningless outside the simulator.
- maintenance_history: caught during Milestone 3 development — this
  column is incremented in the SAME row as the failure it results from
  (see src/simulator/tire_simulator.py), so it directly encodes the
  target for that row. Using it as a feature is training a model to
  detect its own label. Excluded.
- timestamp: not intrinsically predictive on its own (the *_prev,
  *_delta, and rolling features already capture temporal dynamics);
  including raw timestamp risks the model keying on calendar-time
  artifacts of this specific 30-day simulation run rather than
  physical patterns.
- failure, failure_type: the targets themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "pressure",
    "temperature",
    "tread_depth",
    "speed",
    "load",
    "braking_events",
    "acceleration",
    "mileage",
    "pressure_prev",
    "pressure_delta",
    "temperature_prev",
    "temperature_delta",
    "tread_depth_prev",
    "tread_depth_delta",
    "pressure_roll_mean_6",
    "pressure_roll_std_6",
    "temperature_roll_mean_6",
    "temperature_roll_std_6",
    "braking_events_roll_sum_6",
    "tire_reading_index",
    "load_to_nominal_ratio",
    "pressure_deficit",
]

BOOLEAN_FEATURES = [
    "pressure_was_missing",
    "temperature_was_missing",
    "tread_depth_was_missing",
    "pressure_out_of_range",
    "temperature_out_of_range",
    "tread_depth_out_of_range",
]

CATEGORICAL_FEATURES = [
    "road_type",
    "weather",
    "pressure_sensor_fault",
    "temperature_sensor_fault",
    "tread_sensor_fault",
]

TARGET_COLUMN = "failure"

EXCLUDED_COLUMNS_RATIONALE = {
    "tire_id": "identity leakage risk with only 60 tires",
    "vehicle_id": "identity leakage risk",
    "pressure_true": "ground truth not observable by a real system",
    "temperature_true": "ground truth not observable by a real system",
    "tread_depth_true": "ground truth not observable by a real system",
    "maintenance_history": "incremented in the same row as its own failure label — direct leakage",
    "timestamp": "temporal dynamics already captured via lag/rolling features",
    "failure_type": "only known for rows where failure=1; not usable as an input feature",
    "is_synthetic": "constant column, no predictive value",
}


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) using only the approved feature set. Raises if any
    excluded/leaky column is accidentally requested downstream — the
    approved lists above are the single source of truth."""
    missing = [
        c
        for c in NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Expected feature columns missing from dataframe: {missing}")

    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    # Median imputation is required here, not optional: every tire's
    # first reading has NaN lag/delta features (nothing to compute a
    # delta from yet), and rolling std is undefined for a window of a
    # single value. This was caught by running training on the real
    # dataset — the synthetic test fixture didn't happen to include
    # first-row NaNs, so unit tests alone missed it.
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # required given ~0.07% positive rate
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,  # class imbalance handling
                    eval_metric="aucpr",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)