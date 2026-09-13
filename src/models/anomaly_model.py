from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ANOMALY_FEATURES = [
    "pressure",
    "temperature",
    "tread_depth",
    "pressure_delta",
    "temperature_delta",
    "tread_depth_delta",
    "pressure_roll_std_6",
    "temperature_roll_std_6",
]


def build_anomaly_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ANOMALY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Expected feature columns missing from dataframe: {missing}")
    return df[ANOMALY_FEATURES].copy()


def build_isolation_forest_pipeline(contamination: float = 0.1) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=100,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=1,  
                ),
            ),
        ]
    )


def any_ground_truth_fault(row: pd.Series) -> bool:
    return any(
        row.get(col, "none") != "none"
        for col in ("pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault")
    )


def any_rule_based_flag(row: pd.Series) -> bool:
    return any(
        bool(row.get(col, False))
        for col in (
            "pressure_was_missing",
            "temperature_was_missing",
            "tread_depth_was_missing",
            "pressure_out_of_range",
            "temperature_out_of_range",
            "tread_depth_out_of_range",
        )
    )