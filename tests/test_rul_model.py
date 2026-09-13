from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.rul_model import (
    build_linear_regression_pipeline,
    build_random_forest_rul_pipeline,
    build_rul_feature_matrix,
    build_xgboost_rul_pipeline,
)


def make_rul_df(n_rows: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    censored = np.zeros(n_rows, dtype=bool)
    censored[-10:] = True  

    return pd.DataFrame(
        {
            "tire_id": ["T1"] * n_rows,
            "vehicle_id": ["V1"] * n_rows,
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "pressure": rng.uniform(28, 33, n_rows),
            "temperature": rng.uniform(20, 40, n_rows),
            "tread_depth": rng.uniform(1, 9, n_rows),
            "speed": rng.uniform(0, 120, n_rows),
            "load": rng.uniform(300, 700, n_rows),
            "braking_events": rng.integers(0, 3, n_rows),
            "acceleration": rng.normal(0, 1, n_rows),
            "mileage": np.arange(n_rows) * 30.0,
            "pressure_true": rng.uniform(28, 33, n_rows),
            "temperature_true": rng.uniform(20, 40, n_rows),
            "tread_depth_true": rng.uniform(1, 9, n_rows),
            "rul_mileage": np.where(censored, np.nan, rng.uniform(0, 10000, n_rows)),
            "rul_censored": censored,
            "pressure_prev": rng.uniform(28, 33, n_rows),
            "pressure_delta": rng.normal(0, 0.5, n_rows),
            "temperature_prev": rng.uniform(20, 40, n_rows),
            "temperature_delta": rng.normal(0, 1, n_rows),
            "tread_depth_prev": rng.uniform(1, 9, n_rows),
            "tread_depth_delta": rng.normal(0, 0.1, n_rows),
            "pressure_roll_mean_6": rng.uniform(28, 33, n_rows),
            "pressure_roll_std_6": rng.uniform(0, 1, n_rows),
            "temperature_roll_mean_6": rng.uniform(20, 40, n_rows),
            "temperature_roll_std_6": rng.uniform(0, 2, n_rows),
            "braking_events_roll_sum_6": rng.integers(0, 10, n_rows),
            "tire_reading_index": np.arange(n_rows),
            "load_to_nominal_ratio": rng.uniform(0.5, 1.5, n_rows),
            "pressure_deficit": rng.uniform(0, 5, n_rows),
            "pressure_was_missing": False,
            "temperature_was_missing": False,
            "tread_depth_was_missing": False,
            "pressure_out_of_range": False,
            "temperature_out_of_range": False,
            "tread_depth_out_of_range": False,
            "road_type": rng.choice(["highway", "urban", "rural", "off_road"], n_rows),
            "weather": rng.choice(["clear", "rain", "snow"], n_rows),
            "pressure_sensor_fault": "none",
            "temperature_sensor_fault": "none",
            "tread_sensor_fault": "none",
            "is_synthetic": True,
        }
    )


def test_feature_matrix_excludes_censored_rows():
    df = make_rul_df()
    X, y = build_rul_feature_matrix(df)
    assert len(X) == 40 
    assert not y.isna().any()


def test_feature_matrix_excludes_leaky_columns():
    df = make_rul_df()
    X, y = build_rul_feature_matrix(df)
    leaky_columns = [
        "tire_id", "vehicle_id", "timestamp",
        "pressure_true", "temperature_true", "tread_depth_true",
        "rul_mileage", "rul_censored", "is_synthetic",
    ]
    for col in leaky_columns:
        assert col not in X.columns


def test_raises_when_all_rows_censored():
    df = make_rul_df()
    df["rul_censored"] = True
    with pytest.raises(ValueError, match="No non-censored"):
        build_rul_feature_matrix(df)


def test_raises_when_expected_column_missing():
    df = make_rul_df().drop(columns=["pressure_roll_mean_6"])
    with pytest.raises(ValueError, match="missing"):
        build_rul_feature_matrix(df)


def test_linear_regression_pipeline_fits_and_predicts():
    df = make_rul_df()
    X, y = build_rul_feature_matrix(df)
    model = build_linear_regression_pipeline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_random_forest_rul_pipeline_fits_and_predicts():
    df = make_rul_df()
    X, y = build_rul_feature_matrix(df)
    model = build_random_forest_rul_pipeline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_xgboost_rul_pipeline_fits_and_predicts():
    df = make_rul_df()
    X, y = build_rul_feature_matrix(df)
    model = build_xgboost_rul_pipeline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)