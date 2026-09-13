from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_model import (
    ANOMALY_FEATURES,
    any_ground_truth_fault,
    any_rule_based_flag,
    build_anomaly_feature_matrix,
    build_isolation_forest_pipeline,
)


def test_feature_matrix_excludes_ground_truth_and_identity_columns():
    df = pd.DataFrame(
        {col: [0.0, 1.0] for col in ANOMALY_FEATURES}
        | {
            "tire_id": ["T1", "T1"],
            "pressure_true": [30.0, 31.0],
            "temperature_true": [25.0, 26.0],
            "tread_depth_true": [8.0, 8.0],
            "pressure_sensor_fault": ["none", "stuck"],
        }
    )
    X = build_anomaly_feature_matrix(df)
    forbidden = ["tire_id", "pressure_true", "temperature_true", "tread_depth_true", "pressure_sensor_fault"]
    for col in forbidden:
        assert col not in X.columns


def test_raises_on_missing_expected_columns():
    df = pd.DataFrame({"pressure": [30.0]})
    with pytest.raises(ValueError, match="missing"):
        build_anomaly_feature_matrix(df)


def test_any_ground_truth_fault_true_when_any_sensor_faulted():
    row = pd.Series(
        {"pressure_sensor_fault": "none", "temperature_sensor_fault": "stuck", "tread_sensor_fault": "none"}
    )
    assert any_ground_truth_fault(row) is True


def test_any_ground_truth_fault_false_when_all_none():
    row = pd.Series(
        {"pressure_sensor_fault": "none", "temperature_sensor_fault": "none", "tread_sensor_fault": "none"}
    )
    assert any_ground_truth_fault(row) is False


def test_any_rule_based_flag_true_when_any_flag_set():
    row = pd.Series(
        {
            "pressure_was_missing": False,
            "temperature_was_missing": True,
            "tread_depth_was_missing": False,
            "pressure_out_of_range": False,
            "temperature_out_of_range": False,
            "tread_depth_out_of_range": False,
        }
    )
    assert any_rule_based_flag(row) is True


def test_any_rule_based_flag_false_when_all_clear():
    row = pd.Series(
        {
            "pressure_was_missing": False,
            "temperature_was_missing": False,
            "tread_depth_was_missing": False,
            "pressure_out_of_range": False,
            "temperature_out_of_range": False,
            "tread_depth_out_of_range": False,
        }
    )
    assert any_rule_based_flag(row) is False


def test_isolation_forest_flags_injected_outliers_above_chance():
    rng = np.random.default_rng(42)
    n_normal = 200
    n_outliers = 20

    normal = pd.DataFrame(
        {
            "pressure": rng.normal(32, 1, n_normal),
            "temperature": rng.normal(25, 3, n_normal),
            "tread_depth": rng.normal(6, 1, n_normal),
            "pressure_delta": rng.normal(0, 0.2, n_normal),
            "temperature_delta": rng.normal(0, 0.5, n_normal),
            "tread_depth_delta": rng.normal(0, 0.05, n_normal),
            "pressure_roll_std_6": rng.uniform(0, 0.5, n_normal),
            "temperature_roll_std_6": rng.uniform(0, 1, n_normal),
        }
    )
    outliers = pd.DataFrame(
        {
            "pressure": rng.normal(90, 2, n_outliers),  
            "temperature": rng.normal(25, 3, n_outliers),
            "tread_depth": rng.normal(6, 1, n_outliers),
            "pressure_delta": rng.normal(20, 2, n_outliers),  # huge jump
            "temperature_delta": rng.normal(0, 0.5, n_outliers),
            "tread_depth_delta": rng.normal(0, 0.05, n_outliers),
            "pressure_roll_std_6": rng.uniform(0, 0.5, n_outliers),
            "temperature_roll_std_6": rng.uniform(0, 1, n_outliers),
        }
    )
    df = pd.concat([normal, outliers], ignore_index=True)
    is_outlier = np.array([False] * n_normal + [True] * n_outliers)

    contamination = n_outliers / len(df)
    pipeline = build_isolation_forest_pipeline(contamination=contamination)
    pipeline.fit(df[ANOMALY_FEATURES])
    predictions = pipeline.predict(df[ANOMALY_FEATURES])  # -1 = anomaly, 1 = normal
    flagged = predictions == -1

    flagged_rate_among_outliers = flagged[is_outlier].mean()
    flagged_rate_among_normal = flagged[~is_outlier].mean()

    assert flagged_rate_among_outliers > flagged_rate_among_normal
    assert flagged_rate_among_outliers > 0.5  