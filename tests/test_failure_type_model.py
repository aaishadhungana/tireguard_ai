from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.failure_type_model import (
    FAILURE_TYPE_CLASSES,
    build_failure_type_feature_matrix,
    build_random_forest_type_pipeline,
    build_xgboost_type_pipeline,
)


def make_multiclass_failure_df(seed: int = 42) -> pd.DataFrame:
    """20 tires, each with exactly one failure row of a rotating type,
    plus several healthy rows each — enough diversity to exercise the
    multiclass path without needing the full simulator."""
    rng = np.random.default_rng(seed)
    rows = []
    types = FAILURE_TYPE_CLASSES
    for t in range(20):
        tire_id = f"TIRE-{t:03d}"
        assigned_type = types[t % len(types)]
        for i in range(10):
            is_failure = i == 9
            rows.append(
                {
                    "tire_id": tire_id,
                    "vehicle_id": f"VEH-{t // 4:03d}",
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                    "pressure": rng.uniform(20, 33),
                    "temperature": rng.uniform(20, 90),
                    "tread_depth": rng.uniform(1, 9),
                    "speed": rng.uniform(0, 120),
                    "load": rng.uniform(300, 900),
                    "braking_events": rng.integers(0, 3),
                    "acceleration": rng.normal(0, 1),
                    "mileage": i * 30.0,
                    "pressure_true": rng.uniform(20, 33),
                    "temperature_true": rng.uniform(20, 90),
                    "tread_depth_true": rng.uniform(1, 9),
                    "maintenance_history": int(is_failure),
                    "failure": int(is_failure),
                    "failure_type": assigned_type if is_failure else "none",
                    "pressure_prev": rng.uniform(20, 33),
                    "pressure_delta": rng.normal(0, 0.5),
                    "temperature_prev": rng.uniform(20, 90),
                    "temperature_delta": rng.normal(0, 1),
                    "tread_depth_prev": rng.uniform(1, 9),
                    "tread_depth_delta": rng.normal(0, 0.1),
                    "pressure_roll_mean_6": rng.uniform(20, 33),
                    "pressure_roll_std_6": rng.uniform(0, 1),
                    "temperature_roll_mean_6": rng.uniform(20, 90),
                    "temperature_roll_std_6": rng.uniform(0, 2),
                    "braking_events_roll_sum_6": rng.integers(0, 10),
                    "tire_reading_index": i,
                    "load_to_nominal_ratio": rng.uniform(0.5, 1.8),
                    "pressure_deficit": rng.uniform(0, 15),
                    "pressure_was_missing": False,
                    "temperature_was_missing": False,
                    "tread_depth_was_missing": False,
                    "pressure_out_of_range": False,
                    "temperature_out_of_range": False,
                    "tread_depth_out_of_range": False,
                    "road_type": rng.choice(["highway", "urban", "rural", "off_road"]),
                    "weather": rng.choice(["clear", "rain", "snow"]),
                    "pressure_sensor_fault": "none",
                    "temperature_sensor_fault": "none",
                    "tread_sensor_fault": "none",
                    "is_synthetic": True,
                }
            )
    return pd.DataFrame(rows)


def test_feature_matrix_filters_to_failure_rows_only():
    df = make_multiclass_failure_df()
    X, y = build_failure_type_feature_matrix(df)

    assert len(X) == 20  # exactly one failure row per tire
    assert set(y.unique()) == set(FAILURE_TYPE_CLASSES)


def test_feature_matrix_excludes_leaky_columns():
    df = make_multiclass_failure_df()
    X, y = build_failure_type_feature_matrix(df)

    leaky_columns = [
        "tire_id", "vehicle_id", "timestamp",
        "pressure_true", "temperature_true", "tread_depth_true",
        "maintenance_history", "failure", "failure_type", "is_synthetic",
    ]
    for col in leaky_columns:
        assert col not in X.columns


def test_raises_when_no_failure_rows():
    df = make_multiclass_failure_df()
    df["failure"] = 0
    with pytest.raises(ValueError, match="No failure rows"):
        build_failure_type_feature_matrix(df)


def test_raises_when_expected_column_missing():
    df = make_multiclass_failure_df().drop(columns=["pressure_roll_mean_6"])
    with pytest.raises(ValueError, match="missing"):
        build_failure_type_feature_matrix(df)


def test_random_forest_type_pipeline_fits_and_predicts():
    df = make_multiclass_failure_df()
    X, y = build_failure_type_feature_matrix(df)
    y_encoded = pd.Categorical(y, categories=FAILURE_TYPE_CLASSES).codes

    model = build_random_forest_type_pipeline()
    model.fit(X, y_encoded)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(range(len(FAILURE_TYPE_CLASSES))))


def test_xgboost_type_pipeline_fits_and_predicts():
    df = make_multiclass_failure_df()
    X, y = build_failure_type_feature_matrix(df)
    y_encoded = pd.Categorical(y, categories=FAILURE_TYPE_CLASSES).codes

    model = build_xgboost_type_pipeline(num_classes=len(FAILURE_TYPE_CLASSES))
    model.fit(X, y_encoded)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_pooled_cross_validation_produces_predictions_for_every_row():
    """Verifies the pooling logic in train_failure_type_model doesn't
    silently drop rows — every failure row should get exactly one
    pooled prediction across the full GroupKFold run."""
    from src.models.train_failure_type_model import cross_validate_pooled

    df = make_multiclass_failure_df()
    X, y = build_failure_type_feature_matrix(df)
    groups = df.loc[X.index, "tire_id"]
    y_encoded = pd.Categorical(y, categories=FAILURE_TYPE_CLASSES).codes

    y_true, y_pred, fold_info = cross_validate_pooled(
        build_random_forest_type_pipeline, X, np.array(y_encoded), groups, n_folds=5, is_xgb=False
    )

    assert len(y_true) == len(y_pred)
    total_test_rows = sum(f["n_test"] for f in fold_info)
    assert len(y_true) == total_test_rows
    assert len(y_true) == 20