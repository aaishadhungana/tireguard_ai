from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.data_split import (
    tire_group_kfold,
    tire_level_train_test_split,
    verify_no_tire_leakage,
)
from src.models.failure_model import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    compute_scale_pos_weight,
)


def make_processed_df(n_tires: int = 20, rows_per_tire: int = 30, seed: int = 42) -> pd.DataFrame:
    """Builds a minimal dataframe with every column build_feature_matrix
    expects, plus tire_id/failure/maintenance_history, so tests don't
    depend on running the full simulator+pipeline."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tires):
        tire_id = f"TIRE-{t:03d}"
        fails_at = rng.integers(rows_per_tire // 2, rows_per_tire) if rng.random() < 0.5 else None
        maintenance = 0
        for i in range(rows_per_tire):
            is_failure = int(fails_at is not None and i == fails_at)
            if is_failure:
                maintenance += 1
            rows.append(
                {
                    "tire_id": tire_id,
                    "vehicle_id": f"VEH-{t // 4:03d}",
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                    "pressure": rng.uniform(28, 33),
                    "temperature": rng.uniform(20, 40),
                    "tread_depth": rng.uniform(2, 9),
                    "speed": rng.uniform(0, 120),
                    "load": rng.uniform(300, 700),
                    "braking_events": rng.integers(0, 3),
                    "acceleration": rng.normal(0, 1),
                    "mileage": i * 30.0,
                    "pressure_true": rng.uniform(28, 33),
                    "temperature_true": rng.uniform(20, 40),
                    "tread_depth_true": rng.uniform(2, 9),
                    "maintenance_history": maintenance,
                    "failure": is_failure,
                    "failure_type": "structural_degradation" if is_failure else "none",
                    "pressure_prev": rng.uniform(28, 33),
                    "pressure_delta": rng.normal(0, 0.5),
                    "temperature_prev": rng.uniform(20, 40),
                    "temperature_delta": rng.normal(0, 1),
                    "tread_depth_prev": rng.uniform(2, 9),
                    "tread_depth_delta": rng.normal(0, 0.1),
                    "pressure_roll_mean_6": rng.uniform(28, 33),
                    "pressure_roll_std_6": rng.uniform(0, 1),
                    "temperature_roll_mean_6": rng.uniform(20, 40),
                    "temperature_roll_std_6": rng.uniform(0, 2),
                    "braking_events_roll_sum_6": rng.integers(0, 10),
                    "tire_reading_index": i,
                    "load_to_nominal_ratio": rng.uniform(0.5, 1.5),
                    "pressure_deficit": rng.uniform(0, 5),
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
            if is_failure:
                fails_at = None  # only one scripted failure per tire in this fixture
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Group split leakage tests
# ---------------------------------------------------------------------

def test_train_test_split_has_no_tire_overlap():
    df = make_processed_df()
    train_df, test_df = tire_level_train_test_split(df, test_size=0.25, random_state=1)

    train_tires = set(train_df["tire_id"])
    test_tires = set(test_df["tire_id"])
    assert train_tires.isdisjoint(test_tires)
    assert len(train_df) + len(test_df) == len(df)


def test_group_kfold_has_no_tire_overlap_in_any_fold():
    df = make_processed_df()
    for train_idx, test_idx in tire_group_kfold(df, n_splits=5):
        assert verify_no_tire_leakage(df, train_idx, test_idx)


def test_group_kfold_covers_every_row_exactly_once_as_test():
    df = make_processed_df(n_tires=20, rows_per_tire=10)
    all_test_indices = []
    for _, test_idx in tire_group_kfold(df, n_splits=5):
        all_test_indices.extend(test_idx.tolist())

    assert sorted(all_test_indices) == list(range(len(df)))


# ---------------------------------------------------------------------
# Feature exclusion tests — the core leakage-prevention claims
# ---------------------------------------------------------------------

def test_leaky_columns_are_not_in_feature_matrix():
    df = make_processed_df()
    X, y = build_feature_matrix(df)

    leaky_columns = [
        "tire_id",
        "vehicle_id",
        "timestamp",
        "pressure_true",
        "temperature_true",
        "tread_depth_true",
        "maintenance_history",
        "failure",
        "failure_type",
        "is_synthetic",
    ]
    for col in leaky_columns:
        assert col not in X.columns, f"{col} must never be a model feature (leakage risk)"


def test_feature_matrix_contains_expected_columns():
    df = make_processed_df()
    X, y = build_feature_matrix(df)

    for col in NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES:
        assert col in X.columns
    assert y.name == "failure"
    assert set(y.unique()).issubset({0, 1})


def test_missing_expected_column_raises():
    df = make_processed_df().drop(columns=["pressure_roll_mean_6"])
    with pytest.raises(ValueError, match="missing"):
        build_feature_matrix(df)


# ---------------------------------------------------------------------
# Model pipeline smoke tests
# ---------------------------------------------------------------------

def test_scale_pos_weight_computation():
    y = pd.Series([0] * 90 + [1] * 10)
    weight = compute_scale_pos_weight(y)
    assert weight == pytest.approx(9.0)


def test_scale_pos_weight_handles_zero_positives():
    y = pd.Series([0] * 50)
    assert compute_scale_pos_weight(y) == 1.0


def test_logistic_regression_pipeline_fits_and_predicts():
    df = make_processed_df(n_tires=20, rows_per_tire=20)
    X, y = build_feature_matrix(df)
    model = build_logistic_regression_pipeline()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_random_forest_pipeline_fits_and_predicts():
    df = make_processed_df(n_tires=20, rows_per_tire=20)
    X, y = build_feature_matrix(df)
    model = build_random_forest_pipeline()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert len(proba) == len(X)


def test_xgboost_pipeline_fits_and_predicts():
    df = make_processed_df(n_tires=20, rows_per_tire=20)
    X, y = build_feature_matrix(df)
    weight = compute_scale_pos_weight(y)
    model = build_xgboost_pipeline(weight)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert len(proba) == len(X)


def test_pipeline_handles_unseen_categorical_at_inference():
    """OneHotEncoder(handle_unknown='ignore') must not crash if a new
    category appears at inference time (e.g. a road_type not seen in
    training) — this is a realistic production scenario."""
    df = make_processed_df(n_tires=20, rows_per_tire=20)
    X, y = build_feature_matrix(df)
    model = build_logistic_regression_pipeline()
    model.fit(X, y)

    X_new = X.iloc[[0]].copy()
    X_new["road_type"] = "unseen_road_type"
    proba = model.predict_proba(X_new)[:, 1]
    assert len(proba) == 1


def test_pipeline_handles_first_row_nan_features():
    """Every tire's first reading has NaN lag/delta/rolling-std features
    by construction (see features.py) — this was caught by running
    training on real data, where the synthetic test fixture above
    didn't happen to include first-row NaNs. Regression test for that."""
    df = make_processed_df(n_tires=10, rows_per_tire=15)
    df.loc[0, "pressure_prev"] = np.nan
    df.loc[0, "pressure_delta"] = np.nan
    df.loc[0, "pressure_roll_std_6"] = np.nan
    df.loc[0, "temperature_prev"] = np.nan
    df.loc[0, "temperature_delta"] = np.nan

    X, y = build_feature_matrix(df)
    for model_builder in (build_logistic_regression_pipeline, build_random_forest_pipeline):
        model = model_builder()
        model.fit(X, y)  # must not raise ValueError: Input X contains NaN
        proba = model.predict_proba(X)[:, 1]
        assert not np.isnan(proba).any()