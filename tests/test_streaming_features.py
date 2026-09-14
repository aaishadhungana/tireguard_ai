from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import ROLLING_WINDOW as BATCH_ROLLING_WINDOW
from src.data.features import engineer_features
from src.streaming.feature_buffer import ROLLING_WINDOW, StreamingFeatureBuilder, TireFeatureBuffer


def test_rolling_window_constants_match():
    """If these ever drift apart, streaming and batch features would
    silently disagree -- this is a canary for that specific mistake."""
    assert ROLLING_WINDOW == BATCH_ROLLING_WINDOW


def test_first_reading_has_null_lag_features():
    buffer = TireFeatureBuffer()
    features = buffer.add_reading(pressure=32.0, temperature=25.0, tread_depth=8.0, braking_events=0)
    assert features["pressure_prev"] is None
    assert features["pressure_delta"] is None
    assert features["pressure_roll_std_6"] is None  


def test_reading_index_increments():
    buffer = TireFeatureBuffer()
    f1 = buffer.add_reading(32.0, 25.0, 8.0, 0)
    f2 = buffer.add_reading(31.9, 25.5, 7.9, 1)
    assert f1["tire_reading_index"] == 0
    assert f2["tire_reading_index"] == 1


def test_multiple_tires_do_not_interfere():
    builder = StreamingFeatureBuilder()
    builder.process_reading("TIRE-A", 32.0, 25.0, 8.0, 0)
    f = builder.process_reading("TIRE-B", 10.0, 90.0, 1.0, 5)
    assert f["pressure_prev"] is None
    assert f["tire_reading_index"] == 0
    assert builder.known_tire_count() == 2


def test_streaming_features_match_batch_features_exactly():
    rng = np.random.default_rng(7)
    n = 15
    pressures = 32.0 - np.cumsum(rng.uniform(0, 0.3, n))
    temperatures = 25.0 + np.cumsum(rng.normal(0, 1, n))
    treads = 8.0 - np.cumsum(rng.uniform(0, 0.05, n))
    braking = rng.integers(0, 3, n)

    batch_df = pd.DataFrame(
        {
            "tire_id": ["T1"] * n,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="30min"),
            "pressure": pressures,
            "temperature": temperatures,
            "tread_depth": treads,
            "braking_events": braking,
            "load": [500.0] * n,
        }
    )
    batch_result = engineer_features(batch_df)

    buffer = TireFeatureBuffer()
    streaming_rows = []
    for i in range(n):
        streaming_rows.append(
            buffer.add_reading(
                pressure=pressures[i],
                temperature=temperatures[i],
                tread_depth=treads[i],
                braking_events=int(braking[i]),
            )
        )
    streaming_result = pd.DataFrame(streaming_rows)

    shared_columns = [
        "pressure_prev",
        "pressure_delta",
        f"pressure_roll_mean_{ROLLING_WINDOW}",
        f"pressure_roll_std_{ROLLING_WINDOW}",
        "temperature_prev",
        "temperature_delta",
        f"temperature_roll_mean_{ROLLING_WINDOW}",
        f"temperature_roll_std_{ROLLING_WINDOW}",
        "tread_depth_prev",
        "tread_depth_delta",
        "tire_reading_index",
        f"braking_events_roll_sum_{ROLLING_WINDOW}",
    ]

    for col in shared_columns:
        batch_values = batch_result[col].to_numpy(dtype=float)
        streaming_values = streaming_result[col].to_numpy(dtype=float)
        np.testing.assert_allclose(
            streaming_values,
            batch_values,
            equal_nan=True,
            rtol=1e-9,
            err_msg=f"Mismatch in column '{col}' between streaming and batch feature computation",
        )