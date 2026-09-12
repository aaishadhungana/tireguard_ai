from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_telemetry, cleaning_summary
from src.data.features import ROLLING_WINDOW, engineer_features
from src.data.validation import validate_telemetry


def make_raw_df(n_rows: int = 20, tire_id: str = "V-0000-T0") -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="30min")
    failure_col = ["none"] * n_rows
    failure_flag = [0] * n_rows
    if n_rows > 0:
        failure_flag[-1] = 1
        failure_col[-1] = "structural_degradation"
    return pd.DataFrame(
        {
            "tire_id": [tire_id] * n_rows,
            "vehicle_id": ["V-0000"] * n_rows,
            "timestamp": timestamps,
            "pressure": np.linspace(32, 28, n_rows),
            "temperature": np.linspace(25, 45, n_rows),
            "speed": [70.0] * n_rows,
            "load": [500.0] * n_rows,
            "tread_depth": np.linspace(9.0, 5.0, n_rows),
            "mileage": np.arange(n_rows) * 35.0,
            "braking_events": [1] * n_rows,
            "acceleration": [0.1] * n_rows,
            "road_type": ["highway"] * n_rows,
            "weather": ["clear"] * n_rows,
            "maintenance_history": [0] * n_rows,
            "failure": failure_flag,
            "failure_type": failure_col,
            "pressure_sensor_fault": ["none"] * n_rows,
            "temperature_sensor_fault": ["none"] * n_rows,
            "tread_sensor_fault": ["none"] * n_rows,
        }
    )

# Validation tests
def test_valid_dataframe_passes_validation():
    df = make_raw_df()
    report = validate_telemetry(df)
    assert report.is_valid
    assert not report.errors


def test_missing_required_column_fails_validation():
    df = make_raw_df().drop(columns=["pressure"])
    report = validate_telemetry(df)
    assert not report.is_valid
    assert any("Missing required columns" in e for e in report.errors)


def test_invalid_road_type_fails_validation():
    df = make_raw_df()
    df.loc[0, "road_type"] = "moon_surface"
    report = validate_telemetry(df)
    assert not report.is_valid
    assert any("road_type" in e for e in report.errors)


def test_duplicate_key_fails_validation():
    df = make_raw_df()
    dup_row = df.iloc[[0]]
    df = pd.concat([df, dup_row], ignore_index=True)
    report = validate_telemetry(df)
    assert not report.is_valid
    assert any("duplicate" in e.lower() for e in report.errors)


def test_out_of_range_values_are_warnings_not_errors():
    df = make_raw_df()
    df.loc[0, "pressure"] = 500.0  
    report = validate_telemetry(df)
    assert report.is_valid  
    assert report.out_of_range_counts["pressure"] == 1
    assert any("pressure" in w for w in report.warnings)


def test_empty_dataframe_fails_validation():
    df = make_raw_df(n_rows=0)
    report = validate_telemetry(df)
    assert not report.is_valid


# Cleaning tests
def test_missing_values_are_imputed_per_tire():
    df = make_raw_df()
    df.loc[5, "pressure"] = np.nan

    cleaned = clean_telemetry(df)

    assert cleaned["pressure"].isna().sum() == 0
    assert cleaned.loc[5, "pressure_was_missing"] == True  # noqa: E712
    assert cleaned.loc[5, "pressure"] == pytest.approx(cleaned.loc[4, "pressure"])


def test_leading_missing_value_is_backfilled():
    df = make_raw_df()
    df.loc[0, "temperature"] = np.nan

    cleaned = clean_telemetry(df)

    assert cleaned["temperature"].isna().sum() == 0
    assert cleaned.loc[0, "temperature"] == pytest.approx(cleaned.loc[1, "temperature"])


def test_imputation_does_not_cross_tire_boundaries():
    df_a = make_raw_df(n_rows=5, tire_id="TIRE-A")
    df_b = make_raw_df(n_rows=5, tire_id="TIRE-B")
    df_b["pressure"] = 99.0  
    df_b.loc[0, "pressure"] = np.nan
    combined = pd.concat([df_a, df_b], ignore_index=True)

    cleaned = clean_telemetry(combined)

    tire_b_first_pressure = cleaned[cleaned["tire_id"] == "TIRE-B"].iloc[0]["pressure"]
    assert tire_b_first_pressure == pytest.approx(99.0)


def test_out_of_range_flag_set_without_altering_value():
    df = make_raw_df()
    df.loc[3, "temperature"] = 999.0

    cleaned = clean_telemetry(df)

    assert cleaned.loc[3, "temperature_out_of_range"] == True  # noqa: E712
    assert cleaned.loc[3, "temperature"] == 999.0  


def test_cleaning_summary_counts_match_actual_flags():
    df = make_raw_df()
    df.loc[2, "pressure"] = np.nan
    df.loc[7, "tread_depth"] = np.nan

    cleaned = clean_telemetry(df)
    summary = cleaning_summary(cleaned)

    assert summary["pressure_imputed_count"] == 1
    assert summary["tread_depth_imputed_count"] == 1
    assert summary["temperature_imputed_count"] == 0


# Feature engineering 
def test_feature_columns_are_added():
    df = clean_telemetry(make_raw_df())
    featured = engineer_features(df)

    assert "pressure_delta" in featured.columns
    assert f"pressure_roll_mean_{ROLLING_WINDOW}" in featured.columns
    assert "tire_reading_index" in featured.columns


def test_first_row_lag_features_are_null():
    df = clean_telemetry(make_raw_df())
    featured = engineer_features(df)

    first_row = featured.iloc[0]
    assert pd.isna(first_row["pressure_prev"])
    assert pd.isna(first_row["pressure_delta"])


def test_rolling_features_unaffected_by_future_rows():
    """The core leakage check: compute features on the full series, then
    again after corrupting only a LATER row, and confirm every EARLIER
    row's engineered features are byte-identical. If rolling/shift were
    accidentally centered or looked ahead, this would fail."""
    df = clean_telemetry(make_raw_df(n_rows=20))
    featured_original = engineer_features(df)

    df_mutated = df.copy()
    df_mutated.loc[15:, "pressure"] = 999.0  
    df_mutated.loc[15:, "temperature"] = 999.0
    featured_mutated = engineer_features(df_mutated)

    cols_to_check = [
        "pressure_prev",
        "pressure_delta",
        f"pressure_roll_mean_{ROLLING_WINDOW}",
        f"pressure_roll_std_{ROLLING_WINDOW}",
        "temperature_prev",
        "temperature_delta",
    ]
    for col in cols_to_check:
        pd.testing.assert_series_equal(
            featured_original.loc[:13, col],
            featured_mutated.loc[:13, col],
            check_names=False,
        )


def test_rolling_features_do_not_cross_tire_boundaries():
    df_a = make_raw_df(n_rows=5, tire_id="TIRE-A")
    df_b = make_raw_df(n_rows=5, tire_id="TIRE-B")
    df_b["pressure"] = 10.0  # distinct baseline
    combined = pd.concat([df_a, df_b], ignore_index=True)
    cleaned = clean_telemetry(combined)

    featured = engineer_features(cleaned)

    tire_b_first_row = featured[featured["tire_id"] == "TIRE-B"].iloc[0]
    assert pd.isna(tire_b_first_row["pressure_prev"])
    assert tire_b_first_row["tire_reading_index"] == 0


def test_reading_index_resets_per_tire():
    df_a = make_raw_df(n_rows=5, tire_id="TIRE-A")
    df_b = make_raw_df(n_rows=3, tire_id="TIRE-B")
    combined = pd.concat([df_a, df_b], ignore_index=True)
    cleaned = clean_telemetry(combined)

    featured = engineer_features(cleaned)

    tire_a_indices = featured[featured["tire_id"] == "TIRE-A"]["tire_reading_index"].tolist()
    tire_b_indices = featured[featured["tire_id"] == "TIRE-B"]["tire_reading_index"].tolist()
    assert tire_a_indices == [0, 1, 2, 3, 4]
    assert tire_b_indices == [0, 1, 2]