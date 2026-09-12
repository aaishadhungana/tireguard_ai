from __future__ import annotations

import pandas as pd

ROLLING_WINDOW = 6  # 6 steps = 3 hours at 30-min sampling interval


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tire_id", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("tire_id", group_keys=False)

    # ---- Lag / delta features (previous reading only) ----
    for col in ("pressure", "temperature", "tread_depth"):
        df[f"{col}_prev"] = grouped[col].shift(1)
        df[f"{col}_delta"] = df[col] - df[f"{col}_prev"]

    # ---- Trailing rolling stats (current + past only, never centered) ----
    for col in ("pressure", "temperature"):
        df[f"{col}_roll_mean_{ROLLING_WINDOW}"] = grouped[col].transform(
            lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        df[f"{col}_roll_std_{ROLLING_WINDOW}"] = grouped[col].transform(
            lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).std()
        )

    df[f"braking_events_roll_sum_{ROLLING_WINDOW}"] = grouped["braking_events"].transform(
        lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).sum()
    )

    # ---- Per-tire elapsed time / reading index (age of the tire in the
    # dataset, not calendar time) 
    df["tire_reading_index"] = grouped.cumcount()

    # ---- Derived domain ratios (no temporal dependency, safe as-is) ----
    df["load_to_nominal_ratio"] = df["load"] / 500.0  # NOMINAL_LOAD_KG from physics.py
    df["pressure_deficit"] = (32.0 - df["pressure"]).clip(lower=0.0)  # NOMINAL_PRESSURE_PSI

    return df


LEAKAGE_SAFE_FEATURE_COLUMNS = [
    "pressure_prev",
    "pressure_delta",
    "temperature_prev",
    "temperature_delta",
    "tread_depth_prev",
    "tread_depth_delta",
    f"pressure_roll_mean_{ROLLING_WINDOW}",
    f"pressure_roll_std_{ROLLING_WINDOW}",
    f"temperature_roll_mean_{ROLLING_WINDOW}",
    f"temperature_roll_std_{ROLLING_WINDOW}",
    f"braking_events_roll_sum_{ROLLING_WINDOW}",
    "tire_reading_index",
    "load_to_nominal_ratio",
    "pressure_deficit",
]