from __future__ import annotations

import pandas as pd

from src.data.validation import PLAUSIBLE_RANGES

SENSOR_VALUE_COLUMNS = ["pressure", "temperature", "tread_depth"]


def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tire_id", "timestamp"]).reset_index(drop=True)

    for col in SENSOR_VALUE_COLUMNS:
        was_missing_col = f"{col}_was_missing"
        df[was_missing_col] = df[col].isna()

        df[col] = df.groupby("tire_id")[col].transform(lambda s: s.ffill().bfill())

    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        flag_col = f"{col}_out_of_range"
        df[flag_col] = (df[col] < low) | (df[col] > high)

    return df


def cleaning_summary(df: pd.DataFrame) -> dict:
    summary: dict = {"row_count": len(df)}
    for col in SENSOR_VALUE_COLUMNS:
        was_missing_col = f"{col}_was_missing"
        if was_missing_col in df.columns:
            summary[f"{col}_imputed_count"] = int(df[was_missing_col].sum())
    for col in PLAUSIBLE_RANGES:
        flag_col = f"{col}_out_of_range"
        if flag_col in df.columns:
            summary[f"{col}_out_of_range_count"] = int(df[flag_col].sum())
    return summary