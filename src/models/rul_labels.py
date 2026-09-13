from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.physics import LEGAL_MIN_TREAD_DEPTH_MM

RESET_JUMP_THRESHOLD_MM = 1.0


def _segment_cycles(tread_series: pd.Series) -> np.ndarray:
    diffs = tread_series.diff().fillna(0.0)
    is_reset = diffs > RESET_JUMP_THRESHOLD_MM
    return is_reset.cumsum().to_numpy()


def compute_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tire_id", "timestamp"]).reset_index(drop=True)

    rul_values = np.full(len(df), np.nan)
    censored = np.zeros(len(df), dtype=bool)

    for tire_id, group in df.groupby("tire_id", sort=False):
        idx = group.index.to_numpy()
        tread = group["tread_depth_true"].to_numpy()
        mileage = group["mileage"].to_numpy()

        cycle_ids = _segment_cycles(group["tread_depth_true"])

        for cycle in np.unique(cycle_ids):
            cycle_mask = cycle_ids == cycle
            cycle_positions = np.where(cycle_mask)[0]
            cycle_tread = tread[cycle_positions]
            cycle_mileage = mileage[cycle_positions]

            below_threshold = np.where(cycle_tread <= LEGAL_MIN_TREAD_DEPTH_MM)[0]

            if len(below_threshold) == 0:
                for pos in cycle_positions:
                    censored[idx[pos]] = True
                continue

            wear_out_pos = below_threshold[0]
            wear_out_mileage = cycle_mileage[wear_out_pos]

            for local_i, pos in enumerate(cycle_positions):
                rul_values[idx[pos]] = wear_out_mileage - cycle_mileage[local_i]

    df["rul_mileage"] = rul_values
    df["rul_censored"] = censored
    return df


def rul_labeling_summary(df: pd.DataFrame) -> dict:
    """Concrete counts for reporting — never asserted from memory."""
    total = len(df)
    censored = int(df["rul_censored"].sum())
    labeled = total - censored
    return {
        "total_rows": total,
        "labeled_rows": labeled,
        "censored_rows": censored,
        "censored_fraction": round(censored / total, 4) if total else None,
        "rul_mean_mileage": round(float(df.loc[~df["rul_censored"], "rul_mileage"].mean()), 2) if labeled else None,
        "rul_max_mileage": round(float(df.loc[~df["rul_censored"], "rul_mileage"].max()), 2) if labeled else None,
    }