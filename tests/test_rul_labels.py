from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.rul_labels import compute_rul_labels, rul_labeling_summary
from src.simulator.physics import LEGAL_MIN_TREAD_DEPTH_MM


def make_df(tire_id: str, tread_values: list, mileage_values: list) -> pd.DataFrame:
    n = len(tread_values)
    return pd.DataFrame(
        {
            "tire_id": [tire_id] * n,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "mileage": mileage_values,
            "tread_depth_true": tread_values,
        }
    )


def test_simple_monotonic_wear_produces_correct_rul():
    tread = [10.0, 8.0, 6.0, 4.0, 1.6]
    mileage = [0, 100, 200, 300, 400]
    df = make_df("T1", tread, mileage)

    labeled = compute_rul_labels(df)

    expected_rul = [400, 300, 200, 100, 0]
    assert labeled["rul_mileage"].tolist() == pytest.approx(expected_rul)
    assert not labeled["rul_censored"].any()


def test_rul_decreases_monotonically_within_a_cycle():
    tread = [9.5, 7.2, 5.0, 3.1, 1.5]
    mileage = [0, 120, 250, 390, 500]
    df = make_df("T2", tread, mileage)

    labeled = compute_rul_labels(df)
    rul = labeled["rul_mileage"].tolist()

    assert all(rul[i] > rul[i + 1] for i in range(len(rul) - 1))


def test_cycle_never_reaching_threshold_is_censored():
    tread = [10.0, 9.0, 8.0, 7.0]
    mileage = [0, 100, 200, 300]
    df = make_df("T3", tread, mileage)

    labeled = compute_rul_labels(df)

    assert labeled["rul_censored"].all()
    assert labeled["rul_mileage"].isna().all()


def test_reset_correctly_segments_cycles():
    tread = [5.0, 3.0, 1.5, 10.0, 9.0, 8.0]
    mileage = [0, 100, 200, 250, 350, 450]
    df = make_df("T4", tread, mileage)

    labeled = compute_rul_labels(df)

    assert labeled["rul_mileage"].iloc[0] == pytest.approx(200)
    assert labeled["rul_mileage"].iloc[1] == pytest.approx(100)
    assert labeled["rul_mileage"].iloc[2] == pytest.approx(0)
    assert not labeled["rul_censored"].iloc[0:3].any()

    assert labeled["rul_censored"].iloc[3:6].all()
    assert labeled["rul_mileage"].iloc[3:6].isna().all()


def test_multiple_tires_do_not_interfere():
    df_a = make_df("TIRE-A", [5.0, 3.0, 1.5], [0, 100, 200])
    df_b = make_df("TIRE-B", [10.0, 9.0, 8.0], [0, 50, 100])  # never reaches threshold
    combined = pd.concat([df_a, df_b], ignore_index=True)

    labeled = compute_rul_labels(combined)

    tire_a = labeled[labeled["tire_id"] == "TIRE-A"]
    tire_b = labeled[labeled["tire_id"] == "TIRE-B"]

    assert not tire_a["rul_censored"].any()
    assert tire_b["rul_censored"].all()


def test_small_sensor_noise_is_not_mistaken_for_a_reset():
    tread = [5.0, 5.05, 4.9, 3.0, 1.5]  
    mileage = [0, 50, 100, 150, 200]
    df = make_df("T5", tread, mileage)

    labeled = compute_rul_labels(df)

    assert not labeled["rul_censored"].any()
    assert labeled["rul_mileage"].iloc[0] == pytest.approx(200)


def test_summary_counts_match_actual_data():
    tread = [5.0, 3.0, 1.5, 10.0, 9.0]
    mileage = [0, 100, 200, 250, 350]
    df = make_df("T6", tread, mileage)
    labeled = compute_rul_labels(df)

    summary = rul_labeling_summary(labeled)
    assert summary["total_rows"] == 5
    assert summary["labeled_rows"] == 3
    assert summary["censored_rows"] == 2
    assert summary["censored_fraction"] == pytest.approx(0.4)


def test_rul_never_negative():
    tread = [10.0, 8.0, 6.0, 4.0, 1.6]
    mileage = [0, 100, 200, 300, 400]
    df = make_df("T7", tread, mileage)
    labeled = compute_rul_labels(df)
    assert (labeled.loc[~labeled["rul_censored"], "rul_mileage"] >= 0).all()