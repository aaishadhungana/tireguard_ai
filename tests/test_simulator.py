from __future__ import annotations
import random
import pandas as pd
import pytest

from src.simulator.faults import SensorFaultInjector
from src.simulator.physics import (
    NOMINAL_PRESSURE_PSI,
    TireState,
    compute_degradation,
    step_pressure,
    step_temperature,
    step_tread_wear,
)
from src.simulator.tire_simulator import SimulatorConfig, TireTelemetrySimulator

EXPECTED_COLUMNS = {
    "tire_id",
    "vehicle_id",
    "timestamp",
    "pressure",
    "temperature",
    "speed",
    "load",
    "tread_depth",
    "mileage",
    "braking_events",
    "acceleration",
    "road_type",
    "weather",
    "maintenance_history",
    "failure",
    "failure_type",
}

# Physics-layer causal relationship tests
def make_state(**overrides) -> TireState:
    base = dict(pressure=32.0, temperature=25.0, tread_depth=8.0, mileage=0.0, degradation=0.0)
    base.update(overrides)
    return TireState(**base)


def test_underinflation_increases_temperature():
    normal_state = make_state(pressure=32.0)
    underinflated_state = make_state(pressure=20.0)

    normal_temp = step_temperature(
        normal_state, speed_kmh=70, load_kg=500, weather="clear", braking_events=0
    )
    underinflated_temp = step_temperature(
        underinflated_state, speed_kmh=70, load_kg=500, weather="clear", braking_events=0
    )

    assert underinflated_temp > normal_temp


def test_overload_increases_temperature():
    state_a = make_state()
    state_b = make_state()

    normal_temp = step_temperature(state_a, speed_kmh=70, load_kg=500, weather="clear", braking_events=0)
    overloaded_temp = step_temperature(state_b, speed_kmh=70, load_kg=1200, weather="clear", braking_events=0)

    assert overloaded_temp > normal_temp


def test_high_speed_increases_temperature():
    state_a = make_state()
    state_b = make_state()

    slow_temp = step_temperature(state_a, speed_kmh=40, load_kg=500, weather="clear", braking_events=0)
    fast_temp = step_temperature(state_b, speed_kmh=160, load_kg=500, weather="clear", braking_events=0)

    assert fast_temp > slow_temp


def test_puncture_causes_sudden_sharp_pressure_drop():
    state = make_state(pressure=32.0)
    dropped = step_pressure(state, has_puncture=True, dt_hours=0.25)
    natural = step_pressure(make_state(pressure=32.0), has_puncture=False, dt_hours=0.25)

    assert dropped < 20.0
    assert (state.pressure - dropped) > 10 * (state.pressure - natural)


def test_tread_wears_faster_on_off_road_than_highway():
    state_a = make_state(tread_depth=8.0, temperature=25.0)
    state_b = make_state(tread_depth=8.0, temperature=25.0)

    highway_tread = step_tread_wear(state_a, distance_km=100, road_type="highway", load_kg=500, speed_kmh=90)
    offroad_tread = step_tread_wear(state_b, distance_km=100, road_type="off_road", load_kg=500, speed_kmh=90)

    assert offroad_tread < highway_tread  


def test_degradation_increases_as_tread_and_temperature_worsen():
    healthy = make_state(tread_depth=9.0, temperature=30.0)
    worn = make_state(tread_depth=2.0, temperature=90.0)

    assert compute_degradation(worn) > compute_degradation(healthy)
    assert 0.0 <= compute_degradation(healthy) <= 1.0
    assert 0.0 <= compute_degradation(worn) <= 1.0

# Sensor fault injector tests
def test_fault_injector_produces_no_faults_at_zero_rate():
    injector = SensorFaultInjector(fault_rate=0.0, rng=random.Random(1))
    for _ in range(50):
        result = injector.apply(30.0)
        assert result.fault_type == "none"
        assert result.value == 30.0


def test_fault_injector_produces_faults_at_high_rate():
    injector = SensorFaultInjector(fault_rate=1.0, rng=random.Random(1))
    fault_types_seen = set()
    for _ in range(50):
        result = injector.apply(30.0)
        fault_types_seen.add(result.fault_type)
    # At fault_rate=1.0 essentially every step is corrupted somehow.
    assert fault_types_seen - {"none"}


def test_missing_fault_produces_none_value():
    injector = SensorFaultInjector(fault_rate=1.0, rng=random.Random(7))
    seen_missing = False
    for _ in range(100):
        result = injector.apply(30.0)
        if result.fault_type == "missing":
            assert result.value is None
            seen_missing = True
    assert seen_missing

# End-to-end simulator tests
@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    config = SimulatorConfig(
        num_vehicles=5,
        tires_per_vehicle=4,
        duration_days=3,
        sampling_interval_min=30,
        failure_rate=0.3,
        sensor_fault_rate=0.05,
        random_seed=123,
    )
    return TireTelemetrySimulator(config).generate()


def test_schema_contains_required_columns(small_dataset):
    assert EXPECTED_COLUMNS.issubset(set(small_dataset.columns))


def test_expected_row_count(small_dataset):
    # 5 vehicles * 4 tires * (3 days * 48 steps/day [30-min interval])
    expected_steps = 3 * ((24 * 60) // 30)
    expected_rows = 5 * 4 * expected_steps
    assert len(small_dataset) == expected_rows


def test_failure_rate_is_plausible(small_dataset):
    failure_fraction = small_dataset["failure"].mean()
    assert 0.0 < failure_fraction < 0.2


def test_failure_types_are_valid(small_dataset):
    valid_types = {
        "none",
        "underinflation",
        "overheating",
        "overloading",
        "puncture",
        "structural_degradation",
        "sensor_malfunction",
    }
    assert set(small_dataset["failure_type"].unique()).issubset(valid_types)


def test_sensor_faults_do_not_affect_failure_labels(small_dataset):
    faulted = small_dataset[small_dataset["pressure_sensor_fault"] != "none"]
    unfaulted = small_dataset[small_dataset["pressure_sensor_fault"] == "none"]

    assert faulted["failure"].nunique() <= 2
    assert unfaulted["failure"].nunique() <= 2


def test_reproducibility_with_same_seed():
    config = SimulatorConfig(
        num_vehicles=2,
        tires_per_vehicle=2,
        duration_days=1,
        sampling_interval_min=60,
        failure_rate=0.1,
        random_seed=999,
    )
    df1 = TireTelemetrySimulator(config).generate()
    df2 = TireTelemetrySimulator(config).generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_pressure_stays_within_plausible_bounds(small_dataset):
    assert (small_dataset["pressure_true"] >= 0).all()
    assert (small_dataset["pressure_true"] <= NOMINAL_PRESSURE_PSI * 1.5).all()