from __future__ import annotations

import pytest

from src.streaming.message_schema import row_to_message
from src.streaming.subscriber import LiveRiskProcessor, assemble_feature_row


def make_valid_row(**overrides):
    base = dict(
        tire_id="V-9999-T0",
        vehicle_id="V-9999",
        timestamp="2026-01-01T00:00:00",
        pressure=32.0,
        temperature=25.0,
        speed=70.0,
        load=500.0,
        tread_depth=8.0,
        mileage=1000.0,
        braking_events=1,
        acceleration=0.1,
        road_type="highway",
        weather="clear",
    )
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def processor():
    try:
        return LiveRiskProcessor()
    except FileNotFoundError:
        pytest.skip("No trained failure model available -- run train_failure_model.py first.")


def test_process_message_returns_risk_assessment(processor):
    message = row_to_message(make_valid_row())
    result = processor.process_message(message.to_json())

    assert result["tire_id"] == "V-9999-T0"
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_process_message_builds_history_across_calls(processor):
    tire_id = "V-8888-T3"
    msg1 = row_to_message(make_valid_row(tire_id=tire_id, vehicle_id="V-8888", pressure=32.0))
    msg2 = row_to_message(make_valid_row(tire_id=tire_id, vehicle_id="V-8888", pressure=31.5))

    result1 = processor.process_message(msg1.to_json())
    result2 = processor.process_message(msg2.to_json())

    assert result1["tire_id"] == tire_id
    assert result2["tire_id"] == tire_id


def test_assemble_feature_row_defaults_fault_columns_to_none():
    message = row_to_message(make_valid_row())
    row = assemble_feature_row(message, computed={"pressure_prev": None, "pressure_delta": None})
    assert row["pressure_sensor_fault"] == "none"
    assert row["temperature_sensor_fault"] == "none"
    assert row["tread_sensor_fault"] == "none"


def test_assemble_feature_row_flags_out_of_range_pressure():
    message = row_to_message(make_valid_row(pressure=150.0))
    row = assemble_feature_row(message, computed={})
    assert row["pressure_out_of_range"] is True


def test_assemble_feature_row_does_not_flag_normal_pressure():
    message = row_to_message(make_valid_row(pressure=32.0))
    row = assemble_feature_row(message, computed={})
    assert row["pressure_out_of_range"] is False