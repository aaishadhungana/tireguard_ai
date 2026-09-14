from __future__ import annotations

import json

import pytest

from src.streaming.message_schema import REQUIRED_FIELDS, TelemetryMessage, row_to_message


def make_valid_row(**overrides):
    base = dict(
        tire_id="V-0001-T0",
        vehicle_id="V-0001",
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


def test_row_to_message_round_trip():
    row = make_valid_row()
    message = row_to_message(row)
    json_payload = message.to_json()
    reconstructed = TelemetryMessage.from_json(json_payload)
    assert reconstructed == message


def test_from_json_raises_on_missing_fields():
    incomplete = json.dumps({"tire_id": "T1", "vehicle_id": "V1"})
    with pytest.raises(ValueError, match="missing required fields"):
        TelemetryMessage.from_json(incomplete)


def test_topic_includes_vehicle_and_tire_id():
    row = make_valid_row(vehicle_id="V-0042", tire_id="V-0042-T2")
    message = row_to_message(row)
    topic = message.topic()
    assert "V-0042" in topic
    assert "V-0042-T2" in topic


def test_row_to_message_converts_types_correctly():
    row = make_valid_row(pressure="32.5", braking_events="2")
    message = row_to_message(row)
    assert isinstance(message.pressure, float)
    assert isinstance(message.braking_events, int)
    assert message.pressure == 32.5
    assert message.braking_events == 2


def test_required_fields_matches_dataclass_fields():
    dataclass_fields = set(TelemetryMessage.__dataclass_fields__.keys())
    assert REQUIRED_FIELDS == dataclass_fields