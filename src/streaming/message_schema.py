from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

TELEMETRY_TOPIC_PREFIX = "tireguard/telemetry"

REQUIRED_FIELDS = {
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
}


@dataclass
class TelemetryMessage:
    tire_id: str
    vehicle_id: str
    timestamp: str
    pressure: float
    temperature: float
    speed: float
    load: float
    tread_depth: float
    mileage: float
    braking_events: int
    acceleration: float
    road_type: str
    weather: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(payload: str) -> "TelemetryMessage":
        data = json.loads(payload)
        missing = REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(f"Telemetry message missing required fields: {sorted(missing)}")
        return TelemetryMessage(**{k: data[k] for k in TelemetryMessage.__dataclass_fields__})

    def topic(self) -> str:
        return f"{TELEMETRY_TOPIC_PREFIX}/{self.vehicle_id}/{self.tire_id}"


def row_to_message(row: dict) -> TelemetryMessage:
    """Builds a TelemetryMessage from a raw simulator/dataset row (a
    dict with at least REQUIRED_FIELDS present)."""
    timestamp = row["timestamp"]
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()
    return TelemetryMessage(
        tire_id=row["tire_id"],
        vehicle_id=row["vehicle_id"],
        timestamp=str(timestamp),
        pressure=float(row["pressure"]),
        temperature=float(row["temperature"]),
        speed=float(row["speed"]),
        load=float(row["load"]),
        tread_depth=float(row["tread_depth"]),
        mileage=float(row["mileage"]),
        braking_events=int(row["braking_events"]),
        acceleration=float(row["acceleration"]),
        road_type=str(row["road_type"]),
        weather=str(row["weather"]),
    )