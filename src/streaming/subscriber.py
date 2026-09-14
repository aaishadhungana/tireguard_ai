from __future__ import annotations

import argparse

import pandas as pd
import paho.mqtt.client as mqtt

from src.data.validation import PLAUSIBLE_RANGES
from src.models.predict import FailurePredictor
from src.streaming.feature_buffer import StreamingFeatureBuilder
from src.streaming.message_schema import TELEMETRY_TOPIC_PREFIX, TelemetryMessage
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_BROKER_HOST = "localhost"
DEFAULT_BROKER_PORT = 1883


def assemble_feature_row(message, computed):
    """Combines the live message + feature_buffer's computed history-
    dependent features into the full row shape src.models.failure_model
    expects. Categorical fault columns and *_was_missing are defaulted
    per this module's stated scope limitations above."""
    row = {
        "pressure": message.pressure,
        "temperature": message.temperature,
        "tread_depth": message.tread_depth,
        "speed": message.speed,
        "load": message.load,
        "braking_events": message.braking_events,
        "acceleration": message.acceleration,
        "mileage": message.mileage,
        "road_type": message.road_type,
        "weather": message.weather,
        "pressure_sensor_fault": "none",
        "temperature_sensor_fault": "none",
        "tread_sensor_fault": "none",
        "pressure_was_missing": False,
        "temperature_was_missing": False,
        "tread_depth_was_missing": False,
        "load_to_nominal_ratio": message.load / 500.0,
        "pressure_deficit": max(0.0, 32.0 - message.pressure),
        **computed,
    }
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col in row:
            row[f"{col}_out_of_range"] = not (low <= row[col] <= high)
    return row


class LiveRiskProcessor:
    """Holds the feature builder and predictor across the whole
    subscriber session -- this is the stateful object the MQTT callback
    delegates to."""

    def __init__(self):
        self.feature_builder = StreamingFeatureBuilder()
        self.predictor = FailurePredictor()
        self.messages_processed = 0

    def process_message(self, payload):
        message = TelemetryMessage.from_json(payload)
        computed = self.feature_builder.process_reading(
            tire_id=message.tire_id,
            pressure=message.pressure,
            temperature=message.temperature,
            tread_depth=message.tread_depth,
            braking_events=message.braking_events,
        )
        row = assemble_feature_row(message, computed)
        row_df = pd.DataFrame([row])
        predictions = self.predictor.predict(row_df)
        prediction = predictions[0]

        self.messages_processed += 1
        return {
            "tire_id": message.tire_id,
            "vehicle_id": message.vehicle_id,
            "timestamp": message.timestamp,
            "failure_probability": prediction.failure_probability,
            "risk_level": prediction.risk_level,
        }


def run_subscriber(broker_host=DEFAULT_BROKER_HOST, broker_port=DEFAULT_BROKER_PORT, message_limit=None):
    processor = LiveRiskProcessor()
    log.info("Loaded failure model: %s", processor.predictor.metadata.get("model_type"))

    def on_connect(client, userdata, flags, reason_code, properties=None):
        log.info("Connected to MQTT broker (reason_code=%s)", reason_code)
        client.subscribe(f"{TELEMETRY_TOPIC_PREFIX}/#", qos=1)

    def on_message(client, userdata, msg):
        try:
            result = processor.process_message(msg.payload.decode("utf-8"))
            log_fn = log.warning if result["risk_level"] in ("MEDIUM", "HIGH") else log.info
            log_fn(
                "[%s] tire=%s failure_probability=%.3f risk=%s",
                result["timestamp"], result["tire_id"], result["failure_probability"], result["risk_level"],
            )
        except Exception:
            log.exception("Failed to process message: %s", msg.payload[:200])

        if message_limit and processor.messages_processed >= message_limit:
            client.disconnect()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    log.info("Connecting to MQTT broker at %s:%d", broker_host, broker_port)
    client.connect(broker_host, broker_port)
    client.loop_forever()

    log.info(
        "Subscriber stopped. Processed %d messages across %d tires.",
        processor.messages_processed, processor.feature_builder.known_tire_count(),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Subscribe to live telemetry and run risk scoring.")
    parser.add_argument("--broker-host", type=str, default=DEFAULT_BROKER_HOST)
    parser.add_argument("--broker-port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N messages (default: run forever).")
    return parser.parse_args()


def main():
    args = parse_args()
    run_subscriber(broker_host=args.broker_host, broker_port=args.broker_port, message_limit=args.limit)


if __name__ == "__main__":
    main()