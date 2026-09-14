from __future__ import annotations

import argparse
import time

import pandas as pd
import paho.mqtt.client as mqtt

from src.config.settings import settings
from src.streaming.message_schema import row_to_message
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_BROKER_HOST = "localhost"
DEFAULT_BROKER_PORT = 1883


def _find_latest_raw_file():
    candidates = sorted(settings.data_raw_dir.glob("tire_telemetry_synthetic_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No raw telemetry CSVs found in {settings.data_raw_dir}. "
            f"Run `python -m src.simulator.generate_dataset` first."
        )
    return candidates[-1]


def publish_stream(input_path, broker_host=DEFAULT_BROKER_HOST, broker_port=DEFAULT_BROKER_PORT, messages_per_second=20.0, limit=None):
    log.info("Loading telemetry from %s", input_path)
    df = pd.read_csv(input_path)
    if limit:
        df = df.head(limit)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    log.info("Connecting to MQTT broker at %s:%d", broker_host, broker_port)
    client.connect(broker_host, broker_port)
    client.loop_start()

    delay = 1.0 / messages_per_second if messages_per_second > 0 else 0.0
    published = 0

    try:
        for _, row in df.iterrows():
            message = row_to_message(row.to_dict())
            client.publish(message.topic(), message.to_json(), qos=1)
            published += 1
            if published % 500 == 0:
                log.info("Published %d messages...", published)
            if delay:
                time.sleep(delay)
    finally:
        client.loop_stop()
        client.disconnect()

    log.info("Finished publishing %d messages.", published)
    return published


def parse_args():
    parser = argparse.ArgumentParser(description="Publish telemetry over MQTT.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--broker-host", type=str, default=DEFAULT_BROKER_HOST)
    parser.add_argument("--broker-port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--rate", type=float, default=20.0, help="Messages per second.")
    parser.add_argument("--limit", type=int, default=None, help="Max messages to publish (default: all rows).")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input or _find_latest_raw_file()
    publish_stream(
        input_path,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        messages_per_second=args.rate,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()