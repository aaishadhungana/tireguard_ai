from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.db.session import create_all_tables, drop_all_tables, get_session
from src.db.repository import bulk_insert_telemetry, upsert_tire, upsert_vehicle
from src.utils.logger import get_logger

log = get_logger(__name__)

BATCH_SIZE = 5000


def _find_latest_processed_file():
    candidates = sorted(settings.data_processed_dir.glob("tire_telemetry_processed_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No processed datasets found in {settings.data_processed_dir}.")
    return candidates[-1]


def load_dataset(input_path, reset=False):
    if reset:
        log.warning("Dropping all existing tables before reload (--reset was passed).")
        drop_all_tables()

    log.info("Creating tables if they don't already exist...")
    create_all_tables()

    log.info("Reading %s", input_path)
    df = pd.read_parquet(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    vehicle_ids = df["vehicle_id"].unique()
    tire_to_vehicle = df.drop_duplicates("tire_id").set_index("tire_id")["vehicle_id"].to_dict()

    log.info("Inserting %d vehicles and %d tires...", len(vehicle_ids), len(tire_to_vehicle))
    with get_session() as session:
        for vehicle_id in vehicle_ids:
            upsert_vehicle(session, vehicle_id)
        for tire_id, vehicle_id in tire_to_vehicle.items():
            upsert_tire(session, tire_id, vehicle_id)

    log.info("Inserting %d telemetry rows in batches of %d...", len(df), BATCH_SIZE)
    telemetry_cols = [
        "tire_id", "timestamp", "pressure", "temperature", "speed", "load",
        "tread_depth", "mileage", "braking_events", "acceleration",
        "road_type", "weather", "failure", "failure_type",
    ]
    records = df[telemetry_cols].to_dict(orient="records")

    inserted = 0
    with get_session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            inserted += bulk_insert_telemetry(session, batch)
            if inserted % (BATCH_SIZE * 4) == 0 or inserted == len(records):
                log.info("Inserted %d / %d telemetry rows...", inserted, len(records))

    log.info(
        "Load complete: %d vehicles, %d tires, %d telemetry rows.",
        len(vehicle_ids), len(tire_to_vehicle), inserted,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Load processed telemetry into the database.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--reset", action="store_true", help="Drop all tables before loading.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input) if args.input else _find_latest_processed_file()
    load_dataset(input_path, reset=args.reset)


if __name__ == "__main__":
    main()