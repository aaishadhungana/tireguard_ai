from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from src.config.settings import settings
from src.simulator.tire_simulator import SimulatorConfig, TireTelemetrySimulator
from src.utils.logger import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tire telemetry data.")
    parser.add_argument("--vehicles", type=int, default=settings.sim_num_vehicles)
    parser.add_argument("--tires-per-vehicle", type=int, default=settings.sim_tires_per_vehicle)
    parser.add_argument("--days", type=int, default=settings.sim_duration_days)
    parser.add_argument(
        "--interval-min", type=int, default=settings.sim_sampling_interval_min
    )
    parser.add_argument("--failure-rate", type=float, default=settings.sim_failure_rate)
    parser.add_argument("--sensor-fault-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.ensure_directories()

    config = SimulatorConfig(
        num_vehicles=args.vehicles,
        tires_per_vehicle=args.tires_per_vehicle,
        duration_days=args.days,
        sampling_interval_min=args.interval_min,
        failure_rate=args.failure_rate,
        sensor_fault_rate=args.sensor_fault_rate,
        random_seed=args.seed,
    )

    log.info(
        "Generating synthetic telemetry: %d vehicles x %d tires, %d days @ %d min intervals "
        "(seed=%d, failure_rate=%.3f)",
        config.num_vehicles,
        config.tires_per_vehicle,
        config.duration_days,
        config.sampling_interval_min,
        config.random_seed,
        config.failure_rate,
    )

    simulator = TireTelemetrySimulator(config)
    df = simulator.generate()

    log.info("Generated %d rows across %d unique tires", len(df), df["tire_id"].nunique())
    log.info(
        "Failure rows: %d (%.3f%% of rows) | Failure types: %s",
        int(df["failure"].sum()),
        100 * df["failure"].mean(),
        df.loc[df["failure"] == 1, "failure_type"].value_counts().to_dict(),
    )
    fault_cols = ["pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault"]
    for col in fault_cols:
        log.info("%s distribution: %s", col, df[col].value_counts().to_dict())

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_name = args.output_name or f"tire_telemetry_synthetic_{timestamp_str}.csv"
    output_path = settings.data_raw_dir / output_name

    df.to_csv(output_path, index=False)
    log.info("Wrote dataset to %s", output_path)

    metadata = {
        "is_synthetic": True,
        "generator": "src.simulator.tire_simulator.TireTelemetrySimulator",
        "generated_at_utc": timestamp_str,
        "config": vars(config) if hasattr(config, "__dict__") else config.__dict__,
        "row_count": len(df),
        "note": (
            "All values in this file are synthetically generated from simplified, "
            "illustrative tire-engineering relationships. Not derived from real "
            "manufacturer or fleet data."
        ),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()