from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.data.cleaning import clean_telemetry, cleaning_summary
from src.data.features import engineer_features
from src.data.validation import validate_telemetry
from src.utils.logger import get_logger

log = get_logger(__name__)


def _find_latest_raw_file() -> Path:
    candidates = sorted(settings.data_raw_dir.glob("tire_telemetry_synthetic_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No raw telemetry CSVs found in {settings.data_raw_dir}. "
            f"Run `python -m src.simulator.generate_dataset` first."
        )
    return candidates[-1]


def run_pipeline(input_path: Path, output_name: str | None = None) -> Path:
    settings.ensure_directories()

    log.info("Reading raw telemetry from %s", input_path)
    df = pd.read_csv(input_path)

    log.info("Validating %d rows...", len(df))
    report = validate_telemetry(df)
    log.info("\n%s", report.summary())
    if not report.is_valid:
        log.error("Validation FAILED — aborting pipeline. Fix the raw data before proceeding.")
        sys.exit(1)

    log.info("Cleaning telemetry...")
    cleaned = clean_telemetry(df)
    summary = cleaning_summary(cleaned)
    log.info("Cleaning summary: %s", summary)

    log.info("Engineering features...")
    featured = engineer_features(cleaned)
    log.info("Feature columns added: %d", len(featured.columns) - len(cleaned.columns))

    output_name = output_name or input_path.stem.replace(
        "tire_telemetry_synthetic", "tire_telemetry_processed"
    ) + ".parquet"
    output_path = settings.data_processed_dir / output_name
    featured.to_parquet(output_path, index=False)
    log.info("Wrote processed dataset to %s (%d rows, %d columns)", output_path, len(featured), len(featured.columns))

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data engineering pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw CSV. Defaults to the most recent file in data/raw/.",
    )
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else _find_latest_raw_file()
    run_pipeline(input_path, args.output_name)


if __name__ == "__main__":
    main()