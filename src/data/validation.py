from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = {
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
    "pressure_sensor_fault",
    "temperature_sensor_fault",
    "tread_sensor_fault",
}

VALID_ROAD_TYPES = {"highway", "urban", "rural", "off_road"}
VALID_WEATHER = {"clear", "rain", "snow", "extreme_heat"}
VALID_SENSOR_FAULTS = {"none", "stuck", "missing", "drift", "spoofed"}

PLAUSIBLE_RANGES = {
    "pressure": (0.0, 100.0),
    "temperature": (-40.0, 250.0),
    "speed": (0.0, 250.0),
    "load": (0.0, 3000.0),
    "tread_depth": (0.0, 15.0),
}


@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    out_of_range_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Validation: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Rows: {self.row_count}",
        ]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_telemetry(df: pd.DataFrame) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return ValidationReport(is_valid=False, errors=errors, row_count=len(df))

    if df.empty:
        errors.append("Dataframe is empty.")
        return ValidationReport(is_valid=False, errors=errors, row_count=0)

    bad_road_types = set(df["road_type"].dropna().unique()) - VALID_ROAD_TYPES
    if bad_road_types:
        errors.append(f"Unexpected road_type values: {sorted(bad_road_types)}")

    bad_weather = set(df["weather"].dropna().unique()) - VALID_WEATHER
    if bad_weather:
        errors.append(f"Unexpected weather values: {sorted(bad_weather)}")

    for col in ("pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault"):
        bad_faults = set(df[col].dropna().unique()) - VALID_SENSOR_FAULTS
        if bad_faults:
            errors.append(f"Unexpected {col} values: {sorted(bad_faults)}")

    dup_key_count = df.duplicated(subset=["tire_id", "timestamp"]).sum()
    if dup_key_count > 0:
        errors.append(f"{dup_key_count} duplicate (tire_id, timestamp) rows found.")

    out_of_range_counts: dict[str, int] = {}
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        out_of_range = ((series < low) | (series > high)).sum()
        out_of_range_counts[col] = int(out_of_range)
        if out_of_range > 0:
            pct = 100 * out_of_range / len(series) if len(series) else 0.0
            warnings.append(
                f"{col}: {out_of_range} rows ({pct:.2f}%) outside plausible range "
                f"[{low}, {high}] — expected from spoofed-sensor rows, verify against "
                f"pressure_sensor_fault/temperature_sensor_fault/tread_sensor_fault."
            )

    missing_fraction = df[["pressure", "temperature", "tread_depth"]].isna().mean()
    for col, frac in missing_fraction.items():
        if frac > 0.5:
            errors.append(
                f"{col} is missing in {frac:.1%} of rows — exceeds plausible sensor "
                f"dropout rate, likely a generation or ingestion bug."
            )
        elif frac > 0.0:
            warnings.append(f"{col} missing in {frac:.2%} of rows (expected from 'missing' sensor faults).")

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        row_count=len(df),
        out_of_range_counts=out_of_range_counts,
    )