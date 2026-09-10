from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # ---- App ----
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    random_seed: int = field(default_factory=lambda: _get_int("RANDOM_SEED", 42))

    # ---- Paths----
    data_raw_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
    )
    data_processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    )
    model_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
    )
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    # ---- Simulator ----
    sim_num_vehicles: int = field(default_factory=lambda: _get_int("SIM_NUM_VEHICLES", 50))
    sim_tires_per_vehicle: int = field(
        default_factory=lambda: _get_int("SIM_TIRES_PER_VEHICLE", 4)
    )
    sim_duration_days: int = field(default_factory=lambda: _get_int("SIM_DURATION_DAYS", 30))
    sim_sampling_interval_min: int = field(
        default_factory=lambda: _get_int("SIM_SAMPLING_INTERVAL_MIN", 15)
    )
    sim_failure_rate: float = field(
        default_factory=lambda: _get_float("SIM_FAILURE_RATE", 0.05)
    )

    def ensure_directories(self) -> None:

        for path in (self.data_raw_dir, self.data_processed_dir, self.model_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()