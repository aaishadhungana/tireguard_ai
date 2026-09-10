from __future__ import annotations

import logging

from src.config.settings import Settings, settings
from src.utils.logger import get_logger


def test_settings_defaults_are_sane():
    s = Settings()
    assert s.app_env in {"development", "testing", "production"}
    assert s.random_seed == 42
    assert s.sim_failure_rate >= 0.0


def test_ensure_directories_creates_paths(tmp_path, monkeypatch):
    test_settings = Settings(
        data_raw_dir=tmp_path / "raw",
        data_processed_dir=tmp_path / "processed",
        model_dir=tmp_path / "models",
        log_dir=tmp_path / "logs",
    )
    test_settings.ensure_directories()

    assert test_settings.data_raw_dir.exists()
    assert test_settings.data_processed_dir.exists()
    assert test_settings.model_dir.exists()
    assert test_settings.log_dir.exists()


def test_singleton_settings_importable():
    assert settings.random_seed == 42 or isinstance(settings.random_seed, int)


def test_get_logger_returns_namespaced_logger():
    logger = get_logger("tests.test_foundation")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "tireguard.tests.test_foundation"
    logger.info("Logger smoke test message.")