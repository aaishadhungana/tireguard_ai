from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config.settings import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.ensure_directories()

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger("tireguard")
    root.setLevel(level)
    root.propagate = False

    if not root.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            settings.log_dir / "tireguard.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. get_logger(__name__).

    All loggers share the 'tireguard' root logger's handlers/level, so
    console + file output stays consistent everywhere in the codebase.
    """
    _configure_root_logger()
    return logging.getLogger(f"tireguard.{name}")