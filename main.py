from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    settings.ensure_directories()

    log.info("TireGuard AI — Milestone 0 (Project Foundation)")
    log.info("Environment: %s", settings.app_env)
    log.info("Random seed: %s", settings.random_seed)
    log.info("Data raw dir: %s", settings.data_raw_dir)
    log.info("Data processed dir: %s", settings.data_processed_dir)
    log.info("Model dir: %s", settings.model_dir)
    log.info("Log dir: %s", settings.log_dir)
    log.info("Scaffold OK — ready for Milestone 1 (Tire Telemetry Simulator).")


if __name__ == "__main__":
    main()