from __future__ import annotations

import joblib
from fastapi import HTTPException

from src.config.settings import settings
from src.db.session import get_session_factory
from src.models.explain import RootCauseExplainer
from src.models.predict import FailurePredictor
from src.utils.logger import get_logger

log = get_logger(__name__)


class AppState:
    def __init__(self):
        self.failure_predictor = None
        self.root_cause_explainer = None
        self.rul_model = None
        self.load_errors = {}

    def load(self):
        log.info("Loading application state...")

        try:
            self.failure_predictor = FailurePredictor()
            self.root_cause_explainer = RootCauseExplainer(self.failure_predictor.model)
            log.info("Loaded failure prediction model: %s", self.failure_predictor.metadata.get("model_type"))
        except Exception as e:
            log.error("Failed to load failure model: %s", e)
            self.load_errors["failure_model"] = str(e)

        try:
            rul_path = settings.model_dir / "rul_model.joblib"
            if rul_path.exists():
                self.rul_model = joblib.load(rul_path)
                log.info("Loaded RUL model from %s", rul_path)
            else:
                self.load_errors["rul_model"] = f"No RUL model found at {rul_path}"
        except Exception as e:
            log.error("Failed to load RUL model: %s", e)
            self.load_errors["rul_model"] = str(e)

app_state = AppState()


def get_app_state():
    return app_state


def get_db():
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()