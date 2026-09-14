from __future__ import annotations

import joblib
import pandas as pd
from fastapi import HTTPException

from src.config.settings import settings
from src.models.explain import RootCauseExplainer
from src.models.predict import FailurePredictor
from src.utils.logger import get_logger

log = get_logger(__name__)


class AppState:
    def __init__(self):
        self.dataset = None
        self.failure_predictor = None
        self.root_cause_explainer = None
        self.rul_model = None
        self.load_errors = {}

    def load(self):
        log.info("Loading application state...")

        try:
            candidates = sorted(settings.data_processed_dir.glob("tire_telemetry_processed_*.parquet"))
            if not candidates:
                raise FileNotFoundError("No processed datasets found. Run the data pipeline first.")
            self.dataset = pd.read_parquet(candidates[-1])
            log.info("Loaded dataset: %s (%d rows)", candidates[-1].name, len(self.dataset))
        except Exception as e:
            log.error("Failed to load dataset: %s", e)
            self.load_errors["dataset"] = str(e)

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

    def get_tire_rows(self, tire_id):
        if self.dataset is None:
            raise HTTPException(status_code=503, detail="Dataset not loaded -- service is not fully initialized.")
        rows = self.dataset[self.dataset["tire_id"] == tire_id]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"No data found for tire_id '{tire_id}'.")
        return rows.sort_values("timestamp")

app_state = AppState()


def get_app_state():
    return app_state