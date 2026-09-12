from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd

from src.config.settings import settings
from src.models.failure_model import build_feature_matrix


@dataclass
class FailurePrediction:
    failure_probability: float
    risk_level: str
    model_type: str


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.3:
        return "MEDIUM"
    return "LOW"


class FailurePredictor:
    """Loads the serialized model once; call .predict(df) for one or
    more rows already in the processed-feature schema (i.e. the output
    of src.data.pipeline, NOT raw simulator CSV)."""

    def __init__(self, model_path=None, metadata_path=None):
        self.model_path = model_path or settings.model_dir / "failure_model.joblib"
        self.metadata_path = metadata_path or settings.model_dir / "failure_model.meta.json"

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {self.model_path}. "
                f"Run `python -m src.models.train_failure_model` first."
            )

        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

    def predict(self, df: pd.DataFrame) -> list[FailurePrediction]:
        """df must contain at least the feature columns this model was
        trained on (see self.metadata['feature_columns']) — a dummy
        'failure' column is added if missing since build_feature_matrix
        expects a target column, but its value is never used here."""
        df = df.copy()
        if "failure" not in df.columns:
            df["failure"] = 0  # placeholder, unused for inference

        X, _ = build_feature_matrix(df)
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            FailurePrediction(
                failure_probability=round(float(p), 4),
                risk_level=_risk_level(p),
                model_type=self.metadata["model_type"],
            )
            for p in probabilities
        ]