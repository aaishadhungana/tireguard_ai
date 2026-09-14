from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api import services
from src.api.dependencies import AppState, get_app_state
from src.api.schemas import FailurePredictionResponse, TelemetryReading

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/failure", response_model=FailurePredictionResponse)
def predict_failure(reading: TelemetryReading, state: AppState = Depends(get_app_state)):
    return services.predict_failure_for_reading(reading, state)