from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api import services
from src.api.dependencies import AppState, get_app_state
from src.api.schemas import RootCauseResponse, RULResponse, TireHistoryResponse

router = APIRouter(prefix="/tires", tags=["tires"])


@router.get("/{tire_id}/history", response_model=TireHistoryResponse)
def tire_history(
    tire_id: str,
    limit: int = Query(default=50, ge=1, le=1000),
    state: AppState = Depends(get_app_state),
):
    return services.get_tire_history(tire_id, state, limit)


@router.get("/{tire_id}/rul", response_model=RULResponse)
def tire_rul(tire_id: str, state: AppState = Depends(get_app_state)):
    return services.get_rul_for_tire(tire_id, state)


@router.get("/{tire_id}/root-cause", response_model=RootCauseResponse)
def tire_root_cause(tire_id: str, state: AppState = Depends(get_app_state)):
    return services.get_root_cause_for_tire(tire_id, state)