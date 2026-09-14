from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api import services
from src.api.dependencies import AppState, get_app_state
from src.api.schemas import AlertsResponse, FleetStatsResponse

router = APIRouter(prefix="/fleet", tags=["fleet"])


@router.get("/stats", response_model=FleetStatsResponse)
def fleet_stats(state: AppState = Depends(get_app_state)):
    return services.get_fleet_stats(state)


@router.get("/alerts", response_model=AlertsResponse)
def fleet_alerts(
    risk_threshold: str = Query(default="MEDIUM", pattern="^(LOW|MEDIUM|HIGH)$"),
    state: AppState = Depends(get_app_state),
):
    return services.get_alerts(state, risk_threshold)