from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api import services
from src.api.dependencies import AppState, get_app_state, get_db
from src.api.schemas import AlertsResponse, FleetStatsResponse, FleetTiresResponse

router = APIRouter(prefix="/fleet", tags=["fleet"])


@router.get("/stats", response_model=FleetStatsResponse)
def fleet_stats(session: Session = Depends(get_db)):
    return services.get_fleet_stats(session)


@router.get("/tires", response_model=FleetTiresResponse)
def fleet_tires(
    state: AppState = Depends(get_app_state),
    session: Session = Depends(get_db),
):
    """Every tire's current status -- powers the dashboard's fleet grid.
    /fleet/stats intentionally stays aggregate-only; this is the
    per-tire complement to it."""
    return services.get_fleet_tires(session, state)


@router.get("/alerts", response_model=AlertsResponse)
def fleet_alerts(
    risk_threshold: str = Query(default="MEDIUM", pattern="^(LOW|MEDIUM|HIGH)$"),
    state: AppState = Depends(get_app_state),
    session: Session = Depends(get_db),
):
    return services.get_alerts(session, state, risk_threshold)