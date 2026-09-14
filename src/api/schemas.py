from __future__ import annotations

from pydantic import BaseModel, Field


class TelemetryReading(BaseModel):

    tire_id: str
    vehicle_id: str
    pressure: float = Field(..., ge=0, le=150, description="psi")
    temperature: float = Field(..., ge=-40, le=200, description="Celsius")
    speed: float = Field(..., ge=0, le=300, description="km/h")
    load: float = Field(..., ge=0, le=3000, description="kg")
    tread_depth: float = Field(..., ge=0, le=20, description="mm")
    mileage: float = Field(..., ge=0)
    braking_events: int = Field(..., ge=0)
    acceleration: float = 0.0
    road_type: str = "highway"
    weather: str = "clear"


class FailurePredictionResponse(BaseModel):
    tire_id: str
    failure_probability: float
    risk_level: str
    model_type: str


class RootCauseContribution(BaseModel):
    feature: str
    value: float | str
    contribution: float
    direction: str
    source: str


class RootCauseResponse(BaseModel):
    tire_id: str
    failure_probability: float
    risk_level: str
    top_model_contributions: list[RootCauseContribution]
    triggered_engineering_rules: list[RootCauseContribution]
    caveat: str


class RULResponse(BaseModel):
    tire_id: str
    estimated_rul_km: float
    model_type: str
    note: str = (
        "RUL estimates wear-based remaining life only (time until tread "
        "reaches the legal minimum). It does not cover sudden failure modes "
        "like puncture."
    )


class TireHistoryRow(BaseModel):
    timestamp: str
    pressure: float
    temperature: float
    tread_depth: float
    speed: float
    load: float
    failure: int


class TireHistoryResponse(BaseModel):
    tire_id: str
    row_count: int
    rows: list[TireHistoryRow]


class FleetStatsResponse(BaseModel):
    total_tires: int
    total_vehicles: int
    total_rows: int
    failure_rate_pct: float
    failure_type_counts: dict
    data_source_note: str = (
        "Statistics computed from the most recent processed dataset loaded "
        "into memory at API startup -- this is a temporary dev-mode data "
        "source, not a persistent database (that's Milestone 11)."
    )


class AlertItem(BaseModel):
    tire_id: str
    vehicle_id: str
    failure_probability: float
    risk_level: str
    timestamp: str


class AlertsResponse(BaseModel):
    alert_count: int
    alerts: list[AlertItem]


class ErrorResponse(BaseModel):
    detail: str