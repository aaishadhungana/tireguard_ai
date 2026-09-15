"""
Repository layer: every database query in this project goes through a
function here. Routers/services call these functions; they never
construct a SQLAlchemy query directly -- this is what keeps persistence
concerns out of the API layer, mirroring the same separation this
project already applies to ML inference (src/api/services.py never
calls joblib.load directly either).
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Alert, Prediction, Telemetry, Tire, Vehicle


def upsert_vehicle(session: Session, vehicle_id: str) -> None:
    if session.get(Vehicle, vehicle_id) is None:
        session.add(Vehicle(vehicle_id=vehicle_id))
        session.flush()


def upsert_tire(session: Session, tire_id: str, vehicle_id: str) -> None:
    if session.get(Tire, tire_id) is None:
        session.add(Tire(tire_id=tire_id, vehicle_id=vehicle_id))
        session.flush()  


def bulk_insert_telemetry(session: Session, rows: list) -> int:
    session.bulk_insert_mappings(Telemetry, rows)
    return len(rows)


def get_tire_history(session: Session, tire_id: str, limit: int = 50):
    stmt = (
        select(Telemetry)
        .where(Telemetry.tire_id == tire_id)
        .order_by(Telemetry.timestamp.desc())
        .limit(limit)
    )
    rows = list(session.scalars(stmt))
    rows.reverse()  
    return rows


def get_latest_telemetry(session: Session, tire_id: str):
    stmt = (
        select(Telemetry)
        .where(Telemetry.tire_id == tire_id)
        .order_by(Telemetry.timestamp.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def tire_exists(session: Session, tire_id: str) -> bool:
    return session.get(Tire, tire_id) is not None


def insert_prediction(session: Session, tire_id: str, failure_probability: float, risk_level: str, model_type: str):
    prediction = Prediction(
        tire_id=tire_id,
        failure_probability=failure_probability,
        risk_level=risk_level,
        model_type=model_type,
    )
    session.add(prediction)
    session.flush()
    return prediction


def insert_alert(session: Session, tire_id: str, risk_level: str, failure_probability: float, message: str):
    alert = Alert(
        tire_id=tire_id,
        risk_level=risk_level,
        failure_probability=failure_probability,
        message=message,
    )
    session.add(alert)
    session.flush()
    return alert


def get_alerts(session: Session, risk_levels: list, limit: int = 100):
    stmt = (
        select(Alert)
        .where(Alert.risk_level.in_(risk_levels))
        .order_by(Alert.failure_probability.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def get_fleet_stats(session: Session) -> dict:
    total_tires = session.scalar(select(func.count(func.distinct(Telemetry.tire_id)))) or 0
    total_vehicles = session.scalar(select(func.count()).select_from(Vehicle)) or 0
    total_rows = session.scalar(select(func.count()).select_from(Telemetry)) or 0
    total_failures = session.scalar(select(func.count()).where(Telemetry.failure == 1)) or 0

    failure_type_rows = session.execute(
        select(Telemetry.failure_type, func.count())
        .where(Telemetry.failure == 1)
        .group_by(Telemetry.failure_type)
    ).all()

    return {
        "total_tires": total_tires,
        "total_vehicles": total_vehicles,
        "total_rows": total_rows,
        "failure_rate_pct": round(100 * total_failures / total_rows, 4) if total_rows else 0.0,
        "failure_type_counts": {ftype: count for ftype, count in failure_type_rows},
    }


def get_all_tire_ids(session: Session):
    return list(session.scalars(select(Tire.tire_id)))


def get_all_telemetry(session: Session):
    stmt = select(Telemetry).order_by(Telemetry.tire_id, Telemetry.timestamp)
    return list(session.scalars(stmt))


def get_tire_vehicle_map(session: Session) -> dict:
    return dict(session.execute(select(Tire.tire_id, Tire.vehicle_id)).all())