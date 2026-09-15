"""
Database schema for TireGuard AI.

SCOPE DECISIONS (stated explicitly, per this project's "avoid
overengineering" rule):
- Plain PostgreSQL, not TimescaleDB. TimescaleDB's hypertables,
  continuous aggregates, and retention policies earn their complexity
  at genuinely large time-series scale (millions+ rows, long retention
  windows). This project's telemetry volume comfortably fits in a
  normal indexed Postgres table. Worth revisiting if/when telemetry
  volume grows by an order of magnitude or more.
- create_all() for schema setup, not Alembic migrations. Alembic earns
  its complexity once a schema has evolved across multiple deployed
  versions with real data to migrate. At this stage there is one
  schema version and no deployed data to preserve across changes -- a
  migration framework would be process overhead with no current
  benefit. Worth adding before any production deployment.

RELATIONSHIPS:
    Vehicle (1) -> (many) Tire
    Tire (1) -> (many) Telemetry
    Tire (1) -> (many) Prediction
    Tire (1) -> (many) Alert
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Vehicle(Base):
    __tablename__ = "vehicles"

    vehicle_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    tires: Mapped[list["Tire"]] = relationship(back_populates="vehicle", cascade="all, delete-orphan")


class Tire(Base):
    __tablename__ = "tires"

    tire_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    vehicle_id: Mapped[str] = mapped_column(ForeignKey("vehicles.vehicle_id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    vehicle: Mapped["Vehicle"] = relationship(back_populates="tires")
    telemetry: Mapped[list["Telemetry"]] = relationship(back_populates="tire", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="tire", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="tire", cascade="all, delete-orphan")


class Telemetry(Base):
    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tire_id: Mapped[str] = mapped_column(ForeignKey("tires.tire_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    pressure: Mapped[float] = mapped_column(Float)
    temperature: Mapped[float] = mapped_column(Float)
    speed: Mapped[float] = mapped_column(Float)
    load: Mapped[float] = mapped_column(Float)
    tread_depth: Mapped[float] = mapped_column(Float)
    mileage: Mapped[float] = mapped_column(Float)
    braking_events: Mapped[int] = mapped_column(Integer)
    acceleration: Mapped[float] = mapped_column(Float, default=0.0)
    road_type: Mapped[str] = mapped_column(String(32))
    weather: Mapped[str] = mapped_column(String(32))
    failure: Mapped[int] = mapped_column(Integer, default=0)
    failure_type: Mapped[str] = mapped_column(String(64), default="none")

    tire: Mapped["Tire"] = relationship(back_populates="telemetry")

    __table_args__ = (
        Index("ix_telemetry_tire_timestamp", "tire_id", "timestamp"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tire_id: Mapped[str] = mapped_column(ForeignKey("tires.tire_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_utcnow)

    failure_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    model_type: Mapped[str] = mapped_column(String(64))

    tire: Mapped["Tire"] = relationship(back_populates="predictions")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tire_id: Mapped[str] = mapped_column(ForeignKey("tires.tire_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=_utcnow)

    risk_level: Mapped[str] = mapped_column(String(16))
    failure_probability: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(String(512))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)

    tire: Mapped["Tire"] = relationship(back_populates="alerts")