from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from src.config.settings import settings
from src.db import repository
from src.db.models import Tire, Vehicle
from src.db.session import create_all_tables, drop_all_tables, get_session


def _test_database_url() -> str:
    override = os.getenv("TEST_DATABASE_URL")
    if override:
        return override
    return settings.database_url + "_test"


TEST_DB_URL = _test_database_url()


@pytest.fixture()
def db_session():
    try:
        drop_all_tables(TEST_DB_URL)
        create_all_tables(TEST_DB_URL)
    except Exception as e:
        pytest.skip(f"Test PostgreSQL database not available: {e}")

    with get_session(TEST_DB_URL) as session:
        yield session

    drop_all_tables(TEST_DB_URL)


def make_telemetry_row(tire_id, ts, **overrides):
    row = dict(
        tire_id=tire_id,
        timestamp=ts,
        pressure=32.0,
        temperature=25.0,
        speed=70.0,
        load=500.0,
        tread_depth=8.0,
        mileage=1000.0,
        braking_events=1,
        acceleration=0.1,
        road_type="highway",
        weather="clear",
        failure=0,
        failure_type="none",
    )
    row.update(overrides)
    return row


def test_vehicle_tire_relationship(db_session):
    repository.upsert_vehicle(db_session, "V-0001")
    repository.upsert_tire(db_session, "V-0001-T0", "V-0001")
    db_session.commit()

    vehicle = db_session.get(Vehicle, "V-0001")
    assert vehicle is not None
    assert len(vehicle.tires) == 1
    assert vehicle.tires[0].tire_id == "V-0001-T0"


def test_upsert_is_idempotent(db_session):
    repository.upsert_vehicle(db_session, "V-0002")
    repository.upsert_vehicle(db_session, "V-0002")
    db_session.commit()

    count = db_session.query(Vehicle).filter_by(vehicle_id="V-0002").count()
    assert count == 1


def test_cascade_delete_removes_tires():
    try:
        drop_all_tables(TEST_DB_URL)
        create_all_tables(TEST_DB_URL)
    except Exception as e:
        pytest.skip(f"Test PostgreSQL database not available: {e}")

    with get_session(TEST_DB_URL) as session:
        repository.upsert_vehicle(session, "V-0003")
        repository.upsert_tire(session, "V-0003-T0", "V-0003")

    with get_session(TEST_DB_URL) as session:
        vehicle = session.get(Vehicle, "V-0003")
        session.delete(vehicle)

    with get_session(TEST_DB_URL) as session:
        assert session.get(Tire, "V-0003-T0") is None
    drop_all_tables(TEST_DB_URL)


def test_bulk_insert_and_history_ordering(db_session):
    repository.upsert_vehicle(db_session, "V-0004")
    repository.upsert_tire(db_session, "V-0004-T0", "V-0004")

    rows = [
        make_telemetry_row("V-0004-T0", datetime(2026, 1, 1, i, tzinfo=timezone.utc), pressure=32.0 - i)
        for i in range(5)
    ]
    repository.bulk_insert_telemetry(db_session, rows)
    db_session.commit()

    history = repository.get_tire_history(db_session, "V-0004-T0", limit=10)
    assert len(history) == 5
    timestamps = [h.timestamp for h in history]
    assert timestamps == sorted(timestamps)


def test_history_limit_returns_most_recent_rows(db_session):
    repository.upsert_vehicle(db_session, "V-0005")
    repository.upsert_tire(db_session, "V-0005-T0", "V-0005")

    rows = [
        make_telemetry_row("V-0005-T0", datetime(2026, 1, 1, i, tzinfo=timezone.utc), mileage=float(i))
        for i in range(10)
    ]
    repository.bulk_insert_telemetry(db_session, rows)
    db_session.commit()

    history = repository.get_tire_history(db_session, "V-0005-T0", limit=3)
    assert len(history) == 3
    assert [h.mileage for h in history] == [7.0, 8.0, 9.0]


def test_get_latest_telemetry(db_session):
    repository.upsert_vehicle(db_session, "V-0006")
    repository.upsert_tire(db_session, "V-0006-T0", "V-0006")
    rows = [
        make_telemetry_row("V-0006-T0", datetime(2026, 1, 1, i, tzinfo=timezone.utc), mileage=float(i))
        for i in range(3)
    ]
    repository.bulk_insert_telemetry(db_session, rows)
    db_session.commit()

    latest = repository.get_latest_telemetry(db_session, "V-0006-T0")
    assert latest.mileage == 2.0


def test_history_for_unknown_tire_returns_empty(db_session):
    history = repository.get_tire_history(db_session, "NONEXISTENT", limit=10)
    assert history == []


def test_fleet_stats_counts_are_accurate(db_session):
    repository.upsert_vehicle(db_session, "V-0007")
    repository.upsert_tire(db_session, "V-0007-T0", "V-0007")
    repository.upsert_tire(db_session, "V-0007-T1", "V-0007")

    rows = [make_telemetry_row("V-0007-T0", datetime(2026, 1, 1, i, tzinfo=timezone.utc)) for i in range(5)]
    rows += [
        make_telemetry_row(
            "V-0007-T1", datetime(2026, 1, 1, i, tzinfo=timezone.utc),
            failure=1 if i == 2 else 0, failure_type="puncture" if i == 2 else "none",
        )
        for i in range(5)
    ]
    repository.bulk_insert_telemetry(db_session, rows)
    db_session.commit()

    stats = repository.get_fleet_stats(db_session)
    assert stats["total_tires"] == 2
    assert stats["total_vehicles"] == 1
    assert stats["total_rows"] == 10
    assert stats["failure_rate_pct"] == pytest.approx(10.0)
    assert stats["failure_type_counts"] == {"puncture": 1}


def test_insert_and_query_alerts(db_session):
    repository.upsert_vehicle(db_session, "V-0008")
    repository.upsert_tire(db_session, "V-0008-T0", "V-0008")
    repository.insert_alert(db_session, "V-0008-T0", "HIGH", 0.9, "test alert")
    repository.insert_alert(db_session, "V-0008-T0", "LOW", 0.1, "should not appear")
    db_session.commit()

    alerts = repository.get_alerts(db_session, risk_levels=["HIGH", "MEDIUM"])
    assert len(alerts) == 1
    assert alerts[0].risk_level == "HIGH"