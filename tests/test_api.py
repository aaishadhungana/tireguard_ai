from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_tire_id():
    """Pulls an actual tire_id directly from the database -- requires
    `python -m src.db.load_data` to have been run first, same as any
    other test depending on real loaded data."""
    from src.db.session import get_session
    from src.db.repository import get_all_tire_ids

    try:
        with get_session() as session:
            tire_ids = get_all_tire_ids(session)
    except Exception:
        pytest.skip("Database not available -- run `python -m src.db.load_data` first.")
    if not tire_ids:
        pytest.skip("Database has no tires loaded -- run `python -m src.db.load_data` first.")
    return tire_ids[0]


def test_root_returns_service_info(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "TireGuard AI API"


def test_health_reports_component_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "components" in body
    assert set(body["components"].keys()) == {"database", "failure_model", "rul_model"}


def test_tire_history_returns_real_rows(client, real_tire_id):
    response = client.get(f"/tires/{real_tire_id}/history?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["tire_id"] == real_tire_id
    assert body["row_count"] <= 10
    assert body["row_count"] > 0
    assert "pressure" in body["rows"][0]


def test_tire_history_unknown_tire_returns_404(client):
    response = client.get("/tires/NONEXISTENT-TIRE-ID/history")
    assert response.status_code == 404


def test_tire_rul_returns_plausible_estimate(client, real_tire_id):
    response = client.get(f"/tires/{real_tire_id}/rul")
    assert response.status_code == 200
    body = response.json()
    assert body["estimated_rul_km"] >= 0.0


def test_tire_root_cause_separates_sources(client, real_tire_id):
    response = client.get(f"/tires/{real_tire_id}/root-cause")
    assert response.status_code == 200
    body = response.json()
    for contribution in body["top_model_contributions"]:
        assert contribution["source"] == "model_contribution"
    for rule in body["triggered_engineering_rules"]:
        assert rule["source"] == "engineering_rule"


def test_predict_failure_with_valid_reading(client):
    payload = {
        "tire_id": "TEST-TIRE-001",
        "vehicle_id": "TEST-VEHICLE-001",
        "pressure": 32.0,
        "temperature": 25.0,
        "speed": 70.0,
        "load": 500.0,
        "tread_depth": 8.0,
        "mileage": 1000.0,
        "braking_events": 1,
    }
    response = client.post("/predict/failure", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["tire_id"] == "TEST-TIRE-001"
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_failure_rejects_invalid_pressure(client):
    """Pydantic's Field(ge=0, le=150) constraint should reject this
    before the handler even runs -- a real validation check, not just
    'does the endpoint exist.'"""
    payload = {
        "tire_id": "TEST-TIRE-002",
        "vehicle_id": "TEST-VEHICLE-002",
        "pressure": -50.0,  # invalid
        "temperature": 25.0,
        "speed": 70.0,
        "load": 500.0,
        "tread_depth": 8.0,
        "mileage": 1000.0,
        "braking_events": 1,
    }
    response = client.post("/predict/failure", json=payload)
    assert response.status_code == 422


def test_predict_failure_rejects_missing_required_field(client):
    payload = {"tire_id": "TEST-TIRE-003"}  # missing everything else
    response = client.post("/predict/failure", json=payload)
    assert response.status_code == 422


def test_fleet_tires_returns_all_tires_with_status(client):
    response = client.get("/fleet/tires")
    assert response.status_code == 200
    body = response.json()
    assert body["tire_count"] > 0
    first = body["tires"][0]
    assert set(["tire_id", "vehicle_id", "pressure", "risk_level", "failure_probability"]) <= set(first.keys())


def test_fleet_stats_returns_sane_numbers(client):
    response = client.get("/fleet/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_tires"] > 0
    assert body["total_rows"] > 0
    assert 0.0 <= body["failure_rate_pct"] <= 100.0


def test_fleet_alerts_high_threshold_returns_fewer_than_low(client):
    low_response = client.get("/fleet/alerts?risk_threshold=LOW")
    high_response = client.get("/fleet/alerts?risk_threshold=HIGH")
    assert low_response.status_code == 200
    assert high_response.status_code == 200
    assert low_response.json()["alert_count"] >= high_response.json()["alert_count"]


def test_fleet_alerts_rejects_invalid_threshold(client):
    response = client.get("/fleet/alerts?risk_threshold=EXTREME")
    assert response.status_code == 422


def test_unhandled_exception_returns_structured_500(monkeypatch):
    """Forces an unexpected exception inside a handler and checks the
    caller gets a clean structured error, not a raw traceback. Uses a
    dedicated client with raise_server_exceptions=False, since
    TestClient's default behavior is to re-raise server-side exceptions
    for debugging -- exactly what this test needs to disable to verify
    the app's OWN exception handler actually runs."""
    from src.api import services

    def broken(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(services, "get_fleet_stats", broken)

    with TestClient(app, raise_server_exceptions=False) as isolated_client:
        response = isolated_client.get("/fleet/stats")
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error."}