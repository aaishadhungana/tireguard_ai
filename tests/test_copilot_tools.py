from __future__ import annotations

import pytest

from src.api.dependencies import AppState
from src.copilot import tools
from src.db.repository import get_all_tire_ids
from src.db.session import get_session


@pytest.fixture(scope="module")
def db_session():
    try:
        with get_session() as session:
            tire_ids = get_all_tire_ids(session)
            if not tire_ids:
                pytest.skip("Database has no tires loaded -- run `python -m src.db.load_data` first.")
    except Exception as e:
        pytest.skip(f"Database not available: {e}")

    with get_session() as session:
        yield session


@pytest.fixture(scope="module")
def state():
    s = AppState()
    s.load()
    if s.failure_predictor is None:
        pytest.skip("Failure model not available -- run train_failure_model.py first.")
    return s


@pytest.fixture(scope="module")
def real_tire_id(db_session):
    return get_all_tire_ids(db_session)[0]


def test_get_tire_status_returns_real_data(db_session, state, real_tire_id):
    result = tools.get_tire_status(db_session, state, real_tire_id)
    assert result["tire_id"] == real_tire_id
    assert result["latest_reading"] is not None
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= result["failure_probability"] <= 1.0


def test_get_tire_status_unknown_tire_returns_error(db_session, state):
    result = tools.get_tire_status(db_session, state, "NONEXISTENT-TIRE-ID")
    assert "error" in result


def test_get_root_cause_separates_sources(db_session, state, real_tire_id):
    result = tools.get_root_cause(db_session, state, real_tire_id)
    assert "error" not in result
    for c in result["top_model_contributions"]:
        assert c["source"] == "model_contribution"
    for c in result["triggered_engineering_rules"]:
        assert c["source"] == "engineering_rule"


def test_get_fleet_alerts_respects_threshold(db_session, state):
    low = tools.get_fleet_alerts(db_session, state, "LOW")
    high = tools.get_fleet_alerts(db_session, state, "HIGH")
    assert low["alert_count"] >= high["alert_count"]


def test_get_fleet_alerts_default_threshold_is_medium(db_session, state):
    default = tools.get_fleet_alerts(db_session, state)
    explicit = tools.get_fleet_alerts(db_session, state, "MEDIUM")
    assert default["alert_count"] == explicit["alert_count"]


def test_get_fleet_stats_returns_real_aggregates(db_session, state):
    result = tools.get_fleet_stats(db_session, state)
    assert result["total_tires"] > 0
    assert result["total_vehicles"] > 0
    assert 0.0 <= result["failure_rate_pct"] <= 100.0


def test_get_tire_history_respects_limit(db_session, state, real_tire_id):
    result = tools.get_tire_history(db_session, state, real_tire_id, limit=5)
    assert result["row_count"] <= 5
    assert len(result["rows"]) == result["row_count"]


def test_get_tire_history_unknown_tire_returns_error(db_session, state):
    result = tools.get_tire_history(db_session, state, "NONEXISTENT-TIRE-ID")
    assert "error" in result


def test_execute_tool_routes_correctly(db_session, state, real_tire_id):
    result = tools.execute_tool("get_tire_status", {"tire_id": real_tire_id}, db_session, state)
    assert result["tire_id"] == real_tire_id


def test_execute_tool_unknown_tool_returns_error(db_session, state):
    result = tools.execute_tool("not_a_real_tool", {}, db_session, state)
    assert "error" in result
    assert "Unknown tool" in result["error"]


def test_execute_tool_passes_kwargs_correctly(db_session, state):
    result = tools.execute_tool("get_fleet_alerts", {"risk_threshold": "HIGH"}, db_session, state)
    assert "error" not in result
    assert "alert_count" in result


def test_all_tool_definitions_have_matching_dispatch_entries():
    defined_names = {t["name"] for t in tools.TOOL_DEFINITIONS}
    dispatched_names = set(tools.TOOL_DISPATCH.keys())
    assert defined_names == dispatched_names