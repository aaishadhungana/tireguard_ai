from __future__ import annotations

import os

import pytest

from src.api.dependencies import AppState
from src.config.settings import settings
from src.copilot.agent import CopilotError, ask_copilot
from src.db.repository import get_all_tire_ids
from src.db.session import get_session


def test_ask_copilot_raises_without_api_key():
    """Settings is a frozen dataclass by design (immutable config) --
    monkeypatch.setattr can't touch it normally, so this bypasses the
    freeze deliberately via object.__setattr__ and restores the
    original value afterward, rather than weakening the dataclass's
    immutability for the sake of this one test."""
    original = settings.llm_api_key
    object.__setattr__(settings, "llm_api_key", "")
    try:
        with pytest.raises(CopilotError, match="LLM_API_KEY is not set"):
            ask_copilot("How many tires are in the fleet?", session=None, state=None)
    finally:
        object.__setattr__(settings, "llm_api_key", original)


@pytest.fixture(scope="module")
def db_session():
    try:
        with get_session() as session:
            tire_ids = get_all_tire_ids(session)
            if not tire_ids:
                pytest.skip("Database has no tires loaded.")
    except Exception as e:
        pytest.skip(f"Database not available: {e}")

    with get_session() as session:
        yield session


@pytest.fixture(scope="module")
def state():
    s = AppState()
    s.load()
    if s.failure_predictor is None:
        pytest.skip("Failure model not available.")
    return s


@pytest.mark.skipif(
    not os.getenv("LLM_API_KEY"),
    reason="LLM_API_KEY not set -- this test makes a REAL call to Gemini's API "
    "and is skipped without a real key. Get a free key at "
    "https://aistudio.google.com/apikey and set LLM_API_KEY to run it.",
)
def test_ask_copilot_with_real_api_key(db_session, state):
    result = ask_copilot("How many total tires are in the fleet?", db_session, state)

    assert result["answer"]
    assert len(result["tool_calls"]) > 0

    called_fleet_stats = any(c["tool"] == "get_fleet_stats" for c in result["tool_calls"])
    assert called_fleet_stats, (
        f"Expected get_fleet_stats to be called for a fleet-count question, "
        f"got: {[c['tool'] for c in result['tool_calls']]}"
    )

    real_stats = result["tool_calls"][0]["result"]
    assert str(real_stats["total_tires"]) in result["answer"], (
        "The final answer should mention the actual tire count returned by the "
        "tool, not a different or rounded number."
    )