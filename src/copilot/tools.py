from __future__ import annotations

from fastapi import HTTPException

from src.api import services

TOOL_DEFINITIONS = [
    {
        "name": "get_tire_status",
        "description": (
            "Get a specific tire's current sensor readings, failure probability, "
            "risk level, and remaining useful life (RUL) estimate. Use this when "
            "asked about one specific tire by its ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tire_id": {"type": "string", "description": "The tire's ID, e.g. 'V-0001-T0'."},
            },
            "required": ["tire_id"],
        },
    },
    {
        "name": "get_root_cause",
        "description": (
            "Get the model-contribution (SHAP) and engineering-rule explanation "
            "for why a specific tire has its current failure risk. Use this when "
            "asked WHY a tire is at risk, not just what its risk level is."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tire_id": {"type": "string", "description": "The tire's ID."},
            },
            "required": ["tire_id"],
        },
    },
    {
        "name": "get_fleet_alerts",
        "description": (
            "Get every tire currently at or above a given risk level, ranked by "
            "failure probability. Use this for questions like 'which tires need "
            "attention' or 'what's critical right now'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "risk_threshold": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH"],
                    "description": "Minimum risk level to include. Defaults to MEDIUM.",
                },
            },
        },
    },
    {
        "name": "get_fleet_stats",
        "description": (
            "Get fleet-wide aggregate statistics: total vehicles/tires, overall "
            "failure rate, and failure-type breakdown. Use this for summary/overview "
            "questions about the whole fleet rather than a specific tire."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_tire_history",
        "description": (
            "Get a specific tire's recent telemetry readings (pressure, "
            "temperature, tread depth, speed, load) over time. Use this when asked "
            "about a tire's recent trend or historical readings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tire_id": {"type": "string", "description": "The tire's ID."},
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent readings to return. Defaults to 20.",
                },
            },
            "required": ["tire_id"],
        },
    },
]


def _safe(fn, *args, **kwargs) -> dict:
    """Every service function can raise HTTPException (tire not found,
    model not loaded) -- converted here to structured data the LLM can
    report as a fact, rather than crashing the copilot request."""
    try:
        return fn(*args, **kwargs)
    except HTTPException as e:
        return {"error": e.detail}


def get_tire_status(session, state, tire_id: str) -> dict:
    history = _safe(services.get_tire_history, tire_id, session, 1)
    if "error" in history:
        return history

    rul = _safe(services.get_rul_for_tire, tire_id, session, state)
    root_cause = _safe(services.get_root_cause_for_tire, tire_id, session, state)

    latest = history["rows"][-1] if history.get("rows") else None
    return {
        "tire_id": tire_id,
        "latest_reading": latest,
        "failure_probability": root_cause.get("failure_probability") if "error" not in root_cause else None,
        "risk_level": root_cause.get("risk_level") if "error" not in root_cause else None,
        "estimated_rul_km": rul.get("estimated_rul_km") if "error" not in rul else None,
        "rul_note": rul.get("error") if "error" in rul else None,
    }


def get_root_cause(session, state, tire_id: str) -> dict:
    return _safe(services.get_root_cause_for_tire, tire_id, session, state)


def get_fleet_alerts(session, state, risk_threshold: str = "MEDIUM") -> dict:
    return _safe(services.get_alerts, session, state, risk_threshold)


def get_fleet_stats(session, state) -> dict:
    return _safe(services.get_fleet_stats, session)


def get_tire_history(session, state, tire_id: str, limit: int = 20) -> dict:
    return _safe(services.get_tire_history, tire_id, session, limit)


TOOL_DISPATCH = {
    "get_tire_status": get_tire_status,
    "get_root_cause": get_root_cause,
    "get_fleet_alerts": get_fleet_alerts,
    "get_fleet_stats": get_fleet_stats,
    "get_tire_history": get_tire_history,
}


def execute_tool(tool_name: str, tool_input: dict, session, state) -> dict:
    """Single entry point the agent loop calls -- looks up the right
    function and calls it with the LLM-provided arguments. Unknown tool
    names return a structured error rather than raising, since a
    malformed tool call from the LLM shouldn't crash the whole request."""
    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return fn(session, state, **tool_input)