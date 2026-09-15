from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from src.data.features import engineer_features
from src.db import repository
from src.streaming.message_schema import row_to_message
from src.streaming.subscriber import assemble_feature_row
from src.streaming.feature_buffer import TireFeatureBuffer

RAW_TELEMETRY_COLUMNS = [
    "tire_id", "timestamp", "pressure", "temperature", "speed", "load",
    "tread_depth", "mileage", "braking_events", "acceleration",
    "road_type", "weather", "failure", "failure_type",
]


def _telemetry_rows_to_df(rows):
    return pd.DataFrame(
        [{col: getattr(row, col) for col in RAW_TELEMETRY_COLUMNS} for row in rows]
    )


def _add_default_flag_columns(df):
    for col in ("pressure_was_missing", "temperature_was_missing", "tread_depth_was_missing",
                "pressure_out_of_range", "temperature_out_of_range", "tread_depth_out_of_range"):
        df[col] = False
    for col in ("pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault"):
        df[col] = "none"
    return df


def _build_engineered_row_for_tire(tire_id, session):
    rows = repository.get_tire_history(session, tire_id, limit=100_000)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data found for tire_id '{tire_id}'.")
    df = _telemetry_rows_to_df(rows)
    featured = engineer_features(df)
    featured = _add_default_flag_columns(featured)
    return featured.iloc[-1]


def predict_failure_for_reading(reading, state):
    if state.failure_predictor is None:
        raise HTTPException(status_code=503, detail="Failure prediction model is not loaded.")

    message = row_to_message(
        {
            "tire_id": reading.tire_id,
            "vehicle_id": reading.vehicle_id,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "pressure": reading.pressure,
            "temperature": reading.temperature,
            "speed": reading.speed,
            "load": reading.load,
            "tread_depth": reading.tread_depth,
            "mileage": reading.mileage,
            "braking_events": reading.braking_events,
            "acceleration": reading.acceleration,
            "road_type": reading.road_type,
            "weather": reading.weather,
        }
    )
    buffer = TireFeatureBuffer()
    computed = buffer.add_reading(
        pressure=reading.pressure,
        temperature=reading.temperature,
        tread_depth=reading.tread_depth,
        braking_events=reading.braking_events,
    )
    row = assemble_feature_row(message, computed)
    row_df = pd.DataFrame([row])

    prediction = state.failure_predictor.predict(row_df)[0]
    return {
        "tire_id": reading.tire_id,
        "failure_probability": prediction.failure_probability,
        "risk_level": prediction.risk_level,
        "model_type": prediction.model_type,
    }


def get_root_cause_for_tire(tire_id, session, state):
    if state.root_cause_explainer is None:
        raise HTTPException(status_code=503, detail="Root cause explainer is not loaded.")

    latest_row = _build_engineered_row_for_tire(tire_id, session)
    from src.models.failure_model import build_feature_matrix

    row_df = pd.DataFrame([latest_row])
    X, _ = build_feature_matrix(row_df)
    explanation = state.root_cause_explainer.explain(X.iloc[0])

    return {
        "tire_id": tire_id,
        "failure_probability": explanation.failure_probability,
        "risk_level": explanation.risk_level,
        "top_model_contributions": [
            {"feature": c.feature, "value": c.value, "contribution": c.contribution, "direction": c.direction, "source": c.source}
            for c in explanation.top_model_contributions
        ],
        "triggered_engineering_rules": [
            {"feature": c.feature, "value": c.value, "contribution": c.contribution, "direction": c.direction, "source": c.source}
            for c in explanation.triggered_engineering_rules
        ],
        "caveat": explanation.caveat,
    }


def get_rul_for_tire(tire_id, session, state):
    if state.rul_model is None:
        raise HTTPException(status_code=503, detail="RUL model is not loaded.")

    latest_row = _build_engineered_row_for_tire(tire_id, session)
    from src.models.rul_model import NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES

    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    row_df = pd.DataFrame([latest_row])[feature_cols]
    predicted_rul = float(state.rul_model.predict(row_df)[0])

    return {
        "tire_id": tire_id,
        "estimated_rul_km": round(max(0.0, predicted_rul), 1),
        "model_type": type(state.rul_model.named_steps["model"]).__name__,
    }


def get_tire_history(tire_id, session, limit):
    rows = repository.get_tire_history(session, tire_id, limit)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data found for tire_id '{tire_id}'.")
    return {
        "tire_id": tire_id,
        "row_count": len(rows),
        "rows": [
            {
                "timestamp": str(r.timestamp),
                "pressure": r.pressure,
                "temperature": r.temperature,
                "tread_depth": r.tread_depth,
                "speed": r.speed,
                "load": r.load,
                "failure": r.failure,
            }
            for r in rows
        ],
    }


def get_fleet_stats(session):
    return repository.get_fleet_stats(session)


def get_alerts(session, state, risk_threshold="MEDIUM"):
    if state.failure_predictor is None:
        raise HTTPException(status_code=503, detail="Failure prediction model is not loaded.")

    all_rows = repository.get_all_telemetry(session)
    if not all_rows:
        raise HTTPException(status_code=503, detail="No telemetry data available.")

    df = _telemetry_rows_to_df(all_rows)
    featured = engineer_features(df)
    featured = _add_default_flag_columns(featured)

    tire_vehicle_map = repository.get_tire_vehicle_map(session)
    latest_per_tire = featured.sort_values("timestamp").groupby("tire_id").tail(1)

    predictions = state.failure_predictor.predict(latest_per_tire)
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    threshold_level = risk_order.get(risk_threshold.upper(), 1)

    alerts = []
    for (_, row), pred in zip(latest_per_tire.iterrows(), predictions):
        if risk_order.get(pred.risk_level, 0) >= threshold_level:
            alerts.append(
                {
                    "tire_id": row["tire_id"],
                    "vehicle_id": tire_vehicle_map.get(row["tire_id"], "unknown"),
                    "failure_probability": pred.failure_probability,
                    "risk_level": pred.risk_level,
                    "timestamp": str(row["timestamp"]),
                }
            )

    alerts.sort(key=lambda a: a["failure_probability"], reverse=True)
    return {"alert_count": len(alerts), "alerts": alerts}