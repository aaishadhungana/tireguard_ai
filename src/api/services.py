from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from src.streaming.message_schema import row_to_message
from src.streaming.subscriber import assemble_feature_row
from src.streaming.feature_buffer import TireFeatureBuffer


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


def get_latest_tire_row(tire_id, state):
    rows = state.get_tire_rows(tire_id)
    return rows.iloc[-1]


def get_root_cause_for_tire(tire_id, state):
    if state.root_cause_explainer is None:
        raise HTTPException(status_code=503, detail="Root cause explainer is not loaded.")

    latest_row = get_latest_tire_row(tire_id, state)
    from src.models.failure_model import build_feature_matrix

    row_df = pd.DataFrame([latest_row])
    X, _ = build_feature_matrix(row_df)
    explanation = state.root_cause_explainer.explain(X.iloc[0])

    return {
        "tire_id": tire_id,
        "failure_probability": explanation.failure_probability,
        "risk_level": explanation.risk_level,
        "top_model_contributions": [
            {
                "feature": c.feature,
                "value": c.value,
                "contribution": c.contribution,
                "direction": c.direction,
                "source": c.source,
            }
            for c in explanation.top_model_contributions
        ],
        "triggered_engineering_rules": [
            {
                "feature": c.feature,
                "value": c.value,
                "contribution": c.contribution,
                "direction": c.direction,
                "source": c.source,
            }
            for c in explanation.triggered_engineering_rules
        ],
        "caveat": explanation.caveat,
    }


def get_rul_for_tire(tire_id, state):
    if state.rul_model is None:
        raise HTTPException(status_code=503, detail="RUL model is not loaded.")

    latest_row = get_latest_tire_row(tire_id, state)
    from src.models.rul_model import NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES

    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    row_df = pd.DataFrame([latest_row])[feature_cols]
    predicted_rul = float(state.rul_model.predict(row_df)[0])

    return {
        "tire_id": tire_id,
        "estimated_rul_km": round(max(0.0, predicted_rul), 1),
        "model_type": type(state.rul_model.named_steps["model"]).__name__,
    }


def get_tire_history(tire_id, state, limit):
    rows = state.get_tire_rows(tire_id).tail(limit)
    return {
        "tire_id": tire_id,
        "row_count": len(rows),
        "rows": [
            {
                "timestamp": str(r["timestamp"]),
                "pressure": float(r["pressure"]),
                "temperature": float(r["temperature"]),
                "tread_depth": float(r["tread_depth"]),
                "speed": float(r["speed"]),
                "load": float(r["load"]),
                "failure": int(r["failure"]),
            }
            for _, r in rows.iterrows()
        ],
    }


def get_fleet_stats(state):
    if state.dataset is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    df = state.dataset
    return {
        "total_tires": int(df["tire_id"].nunique()),
        "total_vehicles": int(df["vehicle_id"].nunique()),
        "total_rows": len(df),
        "failure_rate_pct": round(100 * df["failure"].mean(), 4),
        "failure_type_counts": df.loc[df["failure"] == 1, "failure_type"].value_counts().to_dict(),
    }


def get_alerts(state, risk_threshold="MEDIUM"):
    if state.dataset is None or state.failure_predictor is None:
        raise HTTPException(status_code=503, detail="Dataset or model not loaded.")

    df = state.dataset
    latest_per_tire = df.sort_values("timestamp").groupby("tire_id").tail(1)

    predictions = state.failure_predictor.predict(latest_per_tire)
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    threshold_level = risk_order.get(risk_threshold.upper(), 1)

    alerts = []
    for (_, row), pred in zip(latest_per_tire.iterrows(), predictions):
        if risk_order.get(pred.risk_level, 0) >= threshold_level:
            alerts.append(
                {
                    "tire_id": row["tire_id"],
                    "vehicle_id": row["vehicle_id"],
                    "failure_probability": pred.failure_probability,
                    "risk_level": pred.risk_level,
                    "timestamp": str(row["timestamp"]),
                }
            )

    alerts.sort(key=lambda a: a["failure_probability"], reverse=True)
    return {"alert_count": len(alerts), "alerts": alerts}