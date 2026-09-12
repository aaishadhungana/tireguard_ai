from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.explain import (
    Contribution,
    RootCauseExplainer,
    _check_engineering_rules,
)
from src.models.failure_model import build_feature_matrix, build_random_forest_pipeline
from src.simulator.physics import LEGAL_MIN_TREAD_DEPTH_MM, MIN_SAFE_PRESSURE_PSI


def make_training_df(n_tires: int = 20, rows_per_tire: int = 15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tires):
        tire_id = f"TIRE-{t:03d}"
        will_fail = t < 5
        for i in range(rows_per_tire):
            is_failure = int(will_fail and i == rows_per_tire - 1)
            rows.append(
                {
                    "tire_id": tire_id,
                    "vehicle_id": f"VEH-{t // 4:03d}",
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                    "pressure": rng.uniform(18, 33) if will_fail else rng.uniform(28, 33),
                    "temperature": rng.uniform(20, 40),
                    "tread_depth": rng.uniform(1, 3) if will_fail else rng.uniform(5, 9),
                    "speed": rng.uniform(0, 120),
                    "load": rng.uniform(300, 700),
                    "braking_events": rng.integers(0, 3),
                    "acceleration": rng.normal(0, 1),
                    "mileage": i * 30.0,
                    "maintenance_history": is_failure,
                    "failure": is_failure,
                    "failure_type": "structural_degradation" if is_failure else "none",
                    "pressure_prev": rng.uniform(28, 33),
                    "pressure_delta": rng.normal(0, 0.5),
                    "temperature_prev": rng.uniform(20, 40),
                    "temperature_delta": rng.normal(0, 1),
                    "tread_depth_prev": rng.uniform(2, 9),
                    "tread_depth_delta": rng.normal(0, 0.1),
                    "pressure_roll_mean_6": rng.uniform(28, 33),
                    "pressure_roll_std_6": rng.uniform(0, 1),
                    "temperature_roll_mean_6": rng.uniform(20, 40),
                    "temperature_roll_std_6": rng.uniform(0, 2),
                    "braking_events_roll_sum_6": rng.integers(0, 10),
                    "tire_reading_index": i,
                    "load_to_nominal_ratio": rng.uniform(0.5, 1.5),
                    "pressure_deficit": rng.uniform(0, 5),
                    "pressure_was_missing": False,
                    "temperature_was_missing": False,
                    "tread_depth_was_missing": False,
                    "pressure_out_of_range": False,
                    "temperature_out_of_range": False,
                    "tread_depth_out_of_range": False,
                    "road_type": rng.choice(["highway", "urban", "rural", "off_road"]),
                    "weather": rng.choice(["clear", "rain", "snow"]),
                    "pressure_sensor_fault": "none",
                    "temperature_sensor_fault": "none",
                    "tread_sensor_fault": "none",
                    "is_synthetic": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_explainer():
    df = make_training_df()
    X, y = build_feature_matrix(df)
    pipeline = build_random_forest_pipeline()
    pipeline.fit(X, y)
    return RootCauseExplainer(pipeline), X, pipeline


# Core mathematical property
def test_shap_contributions_reconstruct_model_probability(trained_explainer):
    explainer, X, pipeline = trained_explainer
    row = X.iloc[0]

    explanation = explainer.explain(row, top_n=100)  

    actual_proba = pipeline.predict_proba(row.to_frame().T)[0, 1]
    row_df = row.to_frame().T
    X_transformed = explainer.preprocess.transform(row_df)
    shap_values = explainer.explainer.shap_values(X_transformed)
    if shap_values.ndim == 3:
        sv_row = shap_values[0, :, 1]
        base = explainer.explainer.expected_value[1]
    else:
        sv_row = shap_values[0]
        base = explainer.explainer.expected_value
    reconstructed = base + sv_row.sum()

    assert reconstructed == pytest.approx(actual_proba, abs=1e-6)


def test_explanation_probability_matches_pipeline_predict_proba(trained_explainer):
    explainer, X, pipeline = trained_explainer
    row = X.iloc[0]
    explanation = explainer.explain(row)
    actual_proba = pipeline.predict_proba(row.to_frame().T)[0, 1]
    assert explanation.failure_probability == pytest.approx(actual_proba, abs=1e-4)


# Source tagging discipline, the central design requirement
def test_model_contributions_are_tagged_model_contribution(trained_explainer):
    explainer, X, _ = trained_explainer
    row = X.iloc[0]
    explanation = explainer.explain(row)
    for c in explanation.top_model_contributions:
        assert c.source == "model_contribution"


def test_engineering_rules_are_tagged_engineering_rule(trained_explainer):
    explainer, X, _ = trained_explainer
    row = X.iloc[0].copy()
    row["pressure"] = 15.0 
    explanation = explainer.explain(row)
    assert len(explanation.triggered_engineering_rules) > 0
    for c in explanation.triggered_engineering_rules:
        assert c.source == "engineering_rule"


def test_sources_are_never_mixed_in_same_list(trained_explainer):
    explainer, X, _ = trained_explainer
    row = X.iloc[0].copy()
    row["pressure"] = 15.0
    row["tread_depth"] = 1.0
    explanation = explainer.explain(row)

    model_sources = {c.source for c in explanation.top_model_contributions}
    rule_sources = {c.source for c in explanation.triggered_engineering_rules}
    assert model_sources <= {"model_contribution"}
    assert rule_sources <= {"engineering_rule"}

# Engineering rule correctness, exact threshold checks
def test_pressure_rule_triggers_below_threshold():
    row = pd.Series({"pressure": MIN_SAFE_PRESSURE_PSI - 1, "tread_depth": 8.0, "temperature": 25.0, "load": 500.0})
    triggered = _check_engineering_rules(row)
    features_triggered = {c.feature for c in triggered}
    assert "pressure" in features_triggered


def test_pressure_rule_does_not_trigger_above_threshold():
    row = pd.Series({"pressure": MIN_SAFE_PRESSURE_PSI + 5, "tread_depth": 8.0, "temperature": 25.0, "load": 500.0})
    triggered = _check_engineering_rules(row)
    features_triggered = {c.feature for c in triggered}
    assert "pressure" not in features_triggered


def test_tread_rule_triggers_near_legal_minimum():
    row = pd.Series({"pressure": 32.0, "tread_depth": LEGAL_MIN_TREAD_DEPTH_MM, "temperature": 25.0, "load": 500.0})
    triggered = _check_engineering_rules(row)
    features_triggered = {c.feature for c in triggered}
    assert "tread_depth" in features_triggered


def test_no_rules_trigger_for_healthy_row():
    row = pd.Series({"pressure": 32.0, "tread_depth": 8.0, "temperature": 25.0, "load": 500.0})
    triggered = _check_engineering_rules(row)
    assert triggered == []


def test_engineering_rules_handle_missing_values_gracefully():
    row = pd.Series({"pressure": np.nan, "tread_depth": np.nan, "temperature": np.nan, "load": np.nan})
    triggered = _check_engineering_rules(row)  # must not raise
    assert triggered == []

# Categorical feature name mapping
def test_categorical_contributions_map_back_to_original_column(trained_explainer):
    explainer, X, _ = trained_explainer
    row = X.iloc[0]
    explanation = explainer.explain(row, top_n=100)
    feature_names = {c.feature for c in explanation.top_model_contributions}
    for name in feature_names:
        assert "_highway" not in name
        assert "_urban" not in name
        assert "_rural" not in name