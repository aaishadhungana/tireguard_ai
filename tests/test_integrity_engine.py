from __future__ import annotations

import pandas as pd
import pytest

from src.models.integrity_engine import (
    IMPLAUSIBLE_PRESSURE_MAX_PSI,
    check_absolute_implausibility,
    check_physics_consistency,
    check_replay_pattern,
    compute_integrity_score,
)


def make_row(**overrides) -> pd.Series:
    base = dict(
        pressure=32.0,
        temperature=25.0,
        speed=70.0,
        load=500.0,
        weather="clear",
        braking_events=0,
    )
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------
# Absolute implausibility
# ---------------------------------------------------------------------

def test_implausibility_triggers_on_high_pressure():
    row = make_row(pressure=80.0)
    result = check_absolute_implausibility(row)
    assert result.triggered
    assert result.source == "engineering_rule"


def test_implausibility_does_not_trigger_on_normal_pressure():
    row = make_row(pressure=32.0)
    result = check_absolute_implausibility(row)
    assert not result.triggered


def test_implausibility_boundary_exact_threshold_does_not_trigger():
    row = make_row(pressure=IMPLAUSIBLE_PRESSURE_MAX_PSI)
    result = check_absolute_implausibility(row)
    assert not result.triggered


def test_implausibility_handles_missing_values_gracefully():
    row = make_row(pressure=float("nan"))
    result = check_absolute_implausibility(row)  # must not raise
    assert not result.triggered


# ---------------------------------------------------------------------
# Physics consistency
# ---------------------------------------------------------------------

def test_physics_consistency_flags_large_deviation():
    # Extremely high temperature with normal pressure/speed/load — not
    # what physics would predict.
    row = make_row(pressure=32.0, speed=50.0, load=500.0, temperature=120.0)
    result = check_physics_consistency(row)
    assert result.triggered


def test_physics_consistency_passes_for_consistent_reading():
    row = make_row(pressure=32.0, speed=50.0, load=500.0, weather="clear", braking_events=0, temperature=22.0)
    result = check_physics_consistency(row)
    assert not result.triggered


def test_physics_consistency_handles_missing_values_gracefully():
    row = make_row(temperature=float("nan"))
    result = check_physics_consistency(row)  # must not raise
    assert not result.triggered


# ---------------------------------------------------------------------
# Replay pattern
# ---------------------------------------------------------------------

def test_replay_triggers_on_identical_run():
    values = [32.0, 32.1, 30.0, 30.0, 30.0, 30.0]
    result = check_replay_pattern(values)
    assert result.triggered


def test_replay_does_not_trigger_on_varying_values():
    values = [32.0, 31.8, 31.5, 31.2, 30.9]
    result = check_replay_pattern(values)
    assert not result.triggered


def test_replay_does_not_trigger_with_insufficient_history():
    values = [30.0, 30.0]
    result = check_replay_pattern(values)
    assert not result.triggered


# ---------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------

def test_composite_score_high_for_clean_row_and_normal_ml_score():
    row = make_row()
    report = compute_integrity_score(row, ml_anomaly_score=0.95)
    assert report.integrity_score > 0.7
    assert report.risk_level == "LOW"


def test_composite_score_drops_when_rule_triggers():
    row = make_row(pressure=80.0)
    report = compute_integrity_score(row, ml_anomaly_score=0.95)
    assert report.integrity_score < 0.95  # penalized relative to ML score alone
    assert any("absolute_implausibility" in r for r in report.reasons)


def test_composite_score_bounded_between_zero_and_one():
    row = make_row(pressure=80.0, temperature=200.0)
    report = compute_integrity_score(row, ml_anomaly_score=0.01)
    assert 0.0 <= report.integrity_score <= 1.0


def test_composite_score_reports_no_trigger_message_when_clean():
    row = make_row()
    report = compute_integrity_score(row, ml_anomaly_score=0.9)
    assert report.reasons == ["No rule-based integrity checks triggered."]


def test_caveat_is_always_present_and_not_overwritten():
    row = make_row(pressure=80.0)
    report = compute_integrity_score(row, ml_anomaly_score=0.5)
    assert "not evidence of an actual cyberattack" in report.caveat.lower() or "not evidence" in report.caveat.lower()