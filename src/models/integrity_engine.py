"""
Sensor / Cyber Integrity Engine.

WHAT THIS DOES NOT CLAIM (required reading before using this module):
every check here is a PATTERN MATCH against known simulated fault/attack
signatures. A physically-impossible reading could be a spoofing attempt,
an exotic hardware failure, or a data-pipeline bug, this module can say
"this looks like the kind of thing spoofing would produce," never "this
tire was hacked." No output of this module is evidence of an actual
cyberattack.

THREE SEPARATE, EXPLICITLY-LABELED SIGNALS combined into one score:

1. "absolute_implausibility" (engineering_rule): a tighter threshold
   than Milestone 2's generic validation range. Found by inspecting
   real data: Milestone 2's pressure range (0-100 psi) catches only
   0.3% of actual spoofed pressure readings (which run 53-101 psi) —
   far too loose, since a real tire practically never exceeds the low
   40s psi even severely overinflated. A domain-specific 50 psi ceiling
   catches 100% of this dataset's spoofed pressure values.

2. "physics_consistency" (engineering_rule): reuses
   physics.py's compute_target_temperature exactly (not a re-derived
   approximation) to check whether temperature is consistent with what
   pressure/load/speed/weather would predict. HONEST LIMITATION FOUND
   BY TESTING (not assumed): this check does NOT discriminate this
   dataset's spoofed rows from clean ones (mean deviation 4.79 vs 4.08
   -- barely different). The simulator's spoof multiplier always
   INFLATES pressure, but the underinflation-heat rule only responds to
   LOW pressure -- there's no "overinflation causes heat" rule for it to
   violate. This check is retained because it's still the physically
   correct thing to test and WOULD catch a different spoofing direction
   (artificially low pressure), but it is not doing meaningful work on
   THIS dataset's spoof pattern, and the evaluation report says so.

3. "ml_anomaly" (model_contribution): Milestone 7's Isolation Forest
   anomaly score, reused rather than duplicated.

Sensor Integrity Score: 0 (untrustworthy) to 1 (fully trustworthy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.simulator.physics import compute_target_temperature

IMPLAUSIBLE_PRESSURE_MAX_PSI = 50.0
IMPLAUSIBLE_PRESSURE_MIN_PSI = 10.0
IMPLAUSIBLE_TEMPERATURE_MAX_C = 150.0
PHYSICS_CONSISTENCY_DEVIATION_THRESHOLD_C = 20.0
REPLAY_MIN_CONSECUTIVE_IDENTICAL = 4


@dataclass
class IntegrityCheckResult:
    check_name: str
    triggered: bool
    detail: str
    source: str  # "engineering_rule" | "model_contribution"


@dataclass
class SensorIntegrityReport:
    integrity_score: float  # 0 (untrustworthy) - 1 (fully trustworthy)
    risk_level: str  # "LOW" | "MEDIUM" | "HIGH"
    reasons: list = field(default_factory=list)
    caveat: str = (
        "This score reflects pattern-matching against known simulated fault "
        "signatures. It is NOT evidence of an actual cyberattack or intrusion -- "
        "a flagged reading could equally be a hardware fault or data pipeline bug."
    )


def check_absolute_implausibility(row: pd.Series) -> IntegrityCheckResult:
    pressure = row.get("pressure")
    if pressure is not None and not pd.isna(pressure):
        if pressure > IMPLAUSIBLE_PRESSURE_MAX_PSI or pressure < IMPLAUSIBLE_PRESSURE_MIN_PSI:
            return IntegrityCheckResult(
                check_name="absolute_implausibility",
                triggered=True,
                detail=f"pressure={pressure:.1f} psi is outside the physically plausible "
                f"range [{IMPLAUSIBLE_PRESSURE_MIN_PSI}, {IMPLAUSIBLE_PRESSURE_MAX_PSI}] psi",
                source="engineering_rule",
            )

    temperature = row.get("temperature")
    if temperature is not None and not pd.isna(temperature) and temperature > IMPLAUSIBLE_TEMPERATURE_MAX_C:
        return IntegrityCheckResult(
            check_name="absolute_implausibility",
            triggered=True,
            detail=f"temperature={temperature:.1f}C exceeds physically plausible maximum "
            f"({IMPLAUSIBLE_TEMPERATURE_MAX_C}C)",
            source="engineering_rule",
        )

    return IntegrityCheckResult("absolute_implausibility", False, "within plausible range", "engineering_rule")


def check_physics_consistency(row: pd.Series) -> IntegrityCheckResult:
    required = ["pressure", "speed", "load", "weather", "braking_events", "temperature"]
    if any(pd.isna(row.get(c)) for c in required):
        return IntegrityCheckResult("physics_consistency", False, "insufficient data to check", "engineering_rule")

    expected_temp = compute_target_temperature(
        pressure=row["pressure"],
        speed_kmh=row["speed"],
        load_kg=row["load"],
        weather=row["weather"],
        braking_events=row["braking_events"],
    )
    deviation = abs(row["temperature"] - expected_temp)

    if deviation > PHYSICS_CONSISTENCY_DEVIATION_THRESHOLD_C:
        return IntegrityCheckResult(
            check_name="physics_consistency",
            triggered=True,
            detail=f"temperature={row['temperature']:.1f}C deviates {deviation:.1f}C from the "
            f"~{expected_temp:.1f}C expected given current pressure/load/speed/weather",
            source="engineering_rule",
        )

    return IntegrityCheckResult(
        "physics_consistency", False, f"consistent with physics (deviation {deviation:.1f}C)", "engineering_rule"
    )


def check_replay_pattern(recent_values: list) -> IntegrityCheckResult:
    """recent_values: the last N readings of ONE sensor for ONE tire, in
    time order, most recent last. Flags a suspiciously long run of
    EXACTLY identical values -- longer than ordinary 'stuck sensor'
    noise would typically produce within a short rolling window."""
    if len(recent_values) < REPLAY_MIN_CONSECUTIVE_IDENTICAL:
        return IntegrityCheckResult("replay_pattern", False, "insufficient history to check", "engineering_rule")

    last_n = recent_values[-REPLAY_MIN_CONSECUTIVE_IDENTICAL:]
    if len(set(last_n)) == 1:
        return IntegrityCheckResult(
            check_name="replay_pattern",
            triggered=True,
            detail=f"last {REPLAY_MIN_CONSECUTIVE_IDENTICAL} readings are exactly identical ({last_n[0]})",
            source="engineering_rule",
        )

    return IntegrityCheckResult("replay_pattern", False, "no suspicious repetition", "engineering_rule")


def _risk_level(score: float) -> str:
    if score < 0.4:
        return "HIGH"
    if score < 0.7:
        return "MEDIUM"
    return "LOW"


def compute_integrity_score(row: pd.Series, ml_anomaly_score: float, recent_pressure_values=None) -> SensorIntegrityReport:
    """ml_anomaly_score: Milestone 7's Isolation Forest score, rescaled
    to [0, 1] where 0 = most anomalous, 1 = most normal (caller's
    responsibility -- see run_integrity_engine.py for the rescaling)."""
    checks = [
        check_absolute_implausibility(row),
        check_physics_consistency(row),
    ]
    if recent_pressure_values is not None:
        checks.append(check_replay_pattern(recent_pressure_values))

    triggered_checks = [c for c in checks if c.triggered]

    score = ml_anomaly_score
    for _ in triggered_checks:
        score *= 0.4
    score = max(0.0, min(1.0, score))

    reasons = [f"{c.check_name}: {c.detail}" for c in triggered_checks]
    if not reasons:
        reasons = ["No rule-based integrity checks triggered."]

    return SensorIntegrityReport(
        integrity_score=round(score, 4),
        risk_level=_risk_level(score),
        reasons=reasons,
    )