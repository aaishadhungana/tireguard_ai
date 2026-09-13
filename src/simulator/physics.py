from __future__ import annotations

from dataclasses import dataclass

NOMINAL_PRESSURE_PSI = 32.0
MIN_SAFE_PRESSURE_PSI = 24.0
NOMINAL_LOAD_KG = 500.0
MAX_RATED_LOAD_KG = 900.0
AMBIENT_TEMP_C = 22.0
NEW_TREAD_DEPTH_MM = 10.0
LEGAL_MIN_TREAD_DEPTH_MM = 1.6

ROAD_TYPE_WEAR_FACTOR = {
    "highway": 0.8,
    "urban": 1.0,
    "rural": 1.1,
    "off_road": 1.6,
}

WEATHER_TEMP_OFFSET_C = {
    "clear": 0.0,
    "rain": -4.0,
    "snow": -8.0,
    "extreme_heat": 10.0,
}


@dataclass
class TireState:
    pressure: float
    temperature: float
    tread_depth: float
    mileage: float
    degradation: float  # 0 = new, 1 = end-of-life. Internal, not a raw sensor.


def step_pressure(state: TireState, has_puncture: bool, dt_hours: float) -> float:
    if has_puncture:
        return max(0.0, state.pressure - state.pressure * 0.6)

    natural_leak = 0.1 * (dt_hours / 24.0)
    return max(0.0, state.pressure - natural_leak)


def compute_target_temperature(
    pressure: float,
    speed_kmh: float,
    load_kg: float,
    weather: str,
    braking_events: int,
) -> float:
    underinflation_ratio = max(0.0, (NOMINAL_PRESSURE_PSI - pressure) / NOMINAL_PRESSURE_PSI)
    overload_ratio = max(0.0, (load_kg - NOMINAL_LOAD_KG) / NOMINAL_LOAD_KG)
    speed_factor = max(0.0, (speed_kmh - 60.0) / 100.0)

    heat_from_underinflation = underinflation_ratio * 35.0
    heat_from_overload = overload_ratio * 20.0
    heat_from_speed = speed_factor * 15.0
    heat_from_braking = braking_events * 1.5

    weather_offset = WEATHER_TEMP_OFFSET_C.get(weather, 0.0)

    return (
        AMBIENT_TEMP_C
        + weather_offset
        + heat_from_underinflation
        + heat_from_overload
        + heat_from_speed
        + heat_from_braking
    )


def step_temperature(
    state: TireState,
    speed_kmh: float,
    load_kg: float,
    weather: str,
    braking_events: int,
) -> float:
    """Temperature responds to underinflation, overload, speed, weather,
    and hard braking. This is the central causal hub: most downstream
    failure modes trace back through temperature."""
    target_temp = compute_target_temperature(
        state.pressure, speed_kmh, load_kg, weather, braking_events
    )

    return state.temperature + (target_temp - state.temperature) * 0.4


def step_tread_wear(
    state: TireState,
    distance_km: float,
    road_type: str,
    load_kg: float,
    speed_kmh: float,
) -> float:
    """Tread wears down as a function of distance, road roughness, load,
    and speed. Higher temperature (already influenced by underinflation
    upstream) also accelerates wear — this is the
    underinflation -> heat -> wear chain from the spec."""
    road_factor = ROAD_TYPE_WEAR_FACTOR.get(road_type, 1.0)
    load_factor = 1.0 + max(0.0, (load_kg - NOMINAL_LOAD_KG) / NOMINAL_LOAD_KG)
    speed_factor = 1.0 + max(0.0, (speed_kmh - 80.0) / 200.0)
    temp_factor = 1.0 + max(0.0, (state.temperature - 60.0) / 100.0)

    wear_mm = distance_km * 0.000177 * road_factor * load_factor * speed_factor * temp_factor
    return max(0.0, state.tread_depth - wear_mm)


def compute_degradation(state: TireState) -> float:

    tread_component = 1.0 - max(
        0.0,
        min(
            1.0,
            (state.tread_depth - LEGAL_MIN_TREAD_DEPTH_MM)
            / (NEW_TREAD_DEPTH_MM - LEGAL_MIN_TREAD_DEPTH_MM),
        ),
    )
    heat_component = max(0.0, min(1.0, (state.temperature - 40.0) / 80.0))

    return max(0.0, min(1.0, 0.7 * tread_component + 0.3 * heat_component))