from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.simulator.faults import SensorFaultInjector
from src.simulator.physics import (
    AMBIENT_TEMP_C,
    LEGAL_MIN_TREAD_DEPTH_MM,
    MIN_SAFE_PRESSURE_PSI,
    NEW_TREAD_DEPTH_MM,
    NOMINAL_LOAD_KG,
    NOMINAL_PRESSURE_PSI,
    TireState,
    compute_degradation,
    step_pressure,
    step_temperature,
    step_tread_wear,
)

OVERHEATING_FAILURE_TEMP_C = 110.0

ROAD_TYPES = ["highway", "urban", "rural", "off_road"]
WEATHER_TYPES = ["clear", "rain", "snow", "extreme_heat"]

FAILURE_TYPES = [
    "underinflation",
    "overheating",
    "overloading",
    "puncture",
    "structural_degradation",
    "sensor_malfunction",
]


@dataclass
class SimulatorConfig:
    num_vehicles: int
    tires_per_vehicle: int
    duration_days: int
    sampling_interval_min: int
    failure_rate: float
    sensor_fault_rate: float = 0.03
    random_seed: int = 42


@dataclass
class _TireContext:
    tire_id: str
    vehicle_id: str
    state: TireState
    pressure_fault: SensorFaultInjector
    temperature_fault: SensorFaultInjector
    tread_fault: SensorFaultInjector
    mileage: float = 0.0
    braking_history: int = 0
    maintenance_history: int = field(default=0)

    scripted_failure_type: Optional[str] = None
    scripted_failure_step: Optional[int] = None


class TireTelemetrySimulator:
  
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.np_rng = np.random.default_rng(config.random_seed)
        self.steps_per_day = max(1, (24 * 60) // config.sampling_interval_min)
        self.total_steps = self.steps_per_day * config.duration_days
        self.dt_hours = config.sampling_interval_min / 60.0

    def _build_fleet(self) -> list[_TireContext]:
        total_tires = self.config.num_vehicles * self.config.tires_per_vehicle
        tire_keys = [
            (v, t)
            for v in range(self.config.num_vehicles)
            for t in range(self.config.tires_per_vehicle)
        ]

        num_scripted = round(self.config.failure_rate * total_tires)
        scripted_indices = set(self.rng.sample(range(total_tires), k=min(num_scripted, total_tires)))

        type_cycle = FAILURE_TYPES.copy()
        self.rng.shuffle(type_cycle)
        scripted_type_by_index: dict[int, str] = {}
        for i, tire_index in enumerate(sorted(scripted_indices)):
            scripted_type_by_index[tire_index] = type_cycle[i % len(type_cycle)]

        fleet: list[_TireContext] = []
        for tire_index, (v, t) in enumerate(tire_keys):
            vehicle_id = f"V-{v:04d}"
            tire_id = f"{vehicle_id}-T{t}"
            initial_state = TireState(
                pressure=self.rng.uniform(30.0, 34.0),
                temperature=self.rng.uniform(20.0, 28.0),
                tread_depth=self.rng.uniform(7.0, NEW_TREAD_DEPTH_MM),
                mileage=0.0,
                degradation=0.0,
            )

            will_fail = tire_index in scripted_indices
            failure_type = scripted_type_by_index.get(tire_index)

            lower_bound = max(1, int(self.total_steps * 0.12))
            upper_bound = max(lower_bound + 1, int(self.total_steps * 0.45))
            lower_bound = min(lower_bound, self.total_steps - 2)
            upper_bound = min(upper_bound, self.total_steps - 1)
            if upper_bound <= lower_bound:
                upper_bound = lower_bound + 1
            failure_step = self.rng.randint(lower_bound, upper_bound) if will_fail else None

            fleet.append(
                _TireContext(
                    tire_id=tire_id,
                    vehicle_id=vehicle_id,
                    state=initial_state,
                    scripted_failure_type=failure_type,
                    scripted_failure_step=failure_step,
                    pressure_fault=SensorFaultInjector(
                        self.config.sensor_fault_rate, self.rng
                    ),
                    temperature_fault=SensorFaultInjector(
                        self.config.sensor_fault_rate, self.rng
                    ),
                    tread_fault=SensorFaultInjector(
                        self.config.sensor_fault_rate, self.rng
                    ),
                )
            )
        return fleet

    def _driving_conditions(self, ctx: _TireContext, step: int) -> dict:
        """Sample this timestep's driving conditions. If this tire is
        scripted to fail via a stress-driven mode (overloading,
        overheating, underinflation), conditions are biased toward that
        stress as the failure step approaches, so the failure emerges
        causally rather than being stamped on at the end."""
        road_type = self.rng.choice(ROAD_TYPES)
        weather = self.rng.choice(WEATHER_TYPES)
        speed_kmh = max(0.0, self.np_rng.normal(70, 20))
        load_kg = max(50.0, self.np_rng.normal(NOMINAL_LOAD_KG, 80))
        braking_events = self.np_rng.poisson(1.0)

        approaching_failure = (
            ctx.scripted_failure_step is not None
            and ctx.scripted_failure_step - 200 <= step <= ctx.scripted_failure_step
        )
        if approaching_failure:
            progress = 1.0 - max(0.0, (ctx.scripted_failure_step - step)) / 200.0
            if ctx.scripted_failure_type == "overloading":
                load_kg += progress * 500.0
            elif ctx.scripted_failure_type == "overheating":
                speed_kmh += progress * 60.0
            elif ctx.scripted_failure_type == "underinflation":
                # Handled via direct pressure decay below.
                pass

        has_puncture = (
            ctx.scripted_failure_type == "puncture" and step == ctx.scripted_failure_step
        )

        return dict(
            road_type=road_type,
            weather=weather,
            speed_kmh=speed_kmh,
            load_kg=load_kg,
            braking_events=int(braking_events),
            has_puncture=has_puncture,
        )

    def _apply_underinflation_script(self, ctx: _TireContext, step: int) -> None:
        if ctx.scripted_failure_type != "underinflation":
            return
        if ctx.scripted_failure_step is None:
            return
        if step > ctx.scripted_failure_step - 300:

            ctx.state.pressure = max(5.0, ctx.state.pressure - 0.03)

    def _label_failure(self, ctx: _TireContext, step: int, degradation: float) -> tuple[bool, str]:
        if ctx.scripted_failure_step == step:
            return True, ctx.scripted_failure_type

        if ctx.scripted_failure_step is not None and step < ctx.scripted_failure_step:
            return False, "none"

        if ctx.state.tread_depth <= LEGAL_MIN_TREAD_DEPTH_MM:
            return True, "structural_degradation"

        if ctx.state.temperature >= OVERHEATING_FAILURE_TEMP_C:
            return True, "overheating"

        if ctx.state.pressure <= MIN_SAFE_PRESSURE_PSI * 0.5:
            return True, "underinflation"

        return False, "none"

    def generate(self) -> pd.DataFrame:
        fleet = self._build_fleet()
        records: list[dict] = []

        start_ts = pd.Timestamp("2026-01-01")

        for step in range(self.total_steps):
            timestamp = start_ts + pd.Timedelta(minutes=step * self.config.sampling_interval_min)

            for ctx in fleet:
                conditions = self._driving_conditions(ctx, step)
                self._apply_underinflation_script(ctx, step)

                true_pressure = step_pressure(
                    ctx.state, conditions["has_puncture"], self.dt_hours
                )
                ctx.state.pressure = true_pressure

                true_temperature = step_temperature(
                    ctx.state,
                    speed_kmh=conditions["speed_kmh"],
                    load_kg=conditions["load_kg"],
                    weather=conditions["weather"],
                    braking_events=conditions["braking_events"],
                )
                ctx.state.temperature = true_temperature

                distance_km = conditions["speed_kmh"] * self.dt_hours
                true_tread = step_tread_wear(
                    ctx.state,
                    distance_km=distance_km,
                    road_type=conditions["road_type"],
                    load_kg=conditions["load_kg"],
                    speed_kmh=conditions["speed_kmh"],
                )
                ctx.state.tread_depth = true_tread
                ctx.mileage += distance_km
                ctx.braking_history += conditions["braking_events"]

                degradation = compute_degradation(ctx.state)
                ctx.state.degradation = degradation

                failure, failure_type = self._label_failure(ctx, step, degradation)
                if failure:
                    ctx.maintenance_history += 1
                    ctx.scripted_failure_step = None
                    ctx.state.tread_depth = NEW_TREAD_DEPTH_MM
                    ctx.state.pressure = NOMINAL_PRESSURE_PSI
                    ctx.state.temperature = AMBIENT_TEMP_C

                pressure_obs = ctx.pressure_fault.apply(true_pressure)
                temperature_obs = ctx.temperature_fault.apply(true_temperature)
                tread_obs = ctx.tread_fault.apply(true_tread)

                records.append(
                    {
                        "tire_id": ctx.tire_id,
                        "vehicle_id": ctx.vehicle_id,
                        "timestamp": timestamp,
                        "pressure": pressure_obs.value,
                        "temperature": temperature_obs.value,
                        "speed": round(conditions["speed_kmh"], 2),
                        "load": round(conditions["load_kg"], 2),
                        "tread_depth": tread_obs.value,
                        "mileage": round(ctx.mileage, 2),
                        "braking_events": conditions["braking_events"],
                        "acceleration": round(self.np_rng.normal(0, 1.5), 3),
                        "road_type": conditions["road_type"],
                        "weather": conditions["weather"],
                        "maintenance_history": ctx.maintenance_history,
                        "failure": int(failure),
                        "failure_type": failure_type,
            
                        "pressure_sensor_fault": pressure_obs.fault_type,
                        "temperature_sensor_fault": temperature_obs.fault_type,
                        "tread_sensor_fault": tread_obs.fault_type,
                        "pressure_true": round(true_pressure, 3),
                        "temperature_true": round(true_temperature, 3),
                        "tread_depth_true": round(true_tread, 3),
                    }
                )

        df = pd.DataFrame.from_records(records)
        df["is_synthetic"] = True
        return df