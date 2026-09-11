from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

SENSOR_FAULT_TYPES = [
    "none",
    "stuck",
    "missing",
    "drift",
    "spoofed",
]


@dataclass
class SensorFaultResult:
    value: Optional[float] 
    fault_type: str


class SensorFaultInjector:
    def __init__(self, fault_rate: float, rng: random.Random):
        self.fault_rate = fault_rate
        self.rng = rng
        self._stuck_value: Optional[float] = None
        self._stuck_remaining: int = 0
        self._drift_offset: float = 0.0

    def apply(self, true_value: float) -> SensorFaultResult:
        if self._stuck_remaining > 0:
            self._stuck_remaining -= 1
            return SensorFaultResult(value=self._stuck_value, fault_type="stuck")

        if self._drift_offset != 0.0:
            self._drift_offset += self.rng.uniform(0.0, 0.05) * abs(self._drift_offset or 1.0)
            if self.rng.random() < 0.02:  
                self._drift_offset = 0.0
            return SensorFaultResult(value=true_value + self._drift_offset, fault_type="drift")

        if self.rng.random() >= self.fault_rate:
            return SensorFaultResult(value=true_value, fault_type="none")

        fault_type = self.rng.choice(["stuck", "missing", "drift", "spoofed"])

        if fault_type == "stuck":
            self._stuck_value = true_value
            self._stuck_remaining = self.rng.randint(2, 8)
            return SensorFaultResult(value=self._stuck_value, fault_type="stuck")

        if fault_type == "missing":
            return SensorFaultResult(value=None, fault_type="missing")

        if fault_type == "drift":
            self._drift_offset = self.rng.uniform(0.5, 3.0) * self.rng.choice([-1, 1])
            return SensorFaultResult(value=true_value + self._drift_offset, fault_type="drift")

        spoofed_value = true_value * self.rng.uniform(1.8, 3.0)
        return SensorFaultResult(value=spoofed_value, fault_type="spoofed")