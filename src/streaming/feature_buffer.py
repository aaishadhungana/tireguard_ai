from __future__ import annotations

from collections import defaultdict, deque

from src.models.failure_model import NUMERIC_FEATURES  # noqa: F401 (documents the target feature set)

ROLLING_WINDOW = 6  # must match src/data/features.py's ROLLING_WINDOW exactly

_TRACKED_SENSORS = ("pressure", "temperature")


class TireFeatureBuffer:
    """Holds recent raw readings for ONE tire and computes lag/delta/
    rolling features for each new reading as it arrives."""

    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
        self._history = {sensor: deque(maxlen=window) for sensor in _TRACKED_SENSORS}
        self._tread_history: deque = deque(maxlen=window)
        self._braking_history: deque = deque(maxlen=window)
        self._reading_index = 0

    def add_reading(self, pressure: float, temperature: float, tread_depth: float, braking_events: int) -> dict:
        """Adds one new reading and returns the computed feature dict
        for THIS reading — matching src/data/features.py's column names
        exactly (pressure_prev, pressure_delta, pressure_roll_mean_6,
        etc.), computed from history BEFORE this reading is added to it
        (so a delta/rolling stat never includes the current value twice
        — mirrors pandas' shift(1)-before-rolling semantics)."""
        features: dict = {}

        for sensor, value in (("pressure", pressure), ("temperature", temperature)):
            history = self._history[sensor]
            if len(history) == 0:
                features[f"{sensor}_prev"] = None
                features[f"{sensor}_delta"] = None
            else:
                prev_value = history[-1]
                features[f"{sensor}_prev"] = prev_value
                features[f"{sensor}_delta"] = value - prev_value

            window_values = (list(history) + [value])[-self.window:]
            features[f"{sensor}_roll_mean_{self.window}"] = sum(window_values) / len(window_values)
            if len(window_values) > 1:
                mean = features[f"{sensor}_roll_mean_{self.window}"]
                variance = sum((v - mean) ** 2 for v in window_values) / (len(window_values) - 1)
                features[f"{sensor}_roll_std_{self.window}"] = variance ** 0.5
            else:
                features[f"{sensor}_roll_std_{self.window}"] = None  # matches pandas: std of 1 value is NaN

            history.append(value)

        if len(self._tread_history) == 0:
            features["tread_depth_prev"] = None
            features["tread_depth_delta"] = None
        else:
            prev_tread = self._tread_history[-1]
            features["tread_depth_prev"] = prev_tread
            features["tread_depth_delta"] = tread_depth - prev_tread
        self._tread_history.append(tread_depth)

        braking_window = (list(self._braking_history) + [braking_events])[-self.window:]
        features[f"braking_events_roll_sum_{self.window}"] = sum(braking_window)
        self._braking_history.append(braking_events)

        features["tire_reading_index"] = self._reading_index
        self._reading_index += 1

        return features


class StreamingFeatureBuilder:
    """Manages one TireFeatureBuffer per tire_id seen so far. This is
    the stateful object a long-running subscriber process holds onto
    across the whole stream."""

    def __init__(self):
        self._buffers: dict = defaultdict(TireFeatureBuffer)

    def process_reading(
        self, tire_id: str, pressure: float, temperature: float, tread_depth: float, braking_events: int
    ) -> dict:
        buffer = self._buffers[tire_id]
        return buffer.add_reading(pressure, temperature, tread_depth, braking_events)

    def known_tire_count(self) -> int:
        return len(self._buffers)