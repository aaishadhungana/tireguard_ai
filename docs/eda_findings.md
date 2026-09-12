# TireGuard AI — Exploratory Data Analysis Findings

> Generated programmatically by `src/data/run_eda.py` from the actual processed dataset. Every number below was computed at run time from the data referenced — none are estimated or illustrative.

## Dataset Overview

- Rows: **115,200**
- Unique tires: **80**
- Unique vehicles: **20**
- Date range: **2026-01-01 00:00:00** to **2026-01-30 23:30:00**
- Overall failure rate: **0.0694%** of rows

## Failure Type Distribution

- `structural_degradation`: 80

![Failure type distribution](figures/failure_type_distribution.png)

**Observation:** as flagged at the end of Milestone 1, the failure-type mix is not balanced across scripted failure modes at current simulator settings — this is visible directly in the chart above and will inform whether Milestone 3 needs class-weighting, SMOTE, or simulator re-tuning before failure-type classification (Milestone 4) is workable.

## Pressure and Temperature vs. Failure Outcome

- Mean true pressure — failed rows: **30.042** psi vs. non-failed rows: **31.1** psi
- Mean true temperature — failed rows: **28.665** °C vs. non-failed rows: **27.421** °C

![Pressure by failure](figures/pressure_by_failure.png)

![Temperature by failure](figures/temperature_by_failure.png)

## Tread Depth Trajectories (Sample Tires)

![Tread depth trajectories](figures/tread_depth_trajectories.png)

**Observation:** trajectories show gradual wear punctuated by sharp resets — the resets are the maintenance/replacement events introduced in the Milestone 1 fix (a failed tire's tread is reset rather than continuing to wear past the legal minimum indefinitely).

## Sensor Fault Distribution

![Sensor fault distribution](figures/sensor_fault_distribution.png)

- `pressure_sensor_fault`: {'none': 80899, 'drift': 29091, 'stuck': 3911, 'spoofed': 661, 'missing': 638}
- `temperature_sensor_fault`: {'none': 79300, 'drift': 31197, 'stuck': 3491, 'missing': 615, 'spoofed': 597}
- `tread_sensor_fault`: {'none': 77855, 'drift': 32423, 'stuck': 3711, 'missing': 619, 'spoofed': 592}

**Important finding:** `drift` alone accounts for **25.3%** of `pressure_sensor_fault` rows — far above the configured 3% sensor fault rate. This is expected given the simulator's design (drift persists across many consecutive steps once triggered, with only a small per-step chance of self-correcting), but it means the `sensor_fault_rate` config parameter does NOT directly correspond to "percent of rows affected" — it's closer to "percent of steps where a NEW fault episode begins." Worth fixing the parameter's naming/semantics or the persistence model before Milestone 7 relies on it.

## Missing-Value Imputation Summary

- `pressure`: 0.554% of rows were imputed (forward/backward-filled per tire)
- `temperature`: 0.534% of rows were imputed (forward/backward-filled per tire)
- `tread_depth`: 0.537% of rows were imputed (forward/backward-filled per tire)

## Correlation Between Core Variables

![Correlation heatmap](figures/correlation_heatmap.png)

- Pressure–temperature correlation (true values): **-0.259** — negative, consistent with the simulator's underinflation→heat causal rule (lower pressure is associated with higher temperature).

## Descriptive Statistics

| Variable | Mean | Std | Min | Max |
|---|---|---|---|---|
| pressure_true | 31.099 | 1.129 | 28.024 | 33.987 |
| temperature_true | 27.422 | 3.891 | 15.909 | 43.289 |
| tread_depth_true | 6.113 | 2.267 | 1.588 | 9.997 |
| speed | 70.061 | 19.994 | 0.0 | 151.71 |
| load | 499.541 | 80.072 | 172.84 | 892.9 |

## Known Limitations Carried Into Milestone 3

- Failure-type imbalance (see above) may require class-weighting or simulator re-tuning before Milestone 4 (failure-type classification) is meaningful.
- `sensor_malfunction` failures still have no distinct telemetry signature (carried over from Milestone 1) — worth deciding whether to fix in the simulator or simply exclude this failure type from Milestone 4's target classes.
- The `sensor_fault_rate` config parameter's effective meaning (see drift finding above) should be documented or fixed before Milestone 7/8 tune detection thresholds against it.