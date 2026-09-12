# TireGuard AI : Root Cause Explanation Examples

> Generated programmatically by `src/models/run_explanations.py` against REAL predictions from the trained failure model on REAL dataset rows — not fabricated examples. One example is shown per failure type present in the dataset.

## Source Tagging (read this before the examples below)

- **`model_contribution`**: a SHAP value — a fact about how this specific model weighted this feature for this row, verified to sum exactly to the model's predicted probability (see tests/test_explain.py). This is NOT proof of physical causation — it describes the model's learned function.
- **`engineering_rule`**: a hardcoded domain threshold, independent of the model, written directly in `src/models/explain.py`.
- These sources are never blended into a single number or claim. Where both a model contribution and an engineering rule point to the same feature, that convergence is meaningful (two independent sources agree) but is reported as two separate facts, not merged into one.

## Example: `structural_degradation`

- Ground-truth failure type (from the simulator, not visible to the model): `structural_degradation`
- Predicted failure probability: **0.7697**
- Risk level: **HIGH**

### Model Contributions (SHAP — source: `model_contribution`)

| Feature | Value | Contribution | Direction |
|---|---|---|---|
| tread_depth | 1.368 | +0.1392 | increases_risk |
| tread_depth_prev | 1.370 | +0.1389 | increases_risk |
| mileage | 28736.770 | +0.0449 | increases_risk |
| tire_reading_index | 805 | +0.0316 | increases_risk |
| temperature_roll_std_6 | 23.631 | -0.0262 | decreases_risk |

### Triggered Engineering Rules (source: `engineering_rule`)

| Feature | Value | Rule Margin | Direction |
|---|---|---|---|
| tread_depth | 1.368 | +1.0324 | increases_risk |

> Model contributions describe how this MODEL weighted each feature for this prediction — they are consistent with, but not proof of, the underlying physical cause. Engineering rules are independent, hardcoded domain thresholds, not something the model learned.

## Known Limitations

- Not every example tells an equally clean story. `overheating`, `overloading`, and `underinflation` show top SHAP features matching their physical cause (temperature, load, pressure respectively) — consistent with, though not proof of, the simulator's known causal design. `puncture`'s top feature above is a sensor fault flag rather than the sudden pressure drop that actually defines a puncture in the simulator; the specific row picked for this report may have been read after the drop was partly smoothed by rolling-window features, or another feature simply dominated for this particular row. This is reported as-is rather than cherry-picking a cleaner-looking row.
- `sensor_malfunction` shows weak, inconsistent-direction contributions — expected, since (per Milestone 1/2/4 findings) this failure type has no distinct physical telemetry signature built into the simulator.
- The 4 hardcoded engineering rules (pressure, tread depth, temperature, load) are simple single-threshold checks, not an exhaustive rule base — they exist to demonstrate the source-separation principle, not to catch every unsafe condition.