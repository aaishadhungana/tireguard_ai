# TireGuard AI — Sensor Anomaly Detection Evaluation

> Generated programmatically by `src/models/train_anomaly_model.py`.

## An Unusual Advantage For This Evaluation (Read First)

Because this is synthetic data with simulator-injected sensor faults (`pressure_sensor_fault`, `temperature_sensor_fault`, `tread_sensor_fault` columns), precision/recall against a KNOWN ground truth can actually be computed here. Most real-world anomaly detection cannot do this — there is no ground truth to check against in production, only after-the-fact manual review of flagged cases. The strong numbers below reflect this synthetic advantage and should not be read as what to expect when this same approach is pointed at real fleet data with no known answer key.

- Total rows evaluated: **115,200**
- Rows with a real injected sensor fault (any of the 3 sensors): **66.95%**

## Important Caveat: 'Anomaly' Is Not a Coherent Frame Here

At current simulator settings, **67.0% of rows have a real sensor fault** — a direct consequence of a Milestone 2 finding: sensor drift persists across many consecutive steps once triggered, so the effective fault rate is far higher than the `sensor_fault_rate` config parameter's nominal value, and this compounds further across 3 independently-faulted sensors combined with 'any of 3' logic. IsolationForest's `contamination` parameter is capped at 0.5 (it assumes anomalies are a MINORITY by design), so it was clamped to 0.5 here rather than the true 67.0%. The practical implication: at this fault rate, 'is this row anomalous' is close to a coin flip rather than a rare-event detection problem, and the metrics below should be read with that in mind. The right fix is upstream, in the simulator's fault injection persistence model (flagged in Milestone 2, not re-fixed here to keep this milestone scoped to detection rather than reopening simulator design).

## Two Separate Detection Approaches (Never Blended)

- **`rule_based`**: Milestone 2's existing out-of-range / missing-value flags (simple, auditable thresholds).
- **`ml_based`**: Isolation Forest trained on observed sensor readings and their derived features only (never on the ground-truth fault columns or true physical values, which a real system would not have).

## Results Against Ground Truth

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| rule_based | 1.000 | 0.090 | 0.166 |
| ml_based (Isolation Forest) | 0.783 | 0.585 | 0.670 |

### Rule-Based Confusion Matrix

```
                Predicted
                Normal  Anomaly
Actual Normal    38070        0
Actual Anomaly   70150     6980
```

### ML-Based Confusion Matrix

```
                Predicted
                Normal  Anomaly
Actual Normal    25582    12488
Actual Anomaly   32018    45112
```

## Agreement Between The Two Approaches

- Both flagged: **5,895** rows
- Only rule_based flagged: **1,085** rows
- Only ml_based flagged: **51,705** rows
- Neither flagged: **56,515** rows

Rows flagged by ML but not by rules are the more interesting case — these are anomalies the fixed-threshold rules structurally cannot catch (e.g. `drift`, which stays within any single-reading plausible range but looks anomalous relative to that tire's own recent trend).

## Known Limitations

- Evaluated only on this specific simulator run's fault mix and rates — generalization to different fault distributions is untested.
- Isolation Forest's `contamination` parameter was set from this dataset's OWN observed fault rate, which a real deployment would not know in advance and would need to estimate or tune differently.
- Does not distinguish WHICH sensor is faulted or what fault type — just whether the row looks anomalous overall. A finer-grained per-sensor, per-fault-type model is a reasonable next iteration.