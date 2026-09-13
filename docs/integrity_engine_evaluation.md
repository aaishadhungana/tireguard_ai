# TireGuard AI — Sensor Integrity Engine Evaluation

> Generated programmatically by `src/models/run_integrity_engine.py` on a random sample of 5,000 rows (full-dataset row-by-row evaluation would be slow with unvectorized rule checks -- sampling is stated explicitly, not silently partial).

## Detection Rate By Actual Fault Type

This is the key result for this milestone: does the combined engine catch DIFFERENT fault signatures differently? (an aggregate-only metric would hide this.)

| Actual Fault Type | Rows in Sample | Flagged (MEDIUM/HIGH risk) | Detection Rate |
|---|---|---|---|
| drift | 2863 | 269 | 9.4% |
| missing | 65 | 6 | 9.2% |
| none | 1653 | 48 | 2.9% |
| spoofed | 66 | 55 | 83.3% |
| stuck | 353 | 26 | 7.4% |

## Real Example Outputs (Spec Format)

### Flagged (actual: spoofed)

```
Sensor Integrity Score: 0.2025
Risk: HIGH

Reason:
physics_consistency: temperature=59.7C deviates 41.8C from the ~17.9C expected given current pressure/load/speed/weather
```

> This score reflects pattern-matching against known simulated fault signatures. It is NOT evidence of an actual cyberattack or intrusion -- a flagged reading could equally be a hardware fault or data pipeline bug.

### Not flagged (actual: clean)

```
Sensor Integrity Score: 0.9859
Risk: LOW

Reason:
No rule-based integrity checks triggered.
```

> This score reflects pattern-matching against known simulated fault signatures. It is NOT evidence of an actual cyberattack or intrusion -- a flagged reading could equally be a hardware fault or data pipeline bug.


## Honest Finding: Physics-Consistency Check's Limited Contribution Here

The physics-consistency check does not meaningfully discriminate this dataset's spoofed rows (mean deviation from expected temperature: 4.79C for spoofed vs 4.08C for clean rows -- barely different). This is because the simulator's spoof multiplier always INFLATES pressure, but the underinflation-heat causal rule only responds to LOW pressure -- there is no 'overinflation causes heat' rule for an inflated spoof to violate. The absolute-implausibility check (a tight 50 psi ceiling) is doing essentially all the useful work for spoofed-pressure detection in this dataset. The physics-consistency check is retained because it is still physically correct and would catch a DIFFERENT spoofing direction (artificially LOW pressure) that this particular simulator run doesn't happen to produce.

## Score Distribution

- Mean integrity score (sample): 0.904
- Rows flagged MEDIUM or HIGH risk: 404 (8.1% of sample)

## Known Limitations

- Replay-pattern check was not exercised in this evaluation (requires per-tire historical sequences, not included in this row-sampled evaluation).
- Rule checks are unvectorized Python (row-by-row) -- fine for this milestone's evaluation purposes, but would need vectorizing for real-time production use on a full fleet stream.
- Physics-consistency check's weak contribution here is a property of THIS dataset's specific spoof direction, not a fundamental flaw in the check itself -- see the honest finding above.