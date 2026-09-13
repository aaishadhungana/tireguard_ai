# TireGuard AI — Remaining Useful Life (RUL) Model Evaluation

> Generated programmatically by `src/models/train_rul_model.py`. All metrics computed via tire-level GroupKFold cross-validation.

## RUL Definition

RUL = remaining mileage until a tire's tread depth reaches the legal minimum (1.6mm) within its CURRENT wear cycle (a cycle resets after any failure/replacement event — see `src/models/rul_labels.py`). This covers wear-based degradation only, not sudden failure modes like puncture (already covered by Milestones 3-5).

## Label Coverage (Read This Before The Metrics)

- Total rows: **115,200**
- Rows with a computable RUL label: **71,654**
- Censored rows (wear cycle still in progress when the simulation ended — RUL genuinely unknown, not fabricated): **43,546** (37.8% of all rows)
- Mean RUL among labeled rows: **15,965 km**
- Max RUL among labeled rows: **38,445 km**

Nearly half the dataset is censored at current simulator settings (30-day duration is comparable to one full tire wear cycle, so many cycles are still in progress when the window ends). This is expected, not a bug — but it does mean the model is trained on a biased sample of 'cycles that finished within 30 days,' which may skew toward faster-wearing tires. Longer simulation runs would reduce this censoring.

## Cross-Validation Results (5-fold, grouped by tire)

| Model | MAE (km) | RMSE (km) | R² |
|---|---|---|---|
| Linear Regression | 3763.69 ± 553.44 | 8075.41 ± 6919.81 | -0.24 ± 1.97 |
| Random Forest | 1011.18 ± 88.25 | 1960.03 ± 172.96 | 0.96 ± 0.01 |
| XGBoost | 1086.66 ± 108.3 | 1917.6 ± 197.33 | 0.96 ± 0.01 |

## Per-Fold Detail

### Linear Regression

- Fold 0: n=14340, MAE=3242.77, RMSE=3908.18, R²=0.8301
- Fold 1: n=14337, MAE=3781.75, RMSE=6502.2, R²=0.5328
- Fold 2: n=14320, MAE=4816.14, RMSE=21781.19, R²=-4.1764
- Fold 3: n=14318, MAE=3465.29, RMSE=4036.65, R²=0.8195
- Fold 4: n=14339, MAE=3512.52, RMSE=4148.83, R²=0.8098

### Random Forest

- Fold 0: n=14340, MAE=1018.74, RMSE=2144.01, R²=0.9489
- Fold 1: n=14337, MAE=992.57, RMSE=1968.98, R²=0.9572
- Fold 2: n=14320, MAE=908.11, RMSE=1696.18, R²=0.9686
- Fold 3: n=14318, MAE=964.79, RMSE=1847.97, R²=0.9622
- Fold 4: n=14339, MAE=1171.67, RMSE=2143.0, R²=0.9492

### XGBoost

- Fold 0: n=14340, MAE=1069.77, RMSE=2064.58, R²=0.9526
- Fold 1: n=14337, MAE=1075.79, RMSE=1940.08, R²=0.9584
- Fold 2: n=14320, MAE=962.15, RMSE=1624.6, R²=0.9712
- Fold 3: n=14318, MAE=1038.03, RMSE=1780.48, R²=0.9649
- Fold 4: n=14339, MAE=1287.58, RMSE=2178.26, R²=0.9476

## Known Limitations

- ~48% censoring means training data skews toward tires that wore out within the 30-day window, likely a biased (faster-wearing) sample relative to the full fleet.
- MAE in km should be read relative to typical tire lifetime (~45,000-50,000 km), an MAE of a few thousand km is a meaningfully useful estimate; an MAE approaching that full range is not.
- No hyperparameter tuning performed.
- Recommended next step: regenerate with a longer simulation duration (e.g. 60-90 days) to reduce censoring before relying on this model.