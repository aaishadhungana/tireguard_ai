# TireGuard AI — Failure Prediction Model Evaluation

> Generated programmatically by `src/models/train_failure_model.py`. Every metric below was computed at run time via tire-level GroupKFold cross-validation — none are estimated or illustrative.

## Dataset

- Total rows: **115,200**
- Total failures: **80** (0.0694% of rows)
- Class imbalance ratio (negative:positive): **1439.0:1**

## Why Tire-Level Cross-Validation (Not a Single Time-Based Split)

A global timestamp cutoff was evaluated first and rejected: every failure in this dataset occurs between day 10 and day 24 of the 30-day simulation run, so any reasonable train/test time cutoff puts close to zero failures in the test set. Instead, tires are split into groups (GroupKFold), so each tire's full history stays on one side of the split — this prevents leakage while keeping failure examples proportionally represented across folds.

## Cross-Validation Results (5-fold, grouped by tire)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.0015 ± 0.0004 | 0.5875 ± 0.0935 | 0.003 ± 0.0007 | 0.7132 ± 0.0404 | 0.0024 ± 0.0007 |
| Random Forest | 0.4558 ± 0.1326 | 0.65 ± 0.0848 | 0.5183 ± 0.0818 | 0.9888 ± 0.0167 | 0.5525 ± 0.0708 |
| XGBoost | 0.2761 ± 0.1407 | 0.525 ± 0.0935 | 0.3534 ± 0.1338 | 0.9866 ± 0.0125 | 0.3371 ± 0.2027 |

## Per-Fold Detail

### Logistic Regression

- Fold 0: 16 tires, 23040 rows, 16 failures — precision=0.0015, recall=0.75, f1=0.003, roc_auc=0.758, pr_auc=0.0037
- Fold 1: 16 tires, 23040 rows, 16 failures — precision=0.0015, recall=0.625, f1=0.003, roc_auc=0.7554, pr_auc=0.0017
- Fold 2: 16 tires, 23040 rows, 16 failures — precision=0.0012, recall=0.5, f1=0.0023, roc_auc=0.6697, pr_auc=0.0021
- Fold 3: 16 tires, 23040 rows, 16 failures — precision=0.0012, recall=0.5, f1=0.0025, roc_auc=0.6639, pr_auc=0.0019
- Fold 4: 16 tires, 23040 rows, 16 failures — precision=0.0022, recall=0.5625, f1=0.0043, roc_auc=0.719, pr_auc=0.0025

### Random Forest

- Fold 0: 16 tires, 23040 rows, 16 failures — precision=0.3824, recall=0.8125, f1=0.52, roc_auc=0.9981, pr_auc=0.6466
- Fold 1: 16 tires, 23040 rows, 16 failures — precision=0.2632, recall=0.625, f1=0.3704, roc_auc=0.9945, pr_auc=0.4525
- Fold 2: 16 tires, 23040 rows, 16 failures — precision=0.5556, recall=0.625, f1=0.5882, roc_auc=0.9991, pr_auc=0.6002
- Fold 3: 16 tires, 23040 rows, 16 failures — precision=0.4348, recall=0.625, f1=0.5128, roc_auc=0.9968, pr_auc=0.4924
- Fold 4: 16 tires, 23040 rows, 16 failures — precision=0.6429, recall=0.5625, f1=0.6, roc_auc=0.9556, pr_auc=0.5709

### XGBoost

- Fold 0: 16 tires, 23040 rows, 16 failures — precision=0.2368, recall=0.5625, f1=0.3333, roc_auc=0.9972, pr_auc=0.3206
- Fold 1: 16 tires, 23040 rows, 16 failures — precision=0.1522, recall=0.4375, f1=0.2258, roc_auc=0.9951, pr_auc=0.1411
- Fold 2: 16 tires, 23040 rows, 16 failures — precision=0.2, recall=0.5, f1=0.2857, roc_auc=0.9924, pr_auc=0.147
- Fold 3: 16 tires, 23040 rows, 16 failures — precision=0.55, recall=0.6875, f1=0.6111, roc_auc=0.9856, pr_auc=0.6957
- Fold 4: 16 tires, 23040 rows, 16 failures — precision=0.2414, recall=0.4375, f1=0.3111, roc_auc=0.9628, pr_auc=0.3809

## Why PR-AUC Matters More Than ROC-AUC Here

With roughly 0.07% of rows being failures, ROC-AUC can look deceptively high because it's dominated by the huge number of true negatives. PR-AUC focuses on the precision/recall tradeoff for the rare positive class and is the more honest metric for this safety-sensitive, heavily imbalanced problem.

## Known Limitations

- Only 60 positive examples exist in the entire dataset (one per tire). Cross-validated metrics are reported as mean ± std across folds, but with this few positives, fold-to-fold variance is inherently high and these numbers should be treated as directional, not precise.
- No hyperparameter tuning was performed — models use reasonable defaults. Tuning is deferred until the class-imbalance / failure-type imbalance issues (flagged in Milestone 2's EDA) are addressed, since tuning against unstable metrics would not be meaningful.
- The extreme rarity of failures in this simulator run suggests the simulator's failure_rate parameter and/or duration should be increased for more statistically robust future model iterations.