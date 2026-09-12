# TireGuard AI — Failure Prediction Model Evaluation

> Generated programmatically by `src/models/train_failure_model.py`. Every metric below was computed at run time via tire-level GroupKFold cross-validation — none are estimated or illustrative.

## Dataset

- Total rows: **86,400**
- Total failures: **60** (0.0694% of rows)
- Class imbalance ratio (negative:positive): **1439.0:1**

## Why Tire-Level Cross-Validation (Not a Single Time-Based Split)

A global timestamp cutoff was evaluated first and rejected: every failure in this dataset occurs between day 10 and day 24 of the 30-day simulation run, so any reasonable train/test time cutoff puts close to zero failures in the test set. Instead, tires are split into groups (GroupKFold), so each tire's full history stays on one side of the split — this prevents leakage while keeping failure examples proportionally represented across folds.

## Cross-Validation Results (5-fold, grouped by tire)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.0017 ± 0.0003 | 0.5667 ± 0.0972 | 0.0033 ± 0.0006 | 0.7308 ± 0.0466 | 0.0038 ± 0.0013 |
| Random Forest | 0.2201 ± 0.1241 | 0.4167 ± 0.1178 | 0.2555 ± 0.0873 | 0.9568 ± 0.0352 | 0.2825 ± 0.0956 |
| XGBoost | 0.2655 ± 0.1446 | 0.4333 ± 0.1616 | 0.3194 ± 0.1439 | 0.9453 ± 0.0407 | 0.2882 ± 0.1584 |

## Per-Fold Detail

### Logistic Regression

- Fold 0: 12 tires, 17280 rows, 12 failures — precision=0.0021, recall=0.6667, f1=0.0043, roc_auc=0.7167, pr_auc=0.0038
- Fold 1: 12 tires, 17280 rows, 12 failures — precision=0.0017, recall=0.5833, f1=0.0035, roc_auc=0.7051, pr_auc=0.0014
- Fold 2: 12 tires, 17280 rows, 12 failures — precision=0.0016, recall=0.5, f1=0.0031, roc_auc=0.7494, pr_auc=0.0043
- Fold 3: 12 tires, 17280 rows, 12 failures — precision=0.0012, recall=0.4167, f1=0.0025, roc_auc=0.6729, pr_auc=0.0051
- Fold 4: 12 tires, 17280 rows, 12 failures — precision=0.0017, recall=0.6667, f1=0.0033, roc_auc=0.8101, pr_auc=0.0042

### Random Forest

- Fold 0: 12 tires, 17280 rows, 12 failures — precision=0.0769, recall=0.3333, f1=0.125, roc_auc=0.9196, pr_auc=0.309
- Fold 1: 12 tires, 17280 rows, 12 failures — precision=0.2692, recall=0.5833, f1=0.3684, roc_auc=0.9101, pr_auc=0.3601
- Fold 2: 12 tires, 17280 rows, 12 failures — precision=0.1176, recall=0.5, f1=0.1905, roc_auc=0.9827, pr_auc=0.0944
- Fold 3: 12 tires, 17280 rows, 12 failures — precision=0.4286, recall=0.25, f1=0.3158, roc_auc=0.9742, pr_auc=0.3315
- Fold 4: 12 tires, 17280 rows, 12 failures — precision=0.2083, recall=0.4167, f1=0.2778, roc_auc=0.9975, pr_auc=0.3174

### XGBoost

- Fold 0: 12 tires, 17280 rows, 12 failures — precision=0.4118, recall=0.5833, f1=0.4828, roc_auc=0.8947, pr_auc=0.3833
- Fold 1: 12 tires, 17280 rows, 12 failures — precision=0.1481, recall=0.3333, f1=0.2051, roc_auc=0.9093, pr_auc=0.1716
- Fold 2: 12 tires, 17280 rows, 12 failures — precision=0.2059, recall=0.5833, f1=0.3043, roc_auc=0.9861, pr_auc=0.3062
- Fold 3: 12 tires, 17280 rows, 12 failures — precision=0.1, recall=0.1667, f1=0.125, roc_auc=0.9387, pr_auc=0.0635
- Fold 4: 12 tires, 17280 rows, 12 failures — precision=0.4615, recall=0.5, f1=0.48, roc_auc=0.9976, pr_auc=0.5164

## Why PR-AUC Matters More Than ROC-AUC Here

With roughly 0.07% of rows being failures, ROC-AUC can look deceptively high because it's dominated by the huge number of true negatives. PR-AUC focuses on the precision/recall tradeoff for the rare positive class and is the more honest metric for this safety-sensitive, heavily imbalanced problem.

## Known Limitations

- Only 60 positive examples exist in the entire dataset (one per tire). Cross-validated metrics are reported as mean ± std across folds, but with this few positives, fold-to-fold variance is inherently high and these numbers should be treated as directional, not precise.
- No hyperparameter tuning was performed — models use reasonable defaults. Tuning is deferred until the class-imbalance / failure-type imbalance issues (flagged in Milestone 2's EDA) are addressed, since tuning against unstable metrics would not be meaningful.
- The extreme rarity of failures in this simulator run suggests the simulator's failure_rate parameter and/or duration should be increased for more statistically robust future model iterations.