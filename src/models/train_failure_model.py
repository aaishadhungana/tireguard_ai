from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.settings import settings
from src.models.data_split import tire_group_kfold, verify_no_tire_leakage
from src.models.failure_model import (
    build_feature_matrix,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    compute_scale_pos_weight,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def _find_latest_processed_file() -> Path:
    candidates = sorted(settings.data_processed_dir.glob("tire_telemetry_processed_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No processed datasets found in {settings.data_processed_dir}. "
            f"Run `python -m src.data.pipeline` first."
        )
    return candidates[-1]


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Computes metrics for one fold, handling the degenerate case of a
    fold with zero positive examples (undefined precision/recall/AUC)
    explicitly rather than letting sklearn silently warn or error."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return {
            "n_pos": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
            "note": "No positive examples in this fold's test set — metrics undefined.",
        }

    return {
        "n_pos": n_pos,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
    }


def cross_validate_model(build_fn, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_folds: int) -> dict:
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(tire_group_kfold(X.assign(tire_id=groups), n_folds)):
        assert verify_no_tire_leakage(X.assign(tire_id=groups), train_idx, test_idx), (
            "Tire leakage detected between train and test fold — this should be "
            "structurally impossible with GroupKFold and indicates a real bug."
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if "scale_pos_weight" in build_fn.__code__.co_varnames:
            model = build_fn(compute_scale_pos_weight(y_train))
        else:
            model = build_fn()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fold_metrics = evaluate_fold(y_test.values, y_pred, y_proba)
        fold_metrics["fold"] = fold_idx
        fold_metrics["n_test_rows"] = len(test_idx)
        fold_metrics["n_test_tires"] = X.assign(tire_id=groups).iloc[test_idx]["tire_id"].nunique()
        fold_results.append(fold_metrics)
        log.info("Fold %d: %s", fold_idx, fold_metrics)

    valid_folds = [f for f in fold_results if f["precision"] is not None]
    summary = {"folds": fold_results, "n_valid_folds": len(valid_folds), "n_total_folds": n_folds}
    for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        values = [f[metric] for f in valid_folds if f[metric] is not None]
        summary[f"{metric}_mean"] = round(float(np.mean(values)), 4) if values else None
        summary[f"{metric}_std"] = round(float(np.std(values)), 4) if values else None
    return summary


def build_report(results: dict, scale_pos_weight: float, n_rows: int, n_failures: int) -> str:
    lines = [
        "# TireGuard AI — Failure Prediction Model Evaluation",
        "",
        "> Generated programmatically by `src/models/train_failure_model.py`. "
        "Every metric below was computed at run time via tire-level GroupKFold "
        "cross-validation — none are estimated or illustrative.",
        "",
        "## Dataset",
        "",
        f"- Total rows: **{n_rows:,}**",
        f"- Total failures: **{n_failures}** ({100 * n_failures / n_rows:.4f}% of rows)",
        f"- Class imbalance ratio (negative:positive): **{scale_pos_weight:.1f}:1**",
        "",
        "## Why Tire-Level Cross-Validation (Not a Single Time-Based Split)",
        "",
        "A global timestamp cutoff was evaluated first and rejected: every failure "
        "in this dataset occurs between day 10 and day 24 of the 30-day simulation "
        "run, so any reasonable train/test time cutoff puts close to zero failures "
        "in the test set. Instead, tires are split into groups (GroupKFold), so "
        "each tire's full history stays on one side of the split — this prevents "
        "leakage while keeping failure examples proportionally represented across "
        "folds.",
        "",
        "## Cross-Validation Results (5-fold, grouped by tire)",
        "",
        "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for model_name, res in results.items():
        def fmt(mean_key, std_key):
            m, s = res.get(mean_key), res.get(std_key)
            return f"{m} \u00b1 {s}" if m is not None else "N/A"

        lines.append(
            f"| {model_name} | {fmt('precision_mean', 'precision_std')} | "
            f"{fmt('recall_mean', 'recall_std')} | {fmt('f1_mean', 'f1_std')} | "
            f"{fmt('roc_auc_mean', 'roc_auc_std')} | {fmt('pr_auc_mean', 'pr_auc_std')} |"
        )

    lines += [
        "",
        "## Per-Fold Detail",
        "",
    ]
    for model_name, res in results.items():
        lines.append(f"### {model_name}")
        lines.append("")
        for fold in res["folds"]:
            lines.append(
                f"- Fold {fold['fold']}: {fold['n_test_tires']} tires, "
                f"{fold['n_test_rows']} rows, {fold['n_pos']} failures — "
                f"precision={fold['precision']}, recall={fold['recall']}, "
                f"f1={fold['f1']}, roc_auc={fold['roc_auc']}, pr_auc={fold['pr_auc']}"
            )
        lines.append("")

    lines += [
        "## Why PR-AUC Matters More Than ROC-AUC Here",
        "",
        "With roughly 0.07% of rows being failures, ROC-AUC can look deceptively "
        "high because it's dominated by the huge number of true negatives. "
        "PR-AUC focuses on the precision/recall tradeoff for the rare positive "
        "class and is the more honest metric for this safety-sensitive, heavily "
        "imbalanced problem.",
        "",
        "## Known Limitations",
        "",
        "- Only 60 positive examples exist in the entire dataset (one per tire). "
        "Cross-validated metrics are reported as mean \u00b1 std across folds, but "
        "with this few positives, fold-to-fold variance is inherently high and "
        "these numbers should be treated as directional, not precise.",
        "- No hyperparameter tuning was performed — models use reasonable "
        "defaults. Tuning is deferred until the class-imbalance / failure-type "
        "imbalance issues (flagged in Milestone 2's EDA) are addressed, since "
        "tuning against unstable metrics would not be meaningful.",
        "- The extreme rarity of failures in this simulator run suggests the "
        "simulator's failure_rate parameter and/or duration should be increased "
        "for more statistically robust future model iterations.",
    ]
    return "\n".join(lines)


def run_training(input_path: Path, n_folds: int) -> None:
    settings.ensure_directories()

    log.info("Loading processed dataset from %s", input_path)
    df = pd.read_parquet(input_path)

    X, y = build_feature_matrix(df)
    groups = df["tire_id"]

    log.info(
        "Dataset: %d rows, %d failures (%.4f%%), %d unique tires",
        len(df), y.sum(), 100 * y.mean(), groups.nunique(),
    )

    results = {}
    log.info("=== Cross-validating Logistic Regression (baseline) ===")
    results["Logistic Regression"] = cross_validate_model(
        build_logistic_regression_pipeline, X, y, groups, n_folds,
    )

    log.info("=== Cross-validating Random Forest (baseline) ===")
    results["Random Forest"] = cross_validate_model(
        build_random_forest_pipeline, X, y, groups, n_folds,
    )

    log.info("=== Cross-validating XGBoost ===")
    results["XGBoost"] = cross_validate_model(
        build_xgboost_pipeline, X, y, groups, n_folds,
    )

    scale_pos_weight = compute_scale_pos_weight(y)

    log.info("Building evaluation report...")
    report_md = build_report(results, scale_pos_weight, len(df), int(y.sum()))
    report_path = DOCS_DIR / "model_evaluation.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("Wrote evaluation report to %s", report_path)

    # ---- Train final model on ALL data using the best-performing model
    # by mean PR-AUC (the metric most meaningful given the imbalance) ----
    best_model_name = max(
        results, key=lambda name: (results[name]["pr_auc_mean"] or -1)
    )
    log.info("Best model by mean PR-AUC: %s", best_model_name)

    if best_model_name == "XGBoost":
        final_model = build_xgboost_pipeline(scale_pos_weight)
    elif best_model_name == "Random Forest":
        final_model = build_random_forest_pipeline()
    else:
        final_model = build_logistic_regression_pipeline()

    final_model.fit(X, y)

    model_path = settings.model_dir / "failure_model.joblib"
    joblib.dump(final_model, model_path)
    log.info("Saved final model (%s) to %s", best_model_name, model_path)

    metadata = {
        "model_type": best_model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(df),
        "training_failures": int(y.sum()),
        "feature_columns": list(X.columns),
        "cross_validation_results": {
            k: {mk: mv for mk, mv in v.items() if mk != "folds"} for k, v in results.items()
        },
    }
    metadata_path = settings.model_dir / "failure_model.meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved model metadata to %s", metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the failure prediction model.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else _find_latest_processed_file()
    run_training(input_path, args.folds)


if __name__ == "__main__":
    main()