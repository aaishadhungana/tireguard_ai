from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import settings
from src.models.data_split import tire_group_kfold, verify_no_tire_leakage
from src.models.rul_labels import compute_rul_labels, rul_labeling_summary
from src.models.rul_model import (
    build_linear_regression_pipeline,
    build_random_forest_rul_pipeline,
    build_rul_feature_matrix,
    build_xgboost_rul_pipeline,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def _find_latest_processed_file():
    candidates = sorted(settings.data_processed_dir.glob("tire_telemetry_processed_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No processed datasets found in {settings.data_processed_dir}.")
    return candidates[-1]


def cross_validate_rul(build_fn, X, y, groups, n_folds):
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(tire_group_kfold(X.assign(tire_id=groups), n_folds)):
        assert verify_no_tire_leakage(X.assign(tire_id=groups), train_idx, test_idx), (
            "Tire leakage detected — structurally should be impossible with GroupKFold."
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(X_train) == 0 or len(X_test) == 0:
            log.warning("Fold %d: empty train or test set, skipping.", fold_idx)
            continue

        model = build_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred) if len(set(y_test)) > 1 else None

        fold_results.append(
            {
                "fold": fold_idx,
                "n_test": len(test_idx),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4) if r2 is not None else None,
            }
        )
        log.info("Fold %d: MAE=%.2f RMSE=%.2f R2=%s (n=%d)", fold_idx, mae, rmse, r2, len(test_idx))

    summary = {"folds": fold_results}
    for metric in ("mae", "rmse", "r2"):
        values = [f[metric] for f in fold_results if f[metric] is not None]
        summary[f"{metric}_mean"] = round(float(np.mean(values)), 2) if values else None
        summary[f"{metric}_std"] = round(float(np.std(values)), 2) if values else None
    return summary


def build_report(results, label_summary):
    lines = [
        "# TireGuard AI — Remaining Useful Life (RUL) Model Evaluation",
        "",
        "> Generated programmatically by `src/models/train_rul_model.py`. "
        "All metrics computed via tire-level GroupKFold cross-validation.",
        "",
        "## RUL Definition",
        "",
        "RUL = remaining mileage until a tire's tread depth reaches the legal "
        "minimum (1.6mm) within its CURRENT wear cycle (a cycle resets after "
        "any failure/replacement event — see `src/models/rul_labels.py`). This "
        "covers wear-based degradation only, not sudden failure modes like "
        "puncture (already covered by Milestones 3-5).",
        "",
        "## Label Coverage (Read This Before The Metrics)",
        "",
        f"- Total rows: **{label_summary['total_rows']:,}**",
        f"- Rows with a computable RUL label: **{label_summary['labeled_rows']:,}**",
        f"- Censored rows (wear cycle still in progress when the simulation "
        f"ended — RUL genuinely unknown, not fabricated): "
        f"**{label_summary['censored_rows']:,}** "
        f"({100 * label_summary['censored_fraction']:.1f}% of all rows)",
        f"- Mean RUL among labeled rows: **{label_summary['rul_mean_mileage']:,.0f} km**",
        f"- Max RUL among labeled rows: **{label_summary['rul_max_mileage']:,.0f} km**",
        "",
        "Nearly half the dataset is censored at current simulator settings (30-day "
        "duration is comparable to one full tire wear cycle, so many cycles are "
        "still in progress when the window ends). This is expected, not a bug — "
        "but it does mean the model is trained on a biased sample of "
        "'cycles that finished within 30 days,' which may skew toward faster-wearing "
        "tires. Longer simulation runs would reduce this censoring.",
        "",
        "## Cross-Validation Results (5-fold, grouped by tire)",
        "",
        "| Model | MAE (km) | RMSE (km) | R² |",
        "|---|---|---|---|",
    ]
    for model_name, res in results.items():
        def fmt(mean_key, std_key):
            m, s = res.get(mean_key), res.get(std_key)
            return f"{m} \u00b1 {s}" if m is not None else "N/A"

        lines.append(
            f"| {model_name} | {fmt('mae_mean', 'mae_std')} | "
            f"{fmt('rmse_mean', 'rmse_std')} | {fmt('r2_mean', 'r2_std')} |"
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
                f"- Fold {fold['fold']}: n={fold['n_test']}, MAE={fold['mae']}, "
                f"RMSE={fold['rmse']}, R\u00b2={fold['r2']}"
            )
        lines.append("")

    lines += [
        "## Known Limitations",
        "",
        "- ~48% censoring means training data skews toward tires that wore out "
        "within the 30-day window, likely a biased (faster-wearing) sample "
        "relative to the full fleet.",
        "- MAE in km should be read relative to typical tire lifetime "
        "(~45,000-50,000 km), an MAE of a few thousand km is a meaningfully "
        "useful estimate; an MAE approaching that full range is not.",
        "- No hyperparameter tuning performed.",
        "- Recommended next step: regenerate with a longer simulation duration "
        "(e.g. 60-90 days) to reduce censoring before relying on this model.",
    ]
    return "\n".join(lines)


def run_training(input_path, n_folds):
    settings.ensure_directories()

    log.info("Loading processed dataset from %s", input_path)
    df = pd.read_parquet(input_path)

    log.info("Computing RUL labels...")
    labeled_df = compute_rul_labels(df)
    label_summary = rul_labeling_summary(labeled_df)
    log.info("Label summary: %s", label_summary)

    X, y = build_rul_feature_matrix(labeled_df)
    groups = labeled_df.loc[X.index, "tire_id"]
    log.info("Training on %d non-censored rows across %d tires", len(X), groups.nunique())

    results = {}
    model_builders = {
        "Linear Regression": build_linear_regression_pipeline,
        "Random Forest": build_random_forest_rul_pipeline,
        "XGBoost": build_xgboost_rul_pipeline,
    }

    trained_models = {}
    for model_name, build_fn in model_builders.items():
        log.info("=== Cross-validating %s ===", model_name)
        results[model_name] = cross_validate_rul(build_fn, X, y, groups, n_folds)

        final_model = build_fn()
        final_model.fit(X, y)
        trained_models[model_name] = final_model

    log.info("Building evaluation report...")
    report_md = build_report(results, label_summary)
    report_path = DOCS_DIR / "rul_evaluation.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("Wrote evaluation report to %s", report_path)

    best_model_name = min(
        results,
        key=lambda name: (results[name]["mae_mean"] if results[name]["mae_mean"] is not None else float("inf")),
    )
    log.info("Best model by mean MAE: %s", best_model_name)

    model_path = settings.model_dir / "rul_model.joblib"
    joblib.dump(trained_models[best_model_name], model_path)
    log.info("Saved final model (%s) to %s", best_model_name, model_path)

    metadata = {
        "model_type": best_model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(X),
        "label_summary": label_summary,
        "cross_validation_results": {
            k: {mk: mv for mk, mv in v.items() if mk != "folds"} for k, v in results.items()
        },
    }
    metadata_path = settings.model_dir / "rul_model.meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved model metadata to %s", metadata_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the RUL model.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input) if args.input else _find_latest_processed_file()
    run_training(input_path, args.folds)


if __name__ == "__main__":
    main()