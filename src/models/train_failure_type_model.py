from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.config.settings import settings
from src.models.data_split import tire_group_kfold, verify_no_tire_leakage
from src.models.failure_type_model import (
    FAILURE_TYPE_CLASSES,
    build_failure_type_feature_matrix,
    build_random_forest_type_pipeline,
    build_xgboost_type_pipeline,
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


def cross_validate_pooled(build_fn, X, y_encoded, groups, n_folds, is_xgb):
    """Runs GroupKFold CV, returns pooled (y_true, y_pred) across all
    folds' test sets — this is what makes per-class metrics meaningful
    with this few examples per class. Also runs per-fold leakage checks."""
    all_true, all_pred = [], []
    fold_info = []

    for fold_idx, (train_idx, test_idx) in enumerate(tire_group_kfold(X.assign(tire_id=groups), n_folds)):
        assert verify_no_tire_leakage(X.assign(tire_id=groups), train_idx, test_idx), (
            "Tire leakage detected — structurally should be impossible with GroupKFold."
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        if len(set(y_train)) < 2:
            log.warning("Fold %d: training set has fewer than 2 classes, skipping fold.", fold_idx)
            continue

        if is_xgb:
            model = build_fn(num_classes=len(FAILURE_TYPE_CLASSES))
        else:
            model = build_fn()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        fold_info.append(
            {"fold": fold_idx, "n_train": len(train_idx), "n_test": len(test_idx)}
        )

    return np.array(all_true), np.array(all_pred), fold_info


def build_report(model_reports, n_failure_rows, class_counts):
    lines = [
        "# TireGuard AI : Failure Type Classification Evaluation",
        "",
        "> Generated programmatically by `src/models/train_failure_type_model.py`. "
        "All metrics are computed from predictions POOLED across all GroupKFold "
        "folds, not averaged per-fold, with 5-6 examples for most classes, a "
        "single fold's per-class recall would swing between 0% and 100% on one "
        "misclassification and would not be a meaningful number on its own.",
        "",
        "## Dataset",
        "",
        f"- Total failure rows classified: **{n_failure_rows}**",
        "- Class counts:",
    ]
    for cls, count in class_counts.items():
        lines.append(f"  - `{cls}`: {count}")

    lines += [
        "",
        "## Why This Is a Hard Problem With This Data",
        "",
        "`structural_degradation` has 68 examples; every other class has 5-6. "
        "A classifier can reach high overall accuracy by mostly predicting the "
        "majority class and still be nearly useless for the rare classes — which "
        "are arguably the more actionable ones (puncture, overheating, etc. call "
        "for different interventions than routine tread wear-out). Per-class "
        "precision/recall below should be read with that in mind, not just the "
        "overall accuracy number.",
        "",
    ]

    for model_name, report in model_reports.items():
        lines += [
            f"## {model_name} — Pooled Results",
            "",
            "### Confusion Matrix (rows = true, columns = predicted)",
            "",
            "```",
            report["confusion_matrix_str"],
            "```",
            "",
            "### Per-Class Metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|",
        ]
        for cls, m in report["per_class"].items():
            lines.append(
                f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1-score']:.3f} | {int(m['support'])} |"
            )
        lines += [
            "",
            f"- Overall accuracy: **{report['accuracy']:.3f}**",
            f"- Macro-averaged F1 (unweighted across classes — the more honest "
            f"summary number given the imbalance): **{report['macro_f1']:.3f}**",
            "",
        ]

    lines += [
        "## Known Limitations",
        "",
        "- Rare-class metrics (5-6 examples each) have wide uncertainty even when "
        "pooled across folds — a couple of examples either way materially changes "
        "the reported precision/recall for those classes.",
        "- `sensor_malfunction` has no distinct telemetry signature in the "
        "simulator (flagged since Milestone 1/2) — expect it to be the hardest "
        "class to distinguish, since nothing in the feature set actually "
        "correlates with it by design.",
        "- No hyperparameter tuning performed. With this little data per class, "
        "tuning would likely overfit to noise rather than find real structure.",
        "- Recommended next step if this needs to be more reliable: increase "
        "the simulator's failure_rate further and/or run a longer duration to "
        "get more examples of each rare failure type, OR treat this as a "
        "lower-priority milestone until then and lean on Milestone 3's binary "
        "failure prediction (which has a much more workable sample size) plus "
        "root-cause explanation (Milestone 5) as the primary user-facing signal.",
    ]
    return "\n".join(lines)


def run_training(input_path, n_folds):
    settings.ensure_directories()

    log.info("Loading processed dataset from %s", input_path)
    df = pd.read_parquet(input_path)

    X, y = build_failure_type_feature_matrix(df)
    groups = df.loc[X.index, "tire_id"]

    class_counts = y.value_counts().to_dict()
    log.info("Failure rows: %d, class counts: %s", len(X), class_counts)

    encoder = LabelEncoder()
    encoder.fit(FAILURE_TYPE_CLASSES)  # fixed order regardless of what's present
    y_encoded = encoder.transform(y)

    model_reports = {}
    model_builders = {
        "Random Forest": (build_random_forest_type_pipeline, False),
        "XGBoost": (build_xgboost_type_pipeline, True),
    }

    trained_models = {}
    for model_name, (build_fn, is_xgb) in model_builders.items():
        log.info("=== Cross-validating %s ===", model_name)
        y_true, y_pred, fold_info = cross_validate_pooled(
            build_fn, X, y_encoded, groups, n_folds, is_xgb
        )
        log.info("Pooled predictions from %d folds, %d total predictions", len(fold_info), len(y_true))

        present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        present_class_names = list(encoder.inverse_transform(present_labels))

        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        cm_lines = ["Predicted ->", "  " + "  ".join(f"{c[:12]:>12}" for c in present_class_names)]
        for i, row in enumerate(cm):
            cm_lines.append(f"{present_class_names[i][:12]:>12} " + " ".join(f"{v:>12}" for v in row))
        cm_str = "\n".join(cm_lines)

        report_dict = classification_report(
            y_true, y_pred, labels=present_labels, target_names=present_class_names,
            output_dict=True, zero_division=0,
        )
        per_class = {k: v for k, v in report_dict.items() if k in present_class_names}

        model_reports[model_name] = {
            "confusion_matrix_str": cm_str,
            "per_class": per_class,
            "accuracy": report_dict["accuracy"],
            "macro_f1": report_dict["macro avg"]["f1-score"],
        }

        # Train final model on ALL failure rows for serialization.
        if is_xgb:
            final_model = build_fn(num_classes=len(FAILURE_TYPE_CLASSES))
        else:
            final_model = build_fn()
        final_model.fit(X, y_encoded)
        trained_models[model_name] = final_model

    log.info("Building evaluation report...")
    report_md = build_report(model_reports, len(X), class_counts)
    report_path = DOCS_DIR / "failure_type_evaluation.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("Wrote evaluation report to %s", report_path)

    best_model_name = max(model_reports, key=lambda name: model_reports[name]["macro_f1"])
    log.info("Best model by macro F1: %s", best_model_name)

    model_path = settings.model_dir / "failure_type_model.joblib"
    joblib.dump(
        {"model": trained_models[best_model_name], "label_encoder": encoder},
        model_path,
    )
    log.info("Saved final model (%s) to %s", best_model_name, model_path)

    metadata = {
        "model_type": best_model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_failure_rows": len(X),
        "class_counts": class_counts,
        "classes": FAILURE_TYPE_CLASSES,
        "macro_f1": model_reports[best_model_name]["macro_f1"],
        "accuracy": model_reports[best_model_name]["accuracy"],
    }
    metadata_path = settings.model_dir / "failure_type_model.meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved model metadata to %s", metadata_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the failure-type classifier.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input) if args.input else _find_latest_processed_file()
    run_training(input_path, args.folds)


if __name__ == "__main__":
    main()