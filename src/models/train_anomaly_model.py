from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.config.settings import settings
from src.models.anomaly_model import (
    ANOMALY_FEATURES,
    any_ground_truth_fault,
    any_rule_based_flag,
    build_anomaly_feature_matrix,
    build_isolation_forest_pipeline,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def _find_latest_processed_file():
    candidates = sorted(settings.data_processed_dir.glob("tire_telemetry_processed_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No processed datasets found in {settings.data_processed_dir}.")
    return candidates[-1]


def build_report(ml_metrics, rule_metrics, agreement, ground_truth_rate, n_rows):
    lines = [
        "# TireGuard AI — Sensor Anomaly Detection Evaluation",
        "",
        "> Generated programmatically by `src/models/train_anomaly_model.py`.",
        "",
        "## An Unusual Advantage For This Evaluation (Read First)",
        "",
        "Because this is synthetic data with simulator-injected sensor faults "
        "(`pressure_sensor_fault`, `temperature_sensor_fault`, `tread_sensor_fault` "
        "columns), precision/recall against a KNOWN ground truth can actually be "
        "computed here. Most real-world anomaly detection cannot do this — there "
        "is no ground truth to check against in production, only after-the-fact "
        "manual review of flagged cases. The strong numbers below reflect this "
        "synthetic advantage and should not be read as what to expect when this "
        "same approach is pointed at real fleet data with no known answer key.",
        "",
        f"- Total rows evaluated: **{n_rows:,}**",
        f"- Rows with a real injected sensor fault (any of the 3 sensors): "
        f"**{100 * ground_truth_rate:.2f}%**",
        "",
    ]

    if ground_truth_rate > 0.5:
        lines += [
            "## Important Caveat: 'Anomaly' Is Not a Coherent Frame Here",
            "",
            f"At current simulator settings, **{100 * ground_truth_rate:.1f}% of rows "
            f"have a real sensor fault** — a direct consequence of a Milestone 2 "
            "finding: sensor drift persists across many consecutive steps once "
            "triggered, so the effective fault rate is far higher than the "
            "`sensor_fault_rate` config parameter's nominal value, and this "
            "compounds further across 3 independently-faulted sensors combined "
            "with 'any of 3' logic. IsolationForest's `contamination` parameter "
            "is capped at 0.5 (it assumes anomalies are a MINORITY by design), so "
            f"it was clamped to 0.5 here rather than the true {100 * ground_truth_rate:.1f}%. "
            "The practical implication: at this fault rate, 'is this row anomalous' "
            "is close to a coin flip rather than a rare-event detection problem, "
            "and the metrics below should be read with that in mind. The right fix "
            "is upstream, in the simulator's fault injection persistence model "
            "(flagged in Milestone 2, not re-fixed here to keep this milestone "
            "scoped to detection rather than reopening simulator design).",
            "",
        ]

    lines += [
        "## Two Separate Detection Approaches (Never Blended)",
        "",
        "- **`rule_based`**: Milestone 2's existing out-of-range / missing-value "
        "flags (simple, auditable thresholds).",
        "- **`ml_based`**: Isolation Forest trained on observed sensor readings "
        "and their derived features only (never on the ground-truth fault "
        "columns or true physical values, which a real system would not have).",
        "",
        "## Results Against Ground Truth",
        "",
        "| Approach | Precision | Recall | F1 |",
        "|---|---|---|---|",
        f"| rule_based | {rule_metrics['precision']:.3f} | {rule_metrics['recall']:.3f} | {rule_metrics['f1']:.3f} |",
        f"| ml_based (Isolation Forest) | {ml_metrics['precision']:.3f} | {ml_metrics['recall']:.3f} | {ml_metrics['f1']:.3f} |",
        "",
        "### Rule-Based Confusion Matrix",
        "",
        "```",
        rule_metrics["confusion_matrix_str"],
        "```",
        "",
        "### ML-Based Confusion Matrix",
        "",
        "```",
        ml_metrics["confusion_matrix_str"],
        "```",
        "",
        "## Agreement Between The Two Approaches",
        "",
        f"- Both flagged: **{agreement['both']:,}** rows",
        f"- Only rule_based flagged: **{agreement['rule_only']:,}** rows",
        f"- Only ml_based flagged: **{agreement['ml_only']:,}** rows",
        f"- Neither flagged: **{agreement['neither']:,}** rows",
        "",
        "Rows flagged by ML but not by rules are the more interesting case — "
        "these are anomalies the fixed-threshold rules structurally cannot catch "
        "(e.g. `drift`, which stays within any single-reading plausible range but "
        "looks anomalous relative to that tire's own recent trend).",
        "",
        "## Known Limitations",
        "",
        "- Evaluated only on this specific simulator run's fault mix and rates — "
        "generalization to different fault distributions is untested.",
        "- Isolation Forest's `contamination` parameter was set from this "
        "dataset's OWN observed fault rate, which a real deployment would not "
        "know in advance and would need to estimate or tune differently.",
        "- Does not distinguish WHICH sensor is faulted or what fault type — "
        "just whether the row looks anomalous overall. A finer-grained "
        "per-sensor, per-fault-type model is a reasonable next iteration.",
    ]
    return "\n".join(lines)


def _metrics_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    cm_str = (
        "                Predicted\n"
        "                Normal  Anomaly\n"
        f"Actual Normal   {cm[0][0]:>6}  {cm[0][1]:>7}\n"
        f"Actual Anomaly  {cm[1][0]:>6}  {cm[1][1]:>7}"
    )
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix_str": cm_str,
    }


def run_training(input_path):
    settings.ensure_directories()

    log.info("Loading processed dataset from %s", input_path)
    df = pd.read_parquet(input_path)

    log.info("Computing ground truth and rule-based labels...")
    ground_truth = df.apply(any_ground_truth_fault, axis=1)
    rule_based_flags = df.apply(any_rule_based_flag, axis=1)

    ground_truth_rate = ground_truth.mean()
    log.info("Ground truth fault rate: %.4f%%", 100 * ground_truth_rate)
    if ground_truth_rate > 0.5:
        log.warning(
            "Ground truth fault rate (%.1f%%) exceeds IsolationForest's max "
            "contamination of 0.5 — 'anomaly' detection is not a coherent frame "
            "when most rows are anomalous. Clamping contamination to 0.5 and "
            "flagging this prominently in the report rather than silently "
            "capping it.",
            100 * ground_truth_rate,
        )
    contamination = min(max(ground_truth_rate, 0.001), 0.5)

    X = build_anomaly_feature_matrix(df)

    log.info("Training Isolation Forest (contamination=%.4f)...", contamination)
    pipeline = build_isolation_forest_pipeline(contamination=contamination)
    pipeline.fit(X)
    ml_predictions = pipeline.predict(X) == -1  # -1 = anomaly

    ml_metrics = _metrics_dict(ground_truth, ml_predictions)
    rule_metrics = _metrics_dict(ground_truth, rule_based_flags)

    log.info("ML-based metrics: %s", {k: v for k, v in ml_metrics.items() if k != "confusion_matrix_str"})
    log.info("Rule-based metrics: %s", {k: v for k, v in rule_metrics.items() if k != "confusion_matrix_str"})

    agreement = {
        "both": int((ml_predictions & rule_based_flags).sum()),
        "rule_only": int((~ml_predictions & rule_based_flags).sum()),
        "ml_only": int((ml_predictions & ~rule_based_flags).sum()),
        "neither": int((~ml_predictions & ~rule_based_flags).sum()),
    }
    log.info("Agreement: %s", agreement)

    report_md = build_report(ml_metrics, rule_metrics, agreement, ground_truth_rate, len(df))
    report_path = DOCS_DIR / "anomaly_detection_evaluation.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("Wrote evaluation report to %s", report_path)

    model_path = settings.model_dir / "anomaly_model.joblib"
    joblib.dump(pipeline, model_path)
    log.info("Saved model to %s", model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(df),
        "feature_columns": ANOMALY_FEATURES,
        "contamination_used": float(contamination),
        "ml_metrics": {k: v for k, v in ml_metrics.items() if k != "confusion_matrix_str"},
        "rule_metrics": {k: v for k, v in rule_metrics.items() if k != "confusion_matrix_str"},
        "agreement": agreement,
    }
    metadata_path = settings.model_dir / "anomaly_model.meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved model metadata to %s", metadata_path)


def main():
    input_path = _find_latest_processed_file()
    run_training(input_path)


if __name__ == "__main__":
    main()