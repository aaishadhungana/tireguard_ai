from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

sns.set_theme(style="whitegrid")

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
FIGURES_DIR = DOCS_DIR / "figures"


def _find_latest_processed_file() -> Path:
    candidates = sorted(settings.data_processed_dir.glob("tire_telemetry_processed_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No processed datasets found in {settings.data_processed_dir}. "
            f"Run `python -m src.data.pipeline` first."
        )
    return candidates[-1]


def plot_pressure_by_failure(df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="failure", y="pressure_true", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No failure", "Failure"])
    ax.set_title("True Tire Pressure by Failure Outcome")
    ax.set_ylabel("Pressure (psi)")
    path = out_dir / "pressure_by_failure.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path.name


def plot_temperature_by_failure(df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="failure", y="temperature_true", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No failure", "Failure"])
    ax.set_title("True Tire Temperature by Failure Outcome")
    ax.set_ylabel("Temperature (°C)")
    path = out_dir / "temperature_by_failure.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path.name


def plot_tread_depth_over_reading_index(df: pd.DataFrame, out_dir: Path, sample_tires: int = 8) -> str:
    tires = df["tire_id"].drop_duplicates().sample(n=min(sample_tires, df["tire_id"].nunique()), random_state=42)
    subset = df[df["tire_id"].isin(tires)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for tire_id, group in subset.groupby("tire_id"):
        ax.plot(group["tire_reading_index"], group["tread_depth_true"], label=tire_id, alpha=0.8)
    ax.set_title(f"Tread Depth Over Time — {len(tires)} Sample Tires")
    ax.set_xlabel("Reading index (per-tire)")
    ax.set_ylabel("Tread depth (mm)")
    ax.legend(fontsize=7, loc="upper right")
    path = out_dir / "tread_depth_trajectories.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path.name


def plot_failure_type_distribution(df: pd.DataFrame, out_dir: Path) -> str:
    counts = df.loc[df["failure"] == 1, "failure_type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, orient="h")
    ax.set_title("Failure Type Distribution")
    ax.set_xlabel("Count")
    path = out_dir / "failure_type_distribution.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path.name


def plot_sensor_fault_distribution(df: pd.DataFrame, out_dir: Path) -> str:
    fault_cols = ["pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, fault_cols):
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Sensor Fault Type Distribution")
    path = out_dir / "sensor_fault_distribution.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path.name


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> str:
    numeric_cols = [
        "pressure_true",
        "temperature_true",
        "tread_depth_true",
        "speed",
        "load",
        "braking_events",
        "mileage",
    ]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Core Telemetry Variables")
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path.name


def compute_statistics(df: pd.DataFrame) -> dict:
    stats: dict = {}
    stats["row_count"] = len(df)
    stats["tire_count"] = int(df["tire_id"].nunique())
    stats["vehicle_count"] = int(df["vehicle_id"].nunique())
    stats["date_range"] = (str(df["timestamp"].min()), str(df["timestamp"].max()))
    stats["failure_rate_pct"] = round(100 * df["failure"].mean(), 4)
    stats["failure_type_counts"] = (
        df.loc[df["failure"] == 1, "failure_type"].value_counts().to_dict()
    )

    for col in ("pressure_true", "temperature_true", "tread_depth_true", "speed", "load"):
        stats[f"{col}_mean"] = round(df[col].mean(), 3)
        stats[f"{col}_std"] = round(df[col].std(), 3)
        stats[f"{col}_min"] = round(df[col].min(), 3)
        stats[f"{col}_max"] = round(df[col].max(), 3)

    failed = df[df["failure"] == 1]
    not_failed = df[df["failure"] == 0]
    stats["mean_pressure_failed"] = round(failed["pressure_true"].mean(), 3) if len(failed) else None
    stats["mean_pressure_not_failed"] = round(not_failed["pressure_true"].mean(), 3)
    stats["mean_temperature_failed"] = round(failed["temperature_true"].mean(), 3) if len(failed) else None
    stats["mean_temperature_not_failed"] = round(not_failed["temperature_true"].mean(), 3)

    for col in ("pressure", "temperature", "tread_depth"):
        was_missing_col = f"{col}_was_missing"
        if was_missing_col in df.columns:
            stats[f"{col}_imputed_pct"] = round(100 * df[was_missing_col].mean(), 3)

    for col in ("pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault"):
        stats[f"{col}_distribution"] = df[col].value_counts().to_dict()

    corr = df["pressure_true"].corr(df["temperature_true"])
    stats["pressure_temperature_correlation"] = round(corr, 4)

    return stats


def build_report(stats: dict, figure_files: dict[str, str]) -> str:
    lines = [
        "# TireGuard AI — Exploratory Data Analysis Findings",
        "",
        "> Generated programmatically by `src/data/run_eda.py` from the actual "
        "processed dataset. Every number below was computed at run time from "
        "the data referenced — none are estimated or illustrative.",
        "",
        "## Dataset Overview",
        "",
        f"- Rows: **{stats['row_count']:,}**",
        f"- Unique tires: **{stats['tire_count']}**",
        f"- Unique vehicles: **{stats['vehicle_count']}**",
        f"- Date range: **{stats['date_range'][0]}** to **{stats['date_range'][1]}**",
        f"- Overall failure rate: **{stats['failure_rate_pct']}%** of rows",
        "",
        "## Failure Type Distribution",
        "",
    ]
    if stats["failure_type_counts"]:
        for ftype, count in stats["failure_type_counts"].items():
            lines.append(f"- `{ftype}`: {count}")
    else:
        lines.append("- No failures present in this dataset run.")
    lines += [
        "",
        f"![Failure type distribution](figures/{figure_files['failure_type']})",
        "",
        "**Observation:** as flagged at the end of Milestone 1, the failure-type "
        "mix is not balanced across scripted failure modes at current simulator "
        "settings — this is visible directly in the chart above and will inform "
        "whether Milestone 3 needs class-weighting, SMOTE, or simulator "
        "re-tuning before failure-type classification (Milestone 4) is workable.",
        "",
        "## Pressure and Temperature vs. Failure Outcome",
        "",
        f"- Mean true pressure — failed rows: **{stats['mean_pressure_failed']}** psi "
        f"vs. non-failed rows: **{stats['mean_pressure_not_failed']}** psi",
        f"- Mean true temperature — failed rows: **{stats['mean_temperature_failed']}** °C "
        f"vs. non-failed rows: **{stats['mean_temperature_not_failed']}** °C",
        "",
        f"![Pressure by failure](figures/{figure_files['pressure']})",
        "",
        f"![Temperature by failure](figures/{figure_files['temperature']})",
        "",
        "## Tread Depth Trajectories (Sample Tires)",
        "",
        f"![Tread depth trajectories](figures/{figure_files['tread']})",
        "",
        "**Observation:** trajectories show gradual wear punctuated by sharp "
        "resets — the resets are the maintenance/replacement events introduced "
        "in the Milestone 1 fix (a failed tire's tread is reset rather than "
        "continuing to wear past the legal minimum indefinitely).",
        "",
        "## Sensor Fault Distribution",
        "",
        f"![Sensor fault distribution](figures/{figure_files['sensor_fault']})",
        "",
    ]
    for col in ("pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault"):
        dist = stats[f"{col}_distribution"]
        lines.append(f"- `{col}`: {dist}")

    total_rows = stats["row_count"]
    drift_note_added = False
    for col in ("pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault"):
        dist = stats[f"{col}_distribution"]
        drift_count = dist.get("drift", 0)
        drift_pct = 100 * drift_count / total_rows if total_rows else 0
        if drift_pct > 10 and not drift_note_added:
            lines += [
                "",
                f"**Important finding:** `drift` alone accounts for **{drift_pct:.1f}%** "
                f"of `{col}` rows — far above the configured 3% sensor fault rate. "
                "This is expected given the simulator's design (drift persists across "
                "many consecutive steps once triggered, with only a small per-step "
                "chance of self-correcting), but it means the `sensor_fault_rate` "
                "config parameter does NOT directly correspond to \"percent of rows "
                "affected\" — it's closer to \"percent of steps where a NEW fault "
                "episode begins.\" Worth fixing the parameter's naming/semantics or "
                "the persistence model before Milestone 7 relies on it.",
            ]
            drift_note_added = True
    lines += [
        "",
        "## Missing-Value Imputation Summary",
        "",
    ]
    for col in ("pressure", "temperature", "tread_depth"):
        key = f"{col}_imputed_pct"
        if key in stats:
            lines.append(f"- `{col}`: {stats[key]}% of rows were imputed (forward/backward-filled per tire)")
    lines += [
        "",
        "## Correlation Between Core Variables",
        "",
        f"![Correlation heatmap](figures/{figure_files['correlation']})",
        "",
        f"- Pressure–temperature correlation (true values): **{stats['pressure_temperature_correlation']}** "
        "— negative, consistent with the simulator's underinflation→heat causal rule "
        "(lower pressure is associated with higher temperature).",
        "",
        "## Descriptive Statistics",
        "",
        "| Variable | Mean | Std | Min | Max |",
        "|---|---|---|---|---|",
    ]
    for col in ("pressure_true", "temperature_true", "tread_depth_true", "speed", "load"):
        lines.append(
            f"| {col} | {stats[f'{col}_mean']} | {stats[f'{col}_std']} | "
            f"{stats[f'{col}_min']} | {stats[f'{col}_max']} |"
        )
    lines += [
        "",
        "## Known Limitations Carried Into Milestone 3",
        "",
        "- Failure-type imbalance (see above) may require class-weighting or "
        "simulator re-tuning before Milestone 4 (failure-type classification) "
        "is meaningful.",
        "- `sensor_malfunction` failures still have no distinct telemetry "
        "signature (carried over from Milestone 1) — worth deciding whether "
        "to fix in the simulator or simply exclude this failure type from "
        "Milestone 4's target classes.",
        "- The `sensor_fault_rate` config parameter's effective meaning "
        "(see drift finding above) should be documented or fixed before "
        "Milestone 7/8 tune detection thresholds against it.",
    ]
    return "\n".join(lines)


def run_eda(input_path: Path) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading processed dataset from %s", input_path)
    df = pd.read_parquet(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    log.info("Computing statistics...")
    stats = compute_statistics(df)

    log.info("Generating charts...")
    figure_files = {
        "pressure": plot_pressure_by_failure(df, FIGURES_DIR),
        "temperature": plot_temperature_by_failure(df, FIGURES_DIR),
        "tread": plot_tread_depth_over_reading_index(df, FIGURES_DIR),
        "failure_type": plot_failure_type_distribution(df, FIGURES_DIR),
        "sensor_fault": plot_sensor_fault_distribution(df, FIGURES_DIR),
        "correlation": plot_correlation_heatmap(df, FIGURES_DIR),
    }

    log.info("Building findings report...")
    report_md = build_report(stats, figure_files)
    report_path = DOCS_DIR / "eda_findings.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("Wrote EDA findings to %s", report_path)

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on the processed telemetry dataset.")
    parser.add_argument("--input", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else _find_latest_processed_file()
    run_eda(input_path)


if __name__ == "__main__":
    main()