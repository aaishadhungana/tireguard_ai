"""
Root-cause explanation engine.

CORE DESIGN PRINCIPLE: every claim this module produces is tagged with exactly one
of these sources, and the tag is never dropped downstream:

- "model_contribution": a SHAP value — a mathematical fact about how
  much a feature moved THIS MODEL's prediction for THIS row, relative
  to the model's average prediction. This is a fact about the model's
  learned function, not a fact about tire physics.
- "engineering_rule": a hardcoded domain threshold (e.g. "pressure below
  24 psi is unsafe") written directly here, independent of the model.
  These come from tire-engineering conventions, not from training data.

WHAT THIS MODULE DOES NOT CLAIM: that the model discovered causation.
The simulator's physics ARE causal by construction (Milestone 1 wrote
the causal rules), but the MODEL only ever sees correlational patterns
in the generated data, it has no mechanism to verify or discover that
causal structure. The most honest claim available is "the model's
explanation is CONSISTENT WITH the known simulator physics" — a
different and weaker claim than "the model found the cause" — and
that phrasing is used throughout this module and its reports.

VERIFIED MATHEMATICAL PROPERTY (checked against the real trained model,
not assumed): SHAP values for the positive class sum, plus the
explainer's base value, to exactly the model's predicted probability
for that row. tests/test_explain.py checks this holds for every
explanation this module produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.simulator.physics import (
    LEGAL_MIN_TREAD_DEPTH_MM,
    MAX_RATED_LOAD_KG,
    MIN_SAFE_PRESSURE_PSI,
)

OVERHEATING_FAILURE_TEMP_C = 110.0


@dataclass
class Contribution:
    feature: str
    value: object  
    contribution: float  
    direction: str  
    source: str  


@dataclass
class RootCauseExplanation:
    failure_probability: float
    risk_level: str
    top_model_contributions: list = field(default_factory=list)
    triggered_engineering_rules: list = field(default_factory=list)
    caveat: str = (
        "Model contributions describe how this MODEL weighted each feature for "
        "this prediction — they are consistent with, but not proof of, the "
        "underlying physical cause. Engineering rules are independent, "
        "hardcoded domain thresholds, not something the model learned."
    )


# ---------------------------------------------------------------------
# Engineering rules: hardcoded, independent of the model. Each rule
# states the exact threshold it checks so it's auditable, not a black box.
# ---------------------------------------------------------------------

def _check_engineering_rules(row: pd.Series) -> list:
    triggered = []

    pressure = row.get("pressure")
    if pressure is not None and not pd.isna(pressure) and pressure < MIN_SAFE_PRESSURE_PSI:
        triggered.append(
            Contribution(
                feature="pressure",
                value=float(pressure),
                contribution=float(MIN_SAFE_PRESSURE_PSI - pressure),
                direction="increases_risk",
                source="engineering_rule",
            )
        )

    tread_depth = row.get("tread_depth")
    if tread_depth is not None and not pd.isna(tread_depth) and tread_depth <= LEGAL_MIN_TREAD_DEPTH_MM * 1.5:
        triggered.append(
            Contribution(
                feature="tread_depth",
                value=float(tread_depth),
                contribution=float(LEGAL_MIN_TREAD_DEPTH_MM * 1.5 - tread_depth),
                direction="increases_risk",
                source="engineering_rule",
            )
        )

    temperature = row.get("temperature")
    if temperature is not None and not pd.isna(temperature) and temperature >= OVERHEATING_FAILURE_TEMP_C * 0.8:
        triggered.append(
            Contribution(
                feature="temperature",
                value=float(temperature),
                contribution=float(temperature - OVERHEATING_FAILURE_TEMP_C * 0.8),
                direction="increases_risk",
                source="engineering_rule",
            )
        )

    load = row.get("load")
    if load is not None and not pd.isna(load) and load > MAX_RATED_LOAD_KG:
        triggered.append(
            Contribution(
                feature="load",
                value=float(load),
                contribution=float(load - MAX_RATED_LOAD_KG),
                direction="increases_risk",
                source="engineering_rule",
            )
        )

    return triggered


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.3:
        return "MEDIUM"
    return "LOW"


class RootCauseExplainer:
    """Wraps a trained failure-prediction Pipeline (preprocess + tree
    model) with SHAP TreeExplainer, producing explanations that keep
    model-derived and rule-derived claims strictly separate.

    Only supports tree-based models (RandomForest, XGBoost), the ones
    this project actually trains, since TreeExplainer requires it and
    is exact (not an approximation) for these model types.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = list(self.preprocess.get_feature_names_out())

    def explain(self, row: pd.Series, top_n: int = 5) -> RootCauseExplanation:
        row_df = row.to_frame().T
        X_transformed = self.preprocess.transform(row_df)

        shap_values = self.explainer.shap_values(X_transformed)
       
        if isinstance(shap_values, list):
            sv_row = shap_values[1][0]  # class-1 (failure) SHAP values
            base_value = self.explainer.expected_value[1]
        elif shap_values.ndim == 3:
            sv_row = shap_values[0, :, 1]
            base_value = self.explainer.expected_value[1]
        else:
            sv_row = shap_values[0]
            base_value = self.explainer.expected_value

        probability = float(base_value + sv_row.sum())
        probability = max(0.0, min(1.0, probability))  

        contributions = []
        for name, value, contrib in zip(self.feature_names, X_transformed[0], sv_row):
            original_col = self._original_column_name(name)
            raw_value = row.get(original_col, value)
            if isinstance(raw_value, (int, float, np.floating)) and not isinstance(raw_value, bool):
                raw_value = float(raw_value)
            contributions.append(
                Contribution(
                    feature=original_col,
                    value=raw_value,
                    contribution=float(contrib),
                    direction="increases_risk" if contrib > 0 else "decreases_risk",
                    source="model_contribution",
                )
            )

        aggregated = {}
        raw_values = {}
        for c in contributions:
            aggregated[c.feature] = aggregated.get(c.feature, 0.0) + c.contribution
            raw_values[c.feature] = c.value
        merged_contributions = [
            Contribution(
                feature=feat,
                value=raw_values[feat],
                contribution=val,
                direction="increases_risk" if val > 0 else "decreases_risk",
                source="model_contribution",
            )
            for feat, val in aggregated.items()
        ]
        merged_contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

        return RootCauseExplanation(
            failure_probability=round(probability, 4),
            risk_level=_risk_level(probability),
            top_model_contributions=merged_contributions[:top_n],
            triggered_engineering_rules=_check_engineering_rules(row),
        )

    def _original_column_name(self, encoded_name: str) -> str:
        """ColumnTransformer prefixes names like 'numeric__pressure' or
        'categorical__road_type_highway'. Strip the transformer prefix
        and, for one-hot dummies, the appended category value."""
        if "__" in encoded_name:
            _, rest = encoded_name.split("__", 1)
        else:
            rest = encoded_name

        categorical_cols = [
            "road_type", "weather",
            "pressure_sensor_fault", "temperature_sensor_fault", "tread_sensor_fault",
        ]
        for col in categorical_cols:
            if rest.startswith(col + "_"):
                return col
        return rest