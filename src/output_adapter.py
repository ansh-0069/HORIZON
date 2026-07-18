from __future__ import annotations

import numpy as np
import pandas as pd
from src.contracts import FORECAST_COLUMNS


def to_submission_schema(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in FORECAST_COLUMNS:
        if column not in output:
            output[column] = "" if column in {"campaign_id", "campaign_name", "quality_flags"} else 0.0
    return output[FORECAST_COLUMNS].sort_values(["horizon_days", "level", "channel", "campaign_type", "campaign_id"], kind="stable")


def validate_submission_schema(frame: pd.DataFrame) -> None:
    """Fail fast if the offline runner is about to write an unusable contract.

    The organizers have not published scorer columns in the supplied guide. This
    validator protects Horizon's currently documented aggregate contract and is
    intentionally centralized with the adapter so the two change together when a
    final official schema is released.
    """
    if list(frame.columns) != FORECAST_COLUMNS:
        raise ValueError("predictions.csv does not match the configured submission column order")
    if frame.empty:
        raise ValueError("predictions.csv must contain at least one forecast row")
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"predictions.csv contains missing values: {missing}")
    required_horizons = {30, 60, 90}
    horizons = set(pd.to_numeric(frame["horizon_days"], errors="coerce").dropna().astype(int))
    if horizons != required_horizons:
        raise ValueError(f"predictions.csv must contain exactly horizons {sorted(required_horizons)}")
    if not set(frame["level"].unique()).issubset({"campaign", "campaign_type", "channel", "overall"}):
        raise ValueError("predictions.csv contains an unknown hierarchy level")
    overall = frame[frame["level"] == "overall"]
    if len(overall) != 3 or set(overall["horizon_days"].astype(int)) != required_horizons:
        raise ValueError("predictions.csv must contain one overall row for each horizon")
    quantile_groups = [
        ("predicted_revenue_p10", "predicted_revenue_p50", "predicted_revenue_p90"),
        ("predicted_spend_p10", "predicted_spend_p50", "predicted_spend_p90"),
        ("predicted_roas_p10", "predicted_roas_p50", "predicted_roas_p90"),
    ]
    numeric_columns = [column for group in quantile_groups for column in group] + [
        "planned_budget", "probability_roas_above_target", "risk_score",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("predictions.csv contains non-finite numeric values")
    for p10, p50, p90 in quantile_groups:
        if not ((numeric[p10] <= numeric[p50]) & (numeric[p50] <= numeric[p90])).all():
            raise ValueError(f"predictions.csv contains unordered quantiles for {p50}")
    if (numeric[["planned_budget", *[column for group in quantile_groups for column in group]]] < 0).any().any():
        raise ValueError("predictions.csv contains negative forecast values")
    if not numeric["probability_roas_above_target"].between(0, 1).all():
        raise ValueError("ROAS probability must be between zero and one")
    if not numeric["risk_score"].between(0, 100).all():
        raise ValueError("risk score must be between zero and one hundred")
    if (frame["forecast_id"].astype(str).str.len() == 0).any() or (frame["model_version"].astype(str).str.len() == 0).any():
        raise ValueError("forecast_id and model_version must be populated")
