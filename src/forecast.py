from __future__ import annotations

import hashlib
import math
import pandas as pd

from src.model import HorizonModel


def _roas(revenue: pd.Series, spend: pd.Series) -> pd.Series:
    return revenue / spend.clip(lower=1e-9)


def _risk(row: pd.Series) -> float:
    width = (row["predicted_revenue_p90"] - row["predicted_revenue_p10"]) / max(row["predicted_revenue_p50"], 1e-9)
    chance_miss = 1.0 - float(row["probability_roas_above_target"])
    return min(100.0, round(100.0 * min(1.0, 0.55 * chance_miss + 0.25 * min(width / 3.0, 1.0) + (0.20 if "extrapolation" in str(row["quality_flags"]) else 0.0)), 1))


def _rollup(leaves: pd.DataFrame, group_columns: list[str], level: str) -> pd.DataFrame:
    numeric = [c for c in leaves.columns if c.startswith("planned_") or c.startswith("predicted_revenue_") or c.startswith("predicted_spend_")]
    result = leaves.groupby(group_columns, dropna=False)[numeric].sum().reset_index()
    result["level"] = level
    if "campaign_type" not in result:
        result["campaign_type"] = "ALL"
    if "channel" not in result:
        result["channel"] = "ALL"
    result["campaign_id"] = "ALL"
    result["campaign_name"] = "ALL"
    result["quality_flags"] = "reconciled_rollup"
    return result


def build_forecast(model: HorizonModel, canonical: pd.DataFrame, horizon_days: int, budget_overrides: dict[str, float] | None = None) -> pd.DataFrame:
    leaves = model.forecast_campaigns(canonical, horizon_days, budget_overrides)
    if leaves.empty:
        raise ValueError("No campaign-level forecasts could be produced")
    channel = _rollup(leaves, ["channel"], "channel")
    campaign_type = _rollup(leaves, ["channel", "campaign_type"], "campaign_type")
    overall = _rollup(leaves.assign(all="ALL"), ["all"], "overall").drop(columns=["all"])
    all_levels = pd.concat([leaves, campaign_type, channel, overall], ignore_index=True, sort=False)
    for quantile in ("p10", "p50", "p90"):
        all_levels[f"predicted_roas_{quantile}"] = _roas(all_levels[f"predicted_revenue_{quantile}"], all_levels[f"predicted_spend_{quantile}"])
    def target_probability(row: pd.Series) -> float:
        median = max(float(row["predicted_roas_p50"]), 1e-9)
        spread = math.log(max(float(row["predicted_roas_p90"]), 1e-9)) - math.log(max(float(row["predicted_roas_p10"]), 1e-9))
        sigma = max(spread / (2.0 * 1.28155), 1e-6)
        z = (math.log(model.target_roas) - math.log(median)) / sigma
        return round(max(0.0, min(1.0, 0.5 * math.erfc(z / math.sqrt(2.0)))), 4)

    all_levels["probability_roas_above_target"] = all_levels.apply(target_probability, axis=1)
    all_levels["risk_score"] = all_levels.apply(_risk, axis=1)
    seed_text = f"{model.model_version}|{canonical['date'].max().date()}|{horizon_days}|{len(canonical)}"
    all_levels["forecast_id"] = hashlib.sha256(seed_text.encode()).hexdigest()[:16]
    all_levels["horizon_days"] = horizon_days
    all_levels["model_version"] = model.model_version
    return all_levels
