from __future__ import annotations

from dataclasses import replace
import math
from typing import Mapping

import pandas as pd

from src.forecast import build_forecast
from src.model import HorizonModel


def validate_budget_overrides(overrides: Mapping[str, float]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for campaign_id, budget in overrides.items():
        value = float(budget)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Budget must be finite and non-negative for campaign {campaign_id}")
        parsed[str(campaign_id)] = value
    return parsed


def simulate_budget_plan(
    model: HorizonModel,
    canonical: pd.DataFrame,
    horizon_days: int,
    campaign_budgets: Mapping[str, float],
    target_roas: float | None = None,
) -> pd.DataFrame:
    if horizon_days not in {30, 60, 90}:
        raise ValueError("Horizon must be 30, 60, or 90 days")
    if target_roas is not None and (not math.isfinite(float(target_roas)) or target_roas <= 0):
        raise ValueError("Target ROAS must be finite and positive")
    scenario_model = replace(model, target_roas=float(target_roas)) if target_roas is not None else model
    return build_forecast(scenario_model, canonical, horizon_days, validate_budget_overrides(campaign_budgets))
