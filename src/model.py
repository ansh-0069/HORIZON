from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd
from src.direct_model import DirectRidgeModel, inference_features


@dataclass
class HorizonModel:
    model_version: str
    global_roas: float
    global_log_sigma: float
    month_roas_factors: dict[int, float]
    direct_models: dict[int, DirectRidgeModel]
    min_history_days: int = 28
    target_roas: float = 4.0

    def forecast_campaigns(self, canonical: pd.DataFrame, horizon_days: int, budget_overrides: dict[str, float] | None = None) -> pd.DataFrame:
        budget_overrides = budget_overrides or {}
        as_of = canonical["date"].max()
        window_start = as_of - pd.Timedelta(days=27)
        recent = canonical[(canonical["date"] >= window_start) & (canonical["date"] <= as_of)].copy()
        future_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        seasonal_factor = sum(self.month_roas_factors.get(int(date.month), 1.0) for date in future_dates) / horizon_days
        group_keys = ["source_system", "source_campaign_id", "channel", "campaign_type", "campaign_name"]
        rows: list[dict[str, object]] = []
        for key, history in canonical.groupby(group_keys, dropna=False):
            source, campaign_id, channel, campaign_type, campaign_name = key
            campaign_recent = recent[(recent["source_system"] == source) & (recent["source_campaign_id"] == campaign_id)]
            active_days = int(campaign_recent["date"].nunique())
            recent_spend = float(campaign_recent["spend"].sum())
            recent_revenue = float(campaign_recent["revenue"].sum())
            # Historical campaigns that are not currently delivering should not
            # silently receive a future budget in a baseline media plan.
            if recent_spend <= 0:
                continue
            historic_spend = float(history["spend"].sum())
            historic_revenue = float(history["revenue"].sum())
            historic_roas = historic_revenue / max(historic_spend, 1e-9)
            recent_roas = recent_revenue / max(recent_spend, 1e-9) if recent_spend > 0 else historic_roas
            reliability = min(1.0, active_days / self.min_history_days)
            roas = reliability * recent_roas + (1.0 - reliability) * historic_roas
            roas = (0.70 * roas + 0.30 * self.global_roas) * seasonal_factor
            daily_spend = recent_spend / max(active_days, 1)
            override = budget_overrides.get(str(campaign_id))
            planned_budget = float(override) if override is not None else max(0.0, daily_spend * horizon_days)
            support = max(historic_spend / max(int(history["date"].nunique()), 1) * horizon_days, 1.0)
            extrapolation = planned_budget > 1.5 * support
            sigma = self.global_log_sigma * (1.0 + (1.0 - reliability) + (0.55 if extrapolation else 0.0))
            # A normalized logarithmic response curve preserves the baseline forecast
            # at the current plan while making incremental budget progressively less productive.
            baseline_budget = max(daily_spend * horizon_days, 1.0)
            saturation_scale = max(baseline_budget * 1.5, 1.0)
            normalization = baseline_budget / (saturation_scale * math.log1p(baseline_budget / saturation_scale))
            revenue_p50 = max(0.0, roas * saturation_scale * math.log1p(planned_budget / saturation_scale) * normalization)
            if horizon_days in self.direct_models:
                direct_features = inference_features(history, str(channel), str(campaign_type), as_of, horizon_days, planned_budget)
                revenue_p10, revenue_p50, revenue_p90 = (float(value[0]) for value in self.direct_models[horizon_days].predict(direct_features))
            else:
                revenue_p10 = revenue_p50 * math.exp(-1.28155 * sigma)
                revenue_p90 = revenue_p50 * math.exp(1.28155 * sigma)
            spend_p10, spend_p90 = planned_budget * 0.90, planned_budget * 1.05
            flags = []
            if active_days < self.min_history_days:
                flags.append("sparse_recent_history")
            if extrapolation:
                flags.append("budget_extrapolation")
            if planned_budget > 1.25 * baseline_budget:
                flags.append("diminishing_returns")
            rows.append({
                "level": "campaign", "channel": channel, "campaign_type": campaign_type,
                "campaign_id": str(campaign_id), "campaign_name": campaign_name,
                "planned_budget": planned_budget, "predicted_revenue_p10": revenue_p10,
                "predicted_revenue_p50": revenue_p50, "predicted_revenue_p90": revenue_p90,
                "predicted_spend_p10": spend_p10, "predicted_spend_p50": planned_budget,
                "predicted_spend_p90": spend_p90, "quality_flags": ";".join(flags),
            })
        return pd.DataFrame(rows)
