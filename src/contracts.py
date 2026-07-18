from __future__ import annotations

CANONICAL_COLUMNS = [
    "source_system", "source_campaign_id", "date", "channel", "campaign_type",
    "campaign_name", "spend", "revenue", "clicks", "impressions", "conversions",
    "configured_budget", "quality_flags",
]

FORECAST_COLUMNS = [
    "forecast_id", "horizon_days", "level", "channel", "campaign_type",
    "campaign_id", "campaign_name", "planned_budget",
    "predicted_revenue_p10", "predicted_revenue_p50", "predicted_revenue_p90",
    "predicted_spend_p10", "predicted_spend_p50", "predicted_spend_p90",
    "predicted_roas_p10", "predicted_roas_p50", "predicted_roas_p90",
    "probability_roas_above_target", "risk_score", "quality_flags", "model_version",
]
