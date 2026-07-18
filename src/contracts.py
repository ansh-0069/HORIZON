from __future__ import annotations

CANONICAL_COLUMNS = [
    "source_system", "source_campaign_id", "date", "channel", "campaign_type",
    "campaign_name", "spend", "revenue", "clicks", "impressions", "conversions",
    "configured_budget", "quality_flags",
]

# Identity columns establish which supplied CSV belongs to each channel. Full
# required-column sets are validated before canonicalization so a partially
# matching file cannot fall through to an unhelpful KeyError later.
SOURCE_IDENTITY_COLUMNS = {
    "google_ads": {"campaign_id", "segments_date", "metrics_cost_micros", "metrics_conversions_value"},
    "microsoft_ads": {"CampaignId", "TimePeriod", "Revenue", "Spend"},
    "meta_ads": {"campaign_id", "date_start", "spend", "conversion"},
}

SOURCE_REQUIRED_COLUMNS = {
    "google_ads": {
        "campaign_id", "segments_date", "campaign_advertising_channel_type", "campaign_name",
        "metrics_cost_micros", "metrics_conversions_value", "metrics_clicks", "metrics_impressions",
        "metrics_conversions", "campaign_budget_amount",
    },
    "microsoft_ads": {
        "CampaignId", "TimePeriod", "CampaignType", "CampaignName", "Revenue", "Spend",
        "Clicks", "Impressions", "Conversions", "DailyBudget",
    },
    "meta_ads": {
        "campaign_id", "date_start", "campaign_name", "conversion", "spend", "clicks",
        "impressions", "daily_budget",
    },
}

TAXONOMY_FILENAME = "campaign_taxonomy.csv"
TAXONOMY_REQUIRED_COLUMNS = {"source_system", "source_campaign_id", "campaign_type"}
