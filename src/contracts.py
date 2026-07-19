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

# Campaign identifiers are opaque external keys.  They must be read as text so
# a leading zero or an identifier larger than an integer type survives source
# ingestion, optional metadata matching, and submission serialization.
SOURCE_CAMPAIGN_ID_COLUMNS = {
    "google_ads": "campaign_id",
    "microsoft_ads": "CampaignId",
    "meta_ads": "campaign_id",
}

# The protected submission schema has one deliberately fixed revenue measure
# per source.  Optional semantic metadata is an attestation of these mappings,
# not a dynamic remapping mechanism.  Supporting a new source measure requires
# an explicit connector/schema change rather than silently forecasting a
# different field than the one declared to a planner.
CANONICAL_REVENUE_FIELDS = {
    "google_ads": "metrics_conversions_value",
    "microsoft_ads": "Revenue",
    "meta_ads": "conversion",
}

TAXONOMY_FILENAME = "campaign_taxonomy.csv"
TAXONOMY_REQUIRED_COLUMNS = {"source_system", "source_campaign_id", "campaign_type"}
TAXONOMY_SUPPORTED_SOURCES = frozenset({"meta_ads"})

# These review fields are deliberately optional so legacy/evaluator uploads
# remain usable.  When present, they make the provenance of planner-facing
# metadata explicit instead of treating the mere presence of a CSV as an
# approval signal.
REVIEW_STATUS_COLUMN = "review_status"
REVIEW_STATUSES = frozenset({"reviewed", "unreviewed"})

# Optional review artifact for a planner upload.  The evaluator can continue
# with the three supplied platform files alone, but a real cross-channel plan
# must make its money, calendar, and attribution assumptions explicit.
SEMANTICS_FILENAME = "source_semantics.csv"
SEMANTICS_REQUIRED_COLUMNS = {"source_system", "currency", "timezone", "attribution_method", "revenue_field"}

# Optional future media plan for the scored offline path. When absent, each
# horizon uses the model's baseline budget defaults. When present, rows bind
# source-qualified campaign budgets to a specific 30/60/90-day horizon.
MEDIA_PLAN_FILENAME = "media_plan.csv"
MEDIA_PLAN_REQUIRED_COLUMNS = {"source_system", "source_campaign_id", "horizon_days", "planned_budget"}
MEDIA_PLAN_HORIZONS = frozenset({30, 60, 90})

# Filenames that must never be treated as advertising-platform exports.
OPTIONAL_DATA_FILENAMES = frozenset({TAXONOMY_FILENAME, SEMANTICS_FILENAME, MEDIA_PLAN_FILENAME})
