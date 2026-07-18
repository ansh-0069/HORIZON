from __future__ import annotations

from typing import Any
import pandas as pd

from src.contracts import CANONICAL_COLUMNS


def _meta_campaign_type(name: object) -> tuple[str, str]:
    text = str(name or "").lower()
    for needle, label in (("shopping", "Shopping"), ("search", "Search"), ("performance", "PerformanceMax"), ("video", "Video"), ("display", "Display")):
        if needle in text:
            return label, "meta_campaign_type_inferred"
    return "Generic", "meta_campaign_type_unknown"


def _taxonomy_mapping(raw: dict[str, pd.DataFrame]) -> dict[tuple[str, str], str]:
    taxonomy = raw.get("campaign_taxonomy")
    if taxonomy is None:
        return {}
    return {
        (str(row.source_system), str(row.source_campaign_id)): str(row.campaign_type)
        for row in taxonomy[["source_system", "source_campaign_id", "campaign_type"]].itertuples(index=False)
        if str(row.campaign_type).strip()
    }


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def canonicalize(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    taxonomy = _taxonomy_mapping(raw)
    google = raw["google_ads"].copy()
    google = _to_numeric(google, ["metrics_cost_micros", "metrics_conversions_value", "metrics_clicks", "metrics_impressions", "metrics_conversions", "campaign_budget_amount"])
    g = pd.DataFrame({
        "source_system": "google_ads",
        "source_campaign_id": google["campaign_id"].astype(str),
        "date": pd.to_datetime(google["segments_date"], errors="coerce"),
        "channel": google["campaign_advertising_channel_type"].fillna("UNKNOWN").astype(str),
        "campaign_type": google["campaign_advertising_channel_type"].fillna("UNKNOWN").astype(str),
        "campaign_name": google["campaign_name"].fillna(google["campaign_id"].astype(str)).astype(str),
        "spend": google["metrics_cost_micros"] / 1_000_000.0,
        "revenue": google["metrics_conversions_value"],
        "clicks": google["metrics_clicks"],
        "impressions": google["metrics_impressions"],
        "conversions": google["metrics_conversions"],
        "configured_budget": google["campaign_budget_amount"],
        "quality_flags": "google_cost_micros_normalized",
    })

    bing = raw["microsoft_ads"].copy()
    bing = _to_numeric(bing, ["Revenue", "Spend", "Clicks", "Impressions", "Conversions", "DailyBudget"])
    b = pd.DataFrame({
        "source_system": "microsoft_ads",
        "source_campaign_id": bing["CampaignId"].astype(str),
        "date": pd.to_datetime(bing["TimePeriod"], errors="coerce"),
        "channel": "MICROSOFT_ADS",
        "campaign_type": bing["CampaignType"].fillna("UNKNOWN").astype(str),
        "campaign_name": bing["CampaignName"].fillna(bing["CampaignId"].astype(str)).astype(str),
        "spend": bing["Spend"], "revenue": bing["Revenue"], "clicks": bing["Clicks"],
        "impressions": bing["Impressions"], "conversions": bing["Conversions"],
        "configured_budget": bing["DailyBudget"], "quality_flags": "",
    })

    meta = raw["meta_ads"].copy()
    meta = _to_numeric(meta, ["conversion", "spend", "clicks", "impressions", "daily_budget"])
    inferred = meta["campaign_name"].map(_meta_campaign_type)
    mapped_types = meta["campaign_id"].astype(str).map(lambda campaign_id: taxonomy.get(("meta_ads", campaign_id)))
    meta_campaign_type = mapped_types.fillna(inferred.map(lambda value: value[0]))
    meta_quality_flags = inferred.map(lambda value: value[1])
    meta_quality_flags = meta_quality_flags.mask(mapped_types.notna(), "meta_campaign_type_mapped")
    m = pd.DataFrame({
        "source_system": "meta_ads",
        "source_campaign_id": meta["campaign_id"].astype(str),
        "date": pd.to_datetime(meta["date_start"], errors="coerce"),
        "channel": "META_ADS",
        "campaign_type": meta_campaign_type,
        "campaign_name": meta["campaign_name"].fillna(meta["campaign_id"].astype(str)).astype(str),
        "spend": meta["spend"], "revenue": meta["conversion"], "clicks": meta["clicks"],
        "impressions": meta["impressions"], "conversions": meta["conversion"],
        "configured_budget": meta["daily_budget"],
        "quality_flags": meta_quality_flags.map(lambda value: f"{value};meta_conversion_treated_as_attributed_revenue"),
    })
    output = pd.concat([g, b, m], ignore_index=True)[CANONICAL_COLUMNS]
    output["spend"] = output["spend"].astype(float)
    output["revenue"] = output["revenue"].astype(float)
    return output
