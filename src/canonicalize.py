from __future__ import annotations

from typing import Any
import pandas as pd

from src.contracts import CANONICAL_COLUMNS, REVIEW_STATUS_COLUMN


MICROSOFT_CAMPAIGN_TYPE_NORMALIZATION = {
    "audience": "DISPLAY",
    "performance max": "PERFORMANCE_MAX",
    "performancemax": "PERFORMANCE_MAX",
    "search": "SEARCH",
    "shopping": "SHOPPING",
}


def _meta_campaign_type(name: object) -> tuple[str, str]:
    text = str(name or "").lower()
    # Meta exports in the supplied schema do not include a campaign-type field.
    # Prefer reviewed taxonomy when available, then infer only explicit naming
    # conventions. The ordering keeps DPA distinct from generic remarketing.
    for needle, label in (
        ("dpa", "META_REMARKETING_DPA"),
        ("dynamic product", "META_REMARKETING_DPA"),
        ("remarketing", "META_REMARKETING"),
        ("retarget", "META_REMARKETING"),
        ("prospecting", "META_PROSPECTING"),
        ("acquisition", "META_PROSPECTING"),
        ("shopping", "META_SHOPPING"),
        ("video", "META_VIDEO"),
        ("display", "META_DISPLAY"),
    ):
        if needle in text:
            return label, "meta_campaign_type_inferred"
    return "Generic", "meta_campaign_type_unknown"


def _taxonomy_mapping(raw: dict[str, pd.DataFrame]) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    """Return declared taxonomy labels plus the subset carrying a review marker.

    Callers must use only the reviewed subset for model-facing hierarchy.  An
    unreviewed mapping remains visible as data-quality evidence but is never a
    silent feature override.
    """
    taxonomy = raw.get("campaign_taxonomy")
    if taxonomy is None:
        return {}, set()
    mapping = {
        (str(row.source_system), str(row.source_campaign_id)): str(row.campaign_type)
        for row in taxonomy[["source_system", "source_campaign_id", "campaign_type"]].itertuples(index=False)
        if str(row.campaign_type).strip()
    }
    if REVIEW_STATUS_COLUMN not in taxonomy.columns:
        return mapping, set()
    reviewed = {
        (str(row.source_system), str(row.source_campaign_id))
        for row in taxonomy[["source_system", "source_campaign_id", REVIEW_STATUS_COLUMN]].itertuples(index=False)
        if str(row.review_status).strip().lower() == "reviewed"
    }
    return mapping, reviewed


def _semantics_reviewed(raw: dict[str, pd.DataFrame]) -> bool:
    """Return whether every declared source-semantic row has been reviewed."""
    semantics = raw.get("source_semantics")
    if semantics is None or REVIEW_STATUS_COLUMN not in semantics.columns:
        return False
    statuses = semantics[REVIEW_STATUS_COLUMN].astype("string").str.strip().str.lower()
    return bool(not statuses.empty and statuses.eq("reviewed").all())


def _meta_revenue_semantics_reviewed(raw: dict[str, pd.DataFrame]) -> bool:
    """Verify that reviewed metadata specifically covers Meta's revenue proxy."""
    semantics = raw.get("source_semantics")
    if semantics is None or not _semantics_reviewed(raw):
        return False
    required = {"source_system", "revenue_field", REVIEW_STATUS_COLUMN}
    if not required.issubset(semantics.columns):
        return False
    meta = semantics[semantics["source_system"].astype(str).str.strip() == "meta_ads"]
    if len(meta) != 1:
        return False
    row = meta.iloc[0]
    return (
        str(row["revenue_field"]).strip() == "conversion"
        and str(row[REVIEW_STATUS_COLUMN]).strip().lower() == "reviewed"
    )


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _campaign_name_or_identifier(values: pd.Series, identifiers: pd.Series) -> pd.Series:
    """Preserve a supplied campaign name or use the already-validated ID.

    Platform exports frequently encode an absent name as an empty string rather
    than a null.  A human-readable label is useful in output, but it is not a
    model identity; fall back to the opaque campaign identifier instead of
    leaking a blank hierarchy value into grouping and CSV output.
    """
    names = values.astype("string").str.strip()
    fallback = identifiers.astype("string").str.strip()
    return names.mask(names.isna() | names.eq(""), fallback)


def _microsoft_campaign_type(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Map Microsoft export labels onto the canonical cross-channel taxonomy.

    Google exports already use canonical upper-snake labels, while Microsoft
    uses presentation variants such as ``PerformanceMax`` and ``Shopping``.
    Treating those strings as separate model categories creates artificial OOV
    fallbacks at longer horizons. This mapping is deterministic platform
    normalization, not an editable taxonomy override; an unrecognized but
    non-blank Microsoft type remains visible as a normalized upper-snake
    label instead of being silently reassigned to a different tactic.
    """
    raw = values.astype("string").str.strip()
    key = raw.str.casefold().str.replace(r"[\s_-]+", " ", regex=True).str.strip()
    normalized = key.map(MICROSOFT_CAMPAIGN_TYPE_NORMALIZATION)
    fallback = raw.str.upper().str.replace(r"[\s-]+", "_", regex=True)
    canonical = normalized.fillna(fallback)
    changed = raw.notna() & canonical.ne(raw)
    flags = pd.Series("microsoft_campaign_type_canonical", index=values.index, dtype="string")
    flags = flags.mask(changed, "microsoft_campaign_type_normalized")
    return canonical, flags


def canonicalize(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    taxonomy, reviewed_taxonomy_keys = _taxonomy_mapping(raw)
    reviewed_semantics = _semantics_reviewed(raw)
    semantics_flag = "source_semantics_reviewed" if reviewed_semantics else "source_semantics_unreviewed"
    google = raw["google_ads"].copy()
    google = _to_numeric(google, ["metrics_cost_micros", "metrics_conversions_value", "metrics_clicks", "metrics_impressions", "metrics_conversions", "campaign_budget_amount"])
    google_channel = google["campaign_advertising_channel_type"].astype("string").str.strip()
    g = pd.DataFrame({
        "source_system": "google_ads",
        "source_campaign_id": google["campaign_id"].astype(str),
        "date": pd.to_datetime(google["segments_date"], errors="coerce"),
        # Channel/type is a required model and hierarchy feature. Preserve a
        # blank/missing source value so validation can block the corrupt export
        # rather than silently converting it into an arbitrary "UNKNOWN".
        "channel": google_channel,
        "campaign_type": google_channel,
        "campaign_name": _campaign_name_or_identifier(google["campaign_name"], google["campaign_id"]),
        "spend": google["metrics_cost_micros"] / 1_000_000.0,
        "revenue": google["metrics_conversions_value"],
        "clicks": google["metrics_clicks"],
        "impressions": google["metrics_impressions"],
        "conversions": google["metrics_conversions"],
        "configured_budget": google["campaign_budget_amount"],
        "quality_flags": f"google_cost_micros_normalized;{semantics_flag}",
    })

    bing = raw["microsoft_ads"].copy()
    bing = _to_numeric(bing, ["Revenue", "Spend", "Clicks", "Impressions", "Conversions", "DailyBudget"])
    microsoft_campaign_type, microsoft_type_flags = _microsoft_campaign_type(bing["CampaignType"])
    b = pd.DataFrame({
        "source_system": "microsoft_ads",
        "source_campaign_id": bing["CampaignId"].astype(str),
        "date": pd.to_datetime(bing["TimePeriod"], errors="coerce"),
        "channel": "MICROSOFT_ADS",
        "campaign_type": microsoft_campaign_type,
        "campaign_name": _campaign_name_or_identifier(bing["CampaignName"], bing["CampaignId"]),
        "spend": bing["Spend"], "revenue": bing["Revenue"], "clicks": bing["Clicks"],
        "impressions": bing["Impressions"], "conversions": bing["Conversions"],
        "configured_budget": bing["DailyBudget"],
        "quality_flags": microsoft_type_flags.map(lambda value: f"{value};{semantics_flag}"),
    })

    meta = raw["meta_ads"].copy()
    meta = _to_numeric(meta, ["conversion", "spend", "clicks", "impressions", "daily_budget"])
    inferred = meta["campaign_name"].map(_meta_campaign_type)
    meta_keys = meta["campaign_id"].astype(str).map(lambda campaign_id: ("meta_ads", campaign_id))
    reviewed_taxonomy = {
        key: value for key, value in taxonomy.items() if key in reviewed_taxonomy_keys
    }
    unreviewed_taxonomy_keys = set(taxonomy) - set(reviewed_taxonomy)
    mapped_types = meta_keys.map(reviewed_taxonomy)
    meta_campaign_type = mapped_types.fillna(inferred.map(lambda value: value[0]))
    meta_quality_flags = inferred.map(lambda value: value[1])
    # Every name-inferred label remains explicitly unreviewed.  A declared but
    # unreviewed mapping is intentionally ignored rather than allowed to alter
    # campaign-type model features.
    fallback_flags = inferred.map(
        lambda value: f"{value[1]};meta_campaign_type_taxonomy_unreviewed_fallback"
    )
    meta_quality_flags = meta_quality_flags.where(mapped_types.notna(), fallback_flags)
    meta_quality_flags = meta_quality_flags.mask(mapped_types.notna(), "meta_campaign_type_mapped_reviewed")
    unreviewed_mapping = meta_keys.isin(unreviewed_taxonomy_keys)
    meta_quality_flags = meta_quality_flags.mask(
        unreviewed_mapping,
        inferred.map(lambda value: f"{value[1]};meta_campaign_type_mapping_unreviewed_fallback"),
    )
    meta_revenue_flag = (
        "meta_conversion_semantics_reviewed"
        if _meta_revenue_semantics_reviewed(raw)
        else "meta_conversion_treated_as_attributed_revenue"
    )
    m = pd.DataFrame({
        "source_system": "meta_ads",
        "source_campaign_id": meta["campaign_id"].astype(str),
        "date": pd.to_datetime(meta["date_start"], errors="coerce"),
        "channel": "META_ADS",
        "campaign_type": meta_campaign_type,
        "campaign_name": _campaign_name_or_identifier(meta["campaign_name"], meta["campaign_id"]),
        "spend": meta["spend"], "revenue": meta["conversion"], "clicks": meta["clicks"],
        "impressions": meta["impressions"], "conversions": meta["conversion"],
        "configured_budget": meta["daily_budget"],
        "quality_flags": meta_quality_flags.map(lambda value: f"{value};{meta_revenue_flag};{semantics_flag}"),
    })
    output = pd.concat([g, b, m], ignore_index=True)[CANONICAL_COLUMNS]
    output["spend"] = output["spend"].astype(float)
    output["revenue"] = output["revenue"].astype(float)
    return output
