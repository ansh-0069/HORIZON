from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from src.contracts import (
    CANONICAL_REVENUE_FIELDS,
    MEDIA_PLAN_FILENAME,
    MEDIA_PLAN_HORIZONS,
    MEDIA_PLAN_REQUIRED_COLUMNS,
    OPTIONAL_DATA_FILENAMES,
    SOURCE_IDENTITY_COLUMNS,
    SOURCE_CAMPAIGN_ID_COLUMNS,
    SOURCE_REQUIRED_COLUMNS,
    REVIEW_STATUS_COLUMN,
    REVIEW_STATUSES,
    SEMANTICS_FILENAME,
    SEMANTICS_REQUIRED_COLUMNS,
    TAXONOMY_FILENAME,
    TAXONOMY_REQUIRED_COLUMNS,
    TAXONOMY_SUPPORTED_SOURCES,
)


def _normalized_text_column(frame: pd.DataFrame, column: str, filename: str) -> pd.Series:
    """Return a stripped string column and reject missing metadata values."""
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or values.fillna("").eq("").any():
        raise ValueError(f"{filename} contains blank {column} values")
    return values


def _validate_optional_review_status(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Normalize an optional review marker without making it evaluator-required.

    A missing marker is intentionally supported for backwards compatibility,
    but downstream quality flags will keep that metadata in the unreviewed
    state.  A present marker must be one of the small, auditable vocabulary.
    """
    if REVIEW_STATUS_COLUMN not in frame.columns:
        return frame
    frame = frame.copy()
    frame[REVIEW_STATUS_COLUMN] = _normalized_text_column(frame, REVIEW_STATUS_COLUMN, filename).str.lower()
    invalid = sorted(set(frame[REVIEW_STATUS_COLUMN]) - set(REVIEW_STATUSES))
    if invalid:
        raise ValueError(
            f"{filename} {REVIEW_STATUS_COLUMN} must be one of {sorted(REVIEW_STATUSES)}; got {invalid}"
        )
    return frame


def _source_campaign_ids(sources: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    """Return source-qualified campaign IDs for validating reviewed mappings."""
    return {
        source: set(frame[SOURCE_CAMPAIGN_ID_COLUMNS[source]].astype("string").str.strip().dropna())
        for source, frame in sources.items()
        if source in SOURCE_CAMPAIGN_ID_COLUMNS
    }


def discover_source_files(data_dir: Path) -> dict[str, Path]:
    if not data_dir.is_dir():
        raise ValueError(f"DATA_DIR does not exist or is not a directory: {data_dir}")
    found: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        if path.name in OPTIONAL_DATA_FILENAMES:
            continue
        try:
            header = set(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            raise ValueError(f"Unable to read CSV header for {path.name}: {exc}") from exc
        matches = [name for name, identity in SOURCE_IDENTITY_COLUMNS.items() if identity.issubset(header)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous source signature for {path.name}: {matches}")
        if matches:
            source = matches[0]
            missing_columns = sorted(SOURCE_REQUIRED_COLUMNS[source] - header)
            if missing_columns:
                raise ValueError(f"{path.name} matches {source} but is missing required columns: {missing_columns}")
            if source in found:
                raise ValueError(f"Multiple files match {source}; keep one current source file")
            found[source] = path
    missing = sorted(set(SOURCE_IDENTITY_COLUMNS) - set(found))
    if missing:
        raise ValueError(f"Missing schema-compatible source files: {', '.join(missing)}")
    return found


def _read_platform_export(path: Path, source: str) -> pd.DataFrame:
    """Read a platform export without coercing its opaque campaign identifier.

    Pandas otherwise infers numeric identifiers, which drops leading zeros and
    turns missing values into the literal string ``"nan"`` later in
    canonicalization.  Normalize and reject invalid IDs at the trust boundary
    before any taxonomy or media-plan mapping can use them.
    """
    identifier = SOURCE_CAMPAIGN_ID_COLUMNS[source]
    try:
        frame = pd.read_csv(path, dtype={identifier: "string"})
    except Exception as exc:
        raise ValueError(f"Unable to read source export {path.name}: {exc}") from exc
    frame[identifier] = _normalized_text_column(frame, identifier, path.name)
    return frame


def media_plan_budget_overrides(media_plan: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Convert a validated media plan into horizon-scoped campaign budget overrides."""
    overrides: dict[int, dict[str, float]] = {horizon: {} for horizon in sorted(MEDIA_PLAN_HORIZONS)}
    for row in media_plan.itertuples(index=False):
        horizon = int(row.horizon_days)
        # Match HorizonModel.campaign_key without importing the model layer.
        key = f"{row.source_system}:{row.source_campaign_id}"
        overrides[horizon][key] = float(row.planned_budget)
    return overrides


def _read_media_plan(path: Path) -> pd.DataFrame:
    try:
        plan = pd.read_csv(path, dtype={"source_system": "string", "source_campaign_id": "string"})
    except Exception as exc:
        raise ValueError(f"Unable to read optional media plan file {MEDIA_PLAN_FILENAME}: {exc}") from exc
    missing = sorted(MEDIA_PLAN_REQUIRED_COLUMNS - set(plan.columns))
    if missing:
        raise ValueError(f"{MEDIA_PLAN_FILENAME} is missing required columns: {missing}")
    if plan.empty:
        return plan.loc[:, sorted(MEDIA_PLAN_REQUIRED_COLUMNS)].copy()
    plan = plan.copy()
    plan["source_system"] = _normalized_text_column(plan, "source_system", MEDIA_PLAN_FILENAME)
    plan["source_campaign_id"] = _normalized_text_column(plan, "source_campaign_id", MEDIA_PLAN_FILENAME)
    plan["horizon_days"] = pd.to_numeric(plan["horizon_days"], errors="coerce")
    plan["planned_budget"] = pd.to_numeric(plan["planned_budget"], errors="coerce")
    unknown_sources = sorted(set(plan["source_system"]) - set(SOURCE_IDENTITY_COLUMNS))
    if unknown_sources:
        raise ValueError(f"{MEDIA_PLAN_FILENAME} has unknown source_system values: {unknown_sources}")
    if plan["horizon_days"].isna().any():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains non-numeric horizon_days values")
    # ``int(30.5)`` silently turns an invalid request into the 30-day plan and
    # ``int(float('inf'))`` raises an implementation exception.  Validate the
    # numeric domain before conversion so optional scenario input has the same
    # precise, fail-closed contract as the JSON product API.
    if not plan["horizon_days"].map(lambda value: math.isfinite(float(value))).all():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains non-finite horizon_days values")
    if not plan["horizon_days"].map(lambda value: float(value).is_integer()).all():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} horizon_days must be exact integer values")
    invalid_horizons = sorted(
        {int(value) for value in plan["horizon_days"] if int(value) not in MEDIA_PLAN_HORIZONS}
    )
    if invalid_horizons:
        raise ValueError(f"{MEDIA_PLAN_FILENAME} horizon_days must be one of {sorted(MEDIA_PLAN_HORIZONS)}; got {invalid_horizons}")
    plan["horizon_days"] = plan["horizon_days"].astype(int)
    if plan["planned_budget"].isna().any():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains non-numeric planned_budget values")
    if (plan["planned_budget"] < 0).any():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains negative planned_budget values")
    if not plan["planned_budget"].map(lambda value: math.isfinite(float(value))).all():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains non-finite planned_budget values")
    duplicate_keys = plan.duplicated(["source_system", "source_campaign_id", "horizon_days"], keep=False)
    if duplicate_keys.any():
        raise ValueError(
            f"{MEDIA_PLAN_FILENAME} has duplicate source/campaign/horizon rows: {int(duplicate_keys.sum())}"
        )
    return plan.loc[:, ["source_system", "source_campaign_id", "horizon_days", "planned_budget"]]


def read_source_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    sources = {
        source: _read_platform_export(path, source)
        for source, path in discover_source_files(data_dir).items()
    }
    taxonomy_path = data_dir / TAXONOMY_FILENAME
    if taxonomy_path.is_file():
        try:
            taxonomy = pd.read_csv(
                taxonomy_path,
                dtype={
                    "source_system": "string",
                    "source_campaign_id": "string",
                    "campaign_type": "string",
                    REVIEW_STATUS_COLUMN: "string",
                },
            )
        except Exception as exc:
            raise ValueError(f"Unable to read optional taxonomy file {TAXONOMY_FILENAME}: {exc}") from exc
        missing = sorted(TAXONOMY_REQUIRED_COLUMNS - set(taxonomy.columns))
        if missing:
            raise ValueError(f"{TAXONOMY_FILENAME} is missing required columns: {missing}")
        taxonomy = taxonomy.copy()
        taxonomy["source_system"] = _normalized_text_column(taxonomy, "source_system", TAXONOMY_FILENAME)
        taxonomy["source_campaign_id"] = _normalized_text_column(
            taxonomy, "source_campaign_id", TAXONOMY_FILENAME
        )
        taxonomy["campaign_type"] = _normalized_text_column(taxonomy, "campaign_type", TAXONOMY_FILENAME)
        taxonomy = _validate_optional_review_status(taxonomy, TAXONOMY_FILENAME)
        unknown_sources = sorted(set(taxonomy["source_system"]) - set(SOURCE_IDENTITY_COLUMNS))
        if unknown_sources:
            raise ValueError(f"{TAXONOMY_FILENAME} has unknown source_system values: {unknown_sources}")
        unsupported_sources = sorted(set(taxonomy["source_system"]) - set(TAXONOMY_SUPPORTED_SOURCES))
        if unsupported_sources:
            raise ValueError(
                f"{TAXONOMY_FILENAME} supports reviewed campaign-type overrides only for "
                f"{sorted(TAXONOMY_SUPPORTED_SOURCES)}; got {unsupported_sources}"
            )
        duplicate_keys = taxonomy.duplicated(["source_system", "source_campaign_id"], keep=False)
        if duplicate_keys.any():
            raise ValueError(f"{TAXONOMY_FILENAME} has duplicate source/campaign mappings: {int(duplicate_keys.sum())}")
        known_campaigns = _source_campaign_ids(sources)
        unknown_campaign_rows = taxonomy.apply(
            lambda row: row["source_campaign_id"] not in known_campaigns[str(row["source_system"])], axis=1
        )
        if unknown_campaign_rows.any():
            examples = taxonomy.loc[unknown_campaign_rows, ["source_system", "source_campaign_id"]].head(5)
            formatted = [f"{row.source_system}:{row.source_campaign_id}" for row in examples.itertuples(index=False)]
            raise ValueError(
                f"{TAXONOMY_FILENAME} maps campaign IDs not present in this upload: {formatted}"
            )
        sources["campaign_taxonomy"] = taxonomy
    semantics_path = data_dir / SEMANTICS_FILENAME
    if semantics_path.is_file():
        try:
            semantics = pd.read_csv(
                semantics_path,
                dtype={
                    "source_system": "string",
                    "currency": "string",
                    "timezone": "string",
                    "attribution_method": "string",
                    "revenue_field": "string",
                    REVIEW_STATUS_COLUMN: "string",
                },
            )
        except Exception as exc:
            raise ValueError(f"Unable to read optional semantics file {SEMANTICS_FILENAME}: {exc}") from exc
        missing = sorted(SEMANTICS_REQUIRED_COLUMNS - set(semantics.columns))
        if missing:
            raise ValueError(f"{SEMANTICS_FILENAME} is missing required columns: {missing}")
        semantics = semantics.copy()
        for column in ("source_system", "currency", "timezone", "attribution_method", "revenue_field"):
            semantics[column] = _normalized_text_column(semantics, column, SEMANTICS_FILENAME)
        semantics = _validate_optional_review_status(semantics, SEMANTICS_FILENAME)
        if semantics["source_system"].duplicated().any():
            raise ValueError(f"{SEMANTICS_FILENAME} has duplicate source_system rows")
        expected_sources = set(SOURCE_IDENTITY_COLUMNS)
        actual_sources = set(semantics["source_system"])
        if actual_sources != expected_sources:
            raise ValueError(f"{SEMANTICS_FILENAME} must declare exactly {sorted(expected_sources)}")
        semantics["currency"] = semantics["currency"].str.upper()
        if semantics["currency"].nunique() != 1:
            raise ValueError(f"{SEMANTICS_FILENAME} contains multiple currencies; normalize before forecasting")
        if semantics["timezone"].nunique() != 1:
            raise ValueError(f"{SEMANTICS_FILENAME} contains multiple timezones; align daily boundaries before forecasting")
        revenue_field_by_source = semantics.set_index("source_system")["revenue_field"].to_dict()
        invalid_revenue_fields = [
            f"{source}:{revenue_field}"
            for source, revenue_field in revenue_field_by_source.items()
            if revenue_field not in sources[source].columns
        ]
        if invalid_revenue_fields:
            raise ValueError(
                f"{SEMANTICS_FILENAME} revenue_field values are not present in their source exports: "
                f"{invalid_revenue_fields}"
            )
        unsupported_revenue_mappings = [
            f"{source}:{revenue_field} (expected {CANONICAL_REVENUE_FIELDS[source]})"
            for source, revenue_field in revenue_field_by_source.items()
            if revenue_field != CANONICAL_REVENUE_FIELDS[source]
        ]
        if unsupported_revenue_mappings:
            raise ValueError(
                f"{SEMANTICS_FILENAME} revenue_field values must match the protected canonical mappings: "
                f"{unsupported_revenue_mappings}"
            )
        sources["source_semantics"] = semantics
    media_plan_path = data_dir / MEDIA_PLAN_FILENAME
    if media_plan_path.is_file():
        media_plan = _read_media_plan(media_plan_path)
        known_campaigns = _source_campaign_ids(sources)
        unknown_plan_rows = media_plan.apply(
            lambda row: row["source_campaign_id"] not in known_campaigns[str(row["source_system"])],
            axis=1,
        )
        if unknown_plan_rows.any():
            examples = media_plan.loc[
                unknown_plan_rows, ["source_system", "source_campaign_id"]
            ].head(5)
            formatted = [f"{row.source_system}:{row.source_campaign_id}" for row in examples.itertuples(index=False)]
            raise ValueError(
                f"{MEDIA_PLAN_FILENAME} maps campaign IDs not present in this upload: {formatted}"
            )
        sources["media_plan"] = media_plan
    return sources
