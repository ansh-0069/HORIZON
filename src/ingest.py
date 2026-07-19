from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from src.contracts import (
    MEDIA_PLAN_FILENAME,
    MEDIA_PLAN_HORIZONS,
    MEDIA_PLAN_REQUIRED_COLUMNS,
    OPTIONAL_DATA_FILENAMES,
    SOURCE_IDENTITY_COLUMNS,
    SOURCE_REQUIRED_COLUMNS,
    SEMANTICS_FILENAME,
    SEMANTICS_REQUIRED_COLUMNS,
    TAXONOMY_FILENAME,
    TAXONOMY_REQUIRED_COLUMNS,
)


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
        plan = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Unable to read optional media plan file {MEDIA_PLAN_FILENAME}: {exc}") from exc
    missing = sorted(MEDIA_PLAN_REQUIRED_COLUMNS - set(plan.columns))
    if missing:
        raise ValueError(f"{MEDIA_PLAN_FILENAME} is missing required columns: {missing}")
    if plan.empty:
        return plan.loc[:, sorted(MEDIA_PLAN_REQUIRED_COLUMNS)].copy()
    plan = plan.copy()
    plan["source_system"] = plan["source_system"].astype(str).str.strip()
    plan["source_campaign_id"] = plan["source_campaign_id"].astype(str).str.strip()
    plan["horizon_days"] = pd.to_numeric(plan["horizon_days"], errors="coerce")
    plan["planned_budget"] = pd.to_numeric(plan["planned_budget"], errors="coerce")
    if plan["source_system"].eq("").any() or plan["source_system"].isna().any():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains blank source_system values")
    if plan["source_campaign_id"].eq("").any() or plan["source_campaign_id"].isna().any():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains blank source_campaign_id values")
    unknown_sources = sorted(set(plan["source_system"]) - set(SOURCE_IDENTITY_COLUMNS))
    if unknown_sources:
        raise ValueError(f"{MEDIA_PLAN_FILENAME} has unknown source_system values: {unknown_sources}")
    if plan["horizon_days"].isna().any():
        raise ValueError(f"{MEDIA_PLAN_FILENAME} contains non-numeric horizon_days values")
    invalid_horizons = sorted({int(value) for value in plan["horizon_days"] if int(value) not in MEDIA_PLAN_HORIZONS})
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
    sources = {source: pd.read_csv(path) for source, path in discover_source_files(data_dir).items()}
    taxonomy_path = data_dir / TAXONOMY_FILENAME
    if taxonomy_path.is_file():
        try:
            taxonomy = pd.read_csv(taxonomy_path)
        except Exception as exc:
            raise ValueError(f"Unable to read optional taxonomy file {TAXONOMY_FILENAME}: {exc}") from exc
        missing = sorted(TAXONOMY_REQUIRED_COLUMNS - set(taxonomy.columns))
        if missing:
            raise ValueError(f"{TAXONOMY_FILENAME} is missing required columns: {missing}")
        duplicate_keys = taxonomy.duplicated(["source_system", "source_campaign_id"], keep=False)
        if duplicate_keys.any():
            raise ValueError(f"{TAXONOMY_FILENAME} has duplicate source/campaign mappings: {int(duplicate_keys.sum())}")
        sources["campaign_taxonomy"] = taxonomy
    semantics_path = data_dir / SEMANTICS_FILENAME
    if semantics_path.is_file():
        try:
            semantics = pd.read_csv(semantics_path)
        except Exception as exc:
            raise ValueError(f"Unable to read optional semantics file {SEMANTICS_FILENAME}: {exc}") from exc
        missing = sorted(SEMANTICS_REQUIRED_COLUMNS - set(semantics.columns))
        if missing:
            raise ValueError(f"{SEMANTICS_FILENAME} is missing required columns: {missing}")
        if semantics["source_system"].duplicated().any():
            raise ValueError(f"{SEMANTICS_FILENAME} has duplicate source_system rows")
        expected_sources = set(SOURCE_IDENTITY_COLUMNS)
        actual_sources = set(semantics["source_system"].astype(str))
        if actual_sources != expected_sources:
            raise ValueError(f"{SEMANTICS_FILENAME} must declare exactly {sorted(expected_sources)}")
        for column in ("currency", "timezone", "attribution_method", "revenue_field"):
            if semantics[column].isna().any() or semantics[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{SEMANTICS_FILENAME} contains blank {column} values")
        if semantics["currency"].astype(str).str.upper().nunique() != 1:
            raise ValueError(f"{SEMANTICS_FILENAME} contains multiple currencies; normalize before forecasting")
        if semantics["timezone"].astype(str).nunique() != 1:
            raise ValueError(f"{SEMANTICS_FILENAME} contains multiple timezones; align daily boundaries before forecasting")
        sources["source_semantics"] = semantics
    media_plan_path = data_dir / MEDIA_PLAN_FILENAME
    if media_plan_path.is_file():
        sources["media_plan"] = _read_media_plan(media_plan_path)
    return sources
