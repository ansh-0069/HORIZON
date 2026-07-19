from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.contracts import (
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
        if path.name == TAXONOMY_FILENAME:
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
    return sources
