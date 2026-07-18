from __future__ import annotations

from pathlib import Path
import pandas as pd


SIGNATURES = {
    "google_ads": {"campaign_id", "segments_date", "metrics_cost_micros", "metrics_conversions_value"},
    "microsoft_ads": {"CampaignId", "TimePeriod", "Revenue", "Spend"},
    "meta_ads": {"campaign_id", "date_start", "spend", "conversion"},
}


def discover_source_files(data_dir: Path) -> dict[str, Path]:
    if not data_dir.is_dir():
        raise ValueError(f"DATA_DIR does not exist or is not a directory: {data_dir}")
    found: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        try:
            header = set(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            raise ValueError(f"Unable to read CSV header for {path.name}: {exc}") from exc
        matches = [name for name, required in SIGNATURES.items() if required.issubset(header)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous source signature for {path.name}: {matches}")
        if matches:
            source = matches[0]
            if source in found:
                raise ValueError(f"Multiple files match {source}; keep one current source file")
            found[source] = path
    missing = sorted(set(SIGNATURES) - set(found))
    if missing:
        raise ValueError(f"Missing schema-compatible source files: {', '.join(missing)}")
    return found


def read_source_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {source: pd.read_csv(path) for source, path in discover_source_files(data_dir).items()}
