"""Create the small, schema-compatible root ``data/`` sample required by the guide.

This release utility is never imported by the evaluator. It preserves the most
recent 90 days of each committed source export so ``./run.sh`` is runnable from
a fresh clone without shipping the full demonstration history twice.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "product" / "supplied_data"
TARGET_DIR = ROOT / "data"
DATE_COLUMNS = {
    "google_ads_campaign_stats.csv": "segments_date",
    "bing_campaign_stats.csv": "TimePeriod",
    "meta_ads_campaign_stats.csv": "date_start",
}


def main() -> None:
    TARGET_DIR.mkdir(exist_ok=True)
    for filename, date_column in DATE_COLUMNS.items():
        frame = pd.read_csv(SOURCE_DIR / filename)
        frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
        sample = frame.loc[frame[date_column] >= frame[date_column].max() - pd.Timedelta(days=89)].sort_values(date_column)
        sample.to_csv(TARGET_DIR / filename, index=False)
        print(f"Wrote {len(sample)} rows to {TARGET_DIR / filename}")


if __name__ == "__main__":
    main()
