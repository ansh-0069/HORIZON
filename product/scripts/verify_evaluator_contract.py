"""Verify a generated CSV against an organizer-provided header fixture.

The public guide requires an exact evaluator format but does not publish that
format. This release-only command makes the remaining external gate explicit:
once organizers provide a header fixture, the team can fail fast before
submission rather than discovering a mismatch in the scorer.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Header fixture does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.reader(handle), None)
    if not row:
        raise ValueError(f"Header fixture is empty: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predictions.csv columns with an official evaluator header")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--official-header", type=Path, required=True)
    args = parser.parse_args()
    actual = read_header(args.predictions)
    expected = read_header(args.official_header)
    if actual != expected:
        raise SystemExit(
            "Evaluator CSV header mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            "Update the versioned OutputSchema and its fixture before submission."
        )
    print(f"Evaluator CSV header matches official fixture: {args.official_header}")


if __name__ == "__main__":
    main()
