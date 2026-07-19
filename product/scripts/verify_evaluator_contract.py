"""Verify a generated CSV against the locked horizon-v1 header, or an official one.

Until organizers publish a scorer header, the repository locks on
``product/tests/fixtures/horizon_v1_header.csv``. Pass ``--official-header`` only
when the organizers provide a replacement fixture.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Support both documented module execution and direct execution from an
# arbitrary working directory.  The verifier is outside the protected runner,
# but release tooling should not rely on an implicit PYTHONPATH.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.output_adapter import FORECAST_COLUMNS

DEFAULT_LOCKED_HEADER = ROOT / "product" / "tests" / "fixtures" / "horizon_v1_header.csv"


def read_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Header fixture does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.reader(handle), None)
    if not row:
        raise ValueError(f"Header fixture is empty: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predictions.csv columns with the locked or official evaluator header")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--official-header",
        type=Path,
        default=DEFAULT_LOCKED_HEADER,
        help="Defaults to the locked horizon-v1 fixture. Override only with an organizer-provided header.",
    )
    args = parser.parse_args()
    expected = read_header(args.official_header)
    if args.official_header.resolve() == DEFAULT_LOCKED_HEADER.resolve() and expected != FORECAST_COLUMNS:
        raise SystemExit("Locked horizon-v1 fixture is out of sync with OutputAdapter.FORECAST_COLUMNS")
    actual = read_header(args.predictions)
    if actual != expected:
        raise SystemExit(
            "Evaluator CSV header mismatch.\n"
            f"Expected ({args.official_header}): {expected}\n"
            f"Actual ({args.predictions}): {actual}\n"
            "Update the versioned OutputSchema and its fixture before submission."
        )
    label = "locked horizon-v1 fixture" if args.official_header.resolve() == DEFAULT_LOCKED_HEADER.resolve() else "official fixture"
    print(f"Evaluator CSV header matches {label}: {args.official_header}")


if __name__ == "__main__":
    main()
