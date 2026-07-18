from __future__ import annotations

import argparse
from pathlib import Path

from src.canonicalize import canonicalize
from src.evaluate import write_evaluation_report
from src.ingest import read_source_files
from src.validate import validate_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Horizon rolling-origin backtests")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/evaluation_report.json"))
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    canonical = canonicalize(read_source_files(args.data_dir))
    validate_canonical(canonical).raise_if_blocking()
    report = write_evaluation_report(canonical, args.output, args.folds)
    print(f"Wrote evaluation report with {len(report['horizons'])} horizons to {args.output}")


if __name__ == "__main__":
    main()
