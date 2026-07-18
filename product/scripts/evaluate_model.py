from __future__ import annotations

import argparse
from pathlib import Path

from src.canonicalize import canonicalize
from src.ingest import read_source_files
from src.validate import validate_canonical
from product.evaluation import write_evaluation_report


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = ROOT / "product"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Horizon rolling-origin backtests")
    parser.add_argument("--data-dir", type=Path, default=PRODUCT_ROOT / "demo_data")
    parser.add_argument("--output", type=Path, default=PRODUCT_ROOT / "models" / "evaluation_report.json")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    canonical = canonicalize(read_source_files(args.data_dir))
    validate_canonical(canonical).raise_if_blocking()
    report = write_evaluation_report(canonical, args.output, args.folds)
    print(f"Wrote evaluation report with {len(report['horizons'])} horizons to {args.output}")


if __name__ == "__main__":
    main()
