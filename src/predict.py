from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd

from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import read_source_files
from src.model import HorizonModel
from src.output_adapter import write_predictions_csv
from src.validate import validate_canonical


def load_model(path: Path) -> HorizonModel:
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, HorizonModel):
        raise TypeError("Model artifact is not a HorizonModel")
    return model


def generate_predictions(data_dir: Path, model_path: Path, output_path: Path) -> int:
    """Run evaluator-safe inference and atomically replace the output CSV."""
    canonical = canonicalize(read_source_files(data_dir))
    quality = validate_canonical(canonical)
    quality.raise_if_blocking()
    model = load_model(model_path)
    forecasts = [build_forecast(model, canonical, horizon) for horizon in (30, 60, 90)]
    output = write_predictions_csv(pd.concat(forecasts, ignore_index=True), output_path)
    print(f"Wrote {len(output)} forecast rows to {output_path}")
    if quality.warnings:
        print("Warnings: " + quality.summary())
    return len(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Horizon aggregate probabilistic forecasts")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        generate_predictions(args.data_dir, args.model, args.output)
    except (FileNotFoundError, IsADirectoryError, OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError, pd.errors.ParserError) as exc:
        parser.exit(2, f"ERROR: offline prediction failed: {exc}\n")


if __name__ == "__main__":
    main()
