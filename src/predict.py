from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd

from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import read_source_files
from src.model import HorizonModel
from src.output_adapter import to_submission_schema, validate_submission_schema
from src.validate import validate_canonical


def load_model(path: Path) -> HorizonModel:
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, HorizonModel):
        raise TypeError("Model artifact is not a HorizonModel")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Horizon aggregate probabilistic forecasts")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = canonicalize(read_source_files(args.data_dir))
    quality = validate_canonical(canonical)
    quality.raise_if_blocking()
    model = load_model(args.model)
    forecasts = [build_forecast(model, canonical, horizon) for horizon in (30, 60, 90)]
    output = to_submission_schema(pd.concat(forecasts, ignore_index=True))
    validate_submission_schema(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} forecast rows to {args.output}")
    if quality.warnings:
        print("Warnings: " + quality.summary())


if __name__ == "__main__":
    main()
