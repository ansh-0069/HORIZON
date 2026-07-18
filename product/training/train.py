from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from product.training.model_builder import fit_horizon_model
from src.canonicalize import canonicalize
from src.ingest import read_source_files
from src.validate import validate_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Horizon forecast artifact outside the submission path")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pickle/model.pkl"))
    args = parser.parse_args()
    canonical = canonicalize(read_source_files(args.data_dir))
    validate_canonical(canonical).raise_if_blocking()
    model = fit_horizon_model(canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(model, handle)
    print(f"Trained {model.model_version}; artifact written to {args.output}")


if __name__ == "__main__":
    main()
