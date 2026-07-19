from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

# Keep the protected entry point fast and deterministic even when it is
# invoked directly (rather than through run.sh).  This must precede pandas,
# which may import NumPy/BLAS during module initialization.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd

from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import media_plan_budget_overrides, read_source_files
from src.model import HorizonModel
from src.output_adapter import write_predictions_csv
from src.validate import validate_canonical


def load_model(path: Path) -> HorizonModel:
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    artifact = path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    manifest_path = path.with_name("model_manifest.json")
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Model manifest is unreadable: {manifest_path}") from exc
        if manifest.get("artifact_sha256") != artifact_sha256:
            raise ValueError("Model artifact SHA-256 does not match model_manifest.json")
    model = pickle.loads(artifact)
    if not isinstance(model, HorizonModel):
        raise TypeError("Model artifact is not a HorizonModel")
    if manifest_path.is_file() and manifest.get("model_version") != model.model_version:
        raise ValueError("Model artifact version does not match model_manifest.json")
    for field in ("training_data_fingerprint", "feature_schema_fingerprint"):
        declared = manifest.get(field) if manifest_path.is_file() else None
        actual = getattr(model, field, "")
        if declared is not None and declared != actual:
            raise ValueError(f"Model artifact {field} does not match model_manifest.json")
    return replace(model, artifact_sha256=artifact_sha256)


def _log(event: str, **fields: object) -> None:
    """Emit deterministic structured diagnostics without contaminating CSV output."""
    print(json.dumps({"event": event, **fields}, sort_keys=True, default=str), file=sys.stderr)


def generate_predictions(data_dir: Path, model_path: Path, output_path: Path) -> int:
    """Run evaluator-safe inference and atomically replace the output CSV."""
    sources = read_source_files(data_dir)
    media_plan = sources.pop("media_plan", None)
    plan_overrides = media_plan_budget_overrides(media_plan) if media_plan is not None else {}
    canonical = canonicalize(sources)
    quality = validate_canonical(canonical)
    quality.raise_if_blocking()
    model = load_model(model_path)
    _log(
        "input_validated",
        rows=len(canonical),
        model_version=model.model_version,
        model_sha256=model.artifact_sha256,
        media_plan_rows=0 if media_plan is None else int(len(media_plan)),
    )
    forecasts = [
        build_forecast(model, canonical, horizon, plan_overrides.get(horizon) or None)
        for horizon in (30, 60, 90)
    ]
    output = write_predictions_csv(pd.concat(forecasts, ignore_index=True), output_path)
    _log("predictions_written", rows=len(output), output_path=output_path, horizons=[30, 60, 90])
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
