from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

from product.evaluation import canonical_fingerprint
from product.training.model_builder import fit_horizon_model
from src.canonicalize import canonicalize
from src.ingest import read_source_files
from src.validate import validate_canonical


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_MODEL_PATH = (ROOT / "pickle" / "model.pkl").resolve()
DEFAULT_PRODUCT_MODEL_PATH = ROOT / "product" / "models" / "horizon_model.pkl"


def safe_product_output_path(output: Path) -> Path:
    """Resolve a training output and reject the frozen evaluator artifact.

    The committed ``pickle/model.pkl`` is protected by a matching manifest and
    is the only artifact the evaluator invokes.  Training from the optional
    product layer must never mutate it: doing so can leave the protected
    manifest stale and turn an otherwise valid checkout into an offline
    inference failure.  Promotion is a separate release operation with its
    own model review and manifest update.
    """
    resolved = output.expanduser().resolve()
    if resolved == PROTECTED_MODEL_PATH:
        raise ValueError(
            "Refusing to overwrite protected evaluator artifact pickle/model.pkl. "
            "Write a product artifact under product/models/ and promote it through "
            "the reviewed release process."
        )
    return resolved


def product_manifest_path(artifact_path: Path) -> Path:
    """Return the separate provenance manifest for an optional product artifact."""
    return artifact_path.with_suffix(".manifest.json")


def _atomic_write(path: Path, payload: bytes) -> None:
    """Atomically replace a local training artifact without a partial pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_product_artifact(model: object, canonical, output: Path) -> tuple[Path, Path]:
    """Persist a product-only artifact plus deterministic training provenance."""
    artifact_path = safe_product_output_path(output)
    artifact = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write(artifact_path, artifact)

    fingerprint = canonical_fingerprint(canonical)
    dates = canonical["date"]
    manifest = {
        "schema_version": "horizon-product-model-manifest-v1",
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "model_version": str(getattr(model, "model_version", "unknown")),
        "training_data_fingerprint": str(getattr(model, "training_data_fingerprint", fingerprint) or fingerprint),
        "feature_schema_fingerprint": str(getattr(model, "feature_schema_fingerprint", "")),
        "canonical_rows": int(len(canonical)),
        "canonical_date_start": str(dates.min().date()),
        "canonical_date_end": str(dates.max().date()),
        "source_systems": sorted(str(value) for value in canonical["source_system"].dropna().unique()),
        "trainer": "product.training.train",
    }
    manifest_path = product_manifest_path(artifact_path)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return artifact_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Horizon forecast artifact outside the submission path")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PRODUCT_MODEL_PATH,
        help="Product-only artifact path (never the protected pickle/model.pkl).",
    )
    args = parser.parse_args()
    try:
        output = safe_product_output_path(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    canonical = canonicalize(read_source_files(args.data_dir))
    validate_canonical(canonical).raise_if_blocking()
    model = fit_horizon_model(canonical)
    artifact_path, manifest_path = write_product_artifact(model, canonical, output)
    print(
        f"Trained {model.model_version}; product artifact written to {artifact_path}; "
        f"provenance manifest written to {manifest_path}"
    )


if __name__ == "__main__":
    main()
