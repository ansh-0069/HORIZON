"""Build and atomically promote a reviewed offline submission artifact.

This command belongs to the optional product/release layer.  ``run.sh`` never
imports it and never trains.  It exists to make the otherwise easy-to-miss
release boundary explicit: a candidate is trained and validated first, then
the protected ``pickle/model.pkl`` and its sibling provenance manifest are
replaced together only when an operator supplies ``--confirm-promote``.

The evaluator artifact intentionally contains data-only model parameters; no
training code, external service configuration, or source rows are serialized
into the runtime path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product.evaluation import canonical_fingerprint
from product.training.model_builder import fit_horizon_model
from src.canonicalize import canonicalize
from src.ingest import read_source_files
from src.predict import load_model
from src.validate import validate_canonical


PROTECTED_MODEL = ROOT / "pickle" / "model.pkl"
PROTECTED_MANIFEST = ROOT / "pickle" / "model_manifest.json"
MANIFEST_SCHEMA_VERSION = "horizon-submission-model-manifest-v2"


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write a local release file atomically and clean its temporary on error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _manifest(model: object, artifact: bytes, canonical) -> dict[str, object]:
    """Create local, reproducible provenance for a sealed inference artifact."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "model_version": str(getattr(model, "model_version", "unknown")),
        "artifact_build_python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "minimum_python_major_minor": "3.11",
        "purpose": "Trusted, pre-trained offline inference artifact for the protected hackathon path.",
        "training_data_fingerprint": str(getattr(model, "training_data_fingerprint", "") or canonical_fingerprint(canonical)),
        "feature_schema_fingerprint": str(getattr(model, "feature_schema_fingerprint", "")),
        "canonical_rows": int(len(canonical)),
        "canonical_date_start": str(canonical["date"].min().date()),
        "canonical_date_end": str(canonical["date"].max().date()),
        "source_systems": sorted(str(value) for value in canonical["source_system"].dropna().unique()),
        "trainer": "product.scripts.promote_submission_model",
    }


def _validate_candidate(model_path: Path, manifest_path: Path, expected_sha: str) -> None:
    """Exercise the same protected loader used by the evaluator before promotion."""
    loaded = load_model(model_path)
    if loaded.artifact_sha256 != expected_sha:
        raise RuntimeError("Candidate artifact SHA differs after protected-loader validation")
    if not str(loaded.model_version):
        raise RuntimeError("Candidate model lacks a non-empty model_version")


def build_and_promote(data_dir: Path, model_path: Path, manifest_path: Path) -> dict[str, object]:
    """Train outside the evaluator, validate, then promote a sealed pair.

    Validation occurs in a sibling temporary directory because ``load_model``
    intentionally looks for a manifest with the fixed filename
    ``model_manifest.json`` next to the candidate pickle.
    """
    canonical = canonicalize(read_source_files(data_dir))
    validate_canonical(canonical).raise_if_blocking()
    model = fit_horizon_model(canonical)
    # The runtime SHA is injected by ``load_model`` via ``dataclasses.replace``.
    # Keeping this field empty in the serialized object avoids an impossible
    # self-referential hash while preserving the manifest/model integrity check.
    model.artifact_sha256 = ""
    artifact = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = _manifest(model, artifact, canonical)
    candidate_root = Path(tempfile.mkdtemp(prefix="horizon-promotion-", dir=model_path.parent))
    candidate_model = candidate_root / "model.pkl"
    candidate_manifest = candidate_root / "model_manifest.json"
    try:
        _atomic_write(candidate_model, artifact)
        _atomic_write(candidate_manifest, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        _validate_candidate(candidate_model, candidate_manifest, str(manifest["artifact_sha256"]))
        # Manifest follows the artifact. A process observing the short update
        # window fails closed on SHA mismatch rather than using an unverified
        # model; normal submission execution starts only after this command
        # returns successfully.
        _atomic_write(model_path, artifact)
        _atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        for path in (candidate_model, candidate_manifest):
            path.unlink(missing_ok=True)
        candidate_root.rmdir()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and promote the sealed offline evaluator artifact")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=PROTECTED_MODEL)
    parser.add_argument("--manifest", type=Path, default=PROTECTED_MANIFEST)
    parser.add_argument(
        "--confirm-promote",
        action="store_true",
        help="Required acknowledgement before replacing the protected evaluator artifact.",
    )
    args = parser.parse_args()
    if not args.confirm_promote:
        parser.error("Refusing to replace the protected artifact without --confirm-promote")
    if args.model.resolve() != PROTECTED_MODEL.resolve() or args.manifest.resolve() != PROTECTED_MANIFEST.resolve():
        parser.error("Promotion targets must be the protected pickle/model.pkl and pickle/model_manifest.json pair")
    manifest = build_and_promote(args.data_dir, args.model, args.manifest)
    print(
        "Promoted {version}; sha256={sha}; rows={rows}; date_range={start}..{end}".format(
            version=manifest["model_version"],
            sha=manifest["artifact_sha256"],
            rows=manifest["canonical_rows"],
            start=manifest["canonical_date_start"],
            end=manifest["canonical_date_end"],
        )
    )


if __name__ == "__main__":
    main()
