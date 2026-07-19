from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.canonicalize import canonicalize
from src.ingest import read_source_files
from src.predict import load_model
from src.validate import validate_canonical
from product.evaluation import DEFAULT_EVALUATION_FOLDS, canonical_fingerprint, write_evaluation_report


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = ROOT / "product"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Horizon rolling-origin backtests")
    parser.add_argument("--data-dir", type=Path, default=PRODUCT_ROOT / "demo_data")
    parser.add_argument("--output", type=Path, default=PRODUCT_ROOT / "models" / "evaluation_report.json")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "pickle" / "model.pkl",
        help="Pre-trained artifact whose provenance is bound to this report; this command never trains it in place.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_EVALUATION_FOLDS,
        help="Chronological folds; defaults to the documented six-fold product/demo-data protocol.",
    )
    args = parser.parse_args()
    if args.folds < 1:
        parser.error("--folds must be at least 1")
    canonical = canonicalize(read_source_files(args.data_dir))
    validate_canonical(canonical).raise_if_blocking()
    model = load_model(args.model)
    data_fingerprint = canonical_fingerprint(canonical)
    training_fingerprint = str(getattr(model, "training_data_fingerprint", "") or "")
    if training_fingerprint and training_fingerprint != data_fingerprint:
        parser.error(
            "--data-dir canonical fingerprint does not match the selected model's training_data_fingerprint; "
            "bind reports only to the artifact's reviewed canonical data."
        )
    report = write_evaluation_report(canonical, args.output, args.folds)
    report["artifact_provenance"] = {
        "artifact_sha256": model.artifact_sha256,
        "model_version": model.model_version,
        "training_data_fingerprint": training_fingerprint or data_fingerprint,
        "feature_schema_fingerprint": str(getattr(model, "feature_schema_fingerprint", "") or ""),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    provenance = report["data_provenance"]
    print(
        f"Wrote evaluation report with {len(report['horizons'])} horizons to {args.output}; "
        f"canonical fingerprint={provenance['canonical_fingerprint']} rows={provenance['canonical_rows']}"
    )


if __name__ == "__main__":
    main()
