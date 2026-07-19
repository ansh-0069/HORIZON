"""Run one non-secret, bounded smoke test for the optional LLM narrator.

This command is deliberately outside the protected submission path. It sends
only a post-forecast evidence packet, never raw channel records or credentials.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from product.app.evidence import OpenAIEvidenceClient, build_evidence_packet, load_evidence_config
from product.app.service import PlannerService


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = ROOT / "product"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the optional grounded evidence narrator without exposing credentials")
    parser.add_argument("--data-dir", type=Path, default=PRODUCT_ROOT / "demo_data")
    parser.add_argument("--model", type=Path, default=ROOT / "pickle" / "model.pkl")
    parser.add_argument("--horizon", type=int, choices=(30, 60, 90), default=60)
    args = parser.parse_args()

    config = load_evidence_config()
    if config is None:
        raise SystemExit("No optional OpenAI evidence configuration is available")
    # This optional verifier uses the same integrity-checked artifact loading,
    # persisted-calibration decision gating, and sealed evidence construction as
    # the local planner. It is not part of the protected evaluator path.
    service = PlannerService(args.data_dir, args.model, allow_live_llm=True)
    forecast = service.forecast({"horizon_days": args.horizon})
    overall = forecast["overall"][0]
    evidence = forecast["evidence"]
    brief = OpenAIEvidenceClient(config).generate(build_evidence_packet(evidence, overall))
    print(
        {
            "status": "passed",
            "forecast_id": str(overall["forecast_id"]),
            "model": config.model,
            "sections": {name: len(brief[name]) for name in ("facts", "assumptions", "recommendations", "limitations")},
        }
    )


if __name__ == "__main__":
    main()
