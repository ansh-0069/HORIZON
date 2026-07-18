from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from product.app.evidence import EvidenceGenerationError, build_evidence_packet, validate_brief
from product.app.service import PlannerService
from product.decisioning.optimizer import recommend_allocation
from product.decisioning.scenario import simulate_budget_plan
from product.training.model_builder import fit_horizon_model

from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import read_source_files
from src.model import HorizonModel
from src.output_adapter import FORECAST_COLUMNS, OutputAdapter, OutputField, OutputSchema, SchemaAdaptationError, SchemaValidationError, to_submission_schema, validate_submission_schema, write_predictions_csv
from src.predict import generate_predictions
from src.validate import validate_canonical


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = ROOT / "product"


class HorizonPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = canonicalize(read_source_files(PRODUCT_ROOT / "demo_data"))
        cls.model = fit_horizon_model(cls.canonical)

    def test_canonicalizes_google_cost_micros(self) -> None:
        google = self.canonical[self.canonical["source_system"] == "google_ads"]
        self.assertGreater(len(google), 0)
        self.assertGreater(google["spend"].sum(), 1.0)
        self.assertLess(google["spend"].sum(), 3_000_000.0)

    def test_quality_has_no_blockers(self) -> None:
        report = validate_canonical(self.canonical)
        self.assertEqual(report.blockers, [])
        self.assertGreaterEqual(len(report.warnings), 1)

    def test_forecast_reconciles_and_has_ordered_quantiles(self) -> None:
        forecast = build_forecast(self.model, self.canonical, 60)
        overall = forecast[forecast["level"] == "overall"].iloc[0]
        leaves = forecast[forecast["level"] == "campaign"]
        self.assertAlmostEqual(float(overall["predicted_revenue_p50"]), float(leaves["predicted_revenue_p50"].sum()), places=6)
        self.assertTrue((forecast["predicted_revenue_p10"] <= forecast["predicted_revenue_p50"]).all())
        self.assertTrue((forecast["predicted_revenue_p50"] <= forecast["predicted_revenue_p90"]).all())
        self.assertTrue(((forecast["probability_roas_above_target"] >= 0) & (forecast["probability_roas_above_target"] <= 1)).all())

    def test_scenario_accepts_override_and_rejects_negative_budget(self) -> None:
        campaign_id = str(self.canonical.iloc[0]["source_campaign_id"])
        simulated = simulate_budget_plan(self.model, self.canonical, 30, {campaign_id: 1000.0}, target_roas=3.5)
        self.assertEqual(set(simulated["horizon_days"]), {30})
        with self.assertRaises(ValueError):
            simulate_budget_plan(self.model, self.canonical, 30, {campaign_id: -1.0})

    def test_pickle_and_submission_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pkl"
            with path.open("wb") as handle:
                pickle.dump(self.model, handle)
            with path.open("rb") as handle:
                restored = pickle.load(handle)
            output = to_submission_schema(build_forecast(restored, self.canonical, 30))
        self.assertEqual(list(output.columns), FORECAST_COLUMNS)
        self.assertFalse(output.empty)
        self.assertFalse(output.isna().any().any())

    def test_submission_contract_validator_rejects_schema_violation(self) -> None:
        output = to_submission_schema(pd.concat([build_forecast(self.model, self.canonical, horizon) for horizon in (30, 60, 90)]))
        validate_submission_schema(output)
        invalid = output.drop(columns=["risk_score"])
        with self.assertRaises(SchemaValidationError):
            validate_submission_schema(invalid)

    def test_output_adapter_uses_optional_defaults_and_required_source_errors(self) -> None:
        forecast = build_forecast(self.model, self.canonical, 30)
        output = to_submission_schema(forecast.drop(columns=["quality_flags"]))
        self.assertEqual(set(output["quality_flags"]), {"none"})
        with self.assertRaises(SchemaAdaptationError):
            to_submission_schema(forecast.drop(columns=["forecast_id"]))

    def test_output_adapter_supports_a_versioned_compatibility_schema(self) -> None:
        schema = OutputSchema(
            version="evaluator-preview-v2",
            fields=(
                OutputField("submission_id", ("forecast_id", "legacy_id"), "string"),
                OutputField("status", (), "string", default="ready"),
            ),
            sort_keys=("submission_id",),
        )
        adapter = OutputAdapter({schema.version: schema}, default_schema_version=schema.version)
        output = adapter.adapt(pd.DataFrame({"legacy_id": ["b", "a"]}))
        self.assertEqual(list(output.columns), ["submission_id", "status"])
        self.assertEqual(output["submission_id"].tolist(), ["a", "b"])
        self.assertEqual(output["status"].tolist(), ["ready", "ready"])

    def test_output_adapter_preserves_existing_csv_when_serialization_fails(self) -> None:
        forecast = build_forecast(self.model, self.canonical, 30)
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "predictions.csv"
            output_path.write_text("previous,output\nunchanged,value\n", encoding="utf-8")
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("simulated disk failure")):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    write_predictions_csv(forecast, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "previous,output\nunchanged,value\n")
            self.assertEqual(list(output_path.parent.glob(".predictions-*.csv")), [])

    def test_prediction_generation_is_atomic_and_missing_model_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "predictions.csv"
            generate_predictions(PRODUCT_ROOT / "demo_data", ROOT / "pickle" / "model.pkl", output)
            self.assertTrue(output.is_file())
            with self.assertRaises(FileNotFoundError):
                generate_predictions(PRODUCT_ROOT / "demo_data", Path(temporary) / "missing.pkl", output)

    def test_optimizer_returns_feasible_reconciled_plan(self) -> None:
        baseline = build_forecast(self.model, self.canonical, 30)
        total = float(baseline[baseline["level"] == "overall"].iloc[0]["planned_budget"])
        result = recommend_allocation(self.model, self.canonical, 30, total, target_roas=3.5, increments=80)
        overall = result.forecast[result.forecast["level"] == "overall"].iloc[0]
        self.assertEqual(result.status, "feasible")
        self.assertAlmostEqual(sum(result.campaign_budgets.values()), total, places=2)
        self.assertGreater(float(overall["predicted_revenue_p50"]), 0.0)

    def test_evidence_brief_is_cited_and_cannot_make_causal_or_numeric_claims(self) -> None:
        forecast = build_forecast(self.model, self.canonical, 30)
        overall = forecast[forecast["level"] == "overall"].iloc[0].to_dict()
        deterministic = {
            "decision": "approve",
            "target_roas": 4.0,
            "drivers": [{"channel": "SEARCH", "expected_revenue": 100.0, "expected_roas": 4.2}],
            "risks": ["No campaign-level extrapolation flags were triggered."],
        }
        packet = build_evidence_packet(deterministic, overall)
        valid = {
            "decision": "approve", "causal_status": "observational_association", "headline": "The scenario merits controlled validation.",
            "facts": [{"text": "The outlook is conditional on supplied attribution.", "evidence_ids": ["forecast_boundary"]}],
            "assumptions": [{"text": "The guardrail posture follows the approved forecast signal.", "evidence_ids": ["overall_guardrail"]}],
            "recommendations": [{"text": "Use a bounded split test before a material allocation change.", "evidence_ids": ["forecast_boundary", "overall_guardrail"]}],
            "limitations": [{"text": "The range represents uncertainty rather than a promise.", "evidence_ids": ["forecast_range"]}],
        }
        self.assertEqual(validate_brief(valid, packet)["headline"], valid["headline"])
        invalid = dict(valid)
        invalid["headline"] = "The change causes ten percent lift."
        with self.assertRaises(EvidenceGenerationError):
            validate_brief(invalid, packet)

    def test_evidence_endpoint_degrades_to_deterministic_brief(self) -> None:
        service = PlannerService.__new__(PlannerService)
        service.forecast = lambda payload: {
            "overall": [{
                "forecast_id": "test-forecast", "probability_roas_above_target": 0.5, "risk_score": 40.0,
                "predicted_revenue_p10": 10.0, "predicted_revenue_p50": 20.0, "predicted_revenue_p90": 30.0,
            }],
            "evidence": {
                "decision": "revise_or_test", "target_roas": 4.0,
                "drivers": [{"channel": "SEARCH", "expected_revenue": 20.0, "expected_roas": 4.0}],
                "risks": ["Forecast is conditional."],
            },
        }
        with patch("product.app.service.load_evidence_config", return_value=None):
            response = service.explain({"scenario": {"horizon_days": 30}})
        self.assertEqual(response["mode"], "deterministic_fallback")
        self.assertEqual(response["forecast_id"], "test-forecast")


if __name__ == "__main__":
    unittest.main()
