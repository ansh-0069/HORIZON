from __future__ import annotations

import csv
import json
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from product.app.evidence import EvidenceClientConfig, EvidenceGenerationError, OpenAIEvidenceClient, build_evidence_packet, validate_brief
from product.app.server import make_handler, validate_live_llm_host
from product.app.service import PlannerService
from product.decisioning.optimizer import recommend_allocation
from product.decisioning.scenario import simulate_budget_plan
from product.evaluation import canonical_fingerprint
from product.training.model_builder import fit_horizon_model

from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import read_source_files
from src.model import HorizonModel
from src.output_adapter import FORECAST_COLUMNS, OutputAdapter, OutputField, OutputSchema, SchemaAdaptationError, SchemaValidationError, to_submission_schema, validate_submission_schema, write_predictions_csv
from src.predict import generate_predictions, load_model
from src.validate import validate_canonical


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = ROOT / "product"


class HorizonPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = canonicalize(read_source_files(PRODUCT_ROOT / "demo_data"))
        cls.model = fit_horizon_model(cls.canonical)

    def _planner_service(self, *, trust_report: dict | None = None) -> PlannerService:
        """Build an in-memory planner without writing a local demo ledger."""
        service = PlannerService.__new__(PlannerService)
        service.canonical = self.canonical
        service.quality = validate_canonical(self.canonical)
        service.model = self.model
        service.allow_live_llm = False
        service.ledger = Mock()
        service._trust_report = trust_report
        return service

    @staticmethod
    def _sufficient_trust_report(horizon_days: int = 60) -> dict:
        return {
            "status": "available",
            "horizons": [
                {
                    "horizon_days": horizon_days,
                    "folds": 6,
                    "median_calibration_samples": 100,
                    "revenue_interval_coverage": 0.80,
                    "roas_interval_coverage": 0.70,
                    "nominal_interval_coverage": 0.80,
                    "revenue_wape": 0.40,
                    "coverage_by_hierarchy": {
                        "campaign": {
                            "revenue_interval_coverage": 0.70,
                            "roas_interval_coverage": 0.60,
                            "missing_forecast_observations": 2,
                            "revenue_coverage_observations": 100,
                        }
                    },
                    "roas_target_probability_reliability": {
                        "status": "evaluated",
                        "observations": 30,
                        "brier_score": 0.12,
                        "expected_calibration_error": 0.06,
                    },
                }
            ],
        }

    def test_canonicalizes_google_cost_micros(self) -> None:
        google = self.canonical[self.canonical["source_system"] == "google_ads"]
        self.assertGreater(len(google), 0)
        self.assertGreater(google["spend"].sum(), 1.0)
        self.assertLess(google["spend"].sum(), 3_000_000.0)

    def test_meta_campaign_names_receive_operational_taxonomy(self) -> None:
        meta = self.canonical[self.canonical["source_system"] == "meta_ads"]
        self.assertFalse(meta.empty)
        self.assertIn("META_PROSPECTING", set(meta["campaign_type"]))
        self.assertIn("META_REMARKETING", set(meta["campaign_type"]))
        self.assertIn("META_REMARKETING_DPA", set(meta["campaign_type"]))
        explicit = meta[meta["campaign_name"].str.contains("prospecting|remarketing|dpa", case=False, na=False)]
        self.assertNotIn("Generic", set(explicit["campaign_type"]))

    def test_quality_has_no_blockers(self) -> None:
        report = validate_canonical(self.canonical)
        self.assertEqual(report.blockers, [])
        self.assertGreaterEqual(len(report.warnings), 1)

    def test_submission_layout_has_root_data_defaults_and_executable_git_mode(self) -> None:
        for filename in ("google_ads_campaign_stats.csv", "bing_campaign_stats.csv", "meta_ads_campaign_stats.csv"):
            self.assertTrue((ROOT / "data" / filename).is_file(), filename)
        runner = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn('DATA_DIR="${1:-./data}"', runner)
        self.assertIn('MODEL_PATH="${2:-./pickle/model.pkl}"', runner)
        self.assertIn('OUTPUT_PATH="${3:-./output/predictions.csv}"', runner)
        staged = subprocess.run(["git", "ls-files", "--stage", "run.sh"], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertTrue(staged.stdout.startswith("100755"), staged.stdout)

    def test_forecast_reconciles_and_has_ordered_quantiles(self) -> None:
        forecast = build_forecast(self.model, self.canonical, 60)
        overall = forecast[forecast["level"] == "overall"].iloc[0]
        leaves = forecast[forecast["level"] == "campaign"]
        # The serving family is selected before artifact promotion.  A direct
        # winner carries its calibrated residual-dependence profile; a
        # statistical winner must explicitly use the safer independent-rank
        # rollup rather than inheriting a direct candidate's copula profile.
        self.assertAlmostEqual(float(overall["planned_budget"]), float(leaves["planned_budget"].sum()), places=6)
        self.assertAlmostEqual(float(overall["predicted_revenue_p50"]), float(leaves["predicted_revenue_p50"].sum()), places=6)
        selected_family = self.model.selected_model_family(60)
        flags = str(overall["quality_flags"])
        if selected_family == "direct_ensemble":
            self.assertIn("historical_residual_factor_copula_rollup", flags)
            self.assertIn("portfolio_oof_residual_calibration", flags)
        else:
            self.assertEqual(selected_family, "statistical_fallback")
            self.assertIn("independent_rank_rollup_fallback", flags)
            self.assertNotIn("historical_residual_factor_copula_rollup", flags)
            self.assertNotIn("portfolio_oof_residual_calibration", flags)
        self.assertTrue((forecast["predicted_revenue_p10"] <= forecast["predicted_revenue_p50"]).all())
        self.assertTrue((forecast["predicted_revenue_p50"] <= forecast["predicted_revenue_p90"]).all())
        self.assertTrue((forecast["predicted_spend_p10"] <= forecast["predicted_spend_p50"]).all())
        self.assertTrue((forecast["predicted_spend_p50"] <= forecast["predicted_spend_p90"]).all())
        self.assertTrue(((forecast["probability_roas_above_target"] >= 0) & (forecast["probability_roas_above_target"] <= 1)).all())

    def test_only_a_selected_direct_model_carries_a_historical_residual_dependence_profile(self) -> None:
        profile = self.model.dependence_for_horizon(60)
        if self.model.selected_model_family(60) == "direct_ensemble":
            self.assertEqual(profile["method"], "hierarchical_residual_factor_copula_v1")
            self.assertGreater(int(profile["sample_count"]), 0)
            self.assertGreater(int(profile["block_count"]), 0)
            weights = profile["factor_weights"]
            self.assertAlmostEqual(sum(float(weights[name]) for name in ("global", "channel", "campaign_type", "idiosyncratic")), 1.0, places=6)
        else:
            self.assertEqual(self.model.selected_model_family(60), "statistical_fallback")
            self.assertEqual(profile, {})

    def test_spend_intervals_use_historical_delivery_uncertainty(self) -> None:
        leaves = self.model.forecast_campaigns(self.canonical, 30)
        self.assertFalse(leaves.empty)
        ratios = leaves["predicted_spend_p10"] / leaves["predicted_spend_p50"].clip(lower=1e-9)
        # Historical CV replaces the old fixed 0.90 / 1.05 cosmetic bands.
        self.assertFalse(((ratios - 0.90).abs() < 1e-12).all())
        self.assertTrue((leaves["predicted_spend_p50"] == leaves["planned_budget"]).all())

    def test_scenario_accepts_override_and_rejects_negative_budget(self) -> None:
        baseline = build_forecast(self.model, self.canonical, 30)
        campaign_key = str(baseline[baseline["level"] == "campaign"].iloc[0]["campaign_key"])
        simulated = simulate_budget_plan(self.model, self.canonical, 30, {campaign_key: 1000.0}, target_roas=3.5)
        self.assertEqual(set(simulated["horizon_days"]), {30})
        for invalid_budget in (-1.0, float("nan"), float("inf")):
            with self.subTest(budget=invalid_budget), self.assertRaises(ValueError):
                simulate_budget_plan(self.model, self.canonical, 30, {campaign_key: invalid_budget})

    def test_zero_budget_scenario_cannot_emit_paid_media_revenue(self) -> None:
        baseline = build_forecast(self.model, self.canonical, 30)
        campaign = baseline[baseline["level"] == "campaign"].iloc[0]
        scenario = simulate_budget_plan(self.model, self.canonical, 30, {str(campaign["campaign_key"]): 0.0})
        zeroed = scenario[(scenario["level"] == "campaign") & (scenario["campaign_key"] == campaign["campaign_key"])].iloc[0]
        self.assertEqual(float(zeroed["planned_budget"]), 0.0)
        self.assertEqual(float(zeroed["predicted_revenue_p10"]), 0.0)
        self.assertEqual(float(zeroed["predicted_revenue_p50"]), 0.0)
        self.assertEqual(float(zeroed["predicted_revenue_p90"]), 0.0)

    def test_ninety_day_default_plan_is_seasonally_marked(self) -> None:
        forecast = build_forecast(self.model, self.canonical, 90)
        leaves = forecast[forecast["level"] == "campaign"]
        self.assertTrue(leaves["quality_flags"].str.contains("seasonally_adjusted_baseline_budget", na=False).any())

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

    def test_submission_header_matches_versioned_fixture(self) -> None:
        fixture = (PRODUCT_ROOT / "tests" / "fixtures" / "horizon_v1_header.csv").read_text(encoding="utf-8").strip().split(",")
        self.assertEqual(FORECAST_COLUMNS, fixture)

    def test_contract_verifier_supports_direct_script_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary) / "predictions.csv"
            predictions.write_text(",".join(FORECAST_COLUMNS) + "\n", encoding="utf-8", newline="")
            verifier = PRODUCT_ROOT / "scripts" / "verify_evaluator_contract.py"
            result = subprocess.run(
                [sys.executable, str(verifier), "--predictions", str(predictions)],
                cwd=Path(temporary),
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Evaluator CSV header matches locked horizon-v1 fixture", result.stdout)

    def test_source_validation_reports_missing_full_schema_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            google_path = data_dir / "google_ads_campaign_stats.csv"
            pd.read_csv(google_path).drop(columns=["metrics_clicks"]).to_csv(google_path, index=False)
            with self.assertRaisesRegex(ValueError, "metrics_clicks"):
                read_source_files(data_dir)

    def test_source_semantics_rejects_unresolved_currency_mix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            (data_dir / "source_semantics.csv").write_text(
                "source_system,currency,timezone,attribution_method,revenue_field\n"
                "google_ads,USD,UTC,platform,metrics_conversions_value\n"
                "microsoft_ads,INR,UTC,platform,Revenue\n"
                "meta_ads,USD,UTC,platform,conversion\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "multiple currencies"):
                read_source_files(data_dir)

    def test_demo_metadata_is_reviewed_and_suppresses_semantic_warnings(self) -> None:
        raw = read_source_files(PRODUCT_ROOT / "demo_data")
        self.assertIn("source_semantics", raw)
        self.assertIn("campaign_taxonomy", raw)
        self.assertTrue(raw["source_semantics"]["review_status"].eq("reviewed").all())
        self.assertTrue(raw["campaign_taxonomy"]["review_status"].eq("reviewed").all())

        canonical = canonicalize(raw)
        report = validate_canonical(canonical)
        semantic_warnings = [
            warning for warning in report.warnings
            if "semantic" in warning.lower() or "taxonomy" in warning.lower() or "revenue" in warning.lower()
        ]
        self.assertEqual(semantic_warnings, [])
        self.assertFalse(
            canonical["quality_flags"].str.contains(
                "unknown|unreviewed|treated_as_attributed", case=False, na=False
            ).any()
        )
        meta = canonical[canonical["source_system"] == "meta_ads"]
        self.assertTrue(meta["quality_flags"].str.contains("meta_campaign_type_mapped_reviewed", na=False).all())
        self.assertTrue(meta["quality_flags"].str.contains("meta_conversion_semantics_reviewed", na=False).all())

    def test_default_evaluator_data_still_allows_missing_review_metadata(self) -> None:
        canonical = canonicalize(read_source_files(ROOT / "data"))
        report = validate_canonical(canonical)
        self.assertEqual(report.blockers, [])
        self.assertTrue(any("semantics manifest absent" in warning for warning in report.warnings))

    def test_unreviewed_semantics_manifest_remains_visible_as_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(PRODUCT_ROOT / "demo_data", data_dir)
            semantics_path = data_dir / "source_semantics.csv"
            semantics = pd.read_csv(semantics_path)
            semantics["review_status"] = "unreviewed"
            semantics.to_csv(semantics_path, index=False)
            report = validate_canonical(canonicalize(read_source_files(data_dir)))
        self.assertTrue(any("semantics manifest absent or unreviewed" in warning for warning in report.warnings))
        self.assertTrue(any("Meta revenue semantics" in warning for warning in report.warnings))

    def test_metadata_rejects_unknown_review_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(PRODUCT_ROOT / "demo_data", data_dir)
            taxonomy_path = data_dir / "campaign_taxonomy.csv"
            taxonomy = pd.read_csv(taxonomy_path)
            taxonomy.loc[0, "review_status"] = "approved"
            taxonomy.to_csv(taxonomy_path, index=False)
            with self.assertRaisesRegex(ValueError, "review_status must be one of"):
                read_source_files(data_dir)

    def test_semantics_manifest_rejects_revenue_field_not_in_source_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(PRODUCT_ROOT / "demo_data", data_dir)
            (data_dir / "source_semantics.csv").write_text(
                "source_system,currency,timezone,attribution_method,revenue_field,review_status\n"
                "google_ads,USD,UTC,platform_attributed,metrics_conversions_value,reviewed\n"
                "microsoft_ads,USD,UTC,platform_attributed,Revenue,reviewed\n"
                "meta_ads,USD,UTC,platform_attributed,Revenue,reviewed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "revenue_field values are not present"):
                read_source_files(data_dir)

    def test_taxonomy_rejects_campaign_not_present_in_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(PRODUCT_ROOT / "demo_data", data_dir)
            taxonomy_path = data_dir / "campaign_taxonomy.csv"
            taxonomy = pd.read_csv(taxonomy_path)
            taxonomy.loc[len(taxonomy)] = {
                "source_system": "meta_ads",
                "source_campaign_id": "not-in-upload",
                "campaign_type": "META_GENERIC",
                "review_status": "reviewed",
            }
            taxonomy.to_csv(taxonomy_path, index=False)
            with self.assertRaisesRegex(ValueError, "not present in this upload"):
                read_source_files(data_dir)

    def test_media_plan_applies_on_scored_predict_path(self) -> None:
        baseline = build_forecast(self.model, self.canonical, 30)
        leaf = baseline[baseline["level"] == "campaign"].iloc[0]
        source_system, campaign_id = str(leaf["campaign_key"]).split(":", 1)
        planned = max(250.0, float(leaf["planned_budget"]) * 1.35)
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(PRODUCT_ROOT / "demo_data", data_dir)
            (data_dir / "media_plan.csv").write_text(
                "source_system,source_campaign_id,horizon_days,planned_budget\n"
                f"{source_system},{campaign_id},30,{planned}\n",
                encoding="utf-8",
            )
            output = Path(temporary) / "predictions.csv"
            generate_predictions(data_dir, ROOT / "pickle" / "model.pkl", output)
            result = pd.read_csv(output)
            matched = result[
                (result["horizon_days"] == 30)
                & (result["level"] == "campaign")
                & (result["channel"].astype(str) == str(leaf["channel"]))
                & (result["campaign_id"].astype(str) == campaign_id)
            ]
            self.assertEqual(len(matched), 1)
            self.assertAlmostEqual(float(matched.iloc[0]["planned_budget"]), planned, places=4)

    def test_media_plan_rejects_invalid_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            (data_dir / "media_plan.csv").write_text(
                "source_system,source_campaign_id,horizon_days,planned_budget\n"
                "google_ads,demo,45,1000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "horizon_days"):
                read_source_files(data_dir)

    def test_forecast_id_binds_data_and_scenario(self) -> None:
        baseline = build_forecast(self.model, self.canonical, 30)
        campaign_key = str(baseline[baseline["level"] == "campaign"].iloc[0]["campaign_key"])
        scenario = build_forecast(self.model, self.canonical, 30, {campaign_key: 1000.0})
        changed_data = self.canonical.copy()
        changed_data.loc[changed_data.index[0], "revenue"] += 1.0
        changed = build_forecast(self.model, changed_data, 30)
        changed_delivery = self.canonical.copy()
        changed_delivery.loc[changed_delivery.index[0], "configured_budget"] += 1.0
        changed_spend_uncertainty = build_forecast(self.model, changed_delivery, 30)
        self.assertNotEqual(baseline.iloc[0]["forecast_id"], scenario.iloc[0]["forecast_id"])
        self.assertNotEqual(baseline.iloc[0]["forecast_id"], changed.iloc[0]["forecast_id"])
        self.assertNotEqual(baseline.iloc[0]["forecast_id"], changed_spend_uncertainty.iloc[0]["forecast_id"])

    def test_unqualified_budget_override_rejects_cross_source_collision(self) -> None:
        collision = self.canonical.copy()
        duplicate = collision.iloc[[0]].copy()
        duplicate["source_system"] = "other_ads"
        duplicate["channel"] = "OTHER"
        duplicate["campaign_type"] = "OTHER"
        collision = pd.concat([collision, duplicate], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Ambiguous unqualified campaign budget override"):
            self.model.forecast_campaigns(collision, 30, {str(self.canonical.iloc[0]["source_campaign_id"]): 1000.0})

    def test_direct_models_use_temporal_calibration_windows(self) -> None:
        for model in self.model.direct_models.values():
            self.assertGreater(model.calibration_sample_count, 0)
            if hasattr(model, "scenario_model"):
                self.assertEqual(model.uncertainty_method, "purged_temporal_selected_two_expert_ensemble_v1")
                self.assertGreater(model.scenario_model.calibration_sample_count, 0)
                self.assertGreater(model.inference_aligned_model.calibration_sample_count, 0)
                self.assertEqual(
                    model.scenario_model.uncertainty_method,
                    "purged_temporal_holdout_support_aware_residual_quantiles_v3",
                )
                self.assertEqual(
                    model.inference_aligned_model.uncertainty_method,
                    "purged_temporal_holdout_inference_aligned_residual_quantiles_v1",
                )
            else:
                self.assertEqual(model.uncertainty_method, "purged_temporal_holdout_support_aware_residual_quantiles_v3")

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

    def test_output_adapter_serializes_stable_numeric_text_and_lf(self) -> None:
        """Serialization—not model logic—owns portable evaluator text output."""
        forecast = build_forecast(self.model, self.canonical, 30)
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "predictions.csv"
            write_predictions_csv(forecast, output_path)
            payload = output_path.read_bytes()
        self.assertNotIn(b"\r\n", payload)
        first = next(csv.DictReader(payload.decode("utf-8").splitlines()))
        for name in (
            "planned_budget",
            "predicted_revenue_p10",
            "predicted_revenue_p50",
            "predicted_revenue_p90",
            "predicted_roas_p50",
            "probability_roas_above_target",
            "risk_score",
        ):
            self.assertRegex(first[name], r"^-?\d+\.\d{6}$", name)

    def test_prediction_generation_is_atomic_and_missing_model_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "predictions.csv"
            generate_predictions(PRODUCT_ROOT / "demo_data", ROOT / "pickle" / "model.pkl", output)
            self.assertTrue(output.is_file())
            with self.assertRaises(FileNotFoundError):
                generate_predictions(PRODUCT_ROOT / "demo_data", Path(temporary) / "missing.pkl", output)

    def test_predictions_are_byte_deterministic_and_runner_caps_blas_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.csv"
            second = Path(temporary) / "second.csv"
            generate_predictions(PRODUCT_ROOT / "demo_data", ROOT / "pickle" / "model.pkl", first)
            generate_predictions(PRODUCT_ROOT / "demo_data", ROOT / "pickle" / "model.pkl", second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
        runner = (ROOT / "run.sh").read_text(encoding="utf-8")
        for variable in ("OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"):
            self.assertIn(variable, runner)

    def test_model_manifest_rejects_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.pkl"
            shutil.copy2(ROOT / "pickle" / "model.pkl", model_path)
            model_path.with_name("model_manifest.json").write_text(
                '{"artifact_sha256":"not-the-model","model_version":"horizon-direct-ridge-v3-seasonal-plan"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_model(model_path)

    def test_optimizer_returns_feasible_reconciled_plan(self) -> None:
        baseline = build_forecast(self.model, self.canonical, 30)
        total = float(baseline[baseline["level"] == "overall"].iloc[0]["planned_budget"])
        result = recommend_allocation(self.model, self.canonical, 30, total, target_roas=3.5, increments=80)
        overall = result.forecast[result.forecast["level"] == "overall"].iloc[0]
        self.assertEqual(result.status, "feasible")
        self.assertAlmostEqual(sum(result.campaign_budgets.values()), total, places=2)
        self.assertGreater(float(overall["predicted_revenue_p50"]), 0.0)
        self.assertIn(result.target_constraint_status, {"marginal_target_met", "marginal_target_relaxed"})

    def test_optimizer_declares_when_marginal_target_is_relaxed(self) -> None:
        baseline = build_forecast(self.model, self.canonical, 30)
        total = float(baseline[baseline["level"] == "overall"].iloc[0]["planned_budget"])
        result = recommend_allocation(self.model, self.canonical, 30, total, target_roas=100.0, increments=80)
        self.assertEqual(result.target_constraint_status, "marginal_target_relaxed")

    def test_planner_uses_the_integrity_checked_submission_model_loader(self) -> None:
        with patch("product.app.service.load_model", return_value=self.model) as loader, patch(
            "product.app.service.DecisionLedger"
        ):
            service = PlannerService(PRODUCT_ROOT / "demo_data", ROOT / "pickle" / "model.pkl")
        loader.assert_called_once_with(ROOT / "pickle" / "model.pkl")
        self.assertIs(service.model, self.model)

    def test_planner_never_runs_a_lazy_backtest_when_report_is_missing(self) -> None:
        service = self._planner_service()
        with tempfile.TemporaryDirectory() as temporary, patch("product.app.service.PRODUCT_ROOT", Path(temporary)), patch(
            "product.evaluation.evaluate_all_horizons", side_effect=AssertionError("backtesting must not run in serving")
        ):
            report = service.trust_report()
        self.assertEqual(report["status"], "not_applicable")
        self.assertFalse(report["evaluation_performed_at_request"])
        self.assertEqual(report["horizons"], [])

    def test_planner_rejects_persisted_report_with_mismatched_artifact_provenance(self) -> None:
        service = self._planner_service()
        expected_fingerprint = canonical_fingerprint(self.canonical)
        artifact_provenance = {
            "artifact_sha256": str(getattr(self.model, "artifact_sha256", "") or ""),
            "model_version": self.model.model_version,
            "training_data_fingerprint": str(
                getattr(self.model, "training_data_fingerprint", "") or expected_fingerprint
            ),
            "feature_schema_fingerprint": str(getattr(self.model, "feature_schema_fingerprint", "") or ""),
        }
        artifact_provenance["artifact_sha256"] = "not-the-loaded-artifact"
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "models" / "evaluation_report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "data_fingerprint": expected_fingerprint,
                        "artifact_provenance": artifact_provenance,
                        "horizons": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch("product.app.service.PRODUCT_ROOT", Path(temporary)):
                report = service.trust_report()
        self.assertEqual(report["status"], "not_applicable")
        self.assertIn("artifact provenance", report["reason"])

    def test_insufficient_persisted_60_day_calibration_forces_revise_or_test(self) -> None:
        insufficient = self._sufficient_trust_report()
        insufficient["horizons"][0].update({"revenue_interval_coverage": 0.50, "roas_interval_coverage": 0.0})
        service = self._planner_service(trust_report=insufficient)
        forecast = build_forecast(self.model, self.canonical, 60)
        evidence = service._evidence(forecast, 4.0, 60)
        self.assertEqual(evidence["decision"], "revise_or_test")
        self.assertEqual(evidence["decision_gates"]["calibration"]["status"], "insufficient")
        self.assertTrue(evidence["decision_gates"]["calibration"]["reasons"])

    def test_unavailable_persisted_60_day_calibration_forces_revise_or_test(self) -> None:
        service = self._planner_service(
            trust_report={"status": "not_applicable", "reason": "fingerprint does not match", "horizons": []}
        )
        forecast = build_forecast(self.model, self.canonical, 60)
        evidence = service._evidence(forecast, 4.0, 60)
        self.assertEqual(evidence["decision"], "revise_or_test")
        self.assertEqual(evidence["decision_gates"]["calibration"]["status"], "unavailable")

    def test_planner_rejects_fractional_non_finite_and_boolean_numeric_payloads(self) -> None:
        service = self._planner_service(trust_report=self._sufficient_trust_report())
        cases = (
            ({"horizon_days": 60.5}, "exact integer"),
            ({"horizon_days": True}, "integer JSON number"),
            ({"horizon_days": 60, "target_roas": float("nan")}, "finite"),
            ({"horizon_days": 60, "target_roas": float("inf")}, "finite"),
            ({"horizon_days": 60, "campaign_budgets": {"google_ads:demo": "100"}}, "JSON number"),
        )
        for payload, message in cases:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                service.forecast(payload)

    def test_meta_revenue_assumption_is_derived_from_canonical_quality_flags(self) -> None:
        service = self._planner_service()
        status, message = service._meta_revenue_semantics()
        self.assertEqual(status, "reviewed")
        self.assertIn("documented as reviewed", message)

        modified = self.canonical.copy()
        mask = modified["source_system"].eq("meta_ads")
        modified.loc[mask, "quality_flags"] = "meta_conversion_treated_as_attributed_revenue;source_semantics_unreviewed"
        service.canonical = modified
        status, message = service._meta_revenue_semantics()
        self.assertEqual(status, "assumed_proxy")
        self.assertIn("revenue proxy", message)

    def test_live_llm_is_server_disabled_unless_explicitly_enabled_on_localhost(self) -> None:
        validate_live_llm_host("127.0.0.1", True)
        validate_live_llm_host("localhost", True)
        with self.assertRaisesRegex(ValueError, "localhost"):
            validate_live_llm_host("0.0.0.0", True)

        service = PlannerService.__new__(PlannerService)
        service.allow_live_llm = False
        service.forecast = lambda payload: {
            "scenario": {"horizon_days": 30, "target_roas": 4.0, "campaign_budgets": {}},
            "overall": [{
                "forecast_id": "test-forecast", "probability_roas_above_target": 0.5, "risk_score": 40.0,
                "predicted_revenue_p10": 10.0, "predicted_revenue_p50": 20.0, "predicted_revenue_p90": 30.0,
            }],
            "evidence": {"decision": "revise_or_test", "target_roas": 4.0, "drivers": [], "risks": []},
        }
        with patch("product.app.service.load_evidence_config") as configuration:
            response = service.explain({"scenario": {"horizon_days": 30}, "prefer_live_llm": True})
        configuration.assert_not_called()
        self.assertEqual(response["mode"], "deterministic_fallback")
        self.assertIn("disabled by this local server", response["message"])

    def test_server_hides_internal_failures_from_api_clients(self) -> None:
        service = SimpleNamespace(forecast=Mock(side_effect=RuntimeError("sensitive local detail")))
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/scenario",
            data=b'{"horizon_days":60}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with patch("product.app.server.LOGGER.exception"), self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=5)
            response = json.loads(raised.exception.read().decode("utf-8"))
        finally:
            thread.join(timeout=5)
            server.server_close()
        self.assertEqual(response["error"]["code"], "FORECAST_FAILED")
        self.assertNotIn("sensitive local detail", response["error"]["message"])
        self.assertIn("local server log", response["error"]["message"])

    def test_data_health_labels_missing_configured_budgets_as_planning_defaults(self) -> None:
        service = self._planner_service()
        health = service.data_health()
        self.assertTrue(health["planning_defaults"])
        self.assertTrue(all(item in health["warnings"] for item in health["planning_defaults"]))

    def test_optimizer_response_and_ledger_use_the_exact_campaign_plan(self) -> None:
        service = self._planner_service(trust_report=self._sufficient_trust_report(30))
        baseline = build_forecast(self.model, self.canonical, 30)
        campaign_key = str(baseline[baseline["level"] == "campaign"].iloc[0]["campaign_key"])
        recommended = {campaign_key: 1234.56}
        reforecast = build_forecast(self.model, self.canonical, 30, recommended)
        fake_result = SimpleNamespace(
            forecast=reforecast,
            campaign_budgets=recommended,
            status="feasible",
            target_constraint_status="marginal_target_met",
            target_roas=4.0,
            achieved_roas_p50=4.2,
            explanation="test recommendation",
        )
        with patch("product.app.service.recommend_allocation", return_value=fake_result):
            response = service.optimize({"horizon_days": 30, "target_roas": 4.0, "total_budget": 1234.56})
        self.assertEqual(response["scenario"]["campaign_budgets"], response["optimization"]["campaign_budgets"])

        service.ledger.record.return_value = {"id": 1}
        service.record_decision({"action": "draft", "scenario": response["scenario"]})
        saved_scenario = service.ledger.record.call_args.args[1]
        saved_summary = service.ledger.record.call_args.args[2]
        self.assertEqual(saved_scenario["campaign_budgets"], response["scenario"]["campaign_budgets"])
        self.assertEqual(saved_summary["forecast_id"], response["overall"][0]["forecast_id"])

    def test_frontend_pins_exact_optimizer_plan_and_marks_changed_inputs_stale(self) -> None:
        script = (PRODUCT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        self.assertIn("latestScenario = copyScenario(scenario)", script)
        self.assertIn("campaign_budgets", script)
        self.assertIn("markScenarioStale", script)
        self.assertIn("scenario:copyScenario(latestScenario)", script)

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
        response = service.explain({"scenario": {"horizon_days": 30}})
        self.assertEqual(response["mode"], "deterministic_evidence_brief")
        self.assertIn("brief", response)
        self.assertEqual(response["brief"]["causal_status"], "observational_association")
        self.assertEqual(response["forecast_id"], "test-forecast")
        with patch("product.app.service.load_evidence_config", return_value=None):
            live = service.explain({"scenario": {"horizon_days": 30}, "prefer_live_llm": True})
        self.assertEqual(live["mode"], "deterministic_fallback")

    def test_evidence_client_accepts_a_guarded_structured_response(self) -> None:
        forecast = build_forecast(self.model, self.canonical, 30)
        overall = forecast[forecast["level"] == "overall"].iloc[0].to_dict()
        packet = build_evidence_packet(
            {"decision": "approve", "target_roas": 4.0, "drivers": [], "risks": []}, overall,
        )
        response = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"decision\":\"approve\",\"causal_status\":\"observational_association\",\"headline\":\"The scenario merits controlled validation.\",\"facts\":[{\"text\":\"The outlook is conditional on supplied attribution.\",\"evidence_ids\":[\"forecast_boundary\"]}],\"assumptions\":[],\"recommendations\":[{\"text\":\"Use a bounded split test before a material allocation change.\",\"evidence_ids\":[\"forecast_boundary\"]}],\"limitations\":[{\"text\":\"The range represents uncertainty rather than a promise.\",\"evidence_ids\":[\"forecast_range\"]}]}"}]}]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                import json
                return json.dumps(response).encode("utf-8")

        with patch("product.app.evidence.urlopen", return_value=FakeResponse()):
            brief = OpenAIEvidenceClient(EvidenceClientConfig(api_key="test-key", model="test-model")).generate(packet)
        self.assertEqual(brief["decision"], "approve")


if __name__ == "__main__":
    unittest.main()
