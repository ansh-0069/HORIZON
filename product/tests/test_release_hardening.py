"""Focused regression tests for evaluator safety and decision-support boundaries.

These tests deliberately avoid fitting the expensive optional training pipeline.
They exercise the deterministic contracts that must remain true after a model
artifact has been promoted: media-plan defaults, model-family selection,
allocation semantics, decision gates, and the protected runner interface.
"""
from __future__ import annotations

import copy
import ast
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from product.app.service import PlannerService
from product.decisioning.optimizer import recommend_allocation
from product.training.model_builder import _select_model_families, fit_horizon_model
from src.budget_plan import campaign_baseline_budget, calendar_matched_prior_year_budget
from src.model import HorizonModel


ROOT = Path(__file__).resolve().parents[2]


def _canonical_rows(
    campaign_id: str,
    dates: pd.DatetimeIndex,
    spend: float,
    *,
    revenue_multiplier: float = 4.0,
    channel: str = "SEARCH",
    campaign_type: str = "BRAND",
) -> pd.DataFrame:
    """Create the minimum canonical slice required by protected inference."""
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_system": "google_ads",
            "source_campaign_id": campaign_id,
            "channel": channel,
            "campaign_type": campaign_type,
            "campaign_name": f"Campaign {campaign_id}",
            "spend": float(spend),
            "revenue": float(spend) * revenue_multiplier,
            "configured_budget": 0.0,
        }
    )
    # Training provenance fingerprints the full canonical contract even when
    # a focused unit test exercises only spend/revenue behavior.
    frame["clicks"] = 1.0
    frame["impressions"] = 10.0
    frame["conversions"] = 1.0
    frame["quality_flags"] = ""
    return frame


class CalendarMatchedPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.as_of = pd.Timestamp("2026-06-30")
        self.recent_dates = pd.date_range(self.as_of - pd.Timedelta(days=27), self.as_of, freq="D")
        self.prior_dates = pd.date_range("2025-07-01", periods=30, freq="D")

    def test_calendar_plan_requires_coverage_and_is_bounded_by_current_pacing(self) -> None:
        # The current 28-day plan is 28 * $10 / 28 * 30 = $300.  A prior-year
        # $1,000/day promotion must be capped at four times that run rate,
        # rather than silently becoming a $30,000 evaluator default.
        current = _canonical_rows("active", self.recent_dates, 10.0)
        prior = _canonical_rows("active", self.prior_dates, 1_000.0)
        history = pd.concat([current, prior], ignore_index=True)

        prior_budget, coverage = calendar_matched_prior_year_budget(history, self.as_of, 30)
        budget, method, returned_coverage = campaign_baseline_budget(history, history, self.as_of, 30)

        self.assertEqual(coverage, 1.0)
        self.assertEqual(prior_budget, 30_000.0)
        self.assertEqual(method, "calendar_matched_prior_year_budget")
        self.assertEqual(returned_coverage, 1.0)
        self.assertEqual(budget, 1_200.0)

    def test_low_prior_year_coverage_falls_back_to_current_run_rate(self) -> None:
        current = _canonical_rows("active", self.recent_dates, 10.0)
        # 23 / 30 mapped dates is below the documented 80% coverage gate.
        prior = _canonical_rows("active", self.prior_dates[:23], 100.0)
        history = pd.concat([current, prior], ignore_index=True)

        prior_budget, coverage = calendar_matched_prior_year_budget(history, self.as_of, 30)
        budget, method, returned_coverage = campaign_baseline_budget(history, history, self.as_of, 30)

        self.assertIsNone(prior_budget)
        self.assertAlmostEqual(coverage, 23 / 30)
        self.assertEqual(method, "recent_run_rate_budget")
        self.assertAlmostEqual(returned_coverage, 23 / 30)
        self.assertEqual(budget, 300.0)

    def test_explicit_active_plan_overrides_default_but_dormant_history_is_not_revived(self) -> None:
        active = _canonical_rows("active", self.recent_dates, 10.0)
        dormant = _canonical_rows("dormant", pd.date_range("2025-03-01", periods=40, freq="D"), 50.0)
        canonical = pd.concat([active, dormant], ignore_index=True)
        model = HorizonModel(
            "test-statistical",
            global_roas=4.0,
            global_log_sigma=0.2,
            month_roas_factors={},
            direct_models={},
            selected_model_families={30: "statistical_fallback"},
        )

        leaves = model.forecast_campaigns(canonical, 30, {"google_ads:active": 777.0})

        self.assertEqual(leaves["campaign_id"].tolist(), ["active"])
        self.assertEqual(float(leaves.iloc[0]["planned_budget"]), 777.0)
        with self.assertRaisesRegex(ValueError, "active forecastable campaign"):
            model.forecast_campaigns(canonical, 30, {"google_ads:dormant": 777.0})


class TemporalModelSelectionTests(unittest.TestCase):
    @staticmethod
    def _selection_canonical() -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=150, freq="D")
        return _canonical_rows("selection", dates, 100.0)

    def test_tournament_breaks_exact_error_ties_in_favor_of_statistical_fallback(self) -> None:
        canonical = self._selection_canonical()

        def forecast_at_tie(model: HorizonModel, *_: object) -> pd.DataFrame:
            return pd.DataFrame(
                [{"level": "overall", "predicted_revenue_p50": 3_000.0}]
            )

        with patch("product.training.model_builder.fit_direct_ensemble", return_value=object()), patch(
            "product.training.model_builder.build_forecast", side_effect=forecast_at_tie
        ):
            selected, records = _select_model_families(canonical, (30,))

        self.assertEqual(selected, {30: "statistical_fallback"})
        self.assertEqual(records[30]["selected_family"], "statistical_fallback")
        self.assertEqual(records[30]["status"], "purged_terminal_tournament")

    def test_tournament_selects_direct_only_when_out_of_time_error_is_lower(self) -> None:
        canonical = self._selection_canonical()

        def forecast_with_direct_winner(model: HorizonModel, *_: object) -> pd.DataFrame:
            value = 2_000.0 if model.selected_model_family(30) == "statistical_fallback" else 2_950.0
            return pd.DataFrame([{"level": "overall", "predicted_revenue_p50": value}])

        with patch("product.training.model_builder.fit_direct_ensemble", return_value=object()), patch(
            "product.training.model_builder.build_forecast", side_effect=forecast_with_direct_winner
        ):
            selected, records = _select_model_families(canonical, (30,))

        self.assertEqual(selected, {30: "direct_ensemble"})
        self.assertLess(records[30]["direct_absolute_error"], records[30]["statistical_absolute_error"])

    def test_training_disabled_is_an_explicit_statistical_fallback_not_an_implicit_direct_default(self) -> None:
        model = fit_horizon_model(self._selection_canonical(), train_direct=False, horizons=(30, 60))

        self.assertEqual(model.direct_models, {})
        self.assertEqual(
            {horizon: model.selected_model_family(horizon) for horizon in (30, 60)},
            {30: "statistical_fallback", 60: "statistical_fallback"},
        )
        self.assertIsNone(
            model.direct_quantiles(pd.DataFrame(), "SEARCH", "BRAND", pd.Timestamp("2025-05-30"), 30, 100.0)
        )

    def test_old_pickles_without_tournament_fields_remain_loadable(self) -> None:
        legacy = HorizonModel(
            "legacy-v5",
            global_roas=4.0,
            global_log_sigma=0.2,
            month_roas_factors={},
            direct_models={},
        )
        del legacy.selected_model_families
        del legacy.model_selection
        del legacy.residual_dependence

        restored = pickle.loads(pickle.dumps(legacy))

        self.assertEqual(restored.selected_model_family(30), "direct_ensemble")
        self.assertEqual(restored.dependence_for_horizon(30), {})
        self.assertIsNone(
            restored.direct_quantiles(pd.DataFrame(), "SEARCH", "BRAND", pd.Timestamp("2025-05-30"), 30, 100.0)
        )


class AllocationAndDecisionGateTests(unittest.TestCase):
    @staticmethod
    def _baseline_leaves() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "level": "campaign",
                    "campaign_key": "google_ads:guardrailed",
                    "channel": "SEARCH",
                    "planned_budget": 100.0,
                    "predicted_revenue_p50": 600.0,
                    "quality_flags": "sparse_recent_history",
                },
                {
                    "level": "campaign",
                    "campaign_key": "google_ads:eligible",
                    "channel": "SEARCH",
                    "planned_budget": 100.0,
                    "predicted_revenue_p50": 400.0,
                    "quality_flags": "",
                },
            ]
        )

    def test_allocator_never_calls_direct_model_or_marks_observational_response_as_causal(self) -> None:
        baseline = self._baseline_leaves()
        simulated = pd.DataFrame(
            [
                {
                    "level": "overall",
                    "predicted_revenue_p50": 1_050.0,
                    "predicted_roas_p50": 5.25,
                }
            ]
        )
        model = SimpleNamespace(direct_quantiles=Mock(side_effect=AssertionError("must not be used by allocation")))

        with patch("product.decisioning.optimizer.build_forecast", return_value=baseline), patch(
            "product.decisioning.optimizer.simulate_budget_plan", return_value=simulated
        ):
            result = recommend_allocation(model, pd.DataFrame(), 30, 300.0, target_roas=4.0, increments=3)

        model.direct_quantiles.assert_not_called()
        self.assertEqual(result.causal_status, "observational_association")
        self.assertTrue(result.validation_required)
        self.assertIn("not finite differences of an observational spend-regression coefficient", result.explanation)
        self.assertEqual(result.guardrailed_campaign_count, 1)
        self.assertLessEqual(result.campaign_budgets["google_ads:guardrailed"], 100.0)

    @staticmethod
    def _fully_evaluated_report() -> dict:
        return {
            "status": "available",
            "horizons": [
                {
                    "horizon_days": 60,
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

    @staticmethod
    def _minimal_service(report: dict) -> PlannerService:
        service = PlannerService.__new__(PlannerService)
        service._trust_report = report
        return service

    def test_decision_approval_gate_requires_campaign_coverage_and_evaluated_draw_share_reliability(self) -> None:
        approved = self._minimal_service(self._fully_evaluated_report()).calibration_posture(60)
        self.assertEqual(approved["status"], "sufficient")

        cases = (
            (
                "campaign revenue",
                lambda report: report["horizons"][0]["coverage_by_hierarchy"]["campaign"].update(
                    {"revenue_interval_coverage": 0.49}
                ),
            ),
            (
                "draw share",
                lambda report: report["horizons"][0]["roas_target_probability_reliability"].update(
                    {"status": "insufficient"}
                ),
            ),
            (
                "Brier",
                lambda report: report["horizons"][0]["roas_target_probability_reliability"].update(
                    {"brier_score": 0.26}
                ),
            ),
        )
        for expected_reason, mutate in cases:
            with self.subTest(expected_reason=expected_reason):
                report = copy.deepcopy(self._fully_evaluated_report())
                mutate(report)
                posture = self._minimal_service(report).calibration_posture(60)
                self.assertEqual(posture["status"], "insufficient")
                self.assertTrue(any(expected_reason.lower() in reason.lower() for reason in posture["reasons"]))


class ProtectedRunnerInterfaceTests(unittest.TestCase):
    @staticmethod
    def _bash() -> str | None:
        """Prefer Git Bash on Windows instead of the WSL launcher alias."""
        if os.name == "nt":
            candidate = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Git" / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
        return shutil.which("bash") or shutil.which("bash.exe")

    def test_runner_rejects_more_than_three_arguments_before_optional_runtime_selection(self) -> None:
        bash = self._bash()
        if bash is None:
            self.skipTest("bash is unavailable in this test environment")
        result = subprocess.run(
            [bash, str(ROOT / "run.sh"), "data", "pickle/model.pkl", "output.csv", "unexpected"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 64, result.stderr)
        self.assertIn("Usage: ./run.sh", result.stderr)

    def test_protected_source_has_no_product_or_network_runtime_imports(self) -> None:
        protected_files = [ROOT / "run.sh", ROOT / "requirements.txt", *sorted((ROOT / "src").glob("*.py"))]
        forbidden_roots = {"product", "openai", "requests", "urllib", "http", "httpx", "aiohttp"}
        for path in protected_files:
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_roots = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            } | {
                str(node.module).split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            with self.subTest(path=path.name):
                self.assertFalse(imported_roots & forbidden_roots, path)

        # Keep this structural check tied to executable behavior: the shell
        # parser must accept the protected command even on a machine without
        # optional product dependencies.
        bash = self._bash()
        if bash is not None:
            syntax = subprocess.run([bash, "-n", str(ROOT / "run.sh")], capture_output=True, text=True, check=False)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)


class PromotionBoundaryTests(unittest.TestCase):
    def test_promotion_cli_refuses_without_confirmation_and_does_not_touch_protected_pair(self) -> None:
        script = ROOT / "product" / "scripts" / "promote_submission_model.py"
        model_path = ROOT / "pickle" / "model.pkl"
        manifest_path = ROOT / "pickle" / "model_manifest.json"
        before_model = model_path.read_bytes()
        before_manifest = manifest_path.read_bytes()

        result = subprocess.run(
            [sys.executable, str(script), "--data-dir", str(ROOT / "data")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("without --confirm-promote", result.stderr)
        self.assertEqual(model_path.read_bytes(), before_model)
        self.assertEqual(manifest_path.read_bytes(), before_manifest)

    def test_candidate_validation_reuses_protected_loader_without_promoting(self) -> None:
        # Candidate validation is intentionally read-only. It proves that the
        # release script exercises the exact evaluator loader before it ever
        # attempts to replace the protected model/manifest pair.
        from product.scripts.promote_submission_model import _validate_candidate

        source_model = ROOT / "pickle" / "model.pkl"
        source_manifest = ROOT / "pickle" / "model_manifest.json"
        before_model = source_model.read_bytes()
        before_manifest = source_manifest.read_bytes()
        expected_sha = str(json.loads(source_manifest.read_text(encoding="utf-8"))["artifact_sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary)
            candidate_model = candidate / "model.pkl"
            candidate_manifest = candidate / "model_manifest.json"
            shutil.copy2(source_model, candidate_model)
            shutil.copy2(source_manifest, candidate_manifest)

            _validate_candidate(candidate_model, candidate_manifest, expected_sha)

        self.assertEqual(source_model.read_bytes(), before_model)
        self.assertEqual(source_manifest.read_bytes(), before_manifest)


if __name__ == "__main__":
    unittest.main()
