from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from product.evaluation import (
    _coverage_records,
    _coverage_summary,
    canonical_fingerprint,
    canonical_provenance,
)
from product.training.direct_ridge import (
    _category_vocabulary,
    _temporal_calibration_partitions,
    fit_direct_ridge,
    training_frame,
)
from product.training.train import (
    DEFAULT_PRODUCT_MODEL_PATH,
    PROTECTED_MODEL_PATH,
    safe_product_output_path,
)
from product.training.model_builder import _fit_fallback_interval_calibration, _statistical_components
from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import read_source_files
from src.model import HorizonModel


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = ROOT / "product"


class TrainingAndEvaluationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = canonicalize(read_source_files(PRODUCT_ROOT / "demo_data"))

    def test_temporal_calibration_purges_overlapping_target_windows(self) -> None:
        frame = training_frame(self.canonical, 30)
        fit, calibration, calibration_start = _temporal_calibration_partitions(frame, 30)

        self.assertFalse(fit.empty)
        self.assertFalse(calibration.empty)
        self.assertTrue((fit["target_end"] < calibration_start).all())
        self.assertTrue((calibration["cutoff"] >= calibration_start).all())

    def test_category_vocabulary_never_includes_future_holdout_values(self) -> None:
        fit = pd.DataFrame(
            {
                "channel": ["SEARCH", "SEARCH"],
                "campaign_type": ["BRAND", "NON_BRAND"],
            }
        )
        later = pd.concat(
            [
                fit,
                pd.DataFrame(
                    {
                        "channel": ["SOCIAL"],
                        "campaign_type": ["FUTURE_CAMPAIGN_TYPE"],
                    }
                ),
            ],
            ignore_index=True,
        )
        vocabulary = _category_vocabulary(fit)

        self.assertNotIn("SOCIAL", vocabulary["channel"])
        self.assertNotIn("FUTURE_CAMPAIGN_TYPE", vocabulary["campaign_type"])
        self.assertIn("FUTURE_CAMPAIGN_TYPE", _category_vocabulary(later)["campaign_type"])

    def test_direct_model_refuses_in_sample_interval_fallback(self) -> None:
        # Eighty examples satisfy the coarse direct-model threshold, but the
        # labels all overlap the first calibration origin.  A valid trainer
        # must return None rather than turn fitted residuals into an interval.
        cutoff_values = pd.date_range("2025-01-01", periods=4, freq="10D")
        frame = pd.DataFrame(
            {
                "cutoff": [cutoff for cutoff in cutoff_values for _ in range(20)],
                "target_end": [cutoff + pd.Timedelta(days=30) for cutoff in cutoff_values for _ in range(20)],
            }
        )
        with patch("product.training.direct_ridge.training_frame", return_value=frame):
            self.assertIsNone(fit_direct_ridge(pd.DataFrame(), 30))

    def test_direct_model_persists_holdout_p50_with_ordered_quantiles(self) -> None:
        model = fit_direct_ridge(self.canonical, 30)
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.uncertainty_method, "purged_temporal_holdout_support_aware_residual_quantiles_v3")
        self.assertGreater(model.calibration_sample_count, 0)
        self.assertLessEqual(model.residual_p10, model.residual_p50)
        self.assertLessEqual(model.residual_p50, model.residual_p90)

    def test_hierarchy_coverage_reports_missing_forecasts_and_actuals(self) -> None:
        forecast = pd.DataFrame(
            [
                {
                    "level": "campaign",
                    "campaign_key": "google_ads:matched",
                    "channel": "SEARCH",
                    "campaign_type": "BRAND",
                    "predicted_revenue_p10": 90.0,
                    "predicted_revenue_p90": 110.0,
                    "predicted_roas_p10": 1.8,
                    "predicted_roas_p90": 2.2,
                },
                {
                    "level": "campaign",
                    "campaign_key": "google_ads:forecast_only",
                    "channel": "SEARCH",
                    "campaign_type": "BRAND",
                    "predicted_revenue_p10": 90.0,
                    "predicted_revenue_p90": 110.0,
                    "predicted_roas_p10": 1.8,
                    "predicted_roas_p90": 2.2,
                },
            ]
        )
        actual = pd.DataFrame(
            [
                {
                    "source_system": "google_ads",
                    "source_campaign_id": "matched",
                    "channel": "SEARCH",
                    "campaign_type": "BRAND",
                    "revenue": 100.0,
                    "spend": 50.0,
                },
                {
                    "source_system": "google_ads",
                    "source_campaign_id": "actual_only",
                    "channel": "SEARCH",
                    "campaign_type": "BRAND",
                    "revenue": 100.0,
                    "spend": 50.0,
                },
            ]
        )
        records = _coverage_records(forecast, actual)
        summary = _coverage_summary([record for record in records if record["level"] == "campaign"])["campaign"]

        self.assertEqual(summary["observations"], 3)
        self.assertEqual(summary["matched_observations"], 1)
        self.assertEqual(summary["missing_forecast_observations"], 1)
        self.assertEqual(summary["missing_actual_observations"], 1)
        self.assertEqual(summary["revenue_coverage_observations"], 2)
        self.assertEqual(summary["revenue_interval_coverage"], 0.5)
        self.assertEqual(summary["roas_coverage_observations"], 2)
        self.assertEqual(summary["roas_interval_coverage"], 0.5)

    def test_product_provenance_binds_the_demo_canonical_data(self) -> None:
        provenance = canonical_provenance(self.canonical)

        self.assertEqual(provenance["canonical_fingerprint"], canonical_fingerprint(self.canonical))
        shuffled = self.canonical.sample(frac=1.0, random_state=17).reset_index(drop=True)
        self.assertEqual(canonical_fingerprint(shuffled), canonical_fingerprint(self.canonical))
        self.assertEqual(provenance["canonical_rows"], len(self.canonical))
        self.assertEqual(provenance["source_systems"], ["google_ads", "meta_ads", "microsoft_ads"])

    def test_statistical_fallback_oof_calibration_widens_only_intervals(self) -> None:
        profile = _fit_fallback_interval_calibration(self.canonical, 30)
        self.assertEqual(profile["status"], "available")
        self.assertEqual(profile["calibration_level"], "campaign_marginal")
        self.assertGreaterEqual(int(profile["origin_count"]), 5)
        self.assertGreaterEqual(int(profile["sample_count"]), 30)
        self.assertGreaterEqual(float(profile["lower_width_multiplier"]), 1.0)
        self.assertGreaterEqual(float(profile["upper_width_multiplier"]), 1.0)
        self.assertGreaterEqual(
            float(profile["calibrated_joint_coverage"]),
            float(profile["base_joint_coverage"]),
        )
        portfolio_profile = profile["portfolio_interval_profile"]
        self.assertEqual(portfolio_profile["status"], "available")
        self.assertEqual(
            portfolio_profile["calibration_purpose"],
            "revenue_interval_only_not_roas_probability_calibration",
        )
        self.assertGreaterEqual(
            float(portfolio_profile["calibrated_joint_coverage"]),
            float(portfolio_profile["base_joint_coverage"]),
        )

        roas, sigma, factors = _statistical_components(self.canonical)
        baseline = HorizonModel(
            "test-fallback-baseline",
            roas,
            sigma,
            factors,
            {},
            selected_model_families={30: "statistical_fallback"},
        ).forecast_campaigns(self.canonical, 30).sort_values("campaign_key", kind="stable").reset_index(drop=True)
        calibrated = HorizonModel(
            "test-fallback-calibrated",
            roas,
            sigma,
            factors,
            {},
            selected_model_families={30: "statistical_fallback"},
            fallback_interval_calibration={30: profile},
        ).forecast_campaigns(self.canonical, 30).sort_values("campaign_key", kind="stable").reset_index(drop=True)
        pd.testing.assert_series_equal(
            baseline["predicted_revenue_p50"],
            calibrated["predicted_revenue_p50"],
            check_names=False,
        )
        self.assertTrue(
            (calibrated["predicted_revenue_p10"] <= baseline["predicted_revenue_p10"]).all()
        )
        self.assertTrue(
            (calibrated["predicted_revenue_p90"] >= baseline["predicted_revenue_p90"]).all()
        )
        self.assertTrue(calibrated["quality_flags"].str.contains("fallback_oof_interval_calibrated").all())

        raw_rollup = build_forecast(
            HorizonModel(
                "test-fallback-raw-rollup",
                roas,
                sigma,
                factors,
                {},
                selected_model_families={30: "statistical_fallback"},
            ),
            self.canonical,
            30,
        )
        calibrated_rollup = build_forecast(
            HorizonModel(
                "test-fallback-calibrated-rollup",
                roas,
                sigma,
                factors,
                {},
                selected_model_families={30: "statistical_fallback"},
                fallback_interval_calibration={30: profile},
            ),
            self.canonical,
            30,
        )
        raw_overall = raw_rollup[raw_rollup["level"] == "overall"].iloc[0]
        calibrated_overall = calibrated_rollup[calibrated_rollup["level"] == "overall"].iloc[0]
        self.assertAlmostEqual(
            float(raw_overall["predicted_revenue_p50"]),
            float(calibrated_overall["predicted_revenue_p50"]),
            places=6,
        )
        self.assertNotEqual(
            float(raw_overall["predicted_revenue_p10"]),
            float(calibrated_overall["predicted_revenue_p10"]),
        )
        self.assertIn(
            "fallback_portfolio_oof_revenue_interval_calibration",
            str(calibrated_overall["quality_flags"]),
        )
        self.assertNotIn("roas_probability_calibration", str(calibrated_overall["quality_flags"]))
        malformed = HorizonModel(
            "test-fallback-malformed-profile",
            roas,
            sigma,
            factors,
            {},
            selected_model_families={30: "statistical_fallback"},
            fallback_interval_calibration={30: {"status": "available", "portfolio_interval_profile": {}}},
        )
        self.assertIsNone(malformed.fallback_portfolio_interval_profile(30))

    def test_optional_trainer_cannot_target_the_protected_artifact(self) -> None:
        self.assertNotEqual(DEFAULT_PRODUCT_MODEL_PATH.resolve(), PROTECTED_MODEL_PATH)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite protected evaluator artifact"):
            safe_product_output_path(PROTECTED_MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
