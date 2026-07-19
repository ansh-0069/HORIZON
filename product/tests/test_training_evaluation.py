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
from src.canonicalize import canonicalize
from src.ingest import read_source_files


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
        self.assertEqual(model.uncertainty_method, "purged_temporal_holdout_residual_quantiles_v2")
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

    def test_optional_trainer_cannot_target_the_protected_artifact(self) -> None:
        self.assertNotEqual(DEFAULT_PRODUCT_MODEL_PATH.resolve(), PROTECTED_MODEL_PATH)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite protected evaluator artifact"):
            safe_product_output_path(PROTECTED_MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
