"""Regression tests for evaluator-safe source and metadata trust boundaries."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from src.canonicalize import canonicalize
from src.ingest import read_source_files
from src.predict import generate_predictions
from src.validate import validate_canonical


ROOT = Path(__file__).resolve().parents[2]
DEMO_DATA = ROOT / "product" / "demo_data"


class DataContractHardeningTests(unittest.TestCase):
    def _copy_demo_data(self, temporary: str) -> Path:
        destination = Path(temporary) / "data"
        shutil.copytree(DEMO_DATA, destination)
        return destination

    def test_semantics_cannot_attest_to_a_different_existing_revenue_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = self._copy_demo_data(temporary)
            semantics_path = data_dir / "source_semantics.csv"
            semantics = pd.read_csv(semantics_path, dtype="string")
            semantics.loc[semantics["source_system"] == "google_ads", "revenue_field"] = "metrics_clicks"
            semantics.to_csv(semantics_path, index=False)
            with self.assertRaisesRegex(ValueError, "protected canonical mappings"):
                read_source_files(data_dir)

    def test_unreviewed_taxonomy_cannot_override_model_feature_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = self._copy_demo_data(temporary)
            meta_path = data_dir / "meta_ads_campaign_stats.csv"
            meta = pd.read_csv(meta_path, dtype={"campaign_id": "string"})
            candidate = meta[
                meta["campaign_name"].astype("string").str.contains("prospecting", case=False, na=False)
            ].iloc[0]
            campaign_id = str(candidate["campaign_id"])
            taxonomy_path = data_dir / "campaign_taxonomy.csv"
            taxonomy = pd.read_csv(taxonomy_path, dtype={"source_campaign_id": "string"})
            matching = taxonomy["source_campaign_id"].eq(campaign_id)
            self.assertTrue(matching.any())
            taxonomy.loc[matching, "campaign_type"] = "UNTRUSTED_OVERRIDE"
            taxonomy.loc[matching, "review_status"] = "unreviewed"
            taxonomy.to_csv(taxonomy_path, index=False)

            canonical = canonicalize(read_source_files(data_dir))
            row = canonical[
                (canonical["source_system"] == "meta_ads")
                & (canonical["source_campaign_id"] == campaign_id)
            ].iloc[0]
            self.assertNotEqual(str(row["campaign_type"]), "UNTRUSTED_OVERRIDE")
            self.assertIn("meta_campaign_type_mapping_unreviewed_fallback", str(row["quality_flags"]))
            self.assertTrue(
                any("campaign taxonomy is incomplete" in warning for warning in validate_canonical(canonical).warnings)
            )

    def test_taxonomy_rejects_sources_without_a_supported_override_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = self._copy_demo_data(temporary)
            google = pd.read_csv(data_dir / "google_ads_campaign_stats.csv", dtype={"campaign_id": "string"})
            taxonomy_path = data_dir / "campaign_taxonomy.csv"
            taxonomy = pd.read_csv(taxonomy_path, dtype={"source_campaign_id": "string"})
            taxonomy.loc[len(taxonomy)] = {
                "source_system": "google_ads",
                "source_campaign_id": str(google.iloc[0]["campaign_id"]),
                "campaign_type": "SEARCH",
                "review_status": "reviewed",
            }
            taxonomy.to_csv(taxonomy_path, index=False)
            with self.assertRaisesRegex(ValueError, r"only for \['meta_ads'\]"):
                read_source_files(data_dir)

    def test_campaign_ids_preserve_leading_zeroes_through_metadata_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = self._copy_demo_data(temporary)
            meta_path = data_dir / "meta_ads_campaign_stats.csv"
            meta = pd.read_csv(meta_path, dtype={"campaign_id": "string"})
            original = str(meta.iloc[0]["campaign_id"])
            opaque = f"000{original}"
            meta.loc[meta["campaign_id"] == original, "campaign_id"] = opaque
            meta.to_csv(meta_path, index=False)
            taxonomy_path = data_dir / "campaign_taxonomy.csv"
            taxonomy = pd.read_csv(taxonomy_path, dtype={"source_campaign_id": "string"})
            taxonomy.loc[taxonomy["source_campaign_id"] == original, "source_campaign_id"] = opaque
            taxonomy.to_csv(taxonomy_path, index=False)

            canonical = canonicalize(read_source_files(data_dir))
            self.assertIn(opaque, set(canonical["source_campaign_id"].astype(str)))

    def test_blank_source_campaign_identifier_fails_before_canonicalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = self._copy_demo_data(temporary)
            meta_path = data_dir / "meta_ads_campaign_stats.csv"
            meta = pd.read_csv(meta_path, dtype={"campaign_id": "string"})
            meta.loc[0, "campaign_id"] = ""
            meta.to_csv(meta_path, index=False)
            with self.assertRaisesRegex(ValueError, "blank campaign_id"):
                read_source_files(data_dir)

    def test_malformed_required_measure_is_a_quality_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            google_path = data_dir / "google_ads_campaign_stats.csv"
            google = pd.read_csv(google_path, dtype={"campaign_id": "string"})
            google["metrics_clicks"] = google["metrics_clicks"].astype("object")
            google.loc[0, "metrics_clicks"] = "not-a-number"
            google.to_csv(google_path, index=False)
            report = validate_canonical(canonicalize(read_source_files(data_dir)))
            self.assertTrue(any("clicks" in blocker for blocker in report.blockers))

    def test_media_plan_rejects_fractional_and_non_finite_horizons(self) -> None:
        """Optional plans must not coerce 30.5 into a scored 30-day scenario."""
        for invalid_horizon, expected in (("30.5", "exact integer"), ("inf", "non-finite")):
            with self.subTest(horizon_days=invalid_horizon), tempfile.TemporaryDirectory() as temporary:
                data_dir = Path(temporary) / "data"
                shutil.copytree(ROOT / "data", data_dir)
                google = pd.read_csv(data_dir / "google_ads_campaign_stats.csv", dtype={"campaign_id": "string"})
                campaign_id = str(google.iloc[0]["campaign_id"])
                (data_dir / "media_plan.csv").write_text(
                    "source_system,source_campaign_id,horizon_days,planned_budget\n"
                    f"google_ads,{campaign_id},{invalid_horizon},1000\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, expected):
                    read_source_files(data_dir)

    def test_media_plan_rejects_unknown_campaign_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            (data_dir / "media_plan.csv").write_text(
                "source_system,source_campaign_id,horizon_days,planned_budget\n"
                "google_ads,not-in-upload,30,1000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not present in this upload"):
                read_source_files(data_dir)

    def test_media_plan_fails_when_campaign_is_not_active_in_forecast_window(self) -> None:
        """Known but dormant campaigns require explicit reactivation, not a silent skip."""
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            google_path = data_dir / "google_ads_campaign_stats.csv"
            google = pd.read_csv(google_path, dtype={"campaign_id": "string"})
            google["segments_date"] = pd.to_datetime(google["segments_date"])
            candidate = google.iloc[-1]
            campaign_id = str(candidate["campaign_id"])
            cutoff = google["segments_date"].max()
            dormant = (google["campaign_id"] == campaign_id) & (
                google["segments_date"] >= cutoff - pd.Timedelta(days=27)
            )
            self.assertTrue(dormant.any())
            google.loc[dormant, "metrics_cost_micros"] = 0
            google.to_csv(google_path, index=False)
            (data_dir / "media_plan.csv").write_text(
                "source_system,source_campaign_id,horizon_days,planned_budget\n"
                f"google_ads,{campaign_id},30,1000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "active forecastable campaign"):
                generate_predictions(data_dir, ROOT / "pickle" / "model.pkl", Path(temporary) / "predictions.csv")

    def test_dormant_portfolio_writes_explicit_zero_plan_forecast(self) -> None:
        """A valid but currently dormant account remains evaluable offline."""
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            for filename, date_column, spend_column in (
                ("google_ads_campaign_stats.csv", "segments_date", "metrics_cost_micros"),
                ("bing_campaign_stats.csv", "TimePeriod", "Spend"),
                ("meta_ads_campaign_stats.csv", "date_start", "spend"),
            ):
                path = data_dir / filename
                frame = pd.read_csv(path)
                frame[date_column] = pd.to_datetime(frame[date_column])
                cutoff = frame[date_column].max()
                frame.loc[frame[date_column] >= cutoff - pd.Timedelta(days=27), spend_column] = 0
                frame.to_csv(path, index=False)
            output = Path(temporary) / "predictions.csv"
            generate_predictions(data_dir, ROOT / "pickle" / "model.pkl", output)
            predictions = pd.read_csv(output)
            overall = predictions[predictions["level"] == "overall"]
            self.assertEqual(set(overall["horizon_days"]), {30, 60, 90})
            self.assertTrue((overall["planned_budget"] == 0).all())
            self.assertTrue((overall["predicted_revenue_p50"] == 0).all())
            self.assertTrue(predictions["quality_flags"].str.contains("portfolio_dormant_zero_plan").any())

    def test_blank_campaign_hierarchy_feature_blocks_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            google_path = data_dir / "google_ads_campaign_stats.csv"
            google = pd.read_csv(google_path, dtype={"campaign_id": "string"})
            google.loc[0, "campaign_advertising_channel_type"] = "  "
            google.to_csv(google_path, index=False)
            report = validate_canonical(canonicalize(read_source_files(data_dir)))
            self.assertTrue(any("blank required values in channel" in blocker for blocker in report.blockers))
            self.assertTrue(any("blank required values in campaign_type" in blocker for blocker in report.blockers))

    def test_negative_configured_budget_is_a_quality_blocker_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            shutil.copytree(ROOT / "data", data_dir)
            google_path = data_dir / "google_ads_campaign_stats.csv"
            google = pd.read_csv(google_path, dtype={"campaign_id": "string"})
            google.loc[0, "campaign_budget_amount"] = -1
            google.to_csv(google_path, index=False)
            report = validate_canonical(canonicalize(read_source_files(data_dir)))
            self.assertTrue(any("configured_budget" in blocker for blocker in report.blockers))

    def test_microsoft_campaign_types_use_stable_cross_channel_labels(self) -> None:
        """Presentation-case Microsoft labels must not become model categories."""
        canonical = canonicalize(read_source_files(ROOT / "data"))
        microsoft = canonical[canonical["source_system"] == "microsoft_ads"]
        observed = set(microsoft["campaign_type"].astype(str))
        self.assertTrue({"DISPLAY", "PERFORMANCE_MAX", "SEARCH", "SHOPPING"}.issubset(observed))
        self.assertFalse({"Audience", "PerformanceMax", "Search", "Shopping"} & observed)
        self.assertTrue(
            microsoft["quality_flags"].str.contains("microsoft_campaign_type_normalized").any()
        )


if __name__ == "__main__":
    unittest.main()
