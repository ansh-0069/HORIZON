from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping
import math
import pickle

import pandas as pd

from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import read_source_files
from src.model import HorizonModel
from src.validate import validate_canonical
from product.app.ledger import DecisionLedger
from product.app.evidence import EvidenceGenerationError, OpenAIEvidenceClient, build_evidence_packet, evidence_status, load_evidence_config
from product.decisioning.optimizer import recommend_allocation
from product.decisioning.scenario import simulate_budget_plan
from product.evaluation import canonical_fingerprint, evaluate_all_horizons

PRODUCT_ROOT = Path(__file__).resolve().parents[1]


class PlannerService:
    """In-memory demo service; the production API can use these same domain functions."""

    def __init__(self, data_dir: Path, model_path: Path) -> None:
        self.canonical = canonicalize(read_source_files(data_dir))
        self.quality = validate_canonical(self.canonical)
        self.quality.raise_if_blocking()
        with model_path.open("rb") as handle:
            self.model: HorizonModel = pickle.load(handle)
        if not isinstance(self.model, HorizonModel):
            raise TypeError("Model artifact is not a HorizonModel")
        self.ledger = DecisionLedger(data_dir.parent / "output" / "horizon_decisions.sqlite")
        self._trust_report: dict[str, Any] | None = None

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        clean = frame.copy().replace({math.nan: None})
        return clean.to_dict(orient="records")

    def data_health(self) -> dict[str, Any]:
        meta = self.canonical[self.canonical["source_system"] == "meta_ads"]
        unknown_meta_types = int(meta["campaign_type"].eq("Generic").sum()) if not meta.empty else 0
        return {
            "status": "warning" if self.quality.warnings else "healthy",
            "rows": int(len(self.canonical)),
            "campaigns": int(self.canonical[["source_system", "source_campaign_id"]].drop_duplicates().shape[0]),
            "date_start": str(self.canonical["date"].min().date()),
            "date_end": str(self.canonical["date"].max().date()),
            "meta_taxonomy_unknown_rows": unknown_meta_types,
            "warnings": self.quality.warnings,
            "blockers": self.quality.blockers,
        }

    def _channel_budget_overrides(self, horizon_days: int, requested: Mapping[str, float]) -> dict[str, float]:
        baseline = build_forecast(self.model, self.canonical, horizon_days)
        leaves = baseline[baseline["level"] == "campaign"].copy()
        overrides: dict[str, float] = {}
        for channel, total in requested.items():
            amount = float(total)
            if amount < 0:
                raise ValueError(f"Channel budget cannot be negative: {channel}")
            subset = leaves[leaves["channel"] == channel]
            if subset.empty:
                raise ValueError(f"Unknown or inactive channel in scenario: {channel}")
            weights = subset["planned_budget"] / max(float(subset["planned_budget"].sum()), 1e-9)
            for campaign_key, weight in zip(subset["campaign_key"], weights, strict=True):
                overrides[str(campaign_key)] = amount * float(weight)
        return overrides

    @staticmethod
    def _evidence(result: pd.DataFrame, target_roas: float) -> dict[str, Any]:
        overall = result[result["level"] == "overall"].iloc[0]
        channels = result[result["level"] == "channel"].sort_values("predicted_revenue_p50", ascending=False)
        risks = result[result["quality_flags"].str.contains("sparse|extrapolation", case=False, na=False)]
        decision = "approve" if float(overall["probability_roas_above_target"]) >= 0.7 and float(overall["risk_score"]) < 45 else "revise_or_test"
        top_drivers = [
            {
                "channel": str(row.channel),
                "expected_revenue": round(float(row.predicted_revenue_p50), 2),
                "expected_roas": round(float(row.predicted_roas_p50), 2),
            }
            for row in channels.head(3).itertuples()
        ]
        risk_items = [str(value) for value in risks["quality_flags"].dropna().unique() if str(value)]
        return {
            "decision": decision,
            "target_roas": target_roas,
            "causal_status": "observational_association",
            "headline": (
                f"The plan has a {float(overall['probability_roas_above_target']):.0%} modeled probability "
                f"of achieving the {target_roas:.2f} ROAS guardrail."
            ),
            "facts": [
                f"Expected 60/90/30-day values are conditional forecasts from existing attributed performance.",
                f"Median revenue is {float(overall['predicted_revenue_p50']):,.0f} with a P10-P90 range of "
                f"{float(overall['predicted_revenue_p10']):,.0f} to {float(overall['predicted_revenue_p90']):,.0f}.",
            ],
            "drivers": top_drivers,
            "risks": risk_items or ["No campaign-level extrapolation flags were triggered."],
            "recommended_validation": (
                "Treat allocation changes as an observational forecast. Use a holdout, geo, or audience split before "
                "claiming incremental lift for a material budget move."
            ),
        }

    def forecast(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        horizon = int(payload.get("horizon_days", 60))
        target_roas = float(payload.get("target_roas", self.model.target_roas))
        if horizon not in {30, 60, 90}:
            raise ValueError("horizon_days must be one of 30, 60, or 90")
        if target_roas <= 0:
            raise ValueError("target_roas must be positive")
        channel_budgets = payload.get("channel_budgets", {})
        campaign_budgets = {str(key): float(value) for key, value in payload.get("campaign_budgets", {}).items()}
        if not isinstance(channel_budgets, Mapping):
            raise ValueError("channel_budgets must be an object")
        overrides = self._channel_budget_overrides(horizon, channel_budgets) if channel_budgets else {}
        overrides.update(campaign_budgets)
        result = simulate_budget_plan(self.model, self.canonical, horizon, overrides, target_roas)
        return {
            "data_health": self.data_health(),
            "evidence": self._evidence(result, target_roas),
            "overall": self._records(result[result["level"] == "overall"]),
            "channels": self._records(result[result["level"] == "channel"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaign_types": self._records(result[result["level"] == "campaign_type"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaigns": self._records(result[result["level"] == "campaign"].sort_values("predicted_revenue_p50", ascending=False)),
        }

    def optimize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        horizon = int(payload.get("horizon_days", 60))
        target_roas = float(payload.get("target_roas", self.model.target_roas))
        total_budget = float(payload["total_budget"])
        result = recommend_allocation(
            self.model,
            self.canonical,
            horizon,
            total_budget,
            target_roas,
            payload.get("channel_minimums", {}),
            payload.get("channel_maximums", {}),
        )
        forecast = result.forecast
        response = {
            "data_health": self.data_health(),
            "evidence": self._evidence(forecast, target_roas),
            "overall": self._records(forecast[forecast["level"] == "overall"]),
            "channels": self._records(forecast[forecast["level"] == "channel"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaign_types": self._records(forecast[forecast["level"] == "campaign_type"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaigns": self._records(forecast[forecast["level"] == "campaign"].sort_values("predicted_revenue_p50", ascending=False)),
            "optimization": {
                "status": result.status,
                "campaign_budgets": result.campaign_budgets,
                "target_constraint_status": result.target_constraint_status,
                "target_roas": result.target_roas,
                "achieved_roas_p50": result.achieved_roas_p50,
                "target_gap_p50": None if result.target_roas is None else result.achieved_roas_p50 - result.target_roas,
                "explanation": result.explanation,
            },
        }
        return response

    def trust_report(self) -> dict[str, Any]:
        if self._trust_report is None:
            report_path = PRODUCT_ROOT / "models" / "evaluation_report.json"
            expected_fingerprint = canonical_fingerprint(self.canonical)
            try:
                stored = json.loads(report_path.read_text(encoding="utf-8"))
                if stored.get("model_family") == self.model.model_version and stored.get("data_fingerprint") == expected_fingerprint:
                    self._trust_report = stored
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            if self._trust_report is None:
                self._trust_report = evaluate_all_horizons(self.canonical, folds=2)
        return self._trust_report

    def llm_status(self) -> dict[str, Any]:
        """Return safe configuration metadata; never expose credentials."""
        return evidence_status()

    def explain(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Generate a cited narrative after, and only after, deterministic forecasting.

        The fallback keeps the product useful offline and prevents the LLM from
        becoming either a prediction dependency or a demo failure point.
        """
        scenario = payload.get("scenario", payload)
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario must be a JSON object")
        forecast = self.forecast(scenario)
        overall = forecast["overall"][0]
        packet = build_evidence_packet(forecast["evidence"], overall)
        config = load_evidence_config()
        if config is None:
            return {
                "mode": "deterministic_fallback",
                "forecast_id": overall["forecast_id"],
                "deterministic_evidence": forecast["evidence"],
                "message": "AI narrative is not configured; the deterministic evidence brief remains available.",
            }
        try:
            brief = OpenAIEvidenceClient(config).generate(packet)
        except EvidenceGenerationError as exc:
            return {
                "mode": "deterministic_fallback",
                "forecast_id": overall["forecast_id"],
                "deterministic_evidence": forecast["evidence"],
                "message": str(exc),
            }
        return {
            "mode": "openai_grounded_narrative",
            "forecast_id": overall["forecast_id"],
            "model": config.model,
            "evidence_packet": packet,
            "brief": brief,
        }

    def record_decision(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "draft"))
        if action not in {"draft", "approved", "revised"}:
            raise ValueError("action must be draft, approved, or revised")
        scenario = payload.get("scenario", {})
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario must be an object")
        forecast = self.forecast(scenario)
        overall = forecast["overall"][0]
        summary = {
            "forecast_id": overall["forecast_id"],
            "horizon_days": overall["horizon_days"],
            "revenue_p50": overall["predicted_revenue_p50"],
            "roas_p50": overall["predicted_roas_p50"],
            "risk_score": overall["risk_score"],
            "decision": forecast["evidence"]["decision"],
        }
        return {"ledger": self.ledger.record(action, scenario, summary), "summary": summary}
