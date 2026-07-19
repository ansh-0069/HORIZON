from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from product.app.evidence import (
    EvidenceGenerationError,
    OpenAIEvidenceClient,
    build_evidence_packet,
    evidence_status,
    load_evidence_config,
)
from product.app.ledger import DecisionLedger
from product.decisioning.optimizer import recommend_allocation
from product.decisioning.scenario import simulate_budget_plan
from product.evaluation import canonical_fingerprint
from src.canonicalize import canonicalize
from src.forecast import build_forecast
from src.ingest import read_source_files
from src.model import HorizonModel
from src.predict import load_model
from src.validate import validate_canonical


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_HORIZONS = frozenset({30, 60, 90})

# The planner may show a forecast whenever inference succeeds, but it must not
# present an approval posture when the persisted validation evidence is too weak
# to support that decision. These are deliberately conservative MVP gates, not
# claims of statistical certification.
MINIMUM_BACKTEST_FOLDS = 5
MINIMUM_CALIBRATION_SAMPLES = 30
MINIMUM_REVENUE_INTERVAL_COVERAGE = 0.60
MINIMUM_ROAS_INTERVAL_COVERAGE = 0.50
MINIMUM_CAMPAIGN_REVENUE_INTERVAL_COVERAGE = 0.50
MINIMUM_CAMPAIGN_ROAS_INTERVAL_COVERAGE = 0.40
MAXIMUM_CAMPAIGN_MISSING_FORECAST_RATE = 0.10
MAXIMUM_REVENUE_WAPE = 0.75
MAXIMUM_TARGET_PROBABILITY_BRIER = 0.25
MAXIMUM_TARGET_PROBABILITY_ECE = 0.20


def _require_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_finite_number(
    value: Any,
    field: str,
    *,
    strictly_positive: bool = False,
    non_negative: bool = False,
) -> float:
    """Accept only JSON numeric scalars with a finite, explicit value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    if strictly_positive and parsed <= 0:
        raise ValueError(f"{field} must be positive")
    if non_negative and parsed < 0:
        raise ValueError(f"{field} cannot be negative")
    return parsed


def _require_horizon(value: Any) -> int:
    """Validate an exact supported integer horizon without lossy coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("horizon_days must be an integer JSON number: 30, 60, or 90")
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError("horizon_days must be an exact integer: 30, 60, or 90")
    horizon = int(parsed)
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError("horizon_days must be one of 30, 60, or 90")
    return horizon


def _validate_budget_mapping(value: Any, field: str) -> dict[str, float]:
    payload = _require_object(value, field)
    parsed: dict[str, float] = {}
    for key, amount in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field} keys must be non-empty strings")
        parsed[key] = _require_finite_number(amount, f"{field}.{key}", non_negative=True)
    return parsed


def _reject_unexpected_fields(payload: Mapping[str, Any], *, allowed: set[str], field: str) -> None:
    unexpected = sorted(str(key) for key in set(payload) - allowed)
    if unexpected:
        raise ValueError(f"{field} contains unsupported fields: {', '.join(unexpected)}")


class PlannerService:
    """In-memory local planner service, intentionally outside the evaluator path.

    This service reads a pre-trained artifact through the same integrity-checked
    loader as the protected runner. It never trains or backtests at request time.
    """

    def __init__(self, data_dir: Path, model_path: Path, *, allow_live_llm: bool = False) -> None:
        self.canonical = canonicalize(read_source_files(data_dir))
        self.quality = validate_canonical(self.canonical)
        self.quality.raise_if_blocking()
        self.model: HorizonModel = load_model(model_path)
        self.allow_live_llm = bool(allow_live_llm)
        self.ledger = DecisionLedger(data_dir.parent / "output" / "horizon_decisions.sqlite")
        self._trust_report: dict[str, Any] | None = None

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        clean = frame.copy().replace({math.nan: None})
        return clean.to_dict(orient="records")

    def _meta_revenue_semantics(self) -> tuple[str, str]:
        """Describe Meta semantics from canonical provenance flags, never by UI default."""
        meta = self.canonical[self.canonical["source_system"] == "meta_ads"]
        if meta.empty:
            return "not_applicable", "No Meta Ads rows are present in the current validated dataset."
        flags = meta["quality_flags"].fillna("").astype(str).str.lower()
        reviewed = flags.str.contains("meta_conversion_semantics_reviewed", na=False)
        proxy = flags.str.contains("meta_conversion_treated_as_attributed_revenue", na=False)
        if bool(reviewed.all()):
            return (
                "reviewed",
                "Meta `conversion` semantics are documented as reviewed for this dataset; they remain platform-attributed, not causal lift.",
            )
        if bool(proxy.any()):
            return (
                "assumed_proxy",
                "Meta `conversion` is treated as a platform-attributed revenue proxy because reviewed revenue semantics are not available for every Meta row.",
            )
        return (
            "unknown",
            "Meta revenue semantics could not be derived from the canonical quality flags; review the source semantic manifest before approval.",
        )

    def data_health(self) -> dict[str, Any]:
        meta = self.canonical[self.canonical["source_system"] == "meta_ads"]
        unknown_meta_types = int(meta["campaign_type"].eq("Generic").sum()) if not meta.empty else 0
        meta_status, meta_assumption = self._meta_revenue_semantics()
        planning_defaults = [
            warning for warning in self.quality.warnings if warning.startswith("missing configured budget rows=")
        ]
        assumptions = [
            meta_assumption,
            "Platform attribution is the source of truth; Horizon does not rebuild attribution or claim causal lift.",
            "Forecast intervals are empirical residual ranges, not guarantees.",
        ]
        return {
            "status": "warning" if self.quality.warnings else "healthy",
            "rows": int(len(self.canonical)),
            "campaigns": int(self.canonical[["source_system", "source_campaign_id"]].drop_duplicates().shape[0]),
            "date_start": str(self.canonical["date"].min().date()),
            "date_end": str(self.canonical["date"].max().date()),
            "meta_taxonomy_unknown_rows": unknown_meta_types,
            "meta_revenue_semantics_status": meta_status,
            "meta_revenue_assumption": meta_assumption,
            "assumptions": assumptions,
            "planning_defaults": planning_defaults,
            "warnings": self.quality.warnings,
            "blockers": self.quality.blockers,
        }

    def _baseline_leaves(self, horizon_days: int) -> pd.DataFrame:
        baseline = build_forecast(self.model, self.canonical, horizon_days)
        return baseline[baseline["level"] == "campaign"].copy()

    def _channel_budget_overrides(self, horizon_days: int, requested: Mapping[str, float]) -> dict[str, float]:
        leaves = self._baseline_leaves(horizon_days)
        overrides: dict[str, float] = {}
        for channel, amount in requested.items():
            subset = leaves[leaves["channel"] == channel]
            if subset.empty:
                raise ValueError(f"Unknown or inactive channel in scenario: {channel}")
            weights = subset["planned_budget"] / max(float(subset["planned_budget"].sum()), 1e-9)
            for campaign_key, weight in zip(subset["campaign_key"], weights, strict=True):
                overrides[str(campaign_key)] = amount * float(weight)
        return overrides

    def _validate_campaign_keys(self, horizon_days: int, campaign_budgets: Mapping[str, float]) -> None:
        if not campaign_budgets:
            return
        known = set(self._baseline_leaves(horizon_days)["campaign_key"].astype(str))
        unknown = sorted(set(campaign_budgets) - known)
        if unknown:
            raise ValueError(
                "campaign_budgets must use active source-qualified campaign keys; unknown keys: "
                + ", ".join(unknown[:3])
            )

    def _parse_forecast_payload(self, payload: Any) -> dict[str, Any]:
        request = _require_object(payload, "scenario")
        _reject_unexpected_fields(
            request,
            allowed={"horizon_days", "target_roas", "channel_budgets", "campaign_budgets"},
            field="scenario",
        )
        horizon = _require_horizon(request.get("horizon_days", 30))
        target_roas = _require_finite_number(
            request.get("target_roas", self.model.target_roas), "target_roas", strictly_positive=True
        )
        channel_budgets = _validate_budget_mapping(request.get("channel_budgets", {}), "channel_budgets")
        campaign_budgets = _validate_budget_mapping(request.get("campaign_budgets", {}), "campaign_budgets")
        self._validate_campaign_keys(horizon, campaign_budgets)
        return {
            "horizon_days": horizon,
            "target_roas": target_roas,
            "channel_budgets": channel_budgets,
            "campaign_budgets": campaign_budgets,
        }

    def _parse_optimization_payload(self, payload: Any) -> dict[str, Any]:
        request = _require_object(payload, "optimization request")
        _reject_unexpected_fields(
            request,
            allowed={"horizon_days", "target_roas", "total_budget", "channel_minimums", "channel_maximums"},
            field="optimization request",
        )
        if "total_budget" not in request:
            raise ValueError("total_budget is required for optimization")
        return {
            "horizon_days": _require_horizon(request.get("horizon_days", 30)),
            "target_roas": _require_finite_number(
                request.get("target_roas", self.model.target_roas), "target_roas", strictly_positive=True
            ),
            "total_budget": _require_finite_number(request["total_budget"], "total_budget", strictly_positive=True),
            "channel_minimums": _validate_budget_mapping(request.get("channel_minimums", {}), "channel_minimums"),
            "channel_maximums": _validate_budget_mapping(request.get("channel_maximums", {}), "channel_maximums"),
        }

    def _not_applicable_trust_report(self, reason: str, expected_fingerprint: str) -> dict[str, Any]:
        return {
            "status": "not_applicable",
            "reason": reason,
            "model_family": self.model.model_version,
            "data_fingerprint": expected_fingerprint,
            "horizons": [],
            "baseline_horizons": [],
            "evaluation_performed_at_request": False,
        }

    def trust_report(self) -> dict[str, Any]:
        """Return only a matching persisted report; never run backtesting in a request."""
        if self._trust_report is not None:
            return self._trust_report
        report_path = PRODUCT_ROOT / "models" / "evaluation_report.json"
        expected_fingerprint = canonical_fingerprint(self.canonical)
        try:
            stored = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._trust_report = self._not_applicable_trust_report(
                f"Persisted evaluation report is unavailable or unreadable: {report_path.name}.", expected_fingerprint
            )
            return self._trust_report
        if not isinstance(stored, dict):
            self._trust_report = self._not_applicable_trust_report(
                "Persisted evaluation report is not a JSON object.", expected_fingerprint
            )
            return self._trust_report
        artifact_provenance = stored.get("artifact_provenance")
        if not isinstance(artifact_provenance, Mapping):
            self._trust_report = self._not_applicable_trust_report(
                "Persisted evaluation report is not bound to a reviewed model artifact.", expected_fingerprint
            )
            return self._trust_report
        if stored.get("data_fingerprint") != expected_fingerprint:
            self._trust_report = self._not_applicable_trust_report(
                "Persisted evaluation report data fingerprint does not match the current validated dataset.", expected_fingerprint
            )
            return self._trust_report
        expected_artifact = {
            "artifact_sha256": str(getattr(self.model, "artifact_sha256", "") or ""),
            "model_version": str(self.model.model_version),
            "training_data_fingerprint": str(getattr(self.model, "training_data_fingerprint", "") or expected_fingerprint),
            "feature_schema_fingerprint": str(getattr(self.model, "feature_schema_fingerprint", "") or ""),
        }
        provenance_mismatch = [
            field
            for field, expected in expected_artifact.items()
            if str(artifact_provenance.get(field, "")) != expected
        ]
        if provenance_mismatch:
            self._trust_report = self._not_applicable_trust_report(
                "Persisted evaluation report artifact provenance does not match the loaded model: "
                + ", ".join(provenance_mismatch),
                expected_fingerprint,
            )
            return self._trust_report
        if not isinstance(stored.get("horizons"), list):
            self._trust_report = self._not_applicable_trust_report(
                "Persisted evaluation report has no horizon results.", expected_fingerprint
            )
            return self._trust_report
        self._trust_report = {
            **stored,
            "status": "available",
            "evaluation_performed_at_request": False,
        }
        return self._trust_report

    @staticmethod
    def _report_metric(report: Mapping[str, Any], field: str) -> float | None:
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    def calibration_posture(self, horizon_days: int) -> dict[str, Any]:
        """Fail closed for decision approval when persisted calibration is weak.

        A report can be shown as unavailable rather than fabricated by running a
        slow backtest on request. The 60-day product decision is therefore
        explicitly blocked when its persisted evidence is missing, stale, small,
        or materially under-covering.
        """
        report = self.trust_report()
        if report.get("status") != "available":
            return {
                "horizon_days": horizon_days,
                "status": "unavailable",
                "reasons": [str(report.get("reason") or "No matching persisted evaluation report is available.")],
                "report_status": str(report.get("status") or "not_applicable"),
            }
        horizon_report = next(
            (
                item
                for item in report.get("horizons", [])
                if isinstance(item, Mapping) and item.get("horizon_days") == horizon_days
            ),
            None,
        )
        if horizon_report is None:
            return {
                "horizon_days": horizon_days,
                "status": "unavailable",
                "reasons": [f"Persisted evaluation report has no {horizon_days}-day result."],
                "report_status": "available",
            }
        folds = self._report_metric(horizon_report, "folds")
        samples = self._report_metric(horizon_report, "median_calibration_samples")
        revenue_coverage = self._report_metric(horizon_report, "revenue_interval_coverage")
        roas_coverage = self._report_metric(horizon_report, "roas_interval_coverage")
        revenue_wape = self._report_metric(horizon_report, "revenue_wape")
        reasons: list[str] = []
        if folds is None or folds < MINIMUM_BACKTEST_FOLDS:
            reasons.append(f"{horizon_days}-day backtest has fewer than {MINIMUM_BACKTEST_FOLDS} folds.")
        if samples is None or samples < MINIMUM_CALIBRATION_SAMPLES:
            reasons.append(
                f"{horizon_days}-day residual calibration has fewer than {MINIMUM_CALIBRATION_SAMPLES} median samples."
            )
        if revenue_coverage is None or revenue_coverage < MINIMUM_REVENUE_INTERVAL_COVERAGE:
            reasons.append(
                f"{horizon_days}-day observed revenue interval coverage is below the {MINIMUM_REVENUE_INTERVAL_COVERAGE:.0%} decision minimum."
            )
        if roas_coverage is None or roas_coverage < MINIMUM_ROAS_INTERVAL_COVERAGE:
            reasons.append(
                f"{horizon_days}-day observed ROAS interval coverage is below the {MINIMUM_ROAS_INTERVAL_COVERAGE:.0%} decision minimum."
            )
        if revenue_wape is None or revenue_wape > MAXIMUM_REVENUE_WAPE:
            reasons.append(
                f"{horizon_days}-day revenue WAPE exceeds the {MAXIMUM_REVENUE_WAPE:.0%} decision maximum."
            )

        hierarchy = horizon_report.get("coverage_by_hierarchy")
        campaign_metrics: dict[str, Any] = {}
        if not isinstance(hierarchy, Mapping) or not isinstance(hierarchy.get("campaign"), Mapping):
            reasons.append("Campaign-level coverage evidence is unavailable for a campaign-level allocation decision.")
        else:
            campaign = hierarchy["campaign"]
            campaign_revenue_coverage = self._report_metric(campaign, "revenue_interval_coverage")
            campaign_roas_coverage = self._report_metric(campaign, "roas_interval_coverage")
            missing_forecasts = self._report_metric(campaign, "missing_forecast_observations")
            evaluable = self._report_metric(campaign, "revenue_coverage_observations")
            missing_rate = (
                missing_forecasts / evaluable
                if missing_forecasts is not None and evaluable is not None and evaluable > 0
                else None
            )
            campaign_metrics = {
                "revenue_interval_coverage": campaign_revenue_coverage,
                "roas_interval_coverage": campaign_roas_coverage,
                "missing_forecast_rate": missing_rate,
                "missing_forecast_observations": None if missing_forecasts is None else int(missing_forecasts),
            }
            if campaign_revenue_coverage is None or campaign_revenue_coverage < MINIMUM_CAMPAIGN_REVENUE_INTERVAL_COVERAGE:
                reasons.append(
                    f"{horizon_days}-day campaign revenue interval coverage is below the {MINIMUM_CAMPAIGN_REVENUE_INTERVAL_COVERAGE:.0%} decision minimum."
                )
            if campaign_roas_coverage is None or campaign_roas_coverage < MINIMUM_CAMPAIGN_ROAS_INTERVAL_COVERAGE:
                reasons.append(
                    f"{horizon_days}-day campaign ROAS interval coverage is below the {MINIMUM_CAMPAIGN_ROAS_INTERVAL_COVERAGE:.0%} decision minimum."
                )
            if missing_rate is None or missing_rate > MAXIMUM_CAMPAIGN_MISSING_FORECAST_RATE:
                reasons.append(
                    f"{horizon_days}-day campaign forecast coverage has more than the {MAXIMUM_CAMPAIGN_MISSING_FORECAST_RATE:.0%} permitted missing-outcome rate."
                )

        probability_reliability = horizon_report.get("roas_target_probability_reliability")
        probability_metrics: dict[str, Any] = {}
        if not isinstance(probability_reliability, Mapping):
            reasons.append("ROAS-target draw-share reliability evidence is unavailable.")
        else:
            probability_status = str(probability_reliability.get("status") or "unavailable")
            brier = self._report_metric(probability_reliability, "brier_score")
            ece = self._report_metric(probability_reliability, "expected_calibration_error")
            probability_metrics = {
                "status": probability_status,
                "observations": self._report_metric(probability_reliability, "observations"),
                "brier_score": brier,
                "expected_calibration_error": ece,
            }
            if probability_status != "evaluated":
                reasons.append("ROAS-target draw share lacks enough independent historical events for approval use.")
            elif brier is None or brier > MAXIMUM_TARGET_PROBABILITY_BRIER:
                reasons.append(
                    f"ROAS-target probability Brier score exceeds the {MAXIMUM_TARGET_PROBABILITY_BRIER:.2f} decision maximum."
                )
            elif ece is None or ece > MAXIMUM_TARGET_PROBABILITY_ECE:
                reasons.append(
                    f"ROAS-target probability calibration error exceeds the {MAXIMUM_TARGET_PROBABILITY_ECE:.0%} decision maximum."
                )
        metrics = {
            "folds": None if folds is None else int(folds),
            "median_calibration_samples": None if samples is None else int(samples),
            "revenue_interval_coverage": revenue_coverage,
            "roas_interval_coverage": roas_coverage,
            "revenue_wape": revenue_wape,
            "nominal_interval_coverage": self._report_metric(horizon_report, "nominal_interval_coverage"),
            "campaign": campaign_metrics,
            "target_probability": probability_metrics,
        }
        return {
            "horizon_days": horizon_days,
            "status": "sufficient" if not reasons else "insufficient",
            "reasons": reasons,
            "metrics": metrics,
            "report_status": "available",
        }

    def _evidence(self, result: pd.DataFrame, target_roas: float, horizon_days: int) -> dict[str, Any]:
        overall = result[result["level"] == "overall"].iloc[0]
        channels = result[result["level"] == "channel"].sort_values("predicted_revenue_p50", ascending=False)
        risks = result[result["quality_flags"].str.contains("sparse|extrapolation", case=False, na=False)]
        model_guardrail_passed = (
            float(overall["probability_roas_above_target"]) >= 0.7 and float(overall["risk_score"]) < 45
        )
        calibration = self.calibration_posture(horizon_days)
        calibration_allows_approval = calibration["status"] == "sufficient"
        decision = "approve" if model_guardrail_passed and calibration_allows_approval else "revise_or_test"
        top_drivers = [
            {
                "channel": str(row.channel),
                "expected_revenue": round(float(row.predicted_revenue_p50), 2),
                "expected_roas": round(float(row.predicted_roas_p50), 2),
            }
            for row in channels.head(3).itertuples()
        ]
        risk_items = [str(value) for value in risks["quality_flags"].dropna().unique() if str(value)]
        if not calibration_allows_approval:
            risk_items.extend(f"Calibration gate: {reason}" for reason in calibration["reasons"])
        probability_status = calibration.get("metrics", {}).get("target_probability", {}).get("status")
        headline = (
            f"The plan has a {float(overall['probability_roas_above_target']):.0%} simulated ROAS-target draw share "
            f"at the {target_roas:.2f} guardrail."
        )
        if probability_status != "evaluated":
            headline += " That draw share is not yet acceptance-calibrated and cannot approve a material allocation alone."
        if not calibration_allows_approval:
            headline = (
                f"The decision posture is revise or test: the persisted {horizon_days}-day calibration evidence "
                f"is {calibration['status']}."
            )
        return {
            "decision": decision,
            "target_roas": target_roas,
            "causal_status": "observational_association",
            "headline": headline,
            "decision_gates": {
                "model_guardrail_passed": model_guardrail_passed,
                "calibration": calibration,
            },
            "facts": [
                f"Expected {horizon_days}-day values are conditional forecasts from existing attributed performance.",
                f"Median revenue is {float(overall['predicted_revenue_p50']):,.0f} with a P10-P90 range of "
                f"{float(overall['predicted_revenue_p10']):,.0f} to {float(overall['predicted_revenue_p90']):,.0f}.",
            ],
            "drivers": top_drivers,
            "risks": list(dict.fromkeys(risk_items)) or ["No campaign-level extrapolation flags were triggered."],
            "recommended_validation": (
                "Treat allocation changes as an observational forecast. Use a holdout, geo, or audience split before "
                "claiming incremental lift for a material budget move."
            ),
        }

    def forecast(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        scenario = self._parse_forecast_payload(payload)
        horizon = scenario["horizon_days"]
        target_roas = scenario["target_roas"]
        overrides = self._channel_budget_overrides(horizon, scenario["channel_budgets"]) if scenario["channel_budgets"] else {}
        overrides.update(scenario["campaign_budgets"])
        result = simulate_budget_plan(self.model, self.canonical, horizon, overrides, target_roas)
        effective_scenario = {
            "horizon_days": horizon,
            "target_roas": target_roas,
            "campaign_budgets": dict(sorted(overrides.items())),
        }
        return {
            "scenario": effective_scenario,
            "data_health": self.data_health(),
            "evidence": self._evidence(result, target_roas, horizon),
            "overall": self._records(result[result["level"] == "overall"]),
            "channels": self._records(result[result["level"] == "channel"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaign_types": self._records(result[result["level"] == "campaign_type"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaigns": self._records(result[result["level"] == "campaign"].sort_values("predicted_revenue_p50", ascending=False)),
        }

    def optimize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self._parse_optimization_payload(payload)
        horizon = request["horizon_days"]
        target_roas = request["target_roas"]
        result = recommend_allocation(
            self.model,
            self.canonical,
            horizon,
            request["total_budget"],
            target_roas,
            request["channel_minimums"],
            request["channel_maximums"],
        )
        forecast = result.forecast
        campaign_budgets = dict(sorted((str(key), float(value)) for key, value in result.campaign_budgets.items()))
        response = {
            # This is the authoritative, exact plan used to create ``forecast``.
            # Clients must persist and brief this campaign-level scenario, rather
            # than reconstructing it from rounded channel totals.
            "scenario": {
                "horizon_days": horizon,
                "target_roas": target_roas,
                "campaign_budgets": campaign_budgets,
            },
            "data_health": self.data_health(),
            "evidence": self._evidence(forecast, target_roas, horizon),
            "overall": self._records(forecast[forecast["level"] == "overall"]),
            "channels": self._records(forecast[forecast["level"] == "channel"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaign_types": self._records(forecast[forecast["level"] == "campaign_type"].sort_values("predicted_revenue_p50", ascending=False)),
            "campaigns": self._records(forecast[forecast["level"] == "campaign"].sort_values("predicted_revenue_p50", ascending=False)),
            "optimization": {
                "status": result.status,
                "campaign_budgets": campaign_budgets,
                "target_constraint_status": result.target_constraint_status,
                "target_roas": result.target_roas,
                "achieved_roas_p50": result.achieved_roas_p50,
                "target_gap_p50": None if result.target_roas is None else result.achieved_roas_p50 - result.target_roas,
                "explanation": result.explanation,
                "allocation_method": getattr(result, "allocation_method", "conservative_monotone_concave_test_priority_v1"),
                "causal_status": getattr(result, "causal_status", "observational_association"),
                "validation_required": bool(getattr(result, "validation_required", True)),
                "eligible_campaign_count": int(getattr(result, "eligible_campaign_count", 0)),
                "guardrailed_campaign_count": int(getattr(result, "guardrailed_campaign_count", 0)),
            },
        }
        return response

    def llm_status(self) -> dict[str, Any]:
        """Return safe configuration metadata; never expose credentials."""
        enabled = bool(getattr(self, "allow_live_llm", False))
        return {
            **evidence_status(),
            "default_mode": "deterministic_evidence_brief",
            "live_narration_requires_explicit_request": True,
            "live_narration_server_enabled": enabled,
            "visible_demo_control_uses_live_narration": False,
            "prediction_dependency": False,
        }

    @staticmethod
    def _deterministic_brief(evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Shape sealed forecast evidence into the UI brief schema without an LLM."""
        facts = list(evidence.get("facts") or [])
        if not facts:
            facts = [
                str(evidence.get("headline") or "Forecast values are conditional on supplied attribution and the selected budget plan.")
            ]
        return {
            "decision": str(evidence.get("decision") or "revise_or_test"),
            "causal_status": str(evidence.get("causal_status") or "observational_association"),
            "headline": str(
                evidence.get("headline")
                or "Decision brief generated from sealed forecast evidence without a live language model."
            ),
            "facts": [{"text": str(text), "evidence_ids": ["forecast_guardrail"]} for text in facts[:3]],
            "assumptions": [
                {
                    "text": "Existing platform attribution is treated as truth; budget effects are observational associations.",
                    "evidence_ids": ["causal_boundary"],
                }
            ],
            "recommendations": [
                {
                    "text": str(
                        evidence.get("recommended_validation")
                        or "Validate material budget moves with a holdout or geo test."
                    ),
                    "evidence_ids": ["validation_plan"],
                }
            ],
            "limitations": [
                {"text": str(risk), "evidence_ids": ["risk_flags"]}
                for risk in list(evidence.get("risks") or [])[:3]
            ],
        }

    def explain(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a sealed evidence brief after deterministic forecasting.

        A networked narrator remains impossible unless the localhost server was
        explicitly started with ``--enable-live-llm`` and the caller separately
        requests it with a JSON boolean. The default never needs an API key.
        """
        request = _require_object(payload, "evidence request")
        _reject_unexpected_fields(request, allowed={"scenario", "prefer_live_llm"}, field="evidence request")
        scenario = request.get("scenario", {})
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario must be a JSON object")
        forecast = self.forecast(scenario)
        overall = forecast["overall"][0]
        evidence = forecast["evidence"]
        packet = build_evidence_packet(evidence, overall)
        effective_scenario = forecast.get("scenario", dict(scenario))
        deterministic = {
            "mode": "deterministic_evidence_brief",
            "forecast_id": overall["forecast_id"],
            "scenario": effective_scenario,
            "deterministic_evidence": evidence,
            "brief": self._deterministic_brief(evidence),
            "message": "Deterministic evidence brief from sealed forecast numbers. No live LLM call was made.",
        }
        requested_live_narration = request.get("prefer_live_llm", False)
        if not isinstance(requested_live_narration, bool):
            raise ValueError("prefer_live_llm must be a JSON boolean when provided")
        if requested_live_narration is not True:
            return deterministic
        if not bool(getattr(self, "allow_live_llm", False)):
            return {
                **deterministic,
                "mode": "deterministic_fallback",
                "message": "Live LLM narration is disabled by this local server; deterministic evidence brief remains available.",
            }
        config = load_evidence_config()
        if config is None:
            return {
                **deterministic,
                "mode": "deterministic_fallback",
                "message": "Live LLM requested but not configured; deterministic evidence brief remains available.",
            }
        try:
            brief = OpenAIEvidenceClient(config).generate(packet)
        except EvidenceGenerationError as exc:
            return {
                **deterministic,
                "mode": "deterministic_fallback",
                "message": str(exc),
            }
        return {
            "mode": "openai_grounded_narrative",
            "forecast_id": overall["forecast_id"],
            "scenario": effective_scenario,
            "model": config.model,
            "evidence_packet": packet,
            "brief": brief,
            "message": "Optional live narrative generated from the sealed evidence packet only.",
        }

    def record_decision(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = _require_object(payload, "decision request")
        _reject_unexpected_fields(request, allowed={"action", "scenario"}, field="decision request")
        action = request.get("action", "draft")
        if not isinstance(action, str) or action not in {"draft", "approved", "revised"}:
            raise ValueError("action must be draft, approved, or revised")
        scenario = request.get("scenario", {})
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario must be a JSON object")
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
        effective_scenario = forecast["scenario"]
        return {
            "ledger": self.ledger.record(action, effective_scenario, summary),
            "scenario": effective_scenario,
            "summary": summary,
        }
