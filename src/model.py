from __future__ import annotations

from dataclasses import dataclass, field
import math
import pandas as pd
from src.direct_model import DirectRidgeModel, inference_features


@dataclass
class HorizonModel:
    model_version: str
    global_roas: float
    global_log_sigma: float
    month_roas_factors: dict[int, float]
    direct_models: dict[int, DirectRidgeModel]
    min_history_days: int = 28
    target_roas: float = 4.0
    artifact_sha256: str = ""
    # Provenance fields are data-only and therefore safe for the protected
    # pickle loader. They bind the shipped artifact to the canonical training
    # contract and one-hot feature schema without importing product code.
    training_data_fingerprint: str = ""
    feature_schema_fingerprint: str = ""
    # Trained outside the protected evaluator path.  The field is deliberately
    # optional so artifacts created before dependence calibration remain
    # loadable; inference uses an explicit independent fallback for those
    # legacy artifacts rather than reintroducing perfect rank correlation.
    residual_dependence: dict[int, dict[str, object]] = field(default_factory=dict)

    def __getattr__(self, name: str) -> object:
        """Supply defaults for fields absent from pre-v4 pickle artifacts."""
        if name == "residual_dependence":
            return {}
        if name in {"training_data_fingerprint", "feature_schema_fingerprint"}:
            return ""
        raise AttributeError(name)

    def dependence_for_horizon(self, horizon_days: int) -> dict[str, object]:
        """Return the persisted residual-dependence profile for a horizon.

        Pickle restores older ``HorizonModel`` instances without newly added
        dataclass attributes.  ``getattr`` is therefore intentional: model
        compatibility is part of the offline submission contract.  Profiles
        are data-only dictionaries so they can be safely checked and evolved
        without importing training-only code during inference.
        """
        profiles = getattr(self, "residual_dependence", {})
        if not isinstance(profiles, dict):
            return {}
        profile = profiles.get(horizon_days, profiles.get(str(horizon_days), {}))
        return dict(profile) if isinstance(profile, dict) else {}

    @staticmethod
    def campaign_key(source_system: object, campaign_id: object) -> str:
        """Return the source-qualified campaign identity used by scenarios."""
        return f"{source_system}:{campaign_id}"

    @staticmethod
    def _seasonal_default_budget_multiplier(
        canonical: pd.DataFrame,
        as_of: pd.Timestamp,
        horizon_days: int,
    ) -> float:
        """Estimate a conservative portfolio plan when no future budget is supplied.

        The evaluator provides historical performance, not a future media plan.
        A 90-day plan based only on the preceding 28 days can carry a holiday
        spike into normal trading months. For that horizon only, compare the
        future calendar-month mix with the recent daily delivery rate. This is
        a deterministic *budget default*, not a revenue feature or an override
        of a planner-provided scenario.
        """
        if horizon_days != 90:
            return 1.0
        recent_start = as_of - pd.Timedelta(days=27)
        recent = canonical[(canonical["date"] >= recent_start) & (canonical["date"] <= as_of)]
        recent_daily_spend = float(recent["spend"].sum()) / 28.0
        if recent_daily_spend <= 0:
            return 1.0
        portfolio_daily = canonical[canonical["date"] <= as_of].groupby("date")["spend"].sum()
        if portfolio_daily.empty:
            return 1.0
        month_daily_mean = portfolio_daily.groupby(portfolio_daily.index.month).mean()
        future_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        seasonal_daily_spend = sum(float(month_daily_mean.get(date.month, recent_daily_spend)) for date in future_dates) / horizon_days
        # Protect sparse or regime-shifted inputs while still allowing a clear
        # seasonal correction when historical coverage supports it.
        return max(0.10, min(2.00, seasonal_daily_spend / recent_daily_spend))

    def direct_quantiles(
        self,
        history: pd.DataFrame,
        channel: str,
        campaign_type: str,
        as_of: pd.Timestamp,
        horizon_days: int,
        planned_budget: float,
    ) -> tuple[float, float, float] | None:
        """Evaluate the exact direct-ridge response used by forecast inference."""
        direct_model = self.direct_models.get(horizon_days)
        if direct_model is None:
            return None
        direct_features = inference_features(history, channel, campaign_type, as_of, horizon_days, planned_budget)
        if direct_model.unknown_category_values(direct_features):
            return None
        return tuple(float(value[0]) for value in direct_model.predict(direct_features))

    def direct_category_oov(
        self,
        history: pd.DataFrame,
        channel: str,
        campaign_type: str,
        as_of: pd.Timestamp,
        horizon_days: int,
        planned_budget: float,
    ) -> bool:
        """Return whether a direct-model feature row has unseen categories."""
        direct_model = self.direct_models.get(horizon_days)
        if direct_model is None:
            return False
        direct_features = inference_features(history, channel, campaign_type, as_of, horizon_days, planned_budget)
        return bool(direct_model.unknown_category_values(direct_features))

    def forecast_campaigns(self, canonical: pd.DataFrame, horizon_days: int, budget_overrides: dict[str, float] | None = None) -> pd.DataFrame:
        budget_overrides = budget_overrides or {}
        provided_override_keys = {str(key) for key in budget_overrides}
        consumed_override_keys: set[str] = set()
        as_of = canonical["date"].max()
        window_start = as_of - pd.Timedelta(days=27)
        recent = canonical[(canonical["date"] >= window_start) & (canonical["date"] <= as_of)].copy()
        portfolio_has_active_delivery = bool((recent["spend"] > 0).any())
        future_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        seasonal_factor = sum(self.month_roas_factors.get(int(date.month), 1.0) for date in future_dates) / horizon_days
        default_budget_multiplier = self._seasonal_default_budget_multiplier(canonical, as_of, horizon_days)
        group_keys = ["source_system", "source_campaign_id", "channel", "campaign_type", "campaign_name"]
        source_counts = canonical.groupby("source_campaign_id", dropna=False)["source_system"].nunique()
        # Reject ambiguity before excluding inactive campaigns. Otherwise an
        # invalid legacy override could appear to work merely because the first
        # colliding campaign has zero recent spend and is skipped below.
        ambiguous_legacy_ids = [
            str(campaign_id)
            for campaign_id, source_count in source_counts.items()
            if int(source_count) > 1 and str(campaign_id) in budget_overrides
        ]
        if ambiguous_legacy_ids:
            raise ValueError(f"Ambiguous unqualified campaign budget override: {sorted(ambiguous_legacy_ids)[0]}")
        rows: list[dict[str, object]] = []
        for key, history in canonical.groupby(group_keys, dropna=False):
            source, campaign_id, channel, campaign_type, campaign_name = key
            campaign_recent = recent[(recent["source_system"] == source) & (recent["source_campaign_id"] == campaign_id)]
            active_days = int(campaign_recent["date"].nunique())
            recent_spend = float(campaign_recent["spend"].sum())
            recent_revenue = float(campaign_recent["revenue"].sum())
            # Historical campaigns that are not currently delivering should not
            # silently receive a future budget in a baseline media plan.  If
            # *every* campaign is dormant, however, retain the hierarchy with
            # an explicit zero-plan fallback so a valid zero-delivery upload
            # still produces a deterministic predictions.csv instead of an
            # avoidable empty-forecast error.
            if recent_spend <= 0 and portfolio_has_active_delivery:
                continue
            historic_spend = float(history["spend"].sum())
            historic_revenue = float(history["revenue"].sum())
            historic_roas = historic_revenue / max(historic_spend, 1e-9)
            recent_roas = recent_revenue / max(recent_spend, 1e-9) if recent_spend > 0 else historic_roas
            reliability = min(1.0, active_days / self.min_history_days)
            roas = reliability * recent_roas + (1.0 - reliability) * historic_roas
            roas = (0.70 * roas + 0.30 * self.global_roas) * seasonal_factor
            daily_spend = recent_spend / max(active_days, 1)
            campaign_key = self.campaign_key(source, campaign_id)
            override = budget_overrides.get(campaign_key)
            override_key = campaign_key if override is not None else None
            # Legacy unqualified IDs remain accepted only when unambiguous. New
            # product callers must use source-qualified campaign keys.
            if override is None and str(campaign_id) in budget_overrides:
                override = budget_overrides[str(campaign_id)]
                override_key = str(campaign_id)
            if override_key is not None:
                consumed_override_keys.add(override_key)
            baseline_budget = (
                max(0.0, daily_spend * horizon_days * default_budget_multiplier)
                if recent_spend > 0
                else 0.0
            )
            planned_budget = float(override) if override is not None else baseline_budget
            support = max(historic_spend / max(int(history["date"].nunique()), 1) * horizon_days, 1.0)
            extrapolation = planned_budget > 1.5 * support
            sigma = self.global_log_sigma * (1.0 + (1.0 - reliability) + (0.55 if extrapolation else 0.0))
            flags = []
            if active_days < self.min_history_days:
                flags.append("sparse_recent_history")
            if recent_spend <= 0:
                flags.append("portfolio_dormant_zero_plan")
            if extrapolation:
                flags.append("budget_extrapolation")
            if planned_budget > 1.25 * baseline_budget:
                flags.append("diminishing_returns")
            if override is None and default_budget_multiplier != 1.0:
                flags.append("seasonally_adjusted_baseline_budget")
            if planned_budget <= 0:
                # A user who sets a campaign plan to zero must never receive a
                # positive paid-media revenue forecast from a model intercept.
                revenue_p10 = revenue_p50 = revenue_p90 = 0.0
                spend_p10 = spend_p50 = spend_p90 = 0.0
            else:
                direct_category_oov = self.direct_category_oov(
                    history, str(channel), str(campaign_type), as_of, horizon_days, planned_budget
                )
                if direct_category_oov:
                    flags.append("direct_model_category_oov_fallback")
                direct_quantiles = self.direct_quantiles(
                    history, str(channel), str(campaign_type), as_of, horizon_days, planned_budget
                ) if not direct_category_oov else None
                if direct_quantiles is not None:
                    revenue_p10, revenue_p50, revenue_p90 = direct_quantiles
                    # Direct ridge is intentionally flexible, but an
                    # out-of-support one-hot/category combination can produce
                    # an implausibly large intercept-led response.  Bound only
                    # those extreme values against the conservative saturated
                    # response curve; ordinary direct forecasts are unchanged.
                    curve_baseline_budget = max(baseline_budget, 1.0)
                    saturation_scale = max(curve_baseline_budget * 1.5, 1.0)
                    normalization = curve_baseline_budget / (
                        saturation_scale * math.log1p(curve_baseline_budget / saturation_scale)
                    )
                    conservative_p50 = max(
                        0.0,
                        roas * saturation_scale * math.log1p(planned_budget / saturation_scale) * normalization,
                    )
                    response_cap = max(
                        conservative_p50 * 2.0,
                        planned_budget * self.global_roas * 2.0,
                        1.0,
                    )
                    if revenue_p50 > response_cap:
                        scale = response_cap / max(revenue_p50, 1e-9)
                        revenue_p10 *= scale
                        revenue_p50 *= scale
                        revenue_p90 *= scale
                        flags.append("direct_model_response_capped")
                else:
                    # A normalized logarithmic response preserves the baseline
                    # forecast at the current plan while making incremental
                    # budget progressively less productive.
                    curve_baseline_budget = max(baseline_budget, 1.0)
                    saturation_scale = max(curve_baseline_budget * 1.5, 1.0)
                    normalization = curve_baseline_budget / (
                        saturation_scale * math.log1p(curve_baseline_budget / saturation_scale)
                    )
                    revenue_p50 = max(
                        0.0,
                        roas * saturation_scale * math.log1p(planned_budget / saturation_scale) * normalization,
                    )
                    revenue_p10 = revenue_p50 * math.exp(-1.28155 * sigma)
                    revenue_p90 = revenue_p50 * math.exp(1.28155 * sigma)
                spend_p10, spend_p50, spend_p90 = self._spend_quantiles(history, horizon_days, planned_budget)
            rows.append({
                "level": "campaign", "channel": channel, "campaign_type": campaign_type,
                "campaign_id": str(campaign_id), "campaign_key": campaign_key, "campaign_name": campaign_name,
                "planned_budget": planned_budget, "predicted_revenue_p10": revenue_p10,
                "predicted_revenue_p50": revenue_p50, "predicted_revenue_p90": revenue_p90,
                "predicted_spend_p10": spend_p10, "predicted_spend_p50": spend_p50,
                "predicted_spend_p90": spend_p90, "quality_flags": ";".join(flags),
            })
        unused_override_keys = sorted(provided_override_keys - consumed_override_keys)
        if unused_override_keys:
            # An explicit plan must never silently become a no-op. A remaining
            # key is unknown, inactive in the current 28-day forecastable
            # window, or a malformed legacy ID; each needs an upstream fix or
            # a deliberate reactivation workflow.
            raise ValueError(
                "Budget override does not match an active forecastable campaign: "
                f"{unused_override_keys[0]}"
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _spend_quantiles(history: pd.DataFrame, horizon_days: int, planned_budget: float) -> tuple[float, float, float]:
        """Empirical spend intervals from rolling horizon delivery, centered on the plan.

        ``predicted_spend_p50`` remains the planner's intended budget. P10/P90 widen
        from the coefficient of variation of historical horizon-length spend totals
        and, when available, under/over-delivery versus configured budgets.
        """
        if planned_budget <= 0:
            return 0.0, 0.0, 0.0
        daily = history.groupby("date", sort=True)["spend"].sum()
        cv = 0.12
        if len(daily) >= max(horizon_days, 14):
            rolled = daily.rolling(window=horizon_days, min_periods=horizon_days).sum().dropna()
            if len(rolled) >= 3:
                mean_spend = float(rolled.mean())
                std_spend = float(rolled.std(ddof=0))
                if mean_spend > 0:
                    cv = min(0.45, max(0.05, std_spend / mean_spend))
        delivery = 1.0
        configured = history["configured_budget"]
        observed = history[(configured > 0) & history["spend"].notna()]
        if len(observed) >= 7:
            ratios = (observed["spend"] / observed["configured_budget"]).clip(lower=0.0, upper=2.0)
            delivery = float(ratios.median())
            delivery = min(1.25, max(0.55, delivery))
        # Center on the plan, then apply delivery bias and historical volatility.
        center = planned_budget * delivery
        z = 1.28155
        spend_p10 = max(0.0, center * (1.0 - z * cv))
        spend_p90 = center * (1.0 + z * cv)
        spend_p50 = planned_budget
        # Keep the plan inside the interval so the fan chart remains coherent.
        spend_p10 = min(spend_p10, spend_p50)
        spend_p90 = max(spend_p90, spend_p50)
        return spend_p10, spend_p50, spend_p90
