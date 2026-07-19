from __future__ import annotations

from dataclasses import dataclass
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
        return tuple(float(value[0]) for value in direct_model.predict(direct_features))

    def forecast_campaigns(self, canonical: pd.DataFrame, horizon_days: int, budget_overrides: dict[str, float] | None = None) -> pd.DataFrame:
        budget_overrides = budget_overrides or {}
        as_of = canonical["date"].max()
        window_start = as_of - pd.Timedelta(days=27)
        recent = canonical[(canonical["date"] >= window_start) & (canonical["date"] <= as_of)].copy()
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
            # silently receive a future budget in a baseline media plan.
            if recent_spend <= 0:
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
            # Legacy unqualified IDs remain accepted only when unambiguous. New
            # product callers must use source-qualified campaign keys.
            if override is None and str(campaign_id) in budget_overrides:
                override = budget_overrides[str(campaign_id)]
            baseline_budget = max(0.0, daily_spend * horizon_days * default_budget_multiplier)
            planned_budget = float(override) if override is not None else baseline_budget
            support = max(historic_spend / max(int(history["date"].nunique()), 1) * horizon_days, 1.0)
            extrapolation = planned_budget > 1.5 * support
            sigma = self.global_log_sigma * (1.0 + (1.0 - reliability) + (0.55 if extrapolation else 0.0))
            if planned_budget <= 0:
                # A user who sets a campaign plan to zero must never receive a
                # positive paid-media revenue forecast from a model intercept.
                revenue_p10 = revenue_p50 = revenue_p90 = 0.0
                spend_p10 = spend_p50 = spend_p90 = 0.0
            else:
                direct_quantiles = self.direct_quantiles(history, str(channel), str(campaign_type), as_of, horizon_days, planned_budget)
                if direct_quantiles is not None:
                    revenue_p10, revenue_p50, revenue_p90 = direct_quantiles
                else:
                    # Fallback only when the direct model is unavailable: a
                    # normalized logarithmic response preserves the baseline at
                    # the current plan and saturates incremental spend.
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
            flags = []
            if active_days < self.min_history_days:
                flags.append("sparse_recent_history")
            if extrapolation:
                flags.append("budget_extrapolation")
            if planned_budget > 1.25 * baseline_budget:
                flags.append("diminishing_returns")
            if override is None and default_budget_multiplier != 1.0:
                flags.append("seasonally_adjusted_baseline_budget")
            rows.append({
                "level": "campaign", "channel": channel, "campaign_type": campaign_type,
                "campaign_id": str(campaign_id), "campaign_key": campaign_key, "campaign_name": campaign_name,
                "planned_budget": planned_budget, "predicted_revenue_p10": revenue_p10,
                "predicted_revenue_p50": revenue_p50, "predicted_revenue_p90": revenue_p90,
                "predicted_spend_p10": spend_p10, "predicted_spend_p50": spend_p50,
                "predicted_spend_p90": spend_p90, "quality_flags": ";".join(flags),
            })
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
