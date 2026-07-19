from __future__ import annotations

from dataclasses import dataclass, field
import math
import pandas as pd
from src.budget_plan import campaign_baseline_budget, seasonal_default_budget_multiplier
from src.direct_model import DirectRidgeEnsembleModel, DirectRidgeModel, inference_features


# This string is intentionally duplicated in the training module instead of
# importing training-only code into the protected evaluator path.  The profile
# itself is plain pickle data, and this marker lets inference reject malformed
# or stale calibration metadata safely.
FALLBACK_INTERVAL_CALIBRATION_METHOD = "expanding_window_oof_statistical_campaign_interval_width_conformal_v1"
FALLBACK_PORTFOLIO_INTERVAL_CALIBRATION_METHOD = "expanding_window_oof_statistical_portfolio_log_residual_quantiles_v1"


@dataclass
class HorizonModel:
    model_version: str
    global_roas: float
    global_log_sigma: float
    month_roas_factors: dict[int, float]
    direct_models: dict[int, DirectRidgeModel | DirectRidgeEnsembleModel]
    min_history_days: int = 28
    target_roas: float = 4.0
    artifact_sha256: str = ""
    # Provenance fields are data-only and therefore safe for the protected
    # pickle loader. They bind the shipped artifact to the canonical training
    # contract and one-hot feature schema without importing product code.
    training_data_fingerprint: str = ""
    feature_schema_fingerprint: str = ""
    # A training-only temporal tournament may retain direct candidates while
    # selecting the conservative statistical family for a particular horizon.
    # Legacy artifacts default to direct serving so their historical behavior
    # remains stable after this field was introduced.
    selected_model_families: dict[int, str] = field(default_factory=dict)
    model_selection: dict[int, dict[str, object]] = field(default_factory=dict)
    # Statistical fallback intervals are calibrated offline from expanding,
    # non-overlapping forecast origins.  The persisted profile scales only the
    # distance from P50 to P10/P90; point forecasts remain exactly unchanged.
    # Keeping it optional preserves compatibility with legacy pickle artifacts.
    fallback_interval_calibration: dict[int, dict[str, object]] = field(default_factory=dict)
    # Trained outside the protected evaluator path.  The field is deliberately
    # optional so artifacts created before dependence calibration remain
    # loadable; inference uses an explicit independent fallback for those
    # legacy artifacts rather than reintroducing perfect rank correlation.
    residual_dependence: dict[int, dict[str, object]] = field(default_factory=dict)

    def __getattr__(self, name: str) -> object:
        """Supply defaults for fields absent from pre-v4 pickle artifacts."""
        if name in {
            "residual_dependence",
            "selected_model_families",
            "model_selection",
            "fallback_interval_calibration",
        }:
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
        if self.selected_model_family(horizon_days) != "direct_ensemble":
            return {}
        profiles = getattr(self, "residual_dependence", {})
        if not isinstance(profiles, dict):
            return {}
        profile = profiles.get(horizon_days, profiles.get(str(horizon_days), {}))
        return dict(profile) if isinstance(profile, dict) else {}

    def selected_model_family(self, horizon_days: int) -> str:
        """Return the serving family selected by an offline temporal tournament."""
        families = getattr(self, "selected_model_families", {})
        if not isinstance(families, dict):
            return "direct_ensemble"
        selected = families.get(horizon_days, families.get(str(horizon_days), "direct_ensemble"))
        return str(selected) if selected in {"direct_ensemble", "statistical_fallback"} else "direct_ensemble"

    def fallback_interval_width_multipliers(self, horizon_days: int) -> tuple[float, float] | None:
        """Return validated offline width scales for statistical fallback leaves.

        This method deliberately does no calibration at inference.  It only
        validates training-produced, data-only metadata and returns a safe
        ``None`` for legacy, malformed, direct-serving, or insufficient-data
        profiles.  Both widths are bounded at one or above so the calibration
        can widen an under-covered fallback interval but can never turn an
        already conservative interval into a narrower one at runtime.
        """
        if self.selected_model_family(horizon_days) != "statistical_fallback":
            return None
        profiles = getattr(self, "fallback_interval_calibration", {})
        if not isinstance(profiles, dict):
            return None
        profile = profiles.get(horizon_days, profiles.get(str(horizon_days), {}))
        if not isinstance(profile, dict):
            return None
        if profile.get("method") != FALLBACK_INTERVAL_CALIBRATION_METHOD:
            return None
        if profile.get("status") != "available":
            return None
        try:
            origin_count = int(profile["origin_count"])
            sample_count = int(profile["sample_count"])
            lower = float(profile["lower_width_multiplier"])
            upper = float(profile["upper_width_multiplier"])
        except (KeyError, TypeError, ValueError):
            return None
        if profile.get("calibration_level") != "campaign_marginal" or origin_count < 5 or sample_count < 30:
            return None
        # The trainer caps multipliers at 4.0. Recheck it here because an
        # artifact is an input boundary in the protected submission path.
        if not all(math.isfinite(value) and 1.0 <= value <= 4.0 for value in (lower, upper)):
            return None
        return lower, upper

    def fallback_portfolio_interval_profile(self, horizon_days: int) -> dict[str, float] | None:
        """Return a validated OOF portfolio *revenue interval* profile.

        The profile is deliberately distinct from ROAS-target probability
        calibration. It contains only centered aggregate revenue residual
        quantiles, fitted outside the evaluator from non-overlapping temporal
        origins. Invalid, sparse, direct-serving, or legacy metadata fails
        closed so protected inference never estimates uncertainty at runtime.
        """
        if self.selected_model_family(horizon_days) != "statistical_fallback":
            return None
        profiles = getattr(self, "fallback_interval_calibration", {})
        if not isinstance(profiles, dict):
            return None
        parent = profiles.get(horizon_days, profiles.get(str(horizon_days), {}))
        if not isinstance(parent, dict):
            return None
        profile = parent.get("portfolio_interval_profile", {})
        if not isinstance(profile, dict):
            return None
        if profile.get("method") != FALLBACK_PORTFOLIO_INTERVAL_CALIBRATION_METHOD:
            return None
        if profile.get("status") != "available":
            return None
        if profile.get("calibration_purpose") != "revenue_interval_only_not_roas_probability_calibration":
            return None
        try:
            origin_count = int(profile["origin_count"])
            residual_p10 = float(profile["residual_p10"])
            residual_p50 = float(profile["residual_p50"])
            residual_p90 = float(profile["residual_p90"])
        except (KeyError, TypeError, ValueError):
            return None
        values = (residual_p10, residual_p50, residual_p90)
        if origin_count < 5 or not all(math.isfinite(value) for value in values):
            return None
        if residual_p10 > residual_p50 or residual_p50 > residual_p90:
            return None
        return {
            "residual_p10": residual_p10,
            "residual_p50": residual_p50,
            "residual_p90": residual_p90,
        }

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
        return seasonal_default_budget_multiplier(canonical, as_of, horizon_days)

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
        if self.selected_model_family(horizon_days) != "direct_ensemble":
            return None
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
        if self.selected_model_family(horizon_days) != "direct_ensemble":
            return False
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
            baseline_budget, baseline_method, _ = campaign_baseline_budget(
                canonical, history, as_of, horizon_days
            ) if recent_spend > 0 else (0.0, "no_recent_delivery", 0.0)
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
            if override is None and baseline_method == "calendar_matched_prior_year_budget":
                flags.append("calendar_matched_prior_year_budget")
            if override is None and baseline_method == "recent_run_rate_budget":
                flags.append("recent_run_rate_baseline_budget")
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
                    fallback_widths = self.fallback_interval_width_multipliers(horizon_days)
                    if fallback_widths is not None and revenue_p50 > 0.0:
                        # Apply the offline OOF calibration in log1p-revenue
                        # space. P50 is intentionally left untouched: model
                        # selection and point-forecast backtests continue to
                        # measure the same statistical response model.
                        lower_width, upper_width = fallback_widths
                        log_p50 = math.log1p(revenue_p50)
                        log_p10 = math.log1p(max(revenue_p10, 0.0))
                        log_p90 = math.log1p(max(revenue_p90, revenue_p50))
                        if lower_width > 1.0:
                            revenue_p10 = max(
                                0.0,
                                math.expm1(log_p50 + (log_p10 - log_p50) * lower_width),
                            )
                        if upper_width > 1.0:
                            revenue_p90 = max(
                                revenue_p50,
                                math.expm1(log_p50 + (log_p90 - log_p50) * upper_width),
                            )
                        flags.append("fallback_oof_interval_calibrated")
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
