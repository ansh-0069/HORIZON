from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_FEATURES = (
    "log_planned_budget", "recent_roas", "long_roas", "log_recent_spend",
    "log_recent_revenue", "recent_active_days", "trend_ratio", "month_sin", "month_cos",
)


@dataclass
class DirectRidgeModel:
    horizon_days: int
    category_columns: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]
    numeric_mean: list[float]
    numeric_scale: list[float]
    coefficients: list[float]
    residual_p10: float
    residual_p90: float
    sample_count: int
    residual_p50: float = 0.0
    calibration_sample_count: int = 0
    uncertainty_method: str = "temporal_holdout_residual_quantiles"
    # A direct model is fitted on all supported historical campaigns, but the
    # amount of recent evidence at inference time is not uniform.  These
    # data-only profiles are learned on the purged calibration partition and
    # widen only the interval around P50 for sparse / extrapolated requests.
    # They are optional so earlier sealed artifacts remain pickle-compatible.
    support_uncertainty_profiles: dict[str, dict[str, float]] = field(default_factory=dict)

    def unknown_category_values(self, frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        """Return feature-category values not represented by this fitted model.

        A one-hot ridge design otherwise encodes an unseen category as an
        all-zero vector.  That is numerically valid but semantically unsafe:
        it makes an unsupported campaign look like the reference category.
        Callers use this explicit signal to fall back to the conservative
        response curve and disclose the out-of-vocabulary condition.
        """
        unknown: dict[str, tuple[str, ...]] = {}
        for column, values in self.categories.items():
            if column not in frame.columns:
                unknown[column] = ("<missing>",)
                continue
            # Values may have been materialized as numpy string scalars while
            # the incoming canonical frame contains Python strings (or vice
            # versa). Normalize both sides before comparison so a supported
            # category never triggers the conservative OOV fallback merely
            # because of its scalar representation.
            observed = {str(value) for value in frame[column].dropna().astype(str)}
            supported = {str(value) for value in values}
            unsupported = tuple(sorted(observed - supported))
            if unsupported:
                unknown[column] = unsupported
        return unknown

    @staticmethod
    def _design(frame: pd.DataFrame, categories: dict[str, tuple[str, ...]], mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        numeric = frame.loc[:, NUMERIC_FEATURES].astype(float).to_numpy()
        if mean is None:
            mean = numeric.mean(axis=0)
        if scale is None:
            scale = numeric.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        standardized = (numeric - mean) / scale
        parts = [np.ones((len(frame), 1)), standardized]
        for column, values in categories.items():
            observed = frame[column].astype(str).to_numpy()
            parts.append(np.column_stack([(observed == value).astype(float) for value in values]))
        return np.column_stack(parts), mean, scale

    def _support_width_multiplier(self, frame: pd.DataFrame) -> np.ndarray:
        """Return deterministic, support-aware interval widening factors.

        This is intentionally *not* a point-forecast adjustment.  A campaign
        with limited recent delivery or a plan beyond observed pacing retains
        the same conditional P50, but its reported uncertainty must not look
        as certain as a well-supported campaign.  Unknown/legacy artifacts
        return the neutral multiplier of one.
        """
        profiles = getattr(self, "support_uncertainty_profiles", {})
        result = np.ones(len(frame), dtype=float)
        if not isinstance(profiles, dict) or frame.empty:
            return result

        def apply_profile(name: str, mask: np.ndarray) -> None:
            profile = profiles.get(name)
            if not isinstance(profile, dict):
                return
            try:
                multiplier = float(profile.get("width_multiplier", 1.0))
            except (TypeError, ValueError):
                return
            if math.isfinite(multiplier) and multiplier > 1.0:
                result[mask] *= min(multiplier, 3.0)

        recent_days = pd.to_numeric(frame.get("recent_active_days"), errors="coerce")
        if recent_days is not None:
            apply_profile("sparse_recent_history", recent_days.fillna(0.0).to_numpy(dtype=float) < 28.0)
        pacing = pd.to_numeric(frame.get("planned_budget_support_ratio"), errors="coerce")
        if pacing is not None:
            apply_profile("budget_extrapolation", pacing.fillna(float("inf")).to_numpy(dtype=float) > 1.5)
        return np.clip(result, 1.0, 4.0)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Direct callers retain the established numeric interface.  HorizonModel
        # performs the compatibility check before invoking this method so the
        # protected forecast path never silently scores an all-zero OOV design.
        x, _, _ = self._design(frame, self.categories, np.asarray(self.numeric_mean), np.asarray(self.numeric_scale))
        predicted_log = x @ np.asarray(self.coefficients)
        residual_p50 = float(getattr(self, "residual_p50", 0.0))
        multiplier = self._support_width_multiplier(frame)
        lower_residual = residual_p50 + (float(self.residual_p10) - residual_p50) * multiplier
        upper_residual = residual_p50 + (float(self.residual_p90) - residual_p50) * multiplier
        p10 = np.maximum(0.0, np.expm1(predicted_log + lower_residual))
        p50 = np.maximum(0.0, np.expm1(predicted_log + residual_p50))
        p90 = np.maximum(0.0, np.expm1(predicted_log + upper_residual))
        return p10, p50, p90


@dataclass
class DirectRidgeEnsembleModel:
    """A deterministic blend of two leakage-safe direct ridge experts.

    ``scenario_model`` learns the historical relationship conditional on the
    realized future media plan.  ``inference_aligned_model`` instead uses the
    deterministic media-plan estimate that would have been available at each
    historical forecast origin.  The latter prevents the common train/serve
    mismatch where a model learns from realized spend but is asked to score a
    run-rate plan.  A purged temporal selection window chooses the blend
    weight offline; protected inference only executes the two fixed models.

    The class deliberately exposes the same narrow interface as
    ``DirectRidgeModel`` so the evaluator path does not acquire any training
    or optional-library dependency.
    """

    horizon_days: int
    scenario_model: DirectRidgeModel
    inference_aligned_model: DirectRidgeModel
    scenario_weight: float
    selection_sample_count: int = 0
    selection_metric: str = "purged_temporal_validation_revenue_wape_v1"
    uncertainty_method: str = "purged_temporal_selected_two_expert_ensemble_v1"

    @property
    def category_columns(self) -> tuple[str, ...]:
        return self.scenario_model.category_columns

    @property
    def categories(self) -> dict[str, tuple[str, ...]]:
        """Expose the union for provenance rendering only.

        Runtime compatibility is stricter than this union: prediction falls
        back unless *both* experts support a category.  The union is useful in
        the immutable feature-schema fingerprint while avoiding accidental
        serialization of a mutable cached value.
        """
        names = set(self.scenario_model.categories) | set(self.inference_aligned_model.categories)
        return {
            name: tuple(
                sorted(
                    set(self.scenario_model.categories.get(name, ()))
                    | set(self.inference_aligned_model.categories.get(name, ()))
                )
            )
            for name in names
        }

    @property
    def sample_count(self) -> int:
        return min(int(self.scenario_model.sample_count), int(self.inference_aligned_model.sample_count))

    @property
    def calibration_sample_count(self) -> int:
        return min(
            int(getattr(self.scenario_model, "calibration_sample_count", 0)),
            int(getattr(self.inference_aligned_model, "calibration_sample_count", 0)),
        )

    def unknown_category_values(self, frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        unknown: dict[str, set[str]] = {}
        for member in (self.scenario_model, self.inference_aligned_model):
            for column, values in member.unknown_category_values(frame).items():
                unknown.setdefault(column, set()).update(str(value) for value in values)
        return {column: tuple(sorted(values)) for column, values in sorted(unknown.items())}

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scenario = self.scenario_model.predict(frame)
        aligned = self.inference_aligned_model.predict(frame)
        weight = min(1.0, max(0.0, float(self.scenario_weight)))
        lower = weight * scenario[0] + (1.0 - weight) * aligned[0]
        middle = weight * scenario[1] + (1.0 - weight) * aligned[1]
        upper = weight * scenario[2] + (1.0 - weight) * aligned[2]

        # Expert disagreement is epistemic uncertainty, not a point shift.
        # Add a bounded quarter-spread around the blended marginal interval so
        # an ensemble cannot look more certain merely because its experts
        # disagree.  The bound keeps a pathological stale model from creating
        # unbounded numbers in evaluator inference.
        disagreement = np.minimum(np.abs(scenario[1] - aligned[1]), np.maximum(middle * 3.0, 1.0))
        lower = np.maximum(0.0, lower - 0.25 * disagreement)
        upper = np.maximum(middle, upper + 0.25 * disagreement)
        lower = np.minimum(lower, middle)
        return lower, middle, upper


def _features(history: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int, planned_budget: float) -> dict[str, Any]:
    recent = history[(history["date"] > cutoff - pd.Timedelta(days=28)) & (history["date"] <= cutoff)]
    previous = history[(history["date"] > cutoff - pd.Timedelta(days=56)) & (history["date"] <= cutoff - pd.Timedelta(days=28))]
    recent_spend = float(recent["spend"].sum())
    recent_revenue = float(recent["revenue"].sum())
    history_spend = float(history["spend"].sum())
    history_revenue = float(history["revenue"].sum())
    previous_spend = float(previous["spend"].sum())
    historical_daily_spend = history_spend / max(float(history["date"].nunique()), 1.0)
    budget_support = max(historical_daily_spend * horizon_days, 1.0)
    return {
        "log_planned_budget": math.log1p(max(planned_budget, 0.0)),
        "recent_roas": recent_revenue / max(recent_spend, 1e-9),
        "long_roas": history_revenue / max(history_spend, 1e-9),
        "log_recent_spend": math.log1p(max(recent_spend, 0.0)),
        "log_recent_revenue": math.log1p(max(recent_revenue, 0.0)),
        "recent_active_days": float(recent["date"].nunique()),
        "trend_ratio": recent_spend / max(previous_spend, 1.0),
        "month_sin": math.sin(2 * math.pi * (cutoff.month + horizon_days / 60.0) / 12.0),
        "month_cos": math.cos(2 * math.pi * (cutoff.month + horizon_days / 60.0) / 12.0),
        # Context fields are intentionally not part of NUMERIC_FEATURES. They
        # let calibrated inference widen uncertainty by support without
        # changing the point-model feature schema or introducing a target leak.
        "planned_budget_support_ratio": max(planned_budget, 0.0) / budget_support,
    }


def inference_features(history: pd.DataFrame, channel: str, campaign_type: str, cutoff: pd.Timestamp, horizon_days: int, planned_budget: float) -> pd.DataFrame:
    row = _features(history, cutoff, horizon_days, planned_budget)
    row.update({"channel": str(channel), "campaign_type": str(campaign_type)})
    return pd.DataFrame([row])
