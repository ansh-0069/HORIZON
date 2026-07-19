from __future__ import annotations

from dataclasses import dataclass
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

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Direct callers retain the established numeric interface.  HorizonModel
        # performs the compatibility check before invoking this method so the
        # protected forecast path never silently scores an all-zero OOV design.
        x, _, _ = self._design(frame, self.categories, np.asarray(self.numeric_mean), np.asarray(self.numeric_scale))
        predicted_log = x @ np.asarray(self.coefficients)
        p10 = np.maximum(0.0, np.expm1(predicted_log + self.residual_p10))
        p50 = np.maximum(0.0, np.expm1(predicted_log + float(getattr(self, "residual_p50", 0.0))))
        p90 = np.maximum(0.0, np.expm1(predicted_log + self.residual_p90))
        return p10, p50, p90


def _features(history: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int, planned_budget: float) -> dict[str, Any]:
    recent = history[(history["date"] > cutoff - pd.Timedelta(days=28)) & (history["date"] <= cutoff)]
    previous = history[(history["date"] > cutoff - pd.Timedelta(days=56)) & (history["date"] <= cutoff - pd.Timedelta(days=28))]
    recent_spend = float(recent["spend"].sum())
    recent_revenue = float(recent["revenue"].sum())
    history_spend = float(history["spend"].sum())
    history_revenue = float(history["revenue"].sum())
    previous_spend = float(previous["spend"].sum())
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
    }


def inference_features(history: pd.DataFrame, channel: str, campaign_type: str, cutoff: pd.Timestamp, horizon_days: int, planned_budget: float) -> pd.DataFrame:
    row = _features(history, cutoff, horizon_days, planned_budget)
    row.update({"channel": str(channel), "campaign_type": str(campaign_type)})
    return pd.DataFrame([row])
