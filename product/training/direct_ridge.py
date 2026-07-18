from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.direct_model import DirectRidgeModel, _features


def training_frame(canonical: pd.DataFrame, horizon_days: int, step_days: int = 21) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["source_system", "source_campaign_id", "channel", "campaign_type", "campaign_name"]
    for key, group in canonical.groupby(keys, dropna=False):
        group = group.sort_values("date")
        minimum = group["date"].min() + pd.Timedelta(days=56)
        maximum = group["date"].max() - pd.Timedelta(days=horizon_days)
        for cutoff in pd.date_range(minimum, maximum, freq=f"{step_days}D"):
            history = group[group["date"] <= cutoff]
            future = group[(group["date"] > cutoff) & (group["date"] <= cutoff + pd.Timedelta(days=horizon_days))]
            planned_budget = float(future["spend"].sum())
            if len(history) < 28 or planned_budget <= 0:
                continue
            feature_row = _features(history, cutoff, horizon_days, planned_budget)
            feature_row.update({
                "cutoff": cutoff,
                "channel": str(key[2]),
                "campaign_type": str(key[3]),
                "target_log_revenue": math.log1p(max(float(future["revenue"].sum()), 0.0)),
            })
            rows.append(feature_row)
    return pd.DataFrame(rows)


def _fit_coefficients(
    frame: pd.DataFrame,
    categories: dict[str, tuple[str, ...]],
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, mean, scale = DirectRidgeModel._design(frame, categories)
    y = frame["target_log_revenue"].to_numpy(dtype=float)
    penalty = np.eye(x.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y), mean, scale


def fit_direct_ridge(canonical: pd.DataFrame, horizon_days: int, ridge_alpha: float = 4.0) -> DirectRidgeModel | None:
    frame = training_frame(canonical, horizon_days)
    if len(frame) < 80:
        return None
    categories = {column: tuple(sorted(frame[column].astype(str).unique())) for column in ("channel", "campaign_type")}
    cutoff_values = sorted(pd.Timestamp(value) for value in frame["cutoff"].unique())
    holdout_cutoffs = max(1, math.ceil(len(cutoff_values) * 0.20))
    calibration_start = cutoff_values[-holdout_cutoffs]
    fit_frame = frame[frame["cutoff"] < calibration_start].copy()
    calibration_frame = frame[frame["cutoff"] >= calibration_start].copy()

    # Quantiles are estimated from residuals on later, unseen training windows.
    # The final point model is then refit on all available historical windows.
    if len(fit_frame) >= 40 and len(calibration_frame) >= 20:
        calibration_coefficients, calibration_mean, calibration_scale = _fit_coefficients(fit_frame, categories, ridge_alpha)
        calibration_x, _, _ = DirectRidgeModel._design(calibration_frame, categories, calibration_mean, calibration_scale)
        calibration_residuals = calibration_frame["target_log_revenue"].to_numpy(dtype=float) - calibration_x @ calibration_coefficients
        uncertainty_method = "temporal_holdout_residual_quantiles"
    else:
        calibration_residuals = np.array([], dtype=float)
        uncertainty_method = "in_sample_residual_fallback"

    coefficients, mean, scale = _fit_coefficients(frame, categories, ridge_alpha)
    x, _, _ = DirectRidgeModel._design(frame, categories, mean, scale)
    fitted_residuals = frame["target_log_revenue"].to_numpy(dtype=float) - x @ coefficients
    residuals = calibration_residuals if len(calibration_residuals) else fitted_residuals
    return DirectRidgeModel(
        horizon_days=horizon_days,
        category_columns=("channel", "campaign_type"),
        categories=categories,
        numeric_mean=mean.tolist(), numeric_scale=scale.tolist(), coefficients=coefficients.tolist(),
        residual_p10=float(np.quantile(residuals, 0.10)),
        residual_p90=float(np.quantile(residuals, 0.90)),
        sample_count=len(frame),
        calibration_sample_count=len(calibration_residuals),
        uncertainty_method=uncertainty_method,
    )
