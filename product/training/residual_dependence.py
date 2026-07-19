"""Offline calibration for dependence-aware portfolio forecast draws.

The direct ridge model calibrates campaign-level marginal intervals.  A
portfolio interval cannot safely be formed by summing every campaign at the
same percentile: that silently assumes perfect rank correlation.  This module
fits a compact hierarchical residual-factor copula from leakage-safe,
expanding-window out-of-fold residuals.  It is training-only and serializes a
small data-only profile into ``HorizonModel``; protected inference performs no
fitting and requires no optional dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from product.training.direct_ridge import expanding_window_oof_residuals


DEPENDENCE_METHOD = "hierarchical_residual_factor_copula_v1"
FALLBACK_METHOD = "independent_rank_fallback_v1"
_MINIMUM_BLOCKS = 6
_MINIMUM_SAMPLES = 80
_MINIMUM_IDIOSYNCRATIC_WEIGHT = 0.15


def _fallback_profile(residuals: pd.DataFrame, reason: str) -> dict[str, object]:
    return {
        "method": FALLBACK_METHOD,
        "calibration_method": "unavailable",
        "fallback_reason": reason,
        "sample_count": int(len(residuals)),
        "block_count": int(residuals["cutoff"].nunique()) if "cutoff" in residuals else 0,
        "factor_weights": {"global": 0.0, "channel": 0.0, "campaign_type": 0.0, "idiosyncratic": 1.0},
    }


def _variance(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.var(array)) if len(array) > 1 else 0.0


def _portfolio_oof_residual_profile(residuals: pd.DataFrame) -> dict[str, object] | None:
    """Estimate aggregate log-residual quantiles from the same OOF blocks.

    Campaign-level marginal intervals and a factor copula describe how leaves
    co-move, but diversification can still make their aggregate range too
    narrow.  These are true portfolio residuals: natural-scale predicted and
    actual campaign revenue are summed within each forecast-origin block
    before the log residual is calculated. This makes the later empirical
    residual adjustment a portfolio calibration, not a sum of campaign
    quantiles or a formal conformal interval.
    """
    required = {"cutoff", "target_log_revenue", "predicted_log_revenue"}
    if not required.issubset(residuals.columns):
        return None
    frame = residuals.loc[:, ["cutoff", "target_log_revenue", "predicted_log_revenue"]].copy()
    frame["cutoff"] = pd.to_datetime(frame["cutoff"], errors="coerce")
    for column in ("target_log_revenue", "predicted_log_revenue"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["cutoff", "target_log_revenue", "predicted_log_revenue"])
    if frame.empty:
        return None
    frame["actual_revenue"] = np.clip(np.expm1(frame["target_log_revenue"].to_numpy(dtype=float)), 0.0, None)
    frame["predicted_revenue"] = np.clip(np.expm1(frame["predicted_log_revenue"].to_numpy(dtype=float)), 0.0, None)
    grouped = frame.groupby("cutoff", sort=False)[["actual_revenue", "predicted_revenue"]].sum()
    if len(grouped) < _MINIMUM_BLOCKS:
        return None
    errors = np.log1p(grouped["actual_revenue"].to_numpy(dtype=float)) - np.log1p(
        grouped["predicted_revenue"].to_numpy(dtype=float)
    )
    errors = errors[np.isfinite(errors)]
    if len(errors) < _MINIMUM_BLOCKS:
        return None
    return {
        "method": "expanding_window_oof_portfolio_log_residual_quantiles_v1",
        "sample_count": int(len(errors)),
        "residual_p10": round(float(np.quantile(errors, 0.10)), 8),
        "residual_p50": round(float(np.quantile(errors, 0.50)), 8),
        "residual_p90": round(float(np.quantile(errors, 0.90)), 8),
    }


def calibrate_residual_dependence(residuals: pd.DataFrame) -> dict[str, object]:
    """Estimate stable factor weights from aligned OOF log-residuals.

    The decomposition intentionally identifies a shared effect only when the
    corresponding group has at least two simultaneous campaigns.  Otherwise a
    singleton group's deviation is retained as idiosyncratic variation instead
    of overfitting a channel/type factor.  Variance energies are normalized
    into latent-Gaussian factor weights; a 15% idiosyncratic floor prevents
    sparse historical data from degenerating back into perfect correlation.
    """
    required = {"cutoff", "channel", "campaign_type", "residual_log_revenue"}
    if not required.issubset(residuals.columns):
        return _fallback_profile(residuals, "missing_residual_columns")
    frame = residuals.loc[:, list(required)].copy()
    frame["residual_log_revenue"] = pd.to_numeric(frame["residual_log_revenue"], errors="coerce")
    frame = frame.dropna(subset=list(required)).copy()
    if frame.empty:
        return _fallback_profile(frame, "no_finite_residuals")
    block_sizes = frame.groupby("cutoff", sort=False).size()
    # A one-campaign block carries a marginal residual but no information about
    # cross-campaign dependence, so exclude it from factor calibration.
    frame = frame[frame["cutoff"].map(block_sizes) >= 2].copy()
    block_count = int(frame["cutoff"].nunique()) if not frame.empty else 0
    if len(frame) < _MINIMUM_SAMPLES or block_count < _MINIMUM_BLOCKS:
        return _fallback_profile(frame, "insufficient_aligned_oof_residuals")

    residual = frame["residual_log_revenue"]
    global_mean = frame.groupby("cutoff", sort=False)["residual_log_revenue"].transform("mean")
    channel_group = [frame["cutoff"], frame["channel"]]
    channel_mean = frame.groupby(channel_group, sort=False)["residual_log_revenue"].transform("mean")
    channel_count = frame.groupby(channel_group, sort=False)["residual_log_revenue"].transform("size")
    type_group = [frame["cutoff"], frame["channel"], frame["campaign_type"]]
    type_mean = frame.groupby(type_group, sort=False)["residual_log_revenue"].transform("mean")
    type_count = frame.groupby(type_group, sort=False)["residual_log_revenue"].transform("size")

    global_effect = global_mean - float(global_mean.mean())
    channel_effect = (channel_mean - global_mean).where(channel_count >= 2, 0.0)
    type_effect = (type_mean - channel_mean).where(type_count >= 2, 0.0)
    idiosyncratic = residual - global_mean - channel_effect - type_effect

    raw = {
        "global": _variance(global_effect),
        "channel": _variance(channel_effect),
        "campaign_type": _variance(type_effect),
        "idiosyncratic": _variance(idiosyncratic),
    }
    total = float(sum(raw.values()))
    if not np.isfinite(total) or total <= 1e-12:
        return _fallback_profile(frame, "degenerate_residual_variance")

    raw_common = float(raw["global"] + raw["channel"] + raw["campaign_type"])
    common_weight = min(1.0 - _MINIMUM_IDIOSYNCRATIC_WEIGHT, raw_common / total)
    if raw_common <= 1e-12:
        weights = {"global": 0.0, "channel": 0.0, "campaign_type": 0.0, "idiosyncratic": 1.0}
    else:
        weights = {
            "global": common_weight * raw["global"] / raw_common,
            "channel": common_weight * raw["channel"] / raw_common,
            "campaign_type": common_weight * raw["campaign_type"] / raw_common,
            "idiosyncratic": 1.0 - common_weight,
        }
    # Convert NumPy values to plain floats before pickling; this makes the
    # artifact portable and keeps serialized metadata inspectable.
    weights = {name: round(float(max(value, 0.0)), 8) for name, value in weights.items()}
    weights["idiosyncratic"] = round(
        float(max(0.0, 1.0 - weights["global"] - weights["channel"] - weights["campaign_type"])),
        8,
    )
    return {
        "method": DEPENDENCE_METHOD,
        "calibration_method": "expanding_window_oof_log_residual_variance_components",
        "residual_space": "log1p_revenue",
        "sample_count": int(len(frame)),
        "block_count": block_count,
        "minimum_idiosyncratic_weight": _MINIMUM_IDIOSYNCRATIC_WEIGHT,
        "factor_weights": weights,
        "portfolio_oof_residual_calibration": _portfolio_oof_residual_profile(residuals),
    }


def fit_residual_dependence(canonical: pd.DataFrame, horizon_days: int) -> dict[str, object]:
    """Fit one dependence profile per forecast horizon outside inference."""
    residuals = expanding_window_oof_residuals(canonical, horizon_days)
    profile = calibrate_residual_dependence(residuals)
    profile["horizon_days"] = int(horizon_days)
    return profile
