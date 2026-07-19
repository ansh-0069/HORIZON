from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from product.training.direct_ridge import fit_direct_ensemble
from src.contracts import CANONICAL_COLUMNS
from product.training.residual_dependence import fit_residual_dependence
from src.direct_model import NUMERIC_FEATURES
from src.forecast import build_forecast
from src.model import HorizonModel


# The fallback model is intentionally compact, so its raw log-normal interval
# can be under-dispersed when daily ROAS volatility is not representative of a
# horizon-level campaign outcome.  These values define a training-only OOF
# calibration protocol.  They are conservative guardrails rather than a claim
# of distribution-free coverage for a non-stationary marketing time series.
FALLBACK_INTERVAL_CALIBRATION_METHOD = "expanding_window_oof_statistical_campaign_interval_width_conformal_v1"
FALLBACK_PORTFOLIO_INTERVAL_CALIBRATION_METHOD = "expanding_window_oof_statistical_portfolio_log_residual_quantiles_v1"
FALLBACK_INTERVAL_CALIBRATION_MIN_TRAINING_DAYS = 90
FALLBACK_INTERVAL_CALIBRATION_MIN_ORIGINS = 5
FALLBACK_INTERVAL_CALIBRATION_MIN_SAMPLES = 30
FALLBACK_INTERVAL_CALIBRATION_MAX_ORIGINS = 12
FALLBACK_INTERVAL_CALIBRATION_MAX_WIDTH = 4.0
FALLBACK_INTERVAL_CALIBRATION_TAIL_COVERAGE = 0.90


def _fallback_calibration_unavailable(
    horizon_days: int,
    *,
    status: str,
    origin_count: int,
    sample_count: int,
) -> dict[str, object]:
    """Return inspectable provenance when an OOF fallback profile is unavailable."""
    return {
        "method": "unavailable",
        "status": status,
        "horizon_days": int(horizon_days),
        "calibration_level": "campaign_marginal",
        "origin_protocol": "expanding_window_non_overlapping_targets",
        "fit_window": "date <= origin",
        "target_window": "origin < date <= origin + horizon_days",
        "minimum_training_days": FALLBACK_INTERVAL_CALIBRATION_MIN_TRAINING_DAYS,
        "minimum_origin_count": FALLBACK_INTERVAL_CALIBRATION_MIN_ORIGINS,
        "minimum_sample_count": FALLBACK_INTERVAL_CALIBRATION_MIN_SAMPLES,
        "origin_count": int(origin_count),
        "sample_count": int(sample_count),
    }


def _fallback_portfolio_calibration_unavailable(
    horizon_days: int,
    *,
    status: str,
    origin_count: int,
) -> dict[str, object]:
    """Return auditable no-op metadata for a sparse fallback portfolio profile."""
    return {
        "method": "unavailable",
        "status": status,
        "horizon_days": int(horizon_days),
        "calibration_level": "portfolio",
        "calibration_purpose": "revenue_interval_only_not_roas_probability_calibration",
        "origin_protocol": "expanding_window_non_overlapping_targets",
        "fit_window": "date <= origin",
        "target_window": "origin < date <= origin + horizon_days",
        "minimum_origin_count": FALLBACK_INTERVAL_CALIBRATION_MIN_ORIGINS,
        "origin_count": int(origin_count),
    }


def _non_overlapping_fallback_origins(canonical: pd.DataFrame, horizon_days: int) -> list[pd.Timestamp]:
    """Choose recent expanding-window origins with disjoint target windows.

    An origin is separated from the next by exactly one forecast horizon, so
    target windows never overlap. A later origin may legitimately train on an
    earlier OOF target because that outcome would have been known by then.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    dates = pd.Series(pd.to_datetime(canonical["date"], errors="coerce")).dropna()
    if dates.empty:
        return []
    earliest = pd.Timestamp(dates.min()).normalize()
    latest = pd.Timestamp(dates.max()).normalize()
    start = earliest + pd.Timedelta(days=FALLBACK_INTERVAL_CALIBRATION_MIN_TRAINING_DAYS)
    end = latest - pd.Timedelta(days=horizon_days)
    if start > end:
        return []
    origins = list(pd.date_range(start, end, freq=f"{horizon_days}D"))
    return [pd.Timestamp(origin) for origin in origins[-FALLBACK_INTERVAL_CALIBRATION_MAX_ORIGINS:]]


def _required_interval_widths(
    actual: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided log-space width scales needed to include OOF outcomes.

    A scale of one means the raw statistical interval already includes the
    observation. Values above one widen only the corresponding side of the
    P50. Invalid or degenerate rows are rejected before this helper is called.
    """
    actual_log = np.log1p(np.clip(np.asarray(actual, dtype=float), 0.0, None))
    p10_log = np.log1p(np.clip(np.asarray(p10, dtype=float), 0.0, None))
    p50_log = np.log1p(np.clip(np.asarray(p50, dtype=float), 0.0, None))
    p90_log = np.log1p(np.clip(np.asarray(p90, dtype=float), 0.0, None))
    lower_base = np.maximum(p50_log - p10_log, 1e-12)
    upper_base = np.maximum(p90_log - p50_log, 1e-12)
    lower = np.maximum(0.0, (p50_log - actual_log) / lower_base)
    upper = np.maximum(0.0, (actual_log - p50_log) / upper_base)
    return lower, upper


def _finite_sample_conformal_width(required: np.ndarray) -> float:
    """Select a conservative one-sided OOF order statistic without runtime fitting."""
    values = np.asarray(required, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot calibrate fallback interval widths without finite OOF residuals")
    ordered = np.sort(values, kind="stable")
    # Finite-sample split-conformal rank for 90% one-sided marginal coverage.
    # The max(1, ...) and zero-based conversion make the policy explicit for
    # small samples; the trainer has a larger minimum sample guard below.
    rank = min(
        len(ordered) - 1,
        max(0, int(math.ceil((len(ordered) + 1) * FALLBACK_INTERVAL_CALIBRATION_TAIL_COVERAGE)) - 1),
    )
    return min(FALLBACK_INTERVAL_CALIBRATION_MAX_WIDTH, max(1.0, float(ordered[rank])))


def _apply_interval_widths(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    lower_width: float,
    upper_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a pair of log1p width scales while leaving P50 untouched."""
    middle = np.log1p(np.clip(np.asarray(p50, dtype=float), 0.0, None))
    lower = np.log1p(np.clip(np.asarray(p10, dtype=float), 0.0, None))
    upper = np.log1p(np.clip(np.asarray(p90, dtype=float), 0.0, None))
    adjusted_p10 = np.expm1(middle + (lower - middle) * float(lower_width))
    adjusted_p90 = np.expm1(middle + (upper - middle) * float(upper_width))
    return np.clip(adjusted_p10, 0.0, None), np.maximum(adjusted_p90, np.asarray(p50, dtype=float))


def _fit_fallback_portfolio_interval_profile(
    records: list[dict[str, object]],
    horizon_days: int,
) -> dict[str, object]:
    """Fit OOF aggregate-revenue residual quantiles for fallback forecasts.

    Unlike the leaf width profile, this profile operates on the aggregate
    revenue path after the hierarchy is rolled up. It is therefore able to
    correct the independent-rank portfolio P10/P90 collapse without moving the
    selected P50. It deliberately calibrates revenue intervals only: ROAS
    target probabilities remain uncalibrated decision-support estimates.
    """
    if len(records) < FALLBACK_INTERVAL_CALIBRATION_MIN_ORIGINS:
        return _fallback_portfolio_calibration_unavailable(
            horizon_days,
            status="insufficient_non_overlapping_oof_origins",
            origin_count=len(records),
        )
    frame = pd.DataFrame(records)
    required = {
        "origin",
        "actual_revenue",
        "predicted_revenue_p10",
        "predicted_revenue_p50",
        "predicted_revenue_p90",
    }
    if not required.issubset(frame.columns):
        return _fallback_portfolio_calibration_unavailable(
            horizon_days,
            status="missing_oof_portfolio_columns",
            origin_count=len(records),
        )
    for column in required - {"origin"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().copy()
    frame = frame[
        (frame["actual_revenue"] >= 0.0)
        & (frame["predicted_revenue_p10"] >= 0.0)
        & (frame["predicted_revenue_p10"] <= frame["predicted_revenue_p50"])
        & (frame["predicted_revenue_p50"] <= frame["predicted_revenue_p90"])
        & (frame["predicted_revenue_p50"] > 0.0)
    ].copy()
    if len(frame) < FALLBACK_INTERVAL_CALIBRATION_MIN_ORIGINS:
        return _fallback_portfolio_calibration_unavailable(
            horizon_days,
            status="insufficient_valid_oof_portfolio_observations",
            origin_count=len(frame),
        )
    actual = frame["actual_revenue"].to_numpy(dtype=float)
    p10 = frame["predicted_revenue_p10"].to_numpy(dtype=float)
    p50 = frame["predicted_revenue_p50"].to_numpy(dtype=float)
    p90 = frame["predicted_revenue_p90"].to_numpy(dtype=float)
    residuals = np.log1p(actual) - np.log1p(p50)
    residual_p10 = float(np.quantile(residuals, 0.10))
    residual_p50 = float(np.quantile(residuals, 0.50))
    residual_p90 = float(np.quantile(residuals, 0.90))
    if not all(math.isfinite(value) for value in (residual_p10, residual_p50, residual_p90)):
        return _fallback_portfolio_calibration_unavailable(
            horizon_days,
            status="non_finite_oof_portfolio_residuals",
            origin_count=len(frame),
        )
    calibrated_p10 = np.expm1(np.log1p(p50) + residual_p10 - residual_p50)
    calibrated_p90 = np.expm1(np.log1p(p50) + residual_p90 - residual_p50)
    return {
        "method": FALLBACK_PORTFOLIO_INTERVAL_CALIBRATION_METHOD,
        "status": "available",
        "horizon_days": int(horizon_days),
        "calibration_level": "portfolio",
        "calibration_purpose": "revenue_interval_only_not_roas_probability_calibration",
        "origin_protocol": "expanding_window_non_overlapping_targets",
        "fit_window": "date <= origin",
        "target_window": "origin < date <= origin + horizon_days",
        "quantile_estimator": "empirical_oof_log_residual_quantiles",
        "origin_count": int(len(frame)),
        "nominal_two_sided_coverage": 0.80,
        "residual_p10": round(residual_p10, 8),
        "residual_p50": round(residual_p50, 8),
        "residual_p90": round(residual_p90, 8),
        "base_lower_coverage": round(float(np.mean(actual >= p10)), 8),
        "base_upper_coverage": round(float(np.mean(actual <= p90)), 8),
        "base_joint_coverage": round(float(np.mean((actual >= p10) & (actual <= p90))), 8),
        "calibrated_lower_coverage": round(float(np.mean(actual >= calibrated_p10)), 8),
        "calibrated_upper_coverage": round(float(np.mean(actual <= calibrated_p90)), 8),
        "calibrated_joint_coverage": round(
            float(np.mean((actual >= calibrated_p10) & (actual <= calibrated_p90))), 8
        ),
        "calibration_limitations": "portfolio_revenue_only; not_roas_probability_calibration; temporal_nonstationarity_can_change_coverage",
    }


def _fit_fallback_interval_calibration(canonical: pd.DataFrame, horizon_days: int) -> dict[str, object]:
    """Fit an honest OOF interval-width profile for statistical fallback leaves.

    Every OOF forecast is produced by a fresh fallback model using rows at or
    before its origin; the next horizon is held out. Origins are
    non-overlapping and only observed campaign/target pairs are scored. The final model
    stores the resulting *widths*, never the source data or a training routine,
    so protected inference remains offline, deterministic, and inference-only.

    The profile is campaign-marginal by design. It improves leaf uncertainty
    without asserting a formal portfolio-level coverage guarantee under cross-
    campaign dependence or seasonality shift.
    """
    origins = _non_overlapping_fallback_origins(canonical, horizon_days)
    records: list[pd.DataFrame] = []
    portfolio_records: list[dict[str, object]] = []
    usable_origins = 0
    minimum_target_days = max(1, int(math.ceil(horizon_days * 0.80)))
    for origin in origins:
        train = canonical[canonical["date"] <= origin].copy()
        target = canonical[
            (canonical["date"] > origin)
            & (canonical["date"] <= origin + pd.Timedelta(days=horizon_days))
        ].copy()
        if train["date"].nunique() < FALLBACK_INTERVAL_CALIBRATION_MIN_TRAINING_DAYS:
            continue
        if target["date"].nunique() < minimum_target_days:
            continue
        model = _selection_model(train, {}, {horizon_days: "statistical_fallback"})
        leaves = model.forecast_campaigns(train, horizon_days)
        if leaves.empty:
            continue
        # Build the exact uncalibrated fallback rollup used at inference. The
        # temporary candidate has no fallback profile, so this cannot learn
        # from its own target window or recursively apply a future profile.
        overall = build_forecast(model, train, horizon_days).query("level == 'overall'")
        if not overall.empty:
            overall_row = overall.iloc[0]
            portfolio_records.append(
                {
                    "origin": origin,
                    "actual_revenue": float(target["revenue"].sum()),
                    "predicted_revenue_p10": float(overall_row["predicted_revenue_p10"]),
                    "predicted_revenue_p50": float(overall_row["predicted_revenue_p50"]),
                    "predicted_revenue_p90": float(overall_row["predicted_revenue_p90"]),
                }
            )
        actual = (
            target.assign(
                campaign_key=(target["source_system"].astype(str) + ":" + target["source_campaign_id"].astype(str))
            )
            .groupby("campaign_key", as_index=False, sort=False)["revenue"]
            .sum()
            .rename(columns={"revenue": "actual_revenue"})
        )
        matched = leaves.loc[
            :,
            [
                "campaign_key",
                "predicted_revenue_p10",
                "predicted_revenue_p50",
                "predicted_revenue_p90",
            ],
        ].merge(actual, on="campaign_key", how="inner", validate="one_to_one")
        if matched.empty:
            continue
        for column in (
            "predicted_revenue_p10",
            "predicted_revenue_p50",
            "predicted_revenue_p90",
            "actual_revenue",
        ):
            matched[column] = pd.to_numeric(matched[column], errors="coerce")
        matched = matched.replace([np.inf, -np.inf], np.nan).dropna().copy()
        matched = matched[
            (matched["actual_revenue"] >= 0.0)
            & (matched["predicted_revenue_p10"] >= 0.0)
            & (matched["predicted_revenue_p10"] <= matched["predicted_revenue_p50"])
            & (matched["predicted_revenue_p50"] <= matched["predicted_revenue_p90"])
            & (matched["predicted_revenue_p50"] > 0.0)
        ].copy()
        if matched.empty:
            continue
        matched["origin"] = origin
        records.append(matched)
        usable_origins += 1

    samples = int(sum(len(record) for record in records))
    portfolio_profile = _fit_fallback_portfolio_interval_profile(portfolio_records, horizon_days)
    if usable_origins < FALLBACK_INTERVAL_CALIBRATION_MIN_ORIGINS:
        unavailable = _fallback_calibration_unavailable(
            horizon_days,
            status="insufficient_non_overlapping_oof_origins",
            origin_count=usable_origins,
            sample_count=samples,
        )
        unavailable["portfolio_interval_profile"] = portfolio_profile
        return unavailable
    if samples < FALLBACK_INTERVAL_CALIBRATION_MIN_SAMPLES:
        unavailable = _fallback_calibration_unavailable(
            horizon_days,
            status="insufficient_oof_campaign_observations",
            origin_count=usable_origins,
            sample_count=samples,
        )
        unavailable["portfolio_interval_profile"] = portfolio_profile
        return unavailable

    frame = pd.concat(records, ignore_index=True)
    actual = frame["actual_revenue"].to_numpy(dtype=float)
    p10 = frame["predicted_revenue_p10"].to_numpy(dtype=float)
    p50 = frame["predicted_revenue_p50"].to_numpy(dtype=float)
    p90 = frame["predicted_revenue_p90"].to_numpy(dtype=float)
    lower_required, upper_required = _required_interval_widths(actual, p10, p50, p90)
    lower_width = _finite_sample_conformal_width(lower_required)
    upper_width = _finite_sample_conformal_width(upper_required)
    adjusted_p10, adjusted_p90 = _apply_interval_widths(p10, p50, p90, lower_width, upper_width)
    base_lower_coverage = float(np.mean(actual >= p10))
    base_upper_coverage = float(np.mean(actual <= p90))
    calibrated_lower_coverage = float(np.mean(actual >= adjusted_p10))
    calibrated_upper_coverage = float(np.mean(actual <= adjusted_p90))
    profile: dict[str, object] = {
        "method": FALLBACK_INTERVAL_CALIBRATION_METHOD,
        "status": "available",
        "horizon_days": int(horizon_days),
        "calibration_level": "campaign_marginal",
        "origin_protocol": "expanding_window_non_overlapping_targets",
        "fit_window": "date <= origin",
        "target_window": "origin < date <= origin + horizon_days",
        "minimum_training_days": FALLBACK_INTERVAL_CALIBRATION_MIN_TRAINING_DAYS,
        "origin_count": int(usable_origins),
        "sample_count": samples,
        "nominal_two_sided_coverage": 0.80,
        "nominal_one_sided_coverage": FALLBACK_INTERVAL_CALIBRATION_TAIL_COVERAGE,
        "lower_width_multiplier": round(float(lower_width), 8),
        "upper_width_multiplier": round(float(upper_width), 8),
        "max_width_multiplier": FALLBACK_INTERVAL_CALIBRATION_MAX_WIDTH,
        "base_lower_coverage": round(base_lower_coverage, 8),
        "base_upper_coverage": round(base_upper_coverage, 8),
        "base_joint_coverage": round(float(np.mean((actual >= p10) & (actual <= p90))), 8),
        "calibrated_lower_coverage": round(calibrated_lower_coverage, 8),
        "calibrated_upper_coverage": round(calibrated_upper_coverage, 8),
        "calibrated_joint_coverage": round(float(np.mean((actual >= adjusted_p10) & (actual <= adjusted_p90))), 8),
        "calibration_limitations": "campaign_marginal_oof; temporal nonstationarity can change future coverage",
    }
    profile["portfolio_interval_profile"] = portfolio_profile
    return profile


def _canonical_training_fingerprint(canonical: pd.DataFrame) -> str:
    """Fingerprint the exact canonical data contract used to train an artifact."""
    columns = list(CANONICAL_COLUMNS)
    payload = canonical.loc[:, columns].sort_values(columns, kind="stable").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_schema_fingerprint(direct_models: dict[int, object]) -> str:
    """Fingerprint one-hot category vocabularies and numerical feature order."""
    payload = {
        "numeric_features": list(NUMERIC_FEATURES),
        "direct_models": {
            str(horizon): {
                "model_class": type(model).__name__,
                "category_columns": list(model.category_columns),
                "categories": {name: list(values) for name, values in sorted(model.categories.items())},
                "scenario_weight": getattr(model, "scenario_weight", None),
                "selection_metric": getattr(model, "selection_metric", None),
            }
            for horizon, model in sorted(direct_models.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _statistical_components(canonical: pd.DataFrame) -> tuple[float, float, dict[int, float]]:
    """Fit the compact hierarchical fallback parameters from one data slice."""
    valid = canonical[(canonical["spend"] > 0) & (canonical["revenue"] >= 0)].copy()
    if valid.empty:
        raise ValueError("Cannot train HorizonModel without positive-spend historical rows")
    roas = float(valid["revenue"].sum() / max(valid["spend"].sum(), 1e-9))
    daily_ratio = (valid["revenue"] / valid["spend"]).clip(lower=1e-5)
    sigma = max(0.20, min(float(daily_ratio.map(math.log).std(ddof=0)), 1.25))
    valid["month"] = valid["date"].dt.month
    monthly = valid.groupby("month")[["revenue", "spend"]].sum()
    factors = (monthly["revenue"] / monthly["spend"].clip(lower=1e-9) / roas).clip(0.55, 1.45)
    return roas, sigma, {int(month): float(value) for month, value in factors.items()}


def _selection_model(
    canonical: pd.DataFrame,
    direct_models: dict[int, object],
    selected_families: dict[int, str],
) -> HorizonModel:
    """Construct an in-memory candidate for a temporal model-family tournament."""
    roas, sigma, factors = _statistical_components(canonical)
    return HorizonModel(
        "horizon-temporal-tournament-candidate-v1",
        roas,
        sigma,
        factors,
        direct_models,
        selected_model_families=selected_families,
    )


def _select_model_families(
    canonical: pd.DataFrame,
    horizons: tuple[int, ...],
) -> tuple[dict[int, str], dict[int, dict[str, object]]]:
    """Choose direct vs statistical serving with a purged terminal tournament.

    At each horizon, both candidates see only rows available at the tournament
    origin and are scored against the following, non-overlapping target window.
    The final artifact is refit after selection, but the decision record keeps
    the exact origin, absolute errors, and winner. A tie deliberately selects
    the lower-variance statistical fallback.
    """
    latest = pd.Timestamp(canonical["date"].max())
    selected: dict[int, str] = {}
    records: dict[int, dict[str, object]] = {}
    for horizon in horizons:
        cutoff = latest - pd.Timedelta(days=horizon)
        train = canonical[canonical["date"] <= cutoff].copy()
        actual = canonical[(canonical["date"] > cutoff) & (canonical["date"] <= cutoff + pd.Timedelta(days=horizon))]
        if train["date"].nunique() < 90 or actual.empty:
            selected[horizon] = "statistical_fallback"
            records[horizon] = {
                "status": "fallback_insufficient_selection_history",
                "selected_family": "statistical_fallback",
                "cutoff": str(cutoff.date()),
            }
            continue
        direct = fit_direct_ensemble(train, horizon)
        fallback_model = _selection_model(train, {}, {horizon: "statistical_fallback"})
        fallback_value = float(
            build_forecast(fallback_model, train, horizon).query("level == 'overall'").iloc[0]["predicted_revenue_p50"]
        )
        actual_value = float(actual["revenue"].sum())
        fallback_error = abs(actual_value - fallback_value)
        if direct is None:
            selected[horizon] = "statistical_fallback"
            records[horizon] = {
                "status": "fallback_direct_candidate_unavailable",
                "selected_family": "statistical_fallback",
                "cutoff": str(cutoff.date()),
                "actual_revenue": actual_value,
                "statistical_p50": fallback_value,
                "statistical_absolute_error": fallback_error,
            }
            continue
        direct_model = _selection_model(train, {horizon: direct}, {horizon: "direct_ensemble"})
        direct_value = float(
            build_forecast(direct_model, train, horizon).query("level == 'overall'").iloc[0]["predicted_revenue_p50"]
        )
        direct_error = abs(actual_value - direct_value)
        winner = "direct_ensemble" if direct_error < fallback_error else "statistical_fallback"
        selected[horizon] = winner
        records[horizon] = {
            "status": "purged_terminal_tournament",
            "selected_family": winner,
            "cutoff": str(cutoff.date()),
            "actual_revenue": actual_value,
            "statistical_p50": fallback_value,
            "direct_p50": direct_value,
            "statistical_absolute_error": fallback_error,
            "direct_absolute_error": direct_error,
            "direct_candidate_type": type(direct).__name__,
            "direct_scenario_weight": getattr(direct, "scenario_weight", None),
        }
    return selected, records


def fit_horizon_model(
    canonical: pd.DataFrame,
    train_direct: bool = True,
    train_dependence: bool = True,
    horizons: tuple[int, ...] = (30, 60, 90),
) -> HorizonModel:
    """Train a model artifact outside the evaluator/import path."""
    roas, sigma, factors = _statistical_components(canonical)
    direct_models = {
        horizon: model
        for horizon in horizons
        if (model := fit_direct_ensemble(canonical, horizon)) is not None
    } if train_direct else {}
    selected_families, selection_records = (
        _select_model_families(canonical, horizons) if train_direct and direct_models
        else ({horizon: "statistical_fallback" for horizon in horizons}, {
            horizon: {"status": "direct_training_disabled", "selected_family": "statistical_fallback"}
            for horizon in horizons
        })
    )
    # Calibrate only horizons actually served by the statistical fallback. A
    # direct model's uncertainty has a separate purged temporal calibration;
    # applying fallback residuals to it would mix incompatible error models.
    fallback_interval_calibration = {
        horizon: _fit_fallback_interval_calibration(canonical, horizon)
        for horizon in horizons
        if selected_families.get(horizon) == "statistical_fallback"
    }
    # Dependence is calibrated only in this training module.  The persisted
    # profile lets evaluator-safe inference assemble joint portfolio draws
    # without fitting, downloading, or retaining source history in memory.
    residual_dependence = {
        horizon: fit_residual_dependence(canonical, horizon)
        for horizon in direct_models
        if selected_families.get(horizon) == "direct_ensemble"
    } if train_dependence and direct_models else {}
    return HorizonModel(
        "horizon-temporal-tournament-v8-calendar-plan-oof-intervals"
        if direct_models
        else "horizon-statistical-v10-calendar-plan-oof-intervals",
        roas,
        sigma,
        factors,
        direct_models,
        training_data_fingerprint=_canonical_training_fingerprint(canonical),
        feature_schema_fingerprint=_feature_schema_fingerprint(direct_models),
        selected_model_families=selected_families,
        model_selection=selection_records,
        fallback_interval_calibration=fallback_interval_calibration,
        residual_dependence=residual_dependence,
    )
