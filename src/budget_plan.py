"""Deterministic baseline media-plan estimation shared by training and inference.

The evaluator does not guarantee a future media plan.  A direct revenue model
therefore needs one stable default that can be reconstructed at a historical
forecast origin as well as at evaluator inference time.  This module contains
only past-data plan estimation; it does not forecast revenue, fit a model, or
perform any network access.
"""
from __future__ import annotations

import math

import pandas as pd


PRIOR_YEAR_MINIMUM_COVERAGE = 0.80


def seasonal_default_budget_multiplier(
    canonical: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_days: int,
) -> float:
    """Return the bounded portfolio month-mix adjustment for a 90-day plan."""
    if horizon_days != 90:
        return 1.0
    historical = canonical[canonical["date"] <= as_of]
    recent = historical[(historical["date"] >= as_of - pd.Timedelta(days=27)) & (historical["date"] <= as_of)]
    recent_daily_spend = float(recent["spend"].sum()) / 28.0
    if recent_daily_spend <= 0.0:
        return 1.0
    portfolio_daily = historical.groupby("date", sort=True)["spend"].sum()
    if portfolio_daily.empty:
        return 1.0
    month_daily_mean = portfolio_daily.groupby(portfolio_daily.index.month).mean()
    future_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    seasonal_daily_spend = sum(float(month_daily_mean.get(date.month, recent_daily_spend)) for date in future_dates) / horizon_days
    return max(0.10, min(2.00, seasonal_daily_spend / recent_daily_spend))


def calendar_matched_prior_year_budget(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_days: int,
    *,
    minimum_coverage: float = PRIOR_YEAR_MINIMUM_COVERAGE,
) -> tuple[float | None, float]:
    """Return a prior-year campaign plan and evidence coverage when supported.

    Each future calendar date is mapped to the same date in the prior year.
    This preserves holiday and promotional pacing without looking beyond the
    forecast origin. A plan is accepted only when at least 80% of the mapped
    daily observations exist. A zero prior-year plan is not treated as evidence
    to suppress an actively delivering campaign, so callers can safely fall
    back to a current run rate.
    """
    if horizon_days <= 0 or history.empty:
        return None, 0.0
    future_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    prior_dates = pd.DatetimeIndex([date - pd.DateOffset(years=1) for date in future_dates])
    daily = history[history["date"] <= as_of].groupby("date", sort=True)["spend"].sum()
    if daily.empty:
        return None, 0.0
    matched = daily.reindex(prior_dates)
    observed = matched.notna()
    coverage = float(observed.mean()) if len(matched) else 0.0
    if coverage < minimum_coverage:
        return None, coverage
    total = float(matched.fillna(0.0).sum())
    if not math.isfinite(total) or total <= 0.0:
        return None, coverage
    return max(0.0, total), coverage


def campaign_baseline_budget(
    canonical: pd.DataFrame,
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_days: int,
) -> tuple[float, str, float]:
    """Build the evaluator-visible plan with calendar and run-rate fallbacks.

    Returns ``(budget, method, prior_year_coverage)``.  The method is emitted
    as a quality flag by inference and is intentionally deterministic so the
    training expert can use exactly the same value at each historical origin.
    """
    recent = history[(history["date"] > as_of - pd.Timedelta(days=28)) & (history["date"] <= as_of)]
    active_days = int(recent["date"].nunique())
    recent_spend = float(recent["spend"].sum())
    if active_days <= 0 or recent_spend <= 0.0:
        return 0.0, "no_recent_delivery", 0.0
    seasonal = seasonal_default_budget_multiplier(canonical, as_of, horizon_days)
    run_rate_budget = recent_spend / active_days * horizon_days * seasonal
    prior_budget, coverage = calendar_matched_prior_year_budget(history, as_of, horizon_days)
    if prior_budget is not None:
        # Calendar history is valuable, but a one-off promotion or a campaign
        # restructure can otherwise create an unbounded default. Bound the
        # prior-year plan against the currently observed active run rate.
        return (
            min(max(prior_budget, 0.25 * run_rate_budget), 4.0 * run_rate_budget),
            "calendar_matched_prior_year_budget",
            coverage,
        )
    return run_rate_budget, "recent_run_rate_budget", coverage
