from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping
import math

import pandas as pd

from product.decisioning.scenario import simulate_budget_plan
from src.forecast import build_forecast
from src.model import HorizonModel


@dataclass(frozen=True)
class OptimizationResult:
    forecast: pd.DataFrame
    campaign_budgets: dict[str, float]
    status: str
    explanation: str


def _fallback_marginal_revenue_per_dollar(base_budget: float, base_roas: float, allocated: float) -> float:
    baseline = max(base_budget, 1.0)
    scale = max(baseline * 1.5, 1.0)
    normalization = baseline / (scale * math.log1p(baseline / scale))
    return max(0.0, base_roas * normalization / (1.0 + allocated / scale))


def recommend_allocation(
    model: HorizonModel,
    canonical: pd.DataFrame,
    horizon_days: int,
    total_budget: float,
    target_roas: float | None = None,
    channel_minimums: Mapping[str, float] | None = None,
    channel_maximums: Mapping[str, float] | None = None,
    max_multiple_of_baseline: float = 2.0,
    increments: int = 160,
) -> OptimizationResult:
    if horizon_days not in {30, 60, 90}:
        raise ValueError("Horizon must be 30, 60, or 90 days")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if max_multiple_of_baseline <= 0:
        raise ValueError("max_multiple_of_baseline must be positive")
    minimums = {str(key): float(value) for key, value in (channel_minimums or {}).items()}
    maximums = {str(key): float(value) for key, value in (channel_maximums or {}).items()}
    if any(value < 0 for value in [*minimums.values(), *maximums.values()]):
        raise ValueError("Channel constraints cannot be negative")
    if sum(minimums.values()) > total_budget + 1e-6:
        raise ValueError("Scenario infeasible: channel minimums exceed total budget")
    for channel, minimum in minimums.items():
        if channel in maximums and minimum > maximums[channel]:
            raise ValueError(f"Scenario infeasible: minimum exceeds maximum for {channel}")

    baseline = build_forecast(model, canonical, horizon_days)
    leaves = baseline[baseline["level"] == "campaign"].copy()
    leaves = leaves[leaves["planned_budget"] > 0].copy()
    if leaves.empty:
        raise ValueError("No active campaigns with baseline budget are available")
    known_channels = set(leaves["channel"])
    unknown = (set(minimums) | set(maximums)) - known_channels
    if unknown:
        raise ValueError(f"Scenario includes inactive/unknown channels: {', '.join(sorted(unknown))}")

    allocations = {str(row.campaign_key): 0.0 for row in leaves.itertuples()}
    channel_allocated: defaultdict[str, float] = defaultdict(float)
    target_relaxed = False
    by_channel = {channel: group.copy() for channel, group in leaves.groupby("channel")}
    step = max(100.0, round(total_budget / max(increments, 1) / 100.0) * 100.0)
    as_of = canonical["date"].max()
    history_by_key = {
        model.campaign_key(source_system, campaign_id): group.copy()
        for (source_system, campaign_id), group in canonical.groupby(["source_system", "source_campaign_id"], dropna=False)
    }

    def marginal_response(row: object, current: float, addition: float) -> tuple[float, float]:
        """Score the exact direct-ridge response used by the forecast when available."""
        campaign_key = str(row.campaign_key)
        history = history_by_key[campaign_key]
        current_quantiles = model.direct_quantiles(
            history, str(row.channel), str(row.campaign_type), as_of, horizon_days, current,
        )
        next_budget = current + addition
        next_quantiles = model.direct_quantiles(
            history, str(row.channel), str(row.campaign_type), as_of, horizon_days, next_budget,
        )
        if current_quantiles is not None and next_quantiles is not None:
            # A direct model may have an intercept. A zero media plan must
            # still represent zero paid-media revenue in the allocator.
            current_revenue = 0.0 if current <= 1e-9 else current_quantiles[1]
            next_revenue = next_quantiles[1]
            return max(0.0, (next_revenue - current_revenue) / max(addition, 1e-9)), next_revenue / max(next_budget, 1e-9)
        return (
            _fallback_marginal_revenue_per_dollar(float(row.planned_budget), float(row.predicted_roas_p50), current),
            float(row.predicted_roas_p50),
        )

    def add_to_best(channel: str, amount: float) -> float:
        remaining = amount
        group = by_channel[channel]
        while remaining > 1e-6:
            candidates = []
            for row in group.itertuples():
                campaign_id = str(row.campaign_key)
                cap = float(row.planned_budget) * max_multiple_of_baseline
                current = allocations[campaign_id]
                if current + 1e-6 >= cap:
                    continue
                candidate_addition = min(step, remaining, cap - current)
                marginal, _ = marginal_response(row, current, candidate_addition)
                candidates.append((marginal, campaign_id, cap))
            if not candidates:
                break
            _, campaign_id, cap = max(candidates, key=lambda item: (item[0], item[1]))
            addition = min(step, remaining, cap - allocations[campaign_id])
            allocations[campaign_id] += addition
            channel_allocated[channel] += addition
            remaining -= addition
        return remaining

    for channel, minimum in sorted(minimums.items()):
        unallocated = add_to_best(channel, minimum)
        if unallocated > 1e-6:
            raise ValueError(f"Scenario infeasible: {channel} minimum exceeds campaign support cap")

    def candidates_for_remaining(remaining_budget: float, enforce_target: bool) -> list[tuple[float, str, str, float, float]]:
        candidates = []
        for row in leaves.itertuples():
            campaign_id = str(row.campaign_key)
            channel = str(row.channel)
            cap = float(row.planned_budget) * max_multiple_of_baseline
            current = allocations[campaign_id]
            channel_cap = maximums.get(channel, math.inf)
            if current + 1e-6 >= cap or channel_allocated[channel] + 1e-6 >= channel_cap:
                continue
            candidate_addition = min(step, remaining_budget, cap - current, channel_cap - channel_allocated[channel])
            if candidate_addition <= 1e-6:
                continue
            marginal, candidate_roas = marginal_response(row, current, candidate_addition)
            if target_roas is not None and candidate_roas < target_roas:
                if enforce_target:
                    continue
                # The caller records that a hard target constraint had to be
                # relaxed to return a feasible full-budget plan.
                marginal *= 0.55
            candidates.append((marginal, campaign_id, channel, cap, channel_cap))
        return candidates

    remaining = total_budget - sum(allocations.values())
    while remaining > 1e-6:
        candidates = candidates_for_remaining(remaining, enforce_target=target_roas is not None)
        if not candidates and target_roas is not None:
            candidates = candidates_for_remaining(remaining, enforce_target=False)
            target_relaxed = bool(candidates)
        if not candidates:
            raise ValueError("Scenario infeasible: total budget exceeds all campaign and channel caps")
        _, campaign_id, channel, cap, channel_cap = max(candidates, key=lambda item: (item[0], item[1]))
        addition = min(step, remaining, cap - allocations[campaign_id], channel_cap - channel_allocated[channel])
        if addition <= 1e-6:
            break
        allocations[campaign_id] += addition
        channel_allocated[channel] += addition
        remaining -= addition

    if remaining > 1e-4:
        raise ValueError("Scenario infeasible: unable to allocate the full budget")
    forecast = simulate_budget_plan(model, canonical, horizon_days, allocations, target_roas)
    return OptimizationResult(
        forecast=forecast,
        campaign_budgets=allocations,
        status="feasible",
        explanation=(
            "Allocation uses discrete marginal returns from the same direct-ridge response used by scenario forecasts "
            "when available, with a documented fallback curve, campaign support caps, channel constraints, and a "
            f"{'controlled ROAS-target relaxation because no fully feasible allocation met the guardrail' if target_relaxed else 'hard ROAS guardrail while feasible'}. "
            "The selected allocation is reforecast through the shared probabilistic model."
        ),
    )
