from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.contracts import CANONICAL_COLUMNS
from src.model import HorizonModel

# Fixed percentile grid keeps offline inference deterministic (no RNG) while
# enabling draw-level hierarchy aggregation and ROAS probabilities.  Individual
# leaf paths contain each grid percentile once; a trained residual-factor
# copula determines their *joint ordering* across campaigns.
DRAW_PERCENTILES = tuple(i / 100.0 for i in range(1, 100))
_NORMAL = NormalDist()


def _roas(revenue: np.ndarray | float, spend: np.ndarray | float) -> np.ndarray | float:
    return revenue / np.clip(spend, 1e-9, None)


def _interp_quantile(p10: float, p50: float, p90: float, percentile: float) -> float:
    """Piecewise log-linear interpolation through empirical P10/P50/P90 knots."""
    eps = 1e-9
    low = max(float(p10), 0.0)
    mid = max(float(p50), 0.0)
    high = max(float(p90), 0.0)
    mid = max(mid, low)
    high = max(high, mid)
    p = min(max(float(percentile), DRAW_PERCENTILES[0]), DRAW_PERCENTILES[-1])
    if low <= eps and mid <= eps and high <= eps:
        return 0.0
    if p <= 0.10:
        return low
    if p == 0.50:
        return mid
    if p == 0.90:
        return high
    if p < 0.50:
        if mid <= low + eps:
            return mid
        t = (p - 0.10) / 0.40
        return math.exp((1.0 - t) * math.log(low + eps) + t * math.log(mid + eps)) - eps
    if p < 0.90:
        if high <= mid + eps:
            return high
        t = (p - 0.50) / 0.40
        return math.exp((1.0 - t) * math.log(mid + eps) + t * math.log(high + eps)) - eps
    return high


def _grid_quantile(draws: np.ndarray, percentile: float) -> float:
    """Return an empirical grid quantile without assuming draw order.

    Under the previous comonotonic implementation draw index and percentile
    were interchangeable.  Dependence-aware paths are intentionally permuted,
    so summaries must sort each marginal before selecting the P10/P50/P90
    grid knots.  The fixed 1%-99% grid includes all reported knots exactly.
    """
    ordered = np.sort(np.asarray(draws, dtype=float), kind="stable")
    index = min(len(ordered) - 1, max(0, int(round(float(percentile) * 100.0)) - 1))
    return float(ordered[index])


def _leaf_draw_paths(row: pd.Series, percentiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    revenue = np.asarray(
        [
            _interp_quantile(row["predicted_revenue_p10"], row["predicted_revenue_p50"], row["predicted_revenue_p90"], p)
            for p in percentiles
        ],
        dtype=float,
    )
    spend = np.asarray(
        [
            _interp_quantile(row["predicted_spend_p10"], row["predicted_spend_p50"], row["predicted_spend_p90"], p)
            for p in percentiles
        ],
        dtype=float,
    )
    return revenue, spend


def _deterministic_factor_path(label: str, draws: int = len(DRAW_PERCENTILES)) -> np.ndarray:
    """Return a stable, stratified standard-normal factor path.

    Hash-sorted percentile strata provide a deterministic alternative to a
    seeded pseudo-random generator.  Every factor has the same 1%-99% marginal
    normal grid but a different ordering, which is both reproducible across
    evaluator runs and suitable for rank-copula construction.
    """
    order = sorted(
        range(draws),
        key=lambda index: hashlib.sha256(f"horizon-factor-copula-v1|{label}|{index}".encode("utf-8")).digest(),
    )
    path = np.empty(draws, dtype=float)
    for rank, index in enumerate(order):
        path[index] = _NORMAL.inv_cdf((rank + 1) / (draws + 1))
    return path


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Map latent values onto the exact fixed marginal percentile grid."""
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    percentiles = np.empty(len(order), dtype=float)
    percentiles[order] = np.asarray(DRAW_PERCENTILES, dtype=float)
    return percentiles


def _dependence_profile(model: HorizonModel, horizon_days: int) -> tuple[str, dict[str, float]]:
    """Validate a persisted profile and return a safe latent-factor fallback.

    Legacy model artifacts do not carry residual-dependence metadata.  They
    use independent deterministic ranks rather than the old, unjustified
    perfectly comonotonic rollup.  A malformed profile follows the same safe
    route and is surfaced in forecast quality flags.
    """
    profile = model.dependence_for_horizon(horizon_days)
    method = str(profile.get("method", "independent_rank_fallback_v1"))
    raw = profile.get("factor_weights", {})
    names = ("global", "channel", "campaign_type", "idiosyncratic")
    if method != "hierarchical_residual_factor_copula_v1" or not isinstance(raw, dict):
        return "independent_rank_fallback_v1", {"global": 0.0, "channel": 0.0, "campaign_type": 0.0, "idiosyncratic": 1.0}
    try:
        weights = {name: max(0.0, float(raw.get(name, 0.0))) for name in names}
    except (TypeError, ValueError):
        return "independent_rank_fallback_v1", {"global": 0.0, "channel": 0.0, "campaign_type": 0.0, "idiosyncratic": 1.0}
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0.0:
        return "independent_rank_fallback_v1", {"global": 0.0, "channel": 0.0, "campaign_type": 0.0, "idiosyncratic": 1.0}
    return method, {name: value / total for name, value in weights.items()}


def _dependence_percentile_paths(
    model: HorizonModel,
    horizon_days: int,
    leaves: pd.DataFrame,
) -> tuple[list[np.ndarray], str]:
    """Build deterministic, dependence-aware marginal rank paths per leaf.

    A hierarchical Gaussian copula uses historical OOF residual variance
    components as global, channel, campaign-type, and idiosyncratic weights.
    The final rank transform preserves each campaign's calibrated P10/P50/P90
    marginal distribution exactly while allowing only historically supported
    shared shocks to co-move.  Runtime is O(campaigns * draws) and contains no
    fitting or random sampling.
    """
    method, weights = _dependence_profile(model, horizon_days)
    global_path = _deterministic_factor_path("global") if weights["global"] > 0.0 else np.zeros(len(DRAW_PERCENTILES))
    paths: list[np.ndarray] = []
    for _, leaf in leaves.iterrows():
        campaign_key = str(leaf["campaign_key"])
        channel = str(leaf["channel"])
        campaign_type = str(leaf["campaign_type"])
        latent = np.zeros(len(DRAW_PERCENTILES), dtype=float)
        if weights["global"] > 0.0:
            latent += math.sqrt(weights["global"]) * global_path
        if weights["channel"] > 0.0:
            latent += math.sqrt(weights["channel"]) * _deterministic_factor_path(f"channel:{channel}")
        if weights["campaign_type"] > 0.0:
            latent += math.sqrt(weights["campaign_type"]) * _deterministic_factor_path(
                f"campaign_type:{channel}:{campaign_type}"
            )
        if weights["idiosyncratic"] > 0.0:
            latent += math.sqrt(weights["idiosyncratic"]) * _deterministic_factor_path(f"campaign:{campaign_key}")
        paths.append(_rank_percentiles(latent))
    return paths, method


def _portfolio_oof_residual_profile(model: HorizonModel, horizon_days: int) -> dict[str, float] | None:
    """Validate the empirical OOF portfolio-residual calibration profile."""
    profile = model.dependence_for_horizon(horizon_days)
    calibration = profile.get("portfolio_oof_residual_calibration", {}) if isinstance(profile, dict) else {}
    if not isinstance(calibration, dict):
        return None
    if calibration.get("method") != "expanding_window_oof_portfolio_log_residual_quantiles_v1":
        return None
    try:
        values = {name: float(calibration[name]) for name in ("residual_p10", "residual_p50", "residual_p90")}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    if values["residual_p10"] > values["residual_p50"] or values["residual_p50"] > values["residual_p90"]:
        return None
    return values


def _portfolio_oof_residual_recalibration(
    revenue_draws: np.ndarray,
    model: HorizonModel,
    horizon_days: int,
    point_forecast: float,
) -> tuple[np.ndarray, bool]:
    """Calibrate portfolio revenue paths with centered OOF aggregate residuals.

    The factor copula supplies campaign co-movement.  This final monotone rank
    transform applies an empirical, leakage-safe *portfolio* residual
    distribution around the factor-copula portfolio point estimate.  It keeps
    scenario draw ordering (and hence revenue/spend dependence) intact while
    correcting the common undercoverage caused by naïve diversification.
    """
    calibration = _portfolio_oof_residual_profile(model, horizon_days)
    if calibration is None:
        return revenue_draws, False
    # Sum of campaign P50 point forecasts is the portfolio point estimate.
    # It avoids allowing strongly skewed leaf tails to move the portfolio
    # median solely because many independent draws happen to land above P50.
    point = max(0.0, float(point_forecast))
    if point <= 0.0:
        return revenue_draws, False
    order = np.argsort(np.asarray(revenue_draws, dtype=float), kind="stable")
    adjusted = np.empty(len(order), dtype=float)
    for rank, index in enumerate(order):
        percentile = DRAW_PERCENTILES[rank]
        if percentile <= 0.10:
            residual = calibration["residual_p10"]
        elif percentile < 0.50:
            fraction = (percentile - 0.10) / 0.40
            residual = (1.0 - fraction) * calibration["residual_p10"] + fraction * calibration["residual_p50"]
        elif percentile == 0.50:
            residual = calibration["residual_p50"]
        elif percentile < 0.90:
            fraction = (percentile - 0.50) / 0.40
            residual = (1.0 - fraction) * calibration["residual_p50"] + fraction * calibration["residual_p90"]
        else:
            residual = calibration["residual_p90"]
        # Preserve the model's P50 point forecast. The OOF median captures
        # historical bias; subtracting it turns residual quantiles into an
        # uncertainty spread rather than silently shifting the point estimate.
        centered_residual = residual - calibration["residual_p50"]
        adjusted[index] = max(0.0, math.expm1(math.log1p(point) + centered_residual))
    return adjusted, True


def _reconcile_portfolio_oof_residual_calibration(
    leaf_revenue: list[np.ndarray],
    model: HorizonModel,
    horizon_days: int,
    point_forecast: float,
) -> tuple[list[np.ndarray], bool]:
    """Apply a calibrated aggregate shock proportionally to each leaf path.

    The earlier implementation changed only the overall row. This routine
    preserves exact draw-level reconciliation by scaling each campaign's
    existing composition within a draw. A zero low-percentile draw uses P50
    campaign weights so the calibrated aggregate still reconciles.
    """
    if not leaf_revenue:
        return leaf_revenue, False
    base = np.vstack([np.asarray(path, dtype=float) for path in leaf_revenue])
    base_total = np.sum(base, axis=0)
    calibrated_total, applied = _portfolio_oof_residual_recalibration(
        base_total, model, horizon_days, point_forecast
    )
    if not applied:
        return leaf_revenue, False
    adjusted = base.copy()
    median_index = len(DRAW_PERCENTILES) // 2
    fallback_weights = np.maximum(base[:, median_index], 0.0)
    if float(fallback_weights.sum()) <= 1e-12:
        fallback_weights = np.ones(len(base), dtype=float)
    fallback_weights = fallback_weights / fallback_weights.sum()
    for index, target_total in enumerate(calibrated_total):
        current_total = float(base_total[index])
        if current_total > 1e-12:
            adjusted[:, index] = base[:, index] * (float(target_total) / current_total)
        elif float(target_total) > 0.0:
            adjusted[:, index] = fallback_weights * float(target_total)
        else:
            adjusted[:, index] = 0.0
    return [adjusted[index].copy() for index in range(len(adjusted))], True


def _summarize_draws(
    revenue_draws: np.ndarray,
    spend_draws: np.ndarray,
    planned_budget: float,
    target_roas: float,
    level: str,
    group: dict[str, object],
    quality_flags: str,
    revenue_p50_override: float | None = None,
) -> dict[str, object]:
    roas_draws = np.asarray(_roas(revenue_draws, spend_draws), dtype=float)
    probability = float(np.mean(roas_draws >= float(target_roas)))
    revenue_p10 = _grid_quantile(revenue_draws, 0.10)
    revenue_p50 = (
        max(0.0, float(revenue_p50_override))
        if revenue_p50_override is not None
        else _grid_quantile(revenue_draws, 0.50)
    )
    revenue_p90 = _grid_quantile(revenue_draws, 0.90)
    # Quantile sums are not generally coherent under a dependence model.
    # The published P50 is therefore reconciled additively through the
    # hierarchy, while uncertainty and threshold probability remain joint-draw
    # statistics. Preserve ordered reported quantiles around that point value.
    revenue_p10 = min(revenue_p10, revenue_p50)
    revenue_p90 = max(revenue_p90, revenue_p50)
    spend_p10 = _grid_quantile(spend_draws, 0.10)
    spend_p50 = _grid_quantile(spend_draws, 0.50)
    spend_p90 = _grid_quantile(spend_draws, 0.90)
    roas_p10 = _grid_quantile(roas_draws, 0.10)
    roas_p50 = revenue_p50 / max(spend_p50, 1e-9)
    roas_p90 = _grid_quantile(roas_draws, 0.90)
    roas_p10 = min(roas_p10, roas_p50)
    roas_p90 = max(roas_p90, roas_p50)
    row: dict[str, object] = {
        "level": level,
        "channel": group.get("channel", "ALL"),
        "campaign_type": group.get("campaign_type", "ALL"),
        "campaign_id": group.get("campaign_id", "ALL"),
        "campaign_name": group.get("campaign_name", "ALL"),
        "planned_budget": float(planned_budget),
        "predicted_revenue_p10": revenue_p10,
        "predicted_revenue_p50": revenue_p50,
        "predicted_revenue_p90": revenue_p90,
        "predicted_spend_p10": spend_p10,
        "predicted_spend_p50": spend_p50,
        "predicted_spend_p90": spend_p90,
        "predicted_roas_p10": roas_p10,
        "predicted_roas_p50": roas_p50,
        "predicted_roas_p90": roas_p90,
        "probability_roas_above_target": round(max(0.0, min(1.0, probability)), 4),
        "quality_flags": quality_flags,
    }
    if "campaign_key" in group:
        row["campaign_key"] = group["campaign_key"]
    return row


def _risk(row: pd.Series) -> float:
    width = (row["predicted_revenue_p90"] - row["predicted_revenue_p10"]) / max(row["predicted_revenue_p50"], 1e-9)
    chance_miss = 1.0 - float(row["probability_roas_above_target"])
    return min(
        100.0,
        round(
            100.0
            * min(
                1.0,
                0.55 * chance_miss
                + 0.25 * min(width / 3.0, 1.0)
                + (0.20 if "extrapolation" in str(row["quality_flags"]) else 0.0),
            ),
            1,
        ),
    )


def _aggregate_level(
    leaf_frame: pd.DataFrame,
    leaf_revenue: list[np.ndarray],
    leaf_spend: list[np.ndarray],
    group_columns: list[str],
    level: str,
    target_roas: float,
    rollup_flag: str,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for key, subset in leaf_frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        group_map = dict(zip(group_columns, key, strict=True))
        positions = subset["_draw_index"].astype(int).tolist()
        revenue = np.sum([leaf_revenue[position] for position in positions], axis=0)
        spend = np.sum([leaf_spend[position] for position in positions], axis=0)
        planned = float(subset["planned_budget"].sum())
        summaries.append(
            _summarize_draws(
                revenue,
                spend,
                planned,
                target_roas,
                level,
                {
                    "channel": group_map.get("channel", "ALL"),
                    "campaign_type": group_map.get("campaign_type", "ALL"),
                    "campaign_id": "ALL",
                    "campaign_name": "ALL",
                },
                rollup_flag,
                revenue_p50_override=float(subset["predicted_revenue_p50"].sum()),
            )
        )
    return summaries


def build_forecast(
    model: HorizonModel,
    canonical: pd.DataFrame,
    horizon_days: int,
    budget_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    leaves = model.forecast_campaigns(canonical, horizon_days, budget_overrides)
    if leaves.empty:
        raise ValueError("No campaign-level forecasts could be produced")
    leaves = leaves.reset_index(drop=True)
    percentile_paths, dependence_method = _dependence_percentile_paths(model, horizon_days, leaves)
    rollup_flag = (
        "historical_residual_factor_copula_rollup"
        if dependence_method == "hierarchical_residual_factor_copula_v1"
        else "independent_rank_rollup_fallback"
    )

    leaf_revenue: list[np.ndarray] = []
    leaf_spend: list[np.ndarray] = []
    for index, leaf in leaves.iterrows():
        revenue_draws, spend_draws = _leaf_draw_paths(leaf, percentile_paths[int(index)])
        leaf_revenue.append(revenue_draws)
        leaf_spend.append(spend_draws)
    leaf_revenue, portfolio_oof_residual_applied = _reconcile_portfolio_oof_residual_calibration(
        leaf_revenue,
        model,
        horizon_days,
        float(leaves["predicted_revenue_p50"].sum()),
    )

    leaf_rows: list[dict[str, object]] = []
    for index, leaf in leaves.iterrows():
        leaf_rows.append(
            _summarize_draws(
                leaf_revenue[int(index)],
                leaf_spend[int(index)],
                float(leaf["planned_budget"]),
                float(model.target_roas),
                "campaign",
                {
                    "channel": leaf["channel"],
                    "campaign_type": leaf["campaign_type"],
                    "campaign_id": leaf["campaign_id"],
                    "campaign_name": leaf["campaign_name"],
                    "campaign_key": leaf["campaign_key"],
                },
                f"{leaf['quality_flags']};additive_p50_reconciled",
                revenue_p50_override=float(leaf["predicted_revenue_p50"]),
            )
        )
        leaf_rows[-1]["_draw_index"] = int(index)

    leaf_frame = pd.DataFrame(leaf_rows)
    aggregate_quality_flags = f"{rollup_flag};additive_p50_reconciled"
    campaign_type_rows = _aggregate_level(
        leaf_frame,
        leaf_revenue,
        leaf_spend,
        ["channel", "campaign_type"],
        "campaign_type",
        float(model.target_roas),
        aggregate_quality_flags,
    )
    channel_rows = _aggregate_level(
        leaf_frame,
        leaf_revenue,
        leaf_spend,
        ["channel"],
        "channel",
        float(model.target_roas),
        aggregate_quality_flags,
    )
    overall_revenue = np.sum(leaf_revenue, axis=0)
    overall_quality_flags = (
        f"{rollup_flag};portfolio_oof_residual_calibration;additive_p50_reconciled"
        if portfolio_oof_residual_applied
        else f"{rollup_flag};additive_p50_reconciled"
    )
    overall_rows = [
        _summarize_draws(
            overall_revenue,
            np.sum(leaf_spend, axis=0),
            float(leaf_frame["planned_budget"].sum()),
            float(model.target_roas),
            "overall",
            {"channel": "ALL", "campaign_type": "ALL", "campaign_id": "ALL", "campaign_name": "ALL"},
            overall_quality_flags,
            revenue_p50_override=float(leaf_frame["predicted_revenue_p50"].sum()),
        )
    ]
    leaf_frame = leaf_frame.drop(columns=["_draw_index"])
    all_levels = pd.DataFrame(leaf_frame.to_dict(orient="records") + campaign_type_rows + channel_rows + overall_rows)
    all_levels["risk_score"] = all_levels.apply(_risk, axis=1)

    # Forecast identity must cover every canonical field that can alter any
    # forecast, interval, risk flag, or output row. In particular,
    # ``configured_budget`` affects spend-delivery uncertainty even though it
    # is not a revenue feature. A partial identity would let two different
    # decision artifacts share a forecast_id.
    identity_columns = list(CANONICAL_COLUMNS)
    canonical_payload = canonical.loc[:, identity_columns].sort_values(identity_columns, kind="stable").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
    )
    scenario_payload = json.dumps(
        {
            "budget_overrides": {str(key): float(value) for key, value in sorted((budget_overrides or {}).items())},
            "target_roas": float(model.target_roas),
            "uncertainty": dependence_method,
            "portfolio_oof_residual_calibration": portfolio_oof_residual_applied,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    seed_text = "|".join(
        [
            model.model_version,
            getattr(model, "artifact_sha256", "") or "unhashed-artifact",
            hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest(),
            str(horizon_days),
            scenario_payload,
        ]
    )
    all_levels["forecast_id"] = hashlib.sha256(seed_text.encode()).hexdigest()[:16]
    all_levels["horizon_days"] = horizon_days
    all_levels["model_version"] = model.model_version
    return all_levels
