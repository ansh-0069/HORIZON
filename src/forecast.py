from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from src.model import HorizonModel

# Fixed percentile grid keeps offline inference deterministic (no RNG) while
# enabling draw-level hierarchy aggregation and ROAS probabilities.
DRAW_PERCENTILES = tuple(i / 100.0 for i in range(1, 100))
_P10_INDEX = DRAW_PERCENTILES.index(0.10)
_P50_INDEX = DRAW_PERCENTILES.index(0.50)
_P90_INDEX = DRAW_PERCENTILES.index(0.90)


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


def _leaf_draw_paths(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    revenue = np.asarray(
        [
            _interp_quantile(row["predicted_revenue_p10"], row["predicted_revenue_p50"], row["predicted_revenue_p90"], p)
            for p in DRAW_PERCENTILES
        ],
        dtype=float,
    )
    spend = np.asarray(
        [
            _interp_quantile(row["predicted_spend_p10"], row["predicted_spend_p50"], row["predicted_spend_p90"], p)
            for p in DRAW_PERCENTILES
        ],
        dtype=float,
    )
    return revenue, spend


def _summarize_draws(
    revenue_draws: np.ndarray,
    spend_draws: np.ndarray,
    planned_budget: float,
    target_roas: float,
    level: str,
    group: dict[str, object],
    quality_flags: str,
) -> dict[str, object]:
    roas_draws = np.asarray(_roas(revenue_draws, spend_draws), dtype=float)
    probability = float(np.mean(roas_draws >= float(target_roas)))
    row: dict[str, object] = {
        "level": level,
        "channel": group.get("channel", "ALL"),
        "campaign_type": group.get("campaign_type", "ALL"),
        "campaign_id": group.get("campaign_id", "ALL"),
        "campaign_name": group.get("campaign_name", "ALL"),
        "planned_budget": float(planned_budget),
        "predicted_revenue_p10": float(revenue_draws[_P10_INDEX]),
        "predicted_revenue_p50": float(revenue_draws[_P50_INDEX]),
        "predicted_revenue_p90": float(revenue_draws[_P90_INDEX]),
        "predicted_spend_p10": float(spend_draws[_P10_INDEX]),
        "predicted_spend_p50": float(spend_draws[_P50_INDEX]),
        "predicted_spend_p90": float(spend_draws[_P90_INDEX]),
        "predicted_roas_p10": float(roas_draws[_P10_INDEX]),
        "predicted_roas_p50": float(roas_draws[_P50_INDEX]),
        "predicted_roas_p90": float(roas_draws[_P90_INDEX]),
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
                "joint_draw_rollup",
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

    leaf_rows: list[dict[str, object]] = []
    leaf_revenue: list[np.ndarray] = []
    leaf_spend: list[np.ndarray] = []
    for index, leaf in leaves.iterrows():
        revenue_draws, spend_draws = _leaf_draw_paths(leaf)
        leaf_revenue.append(revenue_draws)
        leaf_spend.append(spend_draws)
        leaf_rows.append(
            _summarize_draws(
                revenue_draws,
                spend_draws,
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
                str(leaf["quality_flags"]),
            )
        )
        leaf_rows[-1]["_draw_index"] = int(index)

    leaf_frame = pd.DataFrame(leaf_rows)
    campaign_type_rows = _aggregate_level(
        leaf_frame, leaf_revenue, leaf_spend, ["channel", "campaign_type"], "campaign_type", float(model.target_roas)
    )
    channel_rows = _aggregate_level(leaf_frame, leaf_revenue, leaf_spend, ["channel"], "channel", float(model.target_roas))
    overall_rows = [
        _summarize_draws(
            np.sum(leaf_revenue, axis=0),
            np.sum(leaf_spend, axis=0),
            float(leaf_frame["planned_budget"].sum()),
            float(model.target_roas),
            "overall",
            {"channel": "ALL", "campaign_type": "ALL", "campaign_id": "ALL", "campaign_name": "ALL"},
            "joint_draw_rollup",
        )
    ]
    leaf_frame = leaf_frame.drop(columns=["_draw_index"])
    all_levels = pd.DataFrame(leaf_frame.to_dict(orient="records") + campaign_type_rows + channel_rows + overall_rows)
    all_levels["risk_score"] = all_levels.apply(_risk, axis=1)

    identity_columns = ["source_system", "source_campaign_id", "date", "channel", "campaign_type", "campaign_name", "spend", "revenue"]
    canonical_payload = canonical.loc[:, identity_columns].sort_values(identity_columns[:6], kind="stable").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
    )
    scenario_payload = json.dumps(
        {
            "budget_overrides": {str(key): float(value) for key, value in sorted((budget_overrides or {}).items())},
            "target_roas": float(model.target_roas),
            "uncertainty": "joint_draw_rollup_v1",
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
