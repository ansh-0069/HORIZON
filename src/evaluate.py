from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.forecast import build_forecast
from src.model import HorizonModel


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def rolling_origin_backtest(canonical: pd.DataFrame, horizon_days: int, folds: int = 3, step_days: int = 30, train_direct: bool = True) -> dict[str, Any]:
    """Evaluate direct aggregate forecasts using only history available at each cutoff."""
    latest = canonical["date"].max()
    rows: list[dict[str, Any]] = []
    for fold in range(folds, 0, -1):
        cutoff = latest - pd.Timedelta(days=horizon_days + (fold - 1) * step_days)
        train = canonical[canonical["date"] <= cutoff].copy()
        actual = canonical[(canonical["date"] > cutoff) & (canonical["date"] <= cutoff + pd.Timedelta(days=horizon_days))]
        if train["date"].nunique() < 90 or actual.empty:
            continue
        model = HorizonModel.fit(train, train_direct=train_direct)
        forecast = build_forecast(model, train, horizon_days)
        overall = forecast[forecast["level"] == "overall"].iloc[0]
        actual_revenue = float(actual["revenue"].sum())
        actual_spend = float(actual["spend"].sum())
        actual_roas = _safe_ratio(actual_revenue, actual_spend)
        rows.append({
            "cutoff": str(cutoff.date()),
            "actual_revenue": actual_revenue,
            "actual_spend": actual_spend,
            "actual_roas": actual_roas,
            "predicted_revenue_p10": float(overall["predicted_revenue_p10"]),
            "predicted_revenue_p50": float(overall["predicted_revenue_p50"]),
            "predicted_revenue_p90": float(overall["predicted_revenue_p90"]),
            "predicted_roas_p10": float(overall["predicted_roas_p10"]),
            "predicted_roas_p50": float(overall["predicted_roas_p50"]),
            "predicted_roas_p90": float(overall["predicted_roas_p90"]),
        })
    if not rows:
        raise ValueError(f"Insufficient history for {horizon_days}-day rolling-origin backtest")
    frame = pd.DataFrame(rows)
    coverage = ((frame["actual_revenue"] >= frame["predicted_revenue_p10"]) & (frame["actual_revenue"] <= frame["predicted_revenue_p90"])).mean()
    roas_coverage = ((frame["actual_roas"] >= frame["predicted_roas_p10"]) & (frame["actual_roas"] <= frame["predicted_roas_p90"])).mean()
    wape = float((frame["actual_revenue"] - frame["predicted_revenue_p50"]).abs().sum() / max(frame["actual_revenue"].abs().sum(), 1e-9))
    return {
        "horizon_days": horizon_days,
        "folds": len(frame),
        "revenue_interval_coverage": round(float(coverage), 4),
        "roas_interval_coverage": round(float(roas_coverage), 4),
        "revenue_wape": round(wape, 4),
        "fold_results": rows,
    }


def evaluate_all_horizons(canonical: pd.DataFrame, folds: int = 3) -> dict[str, Any]:
    horizons = (30, 60, 90)
    champion = [rolling_origin_backtest(canonical, h, folds, train_direct=True) for h in horizons]
    baseline = [rolling_origin_backtest(canonical, h, folds, train_direct=False) for h in horizons]
    return {
        "model_family": "horizon-direct-ridge-v1",
        "baseline_model_family": "horizon-statistical-v3",
        "horizons": champion,
        "baseline_horizons": baseline,
    }


def write_evaluation_report(canonical: pd.DataFrame, output: Path, folds: int = 3) -> dict[str, Any]:
    report = evaluate_all_horizons(canonical, folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
