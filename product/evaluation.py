from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from product.training.model_builder import fit_horizon_model
from src.forecast import build_forecast


def canonical_fingerprint(canonical: pd.DataFrame) -> str:
    columns = ["source_system", "source_campaign_id", "date", "channel", "campaign_type", "campaign_name", "spend", "revenue"]
    payload = canonical.loc[:, columns].sort_values(columns[:6], kind="stable").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def rolling_origin_backtest(canonical: pd.DataFrame, horizon_days: int, folds: int = 3, step_days: int = 30, train_direct: bool = True) -> dict[str, Any]:
    latest = canonical["date"].max()
    rows: list[dict[str, Any]] = []
    for fold in range(folds, 0, -1):
        cutoff = latest - pd.Timedelta(days=horizon_days + (fold - 1) * step_days)
        train = canonical[canonical["date"] <= cutoff].copy()
        actual = canonical[(canonical["date"] > cutoff) & (canonical["date"] <= cutoff + pd.Timedelta(days=horizon_days))]
        if train["date"].nunique() < 90 or actual.empty:
            continue
        model = fit_horizon_model(train, train_direct=train_direct)
        forecast = build_forecast(model, train, horizon_days)
        overall = forecast[forecast["level"] == "overall"].iloc[0]
        actual_revenue = float(actual["revenue"].sum())
        actual_spend = float(actual["spend"].sum())
        actual_roas = _safe_ratio(actual_revenue, actual_spend)
        rows.append({
            "cutoff": str(cutoff.date()), "actual_revenue": actual_revenue, "actual_spend": actual_spend, "actual_roas": actual_roas,
            "predicted_revenue_p10": float(overall["predicted_revenue_p10"]),
            "predicted_revenue_p50": float(overall["predicted_revenue_p50"]),
            "predicted_revenue_p90": float(overall["predicted_revenue_p90"]),
            "predicted_roas_p10": float(overall["predicted_roas_p10"]),
            "predicted_roas_p50": float(overall["predicted_roas_p50"]),
            "predicted_roas_p90": float(overall["predicted_roas_p90"]),
            "uncertainty_method": (
                model.direct_models[horizon_days].uncertainty_method
                if horizon_days in model.direct_models
                else "statistical_lognormal_fallback"
            ),
            "calibration_sample_count": (
                int(model.direct_models[horizon_days].calibration_sample_count)
                if horizon_days in model.direct_models
                else 0
            ),
        })
    if not rows:
        raise ValueError(f"Insufficient history for {horizon_days}-day rolling-origin backtest")
    frame = pd.DataFrame(rows)
    coverage = ((frame["actual_revenue"] >= frame["predicted_revenue_p10"]) & (frame["actual_revenue"] <= frame["predicted_revenue_p90"])).mean()
    roas_coverage = ((frame["actual_roas"] >= frame["predicted_roas_p10"]) & (frame["actual_roas"] <= frame["predicted_roas_p90"])).mean()
    wape = float((frame["actual_revenue"] - frame["predicted_revenue_p50"]).abs().sum() / max(frame["actual_revenue"].abs().sum(), 1e-9))
    calibration_counts = [int(value) for value in frame["calibration_sample_count"] if int(value) > 0]
    return {
        "horizon_days": horizon_days,
        "folds": len(frame),
        "revenue_interval_coverage": round(float(coverage), 4),
        "roas_interval_coverage": round(float(roas_coverage), 4),
        "nominal_interval_coverage": 0.80,
        "revenue_wape": round(wape, 4),
        "uncertainty_method": str(frame["uncertainty_method"].iloc[-1]),
        "median_calibration_samples": int(pd.Series(calibration_counts).median()) if calibration_counts else 0,
        "fold_results": rows,
    }


def evaluate_all_horizons(canonical: pd.DataFrame, folds: int = 3) -> dict[str, Any]:
    horizons = (30, 60, 90)
    return {
        "model_family": "horizon-direct-ridge-v3-seasonal-plan",
        "baseline_model_family": "horizon-statistical-v5-seasonal-plan",
        "data_fingerprint": canonical_fingerprint(canonical),
        "horizons": [rolling_origin_backtest(canonical, horizon, folds, train_direct=True) for horizon in horizons],
        "baseline_horizons": [rolling_origin_backtest(canonical, horizon, folds, train_direct=False) for horizon in horizons],
    }


def write_evaluation_report(canonical: pd.DataFrame, output: Path, folds: int = 3) -> dict[str, Any]:
    report = evaluate_all_horizons(canonical, folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
