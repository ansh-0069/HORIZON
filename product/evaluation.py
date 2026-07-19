from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from product.training.model_builder import fit_horizon_model
from src.contracts import CANONICAL_COLUMNS
from src.forecast import build_forecast


EVALUATION_REPORT_SCHEMA_VERSION = "horizon-evaluation-v3"
DEFAULT_EVALUATION_FOLDS = 6
# ``None`` resolves to the evaluated horizon.  This keeps primary target
# windows non-overlapping, rather than treating correlated 60/90-day outcomes
# as independent Bernoulli trials in coverage confidence intervals.
DEFAULT_EVALUATION_STEP_DAYS: int | None = None
EVALUATION_HORIZONS = (30, 60, 90)
MINIMUM_TARGET_PROBABILITY_OBSERVATIONS = 20


def canonical_fingerprint(canonical: pd.DataFrame) -> str:
    # Decision uncertainty depends on configured budgets and quality semantics
    # in addition to revenue-model features. Hash the whole canonical contract
    # so a trust report cannot be reused after any inference-relevant source
    # field has changed.
    columns = list(CANONICAL_COLUMNS)
    # Sorting every serialized field makes the provenance hash invariant to
    # source-file row order even if an upstream system emits duplicate
    # identity/date rows with different measured values.
    payload = canonical.loc[:, columns].sort_values(columns, kind="stable").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_provenance(canonical: pd.DataFrame) -> dict[str, object]:
    """Describe exactly which canonical data an evaluation report represents.

    Backtests retrain only on pre-origin rows, so this is not an assertion that
    a committed pickle was evaluated in place.  It is the reproducible data
    identity for the rolling-origin protocol and lets the planner reject a
    report generated from different demo metadata or campaign taxonomy.
    """
    if canonical.empty:
        raise ValueError("Cannot create evaluation provenance for an empty canonical dataset")
    campaign_count = canonical[["source_system", "source_campaign_id"]].drop_duplicates().shape[0]
    return {
        "canonical_fingerprint": canonical_fingerprint(canonical),
        "canonical_rows": int(len(canonical)),
        "canonical_date_start": str(canonical["date"].min().date()),
        "canonical_date_end": str(canonical["date"].max().date()),
        "campaign_count": int(campaign_count),
        "source_systems": sorted(str(value) for value in canonical["source_system"].dropna().unique()),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _wilson_interval(successes: int, observations: int, z_score: float = 1.96) -> list[float | None]:
    """Return a conservative binomial confidence interval for empirical coverage."""
    if observations <= 0:
        return [None, None]
    probability = successes / observations
    denominator = 1.0 + z_score**2 / observations
    center = (probability + z_score**2 / (2.0 * observations)) / denominator
    margin = z_score * ((probability * (1.0 - probability) / observations + z_score**2 / (4.0 * observations**2)) ** 0.5) / denominator
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _hierarchy_actuals(actual: pd.DataFrame, level: str) -> pd.DataFrame:
    """Aggregate holdout outcomes using the same identities as forecast rows."""
    keys_by_level = {
        "overall": [],
        "channel": ["channel"],
        "campaign_type": ["channel", "campaign_type"],
        "campaign": ["campaign_key"],
    }
    if level not in keys_by_level:
        raise ValueError(f"Unsupported hierarchy level: {level}")
    if actual.empty:
        return pd.DataFrame(columns=[*keys_by_level[level], "actual_revenue", "actual_spend"])
    if level == "overall":
        return pd.DataFrame([{"actual_revenue": float(actual["revenue"].sum()), "actual_spend": float(actual["spend"].sum())}])
    if level == "channel":
        return actual.groupby("channel", as_index=False)[["revenue", "spend"]].sum().rename(
            columns={"revenue": "actual_revenue", "spend": "actual_spend"}
        )
    if level == "campaign_type":
        return actual.groupby(["channel", "campaign_type"], as_index=False)[["revenue", "spend"]].sum().rename(
            columns={"revenue": "actual_revenue", "spend": "actual_spend"}
        )
    if level == "campaign":
        frame = actual.copy()
        frame["campaign_key"] = frame["source_system"].astype(str) + ":" + frame["source_campaign_id"].astype(str)
        return frame.groupby("campaign_key", as_index=False)[["revenue", "spend"]].sum().rename(
            columns={"revenue": "actual_revenue", "spend": "actual_spend"}
        )


def _coverage_records(forecast: pd.DataFrame, actual: pd.DataFrame) -> list[dict[str, object]]:
    """Build coverage records without silently dropping unmatched identities.

    A prior inner join removed campaigns that appeared only in the forecast or
    only in the holdout.  That inflated hierarchy coverage and concealed a
    missing forecast.  Forecast-only rows remain visible as missing ground
    truth; actual-only rows are scored as uncovered because a real outcome was
    present without a corresponding prediction.
    """
    records: list[dict[str, object]] = []
    group_keys = {"overall": [], "channel": ["channel"], "campaign_type": ["channel", "campaign_type"], "campaign": ["campaign_key"]}
    prediction_columns = [
        "predicted_revenue_p10",
        "predicted_revenue_p90",
        "predicted_roas_p10",
        "predicted_roas_p90",
    ]
    for level, keys in group_keys.items():
        predicted = forecast.loc[
            forecast["level"] == level,
            [*keys, *prediction_columns],
        ].copy()
        observed = _hierarchy_actuals(actual, level)
        if predicted.empty and observed.empty:
            continue
        join_keys = list(keys)
        if not join_keys:
            # Pandas does not permit an outer merge without a common key.
            # ``__overall_key`` is local to this join and cannot alter
            # forecast identities.
            join_keys = ["__overall_key"]
            predicted["__overall_key"] = 1
            observed["__overall_key"] = 1
        joined = predicted.merge(
            observed,
            on=join_keys,
            how="outer",
            indicator="join_status",
            validate="one_to_one",
        )
        for _, row in joined.iterrows():
            status = str(row["join_status"])
            has_forecast = status != "right_only"
            has_actual = status != "left_only"
            actual_revenue = float(row["actual_revenue"]) if has_actual else None
            actual_spend = float(row["actual_spend"]) if has_actual else None
            actual_roas = (
                _safe_ratio(actual_revenue, actual_spend)
                if actual_revenue is not None and actual_spend is not None
                else None
            )
            revenue_evaluable = has_actual
            revenue_covered = (
                bool(
                    has_forecast
                    and float(row["predicted_revenue_p10"]) <= actual_revenue <= float(row["predicted_revenue_p90"])
                )
                if revenue_evaluable
                else None
            )
            roas_evaluable = actual_roas is not None
            roas_covered = (
                bool(
                    has_forecast
                    and float(row["predicted_roas_p10"]) <= actual_roas <= float(row["predicted_roas_p90"])
                )
                if roas_evaluable
                else None
            )
            records.append(
                {
                    "level": level,
                    "join_status": {
                        "both": "matched",
                        "left_only": "missing_actual",
                        "right_only": "missing_forecast",
                    }[status],
                    "revenue_evaluable": revenue_evaluable,
                    "revenue_covered": revenue_covered,
                    "roas_evaluable": roas_evaluable,
                    "roas_covered": roas_covered,
                }
            )
    return records


def _coverage_summary(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    if not records:
        return {}
    frame = pd.DataFrame(records)
    summary: dict[str, dict[str, object]] = {}
    for level, subset in frame.groupby("level", sort=False):
        observations = int(len(subset))
        matched = int(subset["join_status"].eq("matched").sum())
        missing_forecast = int(subset["join_status"].eq("missing_forecast").sum())
        missing_actual = int(subset["join_status"].eq("missing_actual").sum())
        revenue_evaluable = subset[subset["revenue_evaluable"].eq(True)]
        roas_evaluable = subset[subset["roas_evaluable"].eq(True)]
        revenue_observations = int(len(revenue_evaluable))
        roas_observations = int(len(roas_evaluable))
        revenue_successes = int(revenue_evaluable["revenue_covered"].eq(True).sum())
        roas_successes = int(roas_evaluable["roas_covered"].eq(True).sum())
        summary[str(level)] = {
            "observations": observations,
            "matched_observations": matched,
            "missing_forecast_observations": missing_forecast,
            "missing_actual_observations": missing_actual,
            "revenue_coverage_observations": revenue_observations,
            "revenue_interval_coverage": round(revenue_successes / revenue_observations, 4) if revenue_observations else None,
            "revenue_interval_coverage_95_ci": _wilson_interval(revenue_successes, revenue_observations),
            "roas_coverage_observations": roas_observations,
            "roas_unscorable_observations": int(observations - roas_observations),
            "roas_interval_coverage": round(roas_successes / roas_observations, 4) if roas_observations else None,
            "roas_interval_coverage_95_ci": _wilson_interval(roas_successes, roas_observations),
        }
    return summary


def _target_probability_reliability(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Evaluate the ROAS-target draw share as an empirical event probability.

    Forecast draw frequency is not automatically a calibrated probability.
    This diagnostic records Brier score, reliability bins, and evidence volume
    separately from interval coverage so serving can decline approval when
    there is no basis to treat a simulated draw share as decision-grade.
    """
    pairs: list[tuple[float, float]] = []
    for row in rows:
        probability = row.get("modeled_target_probability")
        outcome = row.get("actual_target_hit")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            continue
        if isinstance(outcome, bool):
            value = float(outcome)
        elif isinstance(outcome, (int, float)) and not isinstance(outcome, bool):
            value = float(outcome)
        else:
            continue
        probability_value = float(probability)
        if math.isfinite(probability_value) and math.isfinite(value):
            pairs.append((min(1.0, max(0.0, probability_value)), 1.0 if value >= 0.5 else 0.0))
    if not pairs:
        return {
            "status": "unavailable",
            "observations": 0,
            "reason": "No finite actual ROAS target events were available for reliability scoring.",
        }
    probabilities = [pair[0] for pair in pairs]
    outcomes = [pair[1] for pair in pairs]
    brier = sum((probability - outcome) ** 2 for probability, outcome in pairs) / len(pairs)
    bin_edges = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0000001))
    bins: list[dict[str, object]] = []
    ece = 0.0
    for lower, upper in bin_edges:
        indices = [index for index, probability in enumerate(probabilities) if lower <= probability < upper]
        if not indices:
            continue
        mean_probability = sum(probabilities[index] for index in indices) / len(indices)
        observed_rate = sum(outcomes[index] for index in indices) / len(indices)
        ece += len(indices) / len(pairs) * abs(mean_probability - observed_rate)
        bins.append(
            {
                "range": f"[{lower:.2f}, {min(upper, 1.0):.2f}{']' if upper > 1.0 else ')'}",
                "observations": len(indices),
                "mean_modeled_probability": round(mean_probability, 4),
                "observed_target_hit_rate": round(observed_rate, 4),
            }
        )
    status = "evaluated" if len(pairs) >= MINIMUM_TARGET_PROBABILITY_OBSERVATIONS else "insufficient_evidence"
    payload: dict[str, object] = {
        "status": status,
        "observations": len(pairs),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "bins": bins,
        "target_definition": "actual_horizon_revenue / actual_horizon_spend >= model target_roas",
    }
    if status != "evaluated":
        payload["reason"] = (
            f"At least {MINIMUM_TARGET_PROBABILITY_OBSERVATIONS} non-overlapping target events are required "
            "before this draw share can be used as a calibrated approval probability."
        )
    return payload


def rolling_origin_backtest(
    canonical: pd.DataFrame,
    horizon_days: int,
    folds: int = DEFAULT_EVALUATION_FOLDS,
    step_days: int | None = DEFAULT_EVALUATION_STEP_DAYS,
    train_direct: bool = True,
) -> dict[str, Any]:
    """Evaluate one horizon with chronological, leakage-safe forecast origins."""
    if horizon_days not in EVALUATION_HORIZONS:
        raise ValueError(f"Unsupported evaluation horizon: {horizon_days}")
    if folds < 1:
        raise ValueError("folds must be at least 1")
    resolved_step_days = horizon_days if step_days is None else step_days
    if resolved_step_days < 1:
        raise ValueError("step_days must be at least 1")
    latest = canonical["date"].max()
    rows: list[dict[str, Any]] = []
    hierarchy_records: list[dict[str, object]] = []
    for fold in range(folds, 0, -1):
        cutoff = latest - pd.Timedelta(days=horizon_days + (fold - 1) * resolved_step_days)
        train = canonical[canonical["date"] <= cutoff].copy()
        actual = canonical[(canonical["date"] > cutoff) & (canonical["date"] <= cutoff + pd.Timedelta(days=horizon_days))]
        if train["date"].nunique() < 90 or actual.empty:
            continue
        # Backtesting one horizon should not need to fit unrelated horizon
        # models.  This keeps the expanded calibration evidence practical
        # without changing the production artifact, which still trains all
        # requested 30/60/90 day models together.
        model = fit_horizon_model(train, train_direct=train_direct, horizons=(horizon_days,))
        forecast = build_forecast(model, train, horizon_days)
        hierarchy_records.extend(_coverage_records(forecast, actual))
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
            "model_version": str(model.model_version),
            "training_data_fingerprint": str(
                getattr(model, "training_data_fingerprint", canonical_fingerprint(train))
                or canonical_fingerprint(train)
            ),
            "feature_schema_fingerprint": str(getattr(model, "feature_schema_fingerprint", "")),
            "target_roas": float(model.target_roas),
            "modeled_target_probability": float(overall["probability_roas_above_target"]),
            "actual_target_hit": None if actual_roas is None else bool(actual_roas >= float(model.target_roas)),
        })
    if not rows:
        raise ValueError(f"Insufficient history for {horizon_days}-day rolling-origin backtest")
    frame = pd.DataFrame(rows)
    coverage = ((frame["actual_revenue"] >= frame["predicted_revenue_p10"]) & (frame["actual_revenue"] <= frame["predicted_revenue_p90"])).mean()
    roas_coverage = ((frame["actual_roas"] >= frame["predicted_roas_p10"]) & (frame["actual_roas"] <= frame["predicted_roas_p90"])).mean()
    wape = float((frame["actual_revenue"] - frame["predicted_revenue_p50"]).abs().sum() / max(frame["actual_revenue"].abs().sum(), 1e-9))
    calibration_counts = [int(value) for value in frame["calibration_sample_count"] if int(value) > 0]
    revenue_successes = int(
        ((frame["actual_revenue"] >= frame["predicted_revenue_p10"]) & (frame["actual_revenue"] <= frame["predicted_revenue_p90"])).sum()
    )
    roas_successes = int(
        ((frame["actual_roas"] >= frame["predicted_roas_p10"]) & (frame["actual_roas"] <= frame["predicted_roas_p90"])).sum()
    )
    return {
        "horizon_days": horizon_days,
        "requested_folds": folds,
        "step_days": resolved_step_days,
        "fold_windows_non_overlapping": bool(resolved_step_days >= horizon_days),
        "folds": len(frame),
        "revenue_interval_coverage": round(float(coverage), 4),
        "revenue_interval_coverage_95_ci": _wilson_interval(revenue_successes, len(frame)),
        "roas_interval_coverage": round(float(roas_coverage), 4),
        "roas_interval_coverage_95_ci": _wilson_interval(roas_successes, len(frame)),
        "nominal_interval_coverage": 0.80,
        "revenue_wape": round(wape, 4),
        "uncertainty_method": str(frame["uncertainty_method"].iloc[-1]),
        "median_calibration_samples": int(pd.Series(calibration_counts).median()) if calibration_counts else 0,
        "model_versions_observed": sorted(str(value) for value in frame["model_version"].unique()),
        "training_data_fingerprints": sorted(str(value) for value in frame["training_data_fingerprint"].unique()),
        "feature_schema_fingerprints": sorted(
            str(value) for value in frame["feature_schema_fingerprint"].unique() if str(value)
        ),
        "coverage_by_hierarchy": _coverage_summary(hierarchy_records),
        "roas_target_probability_reliability": _target_probability_reliability(rows),
        "fold_results": rows,
    }


def _observed_model_versions(reports: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(version)
            for report in reports
            for version in report.get("model_versions_observed", [])
        }
    )


def evaluate_all_horizons(canonical: pd.DataFrame, folds: int = DEFAULT_EVALUATION_FOLDS) -> dict[str, Any]:
    """Run the documented product/demo-data rolling-origin protocol.

    The returned report names its actual fold-level model versions and input
    fingerprint.  It does not imply that the protected evaluator pickle was
    retrained or scored in place.
    """
    candidate_horizons = [
        rolling_origin_backtest(canonical, horizon, folds, train_direct=True)
        for horizon in EVALUATION_HORIZONS
    ]
    baseline_horizons = [
        rolling_origin_backtest(canonical, horizon, folds, train_direct=False)
        for horizon in EVALUATION_HORIZONS
    ]
    candidate_versions = _observed_model_versions(candidate_horizons)
    baseline_versions = _observed_model_versions(baseline_horizons)
    provenance = canonical_provenance(canonical)
    return {
        "report_schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "evaluation_protocol": {
            "name": "rolling_origin_prequential_backtest_v2",
            "default_folds": DEFAULT_EVALUATION_FOLDS,
            "requested_folds": folds,
            "step_days": "horizon_days (non-overlapping primary target windows)",
            "fold_dependence_note": "Target windows do not overlap; expanding training histories still share prior observations, so intervals are descriptive rather than independent-trial guarantees.",
            "horizons_days": list(EVALUATION_HORIZONS),
            "candidate_training": "Each fold uses only canonical rows at or before its forecast origin.",
            "interval_calibration": "Direct-model residual quantiles use a purged later temporal holdout; no in-sample fallback is accepted.",
            "artifact_scope": "This report evaluates fold-specific retraining, not an in-place mutation of the protected evaluator artifact.",
        },
        "data_fingerprint": provenance["canonical_fingerprint"],
        "data_provenance": provenance,
        "model_family": candidate_versions[0] if len(candidate_versions) == 1 else "mixed_fold_model_versions",
        "model_versions_observed": candidate_versions,
        "baseline_model_family": baseline_versions[0] if len(baseline_versions) == 1 else "mixed_fold_model_versions",
        "baseline_model_versions_observed": baseline_versions,
        "horizons": candidate_horizons,
        "baseline_horizons": baseline_horizons,
    }


def write_evaluation_report(
    canonical: pd.DataFrame,
    output: Path,
    folds: int = DEFAULT_EVALUATION_FOLDS,
) -> dict[str, Any]:
    report = evaluate_all_horizons(canonical, folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
