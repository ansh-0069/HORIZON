from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from src.budget_plan import campaign_baseline_budget
from src.direct_model import DirectRidgeEnsembleModel, DirectRidgeModel, _features


CATEGORY_COLUMNS = ("channel", "campaign_type")
MINIMUM_DIRECT_FIT_ROWS = 40
MINIMUM_CALIBRATION_ROWS = 20
MINIMUM_SUPPORT_CALIBRATION_ROWS = 12
MINIMUM_ENSEMBLE_SELECTION_ROWS = 24
ENSEMBLE_SCENARIO_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)

BudgetTrainingMode = Literal["realized_future", "inference_aligned"]


def _inference_aligned_budget(
    canonical: pd.DataFrame,
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int,
) -> float:
    """Estimate the plan visible at a historical forecast origin.

    This is not a future-spend proxy.  It uses the campaign's preceding 28-day
    delivery cadence and the same portfolio calendar adjustment used when the
    evaluator has no supplied media plan.  It therefore gives the second
    ensemble member a train/serve-consistent budget feature.
    """
    budget, _, _ = campaign_baseline_budget(canonical, history, cutoff, horizon_days)
    return budget


def _category_vocabulary(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Return the categorical design vocabulary available at a training point.

    This helper is intentionally called with the *fit* partition for temporal
    validation and OOF residual generation.  Building one-hot columns from
    the complete training frame would disclose categories that first appear in
    the future validation period, even if their coefficients remain zero.
    An unseen category is therefore encoded as the all-zero reference pattern
    during honest validation; the final production refit can then use the
    complete historical vocabulary after calibration is complete.
    """
    return {
        column: tuple(sorted(frame[column].astype(str).unique()))
        for column in CATEGORY_COLUMNS
    }


def training_frame(
    canonical: pd.DataFrame,
    horizon_days: int,
    step_days: int = 21,
    *,
    aligned_cutoffs: bool = False,
    budget_training_mode: BudgetTrainingMode = "realized_future",
) -> pd.DataFrame:
    """Build direct-horizon training examples.

    ``aligned_cutoffs`` is reserved for residual-dependence calibration.  The
    point model retains its existing per-campaign cadence, while the
    dependence estimator needs the same historical forecast-origin dates
    across campaigns in order to measure co-movement rather than accidentally
    treating unrelated dates as a joint residual observation.
    """
    if budget_training_mode not in {"realized_future", "inference_aligned"}:
        raise ValueError(f"Unsupported direct-model budget training mode: {budget_training_mode}")
    rows: list[dict[str, object]] = []
    keys = ["source_system", "source_campaign_id", "channel", "campaign_type", "campaign_name"]
    shared_cutoffs = pd.date_range(
        canonical["date"].min() + pd.Timedelta(days=56),
        canonical["date"].max() - pd.Timedelta(days=horizon_days),
        freq=f"{step_days}D",
    ) if aligned_cutoffs else None
    for key, group in canonical.groupby(keys, dropna=False):
        group = group.sort_values("date")
        minimum = group["date"].min() + pd.Timedelta(days=56)
        maximum = group["date"].max() - pd.Timedelta(days=horizon_days)
        cutoffs = shared_cutoffs if shared_cutoffs is not None else pd.date_range(minimum, maximum, freq=f"{step_days}D")
        for cutoff in cutoffs:
            if cutoff < minimum or cutoff > maximum:
                continue
            history = group[group["date"] <= cutoff]
            future = group[(group["date"] > cutoff) & (group["date"] <= cutoff + pd.Timedelta(days=horizon_days))]
            realized_future_budget = float(future["spend"].sum())
            planned_budget = (
                realized_future_budget
                if budget_training_mode == "realized_future"
                else _inference_aligned_budget(canonical, history, cutoff, horizon_days)
            )
            # Both experts learn revenue only from a non-zero realized paid
            # outcome. The aligned expert changes the *known-at-origin plan*
            # feature, not the target population by silently including zero
            # future-spend leaves that baseline inference would not forecast.
            if len(history) < 28 or realized_future_budget <= 0 or planned_budget <= 0:
                continue
            feature_row = _features(history, cutoff, horizon_days, planned_budget)
            feature_row.update({
                "cutoff": cutoff,
                # Explicitly persist the target boundary.  Temporal splits
                # must reason about the *end* of a label window, not merely
                # its forecast origin.
                "target_end": cutoff + pd.Timedelta(days=horizon_days),
                "source_system": str(key[0]),
                "source_campaign_id": str(key[1]),
                "campaign_key": f"{key[0]}:{key[1]}",
                "channel": str(key[2]),
                "campaign_type": str(key[3]),
                "target_log_revenue": math.log1p(max(float(future["revenue"].sum()), 0.0)),
                "budget_training_mode": budget_training_mode,
            })
            rows.append(feature_row)
    return pd.DataFrame(rows)


def expanding_window_oof_residuals(
    canonical: pd.DataFrame,
    horizon_days: int,
    ridge_alpha: float = 4.0,
    minimum_fit_rows: int = 80,
) -> pd.DataFrame:
    """Return leakage-safe, cross-campaign aligned OOF log-residuals.

    Each residual is produced by a model that only sees training examples
    whose *entire* forecast target window finished before the evaluated
    cutoff.  This is slower than reusing in-sample residuals, but it runs only
    in the offline training workflow and gives the portfolio copula an honest
    estimate of residual co-movement.  A global cutoff cadence is required so
    rows sharing a cutoff represent the same market interval.
    """
    frame = training_frame(canonical, horizon_days, aligned_cutoffs=True)
    if frame.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for cutoff in sorted(pd.Timestamp(value) for value in frame["cutoff"].unique()):
        # A target that ends on the evaluation origin would leak that day's
        # outcome into the model used to score its features.  Require the
        # entire fit target window to end strictly before the OOF origin.
        fit_frame = frame[frame["target_end"] < cutoff]
        evaluation = frame[frame["cutoff"] == cutoff].copy()
        if len(fit_frame) < minimum_fit_rows or evaluation.empty:
            continue
        categories = _category_vocabulary(fit_frame)
        coefficients, mean, scale = _fit_coefficients(fit_frame, categories, ridge_alpha)
        design, _, _ = DirectRidgeModel._design(evaluation, categories, mean, scale)
        evaluation["predicted_log_revenue"] = design @ coefficients
        evaluation["residual_log_revenue"] = evaluation["target_log_revenue"].to_numpy(dtype=float) - evaluation["predicted_log_revenue"]
        rows.append(
            evaluation.loc[
                :,
                [
                    "cutoff",
                    "campaign_key",
                    "channel",
                    "campaign_type",
                    "target_log_revenue",
                    "predicted_log_revenue",
                    "residual_log_revenue",
                ],
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "cutoff",
                "campaign_key",
                "channel",
                "campaign_type",
                "target_log_revenue",
                "predicted_log_revenue",
                "residual_log_revenue",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["cutoff", "campaign_key"], kind="stable"
    ).reset_index(drop=True)


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


def _support_uncertainty_profiles(
    calibration_frame: pd.DataFrame,
    calibration_residuals: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Fit conservative support-conditioned interval width multipliers.

    The calibration partition is chronologically later than the coefficient
    fit.  We measure the 80% log-residual span for sparse and extrapolated
    rows relative to the global span, then enforce a small widening floor. The
    floor is a decision-safety guardrail, not a claim of formal conditional
    coverage.  Profiles are omitted when the purged partition is too small.
    """
    if len(calibration_frame) != len(calibration_residuals) or len(calibration_frame) < MINIMUM_CALIBRATION_ROWS:
        return {}
    residuals = np.asarray(calibration_residuals, dtype=float)
    finite = np.isfinite(residuals)
    if int(finite.sum()) < MINIMUM_CALIBRATION_ROWS:
        return {}
    global_span = float(np.quantile(residuals[finite], 0.90) - np.quantile(residuals[finite], 0.10))
    if not math.isfinite(global_span) or global_span <= 1e-8:
        return {}
    frame = calibration_frame.reset_index(drop=True)
    profiles: dict[str, dict[str, float]] = {}
    masks = {
        "sparse_recent_history": pd.to_numeric(frame.get("recent_active_days"), errors="coerce").fillna(0.0).to_numpy() < 28.0,
        "budget_extrapolation": pd.to_numeric(frame.get("planned_budget_support_ratio"), errors="coerce").fillna(float("inf")).to_numpy() > 1.5,
    }
    for name, mask in masks.items():
        values = residuals[np.asarray(mask, dtype=bool) & finite]
        if len(values) < MINIMUM_SUPPORT_CALIBRATION_ROWS:
            continue
        local_span = float(np.quantile(values, 0.90) - np.quantile(values, 0.10))
        if not math.isfinite(local_span):
            continue
        # At least modestly widen a known lower-support condition. The upper
        # bound prevents a handful of volatile rows from dominating a campaign
        # interval at inference.
        multiplier = min(2.50, max(1.15, local_span / global_span))
        profiles[name] = {
            "width_multiplier": round(float(multiplier), 6),
            "calibration_sample_count": float(len(values)),
            "global_log_residual_span": round(global_span, 6),
            "condition_log_residual_span": round(local_span, 6),
        }
    return profiles


def _temporal_calibration_partitions(
    frame: pd.DataFrame,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Create a purged fit/calibration split from direct training examples."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    cutoff_values = sorted(pd.Timestamp(value) for value in frame["cutoff"].unique())
    if not cutoff_values:
        raise ValueError("Cannot calibrate a direct model without forecast-origin cutoffs")
    holdout_cutoffs = max(1, math.ceil(len(cutoff_values) * 0.20))
    calibration_start = cutoff_values[-holdout_cutoffs]
    # Do not use an origin-only comparison here.  Each row's label spans a
    # horizon, and temporal purity requires the label to end *before* the
    # first calibration origin.
    fit_frame = frame[frame["target_end"] < calibration_start].copy()
    calibration_frame = frame[frame["cutoff"] >= calibration_start].copy()
    return fit_frame, calibration_frame, calibration_start


def _fit_direct_ridge_frame(
    frame: pd.DataFrame,
    horizon_days: int,
    ridge_alpha: float,
    uncertainty_method: str,
) -> DirectRidgeModel | None:
    """Fit one direct ridge expert from a prebuilt leakage-safe frame."""
    if len(frame) < 80:
        return None
    fit_frame, calibration_frame, _ = _temporal_calibration_partitions(frame, horizon_days)

    if len(fit_frame) < MINIMUM_DIRECT_FIT_ROWS or len(calibration_frame) < MINIMUM_CALIBRATION_ROWS:
        return None

    # The calibration model must only know category values observed before the
    # holdout begins.  This avoids future-vocabulary leakage in the one-hot
    # design matrix and makes newly appearing campaign types count as genuine
    # validation risk rather than a free feature.
    calibration_categories = _category_vocabulary(fit_frame)
    calibration_coefficients, calibration_mean, calibration_scale = _fit_coefficients(
        fit_frame, calibration_categories, ridge_alpha
    )
    calibration_x, _, _ = DirectRidgeModel._design(
        calibration_frame, calibration_categories, calibration_mean, calibration_scale
    )
    calibration_residuals = (
        calibration_frame["target_log_revenue"].to_numpy(dtype=float)
        - calibration_x @ calibration_coefficients
    )
    residual_p10, residual_p50, residual_p90 = np.quantile(calibration_residuals, [0.10, 0.50, 0.90])

    # Refit point coefficients on all historical data only after the holdout
    # residuals have been sealed.  This is a normal post-validation refit and
    # does not change the residual quantiles persisted for uncertainty.
    final_categories = _category_vocabulary(frame)
    coefficients, mean, scale = _fit_coefficients(frame, final_categories, ridge_alpha)
    return DirectRidgeModel(
        horizon_days=horizon_days,
        category_columns=CATEGORY_COLUMNS,
        categories=final_categories,
        numeric_mean=mean.tolist(), numeric_scale=scale.tolist(), coefficients=coefficients.tolist(),
        residual_p10=float(residual_p10),
        residual_p50=float(residual_p50),
        residual_p90=float(residual_p90),
        sample_count=len(frame),
        calibration_sample_count=len(calibration_residuals),
        uncertainty_method=uncertainty_method,
        support_uncertainty_profiles=_support_uncertainty_profiles(calibration_frame, calibration_residuals),
    )


def fit_direct_ridge(canonical: pd.DataFrame, horizon_days: int, ridge_alpha: float = 4.0) -> DirectRidgeModel | None:
    """Fit a direct model only when an honest temporal interval is available.

    Point coefficients are refit on all historical examples after interval
    calibration, but P10/P50/P90 residual quantiles are *only* calculated on
    a later, purged chronological holdout.  Returning ``None`` is deliberate:
    when history cannot support a separate calibration window, the caller must
    select the explicit statistical fallback instead of presenting in-sample
    residuals as calibrated uncertainty.
    """
    frame = training_frame(canonical, horizon_days)
    return _fit_direct_ridge_frame(
        frame,
        horizon_days,
        ridge_alpha,
        "purged_temporal_holdout_support_aware_residual_quantiles_v3",
    )


def _ensemble_selection_partitions(frame: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Create a second purged split strictly before the interval holdout."""
    if frame.empty:
        return None
    _, _, calibration_start = _temporal_calibration_partitions(frame, horizon_days)
    selection_pool = frame[frame["target_end"] < calibration_start].copy()
    cutoff_values = sorted(pd.Timestamp(value) for value in selection_pool["cutoff"].unique())
    if not cutoff_values:
        return None
    holdout_cutoffs = max(1, math.ceil(len(cutoff_values) * 0.20))
    selection_start = cutoff_values[-holdout_cutoffs]
    selection_fit = selection_pool[selection_pool["target_end"] < selection_start].copy()
    selection_validation = selection_pool[selection_pool["cutoff"] >= selection_start].copy()
    if len(selection_fit) < MINIMUM_DIRECT_FIT_ROWS or len(selection_validation) < MINIMUM_ENSEMBLE_SELECTION_ROWS:
        return None
    return selection_fit, selection_validation


def _selection_predictions(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    ridge_alpha: float,
) -> np.ndarray:
    categories = _category_vocabulary(fit_frame)
    coefficients, mean, scale = _fit_coefficients(fit_frame, categories, ridge_alpha)
    design, _, _ = DirectRidgeModel._design(validation_frame, categories, mean, scale)
    return np.maximum(0.0, np.expm1(design @ coefficients))


def _select_ensemble_weight(
    scenario_frame: pd.DataFrame,
    aligned_frame: pd.DataFrame,
    horizon_days: int,
    ridge_alpha: float,
) -> tuple[float, int]:
    """Select a revenue-WAPE blend using only a pre-calibration time window."""
    scenario_partitions = _ensemble_selection_partitions(scenario_frame, horizon_days)
    aligned_partitions = _ensemble_selection_partitions(aligned_frame, horizon_days)
    if scenario_partitions is None or aligned_partitions is None:
        return 1.0, 0
    scenario_fit, scenario_validation = scenario_partitions
    aligned_fit, aligned_validation = aligned_partitions
    keys = ["campaign_key", "cutoff"]
    scenario_indexed = scenario_validation.set_index(keys, drop=False)
    aligned_indexed = aligned_validation.set_index(keys, drop=False)
    common = scenario_indexed.index.intersection(aligned_indexed.index)
    if len(common) < MINIMUM_ENSEMBLE_SELECTION_ROWS:
        return 1.0, int(len(common))
    scenario_validation = scenario_indexed.loc[common].reset_index(drop=True)
    aligned_validation = aligned_indexed.loc[common].reset_index(drop=True)
    scenario_prediction = _selection_predictions(scenario_fit, scenario_validation, ridge_alpha)
    aligned_prediction = _selection_predictions(aligned_fit, aligned_validation, ridge_alpha)
    actual = np.maximum(0.0, np.expm1(scenario_validation["target_log_revenue"].to_numpy(dtype=float)))
    denominator = float(np.abs(actual).sum())
    if not math.isfinite(denominator) or denominator <= 1e-9:
        return 1.0, int(len(common))
    scores = {
        weight: float(np.abs(actual - (weight * scenario_prediction + (1.0 - weight) * aligned_prediction)).sum() / denominator)
        for weight in ENSEMBLE_SCENARIO_WEIGHTS
    }
    # Stable ordering makes the artifact reproducible. Prefer the lower blend
    # weight only on an exact tie because it is the train/serve-aligned expert.
    selected = min(scores, key=lambda weight: (scores[weight], weight))
    return float(selected), int(len(common))


def fit_direct_ensemble(
    canonical: pd.DataFrame,
    horizon_days: int,
    ridge_alpha: float = 4.0,
) -> DirectRidgeModel | DirectRidgeEnsembleModel | None:
    """Fit a time-selected, two-expert direct model outside inference.

    Returning a single expert remains a valid safe fallback when a corpus is
    too short for the second purged selection split.  That behavior keeps the
    protected evaluator robust on small organizer datasets while allowing the
    supplied corpus to use the stronger inference-aligned ensemble.
    """
    scenario_frame = training_frame(canonical, horizon_days, budget_training_mode="realized_future")
    aligned_frame = training_frame(canonical, horizon_days, budget_training_mode="inference_aligned")
    scenario_model = _fit_direct_ridge_frame(
        scenario_frame,
        horizon_days,
        ridge_alpha,
        "purged_temporal_holdout_support_aware_residual_quantiles_v3",
    )
    aligned_model = _fit_direct_ridge_frame(
        aligned_frame,
        horizon_days,
        ridge_alpha,
        "purged_temporal_holdout_inference_aligned_residual_quantiles_v1",
    )
    if scenario_model is None:
        return aligned_model
    if aligned_model is None:
        return scenario_model
    weight, sample_count = _select_ensemble_weight(scenario_frame, aligned_frame, horizon_days, ridge_alpha)
    if weight >= 1.0:
        return scenario_model
    if weight <= 0.0:
        return aligned_model
    return DirectRidgeEnsembleModel(
        horizon_days=horizon_days,
        scenario_model=scenario_model,
        inference_aligned_model=aligned_model,
        scenario_weight=weight,
        selection_sample_count=sample_count,
    )
