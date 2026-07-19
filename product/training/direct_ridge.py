from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.direct_model import DirectRidgeModel, _features


CATEGORY_COLUMNS = ("channel", "campaign_type")
MINIMUM_DIRECT_FIT_ROWS = 40
MINIMUM_CALIBRATION_ROWS = 20


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
) -> pd.DataFrame:
    """Build direct-horizon training examples.

    ``aligned_cutoffs`` is reserved for residual-dependence calibration.  The
    point model retains its existing per-campaign cadence, while the
    dependence estimator needs the same historical forecast-origin dates
    across campaigns in order to measure co-movement rather than accidentally
    treating unrelated dates as a joint residual observation.
    """
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
            planned_budget = float(future["spend"].sum())
            if len(history) < 28 or planned_budget <= 0:
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
    if len(frame) < 80:
        return None
    fit_frame, calibration_frame, calibration_start = _temporal_calibration_partitions(frame, horizon_days)

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
        uncertainty_method="purged_temporal_holdout_residual_quantiles_v2",
    )
