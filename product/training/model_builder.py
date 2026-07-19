from __future__ import annotations

import hashlib
import json
import math

import pandas as pd

from product.training.direct_ridge import fit_direct_ridge
from product.training.residual_dependence import fit_residual_dependence
from src.direct_model import NUMERIC_FEATURES
from src.model import HorizonModel


def _canonical_training_fingerprint(canonical: pd.DataFrame) -> str:
    """Fingerprint the exact canonical data contract used to train an artifact."""
    columns = [
        "source_system", "source_campaign_id", "date", "channel",
        "campaign_type", "campaign_name", "spend", "revenue",
    ]
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
                "category_columns": list(model.category_columns),
                "categories": {name: list(values) for name, values in sorted(model.categories.items())},
            }
            for horizon, model in sorted(direct_models.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fit_horizon_model(
    canonical: pd.DataFrame,
    train_direct: bool = True,
    train_dependence: bool = True,
    horizons: tuple[int, ...] = (30, 60, 90),
) -> HorizonModel:
    """Train a model artifact outside the evaluator/import path."""
    valid = canonical[(canonical["spend"] > 0) & (canonical["revenue"] >= 0)].copy()
    if valid.empty:
        raise ValueError("Cannot train HorizonModel without positive-spend historical rows")
    roas = valid["revenue"].sum() / max(valid["spend"].sum(), 1e-9)
    daily_ratio = (valid["revenue"] / valid["spend"]).clip(lower=1e-5)
    sigma = float(daily_ratio.map(math.log).std(ddof=0))
    valid["month"] = valid["date"].dt.month
    monthly = valid.groupby("month")[["revenue", "spend"]].sum()
    factors = (monthly["revenue"] / monthly["spend"].clip(lower=1e-9) / roas).clip(0.55, 1.45)
    direct_models = {
        horizon: model
        for horizon in horizons
        if (model := fit_direct_ridge(canonical, horizon)) is not None
    } if train_direct else {}
    # Dependence is calibrated only in this training module.  The persisted
    # profile lets evaluator-safe inference assemble joint portfolio draws
    # without fitting, downloading, or retaining source history in memory.
    residual_dependence = {
        horizon: fit_residual_dependence(canonical, horizon)
        for horizon in direct_models
    } if train_dependence and direct_models else {}
    return HorizonModel(
        "horizon-direct-ridge-v5-oof-factor-copula" if direct_models else "horizon-statistical-v7-factor-copula",
        float(roas),
        max(0.20, min(sigma, 1.25)),
        {int(month): float(value) for month, value in factors.items()},
        direct_models,
        training_data_fingerprint=_canonical_training_fingerprint(canonical),
        feature_schema_fingerprint=_feature_schema_fingerprint(direct_models),
        residual_dependence=residual_dependence,
    )
