from __future__ import annotations

import math

import pandas as pd

from product.training.direct_ridge import fit_direct_ridge
from src.model import HorizonModel


def fit_horizon_model(canonical: pd.DataFrame, train_direct: bool = True) -> HorizonModel:
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
    direct_models = {horizon: model for horizon in (30, 60, 90) if (model := fit_direct_ridge(canonical, horizon)) is not None} if train_direct else {}
    return HorizonModel(
        "horizon-direct-ridge-v2" if direct_models else "horizon-statistical-v4",
        float(roas),
        max(0.20, min(sigma, 1.25)),
        {int(month): float(value) for month, value in factors.items()},
        direct_models,
    )
