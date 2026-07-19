# Horizon Model Card

## Shipped artifact

| Field | Value |
| --- | --- |
| Artifact | `pickle/model.pkl` |
| Version | `horizon-direct-ridge-v5-oof-factor-copula` |
| SHA-256 | `7f88607a703d529564432a0fa43c99cf2b9669b93948fef67ffffcf5795d83fa` |
| Artifact build runtime | Python 3.12 |
| Minimum supported runtime | Python 3.11 |
| Training-data fingerprint | `e2778df8a45675c6d182173d5400e98339ca6cbf3f8554292faa27dff20a01f4` |
| Feature-schema fingerprint | `f51e8603f9cc1710fd9f7c873163912b52f399afdda92c33a1aa0fce5d520a32` |
| Execution | Deterministic, offline, inference-only |
| Horizons | Aggregate 30, 60, and 90 days |
| Targets | Campaign revenue and spend P10/P50/P90; hierarchy ROAS and ROAS-target probability from joint paths |

The protected manifest contains the same artifact SHA-256, version, and fingerprints. `src.predict` rejects a sealed manifest/model mismatch before inference.

## Method

One direct ridge model is trained per horizon on time-respecting campaign cutoffs. Features include planned budget, recent and long-run ROAS, recent spend/revenue, active days, spend trend, future-horizon month encoding, channel, and campaign type. Training is outside the protected evaluator path; the evaluator only deserializes the pre-trained model.

If a future budget scenario is supplied through product overrides or optional `media_plan.csv`, it takes precedence. Otherwise, 30/60-day baseline budgets use current delivery run rate and the 90-day baseline uses a bounded month-mix adjustment.

Campaign revenue marginal intervals use **purged temporal holdout residual quantiles**. A calibration row is strictly later than its model fit window; the training path does not fall back to in-sample residuals when a safe calibration partition is unavailable. At roll-up time, a deterministic residual-factor copula orders campaign draw paths using global, channel, campaign-type, and idiosyncratic dependence fitted from expanding-window OOF residuals. Portfolio OOF residual calibration then adjusts the aggregate empirical spread. It is deliberately described as OOF residual calibration, **not conformal prediction**.

P50 revenue has an exact additive reconciliation invariant for every horizon/scenario: campaign P50 values sum to campaign-type, channel, and overall P50 values. P10/P90 and ROAS-target probabilities are statistics of joint paths, so they are not additive and must never be summed across rows.

## Validation

`product/models/evaluation_report.json` records a six-fold rolling-origin prequential backtest over the canonical historical training exports. Each fold uses only rows available at its origin; the report also records provenance, calibration sample counts, and hierarchy join completeness.

| Horizon | Revenue WAPE | Revenue interval coverage | 95% CI | ROAS interval coverage | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| 30 days | 63.04% | 66.67% | 30.0%-90.32% | 83.33% | Limited six-fold evidence; not a coverage guarantee |
| 60 days | 103.67% | 33.33% | 9.68%-70.0% | 50.0% | Fails the approval gate; use only for revise-or-test decisions |
| 90 days | 30.82% | 100.0% | 60.97%-100.0% | 66.67% | Limited six-fold evidence; not a coverage guarantee |

The nominal revenue interval is 80%. Six folds are too few to establish stable coverage, and the broad Wilson intervals make this especially clear. The product layer must not describe the 30- or 90-day values as guaranteed calibration. It explicitly fails closed for 60-day approval decisions.

## Uncertainty and decision limits

Revenue P10/P90 use later, purged temporal-holdout residual quantiles. Spend P10/P90 use historical horizon-length spend volatility and delivery versus configured budgets, centered on planned budget as P50. Deterministic factor-copula joint paths and portfolio OOF residual calibration provide aggregate dependence/spread. These are empirical ranges, not conformal guarantees, causal confidence intervals, or Bayesian posteriors. The ROAS-target probability is a guardrail signal, not a calibrated promise.

The scenario optimizer uses the same direct response model where available. It enforces campaign support caps and channel constraints, applies a target-ROAS filter while feasible, and explicitly exposes when that marginal guardrail is relaxed.

## Data contract and semantic limits

Google cost micros are normalized to currency units. Campaign identifiers remain source-qualified to prevent cross-channel collisions. Microsoft export presentation variants are deterministically mapped to the cross-channel vocabulary (`Audience` to `DISPLAY`, `PerformanceMax` to `PERFORMANCE_MAX`, `Search` to `SEARCH`, and `Shopping` to `SHOPPING`); unknown non-blank values remain disclosed upper-snake labels rather than being silently reclassified. Meta campaign types are inferred from explicit name terms unless a **reviewed Meta-only** `campaign_taxonomy.csv` mapping is supplied. Unreviewed mapping rows are retained as warnings but cannot change the model feature; Google and Microsoft external taxonomy mappings are rejected.

`source_semantics.csv` can document currency, timezone, attribution method, and a revenue-field label. Its `revenue_field` must equal the protected canonical source field (`metrics_conversions_value` for Google Ads, `Revenue` for Microsoft Ads, and `conversion` for Meta Ads); it cannot remap revenue to an arbitrary column. Meta `conversion` remains a documented platform-attributed revenue assumption. The model does not perform cross-channel attribution, currency conversion, incrementality measurement, or promotion/shock forecasting.

## Intended use

Use Horizon to compare budget scenarios, inspect plausible revenue/ROAS ranges, and decide when a controlled test is safer than approval. Do not use it as a revenue commitment, a causal attribution system, or a substitute for business-owner validation of semantics and taxonomy.
