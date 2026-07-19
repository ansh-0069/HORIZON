# Horizon Model Card

## Shipped artifact

- **Artifact:** `pickle/model.pkl`
- **Version:** `horizon-direct-ridge-v3-seasonal-plan`
- **Execution:** deterministic, offline, inference-only
- **Horizons:** aggregate 30, 60, and 90 days
- **Targets:** campaign revenue P10/P50/P90; rollups derive spend and ROAS ranges

## Method

One direct ridge model is trained per horizon on time-respecting campaign cutoffs. Features include planned budget, recent and long-run ROAS, recent spend/revenue, active days, spend trend, future-horizon month encoding, channel, and campaign type. The model is trained outside the protected evaluator path and serialized into `pickle/model.pkl`.

If a future budget scenario is supplied, it is used directly. If no plan is supplied, the 30/60-day baseline uses current delivery run rate. The 90-day baseline applies a bounded calendar-month adjustment to avoid extrapolating a short holiday spike across an entire quarter. This is a plan-default assumption, not a claim that the model predicts media spend.

## Validation

The committed `product/models/evaluation_report.json` uses three rolling-origin folds on the committed demo historical data. The current direct model achieved:

| Horizon | Revenue WAPE | Revenue interval coverage | Nominal interval | Median calibration residuals |
| --- | ---: | ---: | ---: | ---: |
| 30 days | 27.7% | 100.0% | 80% | 141 |
| 60 days | 22.2% | 100.0% | 80% | 129 |
| 90 days | 17.4% | 100.0% | 80% | 113 |

Three folds are too few to claim a coverage guarantee. The UI labels those figures as directional and the report retains each fold’s actual and predicted values.

## Uncertainty and decision limits

P10/P90 use residual quantiles from later chronological training windows. They are empirical ranges, not conformal guarantees or causal confidence intervals. Spend P10/P90 remain planning assumptions around the selected plan. The ROAS-target probability is a distributional approximation and should be used as a guardrail signal, not a calibrated promise.

The scenario optimiser uses the same direct response model where available. It enforces campaign support caps and channel constraints, uses the target ROAS as a hard filter while feasible, and explicitly reports when it must relax that guardrail to allocate the requested total budget.

## Data limits

Google cost micros are normalized. Meta campaign types are inferred only from explicit name terms such as prospecting, remarketing, and DPA; unresolved names remain `Generic` unless an optional reviewed `campaign_taxonomy.csv` is provided. Meta `conversion` is treated as platform-attributed revenue only by documented assumption. The model does not perform cross-channel attribution, currency conversion, incrementality measurement, or promotion/shock forecasting.
