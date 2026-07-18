# Current Implementation Truth

## Scope

This document describes the code that executes today. It takes precedence over future-state material in the enterprise plan and production blueprint.

## Protected submission path

`run.sh` accepts zero to three arguments and defaults to `./data`, `./pickle/model.pkl`, and `./output/predictions.csv`. It imports only `src`, NumPy, Pandas, and the Python standard library. It performs deterministic inference only: no training, network calls, credentials, optional product imports, or runtime package downloads.

`src/output_adapter.py` is the sole `predictions.csv` writer. It declares the current `horizon-v1` schema, validates primitive values, maps compatibility aliases, applies explicit defaults only to optional presentation fields, and writes atomically.

## Shipped model

`pickle/model.pkl` contains `horizon-direct-ridge-v2`: one direct ridge model for each 30-, 60-, and 90-day aggregate horizon. Inputs include planned budget, trend, seasonal terms, channel, and campaign type. The runner builds campaign forecasts first and reconciles campaign-type, channel, and overall totals from those leaf outputs.

P10/P50/P90 revenue estimates are derived from chronological holdout residual quantiles created during training. They are empirical prediction ranges, not conformal guarantees, causal estimates, or a Bayesian posterior. The current backtest report records coverage and sample counts by horizon; the 90-day slice has limited history and should be presented as a limitation, not a proof of universal calibration.

## Data semantics and guardrails

Google cost micros are normalized to currency units. The reader requires the full source-specific column contract after detecting its source identity; partial or corrupted source files fail with actionable errors. Duplicate day/campaign observations and inconsistent source-campaign hierarchies block inference. Campaign identifiers are source-qualified internally to prevent cross-channel collisions.

Meta's `conversion` field is treated only as the supplied platform-attributed revenue proxy. A reviewed `campaign_taxonomy.csv` can override name-based campaign-type inference. Neither assumption establishes cross-platform attribution or causality.

## Optional product layer

`product/` contains local demo, scenario-planning, optimizer, evaluation, and evidence code. It is not imported by the evaluator path. The optimizer scores budget candidates with the same direct model used by scenario forecasting; it falls back to a documented conservative response curve only when no compatible direct model exists.

The optional LLM evidence client receives a bounded evidence packet and can only organize cited facts, assumptions, recommendations, and limitations. It does not calculate forecasts. A missing API key, offline environment, quota failure, or service outage falls back to deterministic evidence. The recent external smoke test reached the API but received HTTP 429, so no live narrative should be claimed until project quota is available.

## Open limitations

- The supplied guide does not publish the final evaluator `predictions.csv` schema. The current versioned adapter and fixture make a later contract update localized, but official columns and row granularity must be checked before submission.
- The runner depends on Python 3.11+ because of the pinned NumPy and Pandas versions.
- Direct ridge is a reproducible baseline, not the future ensemble or conformal system described in the blueprint.
- Forecasts are decision support, not promises of revenue or ROAS.
