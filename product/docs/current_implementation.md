# Current Implementation Truth

## Scope

This document describes the code that executes today. It takes precedence over aspirational design notes elsewhere in the repository.

## Protected submission path

`run.sh` accepts zero to three arguments and defaults to `./data`, `./pickle/model.pkl`, and `./output/predictions.csv`. It imports only `src`, NumPy, Pandas, and the Python standard library. It performs deterministic inference only: no training, network calls, credentials, optional product imports, or runtime package downloads.

`src/output_adapter.py` is the sole `predictions.csv` writer. It declares the locked `horizon-v1` schema (21 columns), validates primitive values, maps compatibility aliases, applies explicit defaults only to optional presentation fields, and writes atomically. The locked header fixture is `product/tests/fixtures/horizon_v1_header.csv`.

`pickle/model_manifest.json` records the artifact SHA-256 and model version. When present beside the pickle, `src.predict` rejects mismatched artifacts.

## Shipped model

`pickle/model.pkl` contains `horizon-direct-ridge-v3-seasonal-plan`: one direct ridge model for each 30-, 60-, and 90-day aggregate horizon. Inputs include planned budget, trend, seasonal terms, channel, and campaign type. The runner builds campaign leaf forecasts, then forms deterministic joint draw paths through each leaf's P10/P50/P90 knots on a fixed percentile grid. Campaign-type, channel, and overall rows sum those draws path-wise before deriving ROAS and Pr(ROAS ≥ target). Rollup rows are flagged `joint_draw_rollup`. Under this comonotonic coupling, revenue and spend remain additive at shared percentiles; ROAS and target probabilities are computed from the aggregated paths rather than from ratio-of-quantile shortcuts.

When no future media plan is supplied, the 30/60-day baseline uses current run-rate delivery. The 90-day baseline applies a bounded historical calendar-month adjustment to avoid rolling a short-lived holiday spike into an entire quarter. This is a deterministic plan default only; any channel or campaign scenario budget overrides it. A zero campaign budget produces zero paid-media revenue rather than a positive model-intercept estimate.

Optional `media_plan.csv` in `DATA_DIR` supplies source-qualified campaign budgets per horizon (`source_system`, `source_campaign_id`, `horizon_days`, `planned_budget`) on the scored `src.predict` path. Absent that file, baseline defaults remain in force.

P10/P50/P90 revenue estimates are derived from chronological holdout residual quantiles created during training. Spend P10/P90 use historical horizon-length spend volatility and delivery versus configured budgets, centered on the planned budget as P50. These are empirical planning ranges, not conformal guarantees, causal estimates, or a Bayesian posterior. The current backtest report records coverage and sample counts by horizon; the 90-day slice has limited history and should be presented as a limitation, not a proof of universal calibration.

## Data semantics and guardrails

Google cost micros are normalized to currency units. The reader requires the full source-specific column contract after detecting its source identity; partial or corrupted source files fail with actionable errors. Duplicate day/campaign observations and inconsistent source-campaign hierarchies block inference. Campaign identifiers are source-qualified internally to prevent cross-channel collisions.

**Meta Ads assumption (explicit):** Meta's `conversion` field is treated as the supplied platform-attributed revenue proxy until the dataset owner confirms that business meaning. Explicit campaign-name patterns map prospecting, remarketing, and DPA campaigns into operational Meta types; a reviewed `campaign_taxonomy.csv` can override that mapping. Neither assumption establishes cross-platform attribution or causality.

Optional `source_semantics.csv` can declare currency, timezone, attribution method, and revenue-field labels. When absent, the runner warns that those comparability checks are unreviewed. Optional `media_plan.csv` is validated when present and never required for evaluator success.

## Optional product layer

`product/` contains local demo, scenario-planning, optimizer, evaluation, and evidence code. It is not imported by the evaluator path. The optimizer scores budget candidates with the same direct model used by scenario forecasting; it falls back to a documented conservative response curve only when no compatible direct model exists.

Decision briefs are **deterministic by default**: they organize sealed forecast evidence and never call a network model unless the caller explicitly sets `prefer_live_llm`. A missing API key, offline environment, quota failure, or service outage cannot block forecasting. Do not claim a live LLM narrative in demos unless an explicit smoke test succeeds.

## Open limitations

- The supplied guide does not publish a separate official scorer header. Until organizers provide one, Horizon locks on `horizon-v1` via the adapter and fixture; replace only after an official header arrives.
- The runner depends on Python 3.11+ because of the pinned NumPy and Pandas versions. GitHub Actions recreates that dependency installation and exact evaluator command on each push to `main`.
- Direct ridge with temporal residual quantiles is the shipped method—not an ensemble or conformal system.
- Forecasts are decision support, not promises of revenue or ROAS.
