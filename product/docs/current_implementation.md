# Current Implementation Truth

## Scope

This document describes the code that executes today. It takes precedence over aspirational design notes elsewhere in the repository.

## Protected submission path

`run.sh` accepts zero to three arguments and defaults to `./data`, `./pickle/model.pkl`, and `./output/predictions.csv`. It imports only `src`, NumPy, Pandas, and the Python standard library. It performs deterministic inference only: no training, network calls, credentials, optional product imports, or runtime package downloads.

`src/output_adapter.py` is the sole `predictions.csv` writer. It declares the locked `horizon-v1` schema (21 columns), validates primitive values, maps compatibility aliases, applies explicit defaults only to optional presentation fields, and writes atomically. The locked header fixture is `product/tests/fixtures/horizon_v1_header.csv`.

`pickle/model_manifest.json` records the sealed v5 artifact SHA-256, model version, training-data fingerprint, and feature-schema fingerprint. When present beside the pickle, `src.predict` verifies SHA/version integrity and rejects manifest/model provenance mismatch before inference.

## Shipped model

`pickle/model.pkl` contains `horizon-direct-ridge-v5-oof-factor-copula`: one direct ridge model for each 30-, 60-, and 90-day aggregate horizon. Inputs include planned budget, recent delivery/trend features, future-horizon seasonal terms, channel, and campaign type. The runner builds campaign leaf forecasts, then creates deterministic joint paths using a hierarchical residual-factor copula fitted from expanding-window OOF residuals.

Marginal revenue P10/P90 values use **purged temporal holdout residual quantiles**: calibration observations are later than the data used to fit the corresponding direct model, and the training pipeline rejects in-sample calibration fallback. Portfolio OOF residual calibration is applied to joint leaf paths before hierarchy summary. It is an empirical residual-spread adjustment, **not conformal prediction**.

The P50 revenue point forecast is exactly reconciled: campaign P50 sums equal campaign-type, channel, and overall P50 for the same horizon/scenario. P10/P90 ranges and `probability_roas_above_target` are joint-path statistics and are not additive. Rollup rows carry `historical_residual_factor_copula_rollup`, `portfolio_oof_residual_calibration`, and `additive_p50_reconciled` quality flags when the v5 profile is used.

When no future media plan is supplied, the 30/60-day baseline uses current run-rate delivery. The 90-day baseline applies a bounded historical calendar-month adjustment to avoid rolling a short-lived holiday spike into an entire quarter. This is a deterministic plan default only; any source-qualified campaign scenario budget override takes precedence. A zero campaign budget produces zero paid-media revenue rather than a positive model-intercept estimate. If every campaign is dormant in the current window and no reactivation plan is supplied, inference preserves the hierarchy with explicit zero-plan forecasts and `portfolio_dormant_zero_plan`; it does not fail merely because no campaign is active.

Optional `media_plan.csv` in `DATA_DIR` supplies source-qualified campaign budgets per horizon (`source_system`, `source_campaign_id`, `horizon_days`, `planned_budget`) on the scored `src.predict` path. Horizons must be finite exact integers in `{30,60,90}`; budget values must be finite and non-negative. Each key must exist in the uploaded source data and be active in the forecastable window. Unknown or dormant keys fail closed rather than being silently ignored. Absent that file, baseline defaults remain in force.

## Evaluation posture

The committed evaluation report uses six rolling-origin prequential folds and records overall point error, coverage confidence intervals, and hierarchy join completeness. It is evidence for decision control, not a guarantee of future accuracy.

| Horizon | Revenue WAPE | Revenue coverage (95% CI) | ROAS coverage | Product decision posture |
| --- | ---: | ---: | ---: | --- |
| 30 days | 63.04% | 66.67% (30.0%-90.32%) | 83.33% | Informative but limited by six folds |
| 60 days | 103.67% | 33.33% (9.68%-70.0%) | 50.0% | Fails approval gate; revise or test |
| 90 days | 30.82% | 100.0% (60.97%-100.0%) | 66.67% | Informative but limited by six folds |

The 60-day result is weak on this limited dataset. The product layer must present it as a decision-support caution and forces `revise_or_test`; it is not an approval-grade confidence claim.

## Data semantics and guardrails

Google cost micros are normalized to currency units. The reader requires the full source-specific column contract after detecting source identity; partial or corrupted source files fail with actionable errors. Required metrics must be finite and non-negative. Duplicate day/campaign observations, blank campaign IDs/channels/campaign types, negative supplied configured budgets, and inconsistent source-campaign hierarchies block inference. Blank platform campaign names normalize to their validated campaign ID. Campaign identifiers are source-qualified internally to prevent cross-channel collisions.

**Microsoft campaign-type normalization (explicit):** source presentation labels map deterministically to a cross-channel feature vocabulary: `Audience` to `DISPLAY`, `PerformanceMax` to `PERFORMANCE_MAX`, `Search` to `SEARCH`, and `Shopping` to `SHOPPING`. This is a fixed source adapter, not an editable taxonomy override. An unrecognized non-blank Microsoft label remains an upper-snake, quality-flagged feature rather than being silently assigned to another tactic.

**Meta Ads assumption (explicit):** Meta's `conversion` field is treated as the supplied platform-attributed revenue proxy until the dataset owner confirms that business meaning. Explicit campaign-name patterns map prospecting, remarketing, and DPA campaigns into operational Meta types. A taxonomy override is accepted only for `source_system=meta_ads` and `review_status=reviewed`; unreviewed Meta mappings emit a warning and do not alter model features. Google Ads and Microsoft Ads external taxonomy mappings are rejected, rather than applying unsupported inference.

Optional `source_semantics.csv` can declare currency, timezone, attribution method, and revenue-field labels. A declared revenue field must exactly equal the frozen source field: `metrics_conversions_value` (Google Ads), `Revenue` (Microsoft Ads), or `conversion` (Meta Ads). Semantic metadata cannot redirect revenue extraction to another existing column. When optional semantics are absent or unreviewed, the runner emits a comparability-review warning. Optional `media_plan.csv` is validated when present and never required for evaluator success.

## Optional product layer

`product/` contains local demo, scenario-planning, optimizer, evaluation, and evidence code. It is not imported by the evaluator path. The optimizer scores budget candidates with the same direct model used by scenario forecasting; it falls back to a documented conservative response curve only when a compatible direct model is unavailable, including an unseen categorical value.

Decision briefs are **deterministic by default**: they organize sealed forecast evidence and never call a network model unless a locally enabled server and caller both explicitly request `prefer_live_llm`. A missing API key, offline environment, quota failure, or service outage cannot block forecasting. Do not claim a live LLM narrative in demos unless an explicit smoke test succeeds.

The product server does not evaluate or train during a request. It accepts a persisted evaluation report only when artifact SHA-256, model version, training-data fingerprint, feature-schema fingerprint, and canonical-data fingerprint match the loaded artifact and validated planner data. Otherwise Trust Center status is `not_applicable`. This artifact/report provenance binding prevents stale evidence from being shown as current.

## Open limitations

- The supplied guide does not publish a separate official scorer header. Until organizers provide one, Horizon locks on `horizon-v1` via the adapter and fixture; replace only after an official header arrives.
- The runner depends on Python 3.11+ because of pinned NumPy and Pandas versions. GitHub Actions recreates that dependency installation and exact evaluator command on each push to `main`.
- Direct ridge with purged temporal-holdout residual quantiles, factor-copula paths, and OOF residual calibration is the shipped method; it is neither an ensemble nor a conformal system.
- Forecasts are decision support, not promises of revenue or ROAS.
