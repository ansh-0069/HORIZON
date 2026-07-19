# Horizon Forecasting Submission

## Project Overview

Horizon is an offline, probabilistic marketing revenue-forecasting utility for Google Ads, Meta Ads, and Microsoft Ads. It validates source exports, standardizes them into a canonical campaign-day schema, loads a sealed pre-trained model, and writes 30-, 60-, and 90-day forecasts at campaign, campaign-type, channel, and overall levels.

**This submission requires no internet connection, no API key, and performs inference only.**

The protected evaluator path is intentionally small. It reads CSV inputs, validates and canonicalizes them, loads `pickle/model.pkl`, and atomically writes `predictions.csv`. The local UI, training utilities, evaluation tools, and optional LLM narration live in `product/` and are never imported by `run.sh`.

## Repository Structure

```text
.
|-- run.sh                 Required evaluator entry point
|-- requirements.txt       Pinned protected-runtime dependencies
|-- data/                  Schema-compatible sample inputs
|-- pickle/
|   |-- model.pkl          Sealed pre-trained inference artifact
|   `-- model_manifest.json Artifact integrity and provenance metadata
|-- src/
|   |-- ingest.py          Source discovery and CSV loading
|   |-- canonicalize.py    Canonical schema construction
|   |-- validate.py        Input-quality checks
|   |-- model.py           Read-only model interface
|   |-- forecast.py        Forecast and hierarchy roll-up generation
|   |-- output_adapter.py  Sole predictions.csv writer
|   `-- predict.py         CLI orchestration
`-- product/               Optional product/demo material; not used by run.sh
```

## Requirements

- Bash-compatible shell (the evaluator entry point is `run.sh`)
- Python 3.11+
- `numpy==2.3.5`
- `pandas==3.0.1`

No database, cloud credential, service, network endpoint, GPU, environment variable, or runtime download is required.

## Installation

Install the pinned packages from `requirements.txt` using the evaluator's provisioned Python environment:

```bash
python -m pip install -r requirements.txt
```

For a fully offline local setup, install from a pre-provisioned wheelhouse rather than an online package index:

```bash
python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt
```

The prediction command itself never installs packages and never accesses a network.

## Offline Execution

`run.sh` caps BLAS backends at one thread. The model performs many small, serial linear-algebra operations; this avoids thread-pool overhead and keeps execution deterministic on shared evaluator machines.

When `pickle/model_manifest.json` is present, the runner verifies the artifact SHA-256 and model version before unpickling. The sealed v5 artifact also carries training-data and feature-schema fingerprints; the runner rejects a manifest/model fingerprint mismatch. These are local integrity checks only; they do not download or execute anything.

From the repository root:

```bash
./run.sh
```

Example:

```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

`run.sh` accepts zero to three positional arguments. Its defaults are `./data`, `./pickle/model.pkl`, and `./output/predictions.csv`; supplied arguments override those defaults. It selects `PYTHON_BIN` when explicitly provided, otherwise a runnable `python3`, `python`, or Windows `py -3` launcher; a non-runnable launcher fails with an actionable error. It then calls `python -m src.predict`; it does not train, download, or call an external service.

## Expected Inputs

`DATA_DIR` must contain exactly one schema-compatible CSV for each source. File names are not significant; headers identify the source. Numeric fields must parse as finite, non-negative numbers and date fields must parse as dates. Campaign IDs are preserved as text, including leading zeroes, and blank IDs are rejected.

An optional `campaign_taxonomy.csv` may contain `source_system`, `source_campaign_id`, `campaign_type`, and optional `review_status`. It is a **Meta-only** review artifact: only mappings where `source_system=meta_ads` and `review_status=reviewed` can override name-based Meta campaign-type inference. Missing or unreviewed mappings are accepted as metadata but do not change model features and emit a quality warning. Google Ads and Microsoft Ads taxonomy mappings are rejected rather than silently applying an unsupported taxonomy.

Microsoft `CampaignType` is a fixed source-adapter field, not a taxonomy override: known presentation variants are normalized before feature lookup (`Audience` to `DISPLAY`, `PerformanceMax` to `PERFORMANCE_MAX`, `Search` to `SEARCH`, and `Shopping` to `SHOPPING`). An unrecognized non-blank Microsoft type remains an upper-snake, quality-flagged value rather than being silently reassigned.

An optional `source_semantics.csv` can declare `source_system`, `currency`, `timezone`, `attribution_method`, `revenue_field`, and optional `review_status` for all three sources. A declared `revenue_field` is not a remapping mechanism: it must match the frozen canonical field for that source:

| Source | Canonical revenue field |
| --- | --- |
| Google Ads | `metrics_conversions_value` |
| Microsoft Ads | `Revenue` |
| Meta Ads | `conversion` |

The metadata can document data-owner review, but it cannot cause the protected path to select a different existing revenue column. One common declared currency/timezone is validated when semantics are supplied. Absence of optional semantics does not block evaluator inference and is surfaced as a warning.

An optional `media_plan.csv` may supply future campaign budgets for the scored offline path. Required columns: `source_system`, `source_campaign_id`, `horizon_days` (an exact integer `30`, `60`, or `90`), and `planned_budget` (finite and >= 0). Every key must match a campaign in the supplied source export and an active campaign in the current forecastable window; unknown or dormant keys fail closed instead of silently becoming no-ops. When absent, Horizon uses deterministic baseline-plan defaults. If the entire portfolio is dormant and no plan is supplied, it emits an explicit zero-plan forecast with a `portfolio_dormant_zero_plan` quality flag.

| Source | Required columns |
| --- | --- |
| Google Ads | `campaign_id`, `segments_date`, `campaign_advertising_channel_type`, `campaign_name`, `metrics_cost_micros`, `metrics_conversions_value`, `metrics_clicks`, `metrics_impressions`, `metrics_conversions`, `campaign_budget_amount` |
| Microsoft Ads | `CampaignId`, `TimePeriod`, `CampaignType`, `CampaignName`, `Revenue`, `Spend`, `Clicks`, `Impressions`, `Conversions`, `DailyBudget` |
| Meta Ads | `campaign_id`, `date_start`, `campaign_name`, `conversion`, `spend`, `clicks`, `impressions`, `daily_budget` |

`MODEL_PATH` must point to a compatible `HorizonModel` artifact, normally `pickle/model.pkl`. `OUTPUT_PATH` is the requested location of the generated CSV; its parent directory is created if needed.

## Expected Outputs

On success, `OUTPUT_PATH` is atomically replaced with a non-empty UTF-8 CSV. The current versioned contract (`horizon-v1`) is sorted by horizon and hierarchy and includes:

```text
forecast_id, horizon_days, level, channel, campaign_type, campaign_id,
campaign_name, planned_budget, predicted_revenue_p10,
predicted_revenue_p50, predicted_revenue_p90, predicted_spend_p10,
predicted_spend_p50, predicted_spend_p90, predicted_roas_p10,
predicted_roas_p50, predicted_roas_p90, probability_roas_above_target,
risk_score, quality_flags, model_version
```

`src/output_adapter.py` is the only component that writes this CSV. It applies the declared column order, type checks, compatibility aliases, and explicit defaults. See [`product/docs/output_adapter_contract.md`](product/docs/output_adapter_contract.md) for the schema contract and future-version procedure.

## Model Details

The sealed artifact is `horizon-direct-ridge-v5-oof-factor-copula`. It contains one pre-trained direct ridge model per 30-, 60-, and 90-day horizon with budget, trend, seasonality, channel, and campaign-type features. No fitting, hyperparameter search, feature-store write, or artifact mutation occurs in the runner.

Revenue intervals use **purged temporal holdout residual quantiles**: calibration rows occur strictly after each fit window, and no in-sample residual fallback is accepted. At hierarchy roll-up time, deterministic joint paths use a historical residual-factor copula plus portfolio out-of-fold (OOF) residual calibration. The OOF portfolio adjustment is empirical and **is not conformal prediction**.

P50 is reconciled exactly: for a horizon and scenario, leaf P50 revenue sums to campaign-type, channel, and overall P50 revenue. P10/P90 ranges and ROAS-target probabilities are derived from joint draw paths; they must not be summed across hierarchy rows. These are empirical planning guardrails, not revenue guarantees. See the [current implementation truth](product/docs/current_implementation.md) and [model card](product/docs/model_card.md).

## Data Validation

Before model loading, the pipeline:

- verifies one recognizable CSV per required advertising platform;
- canonicalizes Google cost micros to currency units;
- requires finite, non-negative spend, revenue, clicks, impressions, and conversions;
- rejects missing/blank campaign IDs, channels, or campaign types; normalizes a blank platform campaign name to its validated campaign ID; rejects malformed dates, duplicate `(source_system, campaign_id, date)` records, and incompatible source-campaign hierarchy records;
- validates optional media-plan IDs, horizons, and budgets; and
- reports non-blocking warnings for missing configured budgets and absent/unreviewed semantics or Meta taxonomy metadata.

The output adapter separately rejects missing required output fields, null/non-finite numeric values, invalid integer horizons, and an empty output. It does not invent missing prediction values.

## Assumptions

- Input files represent daily campaign statistics in a consistent currency and attribution convention.
- Campaign IDs are stable within each source system.
- Historical performance is sufficiently representative of the requested forecast period.
- Meta campaign names are deterministically classified as prospecting, remarketing, or DPA when explicit terms are present; only a reviewed Meta taxonomy mapping can override this inference.
- Meta's `conversion` field is treated as the supplied platform-attributed revenue proxy; confirm this business meaning before production use.
- The bundled model is compatible with the input data and requested 30/60/90-day horizons.

## Error Handling

The command fails closed with a non-zero exit status and an actionable `ERROR: offline prediction failed: ...` message for missing model/data paths, unsupported/missing source files, unreadable or malformed CSVs, quality blockers, manifest/artifact mismatches, incompatible pickle artifacts, and output-schema violations.

`predictions.csv` is written through a same-directory temporary file and replaced only after successful serialization. A write failure preserves any pre-existing output file and removes the temporary file.

## Reproducibility

- Dependencies are pinned in `requirements.txt`.
- The sealed artifact is `pickle/model.pkl`; the companion manifest pins SHA-256, model version, training-data fingerprint, and feature-schema fingerprint.
- Execution has no time-dependent API calls, random sampling, or network access.
- Output schema and ordering are versioned in `src/output_adapter.py`.
- The repository includes an optional evaluator rehearsal at `product/scripts/rehearse_submission.py`; it is not imported by `run.sh`.
- GitHub Actions recreates the Python 3.11 dependency install and runs the exact evaluator command on every `main` push.

### Locked output-contract gate

Until organizers publish a different scorer header, this repository locks on `horizon-v1` (`product/tests/fixtures/horizon_v1_header.csv`). After generating predictions, verify:

```bash
python -m product.scripts.verify_evaluator_contract \
  --predictions ./output/predictions.csv
```

If organizers later provide an official one-line header fixture, override explicitly:

```bash
python -m product.scripts.verify_evaluator_contract \
  --predictions ./output/predictions.csv \
  --official-header ./path/to/organizer_predictions_header.csv
```

Immediately before sharing the repository URL, require the reviewed local branch to be pushed:

```bash
python -m product.scripts.release_check --strict --require-upstream-sync
```

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `Usage: ./run.sh ...` | Provide no more than `DATA_DIR MODEL_PATH OUTPUT_PATH`. |
| Python not found | Install Python 3.11+ or set `PYTHON_BIN` to its executable. |
| Missing schema-compatible source files | Put one complete Google, Microsoft, and Meta CSV in `DATA_DIR`; ensure headers match the input table. |
| Duplicate source campaign-day records | Deduplicate upstream data by platform, campaign ID, and date. |
| `Model artifact is not a HorizonModel` | Use the supplied compatible `pickle/model.pkl`. |
| Manifest/fingerprint mismatch | Keep `pickle/model.pkl` and `pickle/model_manifest.json` together from the same release. |
| Output-schema failure | Ensure the pre-trained artifact and repository revision are used together; do not edit forecast output fields. |

## Known Limitations

- The model supports only the three declared source schemas and daily aggregates.
- Optional semantic metadata documents, but does not solve, cross-currency conversion or attribution comparability.
- An optional `media_plan.csv` can bind future campaign budgets to each horizon; without it, baseline plan defaults are used.
- The model does not model promotions, offline conversions, incrementality, attribution-method changes, or unobserved market shocks.
- Forecasts are decision-support estimates, not promises of revenue or ROAS.
- The locked output contract is `horizon-v1` until organizers publish a replacement header.
- The current six-fold backtest has weak 60-day calibration; the product layer fails closed for approval decisions at that horizon. See the model card.

## Future Improvements

- Add the official evaluator output fixture when the final schema is published.
- Add reviewed currency, timezone, attribution, and campaign-hierarchy integrations outside the protected path.
- Add retraining, monitoring, drift detection, and model-registry workflows outside the protected inference path.
- Add versioned media schemas and adapters without changing `run.sh`.
