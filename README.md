# Horizon Forecasting Submission

## Project Overview

Horizon generates probabilistic revenue, spend, and ROAS forecasts from historical Google Ads, Meta Ads, and Microsoft Ads campaign data. It writes forecasts for 30-, 60-, and 90-day horizons at campaign, campaign-type, channel, and overall levels.

**This submission requires no internet connection, no API key, and performs inference only.**

The protected evaluator path is deliberately small: it reads CSV inputs, validates and canonicalizes them, loads the supplied pre-trained `pickle/model.pkl`, and writes `predictions.csv`.

## Repository Structure

```text
.
├── run.sh                 # Required evaluator entry point
├── requirements.txt       # Runtime dependencies only
├── data/                  # Committed schema-compatible sample inputs
├── pickle/
│   └── model.pkl          # Pre-trained inference artifact
├── src/
│   ├── ingest.py          # Source discovery and CSV loading
│   ├── canonicalize.py    # Channel normalization
│   ├── validate.py        # Input-quality checks
│   ├── model.py           # Inference-only model interface
│   ├── forecast.py        # Forecast and roll-up generation
│   ├── output_adapter.py  # Sole predictions.csv writer
│   └── predict.py         # CLI orchestration
└── product/               # Optional product/demo material; not used by run.sh
```

## Requirements

- Bash-compatible shell (the evaluator entry point is `run.sh`)
- Python 3.11+
- `numpy==2.3.5`
- `pandas==3.0.1`

No database, cloud credential, service, network endpoint, GPU, or environment variable is required.

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

`run.sh` caps BLAS backends at one thread. The model performs many small,
serial linear-algebra operations; this prevents thread-pool overhead and keeps
runtime deterministic on shared evaluator machines.

When `pickle/model_manifest.json` is present, the runner verifies the model
artifact SHA-256 and version before unpickling it. This is an integrity check
for the committed trusted artifact; it does not download or execute anything.

From the repository root (with the committed sample files in `./data`):

```bash
./run.sh
```

Example:

```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

`run.sh` accepts zero to three positional arguments. Its defaults are `./data`, `./pickle/model.pkl`, and `./output/predictions.csv`; supplied arguments override those defaults. It selects `PYTHON_BIN` when explicitly provided, otherwise `python3`, `python`, or `py`. It then calls `python -m src.predict`; it does not train, download, or call an external service.

## Expected Inputs

`DATA_DIR` must contain exactly one schema-compatible CSV for each source. File names are not significant; headers identify the source. All numeric fields must be parseable as numbers and date fields must be parseable as dates.

An optional `campaign_taxonomy.csv` may contain `source_system`, `source_campaign_id`, and `campaign_type` to override name-based Meta campaign-type inference. It is not required by the evaluator.

An optional `media_plan.csv` may supply future campaign budgets for the scored offline path. Required columns: `source_system`, `source_campaign_id`, `horizon_days` (`30`, `60`, or `90`), and `planned_budget` (≥ 0). When absent, Horizon uses its baseline plan defaults.

| Source | Required columns |
| --- | --- |
| Google Ads | `campaign_id`, `segments_date`, `campaign_advertising_channel_type`, `campaign_name`, `metrics_cost_micros`, `metrics_conversions_value`, `metrics_clicks`, `metrics_impressions`, `metrics_conversions`, `campaign_budget_amount` |
| Microsoft Ads | `CampaignId`, `TimePeriod`, `CampaignType`, `CampaignName`, `Revenue`, `Spend`, `Clicks`, `Impressions`, `Conversions`, `DailyBudget` |
| Meta Ads | `campaign_id`, `date_start`, `campaign_name`, `conversion`, `spend`, `clicks`, `impressions`, `daily_budget` |

`MODEL_PATH` must point to the supplied compatible `HorizonModel` pickle artifact, normally `pickle/model.pkl`. `OUTPUT_PATH` is the requested location of the generated CSV; its parent directory is created if needed.

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

The serialized `HorizonModel` performs deterministic local inference. It uses a pre-trained direct ridge model per 30/60/90-day horizon with budget, trend, seasonality, channel, and campaign-type features. When no future budget is provided, the 90-day baseline plan is seasonally adjusted from historical monthly delivery; an explicit scenario or `media_plan.csv` budget always takes precedence. Revenue P10/P90 are chronological holdout residual quantiles; spend P10/P90 use historical delivery uncertainty around the plan. Hierarchy ROAS and ROAS-target probability are derived from deterministic joint draw paths, then risk score is attached. See the [current implementation truth](product/docs/current_implementation.md) and [model card](product/docs/model_card.md).

The model artifact is loaded read-only. No fitting, hyperparameter search, feature-store write, or artifact mutation occurs during execution.

## Data Validation

Before model loading, the pipeline:

- verifies one recognizable CSV per required advertising platform;
- canonicalizes Google cost micros to currency units;
- checks required dates, spend, revenue, and campaign IDs for nulls;
- rejects negative spend or revenue;
- rejects duplicate `(source_system, campaign_id, date)` records; and
- reports non-blocking warnings for missing configured budgets and uncertain Meta/taxonomy semantics.

The output adapter separately rejects missing required output fields, null/non-finite numeric values, invalid integer horizons, and an empty output. It does not invent missing prediction values.

## Assumptions

- Input files represent daily campaign statistics in a consistent currency and attribution convention.
- Campaign IDs are stable within each source system.
- Historical performance is sufficiently representative of the requested forecast period.
- Meta campaign names are deterministically classified as prospecting, remarketing, or DPA when those explicit terms are present; provide `campaign_taxonomy.csv` for reviewed overrides.
- Meta's `conversion` field is treated as supplied platform-attributed revenue and conversions in this schema; confirm that business meaning before production use.
- The bundled model is compatible with the input data and requested 30/60/90-day horizons.

## Error Handling

The command fails closed with a non-zero exit status and an actionable `ERROR: offline prediction failed: ...` message for missing model/data paths, unsupported/missing source files, unreadable or malformed CSVs, quality blockers, incompatible pickle artifacts, and output-schema violations.

`predictions.csv` is written through a same-directory temporary file and replaced only after successful serialization. A write failure preserves any pre-existing output file and removes the temporary file.

## Reproducibility

- Dependencies are pinned in `requirements.txt`.
- The model is supplied as `pickle/model.pkl` and is never modified by the runner.
- Execution has no time-dependent API calls, random sampling, or network access.
- Output schema and ordering are versioned in `src/output_adapter.py`.
- The repository includes an evaluator rehearsal at `product/scripts/rehearse_submission.py`; it is optional and not imported by `run.sh`.
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

Immediately before sharing the repository URL, also require that the reviewed local branch has been pushed:

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
| Output-schema failure | Ensure the pre-trained artifact and repository revision are used together; do not edit forecast output fields. |

## Known Limitations

- The model supports only the three declared source schemas and daily aggregates.
- An optional `source_semantics.csv` can declare one normalized currency,
  timezone, attribution method, and revenue-field definition for all sources.
  Without it, the runner emits a review warning rather than fabricating those
  assumptions from incomplete platform exports.
- An optional `media_plan.csv` can bind future campaign budgets to each
  horizon on the scored path; without it, baseline plan defaults are used.
- It does not adjust for cross-currency conversion, attribution-method changes, promotions, offline conversions, or unobserved market shocks.
- Forecasts are decision-support estimates, not revenue guarantees.
- The locked output contract is `horizon-v1` until organizers publish a replacement header.
- Meta Ads `conversion` is treated as attributed revenue only by documented assumption.

## Future Improvements

- Add official evaluator output fixtures when a final schema is published.
- Add currency, timezone, attribution, and campaign-hierarchy metadata validation.
- Add retraining, backtesting, calibration monitoring, and drift detection outside the protected inference path.
- Add additional media schemas through versioned ingestion and output adapters without changing `run.sh`.
