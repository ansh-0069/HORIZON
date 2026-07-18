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
- Python 3.10+
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

From the repository root:

```bash
chmod +x run.sh
./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH
```

Example:

```bash
./run.sh ./product/supplied_data ./pickle/model.pkl ./output/predictions.csv
```

`run.sh` requires exactly three arguments. It selects `PYTHON_BIN` when explicitly provided, otherwise `python3`, `python`, or `py`. It then calls `python -m src.predict`; it does not train, download, or call an external service.

## Expected Inputs

`DATA_DIR` must contain exactly one schema-compatible CSV for each source. File names are not significant; headers identify the source. All numeric fields must be parseable as numbers and date fields must be parseable as dates.

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

The serialized `HorizonModel` performs deterministic local inference. It estimates campaign-level future revenue using recent and longer-term performance, budget response, seasonal month factors, and a pre-trained direct ridge model where available. Revenue uncertainty is emitted as P10/P50/P90 estimates; spend intervals, ROAS intervals, ROAS-target probability, and risk score are then derived and reconciled into higher hierarchy levels.

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
- Meta's `conversion` field is treated as both revenue and conversions in this supplied schema; validate that business meaning before production use.
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

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `Usage: ./run.sh ...` | Provide exactly `DATA_DIR MODEL_PATH OUTPUT_PATH`. |
| Python not found | Install Python 3.10+ or set `PYTHON_BIN` to its executable. |
| Missing schema-compatible source files | Put one complete Google, Microsoft, and Meta CSV in `DATA_DIR`; ensure headers match the input table. |
| Duplicate source campaign-day records | Deduplicate upstream data by platform, campaign ID, and date. |
| `Model artifact is not a HorizonModel` | Use the supplied compatible `pickle/model.pkl`. |
| Output-schema failure | Ensure the pre-trained artifact and repository revision are used together; do not edit forecast output fields. |

## Known Limitations

- The model supports only the three declared source schemas and daily aggregates.
- It does not adjust for cross-currency conversion, attribution-method changes, promotions, offline conversions, or unobserved market shocks.
- Forecasts are decision-support estimates, not revenue guarantees.
- The current output contract is a project-defined schema pending any final evaluator template.

## Future Improvements

- Add official evaluator output fixtures when a final schema is published.
- Add currency, timezone, attribution, and campaign-hierarchy metadata validation.
- Add retraining, backtesting, calibration monitoring, and drift detection outside the protected inference path.
- Add additional media schemas through versioned ingestion and output adapters without changing `run.sh`.
