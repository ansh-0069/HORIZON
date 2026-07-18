# Horizon MVP

Horizon is an offline-first probabilistic revenue-planning utility for e-commerce media. It standardizes Google Ads, Microsoft Ads, and Meta Ads campaign data and produces 30/60/90-day revenue and ROAS ranges at campaign, campaign-type, channel, and blended levels.

## What is implemented

- Dynamic source discovery by CSV schema rather than filename.
- Canonicalization with Google cost-micros conversion and explicit Meta metric/taxonomy warnings.
- Blocking data-quality validation for duplicate campaign-days, required values, invalid metrics, and missing sources.
- Pre-trained, pickled statistical forecast artifact with recent performance, seasonal adjustment, sparse-history fallback, and budget extrapolation risk.
- P10/P50/P90 revenue, spend, and ROAS; ROAS target probability and transparent risk score.
- Campaign-to-type-to-channel-to-total reconciled output.
- Offline submission runner with no API or LLM dependency.

## Quick start

Python 3.11+ is required. Install the pinned packages:

```bash
python -m pip install -r requirements.txt
python -m src.train --data-dir ./data --output ./pickle/model.pkl
bash ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

On Windows, use the equivalent command if Bash is unavailable:

```powershell
python -m src.predict --data-dir ./data --model ./pickle/model.pkl --output ./output/predictions.csv
```

## Evaluator contract

The required entry point is:

```bash
./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH
```

It reads schema-compatible CSV files dynamically from `DATA_DIR`, loads the existing artifact from `MODEL_PATH`, and overwrites `OUTPUT_PATH`. It makes no network calls and does not retrain at runtime.

The supplied hackathon PDFs do not specify the final scorer CSV columns. `src/output_adapter.py` centralizes the output schema and is the only module that should change when organizers publish it.

Before submission, run a clean-room rehearsal. It copies data and the pickle into a temporary evaluator-like directory, runs `run.sh`, validates the emitted contract, and confirms that inference did not modify the model artifact:

```bash
python -m scripts.rehearse_submission --data-dir ./data --model ./pickle/model.pkl
```

On Windows, pass a Bash path, for example `--bash "C:\\Program Files\\Git\\bin\\bash.exe"`.

## Local planner UI

The planner UI is intentionally isolated from the evaluator path and uses only the Python standard library for serving:

```bash
python -m app.server --host 127.0.0.1 --port 4174
```

Open [http://127.0.0.1:4174](http://127.0.0.1:4174). It supports 30/60/90-day scenarios, target-ROAS guardrails, Google/Meta/Microsoft budget inputs, channel drill-down, data-health warnings, and a deterministic evidence brief. It does not call an LLM or external service.

The **Recommend allocation** action uses a deterministic discrete optimizer over the same concave campaign response curves used by the forecast. It enforces campaign support caps and optional channel minimum/maximum constraints, penalizes candidates below the selected ROAS guardrail, and validates the chosen allocation through the shared forecasting pipeline. It is decision support, not an automated media-buying action.

## Trust and decision ledger

Run rolling-origin evaluation before demoing or promoting a model:

```bash
python -m scripts.evaluate_model --data-dir ./data --output ./models/evaluation_report.json --folds 3
```

The local Trust Center computes the same report on demand. It compares the direct multi-horizon ridge forecaster against the statistical fallback, displays median forecast error and empirical revenue-interval coverage for 30/60/90-day historical windows, and does not hide a regression behind aggregate scores. **Save decision** persists the scenario, forecast ID, summary, timestamp, and decision state to `output/horizon_decisions.sqlite`; it is an auditable local MVP equivalent of the production decision ledger.

## Data assumptions and warnings

- Existing platform attribution is used as supplied; Horizon does not create an attribution model.
- Google `metrics_cost_micros` is converted by dividing by 1,000,000.
- Meta's `conversion` field is treated provisionally as attributed revenue and produces a semantic-review warning. Confirm it with the organizer/client before a production forecast.
- Mixed currency, invalid/missing revenue or spend, duplicate campaign-day facts, or missing source schemas block a run.
- A forecast is conditional, not causal. The optional production LLM layer explains model evidence but never generates predictions.

## Optional grounded AI narrative

The product has an optional, separate `/api/evidence` endpoint for a decision-ready narrative. It uses an OpenAI Responses API structured-output contract, but only after the deterministic forecast has completed. The endpoint sends a compact evidence packet—not raw campaign rows—and validates all narrative list items against approved evidence IDs. It rejects numerical claims, causal language, attempts to change the deterministic decision, and uncited statements. If the service is unavailable, it returns the deterministic evidence brief instead.

The local key belongs in `.env.local` and must never be committed:

```text
OPENAI_API_KEY=...
# Optional; defaults to gpt-5.6-luna
HORIZON_LLM_MODEL=gpt-5.6-luna
```

This endpoint is excluded from `run.sh`; the offline evaluator still has no API or LLM dependency.

## Output

`predictions.csv` includes a `forecast_id`, horizon, hierarchy level, forecasted revenue/spend/ROAS P10/P50/P90, probability of clearing the default 4.0 ROAS target, risk score, quality flags, and model version.

## Tests

```bash
python -m unittest discover -s tests -v
```
