# Horizon Product Layer

This directory is deliberately outside the hackathon evaluator path. It contains the local planner UI, optional OpenAI narrative integration, demo data, training utilities, backtests, release checks, tests, reports, and reference documents.

## Boundary contract

The protected submission runtime is the repository root only:

```text
run.sh
requirements.txt
src/
pickle/model.pkl
```

`./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH` imports only `src.predict`, uses the pre-trained pickle, and runs offline. It does not import this package. Do not add UI, credentials, network clients, training code, or product dependencies to `src/` or `requirements.txt`.

## Product commands

Run the local decision-support UI:

```bash
python -m product.app.server --host 127.0.0.1 --port 4174
```

Train a replacement model artifact outside the evaluator path:

```bash
python -m product.training.train --data-dir ./product/demo_data --output ./pickle/model.pkl
```

Run rolling-origin evaluation:

```bash
python -m product.scripts.evaluate_model --data-dir ./product/demo_data --folds 3
```

Run tests:

```bash
python -m unittest discover -s product/tests -v
```

Run repository-controlled submission release gates:

```bash
python -m product.scripts.release_check --strict
```

Verify that the local revision is pushed before submitting:

```bash
python -m product.scripts.release_check --strict --require-upstream-sync
```

## Optional OpenAI narrative

Only the **Generate grounded AI brief** UI action can use `OPENAI_API_KEY`; it reads `product/.env.local`. Forecasting, optimization, training, tests, and the submission runner do not need a key or internet connection. If the key, credits, or network are unavailable, the product UI falls back to deterministic evidence.

The API key is deliberately not a submission dependency. Before a live demo, verify provider quota and the selected model with `python -m product.scripts.verify_evidence_narrative`; do not claim a live AI narrative if that explicit smoke test cannot complete.

## Data and artifacts

- `demo_data/`: full history used by the planner UI, training, and evaluation.
- `models/`: backtest and rehearsal reports.
- `output/`: local product outputs and decision ledger; never used by the evaluator.
- `docs/`: implementation truth, model card, architecture notes, and hackathon reference PDFs.
