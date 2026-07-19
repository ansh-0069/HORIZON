# Horizon Product Layer

This directory is deliberately outside the hackathon evaluator path. It contains the local planner UI, optional OpenAI narrative integration, demo data, training utilities, backtests, release checks, tests, reports, and reference documents.

## Boundary contract

The protected submission runtime is the repository root only:

```text
run.sh
requirements.txt
src/
pickle/model.pkl
pickle/model_manifest.json
```

`./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH` imports only `src.predict`, uses the sealed pre-trained pickle, and runs offline. It does not import this package. Do not add UI, credentials, network clients, training code, or product dependencies to `src/` or `requirements.txt`.

The sealed evaluator artifact is `horizon-direct-ridge-v5-oof-factor-copula`. Its protected companion manifest records its SHA-256, version, training-data fingerprint, and feature-schema fingerprint. A product model may be trained locally, but it is not an evaluator artifact until a reviewed promotion replaces both protected files together.

## Product commands

Run the local decision-support UI:

```bash
python -m product.app.server --host 127.0.0.1 --port 4174
```

Train a replacement model artifact outside the evaluator path:

```bash
python -m product.training.train --data-dir ./product/demo_data
```

This writes `product/models/horizon_model.pkl` and a separately versioned `product/models/horizon_model.manifest.json`. The trainer refuses `--output ./pickle/model.pkl` (including path aliases): that evaluator artifact is frozen and may only be changed by the reviewed release/promotion process that updates its matching protected manifest.

Run rolling-origin evaluation:

```bash
python -m product.scripts.evaluate_model --data-dir ./product/demo_data
```

The default is six chronological folds over `product/demo_data`. Each fold builds a fold-specific model using only data available at its forecast origin. The resulting report records canonical-data provenance, fold model versions, calibration sample counts, and hierarchy join completeness; it does not retrain or mutate the protected pickle in place.

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

## Current evaluation posture

The committed six-fold report evaluates overall forecasts with purged temporal-holdout residual quantiles. It is evidence for risk management, not a claim of guaranteed coverage.

| Horizon | Revenue WAPE | Revenue interval coverage (95% CI) | ROAS interval coverage | Decision posture |
| --- | ---: | ---: | ---: | --- |
| 30 days | 63.04% | 66.67% (30.0%-90.32%) | 83.33% | Informative but limited by six folds |
| 60 days | 103.67% | 33.33% (9.68%-70.0%) | 50.0% | Fails the product approval gate; revise or test |
| 90 days | 30.82% | 100.0% (60.97%-100.0%) | 66.67% | Informative but limited by six folds |

The planner therefore treats a 60-day recommendation as `revise_or_test`, even when its point forecast or local model guardrail appears favorable. This is intentional fail-closed product behavior, not a statement that the forecast cannot be viewed.

## Decision briefs (deterministic by default)

The planner **Generate deterministic brief** action organizes sealed forecast evidence and never requires a network call. The visible UI always sends `prefer_live_llm: false`; it does not read an API key or invoke a provider.

Optional live LLM narration is a separate, explicit API-only path. It is disabled server-side by default, even if an API key exists. It can be enabled only for a local development server bound to `127.0.0.1`, `::1`, or `localhost`:

```bash
python -m product.app.server --host 127.0.0.1 --enable-live-llm
```

After that explicit server opt-in, a caller must still send JSON `prefer_live_llm: true` and configure `OPENAI_API_KEY` in `product/.env.local` (or the local process environment). Merely configuring a key does not opt the planner into live narration. A live narrator can summarize a sealed evidence packet only; it cannot change model forecasts, uncertainty ranges, optimization results, or decision status.

Forecasting, optimization, training, tests, and the submission runner do not need a key or internet connection. Do not claim a live AI narrative in demos unless `python -m product.scripts.verify_evidence_narrative` succeeds.

## Planner trust and scenario integrity

The product server does not train or run a rolling-origin backtest while serving a request. It reads `models/evaluation_report.json` only when its artifact provenance exactly matches the loaded model: artifact SHA-256, model version, training-data fingerprint, and feature-schema fingerprint. The report's canonical-data fingerprint must also match the validated planner dataset. Otherwise the Trust Center reports `not_applicable`; it never substitutes on-demand evaluation or stale evidence.

Approval posture is fail-closed on the selected horizon's calibration evidence. An unavailable, stale, undersized, or materially under-covering report forces `revise_or_test`, even if the forecast guardrail itself looks favorable.

Every API response includes the exact source-qualified `campaign_budgets` used for inference. After **Recommend allocation**, the UI pins that returned campaign-level plan for the decision brief and SQLite ledger save. Editing a planner input marks the displayed scenario stale and disables those actions until it is rerun, so the UI cannot save or narrate a rounded channel-total reconstruction.

## Forecast semantics

The v5 model uses purged temporal holdout residual quantiles for campaign marginal revenue intervals. Deterministic factor-copula paths model aggregate dependence, and portfolio OOF residual calibration adjusts the aggregate empirical spread. That portfolio step is **not conformal prediction**. P50 revenue is reconciled exactly across the campaign, campaign-type, channel, and overall hierarchy; P10/P90 ranges and ROAS probabilities come from joint paths and must not be added across rows.

## Prototype disclosure: implemented vs. roadmap

The planner is intentionally a **single-user local prototype**, not a deployed multi-tenant SaaS. Present the following distinction accurately in a live demo:

| Implemented in this repository | Production roadmap; not implemented here |
| --- | --- |
| Offline validation, canonicalization, sealed pre-trained inference, scenario forecasts, and empirical P10/P50/P90 ranges | Tenant identity, permissions, data isolation, and managed cloud deployment |
| Constraint-aware allocation recommendation and local SQLite decision ledger | Job orchestration, observability, and model/feature registry governance |
| Deterministic decision brief grounded in artifact-bound forecast evidence | Reviewed platform attribution integrations and portfolio calibration at agency scale |

This disclosure is also rendered in the UI so that judges can distinguish shipped functionality from the architecture roadmap.

## Data and artifacts

- `demo_data/`: full history used by the planner UI, training, and evaluation.
- `models/`: optional local model artifacts, provenance manifests, backtest, and rehearsal reports.
- `output/`: local product outputs and decision ledger; never used by the evaluator.
- `docs/`: implementation truth, model card, architecture notes, and hackathon reference PDFs.
