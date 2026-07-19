# Live Demo Runbook: Horizon

## Objective

Demonstrate a credible decision-support product, not a dashboard or an LLM wrapper. In six minutes, show that Horizon can ingest validated cross-channel history, simulate a budget plan, expose uncertainty and evidence limits, prioritize a controlled test allocation, and preserve a trustworthy decision record.

The core story is: **a media planner can ask what a future budget plan is likely to deliver, see the uncertainty, and receive a bounded recommendation that knows when it should not approve a plan.**

## Demo stance

- Use the local planner only. Start it without `--enable-live-llm`.
- Open on the default **30-day** view with a ROAS guardrail near sample blended ROAS (~3.2). Lead with a usable planning scenario first.
- After the first simulate, switch to **60 days** as an intentional trust demonstration: its persisted calibration evidence is weak, so Horizon returns `revise_or_test` rather than pretending that a point forecast is approval-grade.
- Describe the displayed draw share as a **simulated guardrail draw share**, not the real-world probability of meeting ROAS.
- Describe allocation as a **shape-constrained, test-priority policy**, not causal incrementality, autonomous bidding, or a promise of optimal spend.
- Use the deterministic brief. The optional LLM is a bounded narration layer and must never be presented as the forecasting system.

## Narration modes and network boundary

| Mode | How it becomes available | Network / credential behavior | What the audience should hear |
| --- | --- | --- | --- |
| Default decision brief | Start the local planner normally and click **Generate deterministic brief**. | No network request, API key, or provider configuration. | "This is the normal product path. It is generated from sealed forecast evidence." |
| Optional live narration | Start a localhost-only server with `--enable-live-llm`, configure a local credential, then click **Generate optional live narration**. | One explicit network request may be made only after all three actions. The browser remains single-origin and never contacts a provider directly. | "This is optional presentation language, not the model or source of record." |
| Safe fallback | The provider is disabled, unconfigured, unavailable, invalid, or the output violates the evidence contract. | No replacement network attempt. The deterministic evidence brief remains available. | "The product preserved the decision evidence rather than accepting unverified prose." |

The live narrator receives only a post-forecast evidence packet: decision posture, approved source IDs, model ranges, bounded contributor summaries, and generic risk statements. It does not receive raw media records, does not control inference, cannot change the decision, and treats every packet field as data rather than instructions.

## Preflight (before judges enter)

1. From the repository root, start the local planner with the sealed artifact:

   ```bash
   python -m product.app.server --host 127.0.0.1 --port 4174
   ```

2. Open `http://127.0.0.1:4174` and wait for the baseline to finish loading.
3. Keep a terminal tab ready with the exact protected-path command, but do not lead with it:

   ```bash
   ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
   ```

4. Confirm the browser shows the default **30-day** baseline (ROAS guardrail ~3.2), Data Health, Trust Center, and the model-boundary panel.
5. Do not configure an API key or enable live narration for the core demo. The disabled **Generate optional live narration** button is useful evidence of a deliberate boundary.
6. Prepare one local backup: a browser screenshot/PDF of the loaded baseline and a terminal capture of a successful protected-path output. Store both locally; they are contingency material, not simulated live results.

## Six-minute script

| Time | Screen and exact action | Talking points | Expected judge takeaway |
| --- | --- | --- | --- |
| 0:00-0:35 | Start on the loaded Horizon home screen (default 30-day). Do not touch controls yet. | "NetElixir planners need to answer: if we change a cross-channel budget for the next 30, 60, or 90 days, what revenue and ROAS range should we plan around? A single point forecast invites false certainty, so Horizon is built around scenario evidence and operational limits." | This solves a real planning decision, not a generic analytics-display problem. |
| 0:35-1:10 | Scroll to **Trust Center: Data health and backtest**. Point to source/semantic warnings if present. | "The product begins with data contract checks. Google cost micros are normalized; invalid numeric fields, blank IDs, duplicate campaign-days, and broken hierarchies fail closed. Meta revenue semantics and taxonomy review are displayed rather than silently inferred." | The system handles the messy agency inputs that make forecast demos unreliable. |
| 1:10-1:50 | Return to controls. Keep 30-day selected. Raise Google by ~10%, click **Simulate plan**. Show revenue/ROAS ranges and draw share. | "This is a conditional scenario: same validated history, a new stated budget plan. The model predicts; no LLM is involved. The exact source-qualified campaign plan is pinned before it can be saved or briefed." | The planner supports a concrete what-if workflow. |
| 1:50-2:25 | Switch horizon to **60 days**, click **Simulate plan** again. Point to Trust Center / decision posture `revise_or_test`. | "Persisted rolling-origin evidence for 60 days is weak on this history, so the product refuses approval-grade posture. That fail-closed behavior is what you want before client spend is committed." | Trust controls are real product logic, not presentation copy. |
| 2:25-2:55 | Stay on the latest scenario and point to **Expected revenue**, **Expected blended ROAS**, **Simulated guardrail draw share**, **Decision posture**, and **Baseline versus current scenario**. | "P10-P90 is an empirical planning range. The dot is P50. The draw share is a model-path diagnostic for the guardrail, not a real-world approval probability. P50 rolls up additively; interval endpoints are joint-path statistics and are not summed across rows." | The team understands probabilistic forecasting and avoids misleading probability language. |
| 2:55-3:35 | Scroll to **Campaign priority queue**. Point out support flags and the campaign-level ranges. | "We retain campaign-level uncertainty instead of hiding risk in a blended KPI. An unseen category or weak support does not silently receive a confident recommendation; it takes the documented conservative fallback path." | The model behaves defensibly at the practical edge cases judges expect. |
| 3:35-4:15 | Click **Prioritize test allocation**. Show **Test-priority allocation** and the target-constraint status. | "This is not an autonomous budget allocator. It applies campaign support caps, channel constraints, and a marginal ROAS guardrail where feasible. If a constraint must be relaxed, the UI says so. The output is a test-priority plan to validate, not a causal claim of incremental revenue." | Recommendations are constrained, explainable, and operationally safe. |
| 4:15-4:50 | Click **Generate deterministic brief**. Show facts, assumptions, validation recommendation, limitations, and evidence IDs. | "The brief is deterministic and comes from sealed forecast evidence. It separates facts, assumptions, recommendations, and limitations. No API key and no internet are required." | AI integration is bounded and evidence-first rather than hallucinated prose. |
| 4:50-5:20 | Expand **Optional live narration (explicit network opt-in)** but do not enable it. Point to the disabled control and the model-boundary panel. | "An optional narrator can be enabled only on localhost with an explicit server flag, local credential, and a separate button click. It receives a sealed evidence packet, must cite evidence IDs, treats packet fields as data, and cannot alter forecasts, ranges, allocation policy, or decision status. The demo does not depend on it." | The LLM is correctly separated from the forecasting and control plane. |
| 5:20-5:45 | Click **Save decision** only after the optimized plan has been returned. Point to the scenario-pinned message. | "The ledger saves the exact campaign-level plan used for inference. If I change an input, the scenario is marked stale and saving/briefing is blocked until I rerun it." | Auditability is implemented at the decision boundary, not merely promised in architecture slides. |
| 5:45-6:00 | Return to the top-level result or use a closing slide. | "Horizon turns fragmented platform exports into a reproducible forecast-and-test workflow: offline inference for evaluation, a product layer for evidence and scenarios, and explicit guardrails wherever confidence is not earned." | The product is ambitious, technically disciplined, and honest about what remains a roadmap. |

## Suggested transitions

Use these short transitions to keep the story coherent:

1. "Before we forecast, we prove the data is fit to forecast."
2. "Before we recommend, we show what the historical evidence can and cannot support."
3. "Before we let someone save a decision, we pin the exact scenario that created it."
4. "Before we add language generation, we lock the numerical evidence and its boundaries."

## Offline and failure backup path

The primary demo is already local/offline: the local planner, model, and demo data run without a network. Do not wait for an optional LLM call during the jury session.

If the browser is unavailable or the local server fails:

1. Say: "The decision UI is a product layer; the evaluator-critical path is the same sealed model running offline."
2. Run the prepared terminal command:

   ```bash
   ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
   ```

3. Open the generated local `./output/predictions.csv` in the editor and show that it includes 30/60/90-day campaign, campaign-type, channel, and overall rows with P10/P50/P90, ROAS, risk, and model version.
4. Use the pre-captured local baseline screenshot/PDF only to explain the product UI flow. State clearly that it is a pre-run local capture, not a live response.
5. If the live narrator has accidentally been enabled or fails, use **Generate deterministic brief**. It is the normal product path and should be framed as the intended fallback, not a degraded demo.

## Claims to make and claims to avoid

| Make this claim | Do not make this claim |
| --- | --- |
| "P10/P50/P90 are model-based planning ranges with historic evidence surfaced in Trust Center." | "The model guarantees a confidence interval or future revenue." |
| "The draw share is a simulated guardrail diagnostic." | "There is an X% probability the client will hit ROAS." |
| "The optimizer prioritizes constrained tests under model support and business guardrails." | "The optimizer discovers causal incremental lift or should autonomously spend the budget." |
| "The deterministic brief is generated from sealed evidence." | "The LLM predicts revenue, determines approval, or explains causal drivers." |
| "The 60-day gate fails closed on weak calibration evidence." | "Weak validation is harmless because the overall P50 looks plausible." |
| "This repository implements a single-user local prototype with a production roadmap." | "Multi-tenancy, authentication, cloud operations, or live integrations are already shipped." |

## Likely judge questions and concise answers

### Why should we trust the uncertainty ranges?

We should treat them as model-based planning ranges, not guarantees. The UI exposes the persisted rolling-origin coverage evidence and fails closed where that evidence is weak. The exact calibration method and artifact provenance are recorded with the release rather than invented in presentation language.

### Why is the 60-day decision posture `revise_or_test`?

The persisted six-fold evidence is not strong enough for approval at that horizon. We keep the forecast viewable because it is useful for planning, but the decision control requires a controlled test rather than allowing a confident approval claim.

### Is the "simulated guardrail draw share" a calibrated probability?

No. It is the share of deterministic model paths above the selected ROAS guardrail. It is useful as a scenario diagnostic, but the UI explicitly distinguishes it from approval-calibrated probability.

### Does the allocation recommendation prove that moving budget will cause more revenue?

No. It is a constrained, model-based test-priority plan. It respects support caps and guardrails, exposes relaxations, and recommends validation. It does not claim incrementality or causal lift.

### What prevents a hallucinating LLM from changing the recommendation?

The model and optimizer run before narration. The default brief is deterministic. Optional live narration is localhost-only, explicit opt-in, receives a sealed evidence packet, treats all packet fields as data, must cite approved evidence IDs, and cannot modify forecast, interval, allocation, or decision fields. Invalid or unavailable narration returns the deterministic brief rather than unverified text.

### How does this survive automated evaluation without internet?

The protected path is `./run.sh DATA_DIR MODEL_PATH OUTPUT_PATH`. It uses only local files, pinned NumPy/Pandas dependencies, the sealed pickle, and `src/`; it performs inference only and writes `predictions.csv`. The UI, training, and optional narration are isolated outside that path.

### What happens when source data is messy or a campaign type is new?

Malformed required data, blank IDs, duplicate campaign-days, and invalid numeric values fail closed. Meta taxonomy overrides require a reviewed Meta-only mapping. For an unsupported categorical value at inference, the forecast takes a documented conservative fallback rather than fabricating a direct-model one-hot feature.

### What would you build next for a NetElixir pilot?

Keep the protected runner unchanged. Add reviewed client attribution/semantic integrations, authenticated tenant isolation, a governed model registry and monitoring workflow, and controlled experiment measurement to validate budget recommendations in real client accounts.

## Final rehearsal checklist

- [ ] Planner starts at `127.0.0.1:4173` without `--enable-live-llm`.
- [ ] Baseline, Data Health, Trust Center, and product-boundary panels are visible.
- [ ] One modest budget change produces a new simulated scenario.
- [ ] **Prioritize test allocation** returns a campaign-level pinned plan.
- [ ] **Generate deterministic brief** works with no network or key.
- [ ] The optional live-narration control is visibly disabled in the normal demo.
- [ ] Changing an input visibly marks the scenario stale.
- [ ] Terminal fallback command and local backup capture are ready.
- [ ] Presenters know not to call draw share a probability, allocation causal, or the LLM a forecasting engine.
