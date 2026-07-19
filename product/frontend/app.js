const money = new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', maximumFractionDigits: 0});
const number = new Intl.NumberFormat('en-US', {maximumFractionDigits: 2});
let baselineBudgets = {};
let latestScenario = null;
let latestScenarioInputFingerprint = null;
let scenarioIsStale = true;
let baselineOverall = null;
let latestEvidence = null;
let latestHealth = null;

const byId = (id) => document.getElementById(id);
const setNotice = (text, kind='ok') => { const node = byId('notice'); node.textContent = text; node.className = `notice ${kind}`; };
const formatMoney = (value) => money.format(Number(value || 0));
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[character]));
const pct = (value) => `${Math.round(Number(value || 0) * 100)}%`;
const signedPp = (delta) => `${delta > 0 ? '+' : ''}${Math.round(delta * 100)} pp`;

function inputFingerprint() {
  return JSON.stringify({
    horizon_days: byId('horizon').value,
    target_roas: byId('target').value,
    google_budget: byId('google').value,
    meta_budget: byId('meta').value,
    microsoft_budget: byId('microsoft').value,
  });
}

function copyScenario(scenario) {
  return {
    horizon_days: Number(scenario.horizon_days),
    target_roas: Number(scenario.target_roas),
    campaign_budgets: {...(scenario.campaign_budgets || {})},
  };
}

function renderScenarioState() {
  const node = byId('scenario-state');
  const save = byId('save');
  const brief = byId('decision-brief');
  if (!latestScenario) {
    node.textContent = 'No forecast scenario is ready to save or brief.';
    node.className = 'delta-banner warn';
    save.disabled = true;
    brief.disabled = true;
    return;
  }
  if (scenarioIsStale) {
    node.textContent = 'Inputs changed after the displayed forecast. Simulate or recommend again before saving or generating a brief.';
    node.className = 'delta-banner warn';
    save.disabled = true;
    brief.disabled = true;
    return;
  }
  const campaignCount = Object.keys(latestScenario.campaign_budgets || {}).length;
  node.textContent = `Current scenario is pinned to the exact forecasted campaign plan (${campaignCount} campaign budgets).`;
  node.className = 'delta-banner ok';
  save.disabled = false;
  brief.disabled = false;
}

function markScenarioStale() {
  if (!latestScenario || scenarioIsStale || inputFingerprint() === latestScenarioInputFingerprint) return;
  scenarioIsStale = true;
  renderScenarioState();
  setNotice('Inputs changed after the displayed forecast. Re-run the scenario before saving or generating a brief.', 'warn');
}

function setCurrentScenario(scenario, requestFingerprint, initializing) {
  latestScenario = copyScenario(scenario);
  latestScenarioInputFingerprint = initializing ? inputFingerprint() : requestFingerprint;
  scenarioIsStale = !initializing && inputFingerprint() !== requestFingerprint;
  renderScenarioState();
}

function requireCurrentScenario(action) {
  if (!latestScenario || scenarioIsStale) {
    setNotice(`Re-run the scenario before ${action}; displayed outputs no longer match the current inputs.`, 'warn');
    return false;
  }
  return true;
}

function renderHealth(health) {
  latestHealth = health;
  const assumption = health.meta_revenue_assumption || (health.assumptions && health.assumptions[0]) || '';
  const planningDefaults = health.planning_defaults || [];
  const dataWarnings = (health.warnings || []).filter((item) => !planningDefaults.includes(item));
  byId('health').innerHTML = `
    <div class="assumption-banner"><strong>Meta revenue assumption:</strong> ${escapeHtml(assumption)}</div>
    <div class="health-row"><span>Source records</span><strong>${number.format(health.rows)}</strong></div>
    <div class="health-row"><span>Campaigns</span><strong>${number.format(health.campaigns)}</strong></div>
    <div class="health-row"><span>Coverage</span><strong>${health.date_start} - ${health.date_end}</strong></div>
    <div class="health-row"><span>Unclassified Meta rows</span><strong>${number.format(health.meta_taxonomy_unknown_rows || 0)}</strong></div>
    <div class="health-row"><span>Gate status</span><strong class="badge ${health.status === 'warning' ? 'warn' : ''}">${health.status}</strong></div>
    ${planningDefaults.length ? `<h3>Planning defaults</h3><ul class="warning-list">${planningDefaults.map((item) => `<li>${escapeHtml(item)}. Baseline media plans infer a recent-spend default where a configured budget is absent.</li>`).join('')}</ul>` : ''}
    ${dataWarnings.length ? `<ul class="warning-list">${dataWarnings.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}`;
}

function renderTrust(trust) {
  if (trust.status !== 'available') {
    byId('trust').innerHTML = `<div class="trust-summary"><h3>Rolling-origin backtest</h3><p><strong>Not applicable for this planner session.</strong> ${escapeHtml(trust.reason || 'No matching persisted evaluation report is available.')} No evaluation is run while the server is handling a request.</p></div>`;
    return;
  }
  const baseline = Object.fromEntries((trust.baseline_horizons || []).map((item) => [item.horizon_days, item]));
  const lines = trust.horizons.map((item) => { const prior = baseline[item.horizon_days]; const comparison = prior ? ` versus ${Math.round(prior.revenue_wape * 100)}% seasonal/statistical baseline` : ''; const calibration = item.median_calibration_samples ? ` Temporal holdout residuals: median ${number.format(item.median_calibration_samples)} samples.` : ''; const nominal = Math.round((item.nominal_interval_coverage || 0.8) * 100); const caution = item.folds < 5 ? ' Small fold count: treat coverage as directional, not a guarantee.' : ''; return `<p><strong>${item.horizon_days} days:</strong> ${Math.round(item.revenue_interval_coverage * 100)}% observed coverage for a nominal ${nominal}% P10-P90 interval across ${item.folds} historical folds; median WAPE ${Math.round(item.revenue_wape * 100)}%${comparison}.${calibration}${caution}</p>`; }).join('');
  byId('trust').innerHTML = `<div class="trust-summary"><h3>Rolling-origin backtest</h3>${lines}</div>`;
}

function renderChannels(channels) {
  byId('channels').innerHTML = `<table><thead><tr><th>Channel</th><th class="number">Budget</th><th class="number">Revenue P50</th><th class="number">ROAS P50</th><th class="number">Guardrail</th></tr></thead><tbody>${channels.map((row) => `<tr><td>${row.channel}</td><td class="number">${formatMoney(row.planned_budget)}</td><td class="number">${formatMoney(row.predicted_revenue_p50)}</td><td class="number">${number.format(row.predicted_roas_p50)}</td><td class="number">${Math.round(row.probability_roas_above_target * 100)}%</td></tr>`).join('')}</tbody></table>`;
}

function renderCampaignTypes(types) {
  byId('campaign-types').innerHTML = `<table><thead><tr><th>Channel / type</th><th class="number">Budget</th><th class="number">Revenue P50</th><th class="number">P10–P90</th><th class="number">ROAS P50</th></tr></thead><tbody>${types.map((row) => `<tr><td><strong>${escapeHtml(row.channel)}</strong><br><span class="table-muted">${escapeHtml(row.campaign_type)}</span></td><td class="number">${formatMoney(row.planned_budget)}</td><td class="number">${formatMoney(row.predicted_revenue_p50)}</td><td class="number">${formatMoney(row.predicted_revenue_p10)} – ${formatMoney(row.predicted_revenue_p90)}</td><td class="number">${number.format(row.predicted_roas_p50)}</td></tr>`).join('')}</tbody></table>`;
}

function renderCampaigns(campaigns) {
  const priority = [...campaigns].sort((left, right) => Number(right.predicted_revenue_p50) - Number(left.predicted_revenue_p50)).slice(0, 10);
  byId('campaigns').innerHTML = `<table><thead><tr><th>Campaign</th><th>Type</th><th class="number">Budget</th><th class="number">Revenue P10–P90</th><th class="number">ROAS P50</th><th>Risk flags</th></tr></thead><tbody>${priority.map((row) => `<tr><td><strong>${escapeHtml(row.campaign_name)}</strong><br><span class="table-muted">${escapeHtml(row.channel)}</span></td><td>${escapeHtml(row.campaign_type)}</td><td class="number">${formatMoney(row.planned_budget)}</td><td class="number">${formatMoney(row.predicted_revenue_p10)} – ${formatMoney(row.predicted_revenue_p90)}</td><td class="number">${number.format(row.predicted_roas_p50)}</td><td>${escapeHtml(row.quality_flags || 'none')}</td></tr>`).join('')}</tbody></table>`;
}

function renderAllocation(optimization, scenario) {
  const container = byId('allocation');
  if (!optimization) {
    container.innerHTML = `<p class="allocation-empty">Run <strong>Recommend allocation</strong> to allocate the entered total budget under campaign support caps and channel constraints.</p>`;
    return;
  }
  const entries = Object.entries(optimization.campaign_budgets || {}).sort(([, left], [, right]) => Number(right) - Number(left)).slice(0, 8);
  const target = optimization.target_roas;
  const achieved = optimization.achieved_roas_p50;
  const relaxed = optimization.target_constraint_status === 'marginal_target_relaxed';
  const targetLine = target == null ? 'No ROAS guardrail was requested.' : `Target ${number.format(target)}; blended P50 ROAS ${number.format(achieved)} (${number.format(optimization.target_gap_p50)} gap).`;
  const campaignCount = Object.keys((scenario || {}).campaign_budgets || {}).length;
  container.innerHTML = `<p class="allocation-status"><strong>${escapeHtml(optimization.status)}</strong> <span class="badge ${relaxed ? 'warn' : ''}">${escapeHtml((optimization.target_constraint_status || 'not_requested').replaceAll('_', ' '))}</span><br>${escapeHtml(targetLine)}<br>${escapeHtml(optimization.explanation)}</p><p class="panel-note">The full ${number.format(campaignCount)}-campaign allocation is the exact scenario pinned for briefs and ledger saves; it is not reconstructed from rounded channel totals.</p><ol class="allocation-list">${entries.map(([campaign, budget]) => `<li><span>${escapeHtml(campaign)}</span><strong>${formatMoney(budget)}</strong></li>`).join('')}</ol>`;
}

function renderUncertainty(overall, initializing) {
  if (initializing) baselineOverall = {...overall};
  const baseline = baselineOverall || overall;
  const rows = [{label: 'Baseline', value: baseline}, {label: 'Current scenario', value: overall}];
  const width = 760;
  const maximum = Math.max(...rows.map(({value}) => Number(value.predicted_revenue_p90 || 0)), 1);
  const x = (value) => 34 + Math.max(0, Number(value || 0)) / maximum * (width - 54);
  const line = ({label, value}, index) => {
    const y = 42 + index * 66;
    const p10 = x(value.predicted_revenue_p10), p50 = x(value.predicted_revenue_p50), p90 = x(value.predicted_revenue_p90);
    return `<text x="0" y="${y + 4}" class="chart-label">${label}</text><line x1="${p10}" x2="${p90}" y1="${y}" y2="${y}" class="chart-range"/><line x1="${p10}" x2="${p10}" y1="${y - 7}" y2="${y + 7}" class="chart-cap"/><line x1="${p90}" x2="${p90}" y1="${y - 7}" y2="${y + 7}" class="chart-cap"/><circle cx="${p50}" cy="${y}" r="6" class="chart-median"/><text x="${Math.min(p50 + 10, width - 84)}" y="${y + 4}" class="chart-value">${formatMoney(value.predicted_revenue_p50)}</text>`;
  };
  byId('uncertainty-chart').innerHTML = `<svg viewBox="0 0 ${width} 126" role="img" aria-label="Revenue uncertainty ranges for baseline and current scenario">${rows.map(line).join('')}<text x="34" y="120" class="chart-note">Range = P10–P90; dot = P50. This is a conditional forecast, not a guarantee.</text></svg>`;
}

function renderGuardrailDelta(overall, initializing) {
  if (initializing || !baselineOverall) {
    byId('guardrail-delta').textContent = 'Baseline loaded. Change a channel budget and Simulate plan to see guardrail probability move.';
    byId('guardrail-delta').className = 'delta-banner';
    return;
  }
  const before = Number(baselineOverall.probability_roas_above_target || 0);
  const after = Number(overall.probability_roas_above_target || 0);
  const delta = after - before;
  const revenueDelta = Number(overall.predicted_revenue_p50 || 0) - Number(baselineOverall.predicted_revenue_p50 || 0);
  const tone = delta < -0.01 ? 'warn' : (delta > 0.01 ? 'ok' : '');
  byId('guardrail-delta').className = `delta-banner ${tone}`;
  byId('guardrail-delta').innerHTML = `<strong>Budget shock impact:</strong> Guardrail probability ${pct(before)} → ${pct(after)} (${signedPp(delta)}). Median revenue ${formatMoney(baselineOverall.predicted_revenue_p50)} → ${formatMoney(overall.predicted_revenue_p50)} (${revenueDelta >= 0 ? '+' : ''}${formatMoney(revenueDelta)}).`;
}

function renderEvidence(evidence) {
  latestEvidence = evidence;
  byId('headline').textContent = evidence.headline;
  const calibration = evidence.decision_gates?.calibration;
  const calibrationLine = calibration ? `<h3>Calibration gate</h3><p><strong>${escapeHtml(calibration.status.replaceAll('_', ' '))}</strong>${calibration.reasons?.length ? `: ${calibration.reasons.map(escapeHtml).join(' ')}` : '.'}</p>` : '';
  byId('evidence').innerHTML = `<p>${evidence.causal_status.replaceAll('_', ' ')}</p>${calibrationLine}<h3>Top modeled contributors</h3><ul>${evidence.drivers.map((item) => `<li><strong>${item.channel}</strong>: ${formatMoney(item.expected_revenue)} expected revenue at ${number.format(item.expected_roas)} ROAS.</li>`).join('')}</ul><h3>Risks</h3><ul>${evidence.risks.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul><h3>Recommended validation</h3><p>${escapeHtml(evidence.recommended_validation)}</p>`;
}

function renderDecisionBrief(brief, mode, message) {
  const container = byId('decision-brief-output');
  const section = (title, items) => items && items.length ? `<h3>${title}</h3><ul>${items.map((item) => `<li>${escapeHtml(item.text)} <span class="citation">${(item.evidence_ids || []).map(escapeHtml).join(', ')}</span></li>`).join('')}</ul>` : '';
  container.innerHTML = `<div class="decision-brief"><p class="eyebrow">${mode === 'openai_grounded_narrative' ? 'OPTIONAL LIVE NARRATIVE · EXPLICIT OPT-IN' : 'DETERMINISTIC DECISION BRIEF · NO LIVE LLM'}</p><h3 class="decision-brief-headline">${escapeHtml(brief.headline)}</h3><p class="causal-boundary">${escapeHtml((brief.causal_status || 'observational_association').replaceAll('_', ' '))}; a narrative cannot change forecast numbers.</p><p class="decision-brief-message">${escapeHtml(message || '')}</p>${section('Facts', brief.facts)}${section('Assumptions', brief.assumptions)}${section('Recommended validation', brief.recommendations)}${section('Limitations', brief.limitations)}</div>`;
}

function render(response, initializing=false, requestFingerprint=null) {
  const overall = response.overall[0];
  byId('revenue').textContent = formatMoney(overall.predicted_revenue_p50);
  byId('revenue-range').textContent = `P10-P90: ${formatMoney(overall.predicted_revenue_p10)} - ${formatMoney(overall.predicted_revenue_p90)}`;
  byId('roas').textContent = number.format(overall.predicted_roas_p50);
  byId('roas-range').textContent = `P10-P90: ${number.format(overall.predicted_roas_p10)} - ${number.format(overall.predicted_roas_p90)}`;
  byId('probability').textContent = pct(overall.probability_roas_above_target);
  byId('risk').textContent = `Risk score: ${number.format(overall.risk_score)} / 100`;
  byId('decision').textContent = response.evidence.decision.replaceAll('_', ' ');
  renderChannels(response.channels); renderCampaignTypes(response.campaign_types); renderCampaigns(response.campaigns); renderAllocation(response.optimization, response.scenario); renderEvidence(response.evidence); renderHealth(response.data_health); renderUncertainty(overall, initializing); renderGuardrailDelta(overall, initializing);
  if (initializing) {
    baselineBudgets = Object.fromEntries(response.channels.map((row) => [row.channel, row.planned_budget]));
    byId('google').value = Math.round(baselineBudgets.SEARCH + (baselineBudgets.SHOPPING || 0) + (baselineBudgets.PERFORMANCE_MAX || 0) + (baselineBudgets.DEMAND_GEN || 0) + (baselineBudgets.DISPLAY || 0) + (baselineBudgets.VIDEO || 0));
    byId('meta').value = Math.round(baselineBudgets.META_ADS || 0);
    byId('microsoft').value = Math.round(baselineBudgets.MICROSOFT_ADS || 0);
  }
  if (response.scenario) setCurrentScenario(response.scenario, requestFingerprint, initializing);
  const stale = Boolean(response.scenario) && scenarioIsStale;
  const metaNote = response.data_health.meta_revenue_assumption ? ` Meta semantics: ${response.data_health.meta_revenue_semantics_status || 'unknown'}.` : '';
  const planningDefaults = response.data_health.planning_defaults || [];
  const dataWarnings = (response.data_health.warnings || []).filter((item) => !planningDefaults.includes(item));
  const healthNotice = dataWarnings.length
    ? 'Forecast generated with data-quality warnings. Review Trust Center before approving.'
    : (planningDefaults.length
      ? 'Forecast generated with planning-default assumptions. Review Trust Center before approving.'
      : 'Forecast generated from the current validated data snapshot.');
  setNotice(
    stale
      ? 'Inputs changed while this forecast was running. The result is displayed for comparison but cannot be saved or briefed until rerun.'
      : (healthNotice + metaNote),
    stale || dataWarnings.length || planningDefaults.length ? 'warn' : 'ok',
  );
}

async function requestScenario(initializing=false, optimize=false) {
  const payload = {horizon_days: Number(byId('horizon').value), target_roas: Number(byId('target').value)};
  if (!initializing) {
    const googleTotal = Number(byId('google').value || 0);
    const baselineGoogle = Object.entries(baselineBudgets).filter(([channel]) => !['META_ADS', 'MICROSOFT_ADS'].includes(channel));
    const baselineGoogleTotal = baselineGoogle.reduce((sum, [, value]) => sum + value, 0);
    for (const [channel, amount] of baselineGoogle) payload.channel_budgets = {...payload.channel_budgets, [channel]: baselineGoogleTotal ? googleTotal * amount / baselineGoogleTotal : 0};
    payload.channel_budgets = {...payload.channel_budgets, META_ADS: Number(byId('meta').value || 0), MICROSOFT_ADS: Number(byId('microsoft').value || 0)};
  }
  const requestFingerprint = inputFingerprint();
  const optimizationPayload = optimize ? {
    horizon_days: payload.horizon_days,
    target_roas: payload.target_roas,
    total_budget: Object.values(payload.channel_budgets || {}).reduce((sum, value) => sum + value, 0),
  } : payload;
  setNotice(optimize ? 'Optimizing allocation under response-curve constraints…' : 'Running probabilistic scenario simulation…', 'loading');
  try {
    const response = await fetch(initializing ? '/api/baseline' : (optimize ? '/api/optimize' : '/api/scenario'), {method: initializing ? 'GET' : 'POST', headers: {'Content-Type': 'application/json'}, body: initializing ? undefined : JSON.stringify(optimizationPayload)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || 'Scenario request failed');
    render(body, initializing, requestFingerprint);
  } catch (error) { setNotice(error.message, 'error'); }
}

async function loadTrust() {
  try {
    const response = await fetch('/api/trust');
    if (!response.ok) throw new Error('Unable to load backtest report');
    renderTrust(await response.json());
  } catch (error) { byId('trust').innerHTML = `<div class="trust-summary"><p>Backtest report unavailable: ${error.message}</p></div>`; }
}

async function saveDecision() {
  if (!requireCurrentScenario('saving a decision')) return;
  try {
    const response = await fetch('/api/decisions', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action:'draft', scenario:copyScenario(latestScenario)})});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || 'Unable to save decision');
    setNotice(`Decision saved to ledger as #${body.ledger.id} using forecast ${body.summary.forecast_id}.`, 'ok');
  } catch (error) { setNotice(error.message, 'error'); }
}

async function requestDecisionBrief() {
  if (!requireCurrentScenario('generating a decision brief')) return;
  const button = byId('decision-brief');
  button.disabled = true;
  setNotice('Generating deterministic decision brief from sealed forecast evidence…', 'loading');
  try {
    // The visible demo action always selects the local, deterministic path.
    // A live narrator requires a separate explicit API request with boolean true.
    const response = await fetch('/api/evidence', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({scenario:copyScenario(latestScenario), prefer_live_llm: false})});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || 'Unable to generate decision brief');
    const brief = body.brief || {
      decision: (body.deterministic_evidence || latestEvidence || {}).decision,
      causal_status: (body.deterministic_evidence || latestEvidence || {}).causal_status,
      headline: (body.deterministic_evidence || latestEvidence || {}).headline,
      facts: ((body.deterministic_evidence || latestEvidence || {}).facts || []).map((text) => ({text, evidence_ids: ['forecast']})),
      assumptions: [{text: (latestHealth && latestHealth.meta_revenue_assumption) || 'Platform attribution treated as truth.', evidence_ids: ['assumption']}],
      recommendations: [{text: (body.deterministic_evidence || latestEvidence || {}).recommended_validation || '', evidence_ids: ['validation']}],
      limitations: ((body.deterministic_evidence || latestEvidence || {}).risks || []).map((text) => ({text, evidence_ids: ['risk']})),
    };
    renderDecisionBrief(brief, body.mode, body.message);
    setNotice(body.message || 'Deterministic decision brief ready. Forecast numbers were not altered.', 'ok');
  } catch (error) { setNotice(error.message, 'error'); }
  finally { button.disabled = false; }
}

byId('forecast').addEventListener('click', () => requestScenario(false));
byId('optimize').addEventListener('click', () => requestScenario(false, true));
byId('save').addEventListener('click', saveDecision);
byId('decision-brief').addEventListener('click', requestDecisionBrief);
for (const id of ['horizon', 'target', 'google', 'meta', 'microsoft']) {
  byId(id).addEventListener('input', markScenarioStale);
  byId(id).addEventListener('change', markScenarioStale);
}
requestScenario(true);
loadTrust();
