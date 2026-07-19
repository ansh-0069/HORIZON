const money = new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', maximumFractionDigits: 0});
const number = new Intl.NumberFormat('en-US', {maximumFractionDigits: 2});
let baselineBudgets = {};
let latestScenario = {horizon_days: 60, target_roas: 4.0};
let baselineOverall = null;

const byId = (id) => document.getElementById(id);
const setNotice = (text, kind='ok') => { const node = byId('notice'); node.textContent = text; node.className = `notice ${kind}`; };
const formatMoney = (value) => money.format(Number(value || 0));
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[character]));

function renderHealth(health) {
  byId('health').innerHTML = `
    <div class="health-row"><span>Source records</span><strong>${number.format(health.rows)}</strong></div>
    <div class="health-row"><span>Campaigns</span><strong>${number.format(health.campaigns)}</strong></div>
    <div class="health-row"><span>Coverage</span><strong>${health.date_start} - ${health.date_end}</strong></div>
    <div class="health-row"><span>Unclassified Meta rows</span><strong>${number.format(health.meta_taxonomy_unknown_rows || 0)}</strong></div>
    <div class="health-row"><span>Gate status</span><strong class="badge ${health.status === 'warning' ? 'warn' : ''}">${health.status}</strong></div>
    ${health.warnings.length ? `<ul class="warning-list">${health.warnings.map((item) => `<li>${item}</li>`).join('')}</ul>` : ''}`;
}

function renderTrust(trust) {
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

function renderAllocation(optimization) {
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
  container.innerHTML = `<p class="allocation-status"><strong>${escapeHtml(optimization.status)}</strong> <span class="badge ${relaxed ? 'warn' : ''}">${escapeHtml((optimization.target_constraint_status || 'not_requested').replaceAll('_', ' '))}</span><br>${escapeHtml(targetLine)}<br>${escapeHtml(optimization.explanation)}</p><ol class="allocation-list">${entries.map(([campaign, budget]) => `<li><span>${escapeHtml(campaign)}</span><strong>${formatMoney(budget)}</strong></li>`).join('')}</ol>`;
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

function renderEvidence(evidence) {
  byId('headline').textContent = evidence.headline;
  byId('evidence').innerHTML = `<p>${evidence.causal_status.replaceAll('_', ' ')}</p><h3>Top modeled contributors</h3><ul>${evidence.drivers.map((item) => `<li><strong>${item.channel}</strong>: ${formatMoney(item.expected_revenue)} expected revenue at ${number.format(item.expected_roas)} ROAS.</li>`).join('')}</ul><h3>Risks</h3><ul>${evidence.risks.map((item) => `<li>${item}</li>`).join('')}</ul><h3>Recommended validation</h3><p>${evidence.recommended_validation}</p>`;
}

function renderAiBrief(response) {
  const container = byId('ai-evidence');
  if (response.mode !== 'openai_grounded_narrative') {
    container.innerHTML = `<p class="ai-fallback"><strong>AI brief unavailable.</strong> ${escapeHtml(response.message)}</p>`;
    return;
  }
  const brief = response.brief;
  const section = (title, items) => items.length ? `<h3>${title}</h3><ul>${items.map((item) => `<li>${escapeHtml(item.text)} <span class="citation">${item.evidence_ids.map(escapeHtml).join(', ')}</span></li>`).join('')}</ul>` : '';
  container.innerHTML = `<div class="ai-brief"><p class="eyebrow">AI NARRATIVE · CITED AND GUARDED</p><h3 class="ai-headline">${escapeHtml(brief.headline)}</h3><p class="causal-boundary">${escapeHtml(brief.causal_status.replaceAll('_', ' '))}; model output cannot change the forecast decision.</p>${section('Facts', brief.facts)}${section('Assumptions', brief.assumptions)}${section('Recommended validation', brief.recommendations)}${section('Limitations', brief.limitations)}</div>`;
}

function render(response, initializing=false) {
  const overall = response.overall[0];
  byId('revenue').textContent = formatMoney(overall.predicted_revenue_p50);
  byId('revenue-range').textContent = `P10-P90: ${formatMoney(overall.predicted_revenue_p10)} - ${formatMoney(overall.predicted_revenue_p90)}`;
  byId('roas').textContent = number.format(overall.predicted_roas_p50);
  byId('roas-range').textContent = `P10-P90: ${number.format(overall.predicted_roas_p10)} - ${number.format(overall.predicted_roas_p90)}`;
  byId('probability').textContent = `${Math.round(overall.probability_roas_above_target * 100)}%`;
  byId('risk').textContent = `Risk score: ${number.format(overall.risk_score)} / 100`;
  byId('decision').textContent = response.evidence.decision.replaceAll('_', ' ');
  renderChannels(response.channels); renderCampaignTypes(response.campaign_types); renderCampaigns(response.campaigns); renderAllocation(response.optimization); renderEvidence(response.evidence); renderHealth(response.data_health); renderUncertainty(overall, initializing);
  if (initializing) {
    baselineBudgets = Object.fromEntries(response.channels.map((row) => [row.channel, row.planned_budget]));
    byId('google').value = Math.round(baselineBudgets.SEARCH + (baselineBudgets.SHOPPING || 0) + (baselineBudgets.PERFORMANCE_MAX || 0) + (baselineBudgets.DEMAND_GEN || 0) + (baselineBudgets.DISPLAY || 0) + (baselineBudgets.VIDEO || 0));
    byId('meta').value = Math.round(baselineBudgets.META_ADS || 0);
    byId('microsoft').value = Math.round(baselineBudgets.MICROSOFT_ADS || 0);
  }
  setNotice(response.data_health.warnings.length ? 'Forecast generated with data-quality warnings. Review the Trust Center before approving a plan.' : 'Forecast generated from the current validated data snapshot.', response.data_health.warnings.length ? 'warn' : 'ok');
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
  latestScenario = payload;
  if (optimize) payload.total_budget = Object.values(payload.channel_budgets || {}).reduce((sum, value) => sum + value, 0);
  setNotice(optimize ? 'Optimizing allocation under response-curve constraints…' : 'Running probabilistic scenario simulation…', 'loading');
  try {
    const response = await fetch(initializing ? '/api/baseline' : (optimize ? '/api/optimize' : '/api/scenario'), {method: initializing ? 'GET' : 'POST', headers: {'Content-Type': 'application/json'}, body: initializing ? undefined : JSON.stringify(payload)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || 'Scenario request failed');
    render(body, initializing);
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
  try {
    const response = await fetch('/api/decisions', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action:'draft', scenario:latestScenario})});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || 'Unable to save decision');
    setNotice(`Decision saved to ledger as #${body.ledger.id} using forecast ${body.summary.forecast_id}.`, 'ok');
  } catch (error) { setNotice(error.message, 'error'); }
}

async function requestAiBrief() {
  const button = byId('ai-brief');
  button.disabled = true;
  setNotice('Generating a grounded narrative from the current forecast evidence…', 'loading');
  try {
    const response = await fetch('/api/evidence', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({scenario:latestScenario})});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || 'Unable to generate grounded brief');
    renderAiBrief(body);
    setNotice(body.mode === 'openai_grounded_narrative' ? 'Grounded AI brief generated. Numeric forecast values remain model-produced.' : 'AI service unavailable; deterministic evidence remains active.', body.mode === 'openai_grounded_narrative' ? 'ok' : 'warn');
  } catch (error) { setNotice(error.message, 'error'); }
  finally { button.disabled = false; }
}

byId('forecast').addEventListener('click', () => requestScenario(false));
byId('optimize').addEventListener('click', () => requestScenario(false, true));
byId('save').addEventListener('click', saveDecision);
byId('ai-brief').addEventListener('click', requestAiBrief);
requestScenario(true);
loadTrust();
