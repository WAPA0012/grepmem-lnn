/**
 * Tier A — constructed multi-turn scenario (the proof of concept).
 *
 * Each scenario is a mini-session:
 *   - A few "focus-establishing" turns that set the agent's context (e.g. a
 *     debugging thread: "auth service throwing ConnectionRefused").
 *   - A corpus of memories, including:
 *       * the ANSWER memory (belongs to the established focus)
 *       * DISTRACTOR memories (lexically match the probe query just as well,
 *         but belong to a DIFFERENT focus — e.g. an unrelated timeout config).
 *   - A PROBE query that is intentionally AMBIGUOUS — it matches both the
 *     answer and a distractor on raw lexical terms, so flat retrieval is a
 *     coin flip. The state-aware scheduler, having accumulated the focus,
 *     boosts the focus-relevant term types and should rank the answer higher.
 *
 * Metric: does the answer rank in top-1 / top-3? Reported per strategy.
 *
 * Strategies compared (the ablation ladder):
 *   flat-rescore   — re-simulated match with all boosts = 1.0 (fair baseline)
 *   scalar-focus   — focus distribution + readout, but FIXED τ (no coupling)
 *   lnn-coupled    — full coupled CfC (confidence modulates focus τ)
 *
 * NOTE on fairness: all three use the SAME re-simulated match code, so the
 * only difference is the boost vector. flat-rescore = boosts all 1.0.
 * scalar-focus = boosts from a focus distribution that does NOT use the
 * confidence-modulated τ (it uses a fixed τ, decoupled). lnn-coupled = the
 * real thing. This isolates the coupling contribution cleanly.
 */
import { createState, step, readoutBoosts, snapshot, BOOST_TABLE } from '../lib/state-scheduler.js';
import { flatRescoreMatch, stateAwareMatch, stateAwareRawScore, combinedStateScore } from '../lib/match-rescore.js';
import { extractFocusSignal, dominantFocus, FOCUS_MODES } from '../lib/focus-classifier.js';
import { loadGrepmem, loadGrepmemGrep } from '../lib/grepmem-shim.js';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const N_SCENARIOS = parseInt(process.env.SCENARIOS || '300', 10);

// ─── Scenario templates ────────────────────────────────────────────────────
// Each template establishes a focus, then probes with an ambiguous query that
// matches both an in-focus answer and an out-of-focus distractor.

// DESIGN PRINCIPLE: answer and distractor must have SYMMETRIC lexical surfaces
// (same number of triggers, same probe-term density) so flat retrieval is a
// genuine coin flip. The answer carries a term of the ESTABLISHED focus's
// favored type; the distractor carries a term of a DIFFERENT focus. The probe
// is ambiguous on shared words. Only the accumulated focus state can break
// the tie. We rank by RAW score (the normalized match saturates at 1.0 and
// hides the signal).
const TEMPLATES = [
  {
    name: 'auth-debug-vs-cache-config',
    // Establishes procedure-debug focus (error/fix vocabulary).
    focusTurns: [
      'auth service throwing ConnectionRefused error',
      'how to fix the auth login failure',
      'the auth timeout error keeps crashing',
    ],
    probe: 'timeout',                          // single shared term — pure tie
    // Answer: debug-favored (carries error word "failure" = synonym-ish cue).
    // Distractor: config-favored (carries config cue "pool").
    // Both contain "timeout" identically; tie broken by focus boost.
    answer: {
      summary: 'auth timeout after login failure error',
      detail: 'timeout set during the failure. Error handling.',
      triggers: ['timeout', 'failure', 'error'],
    },
    distractor: {
      summary: 'cache timeout pool config',
      detail: 'timeout set in the pool. Config value.',
      triggers: ['timeout', 'pool', 'config'],
    },
    expectedFocus: 1, // procedure-debug
  },
  {
    name: 'db-config-vs-log-debug',
    focusTurns: [
      'set the MAX_CONNECTIONS database config',
      'what is the pool size parameter',
      'configure the db connection limit setting',
    ],
    probe: 'limit',
    answer: {
      summary: 'db limit MAX_CONNECTIONS config parameter',
      detail: 'limit is a config parameter. db setting.',
      triggers: ['limit', 'config', 'parameter'],
    },
    distractor: {
      summary: 'log limit error failure throttle',
      detail: 'limit on error logs. failure throttle.',
      triggers: ['limit', 'error', 'failure'],
    },
    expectedFocus: 2, // config-param
  },
  {
    name: 'deploy-exact-vs-test-exact',
    focusTurns: [
      'connect to production at 10.0.0.5 port 6379',
      'what is the 10.0.0.5 server config',
      'the 10.0.0.5 host is down',
    ],
    probe: 'port',
    answer: {
      summary: 'prod 10.0.0.5 port 6379 host',
      detail: '10.0.0.5 port. host production.',
      triggers: ['10.0.0.5', 'port', 'host'],
    },
    distractor: {
      summary: 'test 10.0.0.9 port 6379 host',
      detail: '10.0.0.9 port. host staging.',
      triggers: ['10.0.0.9', 'port', 'host'],
    },
    expectedFocus: 0, // exact-entity
  },
];

// ─── Helpers ────────────────────────────────────────────────────────────────

function nodeFromTemplate(t) {
  return {
    type: 'knowledge',
    summary: t.summary,
    detail: t.detail,
    triggers: t.triggers,
    tags: t.triggers,
    baseSalience: 0.5,
    accessCount: 0,
    lastAccess: new Date().toISOString().slice(0, 10),
  };
}

// Rank by RAW score (normalized match saturates at 1.0 and hides the signal).
// Three variants:
//   flat:       raw term-match only, no boosts, no focus bonus (baseline)
//   coupled:    term-type boosts + lexical-focus bonus, from the coupled CfC
//   scalar:     same but with the decoupled (fixed-τ) scheduler — ablation
function rankFlat(answerNode, distractorNode, terms) {
  const fb = new Array(9).fill(1.0);
  const aRaw = stateAwareRawScore(answerNode, terms, fb);
  const dRaw = stateAwareRawScore(distractorNode, terms, fb);
  return { aRaw, dRaw, picksAnswer: aRaw > dRaw };
}
function rankStateful(answerNode, distractorNode, terms, boosts, state) {
  const aRaw = combinedStateScore(answerNode, terms, boosts, state);
  const dRaw = combinedStateScore(distractorNode, terms, boosts, state);
  return { aRaw, dRaw, picksAnswer: aRaw > dRaw };
}

// A genuine scalar-decay focus scheduler: focus logits AND the lexical-focus
// bag decay with a FIXED τ (no confidence coupling). Readout uses the same
// boost table but gated by a FIXED confidence (constant), and the lexical
// bonus uses the fixed gate. This removes the COUPLING while keeping every
// other mechanism identical — so lnn-coupled vs scalar-focus isolates the
// coupling contribution fairly.
function makeScalarState() {
  return {
    u: [0, 0, 0, 0],
    fixedTau: 2.0,        // fixed, NOT confidence-modulated
    fixedConf: 0.5,       // fixed gate, NOT state-driven
    inputGain: 2.5,
    focusLex: new Map(),  // lexical-focus bag, fixed-τ decay
    tauLex: 2.5,
    lexGain: 1.0,
    // expose conf/confEpsilon so lexicalFocusBonus(state) reads fixedConf
    conf: 0.5,
    confEpsilon: 0.0,
    steps: 0,
  };
}
const SCALAR_STOP = new Set(['the','and','for','not','but','are','was','has','this','that','with','from','can','all','use','need','what','how','why','when','who','which','where','does','did','were','you','your','its']);
function scalarTokenize(query) {
  if (!query) return [];
  const words = new Set();
  const en = query.toLowerCase().match(/[a-z][a-z0-9_]{2,}/g) || [];
  for (const w of en) if (!SCALAR_STOP.has(w)) words.add(w);
  const cn = query.replace(/[^一-鿿]/g, '');
  for (let i = 0; i < cn.length - 1; i++) words.add(cn[i] + cn[i + 1]);
  return [...words];
}
function stepScalar(state, query, dt = 1) {
  const I = extractFocusSignal(query);
  for (let i = 0; i < 4; i++) {
    const target = I[i] * state.inputGain;
    state.u[i] = target + (state.u[i] - target) * Math.exp(-dt / state.fixedTau);
  }
  // lexical-focus bag, FIXED-τ decay (no confidence modulation)
  for (const [w, wt] of state.focusLex) {
    const d = wt * Math.exp(-dt / state.tauLex);
    if (d < 0.01) state.focusLex.delete(w); else state.focusLex.set(w, d);
  }
  for (const w of scalarTokenize(query)) state.focusLex.set(w, (state.focusLex.get(w) || 0) + 1);
  state.steps++;
}
function scalarBoosts(state) {
  // Same boost-table readout as the coupled scheduler, but gated by the FIXED
  // confidence (no state-driven gate). The focus distribution itself used a
  // FIXED τ during stepping (no confidence coupling). This is the ablation.
  const m = Math.max(...state.u);
  const e = state.u.map(x => Math.exp(x - m));
  const sum = e.reduce((a, b) => a + b, 0) || 1;
  const z = e.map(x => x / sum);
  const gate = state.fixedConf;
  const boosts = new Array(9).fill(1.0);
  for (let t = 0; t < 9; t++) {
    let raw = 0;
    for (let m2 = 0; m2 < 4; m2++) raw += z[m2] * BOOST_TABLE[m2][t];
    boosts[t] = 1 + (raw - 1) * gate;
  }
  return boosts;
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  // extractQueryTerms lives in grep.js, not memory-html.js. Note: its `learned`
  // term branch needs globalThis.__SYNONYM_LEARNER (set by MemoryEngine.init).
  // Tier A's synthetic queries don't rely on learned synonyms, so we accept
  // their absence here. Tier B (real engine) will have them.
  const { extractQueryTerms } = await loadGrepmemGrep();
  console.log('═══════════════════════════════════════════════════════════════════════');
  console.log('  Tier A — constructed multi-turn scenario (proof of concept)');
  console.log(`  ${N_SCENARIOS} scenarios × ${TEMPLATES.length} templates, stratified by focus type`);
  console.log('═══════════════════════════════════════════════════════════════════════\n');

  const counts = {
    'flat-rescore': { top1: 0, top3: 0, total: 0, byFocus: {} },
    'scalar-focus': { top1: 0, top3: 0, total: 0, byFocus: {} },
    'lnn-coupled': { top1: 0, top3: 0, total: 0, byFocus: {} },
  };
  const initFocus = (store, f) => { store.byFocus[f] ||= { top1: 0, top3: 0, total: 0 }; };

  for (let n = 0; n < N_SCENARIOS; n++) {
    const tpl = TEMPLATES[n % TEMPLATES.length];
    const answerNode = nodeFromTemplate(tpl.answer);
    const distractorNode = nodeFromTemplate(tpl.distractor);
    const terms = extractQueryTerms(tpl.probe);

    // --- flat-rescore: no boosts, no focus bonus (baseline) ---
    const flatR = rankFlat(answerNode, distractorNode, terms);

    // --- lnn-coupled: accumulate focus over the establishing turns ---
    const coupled = createState();
    for (const ft of tpl.focusTurns) step(coupled, ft);
    const coupledBoosts = readoutBoosts(coupled);
    const coupledR = rankStateful(answerNode, distractorNode, terms, coupledBoosts, coupled);

    // --- scalar-focus ablation: same turns, fixed τ (no coupling) ---
    const scalar = makeScalarState();
    for (const ft of tpl.focusTurns) stepScalar(scalar, ft);
    const scalarBoostsV = scalarBoosts(scalar);
    const scalarR = rankStateful(answerNode, distractorNode, terms, scalarBoostsV, scalar);

    for (const [name, R] of [['flat-rescore', flatR], ['scalar-focus', scalarR], ['lnn-coupled', coupledR]]) {
      counts[name].total++;
      counts[name].top1 += R.picksAnswer ? 1 : 0;
      counts[name].top3 += R.picksAnswer ? 1 : 0; // only 2 candidates, top3 == top1 here
      initFocus(counts[name], tpl.expectedFocus);
      counts[name].byFocus[tpl.expectedFocus].total++;
      counts[name].byFocus[tpl.expectedFocus].top1 += R.picksAnswer ? 1 : 0;
    }
  }

  // ── Report ──
  const focusName = (f) => FOCUS_MODES[f] || `focus-${f}`;
  console.log('  strategy'.padEnd(18) + 'picks-answer%'.padStart(16) + '  by focus type');
  console.log('  ' + '─'.repeat(70));
  for (const name of ['flat-rescore', 'scalar-focus', 'lnn-coupled']) {
    const c = counts[name];
    const pct = (c.top1 / c.total * 100).toFixed(1);
    const byFocus = Object.entries(c.byFocus).map(([f, v]) =>
      `${focusName(f)}:${(v.top1 / v.total * 100).toFixed(0)}%`).join('  ');
    console.log('  ' + name.padEnd(18) + (pct + '%').padStart(16) + '   ' + byFocus);
  }
  console.log('  ' + '─'.repeat(70));

  const flatPct = counts['flat-rescore'].top1 / counts['flat-rescore'].total * 100;
  const scalarPct = counts['scalar-focus'].top1 / counts['scalar-focus'].total * 100;
  const coupledPct = counts['lnn-coupled'].top1 / counts['lnn-coupled'].total * 100;
  console.log(`\n  Verdict:`);
  console.log(`    flat-rescore → lnn-coupled : ${(coupledPct - flatPct).toFixed(1)}pt  (target ≥ +10pt)`);
  console.log(`    scalar-focus → lnn-coupled : ${(coupledPct - scalarPct).toFixed(1)}pt  (>0 means coupling helps)`);
  if (coupledPct - flatPct < 10) {
    console.log(`    ⚠ below +10pt target — see README for honest interpretation`);
  }
  console.log('═══════════════════════════════════════════════════════════════════════');
}

main().catch(e => { console.error(e); process.exit(1); });
