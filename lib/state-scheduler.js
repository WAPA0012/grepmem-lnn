/**
 * State-aware retrieval scheduler — the core of round 2.
 *
 * Maintains a coupled CfC session state z = (z_focus, z_conf) that evolves as
 * the agent receives queries, and a hand-designed readout that maps z to
 * per-term-type boost weights used to re-weight retrieval matches.
 *
 * ── Why this is "liquid" and load-bearing ──────────────────────────────────
 *
 * z_focus ∈ Δ⁴ is a soft distribution over 4 focus modes.
 * z_conf ∈ (0,1) is a confidence channel.
 *
 * They are COUPLED: z_conf modulates the time constant τ of z_focus. High
 * confidence → large τ → focus resists change (inertia, the agent is "sure").
 * Low confidence → small τ → focus re-orients fast toward new input. This
 * cross-channel coupling is the one thing a per-channel independent scalar
 * decay (the `scalar-focus` ablation) is structurally incapable of — it is
 * what makes the CfC genuinely liquid rather than "four scalar decays".
 *
 * ── The CfC step (closed-form, no ODE solver) ──────────────────────────────
 *
 * Each focus logit u_i relaxes toward its input-driven target u_i* with its
 * own (confidence-modulated) time constant:
 *
 *     u_i(t+dt) = u_i* + (u_i(t) − u_i*) · exp(−dt / τ_i(z_conf))     ... (1)
 *
 * where τ_i(z_conf) = τ_base · (ε + z_conf)   — the COUPLING: confidence
 * widens τ, slowing focus drift. z_focus = softmax(u) keeps it a simplex.
 *
 * z_conf itself relaxes toward an input-driven target (signal strength):
 *
 *     z_conf(t+dt) = c* + (z_conf(t) − c*) · exp(−dt / τ_conf)        ... (2)
 *
 * dt here is in "queries" (each query is one integration step, dt=1 by
 * default; a temporal-gap-aware variant multiplies dt by elapsed days for
 * LoCoMo). This mirrors the time-discretization in lnn-decay.js.
 *
 * ── Readout (hand-designed, training-free, auditable) ──────────────────────
 *
 * A fixed 4×9 table maps each focus mode to a boost for each of grepmem's 9
 * term types. The effective boost = Σ_mode z_focus[mode] · table[mode][type],
 * gated by confidence: boost_effective = 1 + (boost − 1) · gate(z_conf).
 * At gate=0 (no confidence) → boost=1 (neutral, behaves like flat retrieval).
 */

import { extractFocusSignal } from './focus-classifier.js';

const N_FOCUS = 4;
const N_TERMS = 9;

// Term types — must stay in sync with grepmem grep.js extractQueryTerms order
// and with match-rescore.js. Order matters for the boost table columns.
const TERM_TYPES = [
  'exact', 'en_word', 'en_ci', 'bigram', 'trigram', 'ip', 'number', 'synonym', 'learned',
];

/**
 * Fixed 4×9 boost table. Rows = focus modes, cols = term types.
 * A boost of 1.0 = no change. >1 = amplify that term type under this focus.
 * Designed by hand from what each focus mode "wants":
 *   exact-entity    → strong on exact/ip (precise lookups)
 *   procedure-debug → strong on synonym/learned/number (error vocab + ports/counts)
 *   config-param    → strong on en_ci/number (keys + numeric values)
 *   narrative-ctx   → strong on bigram/trigram/en_word (natural language)
 * These are the ONLY tunable readout knobs; kept moderate so a wrong focus
 * can't catastrophically amplify noise.
 */
const BOOST_TABLE = [
  // exact  en_word  en_ci  bigram trigram ip    number synonym learned
  [  1.6,   1.0,     1.0,   1.0,   1.0,    1.6,  1.1,   1.0,    1.0 ], // 0 exact-entity
  [  1.0,   1.1,     1.0,   1.0,   1.0,    1.0,  1.3,   1.4,    1.4 ], // 1 procedure-debug
  [  1.0,   1.1,     1.3,   1.0,   1.0,    1.0,  1.3,   1.0,    1.0 ], // 2 config-param
  [  1.0,   1.3,     1.0,   1.3,   1.3,    1.0,  1.0,   1.0,    1.0 ], // 3 narrative-context
];

/**
 * Create a fresh scheduler state. z_focus starts uniform (no commitment);
 * z_conf starts at 0 (no confidence → boosts are gated off → behaves flat).
 *
 * ALSO maintains a `focusLex` map: a CfC-decayed bag of words from recent
 * queries, weighted by recency. This is the channel that makes state-awareness
 * actually fire on English queries — term-type boosts (synonym/learned) only
 * trigger via the bilingual synonym map, but the lexical-focus channel boosts
 * ANY candidate word that recurred in the focus-establishing turns. Both
 * channels use CfC continuous-time decay (the load-bearing LNN property).
 */
export function createState(opts = {}) {
  return {
    // logits for the 4 focus modes (pre-softmax); uniform ⇒ z_focus uniform
    u: [0, 0, 0, 0],
    // confidence channel
    conf: opts.conf0 ?? 0,
    // CfC hyperparameters (the ONLY tuned knobs besides the boost table)
    tauBase: opts.tauBase ?? 2.0,   // focus relaxation time scale (in queries)
    tauConf: opts.tauConf ?? 3.0,   // confidence relaxation time scale
    confEpsilon: opts.confEpsilon ?? 0.1, // τ floor so confidence never fully freezes focus
    // how strongly input signal drives the focus logits
    inputGain: opts.inputGain ?? 2.5,
    // how strongly signal strength drives confidence target
    confGain: opts.confGain ?? 0.4,
    // LEXICAL FOCUS CHANNEL: word → CfC-decayed weight. Each query's words are
    // added; all weights decay by exp(-dt/tauLex) each step. This is what lets
    // the agent "remember the topic" across turns and boost candidates that
    // share that topic's vocabulary. tauLex is confidence-modulated too
    // (high confidence → focus words persist longer). THE COUPLING applies here
    // as well, not just to the focus distribution.
    focusLex: new Map(),
    tauLex: opts.tauLex ?? 2.5,
    lexGain: opts.lexGain ?? 1.0,   // how much each recurring focus word boosts a candidate
    // how many queries seen (for diagnostics)
    steps: 0,
  };
}

function softmax(u) {
  const m = Math.max(...u);
  const e = u.map(x => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map(x => x / s);
}

/**
 * Effective focus time constant for mode i, modulated by confidence.
 * THE COUPLING: higher confidence → larger τ → focus resists change.
 */
function focusTau(state, i) {
  return state.tauBase * (state.confEpsilon + state.conf);
}

/**
 * Advance the state by one query.
 * @param {object} state   scheduler state (mutated)
 * @param {string} query   the incoming query string
 * @param {object} [opts]
 * @param {number} [opts.dt=1]   integration step (use >1 for temporal gaps)
 * @returns {object} {z_focus: number[4], z_conf: number} post-update snapshot
 */
export function step(state, query, opts = {}) {
  const dt = opts.dt ?? 1;
  const I = extractFocusSignal(query);

  // ── Confidence update (Eq. 2) ──
  // Target confidence rises with total signal strength (how "decisive" the
  // query's lexical cues are). If the query carries no focus cues at all,
  // confidence decays toward 0 (agent becomes unsure → focus can drift).
  const signalMass = I.reduce((a, b) => a + b, 0);
  const confTarget = Math.min(1, signalMass * state.confGain);
  state.conf = confTarget + (state.conf - confTarget) * Math.exp(-dt / state.tauConf);

  // ── Focus update (Eq. 1) ──
  // Each logit's target = its input signal × gain. Relax toward it with the
  // confidence-modulated τ. High confidence → slow drift (inertia).
  for (let i = 0; i < N_FOCUS; i++) {
    const target = I[i] * state.inputGain;
    const tau = focusTau(state, i);
    state.u[i] = target + (state.u[i] - target) * Math.exp(-dt / tau);
  }

  // ── Lexical-focus channel update ──
  // Decay all existing focus-word weights by the confidence-modulated τ, then
  // add the current query's words. This is a CfC relaxation on a word-weight
  // map: words that recur across focus-establishing turns accumulate weight;
  // words from one-off queries decay away. Confidence modulates persistence
  // here too (THE COUPLING applies to this channel as well).
  const tauLexEff = state.tauLex * (state.confEpsilon + state.conf);
  for (const [w, wt] of state.focusLex) {
    const decayed = wt * Math.exp(-dt / tauLexEff);
    if (decayed < 0.01) state.focusLex.delete(w);
    else state.focusLex.set(w, decayed);
  }
  // Tokenize the query into focus words (lowercased, ≥3 char english / ≥2 char
  // chinese) and add them — recurring words build up.
  for (const w of tokenizeFocusWords(query)) {
    state.focusLex.set(w, (state.focusLex.get(w) || 0) + 1);
  }

  state.steps++;
  return { z_focus: softmax(state.u), z_conf: state.conf };
}

// Tokenizer for the lexical-focus channel. Conservative: english words ≥3
// chars, chinese bigrams. Filters stopwords so "the/value/is" don't dominate.
const FOCUS_STOPWORDS = new Set(['the', 'and', 'for', 'not', 'but', 'are', 'was', 'has', 'this', 'that', 'with', 'from', 'can', 'all', 'use', 'need', 'what', 'how', 'why', 'when', 'who', 'which', 'where', 'does', 'did', 'was', 'were', 'you', 'your', 'was', 'its']);
function tokenizeFocusWords(query) {
  if (!query) return [];
  const words = new Set();
  const en = query.toLowerCase().match(/[a-z][a-z0-9_]{2,}/g) || [];
  for (const w of en) if (!FOCUS_STOPWORDS.has(w)) words.add(w);
  const cn = query.replace(/[^一-鿿]/g, '');
  for (let i = 0; i < cn.length - 1; i++) words.add(cn[i] + cn[i + 1]);
  return [...words];
}

/**
 * Read out the per-term-type boost vector from the current state.
 * boost_effective[type] = 1 + (rawBoost[type] − 1) · gate(conf)
 * At conf=0 gate=0 → all boosts 1.0 (neutral = flat retrieval).
 * @returns {number[]} length-9 vector aligned with TERM_TYPES
 */
export function readoutBoosts(state) {
  const z = softmax(state.u);
  const gate = state.conf; // linear gate in [0,1]
  const boosts = new Array(N_TERMS).fill(1.0);
  for (let t = 0; t < N_TERMS; t++) {
    let raw = 0;
    for (let m = 0; m < N_FOCUS; m++) raw += z[m] * BOOST_TABLE[m][t];
    boosts[t] = 1 + (raw - 1) * gate;
  }
  return boosts;
}

/**
 * Scalar boost for a single term type (the hot path during re-scoring).
 */
export function boostFor(state, termType) {
  const idx = TERM_TYPES.indexOf(termType);
  if (idx < 0) return 1.0;
  return readoutBoosts(state)[idx];
}

/**
 * Lexical-focus bonus for a candidate node: how much the node's text overlaps
 * with the CfC-decayed focus-word bag. This is the channel that makes
 * state-awareness fire on English queries (term-type boosts alone rarely
 * trigger there). Returns an additive bonus ≥ 0.
 *
 * A node that shares many words with the recent focus-establishing turns
 * gets a larger bonus — it's "on topic". Confidence gates this too.
 * @param {object} state
 * @param {string} text  the candidate's searchable text (summary+detail+...)
 * @returns {number}
 */
export function lexicalFocusBonus(state, text) {
  if (!text || state.focusLex.size === 0) return 0;
  const gate = state.conf;
  if (gate <= 0) return 0;
  const lower = String(text).toLowerCase();
  let bonus = 0;
  for (const [w, wt] of state.focusLex) {
    if (lower.includes(w)) bonus += wt * state.lexGain;
  }
  return bonus * gate;
}

export function snapshot(state) {
  return {
    z_focus: softmax(state.u),
    z_conf: state.conf,
    boosts: readoutBoosts(state),
    steps: state.steps,
  };
}

export { TERM_TYPES, BOOST_TABLE, N_FOCUS, N_TERMS };
