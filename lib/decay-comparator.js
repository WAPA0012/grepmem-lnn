/**
 * Unified decay interface: pick flat or lnn by env, produce a salience for a
 * grepmem memory node.
 *
 * grepmem's effectiveSalience (grep.js:332) is a local hardcoded function.
 * Rather than monkey-patch grepmem internals, this module provides a parallel
 * salience computer with the SAME signature/semantics so an experiment harness
 * can swap decay strategies via GREPMEM_DECAY without touching grepmem source.
 *
 * The flat strategy reproduces grepmem's logic VERBATIM (same constants), so
 * flat results here ≡ grepmem baseline. The lnn strategy is the experimental
 * CfC liquid decay.
 *
 * Both strategies read memory age via env.daysSince (honoring GREPMEM_NOW),
 * which is what lets LongMemEval's historical haystack_dates actually drive
 * decay — something the stock grepmem harness does NOT do (it sets lastAccess
 * to today on add(), zeroing out the age signal).
 */
import { lnnDecay, lnnReinforce, createLnnState, clampSalience } from './lnn-decay.js';
import { decayStrategy, daysSince, DECAY_FLAT, DECAY_LNN } from './env.js';

// grepmem flat-decay constants (grep.js:332-339), reproduced exactly.
const FLAT_DECAY_RATE = 0.005;   // per day
const FLAT_ACCESS_BOOST = 0.03;  // per access
const FLAT_ACCESS_CAP = 0.30;    // max boost from accessCount
const SALIENCE_FLOOR = 0.1;
const SALIENCE_CEIL = 1.0;

/**
 * Flat decay — verbatim reproduction of grepmem's effectiveSalience.
 *
 *   effective = clamp(base + min(access*0.03, 0.30) - days*0.005, 0.1, 1.0)
 *
 * @param {object} node  grepmem memory node
 * @returns {number} salience in [0.1, 1.0]
 */
export function flatSalience(node) {
  const base = node.baseSalience ?? 0.5;
  const boost = Math.min((node.accessCount ?? 0) * FLAT_ACCESS_BOOST, FLAT_ACCESS_CAP);
  const days = daysSince(node.lastAccess);
  const v = base + boost - days * FLAT_DECAY_RATE;
  return Math.min(SALIENCE_CEIL, Math.max(SALIENCE_FLOOR, v));
}

/**
 * LNN liquid decay. Recomputed fresh each evaluation (stateless x; only the
 * per-memory consolidation params wTau/n persist on node._lnn).
 *
 * FAIRNESS INVARIANT: the access-boost (base + access*0.03) is applied to the
 * STARTING level identically to flat — the LNN's distinct contribution is
 * confined to the DECAY term (per-memory, access-adaptive wTau). This makes
 * flat-vs-lnn a clean test of "does liquid, per-memory decay beat global
 * linear decay?" rather than conflating it with access-boost differences.
 *
 * @param {object} node  grepmem memory node
 * @returns {number} salience in [0.1, 1.0]
 */
export function lnnSalience(node) {
  const base = node.baseSalience ?? 0.5;
  // Lazy-init the per-node consolidation state.
  if (!node._lnn) {
    node._lnn = createLnnState({ x0: base });
  }
  // Treat accessCount as the reinforcement count: catch up consolidation so
  // each prior retrieval has slowed this memory's decay (lowered its wTau).
  const targetN = node.accessCount ?? 0;
  while (node._lnn.n < targetN) {
    lnnReinforce(node._lnn, { reactivate: false });
  }
  // FAIRNESS: flat adds +access*0.03 to the starting level. For a like-for-like
  // comparison, the LNN activation level must ALSO rise with access — otherwise
  // we'd be testing "LNN without access-boost vs flat with access-boost", which
  // is an unfair fight and not the hypothesis. The LNN's distinct contribution
  // is in the DECAY RATE (per-memory, access-adaptive wTau), not in denying
  // the access-boost. So: start from the access-boosted level, then apply the
  // liquid decay. This isolates the decay-mechanism difference cleanly.
  const boosted = Math.min(1.0, base + Math.min(targetN * 0.03, 0.30));
  node._lnn.x = boosted;
  const days = daysSince(node.lastAccess);
  return lnnDecay(node._lnn, days);
}

/**
 * Dispatch to the active strategy. This is the single entry point an
 * experiment harness should call.
 * @param {object} node
 * @returns {number}
 */
export function effectiveSalience(node) {
  return decayStrategy() === DECAY_LNN ? lnnSalience(node) : flatSalience(node);
}

/**
 * Score a retrieval hit: match × salience, matching grepmem's
 * results.sort((a,b) => b.match*b.salience - a.match*a.salience).
 * Allows an experiment to inject a custom salience function.
 *
 * @param {{match:number}} hit
 * @param {object} node
 * @param {(node)=>number} [salFn]  override (defaults to effectiveSalience)
 */
export function rankScore(hit, node, salFn = effectiveSalience) {
  return hit.match * salFn(node);
}

export { DECAY_FLAT, DECAY_LNN, clampSalience };
