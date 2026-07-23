/**
 * State-aware match re-scoring.
 *
 * grepmem's result objects expose `_matchedTerms` only as a COUNT — the term
 * identities/types are not recoverable post-hoc (see grep.js:222-233). So to
 * apply state-aware boosts we must RE-SIMULATE the per-term match ourselves,
 * mirroring the round-1 pattern that recomputed salience: import grepmem's
 * exported `extractQueryTerms`, test each term's pattern against the node's
 * searchable text, apply field weights + state boosts, and produce a new
 * match score. This keeps grepmem source untouched.
 *
 * The re-simulation approximates the index-path scoring (FIELD_WEIGHTS
 * {trigger:3.0, tag:2.0, summary_term:1.5, term:1.0}). It is NOT bit-identical
 * to grepmem's `match`, but it preserves the same RELATIVE weighting structure
 * so cross-strategy comparisons are fair.
 */

import { TERM_TYPES, lexicalFocusBonus } from './state-scheduler.js';

// Field weights — mirror index.js FIELD_WEIGHTS so re-simulated scores live
// in the same ballpark as grepmem's real `match`.
const FIELD_WEIGHTS = {
  trigger: 3.0,
  tag: 2.0,
  summary_term: 1.5,
  term: 1.0,
};

// Map each grepmem term.type to a column index in the boost vector (aligned
// with state-scheduler.js TERM_TYPES). Any unknown type → no boost (1.0).
function termTypeIndex(termType) {
  return TERM_TYPES.indexOf(termType);
}

/**
 * Build the searchable surface set for a node, each with its field weight.
 * Returns [{text, weight}] to test term patterns against.
 */
function nodeSurfaces(node) {
  const surfaces = [];
  const triggers = node.triggers || [];
  for (const t of triggers) surfaces.push({ text: String(t), weight: FIELD_WEIGHTS.trigger });
  const tags = node.tags || [];
  for (const t of tags) surfaces.push({ text: String(t), weight: FIELD_WEIGHTS.tag });
  if (node.summary) surfaces.push({ text: String(node.summary), weight: FIELD_WEIGHTS.summary_term });
  const body = node.type === 'conversation' ? (node.conversation || '') : (node.detail || '');
  surfaces.push({ text: String(body || ''), weight: FIELD_WEIGHTS.term });
  return surfaces;
}

/**
 * Re-simulate grepmem's match score for a node under a given boost vector.
 *
 * For each extracted term, for each surface it matches, add:
 *     term.weight × fieldWeight × boost(term.type)
 * This mirrors index.js lookupScored's Σ term.weight × fieldWeight, with the
 * added state-aware boost factor.
 *
 * @param {object} node       grepmem memory node
 * @param {Array} terms       output of grepmem's extractQueryTerms(query)
 * @param {number[]} boosts   length-9 boost vector from state-scheduler readout
 * @returns {number} raw state-aware score (higher = more relevant)
 */
export function stateAwareRawScore(node, terms, boosts) {
  const surfaces = nodeSurfaces(node);
  let score = 0;
  for (const term of terms) {
    const tIdx = termTypeIndex(term.type);
    const boost = tIdx >= 0 ? boosts[tIdx] : 1.0;
    if (boost <= 0) continue;
    const pat = String(term.pattern).toLowerCase();
    for (const s of surfaces) {
      if (s.text.toLowerCase().includes(pat)) {
        score += term.weight * s.weight * boost;
      }
    }
  }
  return score;
}

/**
 * The candidate's full searchable text as one string (for the lexical-focus
 * channel to scan). Mirrors nodeSurfaces but flattened.
 */
export function nodeText(node) {
  const parts = [];
  for (const t of (node.triggers || [])) parts.push(String(t));
  for (const t of (node.tags || [])) parts.push(String(t));
  if (node.summary) parts.push(String(node.summary));
  parts.push(String(node.type === 'conversation' ? (node.conversation || '') : (node.detail || '')));
  return parts.join(' ');
}

/**
 * Combined state-aware raw score = term-type-boosted lexical match
 *                                  + lexical-focus bonus (topic continuity).
 *
 * The two channels are complementary:
 *   - term-type boosts fire when the query generates synonym/learned terms
 *     (mainly bilingual queries)
 *   - lexical-focus bonus fires when the candidate shares vocabulary with the
 *     recent focus-establishing turns (works on any language)
 * Both are state-driven; both use CfC continuous-time decay. The lexical-focus
 * channel is what makes state-awareness observable on English queries.
 *
 * @param {object} node
 * @param {Array} terms
 * @param {number[]} boosts
 * @param {object} state     scheduler state (for lexicalFocusBonus)
 * @returns {number}
 */
export function combinedStateScore(node, terms, boosts, state) {
  const lexical = stateAwareRawScore(node, terms, boosts);
  const focusBonus = lexicalFocusBonus(state, nodeText(node));
  return lexical + focusBonus;
}

/**
 * Normalized state-aware match ∈ [0,1], using the same normalization shape as
 * grepmem (score / (maxPossible × 0.3), clamped to 1).
 */
export function stateAwareMatch(node, terms, boosts) {
  const raw = stateAwareRawScore(node, terms, boosts);
  const maxPossible = terms.reduce((sum, t) => sum + t.weight * FIELD_WEIGHTS.trigger, 0);
  return Math.min(1.0, raw / Math.max(maxPossible * 0.3, 1));
}

/**
 * Flat (no-boost) re-simulated match — the fair within-module baseline.
 * Equivalent to stateAwareMatch with all boosts = 1.0. Used so that
 * `flat-rescore` vs `lnn-coupled` differ ONLY by the boost factor (not by
 * re-simulation noise vs grepmem's real match).
 */
export function flatRescoreMatch(node, terms) {
  return stateAwareMatch(node, terms, new Array(TERM_TYPES.length).fill(1.0));
}

export { FIELD_WEIGHTS };
