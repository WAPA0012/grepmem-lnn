/**
 * Liquid Neural Network time-decay for scalar salience.
 *
 * Based on the Closed-form Continuous-time (CfC) model
 * (Hasani et al., Nature Machine Intelligence 2022), reduced to the
 * pure-decay case (no input signal, I = 0).
 *
 * The CfC ODE for a hidden state x with synaptic input f is:
 *
 *     dx/dt = -(w_tau + f) · x + A · f
 *
 * CfC's defining contribution is the closed-form solution over a step dt,
 * which removes the need for an ODE solver:
 *
 *     x(t+dt) = A · f/(w_tau+f) + [ x(t) - A·f/(w_tau+f) ] · exp(-(w_tau+f)·dt)
 *
 * In the pure-decay regime (f = 0, no external input — salience simply
 * relaxing toward its floor between accesses), this collapses to:
 *
 *     x(t+dt) = A + (x(t) - A) · exp(-w_tau · dt)        ... (Eq. decay)
 *
 * which is the standard linear first-order relaxation with time constant
 *     τ_eff = 1 / w_tau
 *
 * The "liquid" property is twofold:
 *   1. w_tau is PER-MEMORY (each memory has its own τ), unlike grepmem's
 *      flat model which uses one global 0.005 for all.
 *   2. w_tau is STATE-DEPENDENT: each reinforcement (retrieval) grows w_tau
 *      with diminishing returns, so frequently-used memories decay SLOWER
 *      (their τ_eff increases). This is the Ebbinghaus-consolidation effect
 *      that grepmem's DESIGN.md:424 lists as a planned-but-unimplemented goal.
 *
 * References:
 *   - https://www.nature.com/articles/s42256-022-00556-7
 *   - https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc_cell.py
 *   - https://www.mdpi.com/2218-6581/13/9/126  (independent reproduction)
 */

// Default floor. Matches grepmem's clamp lower bound (effectiveSalience ≥ 0.1).
const DEFAULT_A = 0.1;

// Default time-constant parameter w_tau for a fresh memory.
// τ_eff = 1/w_tau; half-life = ln2/w_tau. With w_tau=0.005, half-life ≈ 139
// days. This is in the same order as grepmem's flat ~200-day half-life, so a
// NEVER-reinforced LNN memory decays comparably to the flat baseline.
// Reinforcement (lnnReinforce) then only ever slows decay (decreases w_tau),
// so an accessed LNN memory ≥ its flat counterpart by construction.
const DEFAULT_W_TAU = 0.005;

// Salience clamp bounds (mirror grepmem's effectiveSalience clamp).
const SALIENCE_MIN = 0.1;
const SALIENCE_MAX = 1.0;

/**
 * Create a fresh LNN decay state for a memory node.
 * Attached to a node as `node._lnn`.
 *
 * @param {object} [opts]
 * @param {number} [opts.x0]         initial salience (default 0.5, = baseSalience)
 * @param {number} [opts.wTau]       time-constant param (default DEFAULT_W_TAU)
 * @param {number} [opts.A]          decay floor (default DEFAULT_A)
 * @param {number} [opts.n]          reinforcement count (default 0)
 */
export function createLnnState(opts = {}) {
  return {
    x: opts.x0 ?? 0.5,
    wTau: opts.wTau ?? DEFAULT_W_TAU,
    A: opts.A ?? DEFAULT_A,
    n: opts.n ?? 0,
  };
}

/**
 * Advance an LNN state forward by `days` of pure decay.
 *
 * Implements Eq. (decay): x(t+dt) = A + (x(t) - A) · exp(-w_tau · dt).
 * Mutates state.x in place and returns the clamped salience.
 *
 * @param {object} state   LNN state (mutated)
 * @param {number} days    elapsed days (≥ 0)
 * @returns {number}       clamped salience in [SALIENCE_MIN, SALIENCE_MAX]
 */
export function lnnDecay(state, days) {
  if (days < 0) days = 0;
  // Guard w_tau to avoid div-by-zero / negative exponents blowing up.
  // τ_eff = 1/w_tau must stay positive for a decaying exponential.
  const wTau = Math.max(1e-9, state.wTau);
  const relaxed = state.A + (state.x - state.A) * Math.exp(-wTau * days);
  state.x = relaxed;
  return clampSalience(relaxed);
}

/**
 * Reinforce an LNN state — the Ebbinghaus-consolidation step.
 *
 * Frequently-used memories should decay SLOWER, i.e. have a larger effective
 * time constant τ_eff = 1/w_tau, i.e. a SMALLER w_tau. So each retrieval
 * DECREMENTS w_tau toward a floor: a smaller w_tau ⇒ gentler exp(-w_tau·dt)
 * ⇒ slower decay. The decrement shrinks as 1/(1+n): the first reinforcement
 * is the largest, later ones add ever-smaller consolidation, asymptoting to a
 * floor where the memory is effectively permanent.
 *
 * @param {object} state   LNN state (mutated)
 * @param {object} [opts]
 * @param {boolean} [opts.reactivate]  if true (default), reset x to full (1.0)
 */
export function lnnReinforce(state, opts = {}) {
  // Diminishing-returns decrement toward the floor. n is the prior count.
  // Step shrinks as 1/(1+n), so the first reinforcement is the largest.
  const step = 0.001 / (1 + state.n);
  const WTAU_FLOOR = 1e-4; // τ_eff ceiling ~10000 days — effectively permanent
  state.wTau = Math.max(WTAU_FLOOR, state.wTau - step);
  state.n += 1;
  if (opts.reactivate !== false) {
    state.x = SALIENCE_MAX; // retrieval re-energizes the trace
  }
  return clampSalience(state.x);
}

/**
 * Effective time constant τ_eff (days) for a state. Diagnostic/inspectable.
 * τ_eff = 1/w_tau. Larger ⇒ slower decay.
 */
export function effectiveTau(state) {
  return 1 / Math.max(1e-9, state.wTau);
}

/**
 * Half-life in days: the dt at which x relaxes halfway from 1.0 toward A.
 * Derived from x(dt) = A + (1-A)exp(-w_tau·dt) = (1+A)/2  ⇒  dt = ln2/w_tau.
 */
export function halfLife(state) {
  return Math.LN2 / Math.max(1e-9, state.wTau);
}

export function clampSalience(x) {
  return Math.min(SALIENCE_MAX, Math.max(SALIENCE_MIN, x));
}

// Exported for tests / param sweeps.
export const LNN_DEFAULTS = { DEFAULT_A, DEFAULT_W_TAU, SALIENCE_MIN, SALIENCE_MAX };
