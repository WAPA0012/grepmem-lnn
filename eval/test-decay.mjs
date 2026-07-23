/**
 * Unit tests for the LNN liquid decay math.
 *
 * These verify the JS CfC reduction is mathematically correct by comparing
 * against the closed-form analytical solution computed independently here in
 * the test (not by reusing the implementation under test). Tolerance is
 * tight (1e-9) because the implementation is just Math.exp — any deviation
 * indicates a formula bug, not float noise.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  createLnnState,
  lnnDecay,
  lnnReinforce,
  effectiveTau,
  halfLife,
  clampSalience,
  LNN_DEFAULTS,
} from '../lib/lnn-decay.js';

// ─── Pure-decay closed form (independent reference) ──────────────────────────
// x(t) = A + (x0 - A) · exp(-w_tau · t)
function analytical(x0, wTau, A, t) {
  return A + (x0 - A) * Math.exp(-wTau * t);
}

test('lnnDecay matches the analytical closed-form solution at multiple t', () => {
  const x0 = 0.9, wTau = 0.01, A = 0.1;
  for (const t of [0, 1, 7, 30, 100, 365, 1000]) {
    const s = createLnnState({ x0, wTau, A });
    const got = lnnDecay(s, t);
    const want = clampSalience(analytical(x0, wTau, A, t));
    assert.ok(Math.abs(got - want) < 1e-9,
      `t=${t}: lnnDecay=${got} analytical=${want} delta=${got - want}`);
  }
});

test('lnnDecay is monotonic non-increasing over time', () => {
  const s = createLnnState({ x0: 0.8, wTau: 0.005 });
  let prev = Infinity;
  for (let t = 0; t <= 500; t += 10) {
    const tmp = createLnnState({ x0: 0.8, wTau: 0.005 });
    const v = lnnDecay(tmp, t);
    assert.ok(v <= prev + 1e-12, `decay should be monotonic: t=${t} v=${v} prev=${prev}`);
    prev = v;
  }
});

test('lnnDecay asymptotes to the floor A, never below', () => {
  const s = createLnnState({ x0: 1.0, wTau: 0.01, A: 0.1 });
  const v = lnnDecay(s, 1e6); // effectively infinite time
  assert.ok(v >= LNN_DEFAULTS.SALIENCE_MIN, `never below clamp floor: ${v}`);
  assert.ok(Math.abs(v - 0.1) < 1e-3, `approaches A: ${v}`);
});

test('lnnDecay at t=0 returns the initial salience unchanged', () => {
  const s = createLnnState({ x0: 0.7 });
  assert.equal(lnnDecay(s, 0), 0.7);
});

test('effectiveTau = 1/w_tau', () => {
  const s = createLnnState({ wTau: 0.01 });
  assert.equal(effectiveTau(s), 100);
});

test('halfLife = ln2 / w_tau', () => {
  const s = createLnnState({ wTau: 0.005 });
  assert.ok(Math.abs(halfLife(s) - Math.LN2 / 0.005) < 1e-9);
  // Sanity: at halfLife, value should be midway between x0 and A
  const s2 = createLnnState({ x0: 1.0, wTau: 0.005, A: 0.1 });
  const v = lnnDecay(s2, halfLife(s2));
  const midpoint = (1.0 + 0.1) / 2;
  assert.ok(Math.abs(v - midpoint) < 1e-6, `halfLife point=${v} midpoint=${midpoint}`);
});

// ─── Reinforcement (Ebbinghaus consolidation) ────────────────────────────────

test('lnnReinforce decreases w_tau (slows future decay)', () => {
  const s = createLnnState({ wTau: 0.005 });
  const before = s.wTau;
  lnnReinforce(s, { reactivate: false });
  assert.ok(s.wTau < before, `w_tau should decrease: ${before} → ${s.wTau}`);
});

test('lnnReinforce has diminishing returns (each step smaller than the last)', () => {
  const s = createLnnState({ wTau: 0.005 });
  const steps = [];
  for (let i = 0; i < 10; i++) {
    const w = s.wTau;
    lnnReinforce(s, { reactivate: false });
    steps.push(w - s.wTau);
  }
  for (let i = 1; i < steps.length; i++) {
    assert.ok(steps[i] <= steps[i - 1] + 1e-15,
      `step ${i} (${steps[i]}) should be ≤ step ${i - 1} (${steps[i - 1]})`);
  }
});

test('lnnReinforce never drives w_tau below its floor', () => {
  const s = createLnnState({ wTau: 0.005 });
  for (let i = 0; i < 10000; i++) lnnReinforce(s, { reactivate: false });
  assert.ok(s.wTau > 0, `w_tau stays positive: ${s.wTau}`);
});

test('a reinforced memory decays slower than an unreinforced one after the same time', () => {
  const fresh = createLnnState({ x0: 1.0, wTau: 0.005 });
  const used = createLnnState({ x0: 1.0, wTau: 0.005 });
  for (let i = 0; i < 5; i++) lnnReinforce(used, { reactivate: false });
  const t = 200;
  const vFresh = lnnDecay({ ...createLnnState({ x0: 1.0, wTau: 0.005 }) }, t);
  const vUsed = lnnDecay({ ...used }, t);
  assert.ok(vUsed > vFresh,
    `reinforced memory should retain more: used=${vUsed} fresh=${vFresh}`);
});

test('lnnReinforce with reactivate resets x to full', () => {
  const s = createLnnState({ x0: 0.5 });
  lnnDecay(s, 100); // decay it down
  assert.ok(s.x < 0.5);
  lnnReinforce(s); // default reactivate=true
  assert.equal(s.x, 1.0);
});

// ─── Clamp behavior ──────────────────────────────────────────────────────────

test('clampSalience bounds to [0.1, 1.0]', () => {
  assert.equal(clampSalience(-5), 0.1);
  assert.equal(clampSalience(0.5), 0.5);
  assert.equal(clampSalience(99), 1.0);
});

test('decay never produces values outside [0.1, 1.0] even with extreme inputs', () => {
  for (const wTau of [1e-6, 0.001, 0.01, 1, 100]) {
    for (const t of [0, 0.5, 10, 365, 1e5]) {
      const s = createLnnState({ x0: 1.0, wTau });
      const v = lnnDecay(s, t);
      assert.ok(v >= 0.1 && v <= 1.0, `out of bounds wTau=${wTau} t=${t} v=${v}`);
    }
  }
});
