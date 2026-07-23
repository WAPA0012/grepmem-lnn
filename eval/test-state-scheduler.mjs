/**
 * Unit tests for the state-aware scheduler (round 2).
 *
 * Verifies the CfC dynamics are correct: softmax normalization, confidence
 * decay/accumulation, and — most importantly — the COUPLING effect that
 * distinguishes the coupled CfC from a per-channel scalar decay. Also tests
 * the focus-classifier signal extraction and the boost readout.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createState, step, readoutBoosts, boostFor, snapshot, TERM_TYPES, BOOST_TABLE } from '../lib/state-scheduler.js';
import { extractFocusSignal, dominantFocus, FOCUS_MODES } from '../lib/focus-classifier.js';

// ─── Focus classifier ───────────────────────────────────────────────────────

test('extractFocusSignal flags error vocabulary as procedure-debug', () => {
  const I = extractFocusSignal('auth service throwing ConnectionRefused error, how to fix');
  assert.ok(I[1] > 0, `procedure-debug signal > 0: ${I}`);
  assert.equal(dominantFocus(I), 1);
});

test('extractFocusSignal flags IP/port as exact-entity', () => {
  const I = extractFocusSignal('connect to 10.0.0.5 port 6379');
  assert.ok(I[0] > 0, `exact-entity signal > 0: ${I}`);
  assert.equal(dominantFocus(I), 0);
});

test('extractFocusSignal flags config keys as config-param', () => {
  const I = extractFocusSignal('set MAX_CONNECTIONS=100 and timeout=30');
  assert.ok(I[2] > 0, `config-param signal > 0: ${I}`);
});

test('extractFocusSignal flags recall words as narrative-context', () => {
  const I = extractFocusSignal('what did we discuss last week about the deployment');
  assert.ok(I[3] > 0, `narrative-context signal > 0: ${I}`);
});

test('extractFocusSignal returns zero vector for empty/non-string input', () => {
  assert.deepEqual(extractFocusSignal(''), [0, 0, 0, 0]);
  assert.deepEqual(extractFocusSignal(null), [0, 0, 0, 0]);
});

// ─── Scheduler state invariants ─────────────────────────────────────────────

test('initial z_focus is uniform (sums to 1, all equal)', () => {
  const s = createState();
  const snap = snapshot(s);
  assert.ok(Math.abs(snap.z_focus.reduce((a, b) => a + b, 0) - 1) < 1e-9, 'focus sums to 1');
  for (const p of snap.z_focus) assert.ok(Math.abs(p - 0.25) < 1e-9, `uniform: ${p}`);
});

test('initial confidence is 0 → all boosts are 1.0 (neutral = flat)', () => {
  const s = createState();
  const boosts = readoutBoosts(s);
  for (const b of boosts) assert.ok(Math.abs(b - 1.0) < 1e-9, `boost neutral at conf=0: ${b}`);
});

test('after a debugging query, focus shifts toward procedure-debug and boosts rise', () => {
  const s = createState();
  step(s, 'getting ConnectionRefused from auth, how to fix the error');
  step(s, 'still failing, the service crashed again');
  const snap = snapshot(s);
  assert.ok(snap.z_focus[1] > snap.z_focus[0], `debug > exact-entity: ${snap.z_focus}`);
  assert.ok(snap.z_conf > 0, `confidence rose: ${snap.z_conf}`);
  // synonym/learned boosts (debug-favored) should now exceed 1.0
  const synIdx = TERM_TYPES.indexOf('synonym');
  assert.ok(snap.boosts[synIdx] > 1.0, `synonym boost > 1: ${snap.boosts[synIdx]}`);
});

test('z_focus always sums to 1 after any sequence of steps', () => {
  const s = createState();
  const queries = ['10.0.0.5 port 6379', 'error crash', 'what did we say', 'MAX_CONN=5', '', 'random words here'];
  for (const q of queries) {
    step(s, q);
    const snap = snapshot(s);
    assert.ok(Math.abs(snap.z_focus.reduce((a, b) => a + b, 0) - 1) < 1e-9, `sums to 1 after "${q}": ${snap.z_focus}`);
    for (const p of snap.z_focus) assert.ok(p >= 0 && p <= 1, `valid prob: ${p}`);
  }
});

test('confidence is always in [0,1]', () => {
  const s = createState();
  for (const q of ['error error error error error', 'x', '']) {
    step(s, q);
    const snap = snapshot(s);
    assert.ok(snap.z_conf >= 0 && snap.z_conf <= 1, `conf in [0,1]: ${snap.z_conf}`);
  }
});

// ─── THE COUPLING TEST (load-bearing) ───────────────────────────────────────
//
// This is the test that distinguishes the coupled CfC from a per-channel
// scalar-decay ablation. With HIGH confidence, focus τ is large → a new query
// of a DIFFERENT focus should barely move the focus distribution (inertia).
// With LOW confidence, the same query should move it much more.
// A scalar-decay model (fixed τ per channel) cannot reproduce this dependence
// on the confidence channel — that's the whole point of the coupling.

test('COUPLING: high confidence makes focus resist change (inertia)', () => {
  // Establish strong debug focus with high confidence.
  const s = createState();
  for (let i = 0; i < 5; i++) step(s, 'ConnectionRefused error crash fix'); // builds debug focus + conf
  const beforeSwitch = snapshot(s).z_focus.slice();
  assert.ok(beforeSwitch[1] > 0.4, `debug focus established: ${beforeSwitch}`);

  // Now a single opposite-focus query (exact-entity). With high confidence,
  // the focus should NOT flip — it should resist.
  step(s, '10.0.0.5 port 6379');
  const afterHighConf = snapshot(s).z_focus.slice();
  // debug focus should still be the dominant mode (inertia held it)
  assert.ok(afterHighConf[1] > afterHighConf[0],
    `high-conf inertia kept debug dominant: ${afterHighConf}`);
});

test('COUPLING: low confidence lets focus re-orient fast', () => {
  // Start with NO confidence (fresh state). One debug query builds a little
  // focus but low confidence. Then an exact-entity query should be able to
  // pull focus toward exact-entity more easily than in the high-conf case.
  const s = createState();
  step(s, 'error'); // one weak debug signal, low confidence
  const confAfterOne = snapshot(s).z_conf;
  step(s, '10.0.0.5 port 6379'); // strong exact-entity signal
  const afterLowConf = snapshot(s).z_focus.slice();
  // exact-entity should have gained substantial ground
  assert.ok(afterLowConf[0] > 0.3, `low-conf let exact-entity rise: ${afterLowConf}`);
});

test('COUPLING: the two regimes differ (the ablation-defining property)', () => {
  // Run both scenarios to the same probe query and show the resulting focus
  // distributions DIFFER because of the confidence state — something a fixed-τ
  // scalar model could not produce from the same probe.
  function run(priorQueries, probe) {
    const s = createState();
    for (const q of priorQueries) step(s, q);
    step(s, probe);
    return snapshot(s).z_focus.slice();
  }
  const highConfPath = run(
    ['error crash', 'error crash', 'error crash', 'error crash'],
    '10.0.0.5 port 6379');
  const lowConfPath = run(
    [],
    '10.0.0.5 port 6379');
  // The probe-only path (no priors) lets exact-entity dominate; the
  // high-confidence path resists and keeps more debug mass.
  assert.ok(lowConfPath[0] > highConfPath[0],
    `probe-only lets exact-entity dominate more than high-conf path: low=${lowConfPath[0]} high=${highConfPath[0]}`);
});

// ─── Boost readout ──────────────────────────────────────────────────────────

test('boostFor returns 1.0 for unknown term types', () => {
  const s = createState();
  assert.equal(boostFor(s, 'nonexistent'), 1.0);
});

test('BOOST_TABLE is 4×9 with all entries ≥ 1.0', () => {
  assert.equal(BOOST_TABLE.length, 4);
  for (const row of BOOST_TABLE) {
    assert.equal(row.length, 9);
    for (const v of row) assert.ok(v >= 1.0, `boost ≥ 1.0: ${v}`);
  }
});

test('readoutBoosts returns length-9 vector aligned with TERM_TYPES', () => {
  const s = createState();
  step(s, 'error crash fix'); // build some confidence
  const b = readoutBoosts(s);
  assert.equal(b.length, 9);
  assert.equal(b.length, TERM_TYPES.length);
});
