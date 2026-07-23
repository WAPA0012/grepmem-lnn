/**
 * Bottleneck diagnostic — the go/no-go gate for Round 3 training.
 *
 * Answers: "is R2's LoCoMo failure a lexical ceiling (training is FUTILE) or
 * a fixable focus-signal problem (training COULD help)?"
 *
 * Oracle analysis: for each QA, compute R@5 under cheating conditions that
 * grant perfect information at each stage of the retrieval pipeline:
 *
 *   A. flat               — no state, boosts 1.0 (baseline = current flat R@5)
 *   B. oracle-focus-bonus — give evidence turns a DIRECT bonus (perfect topic ID)
 *   C. oracle-best-boost  — sweep uniform boost magnitudes, report best R@5
 *   D. real-boost-no-bag  — real CfC boosts, lexicalFocusBonus disabled
 *
 * DECISION GATE (the whole point of this script):
 *   B - A > 15pt  → focus channel COULD help if accurate → GO (train readout)
 *   B - A <  5pt  → LEXICAL CEILING → even perfect focus barely helps → NO-GO
 *                  (training is futile; report and end the direction)
 */
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadGrepmemGrep } from '../lib/grepmem-shim.js';
import { stateAwareRawScore } from '../lib/match-rescore.js';
import { createState, step, readoutBoosts } from '../lib/state-scheduler.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, 'locomo', 'locomo10.json');
const N_CONV = parseInt(process.env.CONV || '3', 10);  // conversations to analyze
const TOPK = parseInt(process.env.TOPK || '5', 10);

function loadConv(idx) {
  const raw = JSON.parse(readFileSync(DATA, 'utf8'));
  const sample = raw[idx];
  const conv = sample.conversation;
  const sessions = [];
  for (let i = 1; i <= 35; i++) {
    const key = `session_${i}`;
    if (!conv[key]) break;
    sessions.push({ idx: i, turns: conv[key].map(t => ({ dia_id: t.dia_id, speaker: t.speaker, text: t.text })) });
  }
  return { sessions, qa: sample.qa.filter(q => q.evidence && q.evidence.length > 0) };
}

function turnToNode(turn) {
  return { type: 'conversation', summary: turn.text.slice(0, 80), detail: '', conversation: `${turn.speaker}: ${turn.text}`, triggers: [], tags: [], baseSalience: 0.5, accessCount: 0, lastAccess: '2023-01-01' };
}

async function main() {
  if (!existsSync(DATA)) {
    console.error('Need eval/locomo/locomo10.json (curl it — see README)');
    process.exit(1);
  }
  const { extractQueryTerms } = await loadGrepmemGrep();

  let A = 0, B = 0, D = 0, n = 0;
  const cByMag = {};

  for (let ci = 0; ci < N_CONV; ci++) {
    const { sessions, qa } = loadConv(ci);
    const corpus = [];
    for (const sess of sessions) for (const t of sess.turns) corpus.push({ dia_id: t.dia_id, node: turnToNode(t) });
    const flatBoosts = new Array(9).fill(1.0);

    for (const q of qa) {
      const terms = extractQueryTerms(q.question);
      if (terms.length === 0) continue;
      const evidence = q.evidence;
      n++;

      // A. flat
      const rA = corpus.map(c => ({ d: c.dia_id, s: stateAwareRawScore(c.node, terms, flatBoosts) })).sort((a, b) => b.s - a.s);
      if (rA.slice(0, TOPK).some(r => evidence.includes(r.d))) A++;

      // B. oracle-focus-bonus: evidence turns get +50 (perfect topic identification)
      const rB = corpus.map(c => {
        const base = stateAwareRawScore(c.node, terms, flatBoosts);
        return { d: c.dia_id, s: base + (evidence.includes(c.dia_id) ? 50 : 0) };
      }).sort((a, b) => b.s - a.s);
      if (rB.slice(0, TOPK).some(r => evidence.includes(r.d))) B++;

      // C. uniform-boost sweep
      for (const mag of [1.0, 1.5, 2.0, 3.0, 5.0]) {
        const boosts = new Array(9).fill(mag);
        const rC = corpus.map(c => ({ d: c.dia_id, s: stateAwareRawScore(c.node, terms, boosts) })).sort((a, b) => b.s - a.s);
        cByMag[mag] = (cByMag[mag] || 0) + (rC.slice(0, TOPK).some(r => evidence.includes(r.d)) ? 1 : 0);
      }

      // D. real boosts from accumulated state, no focus-bag bonus
      const maxEvSess = Math.max(...evidence.map(e => parseInt((e.match(/D(\d+):/) || [])[1] || '1', 10)));
      const priorTexts = [];
      for (const sess of sessions) { if (sess.idx >= maxEvSess) break; for (const t of sess.turns) priorTexts.push(t.text); }
      const st = createState();
      for (const txt of priorTexts.slice(-30)) step(st, txt);
      const realBoosts = readoutBoosts(st);
      const rD = corpus.map(c => ({ d: c.dia_id, s: stateAwareRawScore(c.node, terms, realBoosts) })).sort((a, b) => b.s - a.s);
      if (rD.slice(0, TOPK).some(r => evidence.includes(r.d))) D++;
    }
  }

  const pct = (x) => (x / n * 100).toFixed(1);
  let bestMag = 1.0, bestHit = cByMag[1.0] || 0;
  for (const mag of [1.5, 2.0, 3.0, 5.0]) if ((cByMag[mag] || 0) > bestHit) { bestHit = cByMag[mag]; bestMag = mag; }

  console.log('═══════════════════════════════════════════════════════════');
  console.log(`  Round 3 BOTTLENECK DIAGNOSTIC — ${N_CONV} LoCoMo convs, n=${n} QA, R@${TOPK}`);
  console.log('═══════════════════════════════════════════════════════════\n');
  console.log(`  A. flat (no state)              : ${pct(A)}%`);
  console.log(`  B. oracle-focus-bonus (perfect) : ${pct(B)}%`);
  console.log(`  D. real boosts, no focus bag    : ${pct(D)}%`);
  console.log(`  C. uniform-boost sweep:`);
  for (const mag of [1.0, 1.5, 2.0, 3.0, 5.0]) console.log(`       boost=${mag.toFixed(1)}                   : ${pct(cByMag[mag] || 0)}%`);
  console.log(`       → best uniform boost=${bestMag}, R@5=${pct(bestHit)}%`);

  console.log('\n  ── DECISION GATE ──');
  const lift = (B / n * 100) - (A / n * 100);
  console.log(`  Oracle-focus lift (B - A) = ${lift >= 0 ? '+' : ''}${lift.toFixed(1)}pt`);
  let verdict;
  if (lift > 15) {
    verdict = 'GO';
    console.log(`  → ${lift.toFixed(1)}pt > 15pt: focus channel COULD help if accurate.`);
    console.log(`    Training the readout is WORTH IT. Proceed to train-readout.mjs.`);
  } else if (lift > 5) {
    verdict = 'MAYBE';
    console.log(`  → ${lift.toFixed(1)}pt (5-15): modest ceiling. Focus could help marginally;`);
    console.log(`    lexical match already captures most signal. Borderline — train readout, expect small gains.`);
  } else {
    verdict = 'NO-GO';
    console.log(`  → ${lift.toFixed(1)}pt < 5pt: LEXICAL CEILING. Even PERFECT topic identification`);
    console.log(`    barely helps. Training is FUTILE. Direction should end here.`);
  }
  const cLift = (bestHit / n * 100) - (A / n * 100);
  console.log(`  Best-boost lift (C - A) = ${cLift >= 0 ? '+' : ''}${cLift.toFixed(1)}pt — ${cLift > 3 ? 'readout magnitude matters' : 'readout magnitude is NOT the bottleneck'}.`);
  console.log(`\n  VERDICT: ${verdict}`);
  console.log('═══════════════════════════════════════════════════════════');
}
main().catch(e => { console.error(e); process.exit(1); });
