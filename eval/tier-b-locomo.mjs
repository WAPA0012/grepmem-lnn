/**
 * Tier B — LoCoMo real multi-session data.
 *
 * LoCoMo (snap-research/locomo) is ~10 long persona conversations, each with
 * up to 35 sessions spanning weeks. Each turn has a `dia_id` like "D1:3"
 * (session 1, turn 3) and each QA has an `evidence` list of dia_ids pointing
 * at the turns containing the answer. This gives genuine passage-level GT for
 * R@k AND real multi-session temporal gaps — the setting where the coupled
 * CfC's confidence-modulated τ is hypothesized to matter (long horizons,
 * state decay across session gaps).
 *
 * Protocol per conversation:
 *   1. Flatten session_N turns into an ordered corpus, keyed by dia_id.
 *   2. For each QA with non-empty evidence:
 *      - Feed ALL turns from sessions BEFORE the evidence session as the
 *        "accumulated history" that builds scheduler state (each turn's text
 *        treated as a prior query — the agent has "seen" this content).
 *      - Actually: we feed the EARLIER sessions' turns as context to build
 *        focus, then retrieve on the QA question over the FULL corpus and
 *        measure whether the evidence turn(s) rank in top-k.
 *   3. Compare flat (no state) vs scalar-focus vs lnn-coupled, any-hit R@k
 *      against the evidence dia_id set.
 *
 * Honest note: LoCoMo's QA questions are asked by an evaluator about the
 * conversation, not in-conversation follow-ups. So "feeding turns as queries"
 * is an approximation — the state is built from the conversation CONTENT
 * (what the speakers discussed), which is a reasonable proxy for "the agent
 * has been tracking this conversation's topics".
 */
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadGrepmem, loadGrepmemGrep } from '../lib/grepmem-shim.js';
import { createState, step, readoutBoosts } from '../lib/state-scheduler.js';
import { stateAwareRawScore, combinedStateScore, nodeText } from '../lib/match-rescore.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, 'locomo', 'locomo10.json');
const LIMIT_CONV = parseInt(process.env.CONV || '10', 10);   // conversations (max 10)
const LIMIT_QA = parseInt(process.env.QA || '0', 10);        // QA per conv (0 = all)
const TOPK = parseInt(process.env.TOPK || '5', 10);

// ─── Load + flatten LoCoMo ──────────────────────────────────────────────────

function loadLoCoMo() {
  const raw = JSON.parse(readFileSync(DATA, 'utf8'));
  const convs = [];
  for (const sample of raw) {
    const conv = sample.conversation;
    const sessions = []; // [{idx, date, turns:[{dia_id, speaker, text}]}]
    for (let i = 1; i <= 35; i++) {
      const key = `session_${i}`;
      const dateKey = `session_${i}_date_time`;
      if (!conv[key]) break;
      sessions.push({
        idx: i,
        date: conv[dateKey] || '',
        turns: conv[key].map(t => ({ dia_id: t.dia_id, speaker: t.speaker, text: t.text })),
      });
    }
    convs.push({
      sample_id: sample.sample_id,
      sessions,
      qa: sample.qa.filter(q => q.evidence && q.evidence.length > 0),
    });
  }
  return convs;
}

// Build a pseudo-memory node per turn (so match-rescore can score it).
function turnToNode(turn, sessionDate) {
  return {
    type: 'conversation',
    summary: turn.text.slice(0, 80),
    detail: '',
    conversation: `${turn.speaker}: ${turn.text}`,
    triggers: [],
    tags: [],
    baseSalience: 0.5,
    accessCount: 0,
    lastAccess: '2023-01-01',
  };
}

// ─── Retrieval over the full corpus with a given scorer ─────────────────────

function retrieveRanked(corpus, terms, boosts, state, useFocus) {
  const scored = corpus.map(({ dia_id, node }) => {
    const score = useFocus
      ? combinedStateScore(node, terms, boosts, state)
      : stateAwareRawScore(node, terms, boosts);
    return { dia_id, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

function anyHitRecall(ranked, evidenceIds, k) {
  const topk = ranked.slice(0, k).map(r => r.dia_id);
  return topk.some(id => evidenceIds.includes(id)) ? 1 : 0;
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  if (!existsSync(DATA)) {
    console.error(`LoCoMo not found at ${DATA}. Run:`);
    console.error('  curl -L -o eval/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json');
    process.exit(1);
  }
  const { extractQueryTerms } = await loadGrepmemGrep();

  const convs = loadLoCoMo().slice(0, LIMIT_CONV);
  console.log('═══════════════════════════════════════════════════════════════════════');
  console.log('  Tier B — LoCoMo real multi-session data');
  console.log(`  ${convs.length} conversations, R@${TOPK}, evidence-filtered QA`);
  console.log('═══════════════════════════════════════════════════════════════════════\n');

  // Accumulators per strategy, overall + by temporal-span bucket
  const stats = {
    flat: { hit: 0, n: 0, bySpan: {} },
    scalar: { hit: 0, n: 0, bySpan: {} },
    coupled: { hit: 0, n: 0, bySpan: {} },
  };
  const bump = (s, spanBucket, hit) => {
    s.hit += hit; s.n++;
    s.bySpan[spanBucket] ||= { hit: 0, n: 0 };
    s.bySpan[spanBucket].hit += hit; s.bySpan[spanBucket].n++;
  };

  for (const conv of convs) {
    // Flatten corpus: all turns across all sessions, in order.
    const corpus = [];
    for (const sess of conv.sessions) {
      for (const t of sess.turns) {
        corpus.push({ dia_id: t.dia_id, node: turnToNode(t, sess.date) });
      }
    }
    const corpusByDia = new Map(corpus.map(c => [c.dia_id, c]));

    const qaList = LIMIT_QA > 0 ? conv.qa.slice(0, LIMIT_QA) : conv.qa;

    for (const qa of qaList) {
      const evidence = qa.evidence;
      // Determine the evidence session(s) for temporal-span bucketing.
      const evSessions = evidence.map(e => parseInt((e.match(/D(\d+):/) || [])[1] || '1', 10));
      const maxEvSession = Math.max(...evSessions);
      const spanBucket = maxEvSession <= 5 ? 'early(1-5)' : maxEvSession <= 15 ? 'mid(6-15)' : 'late(16+)';

      const terms = extractQueryTerms(qa.question);
      if (terms.length === 0) continue;

      // --- flat: no state, boosts all 1.0 ---
      const flatBoosts = new Array(9).fill(1.0);
      const flatRanked = retrieveRanked(corpus, terms, flatBoosts, null, false);
      bump(stats.flat, spanBucket, anyHitRecall(flatRanked, evidence, TOPK));

      // --- Build scheduler state from PRIOR sessions (before evidence session) ---
      // Feed the earlier sessions' turn texts as "queries" to accumulate focus.
      // This is the state the coupled CfC is supposed to maintain across the
      // temporal gap up to the evidence.
      const priorSessionTexts = [];
      for (const sess of conv.sessions) {
        if (sess.idx >= maxEvSession) break;
        for (const t of sess.turns) priorSessionTexts.push(t.text);
      }
      // Cap how many prior turns we feed (state converges; avoid huge loops)
      const feedTexts = priorSessionTexts.slice(-40);

      // --- scalar-focus ablation (fixed τ) ---
      const scalar = makeScalarStateB();
      for (const txt of feedTexts) stepScalarB(scalar, txt);
      const scalarBoosts = scalarBoostsB(scalar);
      const scalarRanked = retrieveRanked(corpus, terms, scalarBoosts, scalar, true);
      bump(stats.scalar, spanBucket, anyHitRecall(scalarRanked, evidence, TOPK));

      // --- lnn-coupled ---
      const coupled = createState();
      for (const txt of feedTexts) step(coupled, txt);
      const coupledBoosts = readoutBoosts(coupled);
      const coupledRanked = retrieveRanked(corpus, terms, coupledBoosts, coupled, true);
      bump(stats.coupled, spanBucket, anyHitRecall(coupledRanked, evidence, TOPK));
    }
  }

  // ── Report ──
  const pct = (s) => s.n ? (s.hit / s.n * 100).toFixed(1) : '—';
  console.log('  strategy'.padEnd(12) + `R@${TOPK} overall`.padStart(14) + '  by temporal span'.padStart(0));
  console.log('  ' + '─'.repeat(60));
  for (const [name, s] of Object.entries(stats)) {
    const spans = ['early(1-5)', 'mid(6-15)', 'late(16+)']
      .map(b => `${b}:${s.bySpan[b] ? (s.bySpan[b].hit / s.bySpan[b].n * 100).toFixed(0) + '%' : '—'}`).join('  ');
    console.log('  ' + name.padEnd(12) + (pct(s) + '%').padStart(14) + '   ' + spans);
  }
  console.log('  ' + '─'.repeat(60));
  const fR = stats.flat.n ? stats.flat.hit / stats.flat.n * 100 : 0;
  const sR = stats.scalar.n ? stats.scalar.hit / stats.scalar.n * 100 : 0;
  const cR = stats.coupled.n ? stats.coupled.hit / stats.coupled.n * 100 : 0;
  console.log(`\n  Verdict (n=${stats.coupled.n}):`);
  console.log(`    flat → coupled      : ${(cR - fR).toFixed(1)}pt`);
  console.log(`    scalar → coupled    : ${(cR - sR).toFixed(1)}pt  (>0 = coupling helps on long horizons)`);
  // late-span is where coupling should shine most
  const lateC = stats.coupled.bySpan['late(16+)'];
  const lateS = stats.scalar.bySpan['late(16+)'];
  if (lateC && lateS) {
    console.log(`    late-span scalar→coupled : ${((lateC.hit/lateC.n - lateS.hit/lateS.n)*100).toFixed(1)}pt  (the coupling test zone)`);
  }
  console.log('═══════════════════════════════════════════════════════════════════════');
}

// ─── Scalar-focus ablation for Tier B (fixed τ, has focusLex) ───────────────
// Mirrors tier-a's scalar scheduler. Kept inline to avoid cross-file coupling.
import { BOOST_TABLE } from '../lib/state-scheduler.js';
import { extractFocusSignal } from '../lib/focus-classifier.js';
const STOP_B = new Set(['the','and','for','not','but','are','was','has','this','that','with','from','can','all','use','need','what','how','why','when','who','which','where','does','did','were','you','your','its','have','had','they','them','their','been','being','about','into','some','than','then','also','just','very','really','one','two']);
function tokB(q){ if(!q)return[]; const w=new Set(); const en=(q.toLowerCase().match(/[a-z][a-z0-9_]{2,}/g)||[]); for(const x of en)if(!STOP_B.has(x))w.add(x); const cn=q.replace(/[^一-鿿]/g,''); for(let i=0;i<cn.length-1;i++)w.add(cn[i]+cn[i+1]); return[...w]; }
function makeScalarStateB(){ return { u:[0,0,0,0], fixedTau:2.0, fixedConf:0.5, inputGain:2.5, focusLex:new Map(), tauLex:2.5, lexGain:1.0, conf:0.5, confEpsilon:0.0, steps:0 }; }
function stepScalarB(state, query, dt=1){ const I=extractFocusSignal(query); for(let i=0;i<4;i++){ const tg=I[i]*state.inputGain; state.u[i]=tg+(state.u[i]-tg)*Math.exp(-dt/state.fixedTau); } for(const [w,wt] of state.focusLex){ const d=wt*Math.exp(-dt/state.tauLex); if(d<0.01)state.focusLex.delete(w); else state.focusLex.set(w,d); } for(const w of tokB(query))state.focusLex.set(w,(state.focusLex.get(w)||0)+1); state.steps++; }
function scalarBoostsB(state){ const m=Math.max(...state.u); const e=state.u.map(x=>Math.exp(x-m)); const sm=e.reduce((a,b)=>a+b,0)||1; const z=e.map(x=>x/sm); const g=state.fixedConf; const b=new Array(9).fill(1.0); for(let t=0;t<9;t++){ let r=0; for(let m2=0;m2<4;m2++)r+=z[m2]*BOOST_TABLE[m2][t]; b[t]=1+(r-1)*g; } return b; }

main().catch(e => { console.error(e); process.exit(1); });
