/**
 * Main experiment: flat vs lnn decay on LongMemEval-S retrieval (R@5 / R@10).
 *
 * Design (fairness-critical):
 *   - Ingest each question's haystack sessions ONCE into a fresh namespace,
 *     back-filling the historical haystack_dates into node.lastAccess/created
 *     so memory age actually drives decay (stock grepmem sets lastAccess=today
 *     on add(), zeroing the age signal — this is the bug our experiment fixes).
 *   - Run retrieval ONCE to get candidate (id, match) pairs. The `match` score
 *     is decay-independent (pure lexical grep score), so it's identical for
 *     both strategies — only salience differs.
 *   - Re-rank candidates with flat salience and with lnn salience INDEPENDENTLY,
 *     computing R@5/R@10 for each. This isolates the decay effect cleanly.
 *   - Break down by question_type; temporal-reasoning and knowledge-update are
 *     the expected LNN win zones (time-sensitive queries).
 *
 * Env:
 *   LIMIT       number of questions (default 20 for a smoke run; full=500)
 *   TYPES       comma-list of question_type to filter (default: all)
 *   DATA_PATH   override dataset path (default: ../ai-memory/data/longmemeval_s_cleaned.json)
 *   SEED        sampling seed (default 42)
 *   RESUME      '1' (default) resume from per-strategy checkpoint, '0' start fresh
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync, renameSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { loadGrepmem, GREPMEM_ROOT } from '../lib/grepmem-shim.js';
import { flatSalience, lnnSalience } from '../lib/decay-comparator.js';
import { daysSince } from '../lib/env.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const DEFAULT_DATA = join(GREPMEM_ROOT, 'data', 'longmemeval_s_cleaned.json');
const DATA_PATH = process.env.DATA_PATH || DEFAULT_DATA;
const LIMIT = parseInt(process.env.LIMIT || '20', 10);
const TYPES = process.env.TYPES ? process.env.TYPES.split(',').map(s => s.trim()) : null;
const SEED = parseInt(process.env.SEED || '42', 10);
const RESUME = (process.env.RESUME || '1') !== '0';
const CKPT_DIR = join(ROOT, '.ab-decay');
const CKPT_FLAT = join(CKPT_DIR, 'flat.jsonl');
const CKPT_LNN = join(CKPT_DIR, 'lnn.jsonl');

// ─── Helpers ────────────────────────────────────────────────────────────────

// Deterministic shuffle (seeded) so runs are reproducible.
function seededShuffle(arr, seed) {
  const a = arr.slice();
  let s = seed;
  for (let i = a.length - 1; i > 0; i--) {
    s = (s * 9301 + 49297) % 233280;
    const j = Math.floor((s / 233280) * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Parse a LongMemEval date "2023/05/30 (Tue) 23:40" → epoch ms.
function parseLmeDate(s) {
  if (!s) return null;
  const norm = String(s).replace(/\//g, '-').replace(/\s*\([^)]*\)\s*/, ' ').trim();
  const t = new Date(norm).getTime();
  return Number.isNaN(t) ? null : t;
}

// Load checkpoint → Map<qid, {h5,h10,type}>
function loadCkpt(path) {
  const m = new Map();
  if (!existsSync(path)) return m;
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const r = JSON.parse(line);
      m.set(r.qid, r);
    } catch {}
  }
  return m;
}

function appendCkpt(path, record) {
  mkdirSync(CKPT_DIR, { recursive: true });
  // append
  const existing = existsSync(path) ? readFileSync(path, 'utf8') : '';
  writeFileSync(path, existing + JSON.stringify(record) + '\n');
}

// ─── Per-question evaluation ────────────────────────────────────────────────

/**
 * Ingest haystack + retrieve candidates for one question.
 * Returns { candidates: [{id, match, node}], answerIds, questionDate }.
 * candidates are decay-independent (match only); caller re-ranks per strategy.
 */
async function ingestAndRetrieve(item, qi, MemoryEngine) {
  const ns = join(ROOT, `.ab-decay`, `ns-${process.pid}-${qi}`);
  if (existsSync(ns)) rmSync(ns, { recursive: true });
  mkdirSync(ns, { recursive: true });

  const engine = new MemoryEngine({ basePath: ns });
  await engine.init();
  engine.config.dedupThreshold = 1.01; // disable dedup so raw sessions persist

  const questionDateMs = parseLmeDate(item.question_date);
  const sessionIds = item.haystack_session_ids;
  const answerIds = item.answer_session_ids;

  // Ingest each session. Back-fill the historical date into lastAccess/created
  // so decay sees real memory age (stock add() stamps them "today").
  for (let si = 0; si < item.haystack_sessions.length; si++) {
    const turns = item.haystack_sessions[si];
    const origId = sessionIds[si];
    const date = item.haystack_dates ? item.haystack_dates[si] : '';
    const isAnswer = answerIds.includes(origId);
    const body = turns.map(t => `${t.role === 'user' ? 'User' : 'AI'}: ${t.content}`).join('\n');
    const r = await engine.add({
      type: 'conversation',
      summary: `LME ${origId}${isAnswer ? ' [ANSWER]' : ''}`,
      conversation: body,
      timestamp: date,
      author: 'lme',
    });
    // Back-fill historical timestamps onto the stored node.
    const node = engine._articles.get(r.id);
    if (node && date) {
      const dIso = String(date).replace(/\//g, '-').replace(/\s*\([^)]*\)\s*/, ' ').trim();
      node.lastAccess = dIso.slice(0, 10);
      node.created = dIso.slice(0, 10);
    }
  }
  await engine._flush();

  // Retrieve. land() sorts by match*salience using grepmem's flat salience,
  // but we don't care about its ordering — we want the raw match scores to
  // re-rank ourselves. We call land() to get candidates above threshold, then
  // read each node for re-ranking.
  // Set GREPMEM_NOW so any internal salience is relative to the question date.
  if (questionDateMs) process.env.GREPMEM_NOW = item.question_date;
  const grepmemResults = await engine.land(item.question, 0, 'conversation');

  // Collect (id, match, node) — match is decay-independent.
  const candidates = [];
  for (const r of grepmemResults) {
    const node = engine._articles.get(r.id);
    if (!node) continue;
    // Recover original session id from the summary "LME <id>".
    const m = r.summary?.match(/LME (\S+)/);
    const origId = m ? m[1].replace('[ANSWER]', '') : null;
    candidates.push({ id: r.id, origId, match: r.match, node });
  }

  // Cleanup namespace.
  rmSync(ns, { recursive: true });
  delete process.env.GREPMEM_NOW;

  return { candidates, answerIds, questionDate: item.question_date };
}

/**
 * Re-rank candidates with a salience function and compute Hit@K.
 * @returns {{h5:boolean, h10:boolean, top5:string[], top10:string[]}}
 */
function rankAndHit(candidates, salFn, questionDate, answerIds) {
  if (questionDate) process.env.GREPMEM_NOW = questionDate;
  // Clone nodes shallowly so strategy-specific _lnn state doesn't leak across.
  const ranked = candidates.map(c => ({
    origId: c.origId,
    score: c.match * salFn(c.node),
  }));
  delete process.env.GREPMEM_NOW;
  ranked.sort((a, b) => b.score - a.score);
  const top5 = ranked.slice(0, 5).map(r => r.origId);
  const top10 = ranked.slice(0, 10).map(r => r.origId);
  const h5 = top5.some(id => id && answerIds.includes(id));
  const h10 = top10.some(id => id && answerIds.includes(id));
  return { h5, h10, top5, top10 };
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  const { MemoryEngine } = await loadGrepmem();

  if (!existsSync(DATA_PATH)) {
    console.error(`Dataset not found: ${DATA_PATH}`);
    console.error('Set DATA_PATH or ensure ../ai-memory/data/longmemeval_s_cleaned.json exists.');
    process.exit(1);
  }
  console.log(`Loading ${DATA_PATH} ...`);
  const data = JSON.parse(readFileSync(DATA_PATH, 'utf8'));

  let sampled = TYPES ? data.filter(d => TYPES.includes(d.question_type)) : data.slice();
  // Drop abstention questions (no ground truth).
  sampled = sampled.filter(d => !d.question_id.endsWith('_abs'));
  sampled = seededShuffle(sampled, SEED).slice(0, LIMIT);
  console.log(`Evaluating ${sampled.length} questions (TYPES=${TYPES || 'all'}, LIMIT=${LIMIT})`);

  if (!RESUME) {
    for (const p of [CKPT_FLAT, CKPT_LNN]) if (existsSync(p)) rmSync(p);
  }
  const ckFlat = loadCkpt(CKPT_FLAT);
  const ckLnn = loadCkpt(CKPT_LNN);

  const perType = { flat: {}, lnn: {} };
  const initType = (store, type) => {
    store[type] ||= { hit5: 0, hit10: 0, total: 0 };
  };

  for (let qi = 0; qi < sampled.length; qi++) {
    const item = sampled[qi];
    const qid = item.question_id;
    const type = item.question_type;
    initType(perType.flat, type);
    initType(perType.lnn, type);

    // Skip if both strategies already have a checkpoint for this qid.
    const haveFlat = ckFlat.has(qid);
    const haveLnn = ckLnn.has(qid);
    if (haveFlat && haveLnn) {
      const rf = ckFlat.get(qid), rl = ckLnn.get(qid);
      perType.flat[rf.type].hit5 += rf.h5 ? 1 : 0;
      perType.flat[rf.type].hit10 += rf.h10 ? 1 : 0;
      perType.flat[rf.type].total += 1;
      perType.lnn[rl.type].hit5 += rl.h5 ? 1 : 0;
      perType.lnn[rl.type].hit10 += rl.h10 ? 1 : 0;
      perType.lnn[rl.type].total += 1;
      continue;
    }

    const { candidates, answerIds, questionDate } = await ingestAndRetrieve(item, qi, MemoryEngine);

    // Flat strategy
    if (!haveFlat) {
      const { h5, h10 } = rankAndHit(candidates, flatSalience, questionDate, answerIds);
      const rec = { qid, type, h5, h10 };
      appendCkpt(CKPT_FLAT, rec);
      ckFlat.set(qid, rec);
      perType.flat[type].hit5 += h5 ? 1 : 0;
      perType.flat[type].hit10 += h10 ? 1 : 0;
    } else {
      const r = ckFlat.get(qid);
      perType.flat[type].hit5 += r.h5 ? 1 : 0;
      perType.flat[type].hit10 += r.h10 ? 1 : 0;
    }
    perType.flat[type].total += 1;

    // LNN strategy
    if (!haveLnn) {
      const { h5, h10 } = rankAndHit(candidates, lnnSalience, questionDate, answerIds);
      const rec = { qid, type, h5, h10 };
      appendCkpt(CKPT_LNN, rec);
      ckLnn.set(qid, rec);
      perType.lnn[type].hit5 += h5 ? 1 : 0;
      perType.lnn[type].hit10 += h10 ? 1 : 0;
    } else {
      const r = ckLnn.get(qid);
      perType.lnn[type].hit5 += r.h5 ? 1 : 0;
      perType.lnn[type].hit10 += r.h10 ? 1 : 0;
    }
    perType.lnn[type].total += 1;

    const rf = ckFlat.get(qid), rl = ckLnn.get(qid);
    const mark = (rf.h5 === rl.h5) ? '=' : (rl.h5 && !rf.h5 ? 'LNN↑' : (!rl.h5 && rf.h5 ? 'LNN↓' : '='));
    process.stdout.write(`${mark}`);
    if ((qi + 1) % 25 === 0) {
      console.log(`  [${qi + 1}/${sampled.length}] ${flatR5(perType.flat).toFixed(1)}% vs ${flatR5(perType.lnn).toFixed(1)}% (flat vs lnn R@5)`);
    }
  }
  console.log('');

  report(perType);
}

function flatR5(store) {
  let h = 0, t = 0;
  for (const v of Object.values(store)) { h += v.hit5; t += v.total; }
  return t ? (h / t) * 100 : 0;
}

function report(perType) {
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('  flat-decay vs lnn-decay  (R@5 / R@10)');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  question_type'.padEnd(28) + 'flat R@5/R@10'.padStart(16) + 'lnn R@5/R@10'.padStart(16) + 'ΔR@5'.padStart(8));
  console.log('  ' + '─'.repeat(66));

  const types = [...new Set([...Object.keys(perType.flat), ...Object.keys(perType.lnn)])].sort();
  let tH5f = 0, tH10f = 0, tTf = 0, tH5l = 0, tH10l = 0, tTl = 0;
  for (const type of types) {
    const f = perType.flat[type] || { hit5: 0, hit10: 0, total: 0 };
    const l = perType.lnn[type] || { hit5: 0, hit10: 0, total: 0 };
    const r5f = f.total ? (f.hit5 / f.total) * 100 : 0;
    const r10f = f.total ? (f.hit10 / f.total) * 100 : 0;
    const r5l = l.total ? (l.hit5 / l.total) * 100 : 0;
    const r10l = l.total ? (l.hit10 / l.total) * 100 : 0;
    tH5f += f.hit5; tH10f += f.hit10; tTf += f.total;
    tH5l += l.hit5; tH10l += l.hit10; tTl += l.total;
    const d = (r5l - r5f).toFixed(1);
    const sign = d > 0 ? '+' : '';
    console.log(`  ${type.padEnd(28)}${(r5f.toFixed(1) + '/' + r10f.toFixed(1)).padStart(16)}${(r5l.toFixed(1) + '/' + r10l.toFixed(1)).padStart(16)}${(sign + d).padStart(8)}`);
  }
  console.log('  ' + '─'.repeat(66));
  const R5f = tTf ? (tH5f / tTf) * 100 : 0;
  const R10f = tTf ? (tH10f / tTf) * 100 : 0;
  const R5l = tTl ? (tH5l / tTl) * 100 : 0;
  const R10l = tTl ? (tH10l / tTl) * 100 : 0;
  const dR5 = (R5l - R5f).toFixed(1);
  const sign = dR5 > 0 ? '+' : '';
  console.log(`  ${'OVERALL'.padEnd(28)}${(R5f.toFixed(1) + '/' + R10f.toFixed(1)).padStart(16)}${(R5l.toFixed(1) + '/' + R10l.toFixed(1)).padStart(16)}${(sign + dR5).padStart(8)}`);
  console.log('═══════════════════════════════════════════════════════════════');

  // Verdict
  console.log('\nVerdict (per plan §四):');
  console.log(`  Overall ΔR@5 = ${sign}${dR5}pt  (pass if ≥ -0.5pt)`);
  const tempKu = ['temporal-reasoning', 'knowledge-update'];
  let tf5 = 0, tfT = 0, tl5 = 0, tlT = 0;
  for (const t of tempKu) {
    const f = perType.flat[t] || { hit5: 0, total: 0 };
    const l = perType.lnn[t] || { hit5: 0, total: 0 };
    tf5 += f.hit5; tfT += f.total; tl5 += l.hit5; tlT += l.total;
  }
  const tempR5f = tfT ? (tf5 / tfT) * 100 : 0;
  const tempR5l = tlT ? (tl5 / tlT) * 100 : 0;
  const dTemp = (tempR5l - tempR5f).toFixed(1);
  console.log(`  temporal+knowledge ΔR@5 = ${dTemp > 0 ? '+' : ''}${dTemp}pt  (expect positive — LNN's time-awareness zone)`);
}

main().catch(e => { console.error(e); process.exit(1); });
