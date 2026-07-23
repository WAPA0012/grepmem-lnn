/**
 * Diagnostic: dump the match + flat/lnn salience distribution for one
 * temporal-reasoning question, to understand WHY lnn doesn't beat flat.
 */
import { loadGrepmem, GREPMEM_ROOT } from '../lib/grepmem-shim.js';
import { flatSalience, lnnSalience } from '../lib/decay-comparator.js';
import { daysSince } from '../lib/env.js';
import { readFileSync, existsSync, mkdirSync, rmSync } from 'fs';
import { join } from 'path';

const { MemoryEngine } = await loadGrepmem();
const data = JSON.parse(readFileSync(join(GREPMEM_ROOT, 'data', 'longmemeval_s_cleaned.json'), 'utf8'));
const item = data.find(d => d.question_type === 'temporal-reasoning');
console.log('Q:', item.question);
console.log('Q date:', item.question_date);
process.env.GREPMEM_NOW = item.question_date;

const ns = join(process.cwd(), '.diag-ns');
if (existsSync(ns)) rmSync(ns, { recursive: true });
mkdirSync(ns, { recursive: true });
const engine = new MemoryEngine({ basePath: ns });
await engine.init();
engine.config.dedupThreshold = 1.01;
for (let si = 0; si < item.haystack_sessions.length; si++) {
  const turns = item.haystack_sessions[si];
  const origId = item.haystack_session_ids[si];
  const date = item.haystack_dates[si];
  const isAns = item.answer_session_ids.includes(origId);
  const body = turns.map(t => `${t.role === 'user' ? 'User' : 'AI'}: ${t.content}`).join('\n');
  const r = await engine.add({ type: 'conversation', summary: `LME ${origId}${isAns ? ' [ANSWER]' : ''}`, conversation: body, timestamp: date, author: 'lme' });
  const node = engine._articles.get(r.id);
  if (node && date) { const d = String(date).replace(/\//g, '-').replace(/\s*\([^)]*\)\s*/, ' ').trim(); node.lastAccess = d.slice(0, 10); node.created = d.slice(0, 10); }
}
await engine._flush();
const res = await engine.land(item.question, 0, 'conversation');
console.log('\nTop 10 candidates:');
console.log('  match  flat   lnn   days  acc  ans  summary');
for (const r of res.slice(0, 10)) {
  const node = engine._articles.get(r.id);
  const d = daysSince(node.lastAccess);
  const f = flatSalience(node), l = lnnSalience(node);
  console.log(`  ${r.match.toFixed(3)}  ${f.toFixed(3)}  ${l.toFixed(3)}  ${String(d).padStart(4)}  ${String(node.accessCount||0).padStart(3)}  ${r.summary.includes('[ANSWER]')?'★':' '}    ${r.summary.slice(0,30)}`);
}
// Also show how re-ranking differs
console.log('\nRe-rank comparison (top5 origId):');
const ranked = res.map(r => {
  const node = engine._articles.get(r.id);
  const m = r.summary?.match(/LME (\S+)/);
  return { origId: m?m[1].replace('[ANSWER]',''):null, match:r.match, node };
});
const byFlat = [...ranked].sort((a,b)=>(b.match*flatSalience(b.node))-(a.match*flatSalience(a.node))).slice(0,5).map(x=>x.origId);
const byLnn = [...ranked].sort((a,b)=>(b.match*lnnSalience(b.node))-(a.match*lnnSalience(a.node))).slice(0,5).map(x=>x.origId);
console.log('  flat top5:', byFlat);
console.log('  lnn  top5:', byLnn);
console.log('  answer   :', item.answer_session_ids);
rmSync(ns, { recursive: true });
