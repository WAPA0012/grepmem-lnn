/**
 * Cumulative-use experiment — the test that actually exercises the LNN core.
 *
 * WHY THIS EXISTS:
 *   The LongMemEval ab-decay result (R@5 flat == R@5 lnn) is real but
 *   structurally uninformative: that benchmark ingests every memory fresh
 *   (accessCount=0) and asks one retrieval question. The LNN's defining
 *   property — per-memory, access-adaptive time constant τ — is never
 *   exercised there. Every memory has identical τ by construction.
 *
 *   This experiment builds the scenario the hypothesis actually predicts
 *   about: a user with a LONG cumulative history where some topics recur
 *   (get retrieved, get reinforced) and others don't. We then ask: days/weeks
 *   later, does the LNN keep the recurring topic MORE retrievable than flat
 *   does? That's the Ebbinghaus-consolidation claim.
 *
 * DESIGN:
 *   Synthetic user history across N days:
 *     - "config" memories (recurs often — repeatedly retrieved/reinforced)
 *     - "oneoff" memories (mentioned once, never revisited)
 *   After a gap of G days, we probe a query that lexically matches BOTH a
 *   config memory and a oneoff memory equally (same match score), but the
 *   config memory is OLDER. Under flat decay, the older config memory may be
 *   outranked by the newer oneoff. Under LNN, the reinforced config memory
 *   has a slower (larger τ) decay and should stay competitive / win.
 *
 *   We sweep the gap G and the reinforcement count, measuring how often lnn
 *   ranks the reinforced-older memory above the fresh-oneoff compared to flat.
 *
 * Output: a table of (gap_days, reinforces) → { flat_keeps_config %, lnn_keeps_config % }.
 */
import { loadGrepmem } from '../lib/grepmem-shim.js';
import { flatSalience, lnnSalience } from '../lib/decay-comparator.js';
import { daysSince } from '../lib/env.js';
import { existsSync, mkdirSync, rmSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

// ─── Scenario builder ───────────────────────────────────────────────────────
//
// We don't even need grepmem's full ingest for the core comparison — the
// question is purely about salience on two nodes with known (age, access)
// profiles. We construct the two nodes directly and compare how flat vs lnn
// rank them. This is a clean unit-level test of the consolidation effect.
//
// Node A: "config" memory. Created `configAgeDays` ago. Retrieved `reinforces`
//         times (most recent retrieval `lastAccessDaysAgo`).
// Node B: "oneoff" memory. Created `oneoffAgeDays` ago. Never retrieved.
// Both have IDENTICAL lexical match score (controlled). The question: at
// probe time, which ranks higher?

function makeNode({ ageDays, accessCount, lastAccessDaysAgo, baseSalience = 0.5 }) {
  const created = new Date(Date.now() - ageDays * 86400000).toISOString().slice(0, 10);
  const lastAccess = new Date(Date.now() - lastAccessDaysAgo * 86400000).toISOString().slice(0, 10);
  return {
    baseSalience,
    accessCount,
    lastAccess,
    created,
    _lnn: null, // lnnSalience will lazy-init
  };
}

/**
 * Simulate the probe: at "now", compare salience of a reinforced-older config
 * memory vs a fresh oneoff memory, both with equal lexical match.
 * @returns {{flatConfigSal, flatOneoffSal, lnnConfigSal, lnnOneoffSal, flatPicks, lnnPicks}}
 */
function probe({ configAgeDays, reinforces, lastAccessDaysAgo, oneoffAgeDays, match = 0.7 }) {
  // Use GREPMEM_NOW = real now (daysSince computes from now by default).
  delete process.env.GREPMEM_NOW;
  const config = makeNode({ ageDays: configAgeDays, accessCount: reinforces, lastAccessDaysAgo });
  const oneoff = makeNode({ ageDays: oneoffAgeDays, accessCount: 0, lastAccessDaysAgo: oneoffAgeDays });

  const flatConfigSal = flatSalience(config);
  const flatOneoffSal = flatSalience(oneoff);
  const lnnConfigSal = lnnSalience({ ...makeNode({ ageDays: configAgeDays, accessCount: reinforces, lastAccessDaysAgo }) });
  const lnnOneoffSal = lnnSalience({ ...makeNode({ ageDays: oneoffAgeDays, accessCount: 0, lastAccessDaysAgo: oneoffAgeDays }) });

  const flatPicks = (match * flatConfigSal) >= (match * flatOneoffSal) ? 'config' : 'oneoff';
  const lnnPicks = (match * lnnConfigSal) >= (match * lnnOneoffSal) ? 'config' : 'oneoff';
  return { flatConfigSal, flatOneoffSal, lnnConfigSal, lnnOneoffSal, flatPicks, lnnPicks };
}

// ─── Sweep ──────────────────────────────────────────────────────────────────

function run() {
  console.log('═══════════════════════════════════════════════════════════════════════');
  console.log('  Cumulative-use consolidation test');
  console.log('  Scenario: a reinforced config memory, last retrieved a while ago,');
  console.log('  vs a fresh never-retrieved oneoff memory — equal lexical match.');
  console.log('  The config memory is IDENTICALLY old for both strategies (same');
  console.log('  lastAccess). Only the decay RATE differs: LNN slows it via τ.');
  console.log('  Does LNN keep the reinforced one ranked higher than flat does?');
  console.log('═══════════════════════════════════════════════════════════════════════\n');

  // CRITICAL design: the config memory is OLD (lastAccess=daysAgo) but
  // REINFORCED; the oneoff memory is FRESH (lastAccess=2 days) and never
  // retrieved. Equal lexical match. This is the adversarial case: does the
  // old-but-reinforced memory SURVIVE against the fresh one? Under flat decay
  // the old memory may bottom out; under LNN its widened τ should let it
  // stay competitive — that's the consolidation hypothesis.
  //
  // NOTE: the access-boost (access*0.03) is applied to BOTH strategies
  // identically (see decay-comparator.js fairness invariant), so the only
  // difference under test is decay SHAPE (linear vs liquid-exponential).
  const headers = ['cfgAge', 'reinforces', 'flat cfg/one', 'lnn cfg/one', 'flat picks', 'lnn picks', 'winner'];
  const widths = [9, 11, 16, 16, 12, 12, 14];
  console.log('  ' + headers.map((h, i) => h.padEnd(widths[i])).join(''));
  console.log('  ' + '─'.repeat(widths.reduce((a, b) => a + b, 0)));

  let flatKeepsConfig = 0, lnnKeepsConfig = 0, total = 0;
  let lnnStrictlyBetter = 0; // lnn picks config where flat picks oneoff
  let flatStrictlyBetter = 0;

  for (const cfgAge of [7, 30, 60, 100, 150, 200]) {
    for (const reinforces of [0, 1, 3, 5, 10]) {
      // config: OLD (lastAccess = cfgAge days ago), reinforced.
      // oneoff: FRESH (lastAccess = 2 days ago), never retrieved.
      const r = probe({
        configAgeDays: cfgAge,
        reinforces,
        lastAccessDaysAgo: cfgAge,      // config memory is stale
        oneoffAgeDays: 2,
        match: 0.7,
      });
      total++;
      if (r.flatPicks === 'config') flatKeepsConfig++;
      if (r.lnnPicks === 'config') lnnKeepsConfig++;
      if (r.lnnPicks === 'config' && r.flatPicks !== 'config') lnnStrictlyBetter++;
      if (r.flatPicks === 'config' && r.lnnPicks !== 'config') flatStrictlyBetter++;

      const winner = r.lnnPicks === r.flatPicks ? '='
        : (r.lnnPicks === 'config' ? 'LNN' : 'FLAT');
      console.log('  ' + [
        String(cfgAge), String(reinforces),
        `${r.flatConfigSal.toFixed(3)}/${r.flatOneoffSal.toFixed(3)}`,
        `${r.lnnConfigSal.toFixed(3)}/${r.lnnOneoffSal.toFixed(3)}`,
        r.flatPicks, r.lnnPicks, winner,
      ].map((c, i) => c.padEnd(widths[i])).join(''));
    }
  }

  const w = widths.reduce((a, b) => a + b, 0);
  console.log('  ' + '─'.repeat(w));
  console.log(`\n  Summary across ${total} (cfgAge × reinforces) combos:`);
  console.log(`    flat keeps old-reinforced-config over fresh-oneoff : ${flatKeepsConfig}/${total} = ${(flatKeepsConfig/total*100).toFixed(0)}%`);
  console.log(`    lnn  keeps old-reinforced-config over fresh-oneoff : ${lnnKeepsConfig}/${total} = ${(lnnKeepsConfig/total*100).toFixed(0)}%`);
  console.log(`    LNN strictly better (saves an old config flat would drop): ${lnnStrictlyBetter}/${total} = ${(lnnStrictlyBetter/total*100).toFixed(0)}%`);
  console.log(`    FLAT strictly better (keeps config lnn would drop): ${flatStrictlyBetter}/${total} = ${(flatStrictlyBetter/total*100).toFixed(0)}%`);

  console.log('\n  Interpretation:');
  console.log('    This is the adversarial consolidation test. An old-but-reinforced');
  console.log('    memory competes against a fresh never-seen one. Flat\'s linear decay');
  console.log('    bottoms old memories at 0.1 fast; LNN\'s gentler exponential (widened');
  console.log('    by reinforcement) lets the consolidated memory survive longer.');
  console.log('    "LNN strictly better" counts cases LNN saves that flat loses.');
  console.log('═'.repeat(w + 2));
}

run();
