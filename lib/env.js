/**
 * Environment configuration for the decay experiment.
 *
 * Two knobs, both env-driven so the same harness can run flat vs lnn without
 * code changes (mirrors grepmem's env-driven eval scripts):
 *
 *   GREPMEM_DECAY  = 'flat' | 'lnn'   (default 'flat')
 *     Selects the salience decay strategy.
 *
 *   GREPMEM_NOW    = ISO date string | LongMemEval date string
 *     The "current time" used to compute memory age. Defaults to real now().
 *     For benchmarks we set this to each question's question_date so memory
 *     age is measured relative to when the question was asked, not today.
 *     Accepted formats: ISO (2023-05-30T23:40:00Z) and
 *     LongMemEval-style ("2023/05/30 (Tue) 23:40").
 */

export const DECAY_FLAT = 'flat';
export const DECAY_LNN = 'lnn';

export function decayStrategy() {
  const v = (process.env.GREPMEM_DECAY || DECAY_FLAT).toLowerCase();
  return v === DECAY_LNN ? DECAY_LNN : DECAY_FLAT;
}

/**
 * Parse a "now" timestamp from GREPMEM_NOW, falling back to Date.now().
 * Tolerates both ISO and LongMemEval "YYYY/MM/DD (WD) HH:MM" formats.
 * @returns {number} epoch ms
 */
export function nowMs() {
  const raw = process.env.GREPMEM_NOW;
  if (!raw) return Date.now();
  // ISO first
  const iso = new Date(raw).getTime();
  if (!Number.isNaN(iso)) return iso;
  // LongMemEval style: strip the "(Tue)" weekday, turn "/" into "-"
  const stripped = raw.replace(/\s*\([^)]*\)\s*/, ' ').replace(/\//g, '-').trim();
  const t = new Date(stripped).getTime();
  return Number.isNaN(t) ? Date.now() : t;
}

/**
 * Days elapsed since `fromIso` until GREPMEM_NOW (or real now).
 * `fromIso` is a grepmem node's lastAccess/created (ISO date or the
 * LongMemEval haystack_dates format).
 * @param {string} fromIso
 * @returns {number} days, ≥ 0
 */
export function daysSince(fromIso) {
  if (!fromIso) return 0;
  // grepmem stores lastAccess as ISO slice(0,10) e.g. "2023-05-30";
  // haystack_dates is "2023/05/30 (Tue) 02:21". Handle both.
  const norm = String(fromIso).replace(/\//g, '-').replace(/\s*\([^)]*\)\s*/, ' ').trim();
  const from = new Date(norm).getTime();
  if (Number.isNaN(from)) return 0;
  return Math.max(0, (nowMs() - from) / 86400000);
}
