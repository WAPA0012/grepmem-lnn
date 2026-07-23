/**
 * Focus classifier — extracts a lexical input signal vector I from a query,
 * which the CfC state scheduler consumes to update its focus distribution.
 *
 * NO model, NO training. Pure lexical feature counts. This is the "sensors"
 * feeding the CfC dynamics. The CfC's job is to integrate these signals over
 * time with adaptive time constants; THIS module just turns a query string
 * into a fixed-length real vector.
 *
 * The 4 focus modes (must stay in sync with state-scheduler.js FOCUS_MODES):
 *   0 exact-entity      — precise identifiers: IPs, ports, version numbers,
 *                         exact names ("10.0.0.5", "Redis", "port 6379")
 *   1 procedure-debug   — troubleshooting flow: error words, failure signals,
 *                         "how do I", verbs of fixing ("ConnectionRefused",
 *                         "报错", "怎么修")
 *   2 config-param      — configuration keys/values: env vars, settings,
 *                         key=value patterns ("timeout", "max_connections")
 *   3 narrative-context — open-ended recall: natural language, questions about
 *                         past events ("what did we discuss", "上周聊的")
 *
 * Output: I ∈ ℝ⁴, one salience contribution per focus mode, non-negative.
 * The CfC normalizes these into the focus simplex via softmax internally.
 */

// Lexical cues per focus mode. Bilingual (en + zh) where grepmem's synonym
// map is bilingual. Keep these deliberately simple and auditable.
const CUES = {
  // exact-entity: identifiers, addresses, version strings
  0: {
    re: [
      /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b/g, // IP[:port]
      /\bport\s+\d{2,5}\b/gi,
      /\bv?\d+\.\d+\.\d+\b/g,                           // versions 1.2.3
      /#?[A-Z]{2,}-\d{2,}/g,                            // PROJ-123 tickets
    ],
    words: [],
  },
  // procedure-debug: error / fix / failure vocabulary
  1: {
    re: [
      /\b(error|exception|fail(ed|ure)?|crash(ed)?|refused|timeout|denied|invalid|undefined|null\s+pointer|traceback|stacktrace)\b/gi,
      /(报错|失败|崩溃|连不上|挂了|异常|超时|拒绝|无效)/g,
      /\b(how\s+(do|to)|怎么|如何|fix|repair|debug|troubleshoot|resolve)\b/gi,
    ],
    words: [],
  },
  // config-param: keys / settings / env vars
  2: {
    re: [
      /[A-Z][A-Z0-9_]{2,}=/g,            // ENV_VAR=
      /\b[A-Za-z_][A-Za-z0-9_]*\s*[:=]\s*\S+/g, // key: value / key = value
      /\.(env|yml|yaml|json|toml|conf|ini|config)\b/gi,
    ],
    words: [
      'timeout', 'max_connections', 'pool', 'buffer', 'cache', 'ttl',
      'retries', 'limit', 'threshold', 'interval', 'port', 'host',
      '配置', '参数', '设置', '阈值', '上限',
    ],
  },
  // narrative-context: recall / temporal / open-ended
  3: {
    re: [
      /\b(what|when|why|who|which|where)\b/gi,
      /\b(discuss|mentioned|said|told|last\s+(week|time)|yesterday|before|previous|earlier|remember)\b/gi,
      /(什么|什么时候|为什么|谁|哪个|哪|聊过|说过|提到|上周|昨天|之前|记得|之前讨论)/g,
    ],
    words: [],
  },
};

const N_MODES = 4;

/**
 * Extract the input signal vector I ∈ ℝ⁴ from a query.
 * Each component is a non-negative salience for that focus mode, derived
 * from how strongly the query's lexical features match that mode's cues.
 * @param {string} query
 * @returns {number[]} length-4 non-negative vector
 */
export function extractFocusSignal(query) {
  if (!query || typeof query !== 'string') return [0, 0, 0, 0];
  const q = query;
  const I = [0, 0, 0, 0];
  for (let mode = 0; mode < N_MODES; mode++) {
    const cues = CUES[mode];
    let score = 0;
    for (const re of cues.re) {
      const m = q.match(re);
      if (m) score += m.length;
    }
    for (const w of cues.words) {
      const re = new RegExp(`\\b${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
      const m = q.match(re);
      if (m) score += m.length;
    }
    I[mode] = score;
  }
  return I;
}

/**
 * The argmax focus mode for a signal — used for diagnostics and for the
 * focus-match vs topic-switch stratification in experiments.
 */
export function dominantFocus(I) {
  let best = 0;
  for (let i = 1; i < N_MODES; i++) if (I[i] > I[best]) best = i;
  return best;
}

export const FOCUS_MODES = ['exact-entity', 'procedure-debug', 'config-param', 'narrative-context'];
export const N_FOCUS = N_MODES;
