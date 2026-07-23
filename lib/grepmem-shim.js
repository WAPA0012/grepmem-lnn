/**
 * Shim that re-exports grepmem's engine from the sibling ai-memory/ project,
 * resolving the path robustly regardless of where the importer lives.
 *
 * grepmem is a sibling project (../ai-memory/), not an npm dependency. We
 * reference it by absolute path so eval scripts under eval/ and tests can both
 * import the same module without ../../ confusion.
 *
 * IMPORTANT: this experiment does NOT modify grepmem. It only imports the
 * MemoryEngine to ingest/retrieve, then re-ranks results with its own
 * decay strategies. grepmem source stays read-only.
 *
 * ESM `export ... from` requires a literal specifier, so we compute the URL
 * string and re-export named bindings via a dynamic re-export.
 */
import { fileURLToPath } from 'url';
import { pathToFileURL } from 'url';
import { dirname, join } from 'path';

const HERE = dirname(fileURLToPath(import.meta.url));  // .../ai-memory-lnn/lib
const GREPMEM_HTML = join(HERE, '..', '..', 'ai-memory', 'memory-html.js');
const GREPMEM_GREP = join(HERE, '..', '..', 'ai-memory', 'grep.js');
const GREPMEM_URL = pathToFileURL(GREPMEM_HTML).href;
const GREPMEM_GREP_URL = pathToFileURL(GREPMEM_GREP).href;

// Static re-export isn't possible with a computed path in ESM, so expose an
// async loader. Callers do:  const { MemoryEngine } = await loadGrepmem();
// For the grep module (extractQueryTerms etc.), use loadGrepmemGrep().
export async function loadGrepmem() {
  const mod = await import(GREPMEM_URL);
  return mod;
}

export async function loadGrepmemGrep() {
  const mod = await import(GREPMEM_GREP_URL);
  return mod;
}

export const GREPMEM_ROOT = join(HERE, '..', '..', 'ai-memory');
