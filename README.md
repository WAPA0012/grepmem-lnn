# grepmem-lnn

> Liquid Neural Network time-decay for [grepmem](https://github.com/WAPA0012/grepmem) salience.
> Replacing grepmem's flat global linear decay with CfC-inspired continuous-time dynamics where each memory has its own reinforcement-adaptive time constant τ.

**Status: experiment complete. Results below are honest, including the negative one.**

This is research code. It does **not** modify grepmem — it imports grepmem's engine read-only and re-ranks retrieval results with its own decay strategies, so flat vs liquid can be compared cleanly.

---

## The hypothesis

grepmem scores memory relevance as `match × effectiveSalience`, where salience decay is currently **flat and global** (`grep.js:effectiveSalience`):

```js
effectiveSalience = clamp(base + accessCount*0.03 - days*0.005, 0.1, 1.0)
```

Every memory decays at the same fixed linear rate (`0.005/day`), regardless of how often it's used. grepmem's author flags this in `DESIGN.md:424`:

> *Salience decay is flat. All memories decay at the same rate regardless of how often they're used. Planned: Ebbinghaus-style consolidation where frequently retrieved/cited memories decay slower.*

This project tests whether a **Liquid Neural Network (CfC)** formulation — where each memory carries a per-memory, reinforcement-adaptive time constant τ — realizes that planned consolidation.

## The math

Based on the Closed-form Continuous-time model ([Hasani et al., Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00556-7)), reduced to the pure-decay case (no input signal):

```
x(t) = A + (x₀ − A) · exp(−w_tau · t)       τ_eff = 1/w_tau
```

The "liquid" property: `w_tau` is **per-memory and grows under reinforcement** (each retrieval shrinks `w_tau` with diminishing returns → widens `τ_eff` → slower future decay). Unlike the flat model's single global `0.005`, a frequently-retrieved memory's `τ_eff` increases over its lifetime. No ODE solver required — CfC's defining feature is its closed-form solution. Pure JS, ~60 lines.

## Verification (correctness — all passing)

| Check | Method | Result |
|---|---|---|
| **LNN math = analytical closed form** | `test-decay.mjs`: 13 unit tests vs independently-computed `A+(x0-A)·exp(-wτ·t)` | ✅ error < 1e-9 |
| **JS = Python** | `test-equivalence.mjs`: JS vs Python re-derivation of the same closed form | ✅ rel err < 1% |
| **ncps family membership** | drive a real `ncps`/torch CfC layer with zero input | ✅ runs (corroboration, not numerical twin — see note below) |

```bash
npm test   # 16/16 pass
```

> **Honest note on the ncps check.** An ncps `CfC` layer is a *general* CfC instance with random weights and the full synaptic `f·(A−x)` term active. Its zero-input trajectory therefore does **not** numerically track our scalar pure-decay reduction — the internal weights drive `f ≠ 0` even with no external input. The load-bearing correctness proof is JS == analytical-closed-form (verified to 1e-9). The ncps run only corroborates that the reduction is a legitimate special case of the CfC family, qualitatively. I deliberately did **not** overclaim "JS == real LNN numerically."

## Results

### Experiment 1 — LongMemEval-S retrieval benchmark (the *negative* result)

Ran grepmem's own retrieval benchmark (`eval/ab-decay.mjs`) with both decay strategies, 50 questions sampled from the two most time-sensitive categories. Historical `haystack_dates` back-filled into memory age so decay actually fires.

| question_type | flat R@5/R@10 | lnn R@5/R@10 | ΔR@5 |
|---|---|---|---|
| knowledge-update | 89.5 / 94.7 | 89.5 / 89.5 | 0.0 |
| temporal-reasoning | 90.3 / 93.5 | 90.3 / 93.5 | 0.0 |
| **OVERALL** | **90.0 / 94.0** | **90.0 / 92.0** | **0.0** |

**ΔR@5 = 0.0pt. The LNN's core property is not exercised here.** This benchmark ingests every memory fresh (`accessCount=0`) and asks one retrieval question — so every memory has identical `τ` by construction. There is no cumulative reinforcement for the consolidation mechanism to act on. Lexical `match` dominates ranking; salience only matters in near-ties, which both strategies resolve similarly. The small R@10 dip on knowledge-update reflects LNN's gentler exponential decay keeping some stale non-answer memories ranked slightly higher.

**This is a real, honest null result for this benchmark**, not a failure of the implementation.

### Experiment 2 — Adversarial consolidation test (the *positive* result)

Since LongMemEval structurally cannot test the hypothesis, `eval/cumulative-use.mjs` builds the scenario the hypothesis actually predicts about: an **old-but-reinforced** memory competing against a **fresh never-seen** memory, equal lexical match. Does reinforcement let the old memory survive?

Sweeping config-memory age (7–200 days stale) × reinforcement count (0–10 retrievals):

| metric | flat | lnn |
|---|---|---|
| keeps old-reinforced memory over fresh-oneoff | **23%** (7/30) | **47%** (14/30) |
| strictly better than the other strategy | 0/30 | **7/30 (23%)** |

**LNN wins 23% of cases flat loses, and loses 0%.** The win zone is exactly the predicted one: mid-age memories (30–100 days) with moderate reinforcement (3–10 retrievals), where flat's linear decay has bottomed at `0.1` but LNN's consolidation keeps the memory at `0.45–0.67` — enough to outrank a fresh `0.495` oneoff.

This is the Ebbinghaus-consolidation effect, demonstrated.

```
Sample row (cfgAge=60, reinforces=5):
  flat: 0.348 / 0.488  → picks oneoff (old memory too decayed)
  lnn:  0.567 / 0.495  → picks config (consolidation saved it)   ← LNN wins
```

## What this means

- **LNN decay is mathematically sound** (verified to 1e-9 against the closed form, and to <1% against an independent Python re-derivation).
- **It does NOT help on single-shot retrieval benchmarks** like LongMemEval-S — those can't exercise per-memory reinforcement. Don't expect gains there.
- **It DOES help in cumulative-use scenarios** where memories recur and get reinforced: the consolidation keeps important old memories retrievable ~2× more often than flat decay, with zero regressions. This is the use case grepmem's roadmap describes (multi-year conversation, frequently-used config surviving).

## What's here

| Path | Purpose |
|---|---|
| `lib/lnn-decay.js` | Core: JS CfC liquid decay + reinforcement |
| `lib/decay-comparator.js` | Unified `(node) → salience` for `flat` / `lnn` (fairness-invariant) |
| `lib/env.js` | `GREPMEM_DECAY` / `GREPMEM_NOW` |
| `lib/grepmem-shim.js` | Imports grepmem engine from sibling `../ai-memory/` |
| `eval/test-decay.mjs` | 13 unit tests: LNN decay vs analytical solution |
| `eval/test-equivalence.mjs` | JS vs Python numerical equivalence |
| `eval/ab-decay.mjs` | Exp 1: flat vs lnn on LongMemEval-S R@5/R@10 |
| `eval/cumulative-use.mjs` | Exp 2: adversarial consolidation sweep |
| `eval/diag-salience.mjs` | Single-question salience dump (debugging) |
| `python/cfc_reference.py` | Reference CfC via `ncps`/torch |

## Run it

```bash
# Correctness
npm test

# Exp 1 (needs ../ai-memory/data/longmemeval_s_cleaned.json)
LIMIT=50 TYPES=temporal-reasoning,knowledge-update node eval/ab-decay.mjs

# Exp 2 (self-contained, no dataset)
node eval/cumulative-use.mjs
```

## Fairness invariant

To make flat-vs-lnn a clean test of *decay shape* (not access-boost), both strategies apply the identical `base + access*0.03` boost to the starting level. The LNN's distinct contribution is confined to the decay term (per-memory, access-adaptive `w_tau`). See `lib/decay-comparator.js`.

## References

- [Closed-form continuous-time neural networks — Hasani et al., Nature MI 2022](https://www.nature.com/articles/s42256-022-00556-7)
- [ncps reference implementation (Lechner/Hasani)](https://github.com/mlech26l/ncps)
- [grepmem](https://github.com/WAPA0012/grepmem) — the baseline

## License

MIT
