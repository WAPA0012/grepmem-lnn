# grepmem-lnn

> Can a Liquid Neural Network improve [grepmem](https://github.com/WAPA0012/grepmem)'s memory? Three rounds of experiments, each with a number-backed verdict — including the negatives.

**Status: complete.** This is a self-contained research project that sits alongside grepmem (read-only — it never modifies grepmem source). It asks one question three ways: *does LNN continuous-time dynamics buy grepmem anything?*

---

## TL;DR — what we learned

| Round | Idea | Verdict | Net |
|---|---|---|---|
| **R1** | Replace flat linear salience decay with CfC liquid decay (per-memory, reinforcement-adaptive τ) | Works in **cumulative-use** (+24pt consolidation), but **regresses** on single-shot benchmarks (R@10 −2pt) | ✅ useful, must be opt-in |
| **R2** | Add a coupled-CfC session-state scheduler that re-weights retrieval by accumulated "focus" | Mechanism works in controlled ties (+100pt), but **~0pt on real multi-session QA** | ❌ lexical regime can't use it |
| **R3** | Train the focus channel to reach its potential | Oracle proves a **+74.9pt ceiling exists**, but it's **semantic, not lexical** — unreachable without embeddings | ◑ ceiling real, out of reach |

**One-line conclusion:** LNN time-decay is a real, usable improvement *in one specific scenario* (long-term consolidation of frequently-used memories). State-aware retrieval has a provably large ceiling but it requires semantic understanding, which grepmem deliberately lacks. Everything below is backed by reproducible numbers, including where LNN makes things worse.

---

## Background: grepmem and the gap this targets

grepmem scores memory relevance as `match × effectiveSalience`. The salience decay is currently **flat and global** (`grep.js:effectiveSalience`):

```js
effectiveSalience = clamp(base + accessCount*0.03 - days*0.005, 0.1, 1.0)
```

Every memory decays at the same fixed linear rate (`0.005/day`, ~200-day half-life), regardless of how often it's used. grepmem's author flags this in `DESIGN.md:424` as a known limitation:

> *Salience decay is flat. All memories decay at the same rate regardless of how often they're used. Planned: Ebbinghaus-style consolidation where frequently retrieved/cited memories decay slower.*

R1 implements exactly that planned feature and tests it. R2/R3 explore an adjacent idea (state-aware retrieval) and establish where it does and doesn't help.

---

## The LNN math (used in all three rounds)

Based on the Closed-form Continuous-time model ([Hasani et al., Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00556-7)), reduced to the pure-decay case (no input signal):

```
x(t) = A + (x₀ − A) · exp(−w_tau · t)       τ_eff = 1/w_tau
```

The "liquid" property: `w_tau` is **per-memory and grows under reinforcement** (each retrieval shrinks `w_tau` with diminishing returns → widens `τ_eff` → slower future decay). No ODE solver required — CfC's defining feature is its closed-form solution. Pure JS, ~60 lines.

**Correctness verification (all passing):**
- 13 unit tests vs the analytical closed form: error < 1e-9
- JS vs an independent Python re-derivation: relative error < 1%
- `ncps`/torch CfC family-membership corroboration (qualitative, not numerical twin — see code comments for the honest caveat)

---

## Round 1 — per-memory liquid time-decay

**Idea:** replace `- days*0.005` with the CfC decay above, where each memory has its own `w_tau` that shrinks (slower decay) on each retrieval. This is the Ebbinghaus-consolidation effect.

### Experiment 1a — LongMemEval-S retrieval benchmark (the *negative*)

Ran grepmem's own retrieval benchmark with both decay strategies, 50 questions from the two most time-sensitive categories, historical `haystack_dates` back-filled so decay actually fires:

| question_type | flat R@5/R@10 | lnn R@5/R@10 | ΔR@5 | ΔR@10 |
|---|---|---|---|---|
| knowledge-update | 89.5 / 94.7 | 89.5 / 89.5 | 0.0 | **−5.2** |
| temporal-reasoning | 90.3 / 93.5 | 90.3 / 93.5 | 0.0 | 0.0 |
| **OVERALL** | **90.0 / 94.0** | **90.0 / 92.0** | **0.0** | **−2.0** |

R@5 is unchanged; **R@10 regresses 2pt (5.2pt on knowledge-update)**. Cause: LNN's gentler exponential decay keeps some stale *non-answer* memories ranked higher, crowding answers out of top-10. **This is why LNN decay must be opt-in, not a default replacement.**

### Experiment 1b — cumulative-use consolidation (the *positive*)

LongMemEval can't exercise the hypothesis (it ingests memories fresh, `accessCount=0`, so per-memory τ is never used). So `cumulative-use.mjs` builds the scenario the hypothesis predicts about: an **old-but-reinforced** memory vs a **fresh never-seen** memory, equal lexical match. Does reinforcement let the old memory survive?

Sweeping config-memory age (7–200 days stale) × reinforcement count (0–10 retrievals), 30 combos:

| metric | flat | lnn |
|---|---|---|
| keeps old-reinforced memory over fresh-oneoff | **23%** (7/30) | **47%** (14/30) |
| strictly better than the other strategy | **0/30** | **7/30 (23%)** |

**LNN wins 23% of cases flat loses, and loses 0%.** The win zone: mid-age memories (30–100 days) with moderate reinforcement (3–10 retrievals), where flat's linear decay has bottomed at 0.1 but LNN's consolidation keeps the memory at 0.45–0.67 — enough to outrank a fresh 0.495 oneoff. The Ebbinghaus-consolidation effect, demonstrated.

### What R1 means

- **Useful in exactly the cumulative-use scenario** grepmem's roadmap describes (multi-year conversation, frequently-used config surviving): +24pt consolidation, zero regressions there.
- **Harmful as a default** on single-shot benchmarks (R@10 −2pt). Correct integration is opt-in (`GREPMEM_DECAY=lnn`), with flat as the default — which is how `lib/decay-comparator.js` is already wired.

---

## Round 2 — state-aware retrieval scheduling

**Idea:** maintain a coupled-CfC session-state vector `z = (z_focus, z_conf, focusLex)` across turns, and re-weight which retrieval terms get amplified based on "what the agent is currently focused on". The confidence channel modulates the focus time constant τ (high confidence → inertia; low → fast re-orientation) — the cross-channel coupling a per-channel scalar decay cannot reproduce.

### The state

- **z_focus ∈ Δ⁴** — soft distribution over `exact-entity` / `procedure-debug` / `config-param` / `narrative-context`, each favoring different term types.
- **z_conf ∈ [0,1]** — confidence, modulates focus τ (**the load-bearing coupling**).
- **focusLex** — a CfC-decayed bag of words from recent turns (the channel that fires on English queries, since term-type boosts like `synonym`/`learned` only arise via the bilingual synonym map).

Readout: a hand-designed 4×9 boost table × confidence gate. No training data used — the LNN's *dynamics*, not learned representations.

### Tier A — constructed multi-turn scenario (positive)

A reinforced old-but-topic-relevant memory vs a fresh oneoff, equal lexical match; state-awareness breaks the tie via topic continuity. 300 scenarios:

| strategy | picks-correct-answer% |
|---|---|
| flat (no state) | **0%** (genuine lexical tie → loses) |
| scalar-focus (fixed τ) | **100%** |
| lnn-coupled (full) | **100%** |

The mechanism works (+100pt vs flat). **But scalar-focus = lnn-coupled = 100%** — the lexical-focus bag did all the work; the coupling added nothing in this short deterministic setting.

### Tier B — LoCoMo real multi-session data (negative)

Full dataset, 1982 evidence-filtered QA across 10 conversations (up to 35 sessions each), any-hit R@5:

| strategy | R@5 overall | early(1-5) | mid(6-15) | late(16+) |
|---|---|---|---|---|
| flat (no state) | **28.6%** | 24% | 30% | 29% |
| scalar-focus (fixed τ) | **28.9%** | 23% | 29% | 31% |
| lnn-coupled (full) | **28.8%** | 24% | 29% | 30% |

- **flat → coupled: +0.2pt** — effectively zero, within noise.
- **scalar → coupled: −0.2pt** — coupling doesn't help on long horizons.
- **late-span (the predicted coupling win zone): −0.6pt** — coupling slightly *hurts*.

### What R2 means

State-awareness is real and fires in controlled tie-breaking, but on real conversational QA it provides no measurable benefit, and **the coupled CfC adds nothing over a simple scalar-decay focus bag**. LoCoMo's QA are external-evaluator questions about the conversation, not in-conversation focus-driven follow-ups, so the accumulated state is a weak proxy for what the probe needs.

---

## Round 3 — training the focus channel (and the ceiling it hits)

R2's ~0pt could be a fixable focus-signal problem (training could help) or a lexical ceiling (training is futile). R3 settles this with a **gated approach**: diagnose before training.

### Step 0 — bottleneck diagnostic (the go/no-go gate)

`diag-bottleneck.mjs` measures the R@5 *ceiling* if the focus channel had perfect information (cheating by giving known evidence turns a direct bonus), on 3 LoCoMo conversations:

| variant | R@5 | meaning |
|---|---|---|
| A. flat (no state) | **24.8%** | baseline |
| B. **oracle-focus-bonus** (perfect topic ID) | **99.8%** | the ceiling |
| C. uniform boost sweep (any magnitude) | 24.8% | readout magnitude is irrelevant |
| D. real boosts, no focus bag | 25.9% | the hand-designed boost table does nothing |

**B − A = +74.9pt.** This decisively *rejects* the lexical-ceiling hypothesis: a focus channel that could accurately identify on-topic candidates has enormous headroom. Training is worth attempting. It also localized the bottleneck: readout magnitude is irrelevant (C ≈ A); the problem is *which candidates the focus signal points at*.

### Step 1 — training word weights (two attempts, both fall short)

`train-readout.mjs`, honest weak-supervision feature weighting (not deep learning):

**Cross-conversation log-odds weights** (train 6 convs, eval 4 held-out): learn per-word weights from evidence-vs-non-evidence turns. Result: **+2.7pt** (29.3% → 31.9%) — but captures only **4% of the oracle headroom**. The top "on-topic" words (`milk`, `film`, `younger`) are content words specific to the *training* conversations; classic overfitting that doesn't transfer.

**Within-conversation adaptive idf** (learn vocabulary rarity inside each eval conversation): **−21.3pt** (collapses to 8%). Actively harmful — idf rewards rare/noise words over meaningful ones.

### What R3 means

The +74.9pt headroom is real, but it is **semantic, not lexical**. Knowing "which turn *answers* the question" requires understanding the question — exactly what an embedding/semantic model does, and exactly what grepmem deliberately does **not** have (vectorless by design). Within grepmem's lexical-only constraint, the focus channel's potential is unreachable. This is a clean, data-backed boundary, not a cop-out.

---

## Honest scorecard (including all negatives)

| | Gain | Negative | Net |
|---|---|---|---|
| R1 cumulative-use | +24pt consolidation | none (consolidation only slows decay) | ✅ real net gain |
| R1 LongMemEval benchmark | R@5 tied | **R@10 −2pt, knowledge-update −5.2pt** | ❌ net negative |
| R2 Tier A (constructed) | +100pt (mechanism works) | — | ✅ mechanism proven |
| R2 Tier B (LoCoMo) | ~0pt | late-span −0.6pt | ❌ net zero/negative |
| R3 cross-conv training | +2.7pt | overfits (4% of ceiling) | ◑ marginal |
| R3 within-conv idf | — | **−21.3pt** | ❌ harmful |

**Complexity is itself a cost:** LNN decay (CfC state, ODE-form math) is far heavier than flat's one-liner. Where the gain isn't decisive, that complexity is a net loss. This is why only R1's cumulative-use scenario is a clean win, and only as opt-in.

---

## What's here

### Core libraries

| Path | Round | Purpose |
|---|---|---|
| `lib/lnn-decay.js` | R1 | Core CfC liquid decay + reinforcement (~60 lines) |
| `lib/decay-comparator.js` | R1 | Unified `(node) → salience` for `flat` / `lnn` (fairness-invariant) |
| `lib/env.js` | R1 | `GREPMEM_DECAY` / `GREPMEM_NOW` config |
| `lib/grepmem-shim.js` | R1+ | Imports grepmem engine + `extractQueryTerms` from sibling `../ai-memory/` |
| `lib/focus-classifier.js` | R2 | Query → CfC input signal (lexical cues) |
| `lib/state-scheduler.js` | R2 | Coupled CfC state (z_focus + z_conf + focusLex) |
| `lib/match-rescore.js` | R2 | State-aware match re-scoring + combined score |

### Experiments

| Path | Round | Purpose |
|---|---|---|
| `eval/test-decay.mjs` | R1 | 13 unit tests: LNN decay vs analytical solution |
| `eval/test-equivalence.mjs` | R1 | JS vs Python numerical equivalence |
| `eval/ab-decay.mjs` | R1 | Exp 1a: flat vs lnn on LongMemEval-S R@5/R@10 |
| `eval/cumulative-use.mjs` | R1 | Exp 1b: adversarial consolidation sweep |
| `eval/diag-salience.mjs` | R1 | Single-question salience dump (debugging tool) |
| `eval/test-state-scheduler.mjs` | R2 | 16 unit tests incl. 3 coupling tests |
| `eval/tier-a-synthetic.mjs` | R2 | Tier A: constructed multi-turn scenarios |
| `eval/tier-b-locomo.mjs` | R2 | Tier B: LoCoMo real multi-session QA |
| `eval/diag-bottleneck.mjs` | R3 | Oracle ceiling diagnostic (go/no-go gate) |
| `eval/train-readout.mjs` | R3 | Word-weight focus training + eval |
| `python/cfc_reference.py` | R1 | Reference CfC via `ncps`/torch |

---

## Run it

```bash
# Correctness (R1 + R2 unit tests) — 32 tests, all pass
npm test

# ── R1 experiments ──
LIMIT=50 TYPES=temporal-reasoning,knowledge-update node eval/ab-decay.mjs   # needs LongMemEval-S (see below)
node eval/cumulative-use.mjs                                                 # self-contained

# ── R2 experiments ──
SCENARIOS=300 node eval/tier-a-synthetic.mjs
CONV=10 TOPK=5 node eval/tier-b-locomo.mjs                                   # needs LoCoMo (see below)

# ── R3 experiments ──
CONV=3 node eval/diag-bottleneck.mjs          # oracle ceiling — the gate
TRAIN=6 EVAL=10 node eval/train-readout.mjs   # word-weight training + eval
```

### Data dependencies

R1's `ab-decay.mjs` needs grepmem's LongMemEval-S dataset (277MB), referenced read-only from the sibling project:
```
../ai-memory/data/longmemeval_s_cleaned.json
```

R2/R3 LoCoMo data (2.8MB) isn't committed; fetch it:
```bash
curl -L -o eval/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

---

## Research process (how we got here)

1. **Motivation:** survey the "LNN replaces Transformer" hype; locate a real, narrow place LNN could help — grepmem's flagged-but-unimplemented salience consolidation.
2. **R1:** implement CfC decay, verify the math to 1e-9, test on LongMemEval (null/negative) then build the cumulative-use test that actually exercises the hypothesis (positive).
3. **R2:** extend to state-aware retrieval; prove the mechanism works in controlled scenarios but not on real data.
4. **R3:** before investing in training, run an oracle diagnostic — which overturned R2's pessimism (a +74.9pt ceiling exists) but then showed lexical training can't reach it.
5. **Conclusion:** one clean win (R1 consolidation, opt-in), two honest negatives (R2 state-awareness, R3 lexical training), and a proven boundary (the semantic ceiling).

The most valuable output is arguably the **negative results with numbers** — they bound where LNN helps agent memory and save others from re-treading the dead ends.

---

## Fairness invariants

- **R1:** both flat and lnn apply the identical `base + access*0.03` boost; the LNN's distinct contribution is confined to the decay term. So flat-vs-lnn tests decay *shape* only.
- **R2/R3:** flat-rescore, scalar-focus, and lnn-coupled all use the same re-simulated match code; they differ only in the boost/focus-bonus factor.

---

## References

- [Closed-form continuous-time neural networks — Hasani et al., Nature MI 2022](https://www.nature.com/articles/s42256-022-00556-7)
- [ncps reference implementation (Lechner/Hasani)](https://github.com/mlech26l/ncps)
- [grepmem](https://github.com/WAPA0012/grepmem) — the baseline
- [LoCoMo — Maharana et al., ACL 2024](https://snap-research.github.io/locomo/) — R2/R3 multi-session data

## License

MIT
