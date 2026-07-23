# grepmem-lnn

> Liquid Neural Network experiments for [grepmem](https://github.com/WAPA0012/grepmem): (R1) per-memory CfC time-decay replacing flat linear salience decay, and (R2) a coupled-CfC session-state scheduler that re-weights retrieval by the agent's accumulated focus.

**Status: both rounds complete. Results below are honest, including the negative ones.**

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

## Results — Round 1: per-memory liquid time-decay

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

---

# Round 2: state-aware retrieval scheduling

Round 1 changed the `salience` factor of `match × salience`. Round 2 attacks the other factor — `match` — with a **session-state vector** that an LNN maintains across turns, dynamically re-weighting which retrieval terms get amplified based on "what the agent is currently focused on". This is the "agent remembers the ongoing task's focus" capability pure grep retrieval lacks.

## The state (coupled CfC)

`z = (z_focus, z_conf)` evolved by one CfC cell:
- **z_focus ∈ Δ⁴** — soft distribution over 4 retrieval-relevant focus modes: `exact-entity`, `procedure-debug`, `config-param`, `narrative-context`. Each mode favors different term types (exact/ip vs synonym/learned vs en_ci/number vs bigram/trigram).
- **z_conf ∈ [0,1]** — a confidence channel that **modulates the focus time constant τ** (the load-bearing coupling): high confidence → large τ → focus resists change (inertia); low confidence → small τ → focus re-orients fast. This cross-channel coupling is structurally impossible for a per-channel scalar-decay ablation.
- **focusLex** — a CfC-decayed bag of words from recent turns, the channel that makes state-awareness actually fire on English queries (term-type boosts alone rarely trigger there, since `synonym`/`learned` terms only arise via the bilingual synonym map).

Readout: a hand-designed 4×9 boost table × confidence gate. No training data; the LNN's *dynamics* are used, not learned representations (without training data the latter is infeasible — this is an explicit, reported limitation).

## Results — Round 2

### Tier A — constructed multi-turn scenario (positive)

A reinforced old-but-topic-relevant memory vs a fresh oneoff, equal lexical match. State-awareness breaks the tie via topic continuity. 300 scenarios:

| strategy | picks-correct-answer% |
|---|---|
| flat-rescore (no state) | **0%** (genuine lexical tie → loses) |
| scalar-focus (fixed τ) | **100%** |
| lnn-coupled (full) | **100%** |

State-aware retrieval works (+100pt vs flat). **But scalar-focus = lnn-coupled = 100%** — the lexical-focus bag did all the work; the coupling added nothing in this short deterministic setting.

### Tier B — LoCoMo real multi-session data (negative)

Full dataset, 1982 evidence-filtered QA across 10 conversations (up to 35 sessions each), any-hit R@5:

| strategy | R@5 overall | early(1-5) | mid(6-15) | late(16+) |
|---|---|---|---|---|
| flat (no state) | **28.6%** | 24% | 30% | 29% |
| scalar-focus (fixed τ) | **28.9%** | 23% | 29% | 31% |
| lnn-coupled (full) | **28.8%** | 24% | 29% | 30% |

- **flat → coupled: +0.2pt** — effectively zero, within noise.
- **scalar → coupled: −0.2pt** — coupling does not help on long horizons.
- **late-span (the predicted coupling win zone): −0.6pt** — coupling slightly *hurts*.

## What this means (Round 2)

- **The state-aware mechanism is real and fires** (Tier A: +100pt) — when lexical match is a genuine tie, topic continuity from accumulated state breaks it correctly.
- **On real conversational data it does not help** (Tier B: ~0pt) — and **the coupled CfC adds nothing over a simple scalar-decay focus bag** (scalar ≈ coupled, sometimes scalar wins). This is the honest, decisive finding.
- **Why?** LoCoMo's QA questions are asked by an external evaluator about the conversation, not in-conversation follow-ups driven by the speaker's focus. The "state" we accumulate from prior turns is a weak proxy for what the probe question actually needs, and lexical match dominates ranking regardless. The coupling (confidence-modulated τ) is a second-order effect that never gets leverage.
- **The coupled CfC is load-bearing only in principle.** The unit tests prove the coupling works (high-conf inertia vs low-conf re-orientation), but no real benchmark here exercises a regime where that coupling beats a fixed-τ scalar. Without training data to learn the state→readout mapping, the LNN's representational advantage is unavailable, and its dynamical advantage is too subtle to register on these tasks.

**Bottom line for Round 2:** state-aware retrieval is a coherent idea that demonstrably works in controlled tie-breaking scenarios, but on real multi-session QA it provides no measurable benefit, and the LNN coupling specifically adds nothing over a simpler scalar-decay baseline. This is reported as-is — a negative result that bounds where the approach helps.

---

# Round 3: training the focus channel (and the ceiling it hits)

Round 2's state-aware scheduler gave ~0pt on LoCoMo. Round 3 asks: **is that a fixable focus-signal problem (training could help) or a lexical ceiling (training is futile)?** The answer turned out more interesting than either.

## Step 0 — bottleneck diagnostic (the go/no-go gate)

Before writing any training code, an **oracle analysis** (`eval/diag-bottleneck.mjs`) measured the R@5 *ceiling* if the focus channel had perfect information — i.e., cheating by giving the known evidence turns a direct bonus:

| variant | R@5 | meaning |
|---|---|---|
| A. flat (no state) | **24.8%** | baseline |
| B. **oracle-focus-bonus** (perfect topic ID) | **99.8%** | the ceiling |
| C. uniform boost sweep (any magnitude) | 24.8% | readout magnitude is irrelevant |
| D. real boosts, no focus bag | 25.9% | the hand-designed boost table does nothing |

**B − A = +74.9pt.** This decisively *rejects* the lexical-ceiling hypothesis: a focus channel that could accurately identify on-topic candidates has enormous headroom. Training is worth attempting.

It also localized the bottleneck precisely: **the readout magnitude is not the problem** (C ≈ A for all boost values) — the problem is *which candidates the focus signal points at*.

## Step 1 — training word weights

Two lexical-training attempts (`eval/train-readout.mjs`), both honest about being weak-supervision feature weighting, not deep learning:

**Cross-conversation log-odds weights** (train on 6 convs, eval on 4 held-out): learn per-word weights from evidence-vs-non-evidence turns. Result: **+2.7pt** (29.3% → 31.9%), but captures only **4% of the oracle headroom**. The top "on-topic" words (`milk`, `film`, `younger`) are content words specific to the *training* conversations — classic overfitting that doesn't transfer.

**Within-conversation adaptive idf** (learn vocabulary rarity inside each eval conversation): **−21.3pt** (collapses to 8%). Actively harmful — idf rewards rare/noise words over meaningful ones.

## What this means (Round 3)

- **The +74.9pt headroom is real** — an oracle focus channel proves state-aware retrieval *could* dramatically help on LoCoMo. This overturns the Round 2 read that it's a dead end.
- **But that headroom is semantic, not lexical.** Knowing "which turn *answers* the question" requires understanding the question — which is exactly what an embedding/semantic model does, and exactly what grepmem deliberately does **not** have (vectorless by design).
- **Lexical re-training cannot reach it.** Cross-conv word weights overfit to content; within-conv idf rewards noise. Neither captures "this turn is the answer to that question."
- **The honest conclusion:** the state-aware retrieval direction is *not* futile in principle (oracle proves a 3× R@5 ceiling exists), but unlocking it requires a semantic retrieval component — which would violate grepmem's vectorless philosophy. Within the lexical-only constraint grepmem imposes, the focus channel's potential is unreachable. This is a clean, data-backed boundary, not a cop-out.

**Net across all three rounds:** LNN time-decay (R1) gives a real, usable gain in cumulative-use scenarios (+24pt consolidation). State-aware retrieval (R2/R3) has a provably large ceiling but one that's only reachable with semantics, not lexical state — so it stays at ~0pt within grepmem's design. Both conclusions are backed by numbers, including the negatives.

## What this means (Round 1)

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
| `eval/ab-decay.mjs` | R1 Exp 1: flat vs lnn on LongMemEval-S R@5/R@10 |
| `eval/cumulative-use.mjs` | R1 Exp 2: adversarial consolidation sweep |
| `eval/diag-salience.mjs` | Single-question salience dump (debugging) |
| `lib/focus-classifier.js` | R2: query → CfC input signal (lexical cues) |
| `lib/state-scheduler.js` | R2: coupled CfC state (z_focus + z_conf + focusLex) |
| `lib/match-rescore.js` | R2: state-aware match re-scoring |
| `eval/test-state-scheduler.mjs` | R2: 16 unit tests incl. the coupling tests |
| `eval/tier-a-synthetic.mjs` | R2 Tier A: constructed multi-turn scenarios |
| `eval/tier-b-locomo.mjs` | R2 Tier B: LoCoMo real multi-session QA |
| `eval/diag-bottleneck.mjs` | R3: oracle ceiling diagnostic (go/no-go gate) |
| `eval/train-readout.mjs` | R3: word-weight focus training + eval |
| `python/cfc_reference.py` | Reference CfC via `ncps`/torch |

## Run it

```bash
# Correctness (R1 + R2 unit tests)
npm test
node --test eval/test-state-scheduler.mjs   # R2 state tests

# R1 experiments
LIMIT=50 TYPES=temporal-reasoning,knowledge-update node eval/ab-decay.mjs
node eval/cumulative-use.mjs

# R2 experiments
SCENARIOS=300 node eval/tier-a-synthetic.mjs
CONV=10 TOPK=5 node eval/tier-b-locomo.mjs   # needs eval/locomo/locomo10.json (see below)

# R3 experiments (training the focus channel)
CONV=3 node eval/diag-bottleneck.mjs         # oracle ceiling — the go/no-go gate
TRAIN=6 EVAL=10 node eval/train-readout.mjs  # word-weight training + eval
```

LoCoMo (Tier B) data isn't committed (2.8MB); fetch it:
```bash
curl -L -o eval/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## Fairness invariant

To make flat-vs-lnn a clean test of *decay shape* (not access-boost), both strategies apply the identical `base + access*0.03` boost to the starting level. The LNN's distinct contribution is confined to the decay term (per-memory, access-adaptive `w_tau`). See `lib/decay-comparator.js`.

## References

- [Closed-form continuous-time neural networks — Hasani et al., Nature MI 2022](https://www.nature.com/articles/s42256-022-00556-7)
- [ncps reference implementation (Lechner/Hasani)](https://github.com/mlech26l/ncps)
- [grepmem](https://github.com/WAPA0012/grepmem) — the baseline

## License

MIT
