# What Pie uniquely enables

The synthesis document. The other reports catalogue ~289 papers; this one
answers the question that motivated the search: **which inference-time
algorithms are Pie-native, and which are merely awkward elsewhere?**

## How to read this honestly

It would be easy — and wrong — to claim vLLM and SGLang can't do any of this.
They can do a lot. Being precise about the boundary is what makes the claim
useful:

| Capability | vLLM | SGLang | Pie |
|---|---|---|---|
| Change `temperature` / `top_p` / `top_k` | yes | yes | yes |
| Custom logit transform per step | yes — host-side `LogitsProcessor` (Python callback, per step) | yes — host-side | yes — **as a device program** (PTIR) |
| Grammar / JSON-schema constrained decoding | yes — built in (xgrammar/outlines backends) | yes — built in (compressed FSM) | yes — built in **and** guest-composable (mask AND) |
| Prefix sharing across requests | yes — automatic prefix caching | yes — RadixAttention | yes — content-addressed pages, **explicitly addressable** |
| Speculative decoding | yes — engine-provided methods | yes — engine-provided | yes — **user-written speculators** |
| Read hidden states / per-layer activations mid-forward | no | no | yes — `hidden()`, `query()`, `layer()` |
| Write/steer activations mid-forward | no | no | yes — PTIR program |
| Arbitrary custom attention mask per pass | no | no | yes — mask channel |
| Explicit KV fork / snapshot / rollback as an API | no (internal only) | partial (fork in the frontend language) | yes — `ctx.fork()`, snapshot, truncate, `WorkingSet` ops |
| Per-token state without host round-trip | no | no | yes — device-advanced channels |
| Guest-controlled compute allocation | no | no | yes — credit/bid market |

So the honest framing is **three tiers**:

- **Tier 1 — already commodity.** Truncation samplers, JSON-constrained
  decoding, prefix caching, standard speculative decoding. Pie does these, but
  so does everyone. Not a differentiator.
- **Tier 2 — possible elsewhere, but structurally penalised.** Anything needing
  per-token host logic or branch-heavy search: a host-side logits processor
  costs a round-trip per token, and branching costs re-prefill or relies on an
  opaque cache. Pie removes the penalty rather than the possibility.
- **Tier 3 — not expressible without patching the engine.** Everything below
  the "hidden states" line in the table. This is the real frontier.

The eight axes referenced below are defined in `00-pie-capability-map.md` §10.

---

## Tier 3 — the genuinely Pie-native frontier

### A. Algorithms that need more than logits

The single biggest category. A black-box server hands you a distribution over
the vocabulary and nothing else; these methods need the residual stream, a
specific layer, the attention query, or a value head.

| Algorithm | Needs | Report |
|---|---|---|
| **DoLa** — contrast a mature layer against a premature layer | per-layer logits (`layer()`, `hidden()`) | `02`, `06` |
| **Inference-Time Intervention (ITI)** | write a steering vector into attention-head activations | `06` |
| **Contrastive Activation Addition (CAA)** / ActAdd / RepE | add a steering vector to the residual stream mid-forward | `06` |
| **Refusal-direction ablation** | project out a direction from `hidden()` | `06` |
| **SAE feature steering** | encode/decode `hidden()` through a sparse autoencoder at decode | `06` |
| **EAGLE / EAGLE-2 / EAGLE-3** | draft at the *feature* (hidden-state) level, not the token level | `04` |
| **Semantic entropy / confidence-gated decoding** | entropy over hidden-state clusters, not just logits | `06` |
| **Value-guided search, PRM-guided beam** | a scalar value head evaluated on device per candidate | `03` |
| **Query-aware KV selection (Quest, RetrievalAttention)** | the current `query()` to score which KV pages to attend to | `05` |

> Why this matters: on Pie each of these is a traced tensor program shipped as
> Wasm. Everywhere else it is a fork of the engine.

### B. Algorithms that need a custom attention mask

| Algorithm | Needs | Report |
|---|---|---|
| **Tree/multi-candidate speculative verification** (SpecInfer, Sequoia, Medusa trees) | tree attention mask over the draft tree | `04` |
| **StreamingLLM / attention sinks / LM-Infinite (Λ-mask)** | sink + window mask, and position capping | `05` |
| **Beam search with logical ancestry masks** | per-beam visibility mask, no KV duplication | `03` |
| **Prefix-tree / hierarchical attention over shared branches** | mask encoding the branch DAG | `05` |
| **Dual-chunk / landmark / blockwise long-context attention** | restructured mask at inference | `05` |

Pie already ships `attention-sink`, `sliding-window-attention`,
`prefix-tree-kv-cache` and `beam-designb` (logical mask-out + lazy compaction)
as in-repo inferlets — direct evidence the axis is real.

### C. Algorithms that need explicit KV branching and backtracking

Every step-level search method reduces to fork → explore → score → prune →
backtrack. On a black-box server you either re-prefill each branch or hope the
prefix cache helps; neither gives you *rollback*.

| Algorithm | Needs | Report |
|---|---|---|
| **Tree of Thoughts, Graph of Thoughts** | O(1) fork per thought, prune, resume | `03` |
| **MCTS for reasoning** (RAP, TS-LLM, rStar, ReST-MCTS*, LATS) | fork + backtrack + per-node value | `03` |
| **Self-consistency / best-of-N over a shared prefix** | N forks sharing committed pages | `03` |
| **Speculative rejection for best-of-N** | early-kill a branch and reclaim its pages | `03` |
| **Self-Refine / Reflexion / self-correction** | fork a critic off the same prefix without disturbing it | `03` |
| **Tool-call retry / rollback** | snapshot before the call, restore on failure | `03`, `07` |
| **Constrained decoding with backtracking** (when a hard constraint dead-ends) | truncate + resume | `06` |

### D. Algorithms that need per-token state without a host round-trip

Expressible elsewhere via a host-side logits processor, but at a per-token
Python round-trip — which is exactly the cost these methods cannot absorb at
scale. Pie's device-advanced channels make the state device-resident.

Mirostat, DRY / XTC repetition control, green-list watermarking (KGW),
distortion-free / Gumbel watermarking, SynthID-Text, entropy-adaptive
temperature (EDT, AdapT, "entropix"-style), typical / η / ε sampling with
running statistics. See `02`.

### E. Algorithms that combine several distributions per step

Contrastive decoding, context-aware decoding (CAD), classifier-free guidance
for LLMs, DExperts, GeDi, FUDGE, proxy tuning, emulated fine-tuning,
multi-objective decoding-time alignment. All need **two or more forward
contexts advanced in lockstep and their logits combined before sampling** —
Pie can bind both into one frame and combine in PTIR; elsewhere this is two
servers and a host loop. See `02`, `06`.

### F. Algorithms that need a custom draft/verify rule

Self-speculation, native-MTP draft/verify, n-gram / prompt-lookup cacheback,
retrieval-based drafting (REST, SuffixDecoding), layer-skip self-drafting
(LayerSkip, Kangaroo), Jacobi/CLLM consistency decoding, and — the composition
that is genuinely hard elsewhere — **speculation composed with grammar
constraints in a single pass**. Pie ships `mtp-grammar` as an inferlet. See `04`.

### G. Algorithms that need a guest compute-allocation policy

Compute-optimal test-time scaling (Snell et al.), budget forcing (s1),
difficulty-aware routing, adaptive branch budgets in tree search, goodput-aware
speculation length (SmartSpec/TurboSpec). These are *policies over the
scheduler*, and Pie is the only one of the three that lets a guest influence
the bid. See `03`, `04`, `07`.

### H. Algorithms that interleave tool/agent I/O with generation

Multi-agent debate, mixture-of-agents, long-horizon memory compaction (ACON),
sub-agent spawning, tool-augmented search. Running these *inside* the serving
system means a tool call does not cost the context its cache locality — Pie has
HTTP/filesystem/session/MCP plus inter-inferlet launch and messaging. See `07`.

---

## Highest-leverage open projects

Ranked by (novelty of what Pie makes possible) × (how blocked it is elsewhere).
These are candidates for new inferlets, not claims that they already exist.

1. **Activation-steered decoding as a serving feature.** ITI/CAA/RepE/SAE
   steering as a PTIR program, composable with sampling and constraints.
   Nothing in the current serving stack can express this at all.
2. **Query-aware sparse attention driven by `query()`.** Quest- and
   RetrievalAttention-style dynamic KV selection, written as a guest policy
   instead of an engine feature — and therefore tunable per application.
3. **On-device value-guided tree search.** `value_head()` scoring fused into
   the forward pass so MCTS/PRM-beam node evaluation costs no extra round-trip,
   with `fork()` supplying the branches and the bid market the budget.
4. **Layer-contrastive decoding (DoLa) composed with speculation.** Requires
   per-layer logits *and* a custom verify rule simultaneously — the
   intersection of two axes no other engine exposes.
5. **Grammar-aligned decoding (ASAp) with backtracking.** Fixing the known
   distribution distortion of constrained decoding needs resampling and
   rollback, i.e. constraint state *and* KV truncation together.
6. **Watermarking that survives speculative decoding.** Watermark logic must
   run inside the draft/verify loop; today the two features are mutually
   exclusive in practice.
7. **Any-order / diffusion-LM decoding schedules.** Remasking and parallel
   unmasking need arbitrary masks and non-left-to-right readout — mask channel
   plus `readout()` indices are exactly the primitives required.

---

## Caveats

- Tier assignments describe the *programming model*, not performance. Pie being
  able to express an algorithm says nothing about whether its implementation is
  faster than a hand-patched engine.
- vLLM and SGLang move fast; the table reflects the state described in the
  papers surveyed in `07` and should be re-checked before being quoted.
- "Not expressible" always means "without patching engine internals". Anything
  is possible with a fork of the engine; the claim is about what a *user* can
  ship without one.
