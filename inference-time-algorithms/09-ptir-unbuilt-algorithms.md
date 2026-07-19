# Unbuilt PTIR algorithms

The PTIR programming model makes a decode step a *guest-authored device
program*. `mirostat` proved the pattern: a stateful sampler that used to be a
host loop is now a traced tensor program. This document asks what else the
pattern unlocks that **no existing inferlet covers**.

Scope: algorithms whose substance is a **per-step device program** (PTIR),
not host orchestration. Meta-generation (ToT, MCTS, best-of-N, reflexion) is
deliberately excluded — it is host-side and already demonstrated upstream.

---

## Method

Coverage was determined mechanically, not from memory:

```bash
# 1. which PTIR intrinsics do existing inferlets actually call?
grep -rh "intrinsics::" tests/inferlets runtime/engine/tests/inferlets --include=*.rs \
  | sed 's/.*intrinsics::\([a-z_]*\).*/\1/' | sort | uniq -c | sort -rn

# 2. does the CUDA driver bind each intrinsic?
grep -n "struct FireInputs" -A 25 driver/cuda/src/pipeline/tier0/tier0_runner.hpp

# 3. is algorithm X mentioned anywhere in either inferlet set?
grep -rli "<name>" tests/inferlets runtime/engine/tests/inferlets /root/pie/inferlets
```

Inferlet inventory used as the baseline: 14 curated (`tests/inferlets/`) + 34
engine tests (`runtime/engine/tests/inferlets/`) in this branch, plus the 76
upstream examples in `pie-project/pie@main:inferlets/`, so nothing is proposed
that upstream already demonstrates.

## Finding 1 — four intrinsics exist and nothing uses them

| Intrinsic | In PTIR ABI | Bound in CUDA `FireInputs` | Inferlets using it |
|---|---|---|---|
| `logits()` | yes | yes | 41 files, **78 calls** |
| `mtp_logits(k)` | yes | yes | 4 files, **9 calls** |
| `query()` | yes | **yes** (attention tap) | **0** |
| `layer()` | yes | **yes** (attention tap) | **0** |
| `hidden()` | yes | **no** — falls through to `"tier-0: intrinsic not yet bound"` | **0** |
| `value_head()` | yes | **no** — same | **0** |

Two consequences:

- `query()` and `layer()` are **already wired on CUDA and completely
  unexplored**. Everything in Tier B below is buildable today.
- `hidden()` and `value_head()` are declared in the ABI (`interface/ptir/src/op.rs`,
  `header.rs`, `interp.rs`), implemented in the dummy driver and exercised by
  Metal's interpreter test, but the CUDA tier-0 runner has no `hidden` /
  `value_head` field in `FireInputs`. Tier C needs that binding first — a
  small, well-scoped driver change, not a redesign.

## Finding 2 — the op set is not the bottleneck

PTIR already has `ReduceSum`, `ReduceArgmax`, `CumSum`, `SortDesc`, `TopK`,
`Gather`, `GatherRow`, `ScatterAdd`, `ScatterSet`, `Select`, a sort-free
top-k/top-p/min-p mask op (`0x58`), `mask_apply`, and the full elementwise /
comparison / logical set. Entropy is `neg(ReduceSum(p * log p))`; a frequency
penalty is a `ScatterAdd` histogram; a nucleus variant is `SortDesc` + `CumSum`
+ threshold. **Every Tier A item below needs no new opcode.**

---

## Tier A — pure `logits()`, buildable today, zero engine work

Everything here is the mirostat pattern: a traced program over the logit row,
optionally carrying per-sequence state in a device-advanced channel.

| # | Algorithm | Paper | PTIR shape | Why it's a gap |
|---|---|---|---|---|
| A1 | **Locally typical sampling** | Meister et al., [2202.00666](https://arxiv.org/abs/2202.00666) | `H = -ReduceSum(p·log p)`; score `|log p + H|`; `SortDesc` + `CumSum` threshold | Engine ships Argmax/TopP/TopK/MinP/TopKTopP/Multinomial only |
| A2 | **η- and ε-sampling** | Hewitt et al., [2210.15191](https://arxiv.org/abs/2210.15191) | entropy-dependent floor `min(ε, √ε·exp(−H))`, then mask | no entropy-adaptive truncation exists |
| A3 | **Tail-free sampling** | community method (no canonical paper) | `SortDesc` → 2nd difference of the prob curve → cut | absent |
| A4 | **Top-a sampling** | community method | threshold ∝ `p_max²`; one `max_elem` + compare | absent |
| A5 | **XTC (exclude top choices)** | community method | probabilistically drop above-threshold heads | absent |
| A6 | **Frequency / presence / repetition penalty** | standard | `ScatterAdd` token histogram in a channel, subtract before softmax | **not in the `Sampler` enum at all** — a conspicuous hole |
| A7 | **DRY repetition penalty** | community method | n-gram suffix match via `Gather` + compare, penalty scaled by match length | absent; needs the mirostat state pattern |
| A8 | **Entropy-adaptive temperature** | EDT [2403.14541](https://arxiv.org/abs/2403.14541); AdapT [2309.02772](https://arxiv.org/abs/2309.02772) | `T = f(H)` computed on device, fed to the existing temperature path | `entropycheck` *measures* entropy but nothing *acts* on it |
| A9 | **Distortion-free / Gumbel watermark** | Kuditipudi et al., [2307.15593](https://arxiv.org/abs/2307.15593); Aaronson | key the existing Gumbel-max noise off `hash(secret, context)` instead of the RNG counter | **the driver already runs keyed Gumbel-max sampling** (`[key, ctr]` state) — unusually close to free. Only the *greenlist* watermark ([2301.10226](https://arxiv.org/abs/2301.10226)) is implemented |
| A10 | **SynthID-Text tournament sampling** | Dathathri et al., *Nature* 2024 (no arXiv preprint) | multi-round tournament over candidates with keyed g-functions | absent; notable because it is explicitly designed to survive speculative decoding — which pie can compose (`mtp-grammar` proves masks compose with spec) |

**Suggested first build: A6 + A8 + A1.** A6 is a genuine functional hole, A8
reuses `entropycheck`'s measurement plus `mirostat`'s control-word pattern, and
A1 is the canonical "new sampler" demo.

---

## Tier B — score-driven KV policies: blocked on a missing sensor

Pie today has **static** attention policies (`attention-sink`,
`sliding-window-attention`, `windowed-attention` upstream); it has no
**score-driven dynamic** policy. An earlier draft of this document, and of
`candidates.md`, attributed that to `query()` simply being unused. **That was
wrong.** The tier was re-checked against the interface while implementing
Tiers A/D/E, and none of B1–B5 is writable today.

### The half that exists, and the half that does not

| | Capability | Status |
|---|---|---|
| **Actuator** | choose which pages a step attends to | ✅ `Port::Pages`, `Port::PageIndptr` — `interface/ptir/src/registry.rs:102-113` |
| **Actuator** | mask which positions survive | ✅ `Port::AttnMask` — same enum |
| **Sensor** | read `softmax(QK^T)` | ❌ no such intrinsic |
| **Sensor** | read the layer's projected keys `K` | ❌ no such intrinsic |
| **Sensor** | read the layer's projected query `Q` | ✅ `IntrinsicId::Query` |

`IntrinsicId` (`interface/ptir/src/op.rs:49-68`) is exhaustively `Logits,
MtpLogits, Hidden, Query, ValueHead, Layer, MtpDrafts`. `Port`
(`registry.rs:102-113`) is exhaustively `EmbedTokens, EmbedIndptr, Positions,
Pages, PageIndptr, KvLen, WSlot, WOff, Readout, AttnMask` — `Pages`,
`PageIndptr` and `KvLen` carry page **geometry**, never page **contents**.

So `query()` is necessary but not sufficient: every algorithm in this tier
scores positions with `QK^T`, and `K` is not observable. Nor is it
recoverable — `Query`/`Layer` are restricted to `Stage::OnAttnProj |
Stage::OnAttn` (`registry.rs:185`) while `Hidden` is epilogue-only, and the
per-layer `W_k` projection is not exposed in any form, so keys cannot be
recomputed from anything an inferlet can see.

### What would unblock it

**One intrinsic**: either the post-softmax attention probabilities or the
layer's projected keys, readable at `Stage::OnAttn`. Every row below then
becomes a small inferlet, because the expensive half — paged KV plus a
guest-bound attention mask — already exists. That makes this tier the highest
leverage single ABI addition on the list: one tap, five algorithms.

| # | Algorithm | Paper | What it needs | Why it's a gap |
|---|---|---|---|---|
| # | Algorithm | Paper | Signal it needs | Blocked because |
|---|---|---|---|---|
| B1 | **H2O — heavy-hitter KV eviction** | Zhang et al., [2306.14048](https://arxiv.org/abs/2306.14048) | attention mass accumulated per position | needs `softmax(QK^T)`; `K` unreadable |
| B2 | **SnapKV** | Li et al., [2404.14469](https://arxiv.org/abs/2404.14469) | per-head attention over an observation window | same |
| B3 | **TOVA** | Oren et al., [2401.06104](https://arxiv.org/abs/2401.06104) | the current step's attention score | same |
| B4 | **Quest — query-aware page selection** | Tang et al., [2406.10774](https://arxiv.org/abs/2406.10774) | per-page elementwise min/max of `K`, scored against `Q` | needs `K` directly |
| B5 | **RetrievalAttention** | Liu et al., [2409.10516](https://arxiv.org/abs/2409.10516) | an ANN index over `K` | needs `K` directly |

**Still the best first build once unblocked: B4 (Quest).** Pie's KV is
*already* paged and the attention mask is *already* a guest-bound channel, so
Quest is closer to a natural expression here than in any other engine — the
missing piece is purely the key summary, not the policy machinery.

---

## Tier C — needs `hidden()` / `value_head()` bound in the CUDA runner first

Prerequisite: add `hidden` and `value_head` to `FireInputs` in
`driver/cuda/src/pipeline/tier0/tier0_runner.hpp` and bind them in the
epilogue, mirroring what `driver/dummy` and the Metal interpreter test already
do.

**Honest limitation:** `hidden()` is documented as *"the residual stream at
read-out (epilogue)"* — a **read** tap after the last layer. That is enough for
everything in the table below. It is **not** enough for true mid-network
activation steering (ITI / CAA / RepE / SAE feature steering), which needs a
per-layer **write** port that does not exist in the ABI today. Those remain
genuinely blocked, and are the most interesting reason to extend PTIR.

| # | Algorithm | Paper | What it needs |
|---|---|---|---|
| C1 | **EAGLE-style feature-level drafting** | Li et al., [2401.15077](https://arxiv.org/abs/2401.15077) | draft head consumes `hidden()` rather than tokens; pie has MTP but no feature-level draft |
| C2 | **Semantic entropy / confidence-gated decoding** | see `06` | cluster/score `hidden()` to gate or abstain |
| C3 | **Value-head guided beam / PRM-style step scoring** | see `03` | `value_head()` scored on device so search node evaluation costs no extra pass |
| C4 | **DoLa — layer-contrastive decoding** | Chuang et al., [2309.03883](https://arxiv.org/abs/2309.03883) | premature-layer logits; needs a per-layer LM-head readout on top of `layer()` |
| C5 | *(blocked)* **ITI / CAA / ActAdd / RepE steering** | [2306.03341](https://arxiv.org/abs/2306.03341), [2308.10248](https://arxiv.org/abs/2308.10248), [2310.01405](https://arxiv.org/abs/2310.01405) | a **write** port into the residual stream — not in the ABI |

---

## Tier D — multi-distribution: two contexts combined in one step

Pie can bind heterogeneous passes into one frame, so combining two logit rows
before sampling is a natural PTIR program. Nothing does it today except
`contrastive-decoding`, and that one is *same-model, bounded-context*.

| # | Algorithm | Paper | Gap vs. what exists |
|---|---|---|---|
| D1 | **Classifier-free guidance for LLMs** | Sanchez et al., [2306.17806](https://arxiv.org/abs/2306.17806) | `logits = uncond + γ(cond − uncond)`; needs prompted/unprompted contexts in one frame |
| D2 | **Context-aware decoding (CAD)** | Shi et al., [2305.14739](https://arxiv.org/abs/2305.14739) | with-context vs without-context contrast — the anti-hallucination version of D1 |
| D3 | *(blocked)* **Cross-model contrastive decoding** | Li et al., [2210.15097](https://arxiv.org/abs/2210.15097) | existing `contrastive-decoding` is one model with two context lengths; the original is expert **and** amateur *models* |
| D4 | *(blocked)* **DExperts / GeDi / proxy tuning / emulated fine-tuning** | see `02`, `06` | logit arithmetic across 2-3 models |

**D1 and D2 are built** (`classifier-free-guidance`, `context-aware-decoding`);
the two-context frame pattern predicted here holds, with one correction — the
two streams need **two `WorkingSet`s on one `Pipeline`**, not two pipelines, and
the loop must be strictly sequential.

**D3 and D4 are blocked, and not on effort.** Both need two or more *different*
models resident and steppable from a single inferlet, and
`sdk/rust/inferlet/wit/model.wit:3` fixes the opposite invariant:

> The engine serves exactly one model; these are global functions over that
> single bound model (no `model` resource handle).

With no model handle in the WIT surface there is no way for an inferlet to name
a second model, so the arithmetic these methods perform — `logit_expert −
logit_amateur`, or `logit_base + logit_expert − logit_antiexpert` — has no
second operand to fetch. Unblocking needs a multi-model service surface, which
is a substantially larger change than the single attention tap Tier B wants.

---

## Tier E — constrained-decoding correctness

Pie has strong constrained decoding (`grammar`, `grammar-late`, `grammarmb`,
`json-schema-constrained-decoding`, and `mtp-grammar` composing it with
speculation). What it does not have is the *correctness* literature.

| # | Algorithm | Paper | Gap |
|---|---|---|---|
| E1 | **Grammar-aligned decoding (ASAp)** | Park et al., [2405.21047](https://arxiv.org/abs/2405.21047) | hard masking provably distorts the distribution; ASAp corrects it using expected future grammaticality. Needs constraint state **plus** KV truncation/resample — pie has both |
| E2 | **Token healing / tokenizer alignment** | see `06` | masking at token granularity mis-handles prefix-splitting; no inferlet addresses it |

---

## Explicitly excluded — already covered

Not proposed, because an inferlet already demonstrates it:

- samplers: argmax / top-p / top-k / min-p / top-k-top-p / multinomial / temperature (`sampling-primitives`, `multisamp`, `tempgen`, `isolatedtopp`, upstream `sampler-suite`)
- **mirostat v2** (`mirostat`, `mirostat-v2-sampling`) — the model for Tier A
- greenlist watermarking (`greenlist-watermarking`, upstream `watermarking`)
- grammar / JSON-schema constrained decoding, incl. late masking and MTP composition
- speculative decoding: self-spec, MTP native verify, cacheback n-gram, upstream `jacobi-decoding`
- beam search incl. logical mask-out + compaction (`beam`, `beam-designb*`)
- attention sinks, sliding window, prefix-tree KV, prefix-cache grafting
- entropy *measurement* (`entropycheck`) — but see A8, nothing consumes it
- meta-generation: upstream `tree-of-thought`, `graph-of-thought`, `mcts-*`,
  `best-of-n`, `reflexion`, `demo-self-correct`, `skeleton-of-thought`,
  `recursion-of-thought`, `parallel-generation`

---

## Suggested ordering

1. **A6** frequency/presence/repetition penalties — a real functional hole, trivial PTIR.
2. **A8** entropy-adaptive temperature — connects `entropycheck` to `mirostat`'s control-word pattern.
3. **A1/A2** typical + η/ε sampling — canonical "the sampler is user code" demo.
4. **A9** Gumbel distortion-free watermark — the keyed Gumbel machinery already exists.
5. **B4** Quest — first user of `query()`; showcases paged KV + guest mask together.
6. **C-prereq** bind `hidden`/`value_head` in the CUDA tier-0 runner, then **C3/C1**.
7. **D1/D2** CFG and CAD — first multi-context frame program.
8. **E1** ASAp — the most defensible correctness contribution.

Items 1-5 need **no engine changes at all**.
