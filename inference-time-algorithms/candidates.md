# Candidates

Inference-time algorithms that the PTIR programming model makes expressible as
guest code, and that **no existing inferlet covers**.

`mirostat` is the template: a stateful sampler that used to be a host loop is
now a traced device program. Everything here follows that pattern.

**Scope.** Per-step *device programs* (PTIR). Host-side meta-generation (ToT,
MCTS, best-of-N, reflexion) is out of scope — it is already demonstrated
upstream. See "Excluded" at the bottom for the full covered list.

---

## Two facts that set the priorities

### Fact 1 — four intrinsics exist and nothing uses them

| Intrinsic | PTIR ABI | Bound in CUDA `FireInputs` | Inferlets using it |
|---|---|---|---|
| `logits()` | yes | yes | 41 files, **78 calls** |
| `mtp_logits(k)` | yes | yes | 4 files, **9 calls** |
| `query()` | yes | **yes** (attention tap) | **0** |
| `layer()` | yes | **yes** (attention tap) | **0** |
| `hidden()` | yes | **no** — falls through to `"tier-0: intrinsic not yet bound"` | **0** |
| `value_head()` | yes | **no** — same | **0** |

So `query()`/`layer()` are already wired on CUDA and completely unexplored
(→ Tier B, no engine work). `hidden()`/`value_head()` are in the ABI
(`compiler/{ir/src/op,codegen/src/header,eval/src/interp}.rs`), implemented in `driver/dummy`, and
exercised by Metal's interpreter test — but `FireInputs` in
`driver/cuda/src/pipeline/tier0/tier0_runner.hpp` has no field for them
(→ Tier C needs that binding first).

### Fact 2 — the op set is not the bottleneck

PTIR already has `ReduceSum`, `ReduceArgmax`, `CumSum`, `SortDesc`, `TopK`,
`Gather`, `GatherRow`, `ScatterAdd`, `ScatterSet`, `Select`, the sort-free
top-k/top-p/min-p mask op (`0x58`), `mask_apply`, and the full
elementwise/comparison/logical set.

Entropy is `neg(ReduceSum(p * log p))`. A frequency penalty is a `ScatterAdd`
histogram. A nucleus variant is `SortDesc` + `CumSum` + threshold.
**No Tier A candidate needs a new opcode.**

---

## Summary

Effort is relative: **S** = a sampler-sized PTIR program; **M** = program +
per-sequence channel state or mask plumbing; **L** = needs a driver/ABI change
or multi-context orchestration.

| ID | Candidate | Tier | Engine work | Effort | Status |
|---|---|---|---|---|---|
| [A1](#a1) | Locally typical sampling | A | none | S | ✅ `locally-typical-sampling` |
| [A2](#a2) | η- and ε-sampling | A | none | S | ✅ `eta-epsilon-sampling` |
| [A3](#a3) | Tail-free sampling | A | none | S | ✅ `tail-free-sampling` |
| [A4](#a4) | Top-a sampling | A | none | S | ✅ `top-a-sampling` |
| [A5](#a5) | XTC (exclude top choices) | A | none | S | ✅ `xtc-sampling` |
| [A6](#a6) | Frequency / presence / repetition penalty | A | none | M | ✅ `repetition-penalty` |
| [A7](#a7) | DRY repetition penalty | A | none | M | ✅ `dry-repetition-penalty` |
| [A8](#a8) | Entropy-adaptive temperature | A | none | M | ✅ `entropy-adaptive-temperature` |
| [A9](#a9) | Gumbel distortion-free watermark | A | none | M | ✅ `gumbel-watermark` |
| [A10](#a10) | SynthID-Text tournament sampling | A | none | M | ✅ `synthid-tournament-sampling` |
| [B1](#b1) | H2O heavy-hitter eviction | B | **attention/key tap** | M | ⛔ blocked |
| [B2](#b2) | SnapKV | B | **attention/key tap** | M | ⛔ blocked |
| [B3](#b3) | TOVA | B | **attention/key tap** | M | ⛔ blocked |
| [B4](#b4) | Quest query-aware page selection | B | **key tap** | M | ⛔ blocked |
| [B5](#b5) | RetrievalAttention | B | **key tap** | L | ⛔ blocked |
| [C0](#c0) | **Bind `hidden`/`value_head` in CUDA tier-0** | C | driver | M | out of scope |
| [C1](#c1) | EAGLE-style feature-level drafting | C | after C0 | L | out of scope |
| [C2](#c2) | Semantic entropy / confidence gating | C | after C0 | M | out of scope |
| [C3](#c3) | Value-head guided beam / step scoring | C | after C0 | M | out of scope |
| [C4](#c4) | DoLa layer-contrastive decoding | C | after C0 + per-layer readout | L | out of scope |
| [C5](#c5) | ITI / CAA / RepE activation steering | C | **blocked** — needs a write port | L | ⛔ blocked |
| [D1](#d1) | Classifier-free guidance for LLMs | D | none | M | ✅ `classifier-free-guidance` |
| [D2](#d2) | Context-aware decoding (CAD) | D | none | M | ✅ `context-aware-decoding` |
| [D3](#d3) | Cross-model contrastive decoding | D | **multi-model service** | L | ⛔ blocked |
| [D4](#d4) | DExperts / proxy tuning / emulated fine-tuning | D | **multi-model service** | L | ⛔ blocked |
| [E1](#e1) | Grammar-aligned decoding (ASAp) | E | none | L | ✅ `asap-grammar-aligned-decoding` |
| [E2](#e2) | Token healing / tokenizer alignment | E | none | M | ✅ `token-healing` |

**14 of 14 buildable candidates are implemented**, each verified by a build for
`wasm32-wasip2` and a run against a live CUDA engine. Every crate lives in
`tests/inferlets/<name>/`. Faithfulness against the source papers is audited in
[`10-implementation-faithfulness-audit.md`](10-implementation-faithfulness-audit.md).

The nine unbuilt rows are **not** backlog. Tier C is out of scope by request;
B1–B5, C5, D3 and D4 are blocked on missing ABI surface, and the exact missing
primitive for each is recorded in
[`09-ptir-unbuilt-algorithms.md`](09-ptir-unbuilt-algorithms.md).

**Order actually taken:** `A1/A2 → A3/A4/A5 → A6 → A7 → A8 → A9 → A10 → D1 →
D2 → E2 → E1`. B4 was dropped from the plan once the attention tap turned out
to be missing; C0 and its dependants were out of scope.

---

## Tier A — pure `logits()`, buildable today

<a id="a1"></a>
### A1. Locally typical sampling — S
- **Paper:** Meister et al., [2202.00666](https://arxiv.org/abs/2202.00666), *Locally Typical Sampling*
- **Build:** keep tokens whose information content is closest to the distribution's entropy.
- **PTIR:** `H = neg(ReduceSum(p * log p))` → score `abs(log p + H)` → `SortDesc` → `CumSum` → threshold → `mask_apply`.
- **Gap:** the engine ships Argmax / TopP / TopK / MinP / TopKTopP / Multinomial only.
- **Done when:** matches a reference NumPy implementation token-for-token at fixed seed.

<a id="a2"></a>
### A2. η- and ε-sampling — S
- **Paper:** Hewitt et al., [2210.15191](https://arxiv.org/abs/2210.15191), *Truncation Sampling as Language Model Desmoothing*
- **Build:** ε = absolute probability floor; η = `min(ε, sqrt(ε)·exp(−H))`.
- **PTIR:** reuse A1's entropy reduction, then a single comparison mask.
- **Gap:** no entropy-adaptive truncation exists.

<a id="a3"></a>
### A3. Tail-free sampling — S
- **Paper:** community method (no canonical paper — cite the implementation).
- **Build:** cut the tail where the sorted-probability curve flattens.
- **PTIR:** `SortDesc` → first and second differences via shifted `sub` → normalize → `CumSum` threshold.

<a id="a4"></a>
### A4. Top-a sampling — S
- **Paper:** community method.
- **Build:** threshold proportional to `p_max²`, so peaked distributions truncate harder.
- **PTIR:** one `ReduceMax`-style reduction, one `mul`, one comparison.

<a id="a5"></a>
### A5. XTC (exclude top choices) — S
- **Paper:** community method.
- **Build:** with probability `p`, drop all but the least-likely above-threshold candidate — a creativity knob.
- **PTIR:** threshold mask + a Bernoulli draw from the existing RNG stream.

<a id="a6"></a>
### A6. Frequency / presence / repetition penalty — M ⭐ start here
- **Paper:** standard practice (CTRL, Keskar et al.; OpenAI API semantics).
- **Build:** penalize tokens by how often they already appeared.
- **PTIR:** `ScatterAdd` histogram over emitted tokens carried in a device-advanced channel; subtract `α·count + β·present` from logits before sampling.
- **Gap:** **these are not in the `Sampler` surface at all** — the most conspicuous functional hole, and every other engine has them.
- **Done when:** parity with vLLM's `frequency_penalty` / `presence_penalty` / `repetition_penalty` on a fixed prompt set.

<a id="a7"></a>
### A7. DRY repetition penalty — M
- **Paper:** community method (widely deployed in local-inference stacks).
- **Build:** penalize tokens that would extend a repeated n-gram, scaled by match length.
- **PTIR:** suffix match against the emitted-token channel via `Gather` + `eq`, then a length-scaled penalty.
- **Note:** the strongest demonstration that per-token *sequence* state belongs on device.

<a id="a8"></a>
### A8. Entropy-adaptive temperature — M ⭐
- **Papers:** EDT, [2403.14541](https://arxiv.org/abs/2403.14541); AdapT, [2309.02772](https://arxiv.org/abs/2309.02772)
- **Build:** `T = T0 · N^(θ/H)` with `0 < N < 1` — raise temperature when the model is *confused*, lower it when it is confident. High entropy means the model cannot pick a good continuation anyway, so exploring costs little; low entropy means it should commit.
- **PTIR:** entropy reduction (already proven by `entropycheck`) feeding the temperature path as a control word (already proven by `mirostat`).
- **Gap:** `entropycheck` *measures* entropy and nothing *acts* on it. This candidate is literally connecting two existing inferlets.

<a id="a9"></a>
### A9. Gumbel distortion-free watermark — M ⭐
- **Papers:** Kuditipudi et al., [2307.15593](https://arxiv.org/abs/2307.15593), *Robust Distortion-free Watermarks*; Aaronson's Gumbel scheme.
- **Build:** derive the Gumbel noise key from `hash(secret, context)` instead of the RNG counter — output distribution is provably unchanged, unlike greenlist watermarking which shifts it.
- **PTIR:** the CUDA driver **already runs keyed Gumbel-max sampling** with a `[key, ctr]` state driving the noise. This is unusually close to free.
- **Gap:** only the distribution-shifting greenlist scheme ([2301.10226](https://arxiv.org/abs/2301.10226)) is implemented.

<a id="a10"></a>
### A10. SynthID-Text tournament sampling — M
- **Paper:** Dathathri et al., *Nature* 2024, *Scalable watermarking for identifying large language model outputs* (no arXiv preprint).
- **Build:** multi-round tournament over candidate tokens using keyed g-functions.
- **Why interesting here:** explicitly designed to remain compatible with speculative decoding — and pie can compose the two (`mtp-grammar` already proves masks compose with speculation).

---

## Tier B — score-driven KV policies: **blocked, and not on effort**

> **Correction (verified against the interface, not inferred).** This tier was
> originally listed as buildable on the strength of the `query()` tap. That is
> wrong, and none of B1–B5 can be written today. The distinction that matters:
>
> - **The actuator already exists.** `Port::Pages`, `Port::PageIndptr` and
>   `Port::AttnMask` (`compiler/ir/src/registry.rs:102-113`) are all
>   guest-bound, so an inferlet can freely choose *which* pages a step attends
>   to and mask *which* positions survive. Eviction and page selection are
>   expressible.
> - **The sensor does not.** Every one of B1–B5 decides *what* to evict from
>   `softmax(QK^T)` — accumulated attention mass (H2O), a recent-window score
>   (SnapKV), the current step's score (TOVA), a per-page bound on `q·k`
>   (Quest), or a nearest-neighbour lookup over `K` (RetrievalAttention).
>   `IntrinsicId` (`compiler/ir/src/op.rs:49-68`) is exhaustively
>   `Logits, MtpLogits, Hidden, Query, ValueHead, Layer, MtpDrafts`. There is
>   **no key, no attention score and no attention probability tap**, and
>   `Port` has no KV-data port either — `Pages`/`PageIndptr`/`KvLen` carry page
>   *geometry*, never page *contents*.
>
> `query()` alone is therefore insufficient by construction: `QK^T` needs `K`,
> and `K` is not observable from an inferlet. Nor can `K` be reconstructed —
> `Hidden` is epilogue-scoped (`registry.rs:185` restricts `Query`/`Layer` to
> `Stage::OnAttnProj | Stage::OnAttn`, while `Hidden` is epilogue-only), and the
> per-layer `W_k` projection is not exposed in any form.
>
> **What would unblock the whole tier:** one new intrinsic carrying either the
> post-softmax attention probabilities or the layer's projected keys, readable
> at `Stage::OnAttn`. Every algorithm below then becomes a small inferlet,
> because the hard half — paged KV plus a guest-bound attention mask — is
> already in place. This is recorded in
> [`09-ptir-unbuilt-algorithms.md`](09-ptir-unbuilt-algorithms.md).

Pie today has **static** attention policies (`attention-sink`,
`sliding-window-attention`, upstream `windowed-attention`). It has no
**score-driven dynamic** policy, and cannot have one until the tap above lands.

<a id="b1"></a>
### B1. H2O — heavy-hitter KV eviction — M
- **Paper:** Zhang et al., [2306.14048](https://arxiv.org/abs/2306.14048)
- **Build:** accumulate attention mass per position, evict the low-mass tail.
- **Blocked on:** the accumulated attention mass itself. `query()` gives `Q`
  but never `K`, so `softmax(QK^T)` cannot be formed.

<a id="b2"></a>
### B2. SnapKV — M
- **Paper:** Li et al., [2404.14469](https://arxiv.org/abs/2404.14469)
- **Build:** use an observation window of recent attention to pick which KV each head keeps, at prefill time.
- **Blocked on:** per-head attention scores over the observation window.

<a id="b3"></a>
### B3. TOVA — M
- **Paper:** Oren et al., [2401.06104](https://arxiv.org/abs/2401.06104), *Transformers are Multi-State RNNs*
- **Build:** a single step's attention score decides the evicted token — the
  simplest score-driven policy, and so the natural first client of an attention
  tap once one exists.
- **Blocked on:** that single step's attention score.

<a id="b4"></a>
### B4. Quest — query-aware page selection — M ⭐
- **Paper:** Tang et al., [2406.10774](https://arxiv.org/abs/2406.10774)
- **Build:** keep per-page min/max key summaries, score them against the current query, attend only to the top-k pages.
- **Why pie:** the KV is *already* paged and the attention mask is *already* a
  guest-bound channel. This is closer to a natural expression here than in any
  other engine — the strongest showcase in Tier B, *once* the tap exists.
- **Blocked on:** the per-page min/max key summaries. Quest scores a page by
  `max_k q·k` bounded through elementwise key extrema; keys are unreadable.

<a id="b5"></a>
### B5. RetrievalAttention — L
- **Paper:** Liu et al., [2409.10516](https://arxiv.org/abs/2409.10516)
- **Build:** ANN index over keys, retrieve the subset the query actually attends to, offload the rest.
- **Effort:** L because it needs an index structure and host/device memory movement, not just a mask.
- **Blocked on:** read access to `K` to build the ANN index at all.

---

## Tier C — needs `hidden()` / `value_head()` bound first

<a id="c0"></a>
### C0. Prerequisite: bind `hidden` and `value_head` in the CUDA tier-0 runner — M
- **Change:** add `hidden` and `value_head` to `FireInputs`
  (`driver/cuda/src/pipeline/tier0/tier0_runner.hpp`) and bind them in the
  epilogue, mirroring `driver/dummy/src/lib.rs` and
  `driver/metal/tests/pipeline_interp_test.cpp`, which already handle both.
- **Unblocks:** C1-C4.

> **Honest limitation.** `hidden()` is documented as *"the residual stream at
> read-out (epilogue)"* — a **read** tap after the last layer. That is enough
> for C1-C3. It is **not** enough for mid-network activation steering, which
> needs a per-layer **write** port that does not exist in the ABI. See C5.

<a id="c1"></a>
### C1. EAGLE-style feature-level drafting — L
- **Paper:** Li et al., [2401.15077](https://arxiv.org/abs/2401.15077)
- **Build:** draft from hidden-state features rather than sampled tokens, which is why EAGLE beats token-level drafting.
- **Gap:** pie has native MTP drafting but nothing feature-level.

<a id="c2"></a>
### C2. Semantic entropy / confidence-gated decoding — M
- **Build:** score or cluster `hidden()` to gate generation, abstain, or trigger a fallback.
- **See:** `06-constrained-decoding-and-steering.md`.

<a id="c3"></a>
### C3. Value-head guided beam / step scoring — M ⭐
- **Build:** evaluate `value_head()` on device so each search node costs no extra forward pass.
- **Why pie:** combines with `ctx.fork()` for branches and the credit/bid market for per-branch budget — the three-primitive composition nothing else can express.
- **See:** `03-search-and-meta-generation.md`.

<a id="c4"></a>
### C4. DoLa — layer-contrastive decoding — L
- **Paper:** Chuang et al., [2309.03883](https://arxiv.org/abs/2309.03883)
- **Build:** contrast mature-layer against premature-layer logits to improve factuality.
- **Also needs:** a per-layer LM-head readout on top of `layer()`, not just the tap.

<a id="c5"></a>
### C5. ITI / CAA / ActAdd / RepE activation steering — L, **currently blocked**
- **Papers:** ITI [2306.03341](https://arxiv.org/abs/2306.03341); ActAdd [2308.10248](https://arxiv.org/abs/2308.10248); RepE [2310.01405](https://arxiv.org/abs/2310.01405)
- **Blocker:** requires **writing** a steering vector into the residual stream mid-forward. The ABI has read taps only.
- **Why it is still on the list:** this is the single most compelling reason to extend PTIR with a per-layer write port. No serving system can express it today — see `08-pie-uniqueness-matrix.md`.

---

## Tier D — two contexts combined in one step

Pie can bind heterogeneous passes into one frame, so combining two logit rows
before sampling is a natural PTIR program. Only `contrastive-decoding` does
anything like this today, and it is *same-model, bounded-context*.

<a id="d1"></a>
### D1. Classifier-free guidance for LLMs — M
- **Paper:** Sanchez et al., [2306.17806](https://arxiv.org/abs/2306.17806), *Stay on topic with Classifier-Free Guidance*
- **PTIR:** `logits = uncond + γ·(cond − uncond)` over two contexts in one frame.

<a id="d2"></a>
### D2. Context-aware decoding (CAD) — M
- **Paper:** Shi et al., [2305.14739](https://arxiv.org/abs/2305.14739), *Trusting Your Evidence*
- **Build:** contrast with-context against without-context logits — the anti-hallucination form of D1, same program shape.

> **Correction (verified).** D1 and D2 are built and passing. D3 and D4 are
> **blocked, and not on effort**: both require two or more *different* models to
> be resident and steppable from one inferlet, and
> `sdk/rust/inferlet/wit/model.wit:3` states the invariant plainly — *"The
> engine serves exactly one model; these are global functions over that single
> bound model (no `model` resource handle)."* There is no model handle to pass
> around, so no inferlet can name a second model. See
> [`09-ptir-unbuilt-algorithms.md`](09-ptir-unbuilt-algorithms.md).

<a id="d3"></a>
### D3. Cross-model contrastive decoding — L, **blocked**
- **Paper:** Li et al., [2210.15097](https://arxiv.org/abs/2210.15097)
- **Gap:** the existing `contrastive-decoding` inferlet is one model at two context lengths; the original method contrasts an **expert and an amateur model**.
- **Blocked on:** one model per engine. Two contexts are expressible (that is
  exactly what D1/D2 do); two *models* are not.

<a id="d4"></a>
### D4. DExperts / proxy tuning / emulated fine-tuning — L, **blocked**
- **Build:** logit arithmetic across two or three models to transfer tuning effects at decode time.
- **Blocked on:** one model per engine — same wall as D3, and it binds harder
  here because proxy tuning needs *three* models.
- **See:** `02-token-level-decoding-sampling.md`, `06-constrained-decoding-and-steering.md`.

---

## Tier E — constrained-decoding correctness

Pie has strong constrained decoding (`grammar`, `grammar-late`, `grammarmb`,
`json-schema-constrained-decoding`, `mtp-grammar`). What it lacks is the
*correctness* literature on top of it.

<a id="e1"></a>
### E1. Grammar-aligned decoding (ASAp) — L
- **Paper:** Park et al., [2405.21047](https://arxiv.org/abs/2405.21047), *Grammar-Aligned Decoding*
- **Problem it fixes:** hard masking provably distorts the model's distribution — outputs are grammatical but their likelihoods are wrong.
- **Needs:** constraint state **plus** KV truncation and resampling. Pie has both (`truncate`, snapshot, grammar matcher), so this is the most defensible correctness contribution on the list.

<a id="e2"></a>
### E2. Token healing / tokenizer alignment — M
- **Problem:** masking at token granularity mishandles prefixes that split across token boundaries, a known correctness bug in every constrained-decoding stack.
- **Gap:** no inferlet addresses it.

---

## Excluded — already covered

Not proposed, because an existing inferlet demonstrates it.

| Area | Covered by |
|---|---|
| argmax / top-p / top-k / min-p / top-k-top-p / multinomial / temperature | `sampling-primitives`, `multisamp`, `tempgen`, `isolatedtopp`, upstream `sampler-suite` |
| **mirostat v2** (the template for Tier A) | `mirostat`, `mirostat-v2-sampling` |
| greenlist watermarking | `greenlist-watermarking`, upstream `watermarking`, `demo-watermark` |
| grammar / JSON-schema constrained decoding, late masking, spec composition | `grammar`, `grammar-late`, `grammarmb`, `json-schema-constrained-decoding`, `mtp-grammar` |
| speculative decoding (self-spec, MTP verify, n-gram cacheback, Jacobi) | `selfspec`, `specverify`, `mtpverify`, `mtp-native-verify`, `mtp-specdecode`, `cacheback-speculative-decoding`, upstream `jacobi-decoding` |
| beam search incl. logical mask-out + lazy compaction | `beam`, `beam-baseline`, `beam-designb`, `beam-designb-compact`, `beam-search` |
| attention sinks, sliding window, prefix-tree KV, prefix-cache grafting | `attention-sink`, `sliding-window-attention`, `prefix-tree-kv-cache`, `prefix-cache-e2e`, upstream `windowed-attention`, `hierarchical-attention-*` |
| entropy **measurement** | `entropycheck` — but nothing consumes it, hence A8 |
| meta-generation (host-side, out of scope) | upstream `tree-of-thought`, `graph-of-thought`, `mcts-*`, `best-of-n`, `reflexion`, `demo-self-correct`, `skeleton-of-thought`, `recursion-of-thought`, `parallel-generation` |

---

## How this list was derived

Reproducible, not from memory:

```bash
# which PTIR intrinsics do existing inferlets actually call?
grep -rh "intrinsics::" tests/inferlets runtime/engine/tests/inferlets --include=*.rs \
  | sed 's/.*intrinsics::\([a-z_]*\).*/\1/' | sort | uniq -c | sort -rn

# does the CUDA driver bind each intrinsic?
grep -n "struct FireInputs" -A 25 driver/cuda/src/pipeline/tier0/tier0_runner.hpp

# is algorithm X mentioned anywhere in either inferlet set?
grep -rli "<name>" tests/inferlets runtime/engine/tests/inferlets /root/pie/inferlets
```

Baseline inventory: 14 curated (`tests/inferlets/`) + 34 engine tests
(`runtime/engine/tests/inferlets/`) in this branch, plus the 76 upstream
examples in `pie-project/pie@main:inferlets/`, so nothing is proposed that
upstream already demonstrates.

All 20 arXiv citations above were resolved against the arXiv API and matched
their registered titles.
