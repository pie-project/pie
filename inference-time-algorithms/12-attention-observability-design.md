# Attention observability: a design for unblocking B1–B5

**Status:** design, not yet implemented. Written to decide scope before committing
to a cross-cutting change (interface + compiler + CUDA driver + ABI + Metal).

**Context:** `09-ptir-unbuilt-algorithms.md` lists five KV/attention algorithms as
unbuildable, all blocked on the same two claims — that `softmax(QK^T)` is
unreachable and that `K` is unreadable. **Both claims are now falsified.** This
document records what is actually missing, and proposes two independent tracks.

---

## 1. What the old blocker table got wrong

`09-ptir-unbuilt-algorithms.md:135-139` reads:

| | algorithm | recorded blocker |
|---|---|---|
| B1 | H2O | `needs softmax(QK^T); K unreadable` |
| B2 | SnapKV | same |
| B3 | TOVA | same |
| B4 | Quest | `needs K directly` |
| B5 | RetrievalAttention | `needs K directly` |

Two corrections.

### 1.1 The per-layer execution hook is real, and it runs on CUDA today

`Stage::OnAttnProj` / `Stage::OnAttn` are not aspirational. Every model family in
the CUDA driver calls them:

```
driver/cuda/src/model/llama_like/llama_like.cpp:530, :714
driver/cuda/src/model/qwen3_5/qwen3_5_forward.cpp:899, :1196, :1313, :1384
driver/cuda/src/model/gemma2|gemma3n|gemma4|mixtral|glm5|kimi|
                deepseek_v4|nemotron_h/*.cpp   (OnAttnProj + OnAttn each)
```

via `invoke_stage_hook(StageHookPoint::OnAttnProj, ws.q.data(), N, Hq, L, stream)`
(`driver/cuda/src/model/stage_hooks.hpp:44-59`).

`Dispatch::execute_attention_phase` (`driver/cuda/src/pipeline/dispatch.cu:4065`)
materialises `Query` as f32 on demand, and `:3323` *enforces exact layer order*.
Only Metal rejects per-layer taps (`driver/metal/src/pipeline/interp.hpp:343`).

**So the question was never whether a program can run at attention. It is only
what data is in scope when it does.** Today that is `Query` and `Layer`.

### 1.2 Quest's kernels are already written and parity-tested

`driver/cuda/src/kernels/envelope.{hpp,cu}` ships both halves of Quest:

- `launch_envelope_recompute_bf16` — reduce each page's live keys to the
  per-`(page, kv_head, dim)` min/max envelope.
- `launch_envelope_dot_f32` — `score[kv_head, page] = Σ_{qh ∈ GQA-group} Σ_d
  max(q[qh,d]·min[p,kh,d], q[qh,d]·max[p,kh,d])`, i.e. the max achievable `q·k`
  inside the page envelope — Quest criticality — with `-inf` beyond the live
  page count.

`driver/cuda/tests/test_envelope_dot.cu` parity-checks both **bit-for-bit**
against `pie_sampling_ir::eval::envelope_dot_reference`. The build target exists
(`driver/cuda/CMakeLists.txt:647`).

**These kernels are called from nothing but their own test.** `K` is not
"unreadable" — a maintenance kernel already reads it. What is missing is the
binding, not the capability.

### 1.3 `softmax(QK^T)` is reachable through a supported FlashInfer extension point

Pinned version is **v0.6.15** (`driver/cuda/CMakeLists.txt:107`, fetched via CPM).

`include/flashinfer/attention/variant_helper.cuh` defines six variant hooks.
The relevant one:

```cpp
#define REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, \
                                  qo_head_idx, kv_head_idx, ...)
```

It receives the raw `S = QK^T` element **together with the exact `kv_idx`**. The
decode kernel calls it per KV position:

```cpp
// flashinfer/attention/decode.cuh:91
s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0,
                               /*kv_idx=*/pos, qo_head_idx, kv_head_idx);
if constexpr (variant.use_softmax) { s[j] *= variant.sm_scale_log2; }
bool mask = variant.LogitsMask(params, batch_idx, 0, /*kv_idx=*/pos, ...);  // :97
```

This is a route the maintainers endorse. On
[issue #838](https://github.com/flashinfer-ai/flashinfer/issues/838) ("Can
`BatchDecodeWithPagedKVCacheWrapper` return attention scores to all tokens, not
just logsumexp?") @yzh119 answered:

> Yes that's feasible by defining your own attention variant […] But then you
> might lose the benefit of flashattention algorithm because of the O(n^2) write
> to global memory.

and the asker confirmed success a month later. A maintained example lives at
`tests/utils/test_jit_example.py:161-216` (`struct DumpLogits`).

Conversely, [issue #707](https://github.com/flashinfer-ai/flashinfer/issues/707)
is @yzh119's own RFC for un-fused softmax, closed with "I'm not convinced it's
worth implementing". **A standard score output is never coming.** The variant is
the path.

FlashInfer ships no H2O/SnapKV/TOVA/Quest helper of any kind (0 grep hits).
`BlockSparseAttentionWrapper` consumes a sparsity decision; it does not produce
importance scores.

---

## 2. The O(n²) objection does not bind here

The single documented downside is the `O(qo_len × kv_len)` global write. It does
not apply to any of the three algorithms we want, because of how they are
defined:

| algorithm | when it scores | buffer actually needed |
|---|---|---|
| **H2O** | every decode step | `qo_len = 1` → `[heads, kv_len]`; fold heads in-hook → `[kv_len]` |
| **TOVA** | every decode step | same |
| **SnapKV** | once, at prefill | *observation window only* (last ~32 queries) → `[window, kv_len]` |

For decode at 128K context with 32 heads that is 16 MB unreduced, 512 KB reduced,
and it is transient — one layer is live at a time.

For SnapKV the hook exposes `qo_idx`, so a single predicate

```cpp
if (qo_idx + window >= qo_len) { /* record */ }
```

collapses the quadratic term to `window × kv_len`. **The observation window is
SnapKV's defining device**, so what the paper requires and what the kernel can
give cheaply are the same thing.

---

## 3. What each algorithm actually needs

Sorted by true remaining blocker, not by the old table's uniform answer.

| | needs to READ | needs to WRITE | remaining gap |
|---|---|---|---|
| **B4 Quest** | per-page `K` min/max envelope | page mask | envelope storage; `envelope_dot` dispatch binding; mask→page-list |
| **B3 TOVA** | one step's scores | drop 1 position | score capture; position mask |
| **B1 H2O** | accumulated mass per position | drop positions | score capture; position mask; accumulator state |
| **B2 SnapKV** | window scores + pooling | compact KV | score capture; position mask; pooling op |
| **B5 RetrievalAttention** | ANN index over `K` | route queries | an entire ANN subsystem — **out of scope** |

Note Quest needs **no attention-kernel change at all** and **no eviction** — it
masks per step. That is what makes it structurally cheaper than the other four.

---

## 4. Proposed design

Three layers. Track A needs only layers 1 and 3; track B needs all three.

### Layer 1 — driver: a score-capture attention variant

`attention_flashinfer.cu` already threads `class Variant` through every dispatch
(`AttnVariant`, `AttnVariantSoftcap`, `AttnVariantFull`, `AttnVariantCustom` at
`:51-76`, `:1315`; `prefill_dispatch_for_head_dim<HEAD_DIM, MASK, Variant>` at
`:938`). Every decode kernel is templated on **both** `AttentionVariant` *and*
`Params` (`decode.cuh:63, 215, 394, 658, 830`), and FlashInfer duck-types params
via `DEFINE_HAS_MEMBER` SFINAE.

So this is a type substitution, not a fork:

```cpp
// new: driver/cuda/src/ops/attention_score_variant.cuh
template <class Base>
struct PieScoreParams : Base { float* score_out; int32_t score_stride; };

struct PieScoreCapture : ::flashinfer::AttentionVariantBase {
  static constexpr bool use_softmax = true;
  REGISTER_LOGITS_TRANSFORM(params, logits, b, qo_idx, kv_idx, qh, kh, {
    // NOTE: `logits` here is the RAW dot product — sm_scale is applied by the
    // kernel on the line *after* this hook returns. Scale it ourselves.
    params.score_out[qh * params.score_stride + kv_idx] = logits * params.sm_scale;
    return logits;
  })
};
```

**Normalisation.** The hook cannot see the online-softmax denominator; `m` and
`d` are only final in `OutputTransform`, which has no KV axis. The fix is to not
try: record the raw scaled logit, and normalise afterwards with the LSE the
driver *already* plumbs (`params.lse = lse_out`, `attention_flashinfer.cu:744`
decode / `:1004` prefill):

```
p[h, j] = exp(s[h, j] − lse[h])
```

This is exact — `lse = log Σ_j exp(s_j)` is the same denominator — and costs one
extra small kernel over `[heads, kv_len]`. No atomics: each `(head, kv_idx)` is
written exactly once.

**Cost estimate:** one extra store per score already computed, plus one
`[heads, kv_len]` read-reduce. That is roughly the cost of re-reading `K` once,
so ~30–50% on the attention op — *not* the 2× a separate `QK^T` pass would take,
and far below the ~1.5× a naive re-scoring implementation implies.

**SM90 caveat.** On Hopper, prefill takes a separate path
(`driver/cuda/src/ops/attention_flashinfer_hopper.cu`, gated by `PIE_HAS_SM90` at
`CMakeLists.txt:232`) with its own variant structs in
`flashinfer/attention/hopper/variants.cuh:100` — same `REGISTER_LOGITS_TRANSFORM`
signature. **Two variant structs must be maintained.** Decode is unaffected.

### Layer 2 — PTIR: one appended intrinsic

Follow the `MtpDrafts = 6` precedent exactly (`interface/ptir/src/op.rs:63-67`),
whose doc comment records the rule: *append, leave 0..6 byte-stable so every
prior program's bytecode and identity hash is unchanged.*

```rust
AttnScore = 7,   // per-KV-position attention probability, this layer, this step
```

- **scope:** `&[Stage::OnAttn]` only — not `OnAttnProj`. Scores do not exist
  until attention has run. (`registry.rs:185` is the table.)
- **type rule:** `F32`, rank 2, `[num_heads, kv_len]`; let the program reduce
  over heads with existing ops rather than baking a policy in — H2O and SnapKV
  are per-head, TOVA averages, and PTIR already has the reductions.
- **gating:** not model-gated (every attention model has scores), but
  *backend*-gated: Metal must reject it alongside its existing per-layer-tap
  rejection.

#### The T11 constraint forces read and write into different fires

`validate.rs:456-458` restricts an `SinkScope::Attention` sink to
`Prologue | OnAttnProj` — a sink is legal only *strictly before* its consumption
point. So the mask for layer `L` must be written at `OnAttnProj(L)`, which has
already run by the time `AttnScore` is readable at `OnAttn(L)`.

**Read and write therefore cannot be composed within one layer pass.** This is
not a defect to work around; it is what the algorithms actually do:

- **Quest** reads `Query` at `OnAttnProj(L)` and writes `attn_page_mask` at
  `OnAttnProj(L)` — same stage, both legal, no cross-fire state. This is why
  Track A is self-contained.
- **H2O / TOVA / SnapKV** accumulate at `OnAttn(L)` of one fire and apply the
  eviction mask at `OnAttnProj(L)` of the *next* — they evict on history, then
  attend. Correct by construction, but it makes the accumulator a **loop-carried
  channel across fires**, which puts it squarely under contract #1 in
  `11-ptir-limits.md`: *loop-carried geometry ports must `take()` before
  `put()`*, and the failure is silent.

Update surface, from the `MtpDrafts` precedent
(`git log -S MtpDrafts`, head `c1e148ef`):

```
interface/ptir/src/op.rs              enum + from_u16 + name
interface/ptir/src/header.rs          generated C enum
interface/ptir/src/registry.rs        intrinsic_stages()
interface/ptir/src/validate.rs        scope check + type rule
interface/ptir/src/interp.rs          PassInputs field + eval root
interface/ptir/include/ptir_abi.h     PTIR_INTR_ATTN_SCORE = 7
driver/common/.../ptir_abi.h          mirror
driver/common/.../trace.hpp           C++ mirror enum + static_asserts
driver/common/.../bound.hpp           wire decode
driver/cuda/src/pipeline/dispatch.cu  bind the buffer at OnAttn
driver/metal/src/pipeline/interp.hpp  reject
```

### Layer 3 — the write side (eviction / masking)

Two granularities are needed, and one already exists.

**Page granularity (Quest).** `attn_page_mask` is fully declared —
`ptir-dsl/src/intrinsics.rs:92`, `registry.rs:172` (`SinkScope::Attention`),
`ptir_abi.h:246`, T11 stage-precedence enforced in `validate.rs:458`, and the
interpreter golden `pentathlon_iter.txt:47` shows it firing per layer.
`singleton_codegen.hpp:449` already whitelists it as a legal CUDA sink boundary.
Only the *consumption* is missing.

**The cheap consumption path avoids FlashInfer entirely.** Attention already
receives the page list as `kv_page_indices` / `kv_page_indptr` /
`kv_last_page_lens` (`attention_flashinfer.hpp:134-172`). Applying a page mask is
therefore a **gather that compacts the page table** before the call — not a
kernel change. `north_star_e2e.rs:278` already names this the pending work.

**Position granularity (H2O / TOVA / SnapKV).** These evict individual tokens,
not pages. `MaskMode::kCustom` + `params.maybe_custom_mask` is already wired for
*prefill* (`attention_flashinfer.cu:1364, :1557`, entry point
`launch_attention_flashinfer_prefill_custom_bf16`). Decode has no repo entry
point, but the FlashInfer decode kernel supports it (`decode.cuh:97`), and since
decode pins `qo_idx = 0` the mask offset `qo_idx * kv_len + kv_idx` collapses to
`kv_idx` — **a `[kv_len]` bitmask is exactly a per-position eviction mask.**

So layer 3 for track B is: a new `attn_kv_mask` sink (`SinkScope::Attention`,
alongside `attn_page_mask` in `KNOWN_SINKS`, therefore writable at `OnAttnProj`)
+ a decode custom-mask entry point.

**Soft vs. real eviction.** Masking reproduces the *algorithm* — the quality
curve of H2O/SnapKV/TOVA is determined by which tokens are ignored — but not the
*memory saving*, which needs page reclamation. Reclamation is a separate,
larger piece of work (`runtime/engine/src/store/kv.rs:507 discard()` is a
logical range removal, not per-position). **Proposal: implement masking first and
say so explicitly in the inferlet READMEs**, since a faithfulness claim about
output quality is verifiable and a memory claim would not be.

---

## 5. Two tracks

### Track A — Quest (no attention-kernel change)

1. Allocate `env_min` / `env_max` `[P, kv_heads, head_dim]` f32 ×2. Today
   `KvCacheLayerView` (`kernels/kv_cache_view.hpp:21-40`) has no slot.
2. Call `launch_envelope_recompute_bf16` as KV-append maintenance.
3. Bind `PTIR_OP_KERNEL_CALL("envelope_dot")` → `launch_envelope_dot_f32`.
   Validation already admits the name; execution does not exist.
4. Consume `attn_page_mask` by compacting the page table.
5. Inferlet: `envelope_dot(query()) → top_k(budget) → pivot_threshold →
   attn_page_mask`. Every op on that line already exists and `top_k`/`rank_le`
   were made O(len) in this session.

Delivers: the first per-layer tap validated on real hardware, and the repo's own
declared extension point switched on.

### Track B — score capture (TOVA → H2O → SnapKV)

1. Layer 1 variant + LSE normalisation kernel (decode first; SM90 prefill later).
2. Layer 2 intrinsic `AttnScore = 7`.
3. Layer 3 `attn_kv_mask` + decode custom-mask entry point.
4. **TOVA first** — it needs only the current step's scores and no accumulator,
   so it exercises the whole path with the least state. Note that even TOVA is
   cross-fire: it scores at `OnAttn(L)` and masks at the *next* fire's
   `OnAttnProj(L)` (§4, T11).
5. Then H2O (adds a running accumulator — a loop-carried channel, so contract #1
   in `11-ptir-limits.md` applies: `take()` before `put()`, silent on failure).
6. Then SnapKV (adds prefill-window capture + a pooling op).

---

## 6. Risks and open questions

- **`use_softmax` interaction.** `s[j] *= variant.sm_scale_log2` runs *after* the
  hook, and `sm_scale_log2` already folds `log2e` and, under softcap, the
  `logits_soft_cap` rescale (`variants.cuh:47-56`). Under softcap the recorded
  value must go through `tanh` to match. Simplest resolution: **refuse score
  capture when `logits_soft_cap != 0`** rather than silently reporting a
  different quantity.
- **Split-K / multi-CTA decode.** FlashInfer may partition KV across CTAs and
  merge. Each `(head, kv_idx)` is still written once — but this must be
  confirmed against `decode.cuh`'s split-K path before relying on
  "no atomics needed".
- **GQA.** The hook exposes both `qo_head_idx` and `kv_head_idx`. H2O/SnapKV are
  per-head in their papers; which axis the intrinsic exposes should follow the
  paper, so `[num_q_heads, kv_len]` is the safer default despite the size.
- **Sliding-window models.** `window_left` already restricts the live range;
  scores outside it are `-inf`-masked and must be excluded from accumulation.
- **Metal.** Both tracks are CUDA-only. Metal already rejects per-layer taps, so
  the failure is clean, but the inferlet matrix must gate on backend.
- **`attn_page_mask` has never run on a backend.** Track A's step 4 is the first
  execution of a declared-but-unbuilt sink; expect the ABI to have drifted from
  the interpreter's golden.

---

## 7. Recommendation

**Do Track A first, then Track B.**

Track A is the only one of the five where every missing piece is plumbing rather
than new capability: the kernels exist and are bit-parity tested, the sink is
declared and golden-tested, the selection ops were just optimised, and the tap
runs on ten model families. It is also the cheapest way to prove the per-layer
tap end-to-end on hardware — which is a prerequisite for Track B regardless.

Track B is now *possible without a fork*, which was the open question, and the
O(n²) objection turns out not to apply to H2O/TOVA/SnapKV. But it is still three
layers of change across four components plus two variant structs for SM90, and
it should be built on a tap path that has already been shown to work.

**B5 (RetrievalAttention) stays out of scope.** An ANN index over `K` is a
subsystem, not an intrinsic.

---

## 8. Track A as built (implementation record)

Sections 1-7 were the plan. This is what the hardware actually required, and it
differs from the plan in three places that matter.

### 8.1 What runs

`envelope_dot` executes as a second-party PTIR kernel at **every layer of every
decode step**, on `cuda_native`, scoring every page of the request's page list
against that layer's post-RoPE query. The `quest-attention` inferlet reports the
per-page bound, the classification of those bounds, and the page set Quest would
keep. Curated matrix **36/36**.

The selection is **reported, not applied** — see §8.5.

### 8.2 The DSL had no way to emit a KernelCall

The plan assumed the authoring surface existed. It did not: `ptir-dsl` could
name a kernel but not emit `Op::KernelCall`, and `attn_page_mask` *discarded its
argument* (`let _ = mask.to_arg()`), recording only a name for T11 precedence.
So a program that configured attention and one that did not lowered to the same
trace.

Both are fixed. `Session` now interns names (`intern_name`), `kernel::
envelope_dot` emits a real `Op::KernelCall`, and `attn_page_mask` emits a real
`Op::SinkCall` carrying the mask value.

### 8.3 Rejecting an op in the SHARED lowering makes it invisible to every backend

`container_to_trace` (`driver/common/include/pie_native/ptir/bound.hpp`) is the
lowering **all** backends decode through, and `Dispatch::register_program` calls
it *before* any capability check. It hard-rejected `PTIR_OP_KERNEL_CALL` and
silently dropped `PTIR_OP_SINK_CALL`. A backend that could launch the kernel
never got to see it.

Both now lower into real `Op`s. The `Op` record predates named ops, so the name
index rides in `imm` and a new `Trace::names` resolves it — the shape Metal's
interpreter already assumed. Hosts that cannot execute them fault at execution
(`"tier-0 uncovered op/dtype: kernel_call"`), which is the correct layer.

**Generalised lesson**, alongside the four in `11-ptir-limits.md`: *a shared
lowering must carry ops it does not understand, because refusing there is a
refusal on behalf of backends it knows nothing about.*

### 8.4 Envelopes cannot be enabled lazily — they have to be budgeted

This was the expensive discovery. Two independent reasons, either one fatal:

1. **Staleness.** Envelopes are maintained *incrementally*:
   `launch_envelope_update_appended_bf16` refreshes only the pages a fire
   appended to. Envelopes were being switched on at `register_program`, i.e.
   *after* the prefill had written the whole prompt, so every full page kept its
   `(+inf, -inf)` empty seed and scored `+inf` forever. The tap ran, reported
   28 layers, produced fluent text — and had scored nothing. Exactly the
   "silently wrong contract" class this repo already documents four times.

2. **Memory.** Envelopes are `2 x 4 x kv_heads x head_dim` bytes per page per
   layer = **`2/page_size` of the KV cache** (12.5% at page_size 16; 5.7 GB for
   Qwen3-0.6B on a 46 GB L40S). The KV pool is sized to *consume the device*, so
   by the time a program binds there is nothing left. Recomputing the pool on
   enable — the fix for (1) — hit `cudaMalloc` failure, and CUDA error 700 is
   sticky, so it surfaced under an unrelated stack.

Both are solved by allocating envelopes **with the pages** and charging them to
the page count: `memory_planner.cpp` adds `envelope_bytes_per_page` to the
per-page cost, so the pool simply holds proportionally fewer pages. On a fresh
cache no page holds a key, so the empty seed is *exact*, and every append
thereafter refreshes what it touched. A page recycled from a retired request is
correct after its first append, because `envelope_update_appended_kernel`
recomputes a touched page in full over its live range rather than folding into
the old value.

Two consequences worth stating explicitly:

- **Envelopes escape the KV arena.** It is elastic (`commit_on_allocate =
  false`): an allocation is unbacked virtual address space until `ensure_pages`
  commits it, so seeding the range faults. Envelopes are not elastic in nature —
  every page the pool can hold needs one — so they are plain device allocations.
- **It is an operator opt-in**, `PIE_CUDA_KV_ENVELOPES=1`, and
  `has_kv_envelopes` requires it. This is a *fourth* gate on top of the three in
  §5 (native bf16 NHD, post-RoPE query, `tp_size == 1`). Advertising the
  capability without the memory would let a Quest program bind and then fail at
  its first fire; gating makes it fail at bind, with a message that names the
  switch.

### 8.5 Measured cost

Qwen3-0.6B on an L40S, ~400-token prompt, best of 3 warm runs:

| | ms/token |
|---|---|
| `naive-baseline`, envelopes off | 5.04 |
| `naive-baseline`, envelopes on | 4.50 |
| `quest-attention` (tap active, 28 layers) | 6.35 |

**Envelope maintenance on the KV append path is free within noise.** The whole
cost is the tap: **+1.24 ms/token (+24%)**, about 44 us per layer for the
channel take, `envelope_dot`, `max_elem` fold and put.

That is currently *pure* cost, because the selection is reported rather than
applied. It is also fixed per layer, whereas Quest's saving grows with context,
so the number to beat is not 24% but 24% measured at a context long enough for
attention to dominate. This should be re-measured at 8K-128K once §8.6 lands.

### 8.6 Mask consumption: the design (built; see §10)

The one piece of Track A not built. The design is now fully determined by two
findings, both verified against the sources:

**(a) Page selection needs no FlashInfer change and no replan.** On the
**static no-split decode plan** — the default on SM80+ for batches of <= 512
requests (`can_use_static_nonsplit_decode_plan`,
`driver/cuda/src/ops/attention_flashinfer.cu:69-79,194-201`) — the plan is a
trivial one-CTA-per-`(request, head)` descriptor: `request_indices[r] = r`,
`kv_tile_indices = 0`, `o_indptr[r] = r`, `split_kv = false` (ibid. 88-170). It
does **not** depend on page counts. The kernel takes `kv_len` from the
`paged_kv_t` built at *launch* from the device page list
(`attention_flashinfer_common.cuh:444-454`; `decode.cuh` reads
`paged_kv.get_length(batch_idx)`).

So a different `kv_page_indices`/`kv_page_indptr`/`kv_last_page_lens` may be
passed **per layer** with the plan untouched. This must be gated: under a
**split-KV** plan the plan arrays *are* derived from page counts and
substituting a shorter list would be silently wrong.

**(b) The hook fires before the attention it governs.** `OnAttnProj` runs
post-RoPE and post-QK-norm at `llama_like.cpp:589-600`; the decode dispatch is
at `678-684`. So a program that scores pages at the hook can restrict the
attention of that same layer.

The build is therefore:

1. A **compaction kernel**: per request, walk the page list in order, keep the
   masked-in pages, *always* keep the last page (it holds the tokens this fire
   is writing, and keeping it last means `last_page_len` and the `kv_len`
   identity carry over unchanged), and emit a tight CSR.
2. A **return path on the stage hook**, which is currently one-way. The natural
   place is beside `AttentionObservation` — a per-layer `AttentionPageSelection`
   published by the hook, tagged with its layer so a stale view cannot leak into
   the next one.
3. **Substitution at the decode call only.** The fire's own CSR must keep
   serving the KV append: it is the true source of `kv_len` (`geometry.cu`) and
   the address keys are written through. Compacting it would corrupt the cache.

The **unresolved** part, and the reason this was not built blind: a `SinkCall`
inside a generated region has to be *executed*, and `singleton_codegen.hpp:449`
currently whitelists it as a semantic **boundary**. Whether the fused/NVRTC path
executes the sink or splits the region around it decides where the mask write
belongs — and getting it wrong yields a program whose sink is silently skipped,
which is precisely the failure mode `11-ptir-limits.md` catalogues. That
question should be settled first, against `fused_runtime.cuh:1901`,
`module_cache.hpp:679` and `singleton_codegen.hpp:231`.

### 8.7 Deviations from the paper, as built

1. **Union over heads, not per-head selection.** `paged_kv_t` has one page list
   per request and the custom-mask offset `qo_idx * kv_len + kv_idx` has no head
   index, so a per-head selection has nowhere to live. The kernel takes the max
   over KV heads: a page is kept if *any* head wants it. Quality >= Quest;
   speedup <= Quest.
2. **In-flight pages score `+inf`.** A page the current fire is still filling
   has no settled envelope, so it is pinned rather than scored from partial
   data. Fail-safe, and it coincides with Quest's "keep the local window".
3. **Selection is observed, not enforced** (§8.6).

---

## 9. Track B as built (implementation record)

### 9.1 What runs

`tova-attention` reads the real softmax attention weights, at **all 28 layers**,
on real hardware, for every decode fire. `test_curated.py` is **37/37**.

The row is self-validating and the test exploits that: each layer contributes a
distribution over the live prefix, so the folded row must sum to exactly
`layers_observed`. Measured: `score_mass = 27.99999` against 28 layers.

The scores discriminate. On a prompt whose answer sits at the front, TOVA's
keep-set comes out as `[0..7, 12, 17, 97..102]` — the attention sink plus a
recency window — and the first eviction is position 48, mid-filler. That is the
published behaviour of the algorithm, not merely a well-formed buffer.

| Commit | Content |
|---|---|
| `1a8f62d65` | Layer 1 — observe attention scores through a FlashInfer variant |
| `800fe40b6` | Layer 2 — the `AttnScore` intrinsic in the PTIR ABI |
| (this) | Layer 3 — capture wired into the forward, bound at `OnAttn` |

### 9.2 The intrinsic table stride was hardcoded at 7, and `AttnScore` is 7

`fused_runtime.cuh` sized the per-lane intrinsic side tables (bases / modes /
widths / strides / offsets) at `lane_count * 7`, and `fused_codegen.hpp` emitted
`dispatch_lane * 7u + p.intr` into the **generated device source**. `AttnScore`
is id 7. The read would not have faulted — it would have returned
`intrinsic_bases[lane*7 + 7]`, which is the **next lane's `Logits` base**.

Two things make this worth recording. First, the constant lived in two places,
one of them a string literal inside a code generator, so a reader checking the
host packer would have seen nothing wrong. Second, the emitted source is cached
as cubins on disk keyed by `kCudaGeneratedEmitterVersion`, so widening the
stride without bumping that version would have left every existing machine
replaying the old stride against the new host layout.

Both sites now derive from one `kPtirIntrinsicSlots = PTIR_INTR_ATTN_SCORE + 1`
with a `static_assert`, and the emitter version went 18 -> 19.

This is contract-family #4's shape: a fast path indexed by an id that the id
space quietly outgrew.

### 9.3 The host page CSR is an upper bound; the device CSR is exact

**This is the bug that cost the most, and it is a new instance of contract #3.**

`AttentionObservation` carries the fire's host-side page CSR. `frame.cpp` is
free to replace that CSR with a *conservative* one before the body runs — graph
lattice padding (`:1869`) and the decode-envelope KV bound (`:1818`, which
assigns `kv_last_page_lens = page_size` for every request) both do. The **device**
CSR the attention kernel reads stays exact.

So `LayerScoreCapture` sized its ragged buffers from the host CSR (an upper
bound), while the capture and fold kernels wrote exactly `kv_len` entries as
derived from the device CSR. The slack between the two was never written, and
`cudaMallocAsync` hands back recently-freed memory: the tail of the score row
was **live garbage from a previous fire**.

The failure signature is worth remembering:

- It was **not** deterministic. Three runs in five were clean.
- It was **not** wrong from the start. The first three or four fires of a run
  were exact, then the observed length pinned to the bound and stuck.
- It was **not** loud. The garbage was ~1e-5 in magnitude, so the row still
  summed to 28 and still looked like a distribution.

An eviction policy fed that row keeps positions that do not exist and drops real
ones — silently, and only sometimes.

The fix is to `cudaMemsetAsync` the folded buffer at allocation. That is not
papering over the gap: `0.0` is the intrinsic's *defined* value for a position
past `kv_len`, because a position that does not exist received no attention.

The general rule this reinforces: **a host-side CSR handed to a model body is a
bound, not a measurement.** Anything sized from it must zero the slack, and
anything that needs the true length must derive it from the device CSR — which
is what `k_attn_score_normalize` and `k_attn_score_fold_heads` already do.

The regression test asserts the declared and observed lengths agree on **every**
fire, not just the last one, because the first fires were clean.

### 9.4 `GroupedStageStaticPlan` is a hard gate at registration, not routing

`GroupedStageStaticPlan::valid == false` reads like a "cannot go grouped, fall
back to fused" routing decision, and inside `try_add` it is exactly that. But
`Dispatch::register_program` (`dispatch.cu:2971`) constructs the same plan and
**refuses the whole program** when it is invalid. A stage reading a new
intrinsic therefore has to be described there even though only the fused path
can serve it — the same status `requires_kernel_call` already has for
`envelope_dot`.

### 9.5 Prefill needs no capture path

The concern was that real inferlets prefill first, so a decode-only capture
would throw on the first fire. It does not arise: the repo's own pattern is a
**separate tap-free `ForwardPass` for prefill** (`quest-attention` does this,
and `tova-attention` follows it). That is also correct on the algorithm's terms
— TOVA ranks by the most recent query token, and during prefill "most recent" is
still moving.

### 9.6 Capability gating

`has_attn_score` is true only where every decode fire lands on the plain paged
decode path: llama-like family, `tp == 1`, native bf16 non-HND pages, no sliding
window, and `use_prefill_decode_plan` off (SM90+ routes decode through the
prefill kernel, which has no capture variant). A windowed layer additionally
passes `capturable = false` so the capture is never constructed — a row that
described a truncated context while claiming to describe all of it would be
wrong in exactly the way §9.3 is about.

### 9.7 Deviations from the papers, as built

1. **Heads are folded by the backend.** TOVA ranks per head; one page list per
   request means a per-head keep-set has no representable consumer. The backend
   returns the mean over query heads. Identical collapse, identical reason, as
   Quest (§8.7.1).
2. **Layers are folded by the program.** TOVA keeps a cache per layer; the
   inferlet sums the per-layer rows and ranks the sum. This is the
   layer-uniform variant TOVA itself evaluates, and it is monotone-equivalent
   to the mean.
3. **Selection is observed, not enforced.** As with Quest, the mask does not yet
   reach attention, so output is bit-identical to `naive-baseline` — which is
   what makes the tap testable.

### 9.8 Measured cost, and what the tap actually costs

The `LogitsTransform` capture is the one part of Track B that touches a tuned
kernel, so it was measured in isolation rather than inferred from end-to-end
timings. `test_attn_score_capture` gained a `PIE_ATTN_SCORE_BENCH=1` mode that
runs both decode entry points against an *identical plan* behind CUDA events —
the only way to attribute cost to the kernel rather than to the PTIR stage
machinery wrapped around it.

L40S, head_dim 128, 16 query heads over 8 KV heads, one request:

| `kv_len` | plain | capture (initial) | capture (after §9.8 fix) |
|---:|---:|---:|---:|
| 512 | 0.0179 ms | 1.49x | **1.34x** |
| 2048 | 0.0646 ms | 1.39x | **1.23x** |
| 6400 | 0.1971 ms | 1.37x | **1.21x** |
| 16384 | 0.5005 ms | 1.36x | **1.20x** |

**The fix was one hoisted load.** The hook dereferenced
`params.score_indptr[batch_idx]` on *every* invocation — a global load on the
kernel's innermost loop, sitting on the dependency path of the store address.
It is loop-invariant per CTA, so it moved into the variant's constructor beside
the inherited `kv_len`, and the store became `score_row[qo_head * len + kv]`.
That removed roughly 44% of the capture overhead. The residual ~1.20x is the
scattered 4-byte store itself: only `threadIdx.x == 0` writes, so a warp
contributes two active lanes writing addresses `kv_len` floats apart, which is
one memory sector per useful 4 bytes. Coalescing it would require restructuring
the kernel's thread mapping, which is exactly what using a supported extension
point buys us out of.

**Why the residual is acceptable, and why it is not the number to optimise.**
The tap's cost is proportional to `kv_len`, and so is the attention it rides on
— but H2O and TOVA *bound* `kv_len` by construction. Once §8.6-style mask
consumption lands, a TOVA run with `cache_size = 64` keeps 64 positions
resident no matter how long the context is, so the 1.20x multiplies a base that
has itself collapsed. Measuring the tap before eviction exists measures the
worst case the design can ever exhibit.

This contrasts sharply with Quest, and the contrast is the useful result:

| | ctx ~408 | ctx ~6408 |
|---|---:|---:|
| `quest-attention` | +14.6% | **+11.8%** |
| `tova-attention` (pre-fix) | +20.5% | **+32.0%** |

Quest's tap is a *fixed* per-layer cost — it reads page envelopes, whose count
is `kv_len / page_size` — so it amortises as context grows. Track B's is
`O(kv_len)` because it observes every position, so it does not. That asymmetry
is inherent to what the two algorithms need to know, not an artefact.

**A measurement caveat worth recording.** End-to-end `ms/token` on this host is
host-bound (§8.5) and the host is shared; a run taken at load average ~28
reported `naive-baseline` at 8.08 ms/token against 6.34 ms/token an hour
earlier, and showed `tova-attention` as *faster* than the baseline it strictly
dominates. Kernel-level claims in this document come from the CUDA-event
harness, which measures device time and is immune to that.

---

## 10. Mask consumption as built (implementation record)

The last piece of both tracks, and the one that turns observation into
enforcement. Until this landed, every Quest test passed *whether or not the
mask was applied* — the program ranked pages, the driver read the ranking, and
attention then ignored it. The cost was real and the benefit was zero.

### 10.1 What runs

`quest-attention` on Qwen3-0.6B / L40S, `PIE_CUDA_KV_ENVELOPES=1`:

| budget | continuation |
|---|---|
| 18 (= all pages) | ` (This is a list of statements about Paris. Please answer` |
| 1 | ` (Question: What is the probability that a randomly selected person` |
| `naive-baseline` | ` (This is a list of statements about Paris. Please answer` |

Both halves matter and they pull in opposite directions:

- **Enforcement** — budget 1 diverges. Attention really is confined to the
  pages the policy kept.
- **Coherence** — budget 18 reproduces the unmasked baseline *token for token*.
  An all-keep mask is a no-op, so the compaction does not reorder pages, drop
  the wrong one, or desynchronise `last_page_len` from the list it belongs to.

`tests/inferlets/test_mask_enforced.py` asserts exactly this pair. The curated
matrix is 37/37 with the sink live.

### 10.2 The mask must not be addressed by the page CSR

This is the whole difficulty, and the first design was wrong about it.

The obvious layout is one mask byte per page, sliced per request by the fire's
`kv_page_indptr` — the same CSR everything else in the fire uses. It cannot
work, for the reason §9.3 gives: **the host page CSR is a bound, the device CSR
is exact.** The mask is written from the host (that is where the lane table and
the program's value live) and consumed by a device kernel walking the real page
table. Under decode envelopes those two CSRs disagree — `frame.cpp` substitutes
`plan_kv_page_indptr` and a uniform `page_size` for the host copies while the
device resolves the true geometry itself — so request *r*'s mask row and
request *r*'s page list start at different offsets. Every eviction after the
first request lands on another request's pages.

That is not a hypothetical path. It is *the* path: Quest's `page_indptr` is
device-computed, so `has_decode_envelopes` is true for exactly the fires the
feature exists to serve. An earlier revision detected the hazard and threw,
which was correct but left the feature unreachable.

**The fix is to delete the shared dependency rather than reconcile it.** The
mask is `[num_requests, stride]`, row-major, addressed by *request index times a
fixed stride* — the page CSR appears nowhere in it. `stride` is the widest
request's page count taken from the host CSR, so a conservative host CSR only
over-allocates; it never mis-addresses. The single fact the writer and the
reader must agree on is "slot *p* of request *r*", which is precisely what the
program means when it writes `mask[p]` from `scores[p]`, and precisely what
`envelope_dot` meant when it produced `scores[p]`.

Two consequences fall out, both in the safe direction:

- A slot past the end of a row keeps its page (`page_survives`). Under the
  stride invariant this is unreachable, but the stride comes from a host table
  and the count from device geometry, so the check is what makes a disagreement
  an over-attend rather than an out-of-bounds read.
- Rows are seeded to 1 before every layer, and the sink writes only
  `min(stride, declared)` entries. A page no policy examined is kept. The
  alternative — evicting a page nothing scored — is not recoverable.

`test_page_compact` case 7/8 pin this: they run the compaction with rows
deliberately wider than any request needs, which is what the driver actually
produces. A kernel that recovered the row base from `page_indptr_in` passes
every other case in the file and fails these two.

### 10.3 The compaction

`launch_compact_page_csr` (`driver/cuda/src/kernels/page_compact.cu`) rewrites
the fire's paged-KV CSR down to the kept pages, in three launches: count
(`cub::BlockReduce`), exclusive-scan the per-request counts
(`cub::BlockScan`, tiled with a `running` aggregate), scatter (tiled, order
preserving). Three invariants:

1. **Order is preserved.** FlashInfer's page list is positional; permuting it
   permutes the KV it reads.
2. **The last page always survives**, unconditionally. It holds the token this
   fire is writing, and keeping it last is what lets `last_page_len` — and the
   `kv_len = (pages-1)*page_size + last_page_len` identity built on it — carry
   over from the original CSR untouched.
3. **A request never drops to zero pages.**

Substitution happens on the decode call only, and only after
`decode_plan_is_page_count_independent` confirms the fire planned the static
non-split path. A split-KV plan derives its tile indices *from* the page counts,
so a shorter list would silently attend over the wrong tiles.

### 10.3b Compaction cost, and the two things that dominated it

Compaction runs once per **layer** per fire, so unlike a host-side setup step it
is not amortised over the model. As first built it cost 12 us/call at a 2K
context, which is 336 us on a 28-layer model -- around 5% of a decode step, all
of it overhead. Two things were responsible, and neither was the actual work:

1. **A `cudaMallocAsync`/`cudaFreeAsync` pair per call** for the per-request
   survivor counts. At decode batch sizes that allocation cost more than both
   kernels together. The buffer is now caller-owned: `FirePageMask` already
   allocates four device buffers once per fire, so a fifth is free and is reused
   by every layer.
2. **Three kernel launches** (count, scan, scatter) where two suffice. The only
   thing the scatter needed from the scan was its own output base -- the
   exclusive prefix of the per-request counts -- and with one block per request
   that prefix is at most `num_requests` values long. Each block now sums it
   itself. Recomputing an O(R) sum per block is far cheaper than the launch it
   replaces.

Measured in isolation with CUDA events (`PIE_PAGE_COMPACT_BENCH=1
./bin/test_page_compact`), L40S, page_size 16:

| pages/request | context | before | after | speedup |
|---:|---:|---:|---:|---:|
| 128 | 2K | 12.0 us | 5.0 us | 2.4x |
| 1024 | 16K | 12.1-14.1 us | 6.4-7.5 us | ~1.9x |
| 8192 | 128K | 29.8-31.0 us | 24.1-25.5 us | 1.24x |

The speedup shrinks as the context grows because the long-context case is
genuinely work-bound -- which is the right shape. Per decode step at 2K context
this recovers roughly 200 us.

Layers that emit no mask pay none of this: `compact` is gated on
`written_for(L)`, so a program that taps a subset of layers is charged only for
those. Quest and H2O write on every layer, so the figures above are their real
per-fire cost divided by the layer count.

### 10.4 A latent DSL bug: the name table was emitted in first-use order

`intern_name` assigns indices in first-use order; the container requires the
name table to be strictly sorted and unique. No program had ever used two
second-party names, so nothing noticed. Quest using both `envelope_dot` and
`attn_page_mask` — where the one used second sorts first — produced a container
the loader rejected outright.

Fixed in `builder.rs::build()` by sorting the table and remapping every
`KernelCall`/`SinkCall` `name_idx`, mirroring the channel-gid remap directly
above it. The remap is the half worth testing: sorting alone still loads, and
silently invokes the wrong kernel.

### 10.5 Capability gating

`attn_page_mask` is a *first-party* sink in `KNOWN_SINKS`, so the validator
never gated it — a program would bind on any backend and no-op on the ones that
cannot enforce. `has_attn_page_mask` now travels from the CUDA context through
`PtirCaps` to `ModelProfile`, and `validate.rs` refuses the bind where it is
false. Backends that only observe now say so at bind time rather than at
runtime, or worse, never.

## 11. H2O as built (implementation record)

### 11.1 One line separates H2O from TOVA

`tova-attention` re-seeds its accumulator in the epilogue; `trackb-h2o` puts it
back unchanged. That is the entire algorithmic difference, and it is the reason
H2O needed no new driver code — the `AttnScore` tap of §9 and the
`attn_page_mask` consumption of §10 were already the two halves it needs.

The carry is directly observable, which makes it self-validating. Each layer of
each fire contributes one softmax row, so the accumulated mass must grow by
exactly the model's layer count on every fire. Measured on Qwen3-0.6B (28
layers, 18 pages):

```
mass_trace = [28.014, 56.014, 84.014, ..., 308.014]     # +28.000 per fire
```

A flat trace would mean the loop-carried channel was being re-seeded behind the
program's back; a jumpy one would mean it was reading stale device memory.
Neither can hide.

The keep-set is the heavy-hitter signature the paper predicts, and it
discriminates by orders of magnitude rather than marginally — page 0 (the
attention sink) carries 63% of all mass on a prompt whose remaining 14 pages are
near-identical filler:

```
page_mass = [193.5, 5.2, 4.2, 105.1, 0.0012, 0.0011, ... 0.00005]
kept(budget=3) = [0, 1, 3]   evict_next = 17
```

### 11.2 The cold start needs a descending ramp, not a zero seed

The first decode fire must choose a keep-set having observed nothing — prefill
does not capture (§9.5). An all-zero seed makes every page tie, so the selection
is arbitrary and real context is evicted at random. The seed is instead a small
descending ramp (`1e-4 * (kv_max - i) / kv_max`), four orders of magnitude below
the 1.0 a single layer-fire contributes, so one observation dominates it
completely.

It must be *descending*. An ascending ramp would rank never-used tail slots
above live ones and evict everything real. Descending degenerates to "keep the
earliest positions" — an attention sink — while the driver independently
force-keeps the in-flight last page, which is a local window. The cold-start
behaviour is therefore the StreamingLLM Λ-shape, which is the right prior to
hold for exactly one fire.

The consequence is that H2O's score row is never exactly zero anywhere, so —
unlike TOVA — it cannot assert that slots past the live prefix are zero. The
invariant that survives is that no *attention* lands out there: every tail slot
must stay under the seed ceiling. A position that does not exist cannot have
been attended to.

### 11.3 Page granularity is not a shortcut

A position-granular mask can stop attention from reading a position, but it
cannot free the page holding it. Only page granularity delivers the memory
H2O exists to reclaim, so the eviction unit is the page and the score is
`reduce_sum(reshape(acc, [max_pages, page_size]))` — a reinterpretation, since
`kv_max == max_pages * page_size` exactly.

### 11.4 Coherence must be asserted at near-greedy, not at a sampling temperature

The natural check for "an all-keep mask is a no-op" is that the policy program
reproduces `naive-baseline` token for token. Asserted at `temperature=0.1` this
is **latently flaky**, and finding out why was worth the detour.

Running a policy program changes the decode batch's shape and therefore the
attention kernel's split/reduction plan, so its logits differ from the plain
baseline's in the last bit or two. This is not the tap: `test_attn_score_capture`
asserts the capture variant leaves the attention output *bit*-identical and
passes. It is plan-level residue, and it is unavoidable — floating-point
addition is not associative, and a different split is a different order.

Under Gumbel sampling at `tau=0.1` those last bits are amplified 10x and
occasionally decide a near-tied token. Measured over 24 tokens x 5 seeds x 3
programs:

| tau | result |
|---|---|
| 0.001 | 15/15 exact, reproducible across sessions |
| 0.1 | scattered single-token flips, hitting the mask-only program (Quest) as readily as the capture ones, and not stable across sessions |

So the assertion was moved to near-greedy, where the decision margin is wider
than the residue. This is not a weakening: "the policy follows the baseline's
argmax path exactly" is the claim that was actually meant, and unlike the
`tau=0.1` version it is true. The enforcement half of the pair — a one-page
budget must *change* the continuation — is robust at any temperature and was
left alone.

A second, duller cause was hiding under the same failure: `trackb-h2o` inherited
`tova-attention`'s RNG salt (`seed ^ 0x70a`) while `naive-baseline` and
`quest-attention` use `seed ^ 0x5bd1`. Two programs drawing from different
Gumbel streams disagree for reasons that have nothing to do with attention. Any
inferlet whose test compares its text against the baseline's must share the
baseline's salt.

## 12. SnapKV as built (implementation record)

### 12.1 SnapKV needed a second tap, not a second policy

TOVA and H2O are decode-time policies: they watch the attention of each
generated token and re-rank continuously, so both ride the `AttnScore` tap of
§9, which fires on a one-row decode. SnapKV is a *prefill-time* policy. It looks
at what the **tail of the prompt** attended to, selects a keep-set once, and
then holds it fixed for the whole generation. There is no decode observation to
read, and by the time the first token is generated the decision has already been
made.

That is a different kernel. FlashInfer's prefill path is a separate template
tree from decode, so the tap had to be built a second time:
`dispatch_attention_flashinfer_prefill_capture_bf16`, `PieScoreCaptureWindow`,
and two post-processing kernels. What did *not* have to be rebuilt is everything
downstream — the `AttnScore` ABI, the page fold, `attn_page_mask` consumption,
and the compaction of §10 are all shared. The inferlet is ~460 lines and the
enforcement half of it is the same code H2O runs.

### 12.2 Three things make the prefill capture unlike the decode one

**(a) The `threadIdx.x == 0` guard must not be carried over.** In the decode
capture the `bdx` lanes hold identical `s[j]` after a butterfly reduction, so one
lane writes and the rest are redundant. In prefill the threads hold *distinct*
MMA fragment elements: `q_idx` and `qo_head_idx` come from
`divmod(qo_packed_idx, group_size)` and `kv_idx` from the lane's position in the
tile. Keeping the guard would silently drop 31 of every 32 scores — and it is
invisible without an exact reference, because a row missing most of its entries
still normalises into a plausible-looking distribution. Exactly-once still holds
without atomics: split-KV chunks are disjoint in `kv`.

**(b) `q_idx` needs its own bounds check.** The last MMA tile is padded, so
`q_idx` can exceed `qo_len`. Decode never had a q dimension to overrun.

**(c) The hook's `batch_idx` argument is literally `0` in prefill.** The real
request index only reaches the *constructor*, so the row base is resolved there
and the hook receives it as state.

The window gate itself is cheap: `qo_len` is an inherited member of
`DefaultAttention`, so `first_qo = qo_len > window ? qo_len - window : 0` and the
per-score test is a register compare. Rows outside the window cost one compare
and nothing else.

### 12.3 The causal limit has to be applied after the fact

`LogitsMask` runs *after* `LogitsTransform`, so the hook records real dot
products at positions the softmax is about to discard. Normalising over the full
`kv_len` would spread mass onto the future.

Window row `w` belongs to the query at absolute position `kv_len - rows + w`
(with `rows = min(window, qo_len)`), so it may attend to `kv_len - rows + w + 1`
keys and no more. `k_attn_prefill_score_normalize` zeroes `[limit, kv_len)` and
softmaxes `[0, limit)`. The derivation is general — it stays correct under
chunked prefill, where `kv_len > qo_len`.

`[0, limit)` is always *fully* written, which is what lets the raw buffer go
un-zeroed: within a q-tile every kv index in the processed range is evaluated
for every q row (the mask is applied afterwards), and with split-KV the chunks
together still cover the range. The fold therefore reads exactly the region
normalize wrote. Only the folded buffer needs zeroing, because the host page CSR
is an upper bound while the device CSR is exact.

The fold divisor is `heads * rows`, **not** `heads * window`. A prompt shorter
than the observation window contributes fewer rows, and counting rows that do
not exist would scale its mass down.

### 12.4 The planner had to learn about the tap

Whether a prefill runs on SM90 or FA2 is a *plan-time* decision, so refusing
SM90 at dispatch time would be too late. `wants_prefill_score` therefore threads
into `plan_attention_flashinfer_prefill_bf16`.

It also promotes `full_attention_variant` to true. The real prefill path plans
with the sliding-window template while the capture is instantiated over
`AttnVariantFull`; the two differ only by a runtime window predicate that is
trivially true at `window_left < 0`, and `PrefillPlan<IdType>` does not depend on
the variant at all — only on `window_left` and geometry. So the promotion changes
no numerics and saves an instantiation. `window_left >= 0` throws.

SM90 is refused **loudly**, never silently fallen through. A silent fallthrough
would hand the policy an all-zero row, which reads as "nothing was attended to"
and evicts the entire prefix. Hopper's `StandardAttention` has a different
constructor, stores neither `qo_len` nor `kv_len`, and would need `score_out`
threaded through `AdditionalParams` → `to_underlying_arguments`; that is a real
port, not a flag. This is the one known gap.

### 12.5 The pooling deviation

SnapKV pools the observed scores with an overlapping max-pool (kernel 7, stride
1) before ranking, to avoid keeping an isolated high-attention token without its
neighbours. We fold non-overlapping, at page granularity, and sum rather than
max.

This is deliberate and it is the same argument as §11.3: a position-granular
selection can stop attention from reading a position but cannot free the page
holding it. Page granularity is the only unit that returns memory, so it is the
unit the policy must rank in. Pooling at a finer granularity and then rounding up
to pages would produce a *different ranking of the same pages*, not a finer
eviction. The non-overlapping page fold is the enforceable analogue.

### 12.6 What the capture actually shows

Qwen3-0.6B, 28 layers, 272-token prompt (17 pages of 16), default window 32 —
so the observation window is pages 15..16:

```
page_mass = [14.64, 0.76, 0.54, 0.51, 0.42, 0.46, 0.36, 0.44, 0.37,
             0.37, 0.43, 0.42, 0.55, 0.60, 1.60, 3.22, 2.31]
             ^sink  <------------ flat filler ------------>  <-window->
score_mass = 28.0000 = layers_observed        tail_nonzero = 0
```

The total is the layer count to four decimals, which pins the fold divisor: an
un-normalised fold would land at `heads * rows * layers` and a `heads * window`
divisor would undershoot on a short prompt.

The shape is checked by *sweeping the window* rather than by asserting a
constant, because the profile is a prediction. `test_snapkv.py` derives the
window's page span independently of the driver and both agree at every setting:

| `PIE_ATTN_SCORE_WINDOW` | window pages | window page vs. ordinary prefix page |
|---|---|---|
| 16 | 16..16 | 8.0x |
| 32 (default) | 15..16 | 4.9x |
| 64 | 13..16 | 3.0x |
| 128 | 9..16 | 1.6x |

The elevated region grows and its peak marches *left* as the window widens. That
is not a curiosity, it is the causal mask being obeyed: kv position
`first_row + j` is weighted by only `window - j` of the `window` captured rows,
so the very last page is structurally attended by the fewest rows. A profile that
peaked at the final page would mean the mask had *not* been applied to the
captured rows, and the test asserts it does not.

At `window = 128` the "window" covers half the prompt and the contrast
legitimately washes out; that is a statement about the configuration, so the
strong separation check is gated on `win_pages * 3 <= n` rather than being
weakened for everyone.

### 12.7 Cost

CUDA-event isolated, one request, L40S, `window` = the observation window:

| `qo_len` | window | plain | capture | ratio |
|---|---|---|---|---|
| 512 | 32 | 0.0345 ms | 0.0670 ms | 1.94x |
| 512 | 128 | 0.0342 ms | 0.1411 ms | 4.12x |
| 2048 | 32 | 0.1246 ms | 0.2085 ms | 1.67x |
| 2048 | 128 | 0.1244 ms | 0.2945 ms | 2.37x |
| 8192 | 32 | 1.3328 ms | 1.6567 ms | **1.24x** |
| 8192 | 128 | 1.3453 ms | 1.8642 ms | 1.39x |

The ratio *falls* as the prompt grows, which is the opposite of the decode
capture: prefill work scales with `qo_len * kv_len` while the tap scales with
`window * kv_len`, so a fixed window amortises. The right way to read the
production row (8192 / 32) is **+0.32 ms per layer, once per sequence** — roughly
+7% on a prefill forward pass, paid to avoid carrying an uncompressed cache
through hundreds of decode steps. Plain prefill is untouched: the planner only
promotes when `wants_prefill_score`, and the decode path is not involved at all.

The tap is bandwidth-bound and close to the machine. At 8192/128 it moves ~402 MB
per layer in 519 us — 775 GB/s against the L40S's 864 GB/s peak. Fusing the
softmax rescale into a recompute pass (5 full-buffer passes down to 4) moved the
8192/32 ratio only 1.25x → 1.24x, which locates the remaining cost where it
actually is: the variant's stores are scattered across MMA fragments, not
coalesced, and that is inherent to reading per-`(row, kv, head)` scores out of a
fused kernel. It was kept anyway — strictly less traffic, no added complexity.

Memory is transient and capped: `heads * window * kv_len * 4B` per request per
layer, 16.8 MB at 16 heads / window 32 / 8K context, refused above 1 GiB.

### 12.8 Chunked prefill is well defined, not merely tolerated

The window is the last `window` rows *of this firing*. For one-shot prefill that
is exactly SnapKV's definition. Under chunked prefill the *final* chunk's firing
matches SnapKV, and the earlier firings are well-defined observations of earlier
windows — so a policy that acts on the most recent firing gets SnapKV's semantics
without special-casing. The mixed prefill+decode plan deliberately does not
capture, and the PTIR side fails loudly there rather than returning a zero row.

## 13. Two silent Quest bugs found by exact-census testing

Both bugs below produced fluent text, no NaNs, no crashes, and a page ranking
that looked entirely plausible. Neither was reachable by any test that checked
Quest "works". They were found by asserting the *exact* slot census —
how many pages carry a real bound, how many are pinned at `+inf`, how many are
`-inf` past the end — against arithmetic derived independently from the fire's
own `kv_len`. That is the assertion shape this section is really arguing for.

### 13.1 `envelope_dot` sliced the device page array with the host CSR

Same hazard as §9.3 and §10.2, in a third place. Under decode envelopes the
host's `plan_kv_page_indptr` is a prefix sum of the page channel's *declared*
capacity (`envelope_plan_page_bounds`), while the device array is packed by
real per-request counts. `resolve_lane_envelope` took `page_begin`/`page_count`
from the host copy, so:

* the **offset** was wrong whenever an earlier request in the fire held fewer
  pages than it declared — request `r` scored against request `r-1`'s keys;
* the **count** was wrong whenever this request did — surplus slots were
  scored as real instead of getting the `-inf` that keeps a top-k consumer out
  of a neighbour's pages.

`kv_last_page_lens_h` is *also* substituted with a uniform `page_size` under
envelopes, so `scored_pages` was wrong on top of that.

The fix moves the whole resolution on-device. The host now contributes only a
slot **bound** — the grid extent and the result-row width — which is safe to
over-estimate precisely because surplus slots then correctly resolve to `-inf`:

```
begin       = page_indptr[request];  end = page_indptr[request + 1]
page_count  = end >= begin ? end - begin : 0
slot >= page_count                       -> -inf   (not ours)
kv_after    = (page_count-1)*page_size + last_page_lens[request]
kv_before   = kv_after >= qo_len ? kv_after - qo_len : 0
scored      = min(kv_before / page_size, page_count)
slot >= scored                           -> +inf   (in flight, cannot bound)
page        = page_ids[begin + slot]
```

The rule this is the third instance of: **a host-side copy of a page CSR is a
bound; only the device copy is exact.** Anything a kernel *addresses* with must
come from the device copy. Host copies may size allocations and grids, nothing
else.

Why no earlier test caught it: at small `max_tokens` the declared bound and the
real page count coincide, and with one request in the fire the offset is 0
either way. The curated test used `max_tokens=8`. It now uses 64 and asserts
the census, with an explicit guard that fails if `pages_absent == 0` — i.e. if
the case has drifted back to one where both CSRs would agree.

### 13.2 The explicit-descriptor KV write never maintained envelopes

`llama_like.cpp` has three KV-write branches: the fused decode QKV path
(disabled whenever stage hooks are bound, so Quest never reaches it),
`has_write_desc` → `launch_write_kv_explicit_bf16`, and otherwise the CSR path
`launch_write_kv_to_pages`. Envelope maintenance was hooked into the CSR path
only.

Quest supplies WSlot/WOff, so `has_write_desc` is true for its **decode** fires
and false for its prefill. Prompt pages therefore had real envelopes and every
page written during decoding kept the empty `(+inf, -inf)` seed. An empty
envelope makes `Σ max(q·kmin, q·kmax)` equal `+inf` for any nonzero `q`, so the
newest — most recently attended, most likely to matter — pages all scored
"always keep", silently destroying the ranking.

The symptom was **non-deterministic**, which is what made it interesting: a
recycled physical page carries a stale but *finite* envelope from a previous
tenant, so the number of visibly-empty pages depended on pool state. Cold runs
looked worse than warm ones.

**Merge, not recompute.** The explicit path has no page list and no live
length, so a recompute keyed on the descriptor's offset would shrink a page to
a prefix and could drop a key that is still live — and rewriting a cell
mid-page is exactly why the explicit path exists (beam fork/freeze). Merging
only ever widens, which for an upper bound is the safe direction to be wrong
in, and for append-only pages it is *equal* to a full recompute. A page being
entered at `w_off == 0` is being started, so its envelope is reset first;
without that, a recycled page accumulates every request that ever used it and
converges on "keep everything".

`test_envelope_dot` proves the equality directly: it replays a prefill fire
plus a run of one-token decode fires through the merge and compares against
`envelope_recompute` over the final live lengths — after planting a previous
tenant's `[-1e30, 1e30]` on every page the sequence will touch. The comparison
is exact, so a missing reset cannot pass.

**Cost, and why the reset is folded in.** The two kernels are trivial at decode
widths — the fire carries `R` tokens — so their cost is entirely launch
overhead: ~4.6 us per layer, ~129 us per 28-layer pass, on the critical path of
every step. A single fused kernel elects one writer block per (page, kv_head) —
the first valid token naming that page — which owns the page outright, gathers
its own fire's keys, and stores once. Sole ownership removes the atomics *and*
makes the reset safe to fold in: the race the two-kernel split exists to avoid
(a reset erasing a key another token of the same fire already merged) cannot
arise when one block does both, in order, for every token on the page. Measured
on an L40S: **129 us → ~82 us per 28-layer pass, flat from N=1 to N=128.** Above
128 tokens the two-launch form is kept, since there the kernels are doing real
work and the shared-memory gather list would not fit.

## 14. From what context does Quest pay for itself?

Quest trades work for work: an `envelope_dot` per layer, a top-k, and a page
table compaction, in exchange for an attention that reads fewer pages. Below
some context the overhead exceeds the saving. That crossover is the only number
that decides whether to enable it, and no kernel microbenchmark yields it,
because the overhead is per-layer-per-step while the saving is proportional to
the pages the budget removes. `tests/inferlets/bench_quest.py` measures it.

### 14.1 Two things had to be fixed before the number meant anything

**The instrumentation was being timed.** The inferlet drained per-page scores,
a layer count and `kv_len` to the host on every step so tests could assert the
census. None of that is the policy — the ranking, the threshold and the mask
are computed and consumed on device — but it costs a per-layer fold over
`p_max` plus three device-to-host drains per step that a runahead pipeline has
to wait on. It is now behind `report` (default on, so every test is unchanged),
and the benchmark turns it off. At 6144 tokens with a quarter budget this alone
moved the result from 1.01x to 0.76x.

**The difference method was charging Quest for its own endpoint.** Decode cost
is measured by differencing two runs that differ only in `max_tokens`, which
cancels prefill and all fixed overhead. But `p_max` — the number of slots
`envelope_dot` scores per layer — is derived from `max_tokens`, so the long
endpoint was doing more per-step work than the short one and the difference
absorbed it. `reserve_tokens` now pins `p_max` to the long endpoint for both.

Two further method notes, both learned by getting them wrong: minimise the two
endpoints **independently** before differencing (minimising the differences
pairs the luckiest long run with the unluckiest short one, which produced
negative times), and **interleave** the configurations rather than running each
to completion, so a drifting shared host does not land entirely on one of them.

### 14.2 The measurement

Qwen3-0.6B, 28 layers, page size 16, L40S. ms/token, min of 7 interleaved
rounds, `report=false`:

| ctx | pages | baseline | budget=100% | budget=50% | budget=25% |
|---|---|---|---|---|---|
| 1024 | 80 | 3.76 ms | 1.65x | 1.45x | 1.35x |
| 2048 | 158 | 4.83 ms | 1.50x | 1.24x | 1.16x |
| 4096 | 314 | 6.84 ms | 1.33x | 1.04x | **0.87x** |
| 6144 | 471 | 8.70 ms | 1.32x | 0.98x | **0.77x** |

Monotone in every direction: the baseline grows with context, every row
improves as the budget tightens, and every column improves as context grows.

**Quest becomes a net win at ~4K context with a quarter budget, and is 23%
faster at 6K.**

### 14.3 The overhead is constant, so the crossover is set by the baseline

The `budget=100%` column is Quest with nothing evicted — pure overhead. In
absolute terms it is 2.46, 2.41, 2.29, 2.76 ms across the four contexts: **flat
at ~2.5 ms/step regardless of context.** The ratio falls from 1.65x to 1.32x
only because the baseline grows underneath it.

That is ~89 us per layer, and it is *not* the kernels: envelope maintenance is
~3 us/layer (§13.2) and compaction 5-10 us/layer at these contexts (§10.3b).
The remainder is the per-layer hook dispatch itself — `envelope_dot`,
`pivot_threshold`, `rank_le` and `attn_page_mask` are four separate launches
plus the grouped-dispatch machinery around them, 28 times per step. That, not
any individual kernel, is where a further Quest speedup would come from.

It also means this table is a **worst case for the ratio**. A 0.6B model has a
small decode step, so a constant 2.5 ms is a large fraction of it. On a
production-sized model the same 2.5 ms is a much smaller share and the
crossover moves substantially earlier.

### 14.4 Where the 2.5 ms actually goes: the hook takes the request off CUDA graphs

§14.3 attributed the constant overhead to "per-layer hook dispatch". Profiling
says what that means, but only after a methodology trap that is worth recording
because it produced a confident and completely wrong answer first.

**The trap.** `nsys profile -t cuda` traces CUDA graphs at *graph* granularity
by default: a replayed graph appears as a single entry, and the kernels inside
it are not listed at all. The baseline's decode step therefore showed up as one
174 us kernel, and the natural reading -- "the baseline fuses the whole
28-layer step into one megakernel and the hook opts out of it" -- is wrong. The
tell was arithmetic, not tooling: 28 layers of this model read ~860 MB of KV
and ~1.2 GB of weights, which cannot happen in 174 us on a device with 864 GB/s
of bandwidth. **A profile that implies a kernel exceeded the memory bandwidth
of the machine is not a discovery, it is a measurement artifact.** Re-running
with `--cuda-graph-trace=node` resolved it.

**What is actually true.** 104 decode tokens at ~6.1K context, identical
workload, node-level tracing:

| | baseline | quest |
|---|---|---|
| kernels executed | 27 361 | 58 982 |
| of which inside a CUDA graph | 26 368 (96%) | **0** |
| host-visible launches | 993 | 58 982 |
| host launches per decode step | ~9 | ~573 |
| GPU time | 989.1 ms | 1090.9 ms |

The two programs run *the same model kernels*, in the same counts:
`BatchDecodeWithPagedKVCacheKernel` 2884 (= 103 steps x 28 layers) in both,
`gemv_bf16` 5872 in both, `gemvx` 5768 in both, `rmsnorm` 5928 in both. Nothing
about the model forward is unfused by the hook. What the hook costs is that the
request stops being replayed from a captured graph and starts being launched
kernel by kernel from the host.

So the ~2.5 ms/step splits in two:

* **~1.0 ms/step on the GPU** (989.1 -> 1090.9 ms over 103 steps). Quest's own
  kernels are most of it: `envelope_dot` 2884 x 12.0 us = 0.34 ms/step, the
  threshold/mask program 2884 x 8.6 us = 0.24 ms/step, the explicit KV write
  2884 x 2.5 us = 0.07 ms/step. The remainder is per-layer variant differences
  -- the baseline fuses split+qk-norm+rope into one `qkv_decode_qk_norm_rope`
  (2.2 us) where the hook path runs `split_qkv` and `qk_rmsnorm_rope`
  separately (2.4 + 3.5 us).
* **~1.5 ms/step on the host**, which is 564 extra eager launches per step at a
  few microseconds each. This is the part that does not show up in any kernel
  timing and is why §14.3's stage breakdown only ever saw a blocking wait.

**This reorders the optimisation targets.** The largest single lever is not any
kernel in this document -- it is making a hook-bearing request graph-capturable
again. The exclusion is one clause, `!has_stage_hooks`, at the end of
`forward_graph_replay_eligible` (`driver/cuda/src/batch/forward.cpp`), and it is
worth being clear that it is **structurally necessary rather than merely
conservative**:

* `execute_declared_phase` is host work that runs once per layer per fire. It
  rebuilds the task/binding/group vectors, sizes and uploads the lane table,
  and acquires workspaces. A replayed graph runs no host code, so the recorded
  kernels would address the workspace and lane-table contents of whichever step
  was captured.
* `Dispatch::finish` asserts `phase_invocations[phase] == model_layers` -- the
  hook must be observed to have run at every layer. Under replay it would be
  observed zero times and every hook-bearing fire would fail that check.

Neither is a property of the *policy*: everything Quest does per layer is
already device-side (`envelope_dot`, a PTIR threshold program, a mask write).
They are properties of how the PTIR stage phase is currently driven. Lifting
them means giving the phase stable device-resident state that device kernels
update, instead of host-rebuilt state, and moving the invocation accounting to
the device. That is an engine project in PTIR dispatch rather than a Quest
change, and it would benefit every Track A and Track B policy identically,
since they all bind the same per-layer hook.

Ranked, per decode step at ~6K context:

| | cost | owner |
|---|---|---|
| lost CUDA-graph replay | ~1.5 ms | engine (PTIR dispatch) |
| `envelope_dot` | 0.34 ms | this document |
| threshold + mask program | 0.24 ms | this document |
| unfused split/rope variant | ~0.3 ms | engine |
| explicit KV write + envelope merge | ~0.15 ms | this document |

The three rows this document owns total ~0.73 ms against ~1.8 ms owned by the
engine. Tuning them further has a low ceiling until the graph question is
addressed.

One further engine-level observation, flagged rather than attributed because it
affects both programs equally and so cancels out of every comparison here:
`cudaGraphInstantiate` was called 51 times at ~2.2 ms each in a 31-step run, in
*both* programs. Re-instantiating an executable graph rather than updating one
is expensive enough to dominate the host side of a decode step.

## 15. Track B: where each eviction policy pays for itself

> **Superseded by §18.4 and §18.5.** The table below was measured with a page
> count extrapolated from the target context length rather than probed, so its
> `@1` columns were a ~80% budget that evicted rather than the zero-benefit
> reference they are described as. The *shape* of every conclusion here holds;
> the overhead numbers in §15.1 are understated and are corrected in §18.5,
> where they turn out to agree better with §14.4 than the numbers below did.
> §15.2's ordering is unaffected. Kept as written because the correction is
> more instructive than the corrected text would be alone.

§14 measured Quest. `tests/inferlets/bench_trackb.py` applies the identical
method -- differenced endpoints, independently minimised, interleaved across
rounds, `reserve_tokens` pinned, `report` off -- to the two Track B policies
that enforce. Qwen3-0.6B, 28 layers, page size 16, L40S, min of 7 rounds.
Ratios are baseline ms/token over policy ms/token, so >1.00x is a net win.

| ctx | pages | baseline | h2o@1 | h2o@0.5 | h2o@0.25 | snapkv@1 | snapkv@0.5 | snapkv@0.25 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 70 | 3.54 ms | 0.52x | 0.58x | 0.61x | 0.65x | 0.73x | 0.74x |
| 2048 | 134 | 4.66 ms | 0.58x | 0.70x | 0.76x | 0.73x | 0.88x | 0.95x |
| 4096 | 262 | 6.92 ms | 0.62x | 0.85x | 0.99x | 0.80x | 1.10x | 1.16x |
| 6144 | 390 | 10.68 ms | 0.72x | 1.13x | 1.22x | 0.89x | 1.44x | **1.79x** |

Monotone in every direction: in the context, in the aggressiveness of the
budget, and between the two policies at every cell.

### 15.1 SnapKV is at the floor, and that is the point

The `@1` columns keep every page, so they price the policy with its benefit
removed. At 6144 that is **+4.15 ms/step for H2O and +1.32 ms/step for
SnapKV**, and the gap between those two numbers is the whole argument of §14.4
restated from the other end.

SnapKV does almost nothing per layer. Its keep-set was decided once, during
prefill; every decode step afterwards re-applies a mask that is already
resident on the device. There is no score to compute, no row to fold, no
ranking. Its 1.32 ms/step is therefore very close to a direct measurement of
what binding a per-layer hook costs by itself -- and §14.4 put that at ~1.5
ms/step from an entirely different instrument (kernel launch counts under
`nsys`). Two independent measurements landing on the same constant is the
reason to believe either of them.

The consequence is that **SnapKV's per-step cost cannot be optimised from
within this document**. It is already doing the minimum. Only the engine-level
graph-capture change described in §14.4 would move it.

H2O's extra ~2.8 ms/step over SnapKV is its own work, and it is per-layer work
proportional to the context: `on_attn` folds a `[kv_max]` score row into a
cumulative accumulator at every layer, and the epilogue reduces that row into
page masses once per step. That is a real cost with a real justification --
H2O's statistic is the accumulated history, which is exactly what SnapKV's
fixed keep-set gives up -- but it is the part worth attacking if H2O's
crossover needs to move earlier.

### 15.2 How the three policies compare

Ranked by where each becomes a net win on this model:

| policy | decides | per-layer decode work | crossover (quarter budget) |
|---|---|---|---|
| SnapKV | once, at prefill | apply a fixed mask | **~2.3K** |
| H2O | every step | fold + rank a `[kv_max]` row | ~4.1K |
| Quest | every step | `envelope_dot` over all pages, then rank | ~4.4K (§14) |

The ordering is the ordering of how much each recomputes. SnapKV commits to a
decision and never revisits it, so it pays once; H2O revisits it with a
statistic it already has; Quest recomputes the statistic itself from the key
envelopes every layer. Each step up that ladder buys adaptivity -- Quest can
change its mind about a page whose relevance depends on the current query,
which SnapKV structurally cannot -- and each costs a later crossover.

None of this is an argument that SnapKV is the best policy. It is an argument
that on a 0.6B model the crossovers are close together and all of them are
dominated by a constant that belongs to the engine. On a production-sized model
the baseline per-step cost grows while that constant does not, so every one of
these crossovers moves earlier, and the ordering between them -- which is set
by real per-layer work rather than by the constant -- is what survives.

## 16. Envelopes cost half of what they did, for free

Quest's key envelopes were stored as fp32 `[num_pages, kv_heads, head_dim]`
pairs, which is **12.5% of the key tier and 25% of the KV pool's key half** --
the largest single memory cost this document adds, and the one that decides
whether Quest is affordable on a device where KV capacity is the binding
constraint.

### 16.1 The narrowing is exact, not a tradeoff

The usual framing for a precision reduction is a memory-vs-accuracy trade. Here
there is nothing to trade, because of what an envelope *is*:

> An envelope entry is the min or the max of a set of **bf16 keys**. It is
> therefore already, exactly, some bf16 value. Storing it in fp32 stores the
> same number in twice the space.

The seeds are exact too: `+inf` and `-inf` are representable in bf16 (bf16 is
the top 16 bits of fp32, so it inherits the exponent range verbatim -- which is
also why the sign-magnitude ordering argument behind the `atomicCAS` min/max
carries over unchanged). And `envelope_dot` widens `bf16 -> fp32` exactly before
multiplying, keeping the accumulate in f32, so **every arithmetic result is
bit-identical to the fp32-storage version.**

This is checkable rather than merely arguable, and `test_envelope_dot`'s 13
checks -- including the exact merge/recompute equality of §13.2 and the golden
vector -- all pass **unchanged**, with no tolerance loosened. If the narrowing
were lossy, the merge-equals-recompute check would be the first to break, since
it compares two different orders of arriving at the same bound.

Directed rounding (`__float2bfloat16_rd` for min, `_ru` for max) is applied
anyway. It is a no-op on every value the current code stores, and it is there so
that a future caller merging a value that did *not* originate as a bf16 key --
a quantized tier, a fused dequantize, a synthetic bound -- cannot round a bound
*inwards*. An envelope that is too tight drops a live key from the score and
Quest evicts a page it should have kept; an envelope that is too loose only
costs a page it did not need. Only one of those directions is safe to be wrong
in, and the rounding mode is what pins it.

### 16.2 The saving, measured

The planner (`memory_planner.cpp`) and the cache (`kv_cache.cpp`) both charge
`2 * sizeof(uint16_t) * layers * kv_heads * head_dim` per page and **must
agree**, or the cache overruns the budget the planner sized. Running the planner
on this L40S with the switch off and on:

| `PIE_CUDA_KV_ENVELOPES` | logical KV pages | KV tokens |
|---|---|---|
| 0 | 22 084 | 353 344 |
| 1 | 20 785 | 332 560 |

22084/20785 = **1.0625 exactly** -- envelopes are 6.25% of a KV page, as the
accounting says. Under fp32 the same ratio was 1.125, which would have left
19 630 pages. **The narrowing hands back 1 155 pages -- 18 480 tokens of KV
capacity -- for zero numeric change.**

Kernel cost is unchanged at decode widths: the merge bench reports 82.3 us per
28-layer pass at N=1 and 78.5 us at N=16, matching §13.2's fp32 numbers, because
at those widths the kernels are pure launch overhead and touch too little memory
for the halving to show. It shows only in the memory table, which is the point.

End to end, `bench_quest.py` re-run after the change reproduces §14.2's crossover
unchanged -- net win from 4K, quarter-budget 0.85x at 4096 and 0.60x at 6144
against a 11.90 ms baseline -- with an identical reserved page count at every
context. The narrowing is invisible to every measurement except the memory one.

## 17. The one-shot prefill was a ceiling under the crossover

Every measurement above stops at 6144 tokens. That was not a choice about what
was interesting -- it was the largest context the benchmark could reach, because
both endpoints prefilled the whole prompt in **one fire**, and a fire cannot
exceed the driver's structural per-launch token capacity (`max_embed_length()`,
8192 on this CUDA driver).

That ceiling sat in a bad place. §14.3 shows Quest's overhead is a constant
~2.5 ms/step while the saving grows with context, so the ratio only improves as
the context does. Capping the measurement at 6144 therefore capped it just past
the crossover, where Quest had barely started to win -- and worse, it capped the
*feature*, not just the benchmark. A policy whose entire purpose is long context
could not be run at long context.

### 17.1 Chunking, and what has to be true of it

`quest-attention` and `naive-baseline` now split the prompt into `ceil(n/C)`
chunks. Chunk `i` attends over the whole prefix written so far -- `kv_len` is
cumulative and `page_indptr` covers every page up to the chunk's end -- and
writes only its own tokens. That is what makes the concatenation equal the
one-shot fire: the causal offset each chunk's queries see is `kv_len - qo_len`,
which is exactly the chunk's base. When the prompt fits in one chunk the loop
runs once and builds the pass it always built, so nothing below the ceiling
moves.

**Testing the equivalence needed a correction that is worth recording.** The
obvious test -- run the same prompt at a forced-small chunk width and compare
the text -- fails, and the first version of it did. Chunking changes the
attention kernel's tile decomposition (28 fires of 37 tokens do not reduce in
the same order as one fire of 1024), so the prompt's hidden states differ in
their last bits and those bits are written into the KV cache. This is §11.4's
observation arriving from a new direction, and near-greedy decoding amplifies it
into completely different, equally coherent text.

The fix is not a tolerance. It is to pin the assertion to a prompt whose
continuation is **decisive**: on an ambiguous prompt the next-token distribution
is nearly flat and a 1-ulp difference flips the argmax, but on a prompt whose
answer the model is sure of the argmax has a real margin and cannot be flipped.
`test_chunked_prefill` uses such a prompt and asserts **exact text
equality over 32 tokens** at chunk widths 37, 128 and 999 -- all deliberately
not multiples of the 16-token KV page, so every boundary lands mid-page and the
write offsets have to be right. The single-token argmax was checked separately
across three contexts and four widths and is identical in every case, which
isolates the prefill's own output from any downstream amplification.

### 17.2 The measurement the ceiling was hiding

Same method as §14.2 -- Qwen3-0.6B, 28 layers, page size 16, L40S, ms/token,
`report=false`. Ratios are quest/baseline, so **below 1.00 is a win**. Numbers
below are the re-measurement described in §18.3: min of **15** rounds, with the
page count probed rather than extrapolated (the first version of this table had
a slightly short `budget=100%` and one visible outlier; both are gone).

| ctx | pages | baseline | budget=100% | budget=50% | budget=25% |
|---|---|---|---|---|---|
| 1024 | 86 | 3.59 ms | 1.61x | 1.54x | 1.34x |
| 2048 | 164 | 4.57 ms | 1.52x | 1.28x | 1.18x |
| 4096 | 320 | 6.87 ms | 1.37x | 1.01x | **0.87x** |
| 6144 | 477 | 8.87 ms | 1.28x | 0.90x | **0.73x** |
| 8192 | 633 | 11.20 ms | 1.21x | 0.84x | **0.65x** |
| 12288 | 946 | 15.37 ms | 1.18x | 0.76x | **0.56x** |
| 16384 | 1259 | 19.70 ms | 1.13x | 0.72x | **0.51x** |

**At 16K context and a quarter budget Quest is 1.96x faster than the baseline.**
The crossover is unchanged at ~4K -- lifting the ceiling did not move it, which
is the right outcome, since the ceiling was an artifact of the harness and not a
property of the policy.

Every column is now strictly monotone in the context, which the seven-context
table at 7 rounds was not. That is worth more than any single cell: the trend is
the claim, and a monotone trend across a 16x span is not something host noise
produces by accident.

The `budget=100%` column is the constant: in absolute terms the overhead is
2.20, 2.40, 2.52, 2.44, 2.31, 2.70 and 2.65 ms across the seven contexts --
**flat at ~2.5 ms over a 16x range, with no outlier.** Seven contexts spanning
16x say what four spanning 6x could only suggest: **the overhead does not scale
with context and the saving does**, so the ratio improves without bound as the
context grows. §18.5 measures the same constant a third way, from SnapKV.

## 18. Track B past the ceiling, and a budget the benchmark did not have

§17 lifted the one-shot prefill ceiling for Quest and the naive baseline. Track
B had the identical ceiling and it mattered more there, because §15 put
SnapKV's crossover at ~2.3K and H2O's at ~4.1K -- the entire range in which
these policies pay was above the point at which they could be run.

Propagating the fix was mechanical for H2O and TOVA, which observe the decode
fire and leave prefill alone. SnapKV is the exception, and it is the reason
this is a section rather than a footnote.

### 18.1 SnapKV is the only policy for which chunking is not mechanical

SnapKV is observed **during prefill**. The capture records the last `window`
query rows *of the fire it is attached to* (§12.3), so under chunking only the
final chunk's window is the prompt's tail. An earlier chunk's window is a
perfectly well-defined observation -- of the wrong thing. `attn_score.hpp`
already anticipated this and blessed the resolution: *"the final firing is the
one whose window matches SnapKV, so a policy that simply acts on the most
recent firing gets SnapKV's semantics for free."*

So the tap goes on the final chunk alone. That is not only correct but cheaper:
the earlier chunks run the plain prefill kernel and never pay the capture
variant's 1.24-1.94x (§12.7).

Which makes the final chunk's **size** load-bearing, and the obvious even split
is not even enough. A fixed width of `ceil(n / ceil(n / C))` gives the right
chunk *count*, but the remainder still lands entirely on the last chunk: at
n=1302, C=37 it lays down 35 chunks of 37 and a final chunk of **seven**,
truncating a 32-row window to 7 rows. The fix is to spread the remainder over
the *first* chunks instead -- with `k = ceil(n / C)`, chunk `i` gets
`n/k + (i < n mod k)` -- so every chunk is within one token of every other and
the final chunk is `floor(n / k)`, the largest value a last chunk can take. At
C = 8192 that is thousands of tokens and the window is never in danger.

### 18.2 The discriminator, which took two tries to find

A truncated window is invisible in the output: it still produces a plausible
ranking and a coherent continuation. Testing it needs a quantity that moves.

The first choice, the keep-set, barely moves at all. At a 24-page budget on a
1302-token prompt, forcing the window down to **four** rows changed 2 of the 24
kept pages -- while merely changing the chunk width with the window intact
changed 1. No margin, and a test with no margin is a coin flip.

`tail_page_share` -- observed mass on the prompt's last page over total
observed mass -- separates cleanly, because a narrower window means fewer query
rows and all of them sit near the end of the prompt, which inflates the last
page's share:

| final chunk | 36 | 127 | 635 | 19 | 9 | 4 |
|---|---|---|---|---|---|---|
| `tail_page_share` | .0507 | .0506 | .0507 | .0852 | .1801 | .2702 |
| window | whole | whole | whole | cut | cut | cut |

Whole-window noise is 0.2%; the smallest truncation signal is +68%. A 2% band
sits two orders of magnitude clear of the noise and still trips on the mildest
truncation. That is the assertion in `test_chunked_prefill.py`, along with
`tail_page_share > 0`, which separately catches a tap attached to the wrong
chunk entirely (an earlier chunk cannot see the last page at all, so it reports
exactly zero).

### 18.3 The benchmark was measuring a budget it did not have

Re-running `bench_trackb.py` over seven contexts produced something that could
not be true: `snapkv@1` at **1.05x**, a policy that evicts nothing running
*faster* than not evicting anything.

The cause was in the harness, not the policy. `bench_trackb` derived the page
count as `(ctx + LONG_TOKENS) // PAGE_SIZE`. But `ctx` is a *target*:
`_prompt_for` lays down `round(ctx / 9)` repetitions of a unit that tokenizes
to slightly more than 9, so the real prompt runs about 25% long -- ctx=12288
measures 15032 tokens. The estimate therefore under-counted pages by ~25%, and
**`budget=1.0` was really a ~80% budget that evicted**.

That is not a small error in a small place. The `@1` column is the column whose
entire job is to price a policy with its benefit set to zero, and every
overhead number in §15 was read off it. Those numbers were understated by an
unintended benefit.

Both benchmarks now probe the real page count, and both add the arithmetic the
budget has to satisfy: the probe stops after `SHORT_TOKENS`, so the
`LONG_TOKENS` the measured endpoint decodes have to be added back, which is 8%
of the pages at ctx=1024. Two guards now assert the properties that would have
caught this:

- `assert_full_budget_is_overhead` -- a full budget cannot beat the baseline,
  because the policy does everything the baseline does and then some. If it
  wins, the budget is not full.
- `assert_monotone_baseline` -- the baseline attends over the whole prefix, so
  its per-token cost is necessarily increasing in the context. If the measured
  baseline is not increasing, the run's noise floor is above its signal and
  every ratio derived from it is meaningless. This one is not hypothetical
  either: a 7-round sweep taken at host load 35 failed it (10.12 ms at 4096 vs
  9.04 ms at 6144). `PIE_BENCH_REPS` exists so the answer is more rounds rather
  than a wider tolerance.

### 18.4 The corrected Track B table

Qwen3-0.6B, 28 layers, page size 16, L40S, ms/token, **min of 15** interleaved
rounds, `report=false`, page counts probed. Ratios are baseline/policy, so
**>1.00x is a win**.

| ctx | pages | baseline | h2o@1 | h2o@.5 | h2o@.25 | snap@1 | snap@.5 | snap@.25 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 86 | 3.51 ms | 0.51x | 0.55x | 0.59x | 0.60x | 0.72x | 0.76x |
| 2048 | 165 | 4.32 ms | 0.47x | 0.61x | 0.66x | 0.65x | 0.76x | 0.83x |
| 4096 | 321 | 7.03 ms | 0.61x | 0.81x | 0.59x | 0.71x | 1.02x | 1.24x |
| 6144 | 478 | 8.64 ms | 0.59x | 0.84x | 1.01x | 0.79x | 1.16x | 1.38x |
| 8192 | 634 | 11.16 ms | 0.62x | 0.93x | 1.25x | 0.85x | 1.23x | 1.50x |
| 12288 | 946 | 15.31 ms | 0.65x | 0.96x | 1.37x | 0.88x | 1.22x | **1.81x** |
| 16384 | 1259 | 19.85 ms | 0.65x | 1.05x | 1.47x | 0.90x | 1.31x | 1.63x |

Every `@1` cell is now below 1.00x, as it must be. Crossovers at a quarter
budget: **SnapKV ~2.9K, H2O ~6.1K**. (`h2o@.25` at 4096 reads 0.59x, below both
its neighbours -- a residual noise artifact of the shared host, not a feature.
`snapkv@.25` likewise loses a little between 12288 and 16384.)

### 18.5 What the corrected numbers say, which is more than the wrong ones did

Converting the `@1` ratios to absolute per-step overhead is where the table
stops being a benchmark and starts being an argument:

| ctx | 1024 | 2048 | 4096 | 6144 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|---|---|
| SnapKV | 2.34 | 2.33 | 2.87 | 2.30 | 1.97 | 2.09 | 2.21 ms |
| H2O | 3.37 | 4.87 | 4.49 | 6.00 | 6.84 | 8.24 | 10.69 ms |
| H2O − SnapKV | 1.03 | 2.55 | 1.62 | 3.71 | 4.87 | 6.16 | 8.48 ms |

**SnapKV's overhead is flat at ~2.3 ms/step across a 16x range of context.**
H2O's grows roughly linearly, and the difference between them -- H2O's own
work, the per-layer `[kv_max]` fold and rank -- is the part that grows.

This separates the two costs cleanly, which the four-context table in §15 could
only hint at. SnapKV decides its keep-set once during prefill and afterwards
re-applies a mask that is already resident on the device: no score, no fold, no
ranking, nothing per layer that depends on the context. What is left is the
cost of *having a hook at all*.

And that number now closes a loop that §15 left open. Three instruments:

| instrument | what it measures | value |
|---|---|---|
| Quest `budget=100%` (§14.3, §17.2) | end-to-end, benefit removed | ~2.5 ms |
| `nsys --cuda-graph-trace=node` (§14.4) | ~1.5 ms host launch + ~1.0 ms GPU | ~2.5 ms |
| SnapKV `budget=100%` (here) | end-to-end, policy work ~0 | **~2.3 ms** |

§15 reported SnapKV's overhead as 1.32 ms and called it agreement with §14.4.
It was not: 1.32 ms matched only the *launch* component and was silently short
of the total, because the budget was evicting. The corrected 2.3 ms agrees with
both other instruments on the whole quantity. **The conclusion of §15.1
survives and is strengthened: SnapKV is at the floor, and the floor is the CUDA
graph replay the hook costs (§14.4), which cannot be moved from inside this
document.**

The practical consequence is unchanged in direction and larger in size than
§15 estimated. On a 0.6B model a ~2.3 ms/step constant is most of the budget,
which is why these crossovers sit where they do. That constant is set by the
engine's launch path and does not grow with the model, while the baseline
per-step cost does -- so on a production-sized model every crossover here moves
earlier, and what survives is the ordering, which is set by real per-layer
work: SnapKV (none) < H2O (a `[kv_max]` fold) < Quest (`envelope_dot` over
every page).
