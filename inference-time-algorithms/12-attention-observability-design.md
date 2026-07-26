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

Follow the `MtpDrafts = 6` precedent exactly (`compiler/ir/src/op.rs:63-67`),
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
compiler/ir/src/op.rs              enum + from_u16 + name
compiler/codegen/src/header.rs          generated C enum
compiler/ir/src/registry.rs        intrinsic_stages()
compiler/ir/src/validate.rs        scope check + type rule
compiler/eval/src/interp.rs          PassInputs field + eval root
compiler/codegen/include/ptir_abi.h     PTIR_INTR_ATTN_SCORE = 7
driver/common/.../ptir_abi.h          mirror
driver/common/.../trace.hpp           C++ mirror enum + static_asserts
driver/common/.../bound.hpp           wire decode
driver/cuda/src/pipeline/dispatch.cu  bind the buffer at OnAttn
driver/metal/src/pipeline/interp.hpp  reject
```

### Layer 3 — the write side (eviction / masking)

Two granularities are needed, and one already exists.

**Page granularity (Quest).** `attn_page_mask` is fully declared —
`compiler/dsl/src/intrinsics.rs:92`, `registry.rs:172` (`SinkScope::Attention`),
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

The plan assumed the authoring surface existed. It did not: `compiler/dsl` could
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
