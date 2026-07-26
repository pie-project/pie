# What PTIR cannot do yet

The preceding documents ask whether the implementations are faithful to their
papers. This one asks the opposite question: having written thirty inferlets
against PTIR, where does the abstraction actually strain?

The evidence base is the curated suite — 15 inference-time algorithms, plus
search, speculative, KV-layout and composition inferlets, plus a `naive-baseline`
control — all of which now pass on CUDA (`10-implementation-faithfulness-audit.md`,
"What actually runs"). Three axes: does it **fuse**, does it **cost**, does it
**read** well.

---

## 1. Fusion

### The barrier set is narrow, and that is the good news

`region_kind_for_node` (`interface/ptir/src/compiler.rs:1771`) assigns every op
to a region kind. Exactly five op families become their own `Library` region —
`TopK`, `SortDesc`, `CumSum`/`CumProd`, `MatMul`, and second-party
`KernelCall`/`SinkCall` — and `compatible_schedule` (`:1782`) makes the first
four hard barriers on *both* sides, so nothing may fuse across them. Everything
else falls to `Generated` and is greedily merged.

The practical reach of that rule is better than it sounds. Across the thirty
inferlets in `tests/inferlets`, **twenty-seven use none of the barrier ops at
all**:

| Inferlet | barrier ops |
|---|---|
| `locally-typical-sampling` | 2 (`top_k`, `cum_sum`) |
| `tail-free-sampling` | 1 (`top_k`) |
| `beam-search` | 1 (`top_k`) |
| *all 27 others* | **0** |

Softmax, entropy, log-softmax, running penalties, pivot/threshold truncation,
mask application, Gumbel noise, argmax, scatter over a seen-token vector,
nine-round tournament knockouts — none of these break a region. An entire
sampler epilogue, however baroque, compiles to a single fused kernel as long as
it does not need a *ranking* or a *prefix scan*.

And the cost table bears this out exactly: the only two samplers with a cliff
(A1 at 5.37×, A3 at 5.30×) are precisely the two with barrier ops. The other
eight land between 0.84× and 2.15×.

### Where it strains: the library recognizer is a single exact pattern

`recognize_library_dataflows` (`:1332`) loops over one matcher,
`match_nucleus_dataflow`. There is exactly **one** recognized library dataflow in
the entire compiler: `LibraryOp::NucleusSample`, and it matches only the literal
shape

```
reduce_argmax( add( select( pivot_threshold( div( exp( sub(l, bcast(max)) ),
                                                  bcast(sum) ) ),
                           l, const(-inf) ),
                    rng_keyed{gumbel} ) )
```

This is an exact dataflow match, not a rewrite system. Spelling softmax any
other way, or splicing a single `broadcast` between the mask and the argmax,
drops the match. That is a normal, forgivable limitation of a peephole
optimiser — *except* that here a miss was not merely a lost optimisation.

**A pattern miss was a hang.** The generated-code emitter deliberately routed
`pivot_threshold(cummass_le)` to the single-threaded M1 reference, whose
implementation was an O(len³) selection sort executed by thread 0 alone —
unreturnable at a 151936-token vocabulary. Because every other inferlet happened
to match the library pattern, the generated path had no coverage and the defect
sat undiscovered. `consensus-decoding` hit it by adding one `broadcast`.

The fix (`driver/cuda/src/pipeline/generated/fused_codegen.hpp`) ports tier0's
block-cooperative selection loop into the generated path, and
`tests/inferlets/sampling-primitives` now pins it with a keep-mask shaped so it
*cannot* match the library pattern. But the general lesson stands:

> **The generated path is the semantic contract; the library path is the
> optimisation.** Whenever those two diverge in *asymptotics* rather than in
> constants, a fusion heuristic silently becomes a correctness cliff.

`pivot_threshold(rank_le)` is the next candidate — its generated implementation
is O(len²), untested at vocabulary scale.

### Where it strains: no cost model, so barriers are invisible

Nothing in the DSL tells an author that `top_k` costs 5× and `pivot_threshold`
costs 1.15×. The tier-0 `k_topk_rows` kernel is an incremental-threshold
selection that rescans the row once per pick — `O(k · vocab)`, or 33.5 M element
visits per token at `k_max = 128` on a 262144-token vocabulary. A1 and A3 could
both be re-expressed without a ranking (A4 demonstrates the style), but nothing
in the surface language nudges an author that way.

---

## 2. Performance

The full measurement is in `10-implementation-faithfulness-audit.md`. Three
findings belong here because they are properties of **PTIR**, not of the
algorithms:

**The overhead is entirely marginal.** Intercepts land between 87 and 165 ms for
every configuration including the control — that is install plus prefill. No
inferlet carries a heavy one-time setup, so cost scales with tokens generated and
never surprises a short request. For a programmable-serving layer this is the
property that matters most, and PTIR has it.

**The dominant remaining cost is lost pipelining, not lost fusion.** Five of the
six entries above 3× pay for a *serialization*, not for an unfused kernel:

- D1/D2 (classifier-free guidance, context-aware decoding) run two forward
  passes whose combination feeds the next input, so neither may run ahead. 2× is
  the extra pass; the other 2× is the unavailable run-ahead window.
- `greenlist-watermarking`, `json-schema-constrained-decoding` and
  `contrastive-decoding` derive each fire's host input from the *previous* output
  token. They are structurally depth-1. (They had been written as if they were
  not, and the JSON-schema one was silently reusing a stale grammar mask on its
  run-ahead fires — i.e. not enforcing the constraint at all.)

PTIR's run-ahead window is what hides host latency for everything else in the
table. Any algorithm with a host-side dependency on the previous token forfeits
it entirely. **This is the single largest performance limit in the system**, and
it is architectural: the only escape is to move the dependency onto the device.
Both watermarking and grammar masking are expressible as device programs in
principle; neither is today, because the mask construction needs data structures
(a hash-seeded permutation, a DFA transition table) that PTIR has no way to hold
across steps.

**One anomaly remains unexplained.** `synthid-tournament-sampling` returns 1.50,
1.55 or 6.19 s at a fixed 160-token budget — two clean modes ~4× apart with no
input difference. Nine sequential knockout rounds make it the deepest epilogue in
the set, and the barrier cost model that accounts for every other slope does not
predict the split.

---

## 3. Expressiveness

The parts that read well read *very* well. `entropy-adaptive-temperature` is
`T = T₀ · N^(θ/H)` written almost verbatim; `dry-repetition-penalty` reaches
`0.8 · 1.75^(5−2) = 4.2875` exactly. The strain is concentrated in four places.

### Shape polymorphism leaks

`intrinsics::logits()` returns rank-1 `[vocab]` when a fire has a single read-out
row and `[rows, vocab]` otherwise. The type is therefore a function of a runtime
property the author is not looking at. This silently broke `beam-search` at width
1 — the `[B, 1]` score column could not meet a rank-1 `[v]` — and it is the
reason `reshape(&x, [1, vocab])` incantations are scattered through the
inferlets. The defensive spelling, `reshape(intrinsics::logits(), [B, v])`, is
something every author must learn by being bitten.

### Four unchecked contracts that fail silently

None of these is expressible in the type system, and all four produce a wrong
answer rather than an error:

1. **Loop-carried geometry ports must `take()` before they `put()`.** Under
   `DeviceGeometry` the host never drains the ring; `k_stage_readiness` treats a
   put into a full ring as *not ready*, which clears `pass_commit` and turns the
   fire into a **dummy run**.
2. **Every host-`Writer` channel a fire takes must be `put` before that fire's
   `submit`.** Otherwise the fire silently consumes a stale value.
3. **The KV page CSR — not the `KvLen` port — is the wire's source of truth for
   how much KV a lane attends.** `derive_kv_len_kernel`
   (`driver/cuda/src/kernels/geometry.cu:14`) computes
   `kv_len = (page_count - 1) * page_size + last_page_len`, where `page_count`
   comes from `kv_page_indptr` and `last_page_len` is all that survives of the
   `KvLen` port (`descriptor_resolve.hpp:15`:
   `last_page_len = ((len - 1) % page) + 1`). An author who declares
   `page_indptr = [0, reserved_pool_pages]` — the natural reading of "these are
   my pages" — makes attention read every uninitialised cell between the true
   length and the pool boundary.
4. **The logits feeding a `NucleusSample` must come from the logits intrinsic,
   behind at most a `reshape`.** The compiler's matcher
   (`match_nucleus_add_order`, `interface/ptir/src/compiler.rs:1393`) matches on
   *DAG shape only*, so it happily claims a sampler whose logits were produced by
   a `broadcast`, a `gather` or any other op. The driver then assumes what the
   compiler did not check: the nucleus prep in
   `driver/cuda/src/pipeline/generated/fused_runtime.cuh` (~L1007) shrinks the
   scratch slot for the region's logits input to **4 bytes** and marks it elided,
   on the theory that it is the intrinsic and will never be materialised. With
   any other producer the op still runs and writes a full `[rows, vocab]` tensor
   into that 4-byte slot, scribbling over neighbouring values in the same
   per-lane scratch arena.

Between them these cost more debugging time than every genuine algorithm
question in this project combined. All four are mechanically checkable — the
first from the geometry class and the port's loop-carried status, the second by
comparing the takes of a submitted fire against the puts issued before it, the
third by asserting `page_indptr` spans agree with the `KvLen` the same fire
declares, the fourth by having the matcher verify the provenance it is already
relying on — and none is checked.

The third and fourth are the worst, because they are the only ones whose failure
mode is *plausible output*. The other two produce a dummy run or a stale
constant, which shows up immediately. Over-declared pages produce grammatical,
confidently-wrong text — six of the thirty inferlets shipped with it, passing a
green test suite, until an end-to-end test that asserted on *content* rather
than on liveness caught it. Nor does the attention mask save you: `pack_dense_mask.cu`
packs mask cells over the lane's *derived physical* span, so a correctly-sized
mask is simply laid out against the wrong geometry.

The fourth has the same shape of consequence and a nastier root cause: it is a
**disagreement between two components about who validates what**. The compiler
pattern is purely structural; the runtime's optimisation is provenance-dependent.
Neither is wrong in isolation. `consensus-decoding` hit it by broadcasting one
read-out row to `[B, vocab]` so that `B` lanes could draw independent Gumbel
noise from a shared distribution — a completely reasonable program, which the
type system accepts, the compiler compiles, the scheduler fuses, and the GPU
silently mis-executes. The general lesson is that **a fast path selected by
pattern-matching must validate every assumption it makes beyond the pattern**,
because the pattern is what the author is implicitly promised.

### The missing intrinsic sets a hard boundary

Five algorithms in the survey (H2O, SnapKV, TOVA, Quest, RetrievalAttention) are
unimplementable for one reason: they evict or select KV entries by **attention
score**, and `IntrinsicId` (`interface/ptir/src/op.rs:49-68`) exposes logits,
embeddings and geometry but no attention weights. This is not a fusion or
performance limit; it is a hole in the surface area, and it removes the entire
KV-eviction family. `attention-sink` and `sliding-window-attention` are in the
suite only because they select by *position*, which geometry does expose.

### Device programs are finite unrolls

There are no data-dependent loops on device. `dry-repetition-penalty`'s
`max_ngram` cap is not a design choice, it is the unroll depth; the penalty
saturates rather than continuing to grow. This is a fair trade for a
statically-schedulable program, and it happened not to bind on any algorithm
here — but it bounds the class.

---

## Out of scope by design

**One model per engine.** D3/D4-style algorithms that need two different models
resident in one engine are not a PTIR limitation to be fixed; the 1:1
engine-to-model relationship is intentional. Multi-model algorithms compose at
the orchestration layer instead.

---

## Summary

| Axis | Verdict |
|---|---|
| **Fusion** | Strong. 27/30 inferlets compile with zero barriers. The barrier set is small and its cost is exactly where the cliffs are. |
| **Fusion — risk** | The one library pattern is an exact match, and a miss dropped to an asymptotically worse implementation. Fixed; the class of defect is not. |
| **Performance** | Overhead is marginal, never fixed. The real limit is lost run-ahead for any host-dependent step — architectural, not incidental. |
| **Expressiveness** | Algorithms read close to their equations. Four silent-failure contracts and one leaking shape rule are the sharp edges; the missing attention-score intrinsic is the one hard wall. |
