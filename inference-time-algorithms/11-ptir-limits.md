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

And the cost table bears this out: the only two samplers that carry measurable
extra cost (A1 at 1.54×, A3 at 1.49×) are precisely the two with barrier ops.
The other eight land between 0.84× and 2.15×. Note the gap is now *modest* — it
read 5.37× and 5.30× until the `top_k` kernel behind it was fixed, and the part
attributable to the barrier structure itself was always the smaller half.

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

`pivot_threshold(rank_le)` was the next candidate — its generated implementation
was O(len²), untested at vocabulary scale. It is now a radix select, verified at
151936 columns.

### Where it strains: no cost model, so barriers are invisible

Nothing in the DSL tells an author that `top_k` and `cum_sum` are hard schedule
barriers while `pivot_threshold` is not. That gap is real and it is the reason
A1/A3 still sit ~1.5× above the baseline while A4, which needs no ranking, sits
at 1.15×.

What the gap is *not* is the 5.3× cliff those two originally measured. That was
an implementation defect: the `top_k` kernel actually dispatched
(`k_grouped_topk`) rescanned the full row once per pick, `O(k · vocab)`, on a
single 256-thread block. A radix select plus a bitonic sort of the survivors
removed it and made the cost flat in `k` (116 → 5.7 ms/token at `k_max = 1024`).

The episode carries a language-design lesson anyway: an author had no way to
tell the difference between "this op is inherently a barrier" and "this op has a
slow kernel today", because the surface language exposes neither. A1's source
carried a comment confidently attributing its cost to an algorithmic bound that
was really a fixable constant.

---

## 2. Performance

The full measurement is in `10-implementation-faithfulness-audit.md`. Four
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

**One anomaly is now characterized, if not fully explained.**
`synthid-tournament-sampling` returns either ~1.3 s or ~5–6 s at a fixed
160-token budget. What is now established:

- It is **per call, not per session**. Consecutive calls in one process
  alternate between the two modes. (The original reading — that a session picked
  a mode and kept it — was an artefact of the disk cache described below.)
- It is **not a correctness problem**. Output is bit-identical in both modes:
  same text, same `z_score = 3.175`, same `unique_contexts`. Only latency moves.
- It is **host-bound, not device-bound**. The GPU sits at ~25 % mean utilisation
  in *both* modes at full SM clock, with no external process on the device.
- It is **specific to program size, with a sharp knee**. At `depth = 1` and
  `depth = 3` the same inferlet is rock-stable once warm (780–864 ms, ±3 %). At
  the production `depth = 9` it is bimodal. The fast mode (~1.3 s) is the
  *correct* cost; the slow mode is a stall.
- Ruled out: GPU contention, compiler nondeterminism (no hash-ordered iteration
  in `compiler.rs`), run-ahead depth (2, 4, 6, 8 and 12 are all bimodal), and
  read-back ring capacity (extra slack shifts the ratio slightly, fixes
  nothing).

A10 is also the only inferlet that reads back **three** channels per token
rather than one. The working hypothesis is that at `depth = 9` its per-fire host
cost reaches parity with its device time, leaving it balanced exactly on the
pipelining knee — but deepening the run-ahead window does not rescue it, so that
account is incomplete.

**A cold plan costs 12–31 s to compile, and nothing says so.** Fused regions are
NVRTC-compiled on first use and cached on disk at `~/.cache/pie/ptir-cuda` (537
modules, 176 MB on this machine). A cache hit is invisible; a miss is a
30-second stall inside what looks like an ordinary request. The cache key covers
the plan bytes, so *any* shape change mints a new entry — including a changed
channel capacity or a changed `depth`, both of which an author would reasonably
regard as tuning rather than recompilation. This is the largest single latency
in the system and it is entirely undocumented at the surface.

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
| **Performance** | Overhead is marginal, never fixed. The real limit is lost run-ahead for any host-dependent step — architectural, not incidental. The largest single latency is invisible: a cold plan pays a 12–31 s NVRTC compile. |
| **Expressiveness** | Algorithms read close to their equations. Four silent-failure contracts and one leaking shape rule are the sharp edges; the missing attention-score intrinsic is the one hard wall. |
