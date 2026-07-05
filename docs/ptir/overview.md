<!-- Source of truth: wiki page `tensor-ir-plan.md` (slug tensor-ir-log). This folder is the split, on-disk copy for implementation teams. -->

# Serving Programmable Dataflows

PTIR — the pie tensor IR — is the programming model at the inferlet/engine
boundary. An inferlet —
sandboxed wasm driving one model — steers GPU inference by attaching **tensor
programs**, closures traced once into PTIR graphs, to the **forward passes** it
submits; the engine batches instances by traced program, stage by stage:
instances sharing a forward trace co-batch its forward even where their
epilogues differ. Everything per-step or per-instance therefore lives outside
the graph, in **channels**; everything trace-known lives inside it, as
constants and shapes. §1–§5 define the model —
values and memory, programs, pipelining, intrinsics, the forward contract —
§6 stress-tests it on speculative decoding, beam search, contrastive search,
MCTS, and LoRA, §7 records the lowering: commit (§7.1), the fire rule
(§7.2), compilation (§7.3) — and the appendix pins the first-party op set.
The document is self-contained.


0. Design principles

Elegance and performance are not traded off; when they seem to conflict,
either a datum sits in the wrong category or a contract over-specifies
mechanism.

- **D1 — Contracts specify observations, not mechanisms.** `put`/`take`/`read`
  fix what a program may observe, never what the machine does: "blocks" may
  lower to predicated retry, host puts coalesce into one transfer per submit,
  and a `bool` tensor's wire format (packed bits) is the runtime's. Choose the
  *semantic* dtype; leave representation free.
- **D2 — Every datum lives in the category of its variance.** Trace-invariant
  → `Tensor::constant`; per-step → channel contents; per-instance → *data*,
  never a constant (a slot id baked into the trace welds equal programs to
  unequal graphs); shapes bind to trace-known capacities, never live sizes.
- **D3 — Invariants are structural, and the runtime monetizes them.**
  Single-producer/consumer channels → epoch rings, zero-copy commit (§7.1);
  inherit-only slot ids → fence-free concurrent GC (§5.2); channel-effect
  mid-pass hooks → capture; a linear `take`→`put` → in-place update (§1).
- **D4 — Costs are prepaid at the cheap edge of the dataflow.** Divergence
  costs the decode path nothing — siblings freeze their view of the shared
  page, one designated child keeps its tail, dead branches reclaim off-path
  (`compact`, §5.2); headroom `alloc` ahead; masks primed ahead; virtual
  loss instead of waiting. Never fence, never cancel.
- **D5 — The escape hatch is a named op, never a loosened core.**
  `pivot_threshold`, `envelope_dot`, `mask_apply` are op-intrinsics;
  `gather_tokens` — `compact`'s kernel (§5.2) — is the same principle below
  the program surface. Extension creep lands in the second-party registry
  (§4), and the first-party core stays closed and fusable (appendix).


1. Tensors and channels

PTIR separates values from memory:

- A `Tensor` is an **SSA value** — what ops consume and produce, what intrinsics
  return, what `take`/`read` yield inside a program. Immutable, freely
  duplicated; a trace-known one comes from `Tensor::constant`.
- A `Channel` is GPU-resident **ordered memory**: a bounded queue of **cells**,
  each a slot with a full/empty bit. A channel carries tensors. The bit (not
  host code) is what enforces ordering: every `take`/`read`/`put` blocks on the
  bit, so producer→consumer happens-before edges are automatic. Capacity is a
  trace-known constructor arg; the examples use capacity 1 unless noted
  (deeper run-ahead = larger capacity, §3).

Misuse is a type error, not a discipline: ops accept only `Tensor`, descriptors
and `put`/`take`/`read` only `Channel`. (`Tensor` names the value — as in ML usage; the
synchronizing construct is the channel.) CPU-side data is never either — a host
`take` yields a plain wasm `Vec`.

```rust
let a = Channel::new([shape_dim1, shape_dim2], dtype::f32);   // capacity 1; .capacity(n) widens

// semantics by full/empty bit (same on host and inside the program):
a.put(v)   // empty -> fill, set full ; full -> BLOCK until empty (back-pressure)
a.take()   // full  -> return v (a Tensor in-program), set empty ; empty -> BLOCK until full
a.read()   // full  -> return copy, stay full ; empty -> BLOCK until full

// on the host these block asynchronously:
a.take().await   // move the value out to a wasm Vec (cell emptied)
a.read().await   // copy the value out (cell preserved)
a.put(v)         // hand a value to the GPU

// inside a program the same semantics hold, but nothing suspends on-device:
// readiness + resubmission realize them (execution rules below).
```

A `Tensor`'s type — and a `Channel`'s element type — is `(shape, dtype)`: a
concrete scalar type (`dtype::f32 | i32 | u32 | bool`) or a **model-intrinsic
type** bound late from the backend (e.g. `intrinsics::activation_type` —
bf16/fp8/int8/…), so one traced program serves every backend.

The full/empty discipline yields two happens-before edges for free:

- `put_t -> take/read of t`  (a consumer waits until the cell is full)
- `take_t -> put_{t+1}`      (a producer waits until the cell is empty)

A channel is **single-producer, single-consumer**: one writer endpoint, one
reader endpoint, fixed at bind time (the host or a named stage) — a second
endpoint is a bind error, and D3's monetizations (§7.1) assume the pair. The
pair may span pipelines (§6.3's two-model feed): SPSC constrains endpoints,
not clocks.

Two more constructors:

- `Channel::from(v)` — sugar for `new` + `put`: a channel seeded full with `v`.
- `Tensor::constant(v)` — a trace-known **value**, *not* a channel: no
  full/empty bit, freely duplicated. `v` is trace-known, and only trace-known:
  small constants fold to launch immediates, large ones (e.g. a DFA table)
  lower to resident immutable buffers. Channels are memory; tensors are SSA
  values. A per-step value is *not* a constant: host-known → host-fed channel
  (`mask`-style; the bit guarantees submit N sees value N), device-derivable →
  in-graph counter channel (§3's `len`).

> Loop-carried state lives in device-resident channels; ordering comes from the
> bits, never from per-step re-binding by the host. Channels are the *only*
> stateful construct — everything else a program touches is an immutable value.

> **Reserved: `Register`.** A latest-value store (`set` overwrites, `get`
> samples the newest value; no ordering edges, never a readiness input) is
> deliberately *not* in the model — every current need is channel-shaped.
> Device→host status is a loop-carried channel the host peeks with `read()`;
> a host→device knob rides the submit loop (one `put` paired with each
> submit, drained by the epilogue program). A register becomes necessary
> exactly when writer and reader share no clock: engine-published values
> (load, memory pressure), feedback between device programs running at
> different rates (two-model speculation), live control of deeply
> pre-submitted steps, non-perturbing observation taps. (§6.5's live weight
> update walks up to this line and stays channel-shaped: the update rides a
> step's prologue.) Recorded as direction; the name is reserved. Corollary:
> no intrinsic may return a time- or load-varying value — that is a register
> read in disguise (unpaired, non-replayable).

Four execution rules complete the channel semantics:

- **Readiness, not blocking.** On-device, a channel op never suspends a
  running pass. Readiness is checked per stage, as each program starts: every
  channel it `take`s or `read`s must already be full — and, the put side of
  the same discipline, a channel whose *first* op is a `put` must already be
  empty (back-pressure, §7.1). An instance missing an
  input still runs — the batch stays uniform — but on **dummy values** (each
  cell's last committed value, so shapes and bounds always hold), and a pass
  is atomic per instance: unless *every* stage found its inputs full, no take
  consumes and no put lands, and the runtime resubmits the instance in
  submission order. Persistent starvation hits a deadline and poisons (rule
  below). Host-side `put`/`take`/`read` block for real (async).
- **Per-channel program order.** Within one traced program, same-channel
  operations run in program order; different channels schedule by dataflow.
  (Otherwise §6.1's `cursor.take()` could run before the descriptor's
  `cursor.read()` and deadlock the step.)
- **No cancellation.** A submitted pass always runs; state is corrected by
  *later writes* (§6.1's reject tail), never by undoing. `Pipeline::close()`
  (implied by drop) only signals that no further submissions are coming:
  in-flight passes run as far as their inputs allow, parked instances are
  reclaimed at teardown, and channels those programs touched are unspecified
  after `close`.
- **Faults poison.** A pass that dies on a device fault — or starves past its
  readiness deadline, an engine policy, never a per-pass knob — poisons every
  channel it touches; a blocked host
  `take()`/`read()` then errors instead of hanging — the `?` in
  `out.take().await?`.

One cost guarantee (D3): a **linear** `take`→`put` — the taken value flows into
exactly one `put` on the same channel, the counter/ping-pong idiom — lowers to
an in-place update with undo at the mutation's granularity (§7.1). A one-row
`scatter_set` into a `[L_max, d]` history channel (§6.3) costs one row, not a
re-materialization of the tensor.


2. Tensor program

A program is a closure **traced once** into a PTIR graph. Closures return
nothing: their effects are channel `put`/`take`s — an output is just a channel
some attachment point consumes — plus §4's configuration sinks (programs
configure the pass; they never write memory — pages are the working set's,
§5.2).
`if`/`for` are resolved at trace time; data-dependent choice inside the graph
uses `select`; a genuinely different branch is a different traced program
(batch-by-program). An **instance** is one binding of a traced program to
its channels and working set: the trace is the identity the engine batches
by (§5.3), the instance is the state it advances.

```rust
let sampled = Channel::new([1], dtype::i32);

let p = || {
    // traced once; this `if` is resolved at trace time, not per-step.
    let logits = intrinsics::logits();        // stage-scoped value (§5.3)
    if SPECULATIVE {
        sampled.put(reduce_argmax(logits));   // effect = a put into a channel
    }
};
```


3. Pipelining

`Pipeline` submits passes run-ahead: step *t+1* is enqueued before *t*
completes; ordering is carried entirely by the channels' bits.

Five channels, each a single producer/consumer pair:

- `tok`  — loop-carried token. `take`/`put` **ping-pong**: `put_t -> take_{t+1}`
  is `W_t -> R_{t+1}` for free. The host never touches it.
- `len`  — committed length. Descriptor peeks (`read`), sampler advances
  (`take`/`put`) — ordered per-channel (§1); the host never feeds it.
- `out`  — host-facing. Back-pressure on the single cell makes the stream
  **lossless**; deeper run-ahead = capacity N (the channel is the queue;
  capacity 1 here).
- `mask` — host `put`s each step, the program `take`s: the one host-coupled
  edge — a late mask dummy-runs the sampler until it lands (§1). `bool`, so
  the wire carries packed bits (D1).
- `rng`  — sampler state `[key, ctr]`. `take`/`put` ping-pong like `tok`; noise
  is a pure function of `(key, ctr)`, never of timing, so a replayed pipeline
  reproduces its tokens exactly. Host seeds it once, then never touches it.

```rust
let CTR1 = Tensor::constant([0u32, 1]);                    // rng counter increment
let tok  = Channel::new([1], dtype::i32);                  // loop-carried, GPU-resident, host never touches
let out  = Channel::new([1], dtype::i32);                  // host-facing output
let mask = Channel::new([intrinsics::vocab], dtype::bool); // grammar mask (host produces each step);
                                   //   bool is the semantic dtype -- the wire format (packed bits,
                                   //   32x smaller than an f32 bias) is the runtime's (D1)
let len  = Channel::from([1u32]);   // committed length -- in-graph counter, host never feeds it
let rng  = Channel::from([seed, 0u32]);  // sampler state [key, ctr]: the seed is per-request *data*
                                         //   (same traced program); the counter advances in-graph
tok.put([BOS]);                    // seed token -> cell full

let ws = WorkingSet::new();
ws.alloc(div_ceil(1 + MAX_TOKENS, ws.page_size()));
                                   // provision up front (§6.1 grows on demand instead). Prompt
                                   //   prefill is elided: the same contract, chunked -- [chunk]-token
                                   //   rows, no readout until the last (§5.1); here the sequence
                                   //   starts from the seeded BOS
let p  = Pipeline::new();

let fwd = ForwardPass::new();
fwd.embed(&tok, LANE_1);                 // LANE_1 = constant indptr [0, 1]: one lane, one token;
                                   // inside: tok.take() -- consume (ping-pong)
fwd.attn_working_set(&ws, &len);   // sugar arity (§5.1): binds the slot pool, and read range +
                                   //   append slot both derive from `len` (peeked)
fwd.epilogue(|| {
    let logits = intrinsics::logits();
    let r = rng.take();
    let g = gumbel(r, [intrinsics::vocab]);  // composed from the `rng` primitive: -log(-log u), fuses
    let t = reduce_argmax(add(mask_apply(logits, mask.take()), g));
                                   // Gumbel-max: an exact sample from the masked distribution;
                                   //   mask.take() is the step's readiness input (§1). mask_apply =
                                   //   select(m, x, -inf), a composed map.
                                   //   temperature/top-p/min-p are map + pivot-threshold ops on
                                   //   `logits` before the mask; greedy is g = 0.
    rng.put(add(r, CTR1));         // advance the counter (ping-pong, like `tok`)
    tok.put(t);                    // refill loop-carried channel for the next step (W_t)
    len.put(add(len.take(), 1));   // advance geometry in-graph
    out.put(t);                    // publish to the host (lossless via back-pressure)
});

// Software-pipelined: prime mask_0 + submit step 0, then submit t+1 BEFORE
// harvesting t. Without the priming put, sample_0 dummy-runs on an empty mask
// until the deadline poisons it, while the host parks at out.take() (§1).
mask.put(initial_mask());                 // mask_0 from the root grammar state
p.submit(&fwd);                           // step 0 in flight

for _ in 0..MAX_TOKENS {
    p.submit(&fwd);                       // submit t+1 NOW: its fwd runs while the
                                          // host handles t; sample_{t+1} parks on `mask`
    let readout = out.take().await?;      // harvest t: blocks until full, empties -> wasm Vec
    println!("New token: {}", readout);
    if readout == STOP_TOKEN {
        p.close();                        // no further submissions; the parked t+1 is
        break;                            //   reclaimed at teardown (drop implies close)
    }
    mask.put(next_mask(readout));         // readies the in-flight sample_{t+1}
}
```


4. Intrinsics

Intrinsics split by **who owns the contract** — the namespace encodes the
contract, never the mechanism (D1):

- **First-party** (`intrinsics::*`): this spec owns the semantics; every
  conforming backend provides them. Model/backend constants (`page_size`,
  `num_layers`, `vocab`, `activation_type`), stage-scoped pass values
  (`logits()`, `hidden()`, `query()`, `layer`), the sampling primitives (`rng`,
  `pivot_threshold` — replay determinism is spec semantics), the composed ops
  (`gumbel`, `mask_apply` — sugar over the core set, first-party by
  construction). A few are **model-gated** — `mtp_logits()`, `value_head()`:
  spec-owned meaning, availability a model property, checked at bind. There
  is no memory op: programs never write pages — reclamation is the working
  set's (`compact`, §5.2), and its `gather_tokens` kernel is engine
  machinery, not a program op.
- **Second-party** (`intrinsics::kernel::*`): the backend owns the semantics;
  the spec fixes only the calling convention — trace-known shapes, no effects
  beyond declared outputs, and §1's corollary (no time- or load-varying
  returns) binds every party. Availability is a bind-time property: a program
  naming a kernel the backend lacks fails to bind, and the fallback is a
  *different traced program* (batch-by-program), never dynamic dispatch
  in-graph. Extension creep lands here (D5); promotion into first-party is a
  spec event.

```rust
// first-party constants (trace-known, late-bound per backend):
intrinsics::page_size
intrinsics::num_layers
intrinsics::activation_type   // backend dtype: any quantized float (bf16/fp8/int8/...)
...

let a = Channel::new([intrinsics::page_size, intrinsics::num_layers], intrinsics::activation_type);
```

Second-party kernels — big ops wrapping vendor kernels, deliberately outside
the standard op set; in PTIR just `Op::Intrinsics(...)`:

```rust
// pre_attn: max possible q·k per page from the key min/max envelope (Quest);
// the selection feeds this layer's attn_page_mask sink (§6.1)
let score = intrinsics::kernel::envelope_dot(q);            // -> [P_MAX] importance score
intrinsics::kernel::attn_page_mask(pivot_threshold(score, rank_le(budget.read())));

// post_attn (Tier-A): attention mass over the last `window` query rows
// (SnapKV); on_attn runs once per layer (§5.3)
let imp = intrinsics::kernel::attn_score_window(window);         // -> [kv_heads, kv_len], this layer
stats.put(scatter_set(stats.take(), intrinsics::layer, imp));
                                             // [num_layers, kv_heads, kv_len], seeded; linear
                                             //   take->put: an in-place row write (§1)

// (pre-KV-write transforms -- per-token low-bit cache quantization, KIVI /
// TurboQuant -- would need a per-layer tap ahead of the KV append plus a
// payload sink; neither exists in §5.3. Recorded as direction.)

// configuration sinks: ops with no result whose effect is to configure THIS
// pass's forward -- stage-precedence rules and scoping in §5.3.
// 2-D sparse attention, per-head pattern bound as data (MInference):
fwd.prologue(|| intrinsics::kernel::minference_sparse(patterns.read()));
// low-rank deltas at declared projection sites (LoRA, §6.5):
fwd.prologue(|| intrinsics::kernel::lora(A.read(), B.read(), SITES));
```


5. The forward contract

5.1 Ragged inputs

A `ForwardPass` is fully described by ragged tensors (CSR: flat array +
`indptr`). Decode, (chunked) prefill, speculative windows, beams and mixed
batches are not modes — they are the *same* contract with different contents:

```rust
fwd.embed(&toks, &tok_indptr);               // [nnz], [B+1]  token ids per lane; consumes (take).
                                             //        Spliced modalities are sibling ports of the
                                             //        embed family (embed_images, embed_audio --
                                             //        anchors are data); out of scope here
fwd.positions(&pos);                         // [nnz]  RoPE positions (trees/splices are data);
                                             //        defaults to append order
fwd.attn_working_set(&ws,                    // attention's memory, one call: binds the slot pool
                                             //        (one per forward, host owns its shape; §5.2)
    &pages, &page_indptr,                    // page family: lane b reads pages[b][..] (peeked)
    &kv_len,                                 // the length column (peeked): [B] per-lane totals =
                                             //        the PHYSICAL span -- only each lane's last
                                             //        page partial; a frozen fork page (§5.2) is
                                             //        presented full. Sub-page validity is program
                                             //        state, lowered to the attention mask (§5.3):
                                             //        the engine never carries per-page lengths
    &w_slot, &w_off);                        // token family: [nnz] where each token's KV lands --
                                             //        the promise: exactly there; optional --
                                             //        omitted = append at the length column. A
                                             //        trailing trace-known `capacity` caps the
                                             //        slot axis (§6.1)
fwd.readout(&out_idx);                       // [n_out] which positions are read out -- logits(),
                                             //        hidden(), value_head() all follow it;
                                             //        defaults to the last token of each lane

// sugar arity for the single-sequence case (page order = the ws's host order):
fwd.attn_working_set(&ws, &len);             // read range + append slot both derive from `len`
```

| mode               | tokens            | attn write          | readout        |
|--------------------|-------------------|---------------------|----------------|
| decode × B         | 1 per lane        | each lane's tail    | all            |
| (chunked) prefill  | chunk             | contiguous range    | last (or none) |
| speculative (§6.1) | K+1 per lane      | provisional range   | all K+1        |
| beam (§6.2)        | 1 per lane        | private tail slot   | all            |
| contrastive (§6.3) | 1 per candidate   | private tail slot   | all            |
| MCTS eval (§6.4)   | 1 per leaf        | the node's slot     | all            |
| mixed batch        | prefill + decode lanes coexist | each their own | each their own |

One table, two column families. Token-indexed columns (`embed`,
`positions`, attn's `w_slot`/`w_off`) share `tok_indptr`; page-indexed ones
(attn's `pages`) carry their own `page_indptr`; length is `kv_len` totals —
sub-page validity (frozen forks) rides the attention mask (§5.3), and both
derive in-program from one lens channel, the single source (§5.2, §6.2) — and
`readout` picks positions off the token axis. Consumption splits the same way: the
token family **consumes** (`take` — a token is spent by the pass that embeds
it), geometry **peeks** (`read` — geometry is state, not a message).
Host-re-fed geometry (§6.4) is refreshed by a host `take`→`put`: passes in
flight have already peeked their values at pass start, and the pass that
needs the new ones is submitted after the `put`.

Two staticness rules keep capture safe:

- **Shapes are trace-known.** `nnz`, `B`, `n_out` are fixed per traced program;
  a different size is a different bucket, i.e. a different program
  (batch-by-program). Unused lanes are validity-coded, never reshaped away.
- **Contents are data.** Indices, offsets and lengths live in channels —
  host-fed (`mask`-style) or device-computed (§3's `len`, §6.1's `cursor`) —
  and change every step without recompiling.

Rectangular batches are the degenerate ragged case: their `indptr` is an
arithmetic progression, itself trace-known, so it folds to a constant (`LANE_1`,
`LANES_B` in the examples). An `indptr` needs to be a *channel* only when the
lane split moves per step at fixed `nnz` — e.g. a mixed batch re-packed each
step.

5.2 Working-set access

The working set is one flat pool of page slots; a slot id is a **stable
handle** to a (CoW-shared, refcounted) page object — it never renumbers.
Ownership is split:

- **The host owns the shape.** `ws.alloc`/`ws.free` only on the host (memory
  contention and eviction are host decisions). `alloc` grants fresh or
  recycled slot ids, in-flight safe; `free` is **non-compacting** (tombstone +
  free-list) and requires only that the freed slots are unreachable from every
  in-flight pass — survivors keep their ids, so device-resident index tensors
  stay valid across `free`.
- **The program owns geometry, never contents.** Ordinary index tensors are
  the whole authority: which pages a lane reads (`pages`), how far each is
  valid (a lens channel — program state the engine never carries), where its
  tokens land (`w_slot`/`w_off`) — the §5.1 columns. The only content writes
  are the forward's own KV appends — there is no copy op.
  Divergence is a *freeze*, not a copy: each sibling's valid length for the
  shared page just stops advancing (bookkeeping, not an op), while one
  *designated child* may keep writing past every frozen prefix — the
  invariant below permits exactly one such writer. The engine sees the lens
  channel only through its two in-program derivatives: the physical totals
  (§5.1's `kv_len`, frozen pages counted full) and the KV validity mask
  (§5.3's `attn_mask`) — the kernel reads masked residuals but excludes them
  from attention, a bounded sliver (under a page per frozen boundary), so no
  attention kernel ever changes for forks. The other children open
  fresh slots; only a branch that *dies* strands the tokens it wrote past
  the live prefix, as waste for `compact` (below) — the decode path pays
  nothing either way (D4). The ws handle never appears in
  program scope: it exists only in host code and pass descriptors (a pass
  binds exactly one ws; cross-ws sharing is host-side `slice`/`append`/`fork` —
  each builds a new set from existing pages by reference, refcounted CoW:
  §6.5's prefix tuning). Programs never grow or shrink the set; they can only *inherit* slot
  ids (constants, host-fed channels, gathers of existing index tensors),
  which is what makes host-side reachability tracking sound — and it makes a
  program's identity ws-independent: the same traced graph, bound to
  different working sets, is the *same* program (batch-by-program across
  instances).

Reclamation is host-driven and two-tier; neither tier is ever a side effect
of `free`:

- **Whole-page `free`** over stable ids (physical memory is refcounted behind
  the handles; a freed slot's page returns to the arena when its last
  reference drops): **exact** when the host owns the slot map (§6.4), or
  **concurrent mark-sweep** over a peeked snapshot when geometry lives on
  device (§6.2) — no fence either way. The arena recycles a freed page only
  after every pass in flight at `free`-time has retired (an RCU-style grace
  period), so a host bug that frees a still-referenced slot reads its own
  stale data, never another working set's.
- **Token-space `compact`** for the waste `free` cannot see: frozen fork
  tails (mask-excluded residuals, above) and token-scattered eviction (H2O-style
  policies that rewrite the past). `compact` rewrites live tokens densely —
  the `gather_tokens` kernel (D5), engine machinery, never a program op —
  and returns the old→new remap, which the host re-feeds through the
  geometry channels (§5.1's host `take`→`put`). Like index-space
  defragmentation (its degenerate, no-move case), it is an explicit
  quiescent-point operation when geometry lives on device (§6.2 drains its
  in-flight step first) and an ordinary re-feed when the host owns it
  (§6.4); old pages return to the arena after the usual grace period.

One invariant: any number of lanes may *read* a slot up to its declared valid
length (aliasing = prefix sharing); past it, a slot *written* this step
belongs to exactly one lane. The read bound is enforced by the mask the
program feeds (above), never by the kernel; the write bound by the
`w_slot`/`w_off` promise (§5.1).

> **Design note.** [`working-set.wit`](../interface/inferlet/core/wit/working-set.wit)
> today specifies the opposite: `free` densely compacts, `alloc` returns a
> contiguous range, and a `generation` counter lets the *host* detect stale
> indices at submit — a defense that cannot cover device-resident indices
> (§6.2's `pages`). Migrating: `free` stops compacting (`size()` then counts
> live slots, not array length), `alloc` returns slot ids (fresh or recycled)
> instead of a contiguous range, `generation` bumps only on
> `reorder`/`compact`, and explicit `compact()` — now token-space, above —
> replaces compaction-on-free. GC itself needs **no new API**: mark is host
> code, sweep is `free`. Recorded as direction; the wit is unchanged.

5.3 Attachment points

Hooks observe or feed a pass at fixed stages; everything they exchange moves
through channels:

```rust
fwd.prologue(|| { ... });                        // pass prologue: before any KV read -- weight
                                                 //   swap (§6.5), config sinks (§4)
fwd.on_attn_proj(|| intrinsics::kernel::attn_page_mask(...));
                                                 // e.g. Quest page selection (§6.1): a sink,
                                                 //   consumed by this layer's attention
fwd.attn_mask(&m);                               // masks this pass's queries over the KV axis, past
                                                 //   + window: tree-speculation structure, fork
                                                 //   validity (§5.2) -- bool, data, peeked like
                                                 //   geometry (§5.1); absent = causal. Shaped by the
                                                 //   trace-known cap ([*, P_MAX * page]), packed on
                                                 //   the wire (D1)
fwd.on_attn(|| { stats.put(...) });              // e.g. SnapKV window scores (§4; per layer)
fwd.epilogue(|| { ... });                        // sampling programs (§2, §6)
```

No stage is privileged: each hook is an ordinary program under §1's rules —
readiness checked as it starts, effects through channels — and stages differ
in when they run and which intrinsics are in scope (`query()` and
`intrinsics::layer` at the attn taps; `logits()`, `hidden()`, `mtp_logits()`,
`value_head()` at the epilogue). Boundary stages run once per pass; the
anatomical taps (`on_attn_proj`, `on_attn`) run **once per layer**.
`intrinsics::layer` is the invocation's layer index — a replayable
per-invocation value, not a register read (§1) — and per-layer accumulation
is the linear `take`→`scatter_set(layer, ·)`→`put` idiom (§1; §4's SnapKV).
Readiness stays per pass: one check covers all of a tap's invocations.
Batching follows the same per-stage structure: a pass's identity is the tuple
of its stage traces, and instances co-batch stage by stage — a shared forward
batches even where epilogues differ.
Configuration sinks (§4) — `attn_page_mask`, `lora`, `minference_sparse` —
are ordinary ops in a stage program: they take tensors, return nothing, and
their effect is to configure this pass's forward. Each sink names the point
that consumes its effect, and the trace rejects a call at any stage that does
not precede it (pass-wide sinks are prologue-only; `attn_page_mask` in the
prologue masks every layer, at attn-proj that layer alone). Every pass
contains a forward; there is no bare pass. Pre-forward work (a weight swap, a
config sink) rides the `prologue`, so the scheduler's unit stays homogeneous —
nothing tiny ever sits in the stream to order, batch, or retry on its own.


6. Design by examples

6.1 Stress test — MTP speculative decode + grammar constraint + Quest attention

One `ForwardPass`, three independent programs at three stages. Speculation is
*data* (sentinel-coded lanes), the constraint is *data* (per-position masks),
Quest is a *graph-internal selection*; capture stays homogeneous (§5.1) — only
channel contents differ per step.

```
stage             program                                    channel(s)
----------------  -----------------------------------------  -------------------
on_attn_proj      Quest: top-`budget` pages by key envelope  (sink: attn_page_mask)
epilogue          MTP: propose K drafts + sampled verify     toks, prev_drafts, out, draft_out
host loop         grammar: per-position constraint           mask      (host-produced)
```

`K` = MTP draft width. `P_MAX` = trace-known slot capacity
(`div_ceil(MAX_LEN, page)`): per-page values (the page mask) are shaped by the
*cap*, never the live size — the ws grows to `P_MAX` without re-trace; past it
is a new bucket (§5.1). There is no `num_pages` intrinsic: a ws's live size is
runtime state, not a shape.

```rust
// trace-known values (Tensor::constant, K = 4):
let DRAFT_LANES  = Tensor::constant([1u32, 2, 3, 4]);   // draft input lanes toks[1..=K]
let VERIFY_LANES = Tensor::constant([0u32, 1, 2, 3]);   // target positions that verify each draft
let LANE_K1      = Tensor::constant([0u32, K + 1]);     // indptr: one lane, K+1 tokens
let LANE_IDX     = Tensor::constant([0u32, 1, 2, 3, 4]);
let ZERO_IDX     = Tensor::constant([0u32]);
let NEG1         = Tensor::constant(-1i32);             // empty/reject sentinel
let ONES_K       = Tensor::constant([1.0f32; K]);
let ZEROS_K      = Tensor::constant([0.0f32; K]);
let CTR1         = Tensor::constant([0u32, 1]);         // rng counter increment

let ws    = WorkingSet::new();
let page  = ws.page_size();                        // tokens per KV page (token-agnostic set; inferlet owns length)
let P_MAX = div_ceil(MAX_LEN, page);               // trace-known slot capacity: shapes of per-page values --
                                                   //   the ws grows to P_MAX without re-trace; past it = new bucket

// channels (stateful, full/empty cells). Channel::from(v) = new + put (seeded full).
let toks    = Channel::from([BOS, -1, -1, -1, -1]);                // loop-carried window [committed, d1..dK]; dtype i32
                                                                   //   -1 = empty lane (engine contract: negative id = padding, no embed/KV effect)
let prev_drafts = Channel::from([-1i32; K]);                       // device copy of last step's drafts d1..dK -- second
                                                                   //   consumer of the window: embed() takes `toks`,
                                                                   //   verify still needs the values (cf. §6.3 `cands`)
let out     = Channel::new([K + 1], dtype::i32);                   // host-facing: tokens committed this step (lossless via back-pressure)
let draft_out = Channel::new([K], dtype::i32);                     // host-facing: drafts proposed for t+1 (host walks the grammar along these)
let mask    = Channel::new([K + 1, intrinsics::vocab], dtype::bool); // grammar mask (true = allowed); packed on the wire:
                                                                   //   ~80KB/step vs ~2.6MB as an f32 bias (D1)
let budget  = Channel::from([256u32]);                             // Quest knob (pages/step); read() (peek), never consumed.
                                                                   //   Seeded once: to tune it live, feed it per-step like
                                                                   //   `mask` and drain it in the epilogue (a knob is paired
                                                                   //   data -- there is no register, §1)
let cursor  = Channel::from([1u32]);                               // device: committed KV length L = append offset (seeded: BOS)
let rng     = Channel::from([seed, 0u32]);                         // sampler state [key, ctr]; per-request seed is data

let headroom = 2 * (K + 1);                        // spare token-slots: >= (run-ahead depth 1 + 1) windows
let mut clen = 1u32;                               // host mirror of L (coarse page alloc only)
ws.alloc(div_ceil(clen + headroom, page));         // provision spare pages so a full K+1 window always has slots

let p   = Pipeline::new();
let fwd = ForwardPass::new();
fwd.attn_working_set(&ws, &cursor, P_MAX);
                                       // sugar arity (§5.1): read range + append offset derive from
                                       //   `cursor` (peeked) -- the geometry *contents* are a device
                                       //   channel here, spec-dependent. Trailing capacity P_MAX:
                                       //   the binding's trace-known slot cap = the page mask's shape

// --- Quest (inside the captured graph): page importance -> rank-threshold -> selection ---
fwd.on_attn_proj(|| {                                            // runs once per layer (§5.3)
    let q     = intrinsics::query();                             // this layer's projected query
    let score = intrinsics::kernel::envelope_dot(q);             // [P_MAX] importance from this layer's key
                                                                 //   envelope; slots beyond the live size
                                                                 //   are validity-coded (-inf)
    intrinsics::kernel::attn_page_mask(pivot_threshold(score, rank_le(budget.read())));
                                                                 // [P_MAX] bool, top-`budget`: a sink (§4) --
                                                                 //   this layer's attention consumes it; no
                                                                 //   channel, no port (produced and consumed
                                                                 //   inside one forward)
});

// --- MTP speculation + sampled verify (epilogue, outside capture) ---
fwd.embed(&toks, LANE_K1);                                             // one lane, K+1 tokens; inside: toks.take()
fwd.epilogue(|| {
    let logits     = intrinsics::logits();                       // [K+1, vocab]
    let mtp_logits = intrinsics::mtp_logits();                   // [K, vocab] future-token heads
    let m      = mask.take();                                    // [K+1, vocab] bool grammar mask (readiness input, §1)
    let r      = rng.take();
    let g      = gumbel(r, [K + 1, intrinsics::vocab]);          // one fresh Gumbel row per position
    rng.put(add(r, CTR1));                                       // advance counter (ping-pong)
    let picked = reduce_argmax(add(mask_apply(logits, m), g));   // [K+1] Gumbel-max: picked[i] is an exact
                                                                 //   sample from the masked target at position i

    // match-verify: draft at lane i accepted iff it equals the *sampled* token at position i.
    // Lossless: committed tokens ARE the samples; the draft only decides how many commit.
    let d      = prev_drafts.take();                             // [K] = d1..dK, proposed last step
    let head   = gather(picked, VERIFY_LANES);                   // [K] = picked[0..K]
    let hit    = eq(head, d);                                    // [K] bool

    // longest accepted prefix = sum of the leading-AND run
    let runf   = select(hit, ONES_K, ZEROS_K);                   // [K] bool -> f32 {1,0}
    let n_acc  = cast(reduce_sum(cumprod(runf)), dtype::u32);    // scalar: count of leading hits

    // commit lanes 0..=n_acc (accepted drafts + one bonus correction); tail -> -1 sentinel
    let keep   = ge(broadcast(n_acc, [K + 1]), LANE_IDX);        // [K+1] bool: i <= n_acc
    let commit = select(keep, picked, NEG1);                     // [K+1], tail = -1
    out.put(commit);                                             // publish committed tokens

    // next window: lane 0 = bonus token picked[n_acc]; lanes 1..K = fresh MTP drafts.
    // drafts are left unconstrained -- a grammar-illegal draft just fails verify next step.
    let bonus  = gather(picked, n_acc);                          // scalar correction (reduce-result used as index)
    let drf    = reduce_argmax(mtp_logits);                      // [K] fresh drafts for t+1 (greedy on purpose:
                                                                 //   proposals, not samples)
    prev_drafts.put(drf);                                        // refill the device copy (ping-pong)
    draft_out.put(drf);                                          // publish drafts: host needs them to build mask_{t+1} rows 1..K
    cursor.put(add(cursor.take(), add(n_acc, 1)));             // advance L by n_acc+1 (ping-pong); reject tail overwritten next step
    let base   = scatter_set(broadcast(NEG1, [K + 1]), ZERO_IDX, bonus);  // lane 0 = bonus
    toks.put(scatter_set(base, DRAFT_LANES, drf));               // lanes 1..K = drafts
});

// Software-pipelined: prime mask_0 + submit step 0, then submit t+1 BEFORE harvesting t.
mask.put(grammar.speculative_masks(&[]));                       // mask_0: row 0 from the root state; draft rows are
                                                                // don't-care (step-0 draft lanes are -1, never verify)
p.submit(&fwd);                                                 // step 0 in flight

for _ in 0..MAX_STEPS {
    p.submit(&fwd);                                             // submit t+1 NOW: its fwd + Quest attn run
                                                                // ahead; sample_{t+1} parks on `mask` (§1)
    let accepted = out.take().await?;                           // harvest t (t+1's fwd overlaps this) -> wasm Vec
    let drafts   = draft_out.take().await?;                     // drafts d1..dK proposed for t+1 (published by epilogue_t)
    for &t in accepted.iter().take_while(|&&t| t >= 0) {
        if t == STOP { p.close(); return; }                     // no more submissions; the parked t+1
                                                                //   is reclaimed at teardown
        grammar.advance(t);                                     // host FSM walks the committed prefix
        clen += 1;                                              // committed KV grew by one token
    }
    let want = div_ceil(clen + headroom, page);                 // pages needed to keep `headroom` spare tokens
    if want > ws.size() { ws.alloc(want - ws.size()); }         // top up spare slots (in-flight safe: grants never disturb existing ids)
    // row 0 constrains the committed position; row i constrains the slot after drafts d1..d_i.
    mask.put(grammar.speculative_masks(&drafts));               // speculative walk from the committed state; readies sample_{t+1}
}
```

Every on-device line is a core primitive or a named op-intrinsic
(`envelope_dot`); `grammar.*` and the `for` are ordinary host wasm — not IR.

Why the three compose without special cases:

- **Spec ⟂ structure.** Acceptance is a *value* (`n_acc`, sentinel `-1`), never
  a shape: the captured graph is identical whether 1 or K+1 tokens commit.
- **Spec ⟂ memory.** K+1 provisional KV writes per step; `cursor` advances by
  `n_acc+1`, the reject tail is overwritten next step — no per-step free. Host
  memory work is only spare-slot `alloc`, in-flight safe (grants never disturb
  existing ids).
- **Constraint ⟂ spec.** The host walks the grammar along the published drafts
  (`draft_out`) to build `mask` rows; drafts are unconstrained — an illegal
  draft can never equal the masked sample, so it is simply rejected.
- **Verify is a policy, not a mode.** Match-verify (accept iff draft ==
  sample) is lossless but conservative; rejection-sampling verify (accept
  `d_i` w.p. `min(1, p(d_i)/q(d_i))`, resample the residual at the first
  reject) is the same attachment point with a few more ops and the drafter's
  `q` in one more channel.
- **Quest ⟂ both.** The page mask is a sink, produced and consumed layer by
  layer inside one forward, and `budget` is peeked — neither adds an edge to
  the decode chain.
- **Run-ahead survives the constraint.** `submit(t+1)` precedes `out.take(t)`;
  the grammar mask parks *only the sample* — attention and MLP have already
  overlapped — and the `toks` chain never round-trips to the host. What stays
  on the critical path is mask *production*: an incremental matcher first, and
  ultimately a DFA-as-constant walked in-graph (§1), move it off (D4).

> **Design note.** The wit ([`inference.wit`](../interface/inferlet/core/wit/inference.wit))
> still carries scalar geometry args (`inp-start`, `inp-len`, `valid-tokens`,
> `output-start`, `output-len`, `offset`); migrating to §5's index tensors
> needs (1) launch-immediate folding and (2) driver support for
> device-resident offsets. Recorded as direction; the wit is unchanged.

6.2 Beam search — the batch dimension lives in the index data

Lane b's KV sequence is row b of `pages` `[B, P]`: beam reorder is an index
gather, never a KV move — and divergence is a *freeze*, never a copy (§5.2).
Rows alias freely; a fork stops advancing the shared page's valid length for
every reader but one — the parent's *designated child* keeps filling the tail
past its siblings' frozen prefixes (§5.2's invariant: one writer past the
valid length) — so a surviving fork wastes nothing. Only a branch that dies
strands its tail tokens, for `compact` to sweep (§5.2). Valid lengths live in
the program's `lens` channel; the engine sees only their two derivatives —
physical totals (`klen`) and the KV validity mask (`kvm`), both computed in
the same epilogue that updates `lens` (§5.2: one source, no kernel changes).

```rust
let V      = intrinsics::vocab;
let ws     = WorkingSet::new();
let page   = ws.page_size();                    // tokens per KV page
let P      = div_ceil(MAX_LEN, page) + SLACK;   // trace-known row capacity (§6.1's P_MAX): SLACK
                                                //   absorbs frozen-tail waste between compacts (D4)
let LANES  = Tensor::constant([0u32, 1, .., B - 1]);   // lane indices
let LANES_B   = Tensor::constant([0u32, 1, .., B]);      // indptr: one token per lane (§5.1)
let PAGE_ROWS = Tensor::constant([0u32, P, .., B * P]);  // indptr: rectangular [B, P] page rows
let pages  = Channel::from(pages0);             // [B, P] u32: rows share the prompt's pages (aliasing = sharing)
let lens   = Channel::from(lens0);              // [B, P] u32: per-page valid prefix -- PROGRAM state,
                                                //   the single source (§5.2); prompt pages full,
                                                //   fork-frozen pages partial. The engine never sees it
let klen   = Channel::from(klen0);              // [B] u32: physical span (§5.1's kv_len) -- frozen
                                                //   pages counted full; derived from lens each step
let kvm    = Channel::from(kvm0);               // [B, P * page] bool: KV validity mask (§5.3) --
                                                //   kvm[b][j*page+o] = o < lens[b][j]; packed (D1)
let pos    = Channel::from([L0; B]);            // [B] u32: logical length = this step's RoPE position
                                                //   (physical span counts masked holes; logical must
                                                //   not -- explicit positions, §5.1)
let np     = Channel::from(np0);                // [B] u32: live entries per row (tail = entry np-1)
let tslot  = Channel::from(tslot0);             // [B] u32: each lane's tail slot (= its row's last entry)
let tfill  = Channel::from(tfill0);             // [B] u32: each lane's tail fill
let w_slot = Channel::from(wslot0);             // [B] u32 \ where this step's token lands: token family,
let w_off  = Channel::from(woff0);              // [B] u32 / consumed (§5.1); each epilogue refills (ping-pong)
let toks   = Channel::from([BOS; B]);           // [B] i32
let scores = Channel::from([0.0f32; B]);        // [B] f32  running log-prob sums
let fresh  = Channel::new([B], dtype::u32);     // host-fed headroom slots (mask-style, every step; D4)
let out     = Channel::new([B], dtype::i32);    // host-facing tokens (back-pressure)
let out_par = Channel::new([B], dtype::u32);    // host-facing parents: without them the host cannot
                                                //   backtrack the winning hypothesis across reorders
let out_scr = Channel::new([B], dtype::f32);    // host-facing running scores (final ranking)

let p   = Pipeline::new();
let fwd = ForwardPass::new();
fwd.embed(&toks, LANES_B);                      // one token per lane
fwd.positions(&pos);                            // explicit: the append-order default would count
                                                //   masked holes (physical), not logical length
fwd.attn_working_set(&ws,                       // one call binds attention's memory (§5.1):
    &pages, PAGE_ROWS,                          //   rectangular = degenerate ragged (§5.1)
    &klen,                                      //   dense totals; frozen forks presented full (§5.2)
    &w_slot, &w_off);                           //   computed by the *previous* epilogue, like `toks`
fwd.attn_mask(&kvm);                            // sub-page validity rides the mask (§5.2, §5.3)
fwd.epilogue(|| {
    let logits = intrinsics::logits();          // [B, V]
    let cand   = add(broadcast(scores.take(), [B, V]), log_softmax(logits));
    // (stochastic beams = Gumbel-top-k: perturb `cand`, truncate against the
    //  parent's perturbed score carried in one more [B] channel -- same graph shape)
    let (s, i) = top_k(reshape(cand, [B * V]), B);   // lowers to per-row top-B + B^2 merge
    let parent = div(i, V);
    // reorder = gathers; divergence = NOT advancing the inherited lens entry.
    // Each parent designates ONE child to keep filling its tail (any
    // deterministic pick works; here, last child in lane order); the others
    // open fresh slots at offset 0. A surviving fork therefore wastes
    // nothing -- only dead branches leave residue for compact (§5.2).
    // Everything is element-wise except the per-lane-column scatters
    // (composed flat: row b, entry j -> b*P + j).
    let pg   = gather(pages.take(), parent);         // [B, P] row gather -- the reorder
    let pl   = gather(lens.take(), parent);
    let n    = gather(np.take(), parent);
    let tf   = gather(tfill.take(), parent);
    let heir = scatter_set(LANES, parent, LANES);    // heir[p] = p's designated child (duplicate
                                                     //   scatter indices resolve in index order,
                                                     //   last wins; base values are never gathered
                                                     //   -- every parent index is written)
    let cont = and(eq(gather(heir, parent), LANES), lt(tf, page));
                                                     // b continues its parent's tail iff b is the
                                                     //   designated child and the tail has room
    let slot = select(cont, gather(tslot.take(), parent), fresh.take());
    let off  = select(cont, tf, 0);
    let n2   = select(cont, n, add(n, 1));
    let tcol = add(mul(LANES, P), sub(n2, 1));       // flat index of each lane's tail entry
    pages.put(reshape(scatter_set(reshape(pg, [B * P]), tcol, slot), [B, P]));
    let pl2  = reshape(scatter_set(reshape(pl, [B * P]), tcol, add(off, 1)), [B, P]);
    lens.put(pl2);                                   // the source; its two derivatives follow:
    klen.put(add(mul(sub(n2, 1), page), add(off, 1)));   // physical span (frozen pages full)
    kvm.put(reshape(lt(broadcast(iota([page]), [B, P, page]),
                       broadcast(reshape(pl2, [B, P, 1]), [B, P, page])), [B, P * page]));
                                                     // valid iff in-page offset < lens entry
    pos.put(add(pos.take(), 1));                     // logical length (ping-pong)
    np.put(n2);  tslot.put(slot);  tfill.put(add(off, 1));
    w_slot.put(slot);  w_off.put(off);               // next step's write descriptor
    toks.put(rem(i, V));
    scores.put(s);
    out.put(rem(i, V));                              // EOS/stop policy is host-side
    out_par.put(parent);                             // reorder permutation, for host backtracking
    out_scr.put(s);
});

// Host: one program -- a page boundary is just divergence's fresh path taken
// by every heir at once, so no second traced variant, no page schedule.
// Length penalty applies to finished hypotheses only: live beams share one
// length, so raw sums rank identically in-loop.
fresh.put(ws.alloc(B));                         // headroom grant: B ids per step, mostly unused --
p.submit(&fwd);                                 //   unused grants go unreachable and sweep back (D4)
for t in 1..MAX_STEPS {
    fresh.put(ws.alloc(B));                     // fresh or recycled ids (§5.2)
    p.submit(&fwd);                             // submit t NOW; harvest t-1 below (overlap)
    let picked = out.take().await?;             // harvest t-1; lossless via back-pressure
    hyp.push(&picked, out_par.take().await?, out_scr.take().await?);
                                                // host (token, parent, score) log: parents let the
                                                //   host backtrack the winning hypothesis at the end
    if finished(&picked) { p.close(); break; }  // no more submissions; parked work reclaimed
    if t % GC_EVERY == 0 {                      // tier 1 (§5.2): whole-page mark-sweep -- no fence,
        let snap = pages.read().await?;         //   no bubble; peek while step t runs (read = copy)
        ws.free(dead(&mark(&snap)));            //   live = snap entries ∪ ids granted since the peek;
    }                                           //   sound: rows only gather, free never renumbers
    if waste_high(&hyp) {                       // tier 2 (§5.2): token-space compact -- the one
                                                //   quiescent point, at a host-chosen threshold.
                                                //   The host replays the parent log (`hyp`) to
                                                //   model waste exactly: no device readback
        let fin = out.take().await?;            // drain: harvest the in-flight step too; nothing
        hyp.push(&fin, out_par.take().await?, out_scr.take().await?);   // is in flight past here
        if finished(&fin) { p.close(); break; }
        let remap = ws.compact(&live_runs(&hyp));
                                                // live token runs, from the same replay;
                                                //   gather_tokens packs them into fresh slots and
                                                //   returns old->new. Copies ride off the decode
                                                //   path; old pages free after the grace period
        let g = dense_geometry(&hyp, &remap);   // host math: the packed layout -- all pages full
                                                //   except each lane's tail (live forks keep their
                                                //   structural partial page)
        pages.take().await?;   pages.put(g.pages);    // re-feed EVERY geometry channel by the
        lens.take().await?;    lens.put(g.lens);      //   ordinary host take->put (§5.1). The
        klen.take().await?;    klen.put(g.klen);      //   cells are full (the drain retired the
        kvm.take().await?;     kvm.put(g.kvm);        //   last writer), so no take blocks; the
        np.take().await?;      np.put(g.np);          //   next submit reads only the new pages.
        tslot.take().await?;   tslot.put(g.tslot);    //   Values (toks, scores, pos) never move
        tfill.take().await?;   tfill.put(g.tfill);
        w_slot.take().await?;  w_slot.put(g.w_slot);
        w_off.take().await?;   w_off.put(g.w_off);
    }
}
```

- Reorder, divergence, page turnover are `gather`/`select`/`scatter_set` —
  index math only; no program copies or writes a page, and the ws shape
  never changes from the program.
- Divergence costs the decode path nothing (D4), and a *surviving* fork
  wastes nothing: the designated child fills the shared tail's remainder.
  Waste is exactly the dead branches — a hypothesis that falls out of the
  beam strands the tokens it wrote past the live prefix — and it is memory
  plus a bounded residual read: the kernel loads masked slots but the mask
  excludes them from attention, under a page per frozen boundary (§5.2).
  Reclaimed when the host's waste model (replayed from the parent log)
  crosses the compact threshold.
- One program, not two: a page boundary is divergence's fresh-slot path taken
  by every heir at once, so no step needs a second traced variant or a page
  schedule, and `fresh` is plain headroom (D4).
- Reclamation is two-tier (§5.2): concurrent mark-sweep over a peeked
  `pages` snapshot — safe because reachability only shrinks *and* `free`
  never renumbers survivors — plus the explicit quiescent-point `compact`.
  Only the second ever pauses the pipeline, for one drained step, at a
  host-chosen threshold.

6.3 Contrastive search — the lanes belong to the ranking rule

Commit, out of the winner's top-k candidates, the one maximizing
`(1-α)·p(v) − α·max_j cos(h_v, h_j)`. The penalty needs each candidate's
*hidden state*, so candidates must be forwarded before one is chosen — the
lanes belong to the ranking rule. KV machinery is §6.2 verbatim with
`parent = broadcast(w, [k])`; new are one history channel and the selection
rule:

```rust
const ALPHA: f32 = 0.6;                          // trace-known
let len   = Channel::from([l0]);                 // [1] u32 committed length (§3's counter): the
                                                 //   write cursor into `hist`
let hist  = Channel::from(h0);                   // [L_max, d] committed hiddens;
                                                 //   L_max = trace-known cap (§6.1's P_MAX): past it = new bucket
let cp    = Channel::from(p0);                   // [k] p(candidate), from the last winner's logits
let cands = Channel::from(ids0);                 // [k] same ids as `toks` -- second consumer
                                                 // (embed() eats `toks`; the epilogue still needs the values)

fwd.epilogue(|| {                                // descriptor = §6.2's with k lanes
    let logits = intrinsics::logits();           // [k, V]
    let h   = intrinsics::hidden();              // [k, d] last-layer state per lane -- why lanes exist
    let l   = len.take();
    let hh  = hist.take();
    let sim = reduce_max(add(matmul(l2norm(h), transpose(l2norm(hh))),  // [k, L_max] cosine
                             after_mask(l)));    // additive validity: -inf on positions >= l
                                                 //   (composed: select(ge(iota([L_max]), l), -INF, 0))
    let w   = reduce_argmax(sub(mul(cp.take(), 1.0 - ALPHA), mul(sim, ALPHA)));
    out.put(gather(cands.take(), w));            // commit = this step's winning *input*; `out` is
                                                 //   [1] here -- one token a step, not §6.2's [B]
    hist.put(scatter_set(hh, l, gather(h, w)));
    let (np, ni) = top_k(softmax(gather(logits, w)), k);  // winner's row seeds the next candidates
                                                          //   (sampled pool: add gumbel(r, [V]) to the
                                                          //   log-probs before top_k = k draws w/o replacement)
    cp.put(np); toks.put(ni); cands.put(ni);
    len.put(add(l, 1));
    // tail bookkeeping: §6.2's freeze/fresh index math, parent = broadcast(w, [k])
});
```

- The commit is one step late by construction, so the schedule is already
  pipelined: while the host harvests token t, t+1's candidates are in flight.
- `hist` is a linear `take`→`put` (§1, D3): the one-row `scatter_set` lowers
  to an in-place row write, not a per-step re-materialization of `[L_max, d]`.
- Two-model contrastive *decoding* (`log p_expert − log p_amateur`) is the
  same epilogue with the amateur's logits arriving through a channel fed by a
  second pipeline — needs multi-model; recorded as direction.

6.4 MCTS — the tree is a process, the KV tree is index data

Select/expand/backprop are scalar arithmetic under unbounded, data-dependent
control flow — a *process*, not a program: the host keeps the tree. The device
sees only "evaluate B leaves", a plain §5.1 forward whose geometry the host
re-feeds each submission. Sibling paths aliasing the same page rows *is* the
tree.

```rust
// every content is host-fed: the tree decides, channels carry. `toks` is
// consumed (mask-style, §1); geometry is peeked (§5.1) and refreshed by a
// host take -> put before each submit
let ws    = WorkingSet::new();                   // the node->slot map is host state, in the tree
let page  = ws.page_size();
let P     = div_ceil(MAX_DEPTH, page);           // trace-known row capacity (§6.2's P)
let toks  = Channel::new([B], dtype::i32);       // one pending token per selected leaf
let pages = Channel::new([B, P], dtype::u32);    // root->leaf path rows (aliasing = tree structure)
let klen  = Channel::new([B], dtype::u32);       // physical span per leaf path (frozen pages full)
let kvm   = Channel::new([B, P * page], dtype::bool);  // KV validity mask (§5.2, §5.3): a within-page
                                                 //   branch is a page frozen at the fork depth --
                                                 //   children read the shared slot up to the fork
                                                 //   offset, each appends into its own slot; no
                                                 //   copy, no COW pass. Host-computed: the tree
                                                 //   knows every fork offset
let pos   = Channel::new([B], dtype::u32);       // leaf depth = RoPE position (explicit, §6.2)
let wslot = Channel::new([B], dtype::u32);       // host-assigned: a new node's KV is permanent tree state
let woff  = Channel::new([B], dtype::u32);
let vals  = Channel::new([B], dtype::f32);       // out: value head
let prio  = Channel::new([B, k], dtype::f32);    // out: top-k children priors
let pids  = Channel::new([B, k], dtype::i32);

let eval = ForwardPass::new();
eval.embed(&toks, LANES_B);
eval.positions(&pos);
eval.attn_working_set(&ws,
    &pages, PAGE_ROWS,                           // constant indptr (§6.2); rectangular rows
    &klen,                                       // dense totals (§5.1); per-leaf depths AND
                                                 //   within-page forks ride `kvm` (§5.2)
    &wslot, &woff);
eval.attn_mask(&kvm);
eval.epilogue(|| {
    let logits = intrinsics::logits();
    vals.put(intrinsics::value_head());          // [B]
    let (pr, ids) = top_k(softmax(logits), k);
    prio.put(pr); pids.put(ids);                 // children for expansion
});

// Host: parallel MCTS -- virtual loss *is* the run-ahead policy.
feed_and_submit(tree.select(B));                 // batch 0 in flight
loop {
    let leaves = tree.select(B);                 // PUCT + virtual loss (stats trail one batch)
    feed_and_submit(leaves);                     // geometry channels .put(..) -- all puts before
                                                 //   a submit coalesce into one transfer (D1)
    let v = vals.take().await?;                  // batch t-1 lands while t runs
    tree.expand(pids.take().await?, prio.take().await?);   // ws.alloc backs each expanded node's slot
    tree.backprop(v);                            // undo virtual loss here
    ws.free(tree.prune());                       // exact GC, free follows pruning: safe with the next
                                                 //   eval in flight -- pruned nodes are never in the
                                                 //   frontier, and free never renumbers survivors (§5.2)
}
```

- Virtual loss *is* run-ahead made sound: batch t is selected before t−1's
  values land, and a submitted eval is never wrong, only late (§1's
  no-cancellation rule; backprop is the overwrite).
- Reclamation is exact: the host owns the node→slot map, so `alloc` follows
  expansion and `free` follows pruning, concurrently with in-flight evals —
  no mark-sweep, no fence (contrast §6.2). Fork waste is exact too: the host
  knows every freeze offset, so `compact` (§5.2) is an ordinary re-feed at a
  host-chosen moment — no drain needed, geometry is host-owned.
- The non-degenerate ragged case: per-leaf depths differ (`klen`, `pos` and
  the `kvm` rows differ per leaf); multi-token expansion makes `tok_indptr`
  a channel too.

6.5 LoRA — an adapter is data, not an engine object

`W'x = Wx + B(Ax)` at chosen projection sites. PEFT needs no engine adapter
API: the weights are per-instance *data* (D2 — an adapter handle attached to
the pass keys batching on adapter identity; weights in channels keep a
program's identity adapter-independent), and application is a **configuration
sink** (§4) — the same species as `minference_sparse`: one named op, fed
data, configuring the whole forward (D5).

```rust
// rank R and num_layers are trace-known shapes (a different rank = a
// different bucket); the weight *contents* are seeded data: swapping an
// adapter is re-seeding two channels, never re-tracing. The LoRA scale
// alpha/R is folded into b0's seed (per-adapter data, D2).
let lA = Channel::from(a0);                       // [num_layers, R, d]
let lB = Channel::from(b0);                       // [num_layers, d_out, R]

fwd.prologue(|| intrinsics::kernel::lora(lA.read(), lB.read(), SITES));
// a configuration sink (§4) in the pass prologue: the reads are peeks (§1),
// so application adds no edge to the decode chain -- and since the sink takes
// tensors, weights may equally be computed in-graph (a scaled or merged
// adapter) before applying. SITES: trace-known placement (Tensor::constant)
// over the model's site vocabulary (q/k/v/o/up/.., a model intrinsic like
// `vocab`): placement is structure, weights are contents (D2).
```

- **Batching survives mixed adapters.** Instances share the traced program
  (same rank, same placement = same bucket); *which* adapter runs is channel
  contents, so B requests with B different LoRAs batch into one pass — the
  kernel is the fused grouped matmul over the batch's gathered weights.
- **No site enum, no per-layer hook.** How the delta meets the projection
  (split q/k/v vs a fused qkv GEMM) is hidden inside the backend-bound
  intrinsic, exactly as `intrinsics::activation_type` hides dtype (D1: the
  contract is "the delta lands at the declared sites"; the fusion is the
  runtime's). The site vocabulary is model-level — an adapter checkpoint is
  model-bound anyway. The cost, named honestly: the delta *algebra* is not
  programmable — a new PEFT algebra is a new named op (IA³, DoRA), D5's
  price. A truly novel mid-model computation would need per-site attachment
  stages; recorded as direction, like `Register` (§1).
- **The rest of PEFT needs no ops at all.** Prefix tuning is seeded pages
  shared CoW across instances (§5.2's `slice`/`append`); prompt tuning is
  seed embeddings fed like tokens (§5.1).
- **Weights are `read()`, never `take`n; an update rides a step.** A second
  traced variant of the same pass adds a prologue program (§5.3)
  that `take`s the host-fed new pair and `take`→`put`s `lA` and `lB`; the host
  submits that variant for the step after an update lands. The per-instance
  atomic commit (§1) publishes the pair RCU-style — this step and all later
  ones read the new pair, in-flight earlier steps the old, and no reader
  ever tears new `lA` against old `lB`. (Host-side direct `take`→`put`s would
  tear — between the two channels' updates both cells are full and
  committable — and their empty windows would spuriously retry innocent
  passes.) An ES perturbation is not even an update: the per-eval seed is
  one more data arg to the kernel, which regenerates the noise in-graph
  (`rng`-style, §3) — the base weights stay read-only.
- Live weights are the nearest miss yet for the reserved `Register` (§1):
  latest-value reads, skipped versions harmless, writer and reader on
  different clocks. The update-variant keeps it channel-shaped because
  within one pipeline the *step* is the clock; cross-pipeline weight feed (a
  trainer pipeline pushing on its own clock) is where the reservation would
  finally trigger.

> **Design note.** `inference.wit`'s forward-pass `adapter(...)` attachment is
> removed — an engine-object adapter on the pass is exactly the identity leak
> D2 forbids. The `adapter` *resource* (create/load/save) survives for now as
> zo's weight-store handle; direction: adapter weights become tensors the
> inferlet feeds through channels (zo's `adapter-seed` dissolves into the
> kernel's seed argument), and the resource retires too.


7. Lowering

Nothing in this section is program surface: §§1–5 fix every observation, and
these notes record the mechanisms they lower to (D1) — how channel effects
commit (§7.1), when the engine fires a batch (§7.2), how programs become
device code (§7.3).

7.1 Epoch rings — commit is an index bump

A channel of capacity N lowers to a ring of N+1 cells plus its bits; capacity
1 is a double buffer. Within a pass, `take`/`read` resolve against the
committed cell, `put`s land in the pending cell, and publishing is an index
bump — per channel, at pass end, no data moves (D3); capacity > 1 keeps head
and tail, a net `put` bumps head, a net `take` bumps tail. The commit
predicate behind that bump is evaluated in two places:

- **At fire time, structurally.** Per channel, the program's *first* op in
  program order names the required bit: `take`/`read` need full (§1's
  readiness rule), a leading `put` needs empty — the empty check *is*
  back-pressure, and §3's lossless `out` rests on it. The host settles most
  of the predicate without reading device bits: a cell fed (or drained) by a
  pass in flight ahead of this one counts as satisfying its bit, because
  stream order retires the producer first.
- **On device, per stage.** What fire time cannot settle is the genuinely
  late edges — host-fed cells (§3's `mask`), host-drained ones (`out`),
  cross-pipeline feeds. Each stage checks its inputs as it starts (§1); the
  pass's commit flag is the AND of its stages' checks; the bump is
  predicated on the flag. A failed instance dummy-runs, publishes nothing,
  and resubmits — membership in a later fire (§7.2).

So the transaction needs no undo machinery of its own: predication in front,
one predicated index increment per channel behind — O(channels) work in the
tail of the stage's last glue kernel (§7.3). Two lowerings then drop ring
cost where structure allows (D3):

- **In-place, no undo.** A linear `take`→`put` in a stage that no fallible
  stage follows mutates the committed cell directly. The epilogue is the
  last stage, so the ping-pong idiom — `tok`, `rng`, `len` — always
  qualifies, predicated on the accumulated flag at its entry.
- **In-place, undo at mutation granularity.** Ahead of a fallible stage, the
  same idiom saves only what it mutates — §1's cost guarantee; the undo log
  *is* the ring, degenerated from whole cells to the touched rows.

Host-visible channels always keep the full ring: the host peeks the
committed cell wait-free — always a consistent snapshot — while the pending
cell fills next door.

**Within a pass, a channel is a register.** The readiness checks pin each
bit before first use, and per-channel program order (§1) makes the in-pass
trajectory trace-known, so blocking semantics exist only at pass boundaries.
Corollaries: a second `put` overwrites the pending cell — last write wins; a
`take` after an in-pass `put` reads the pending value — pure dataflow, no
extra cell; the committed bit is the last op's direction. (The trajectory is
trace-known, so double `put`s could be rejected statically — but predicated
choice lowers to exactly the last-wins shape, so the register rule is the
contract.)

7.2 The fire rule — quorum on the device's clock

Stage-tuple identity (§5.3) decides who *may* share a launch; the fire rule
decides *when* — and like the readiness deadline it is an engine policy,
never a per-pass knob (§1). One rule, three clauses:

- **Quorum.** The moment every counted pipeline's next pass is structurally
  ready (§7.1's fire-time predicate; late host edges deliberately do not
  gate — they park their stage, never the batch), enqueue the dense batch to
  the driver, one deep, behind the batch in flight. In steady state quorum
  completes mid-flight and the device takes the next batch in stream order:
  bubble zero, and no completion estimate anywhere — enqueue-ahead is what a
  measured lead time would only approximate.
- **Idle escape.** If the device goes idle with the queue empty, fire the
  ready subset immediately. Missing instances are absent — no holes, no
  padding rows — and rejoin a later fire; §1's resubmission is exactly this
  membership.
- **Cold hold.** Nothing in flight at all: hold sub-millisecond for
  arrivals, then fire partial. A step-scale timeout survives only as a hang
  backstop.

The quorum's denominator counts pipelines that *can* be ready this round:
one blocked on host work at a value-dependent boundary — a tool call, §6.2's
drained compact — is absent, not awaited. That keeps both regimes
first-class: decode fleets live in the quorum clause, agentic fleets live in
the escape, and the escape is exactly as dense as its ready set.

Readiness absorbs what is elsewhere scheduler machinery: a late-binding
dirty flag is a full bit, a producer link is a loop-carried channel,
write-transaction ownership is the epoch ring, laggard bookkeeping is
membership recomputed each fire. What remains of scheduling is this rule
plus capacity limits — splitting an over-budget ready set before the fire
decision, chunked prefill included (§5.1) — and capacity accounting itself
shrinks to rows and pages per request; the rest (readout counts, mask bytes,
sampler rows) is trace-known per program, priced once at registration.

> **Design note.** Today's scheduler
> ([`scheduler.rs`](../runtime/src/inference/scheduler.rs)) is
> response-synchronous — a fire cannot start until the previous driver call
> returns — and the run-ahead handover
> ([`run-ahead-submission-scheduler-handover.md`](../docs/run-ahead-submission-scheduler-handover.md))
> reaches this section's architecture bottom-up: its `tensor.write` + dirty
> flag, producer links, per-forward write transactions and parity barrier
> are the channel runtime's special cases, and its firing formula —
> `max(completion − lead_time, collection)` — is superseded by depth-1
> enqueue, whose bubble the EWMA existed to hide. Recorded as direction; the
> scheduler is unchanged.

7.3 Compilation — bind the trunk, fuse the glue

The trunk — attention, the GEMMs — is never expressed in PTIR: §5's
descriptor binds it, the backend provides it. What compiles is the glue: the
stage programs, row-parallel over `[B, ·]`, every shape trace-known (§5.1),
drawn from the closed first-party core plus named intrinsics — D5 is what
keeps the set JIT-able (the appendix lists the core). Three tiers, each a
complete backend:

- **Tier 0 — interpret.** One prebuilt kernel per core op; walk the trace
  launch by launch. Correct on day one per backend: the op set is the entire
  porting surface — a few dozen row-parallel kernels, on Metal one MSL
  library.
- **Tier 1 — fuse.** One kernel per stage program: a lane per
  CTA/threadgroup, reductions row-local, geometry scatters a few words per
  lane (§6.2's epilogue is element-wise math plus two flat scatters). CUDA
  compiles through NVRTC — the Sampling-IR path
  ([`sampling_ir/`](../driver/cuda/src/sampling_ir/),
  [`sampling-ir.md`](../docs/sampling-ir.md)) generalized from one epilogue to every
  stage; Metal through `newLibraryWithSource`, the same runtime-compile
  shape, pipeline states cached under the same trace hash. `top_k`, `matmul`
  and their kin link as library kernels: the JIT emits glue, never
  algorithms.
- **Tier 2 — erase launches.** CUDA graphs keyed by the stage tuple's
  program-set hash alongside today's shape key
  ([`forward_graph.hpp`](../driver/cuda/src/executor/forward_graph.hpp)'s
  `{num_requests, num_tokens, variant}`). Metal has no stream capture; the
  analogue is indirect-command-buffer replay, and it matters less: after
  tier 1 a stage is the trunk's kernels, a glue kernel or two, and §7.1's
  commit bump.

Channels lower to addresses. Shapes are trace-known, so registration
preallocates every instance's cells; a channel op inside a kernel is a
load/store at instance base + channel offset + ring index; the bits are a
bitfield word — the same word the per-stage readiness check reads (§7.1).
Traced-once sets the economics: compilation (tens of milliseconds) runs once
per program per backend at registration, amortized over every instance and
step — the hash→kernel cache discipline Sampling-IR uses today. And batching
never recompiles: same program → rows concatenate; different programs
sharing a forward → one trunk launch over the union, per-program glue
partitioned over the row space (§5.3), which is how the executor already
partitions sampling rows.


Appendix — the first-party op set

D5's claim — the core stays closed and fusable — is checkable only against
the list. Every program op in this document comes from the families below,
closed under their obvious completions; everything else is an
`intrinsics::*` value or an `intrinsics::kernel::*` name (§4). Every op is
value → value over trace-known shapes (§5.1); none touches memory (§5.2).
Host arithmetic on trace-known values (`div_ceil`) is wasm, not IR.

| family        | ops                                                  | notes |
|---------------|------------------------------------------------------|-------|
| map           | `add sub mul div rem neg exp log cast`               | element-wise; broadcasting by shape |
| compare/logic | `eq ne lt le gt ge and or not`                       | `bool` results — packed on the wire (D1) |
| choice        | `select`                                             | the data-dependent branch (§2) |
| shape         | `reshape broadcast transpose`                        | metadata only, no data movement |
| index         | `iota gather scatter_set`                            | `scatter_set` duplicates resolve in index order, last wins — load-bearing in §6.2 |
| reduce/scan   | `reduce_sum reduce_max reduce_argmax cumsum cumprod` | row-local (§7.3) |
| normalize     | `softmax log_softmax l2norm`                         | composed reduce + map; fuse |
| order         | `top_k pivot_threshold rank_le`                      | `rank_le(k)` is the rank predicate `pivot_threshold` cuts at; `top_k` links as a library kernel (§7.3) |
| linear        | `matmul`                                             | the core's one GEMM (§6.3); a library kernel (§7.3) |
| sampling      | `gumbel mask_apply`                                  | composed over `rng` state + map (§3, §4); replay-deterministic |
