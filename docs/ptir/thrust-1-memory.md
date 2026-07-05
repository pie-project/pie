<!-- Source of truth: wiki page `tensor-ir-plan.md` (slug tensor-ir-log). This folder is the split, on-disk copy for implementation teams. -->

# Thrust 1 — Working Set & Attention Memory

**Status:** Ready for execution. Independent of thrusts 2 and 3 (see §6).
**Audience:** Runtime (working set) + driver/CUDA (attention kernels) engineers.
**Realizes:** [`overview.md`](overview.md) §5.1 (the descriptor), §5.2
(ownership and reclamation), and the attention-adjacent second-party kernels
(§4, §6.1–§6.2). Contract provided to the other thrusts: **C1 — geometry is
data** ([`masterplan.md`](masterplan.md) §3).

---

## 1. Goal

Memory belongs to the working set; programs (and, until they exist, host code)
hold only **geometry** — ordinary index data. End state:

- A slot id is a **stable handle** to a refcounted, CoW-shared page object; it
  never renumbers. `free` is non-compacting; `alloc` grants fresh-or-recycled
  ids and is in-flight safe.
- The attention descriptor is one call with two column families (overview
  §5.1): token-indexed (`embed`, `positions`, `w_slot`/`w_off` — consumed) and
  page-indexed (`pages` + `kv_len` totals — peeked). The engine carries **no
  per-page lengths**: sub-page validity (frozen forks) is inferlet/program
  state, lowered to a full-KV attention mask over the existing custom-mask
  path — **no attention-kernel changes anywhere in this thrust** (W11).
- Divergence is a **freeze**: siblings stop advancing their view of a shared
  page; one *designated child* may keep writing past every frozen prefix
  (overview §5.2 invariant). There is **no copy op** anywhere.
- Reclamation is two-tier and host-driven: whole-page `free` (exact, or
  concurrent mark-sweep over a peeked snapshot; RCU-style grace period before
  arena recycling) plus explicit token-space `compact` (the `gather_tokens`
  kernel + an old→new remap the host re-feeds).
- The attention path supports: full-KV validity masks for decode-shaped lanes
  (the fork-sharing enabler), page key envelopes (`envelope_dot`), and dense
  token repacking (`gather_tokens`). In-kernel page skipping
  (`attn_page_mask`) is direction-only under W11 (see M3).
- Descriptor index arrays may live in **device memory** the host never
  materialized (C1's final form), with launch-immediate folding for trace-known
  scalars.

Standalone value: fork-heavy workloads (beam, MCTS, agent trees) get stable
indices, honest reclamation, and prefix-sharing attention on **today's** WIT
surface — no dependency on PTIR programs or the new scheduler.

## 2. Current state

| area | today | anchor |
|---|---|---|
| WIT | `free(indices)` **densely compacts** at call time; `alloc(n)` returns a contiguous `page-range`; `generation` bumps on *every* structural mutation; `slice`/`append`/`fork` are lazy CoW and already exist | `interface/inferlet/core/wit/working-set.wit` |
| runtime | dense ordered array of page slots; compaction-on-free; set-level revert log `pending: Vec<(usize, Option<ObjectId>)>` with `commit_writes`/`abort_writes`; page objects are refcounted CoW; full-page CAS sealing after forward writes | `runtime/src/working_set/kv.rs` (:202, :544, :550) |
| forward txn | per-forward transaction guard exists at the API layer (abort-on-drop, `finalize_forward_txn`) | `runtime/src/api/inference.rs` (:1076–1101) |
| attention | FlashInfer paged attention: page table + per-lane totals, only each lane's **last** page may be partial; no mid-chain partial pages, no per-query page skipping, no attention-score export | `driver/cuda/src/` (attention workspace, kernels) |
| custom masks | BRLE-encoded attention masks exist for the speculative-window verification path; full-KV-axis coverage unverified (M2b task 1) | `driver/cuda/src/brle.{cpp,hpp}`, `driver/cuda/src/spec_expansion.cpp`; scheduler mask-byte accounting (`packed_mask_bytes`) |
| geometry ABI | scalar geometry args (`inp-start`, `inp-len`, `valid-tokens`, `output-start`, `output-len`, `offset`) | `interface/inferlet/core/wit/inference.wit`; overview §6.1 design note |
| graphs | `ForwardGraphKey{num_requests, num_tokens, variant}`; page tables are late-bound buffers | `driver/cuda/src/executor/forward_graph.hpp` (:46) |

The overview's §5.2 design note is this thrust's WIT migration in one
paragraph; treat it as normative.

## 3. Locked design decisions

| # | decision | source |
|---|---|---|
| W1 | A slot id is a stable handle to a refcounted page object; survivors keep their ids across `free` | overview §5.2 |
| W2 | Host owns the shape (`alloc`/`free` host-only); programs own geometry, never contents; slot ids are **inherit-only** (constants, host-fed values, gathers of existing index tensors) — this is what makes reachability tracking sound | overview §5.2 |
| W3 | Divergence = freeze. Any number of lanes read a slot up to its declared valid length; past it, a slot written this step belongs to exactly one lane (the designated child). Only dying branches strand tokens | overview §5.2, §6.2 |
| W4 | `free` is non-compacting (tombstone + free list); precondition is unreachability from every in-flight pass; the arena recycles a freed page only after every pass in flight at free-time has retired (grace period) | overview §5.2 |
| W5 | `compact` is explicit, token-space, never a side effect of `free`; `gather_tokens` is engine machinery (D5), not a program op; it returns an old→new remap the host re-feeds through geometry | overview §5.2 |
| W6 | The engine's only length is `kv_len` physical totals (frozen pages presented full). Per-page validity lives in the program's lens channel; the engine consumes its two in-program derivatives — the totals and the KV validity mask | overview §5.1, §5.2 |
| W7 | `attn_working_set` is one call (no builder), with sugar arities `(&ws, &len)` and `(&ws, &cursor, P_MAX)`; a trailing trace-known capacity caps the slot axis | overview §5.1, §6.1 |
| W8 | `generation` bumps only on `reorder`/`compact` | overview §5.2 design note |
| W9 | `attn_page_mask` is second-party (`intrinsics::kernel::*`): prologue = every layer, attn-proj = that layer alone | overview §4, §5.3 |
| W10 | Per-page shapes bind to the trace-known cap (`P_MAX`), never the live size; there is no `num_pages` intrinsic | overview §6.1 |
| W11 | **No new or modified attention kernels.** Fork validity lowers to the existing custom-mask machinery; masked programs run the masked-attention variant, dense programs keep stock kernels (batch-by-program isolation) | project constraint |

## 4. Boundaries (what this thrust does NOT do)

- **Write-transaction lifecycle** (keying `pending` by forward txn, per-forward
  commit/abort) is thrust 2 phase S4. This thrust changes allocation and
  reclamation in the same files; C4's ownership split applies — coordinate PR
  trains.
- **Channel-fed descriptors** are thrust 3. This thrust proves device-resident
  geometry with plain host-written device buffers (C1 interim form).
- **Scheduling** is untouched; every phase here works under the current
  response-synchronous fire loop.

## 5. Phases

### Phase M1 — Slot-id semantics

Make the overview §5.2 design note true.

Tasks:

1. `kv.rs`: replace the dense array + compaction-on-free with a slot table
   (`Vec<Option<PageSlot>>` + free list). `size()` returns the **live** count.
   `alloc(n)` pops recycled ids before growing; grants never disturb existing
   ids. `free(ids)` tombstones; double-free and out-of-range return errors
   (never trap). `slice`/`append`/`fork` operate over ids unchanged in
   semantics.
2. `generation` bumps only on `reorder`/`compact` (W8). Audit SDK uses of
   `generation` (rust, python, js, bakery) — the stale-index defense moves to
   "compact/reorder only", which is its honest coverage anyway (device-resident
   indices were never protected; overview §5.2 design note).
3. WIT: add `alloc-slots(n: u32) -> result<list<u32>, error>` and
   `free-slots(ids: list<u32>)` alongside the old pair (append-only). Keep
   `alloc`/`free` as shims over the new semantics during migration (old `free`
   = free-slots without compaction — callers that relied on index shifting are
   found by test, expected: none outside SDK internals). Removal is a later
   deprecation PR (C4).
4. Reachability capture: at submit, record the slot set each in-flight pass
   references (from its descriptor). This is the grace-period input for M4 and
   the `free`-precondition check in debug builds.
5. Regenerate SDKs; port `bakery` and the test inferlets.

Exit criteria:

- Soak test (mock driver): randomized alloc/free/fork/append/slice with passes
  continuously in flight; no id ever renumbers; commit/abort behavior
  unchanged; debug-mode free-while-referenced assertion fires on injected bugs.
- All existing inferlet e2e tests pass with the flag on.

### Phase M2 — The length column

**M2a — dense totals (`kv_len [B]`).** Map overview §5.1's `kv_len` onto the
existing per-lane valid-token path end to end (WIT → schema → executor), as a
named column rather than scattered scalars. Low risk; mostly plumbing and
tests.

**M2b — full-KV validity masks (no attention-kernel work, W11).** The
fork-sharing enabler. Frozen mid-chain pages are presented to the kernel as
full (they count fully in `kv_len`), and the invalid residual slots are
excluded by an attention mask the inferlet feeds — the same custom-mask
machinery tree-speculation verification uses today. Semantically exact:
masked positions get `-inf` before softmax, which equals prefix truncation.

Tasks:

1. **Audit the existing custom-mask path.** Today it masks the in-pass
   speculative window; extend the plumbing (WIT `attention-mask` → schema →
   executor) so a mask can cover the **full KV axis** for decode-shaped
   lanes. This is plumbing and shape bookkeeping, not kernel work — the
   underlying masked-attention kernels take `[qo, kv]` masks already.
2. **Execution variant wiring.** Masked decode runs the masked-attention
   path (the prefill-with-mask kernel at `qo = 1`), which excludes it from
   the pure-decode graph variant. Batch-by-program (overview §5.1) confines
   the cost to fork-bearing programs: dense programs keep stock kernels,
   stock graph variants, stock performance.
3. **Explicit positions.** With masked holes, physical span ≠ logical
   length, so fork-bearing programs feed `positions` explicitly (overview
   §6.2's `pos` channel). Verify that path end to end.
4. **Mask sizing and admission.** Decode-shaped masks are `[B, KV_MAX]` bits
   — small, packed on the wire. The scheduler's existing mask-byte capacity
   accounting covers admission; masked *prefill* (chunk × KV bits) is
   admission-limited by the same accounting — document the practical bound.
5. **Plan-on-upper-bound.** Attention plan/workspace sizing keys on the
   trace-known cap (`P_MAX`, W10), not live sizes, so geometry changes never
   re-plan.

Exit criteria:

- Randomized fork-geometry attention (mask lowering) matches a PyTorch
  reference within fp32 accumulation tolerance, including: prompt-page
  aliasing, mid-chain frozen pages, designated-child tails, within-page
  forks at arbitrary offsets (overview §6.4).
- Dense-column programs show no regression (< 2%): stock kernels, stock
  graph variants.
- A beam-shaped microbench (B=8, fork every step) reports the masked-variant
  overhead vs. dense decode (variant switch + residual reads); the number
  goes in this doc.

> **Recorded as direction (perf escape only, gated on the microbench):**
> chunk-at-frozen-boundary + online-softmax state merge (FlashInfer
> cascade), or a per-page-len kernel generalization. Not scheduled.

### Phase M3 — Attention-adjacent kernels

1. **`attn_page_mask` — deferred (W11).** In-kernel page skipping is
   attention-kernel work, so it ships only if the constraint lifts. Its
   availability is bind-time (overview §4): a program naming it fails to
   bind on this backend and falls back to a different traced program, so
   nothing else in the plan blocks on it. The no-kernel fallback for
   pass-granularity selection is page-table selection (the page list is
   index data); per-layer selection has no no-kernel form — recorded as
   direction.
2. **Envelopes + `envelope_dot`:** maintain per-page key min/max at KV-append
   time (driver-side epilog of the KV write; per-model opt-in so only
   Quest-running programs pay), and the `[P_MAX]` importance-score kernel.
   Slots beyond the live size are validity-coded (`-inf`), overview §6.1.
3. **`gather_tokens`:** pack live token runs into fresh slots per a host-given
   run list; the driver-side op behind `compact` (W5). Target: streaming-copy
   bandwidth.

Exit criteria: `envelope_dot` scores match a reference on synthetic pages;
envelope maintenance overhead measured and < 2% on decode; `gather_tokens`
≥ 80% of `cudaMemcpy` bandwidth on page-sized runs. (Both are standalone
reduction/copy kernels, not attention-kernel modifications — confirm they
fall outside the W11 constraint before starting; open question 5. The
no-kernel fallback for `compact` is batched D2D copies over existing copy
ops.)

### Phase M4 — `compact()` and GC helpers

1. `ws.compact(live_runs) -> remap`: host API; issues `gather_tokens`, returns
   old→new; copies ride off the decode path; old pages return to the arena
   after the grace period. Quiescent-point contract documented: when geometry
   lives on device the **caller** drains its in-flight step first (overview
   §6.2); when host-owned it is an ordinary re-feed (overview §6.4).
2. Grace period: recycle a freed page only after every pass in flight at
   free-time retired (uses M1's reachability capture). Property: a host bug
   that frees a still-referenced slot reads its own stale data, never another
   working set's (overview §5.2).
3. Mark-sweep helper (host library, not engine API — overview: "mark is host
   code, sweep is `free`"): `mark(snapshot ∪ ids-granted-since) -> dead`.

Exit criteria: §6.2-shaped beam simulation on mock — host waste model (parent
log replay) matches actual strandings exactly; fuzzed free/compact under
in-flight passes never yields cross-working-set reads (grace period test);
`generation` bumps exactly on compact/reorder.

### Phase M5 — Geometry as data (C1 final form)

1. Driver: descriptor index families readable from device buffers the host
   never wrote this step (device-resident offsets); the executor already
   late-binds page tables into graphs — extend the same pattern to
   `w_slot`/`w_off` and the length column.
2. Launch-immediate folding: trace-known scalars (constant indptr, caps) fold
   into launch parameters instead of buffers (overview §6.1 design note,
   item 1).
3. WIT: begin the migration from scalar geometry args to index tensors behind
   the thrust flag; overview §6.1's design note is updated in the same PR (C4).

Exit criteria: a forward whose `pages`/`kv_len`/mask/`w_slot`/`w_off` contents
were produced by a *previous pass's kernel* into a device buffer (host never
read them) matches the host-fed run bit for bit. This is the handshake test
thrust 3 will bind channels against.

## 6. Interfaces

**Provides:** C1 to thrusts 2 and 3 (schema fields + device-resident
descriptor reads + trace-known-cap sizing); the full-KV mask path (M2b) that
overview §6.2/§6.4 programs lower onto; the M3 standalone kernels
(`envelope_dot`, `gather_tokens`) to thrust 3's second-party registry;
`compact`/grace-period semantics to everyone.

**Consumes:** nothing from thrusts 2 or 3. Every phase is testable under the
current scheduler with host-driven geometry. (Thrust 2's S4 rework of
`pending` lands in the same files; sequence the PR trains, C4.)

## 7. Risks

| risk | mitigation |
|---|---|
| Masked-attention variant too slow for fork-heavy fleets (prefill-path kernel at `qo = 1` + residual reads) | Measured by the M2b microbench; the kernel options stay recorded-as-direction and graduate only on that evidence; batch-by-program confines the cost meanwhile |
| The existing custom-mask path is window-scoped deeper than the ABI (executor assumptions) | M2b task 1 audit runs first; the underlying masked kernels take full `[qo, kv]` masks, so exposure is plumbing depth, not kernel capability |
| WIT migration breaks external inferlets | Append-only + shims (M1.3); old entry points removed only at the C4 deprecation gate |
| Grace-period leaks when a pass never retires | Retirement is bounded by the engine deadline + poison (overview §1); leaked pages surface in the arena accounting probe |
| Envelope maintenance taxes every append | Per-model/per-program opt-in (M3.2); programs that never name `envelope_dot` pay zero |
| `generation` semantics change surprises SDK callers | Audit in M1.2; the counter's new meaning is strictly weaker only for cases it never actually protected |

## 8. Open questions

1. `alloc-slots` return: `list<u32>` only, or keep a contiguous-range fast path
   for the common grow-by-N (prefill) case?
2. Mask wire encoding for full-KV validity masks: BRLE (today's spec-mask
   format) vs dense packed bits — align with thrust 2 (RA §11 Q9) and thrust
   3's packed `bool` channel wire (D1); one answer for all three.
3. Does `positions`' "defaults to append order" survive explicit
   `w_slot`/`w_off`, or should the default derive from the write pair?
   (Fork-bearing programs now always feed positions explicitly — overview
   §6.2; the question remains for the dense path only.)
4. Debug-mode single-writer validation for device-computed `w_slot` (overview
   §5.2 invariant): worth a checked kernel behind a flag?
5. Are the standalone non-attention kernels (`envelope_dot`, `gather_tokens`)
   inside the W11 constraint or out? If banned too: `compact` falls back to
   batched D2D copies over existing copy ops; envelopes and Quest defer
   entirely.

## 9. Code anchors

| area | files |
|---|---|
| WIT working set | `interface/inferlet/core/wit/working-set.wit` |
| WIT forward | `interface/inferlet/core/wit/inference.wit` |
| Runtime working set | `runtime/src/working_set/kv.rs`, `runtime/src/working_set/rs.rs` |
| Forward txn guard | `runtime/src/api/inference.rs` |
| Driver ABI | `interface/driver/src/schema.rs`, `interface/driver/include/pie_driver_abi.h` |
| CUDA attention | `driver/cuda/src/attention_workspace.{cpp,hpp}`, `driver/cuda/src/kernels/`, `driver/cuda/src/kv_cache.{cpp,hpp}` |
| Graph keying | `driver/cuda/src/executor/forward_graph.hpp` |
| Background | [`../docs/working-set-refactor-handover.md`](../docs/working-set-refactor-handover.md), [`../docs/attention-kv-ir.md`](../docs/attention-kv-ir.md) |
