<!-- Source of truth: wiki page `tensor-ir-plan.md` (slug tensor-ir-log). This folder is the split, on-disk copy for implementation teams. -->

# PTIR Master Plan

**Status:** Execution handoff. The design is frozen in [`overview.md`](overview.md);
this folder plans its implementation.
**Audience:** Implementation teams (runtime/Rust, driver/CUDA, SDK/WIT) and their
tech leads.
**Scope:** One master plan and three independently executable thrusts. Read this
file first; each thrust document then stands alone for its team.

---

## 1. North star

PTIR ([`overview.md`](overview.md)) is the target programming model at the
inferlet/engine boundary:

- Inferlets attach **tensor programs** — closures traced once — to the forward
  passes they submit. The engine batches instances by traced program, **stage by
  stage**; instances sharing a forward trace co-batch even where epilogues
  differ.
- **Channels** (GPU-resident cells with full/empty bits) are the *only* stateful
  construct. Everything per-step is channel contents; everything trace-known is
  constants and shapes. Ordering comes from the bits, never from host round
  trips.
- The **working set** owns memory; programs own **geometry, never contents**
  (index tensors: `pages`, length column, `w_slot`/`w_off`). Divergence is a
  freeze, reclamation is two-tier (`free` + token-space `compact`), and there is
  no copy op.
- The engine fires **dense batches on a quorum rule** with a depth-1 driver
  queue: zero bubble in steady state, no completion estimation anywhere.
- Programs compile through **three tiers**: op-per-kernel interpreter, JIT'd
  per-stage fusion (the Sampling-IR path generalized), and launch-erased graph
  replay.

**North-star acceptance:** overview §6.1 (MTP speculation + grammar constraint +
Quest attention, one pass, three programs) and §6.2 (beam search with freeze /
designated-child / compact) run end to end on the CUDA backend under the quorum
scheduler, meeting the performance gates in §5 below.

## 2. Why three thrusts, and why they are independent

The model factors on its own seams — **space** (what memory is), **time** (when
work runs), **compute** (what a program is) — and each seam already has a
shipping carrier in the repo today:

| seam | carrier today | thrust upgrades it to |
|---|---|---|
| space | `working-set.wit` + `runtime/src/working_set/` + FlashInfer paged attention | stable slot ids, full-KV validity masks (no attention-kernel changes), two-tier reclamation, standalone kernels |
| time | `runtime/src/inference/scheduler.rs` (response-synchronous fire loop) | non-blocking execute, run-ahead, quorum fire rule, depth-1 enqueue |
| compute | Sampling-IR JIT (`driver/cuda/src/sampling_ir/`) + sampler ABI | traced stage programs, channels + epoch rings, tiers 0/1/2 |

Because each thrust upgrades one carrier while the other two keep functioning
on today's contracts, the thrusts are independently executable and each lands
standalone value **before** convergence:

- **Thrust 1 — Working set & attention memory**
  ([`thrust-1-memory.md`](thrust-1-memory.md)). Realizes overview §5.1–§5.2 and
  the attention-adjacent kernels. Standalone value: stable slot ids, honest
  reclamation, and fork-friendly attention for today's WIT users — no PTIR
  dependency.
- **Thrust 2 — Pipelined execution & the fire rule**
  ([`thrust-2-scheduler.md`](thrust-2-scheduler.md)). Realizes overview §3 (host
  side) and §7.2; incorporates and partially supersedes
  [`run-ahead-submission-scheduler-handover.md`](../docs/run-ahead-submission-scheduler-handover.md).
  Standalone value: run-ahead decode and zero-bubble batching on the existing
  forward contract.
- **Thrust 3 — PTIR programs: trace, channels, compilation**
  ([`thrust-3-programs.md`](thrust-3-programs.md)). Realizes overview §1–§4,
  §5.3, §7.1, §7.3 and the appendix. Standalone value: generalizes the shipped
  Sampling-IR path — epilogue parity first, then prologue/sinks/taps.

## 3. Convergence contracts

Independence is real only if the meeting points are pinned **now**. Four
contracts; every cross-thrust dependency in the thrust docs cites one of them.

### C1 — Geometry is data (thrust 1 → thrusts 2, 3)

The forward descriptor's index families (`pages`, the length column,
`w_slot`/`w_off`, `readout`) are carried as arrays whose *contents* the driver
reads at execution time, and which may live in **device memory** the host never
materialized. Interim form: host-written buffers (thrust 1 tests this alone).
Final form: PTIR channel cells (thrust 3 binds them). Rules: new descriptor
fields are append-only in `interface/driver/src/schema.rs`; shapes bind to
trace-known caps, never live sizes (overview §5.1); scalar geometry args in
`inference.wit` are deprecated only after the tensor path is proven (overview
§6.1 design note).

### C2 — Readiness is a word the driver waits on (thrust 2 → thrust 3)

The driver blocks (or predicates) at declared cut points on a word in
device/pinned memory with acquire semantics — `cuStreamWaitValue32` on CUDA, a
poll on the mock path. Interim form: per-binding **dirty flags** written by
`tensor.write` (handover §2.6; thrust 2 builds the wait mechanism and the
direct host→driver channel). Final form: **channel full/empty bit words**
updated by epoch-ring commit (overview §7.1; thrust 3 swaps the producer of the
word, not the wait mechanism). Rule: the executor-side wait/predicate code is
written once, keyed by an opaque `(word address, expected value)` binding — it
must not know whether a flag or a channel bit feeds it.

### C3 — Program identity is a hash (thrust 3 → thrust 2)

A pass's batching identity is the tuple of its stage traces (overview §5.3),
carried as a **program-set hash** next to the shape key. Interim form: today's
implicit identity (sampler SoA / `ForwardGraphKey{num_requests, num_tokens,
variant}` in `driver/cuda/src/executor/forward_graph.hpp`). Final form: the
stage-tuple hash as a first-class batch/graph key component. Rule: thrust 2
keys quorum membership and batch formation on an opaque `identity: u64` it
never interprets; thrust 3 defines its computation.

### C4 — File ownership and ABI discipline

- `runtime/src/working_set/*` — owned by thrust 1. Exception: the per-forward
  write-transaction lifecycle (`prepare`/`commit_writes`/`abort_writes` and
  their keying by forward txn) is owned by thrust 2 (its phase S4); thrust 1
  owns allocation/reclamation semantics in the same files. PRs touching the
  other team's functions require their review.
- `runtime/src/inference/*`, `runtime/src/driver/*` — owned by thrust 2.
- `driver/cuda/src/sampling_ir/*` and the new PTIR program dirs — owned by
  thrust 3. Attention/KV kernels (`driver/cuda/src/kernels`, FlashInfer
  integration) — owned by thrust 1.
- `interface/driver/src/schema.rs` and the WIT (`interface/inferlet/`) are
  **append-only**: new fields default-empty, old paths keep working, removal is
  a separate deprecation PR after all three thrusts sign off. One weekly ABI
  sync across the three teams; every schema PR names the contract (C1/C2/C3) it
  serves.

## 4. Shared engineering rules

- **Mock first.** Every phase proves semantics on the in-process/mock driver
  path (`runtime/src/driver/inproc.rs`) before CUDA. The thrust-3 reference
  interpreter (host-side, tier 0) is the golden model for program semantics.
- **Feature flags.** Each thrust lands behind its own flag
  (`ws-slot-ids`, `run-ahead`, `ptir`); default-off until its M1 exit, then
  default-on with the legacy path kept one release.
- **No-regression gate.** Baseline single-stream greedy decode and the existing
  batch throughput benchmarks must not regress while a flag is off, and must
  stay within 2% while it is on (unless the phase explicitly buys latency).
- **Design notes stay honest.** When a thrust changes a WIT or ABI surface that
  `overview.md` marks with a "Recorded as direction" design note, the same PR
  updates the note.

## 5. Milestones

**M0 — plan accepted.** Teams assigned, flags created, ABI sync scheduled.

**M1 — standalone exits** (each thrust independently demonstrable):

| thrust | demo | gate |
|---|---|---|
| 1 | slot-id soak: random alloc/free/fork/append with passes in flight (mock); masked-attention fork lowering matches a reference on randomized fork geometries | no id ever invalidated; mask lowering exact-match; dense-path perf within 2% |
| 2 | 8 concurrent greedy pipelines, run-ahead depth 1, quorum scheduler | inter-batch bubble p50 < 100 µs; single-stream tokens/s no regression; laggard tests green |
| 3 | grammar-constrained decode via a PTIR epilogue program (tier 0, then tier 1) | token-for-token parity with the Sampling-IR path; tier-1 tokens/s within 5% of it |

**M2 — pairwise integrations:**

- **3 + 2 (C2):** channel bits replace dirty flags as the sampler's late-input
  readiness; overview §3's `mask` channel parks only the sample while the
  forward overlaps.
- **3 + 1 (C1):** overview §6.1's `fwd.attn_working_set(&ws, &cursor, P_MAX)` —
  descriptor contents from a device channel the host never reads.
- **2 + 1:** run-ahead decode with in-flight-safe `alloc` top-up (overview
  §6.1's headroom loop).

**M3 — north star:**

- Overview §6.1 end to end on CUDA: MTP + grammar, quorum scheduler, tier-1
  fused epilogue. (Quest's per-layer page mask is optional — gated on backend
  `attn_page_mask` availability, overview §4's bind-time rule; direction-only
  under the no-attention-kernel constraint.)
- Overview §6.2 beam search: freeze/designated-child geometry, tier-1 epilogue,
  mark-sweep `free`, one token-space `compact` under a waste threshold.
- Gates: §6.1 accepted-tokens/s ≥ the current speculative path on the same
  model; beam step time within 10% of a hand-rolled baseline; bubble p50
  < 100 µs sustained; dummy-run (readiness-miss) rate < 1% on the steady-state
  decode fleet.

## 6. Cross-thrust risk register

| risk | owner | mitigation |
|---|---|---|
| Fork validity lowers to the masked-attention variant (prefill-path kernel at `qo = 1` + residual reads): slower for fork-bearing programs | 1 | Attention-kernel work is out by constraint; the M2b microbench gates whether the kernel options (cascade merge, per-page-len patch) ever graduate from recorded-direction; batch-by-program confines the cost to fork programs meanwhile |
| Per-instance predicated commit + resubmission re-runs a trunk on readiness miss | 3 (semantics), 2 (retry) | Structural readiness makes device-side misses the exception (late host edges only); measure dummy-run rate (M3 gate); stage-level resubmission recorded as a later optimization |
| ABI churn across three teams | all | C1–C4: append-only, opaque keys, named contracts, weekly sync |
| CUDA-graph key cardinality (program-set × shape buckets) | 3 | Bucket `B`/`num_tokens` as today; LRU eviction; capture only proven-hot tuples |
| `kv.rs` contention between thrusts 1 and 2 | 1, 2 | C4 ownership split (allocation vs. txn lifecycle); land S4 and M1 in separate PR trains |
| Quorum denominator thrash on agentic fleets | 2 | Denominator counts only pipelines with a submitted next pass; escape path is first-class and measured (escape rate probe) |

## 7. Document map

| doc | role |
|---|---|
| [`overview.md`](overview.md) | The model. Authoritative for semantics; supersedes [`../docs/tensor-ir.md`](../docs/tensor-ir.md) where they conflict |
| [`thrust-1-memory.md`](thrust-1-memory.md) | Working set, length columns, reclamation, attention-adjacent kernels |
| [`thrust-2-scheduler.md`](thrust-2-scheduler.md) | Non-blocking execute, run-ahead, quorum fire rule |
| [`thrust-3-programs.md`](thrust-3-programs.md) | Trace format, channels/epoch rings, compilation tiers |
| [`../docs/sampling-ir.md`](../docs/sampling-ir.md) | The epilogue specialization; still the JIT's design reference |
| [`../docs/run-ahead-submission-scheduler-handover.md`](../docs/run-ahead-submission-scheduler-handover.md) | Mechanism inventory for thrust 2; its parity barrier and `fire_at` formula are superseded by overview §7.2 |
| [`../docs/working-set-refactor-handover.md`](../docs/working-set-refactor-handover.md) | Background for thrust 1's starting state |
