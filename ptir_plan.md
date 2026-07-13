# PTIR Optimizing Compiler Plan

Revision 6. Supersedes the previous drafts. Revision 2 fixed six
architectural decisions up front as contracts, separated grouping from
fusion, made
step-varying extents symbolic in stage signatures, and defined the numeric
equivalence contract before any generated code ships. Revision 3 adds the
Metal track as a first-class part of the plan (Decision 7): Metal executes
generated code only, with the singleton partition of the shared compiler as
its Tier 0 and no interpreter tier. Revision 4 removes composite annotations
and workload-specific opcodes. Revision 5 removes the public nucleus opcode:
generic helpers lower to composable SSA, and the Rust compiler recognizes
exact normalized semantics into internal library regions. Revision 6 makes
generated fusion the only production execution path on every backend:
singleton partitions are test-only, unsupported programs reject at
registration, and handwritten Tier 0 is deleted after the generated oracle
gates pass.

Guiding directive for all tracks: clean implementation over
backward-compatibility staging. No transitional dual production paths are
maintained; superseded execution paths are deleted once the replacement
passes its oracle gates, and parity is always defined against the reference
semantics, never against legacy behavior.

## 1. Purpose

PTIR already provides a useful semantic boundary between inferlets and model
drivers, but the current implementation is primarily a correctness executor.
It preserves typed tensor operations, channel effects, and pass-atomic
publication, while leaving most launch and memory optimization opportunities
unexploited.

This document defines:

- the current execution model and its important limitations;
- seven binding architectural decisions;
- the north-star execution model for PTIR;
- the compiler, runtime, and driver features required to reach it;
- a staged implementation plan, including the Metal track and its
  relationship to the CUDA roadmap;
- correctness and performance criteria for declaring each phase complete.

The objective is not to build a general tensor compiler comparable to TVM or
Triton. PTIR has a smaller and more structured problem:

- a closed operation set (~52 container ops, ~45 of them compute);
- statically declared dtypes and channel shapes, with a small, known set of
  step-varying extents;
- rank at most four;
- stage-local SSA dataflow;
- explicit channel effects;
- a small set of recurring row-wise sampling, reduction, and descriptor
  workloads.

The implementation should exploit these constraints rather than introduce a
general scheduling language.

## 2. Executive Summary

Today, a batch of four compatible pipelines can share one model forward:

```text
forward [a, b, c, d]
```

However, post-forward PTIR execution is still dispatched per instance:

```text
epilogue(a)
epilogue(b)
epilogue(c)
epilogue(d)
```

Each epilogue is then interpreted by Tier 0 as one kernel launch per PTIR
core operation, plus channel movement and commit bookkeeping. Even identical
epilogues are not grouped.

The cost has two independent multipliers, and the plan attacks them
separately:

```text
current:            O(instances x stage ops)

after grouping:     O(unique ready stage signatures x stage ops)

after fusion:       O(unique ready stage signatures x fused/library regions)
```

Grouping (removing the `instances` factor) is a runtime concern: lane tables,
readiness, grouped commit. Fusion (removing the `ops` factor) is a codegen
concern: region formation and generated kernels. Production enables them
together: every accepted program is compiled at registration into fused
generated regions and stock semantic libraries. There is no fire-time
interpreter, handwritten Tier 0, singleton fallback, or compile-miss
degradation. The singleton partition exists only in tests as a generated
differential oracle.

The north star is:

```text
forward [a, b, c, d]

ready group S1 [a, d] -> fused region(s) for S1
ready group S2 [b, c] -> fused region(s) for S2

grouped pass-atomic commit
```

where `S1` and `S2` are canonical stage signatures, not whole-program
identities.

## 3. Current State

All claims in this section were verified against the code at this revision.

### 3.1 What already works

The current system has the semantic substrate needed for an optimizing
compiler:

- canonical PTIR containers and stable whole-program hashes
  (FNV-1a-64 over canonical container bytes, `interface/ptir/src/lib.rs`);
- typed, bound PTIR traces with declared dtypes and shapes;
- explicit stages and stage-local SSA values;
- explicit `ChanTake` and `ChanPut` effects;
- device-resident channel cells with epoch/ticket readiness validation and
  all-or-nothing ticket reservation per pass
  (`runtime/engine/src/pipeline/channel.rs`,
  `runtime/engine/src/pipeline/fire.rs`, `TicketReservation`);
- model-forward batching across multiple PTIR instances;
- per-program attribution of sampled logits rows;
- CUDA Tier 0 execution covering the PTIR operation set;
- exact reference behavior for sampling operations, including top-p;
- keyed RNG as a pure function of (key, counter, element index), with
  counter advancement authored explicitly in the PTIR program;
- a test-only Tier 1 fused-codegen prototype;
- CUDA and Metal interpreter/reference paths suitable for parity testing.

The missing layer is the optimizer and its grouped production execution
runtime.

### 3.2 Current scheduler and forward batching

The runtime scheduler places multiple distinct PTIR instances into one
native launch when structural constraints permit. `LaunchGrouping`
(`runtime/engine/src/scheduler/worker.rs`) merges fires by capacity only:

- no duplicate fire of the same instance in one launch;
- forward-request, token, and page-reference limits;
- solo execution for prebuilt requests;
- solo execution for requests carrying custom masks.

It does not require, or ever consult, program hashes. The CUDA driver
composes the requests into one flat model-forward batch and records
per-program sampled-row offsets (`driver/cuda/src/batch/compose.cpp`).

This means heterogeneous inferlets already share expensive model work.

### 3.3 Current post-forward dispatch

After the model forward:

1. selected bf16 logits rows are gathered and cast into a separate FP32
   buffer (`driver/cuda/src/batch/logits.cpp`);
2. the driver's `Dispatch::run` iterates PTIR instances in launch order and
   calls `fire_async` per instance (`driver/cuda/src/pipeline/dispatch.cu`);
3. on the engine side, `BatchScheduler::retire_ready_launches`
   (`runtime/engine/src/scheduler/worker.rs`) settles each fire
   individually as committed, failed, or retry.

There is no grouping by equal whole programs or equal stages. Two instances
of the same program still execute through separate `fire_async` calls.

### 3.4 Current identity model

`program_hash` identifies the complete canonical PTIR container. It excludes
per-instance data (seeds, working-set bindings, RNG values) by construction.
It is used for registration deduplication, decoded-program caching, and
driver compile-cache identity.

It cannot identify reusable stages across different programs. Programs with
different prologues but identical epilogues have different program hashes.
There is no canonical stage signature today, and the `Stage` type is a
four-variant position tag, not an identity.

The CUDA forward-graph key contains a `program_set_hash` field and a folding
helper (`driver/cuda/src/batch/forward_graph.hpp`), but no construction site
populates it; it currently defaults to zero. It is a designed-in hook for
future in-graph PTIR capture, not live behavior. PTIR execution runs outside
the captured forward graph, as eager launches on the sampling stream.

### 3.5 Current Tier 0 lowering

Production CUDA PTIR execution is operation-oriented
(`driver/cuda/src/pipeline/tier0/tier0_runner.hpp`):

- one prebuilt kernel launch per core operation, with reshape as the alias
  exception;
- every intermediate materialized in per-pass scratch allocations;
- each channel put performed as a separate device-to-device copy into the
  channel's pending cell;
- one `k_commit_bump<<<1,1>>>` kernel per instance publishing the pass
  atomically, guarded by the accumulated readiness predicate.

SDK composites are fully expanded before execution with no provenance:
`softmax` becomes seven core ops (max, broadcast, sub, exp, sum, broadcast,
div) indistinguishable from hand-written ops
(`sdk/rust/ptir-dsl/src/value.rs`). A chat decode epilogue therefore requires
roughly two dozen launches even though most of its state and geometry update
is a straightforward multi-output row program.

The former driver-private fusion-only opcode band is removed. Generated
backends consume the same normalized general SSA as the reference semantics;
there is no second backend-only composite IR.

### 3.6 Current Tier 1 prototype

The existing Tier 1 test support (`driver/cuda/tests/support/tier1_codegen.hpp`)
proves that generated fusion is possible, but its scope is narrow:

- a single stage consisting of a linear per-vocabulary elementwise chain;
- exactly one terminal reduction (argmax, sum, or max);
- string codegen of NVRTC-compilable CUDA-C; compilation and caching are
  left to the caller;
- test-only invocation.

It does not provide arbitrary stage DAGs, multiple reductions, multiple or
mixed-domain outputs, grouped instances, descriptor and channel sinks,
production compilation and caching, or fallback handling.

### 3.7 Shapes are not fully static across fires

Container-declared channel shapes and dtypes are static and part of program
identity. But the bound geometry of a fire is recomputed every submit from
current channel values (`runtime/engine/src/pipeline/fire/geometry.rs`):
`KvLen`, `Pages`, `PageIndptr`, token counts, and sampled-row counts all vary
per decode step. The same hash-identical program produces differently shaped
geometry every token.

Any signature or compile-cache design that bakes these extents in as
constants would churn every step. This motivates the symbolic-extent model in
Decision 3.

### 3.8 Channel readiness already has a validation mechanism

Channels are device-resident bounded queues. Readiness is expressed as
per-channel expected head/tail tickets reserved at submit time;
`TicketReservation` reserves tickets for all of a pass's channels up front,
commits them only on successful submit, and rolls all of them back on
failure (`runtime/engine/src/pipeline/fire.rs`). Cross-instance channels
(extern import/export) are ordered by the shared pipeline FIFO across fires,
not within a single batched forward.

Grouped execution therefore does not need a new readiness scheduler; it
needs to partition already-ticket-validated fires (Decision 6).

### 3.9 Retry-ineligible recurrent state

Fires carrying recurrent-state (RS) slots are retry-ineligible: the engine
rejects a driver RETRY outcome for them
(`runtime/engine/src/scheduler/worker.rs`, `retry_eligible`). RS is in-place
or CoW mutable state for hybrid/linear-attention models; preemption migrates
KV but deliberately leaves RS resident
(`runtime/engine/src/inferlet/process/preemption.rs`). Grouped execution must
respect this (Decision 6).

### 3.10 Important current batching restrictions

Some workloads are deliberately solo today:

- dense device attention masks in multi-program batches;
- custom wire masks combined with device geometry;
- prebuilt/device-geometry requests;
- recurrent-state configurations that cannot safely participate in the
  existing composed path.

These are runtime and attention integration constraints, not merely compiler
limitations. Stage fusion alone will not remove them; Phase 5 addresses the
mask-representation subset.

### 3.11 Metal driver current state

Verified against `driver/metal` after its 2026-07 refactor:

- Metal's production PTIR engine is a CPU host interpreter
  (`driver/metal/src/pipeline/interp.hpp`), a C++ mirror of the Rust
  reference pinned to the same readiness, commit, tie-break, and RNG
  semantics. There is no GPU execution plane for PTIR: no per-op kernels,
  no device channel state, no commit kernel.
- Channel cells are host-resident
  (`driver/metal/src/pipeline/registry.cpp`); pending and commit semantics
  live entirely in the interpreter.
- Logits cross the forward/PTIR seam through a CPU per-element bf16-to-f32
  conversion (`driver/metal/src/batch/forward.cpp`); PTIR runs strictly
  after the forward, synchronized by shared events.
- Kernel compilation is already runtime MSL source compilation
  (`newLibraryWithSource`, Metal 4 compiler); there is no binary-archive
  cache and no function-constant specialization.
- The decode path is one command buffer per step with argument tables built
  once and reused; decode kernels already use width-32 simdgroup
  reductions, matching the canonical reduction tree width.
- Keyed RNG constants exist byte-identical to CUDA's, but only in the CPU
  interpreter; there is no device RNG kernel.
- Forward batching is sealed at one request; the paged multi-request path
  is gated, not faked.
- The shared surface (`driver/common/include/pie_native`: op table, wire
  format, geometry helpers, driver ABI) is already common to both drivers.

The consequence: Metal has no legacy device tier to preserve. It can adopt
the generated-execution end state directly (Decision 7) without porting the
CUDA Tier 0 kernels or keeping an interpreter tier alive.

## 4. Architectural Decisions

These seven decisions are binding for the rest of the document. Each removes
a class of future rework.

### Decision 1: The plan is computed in Rust; drivers execute plans

Normalization, stage signatures, region partitioning, library-island cuts,
and the lane-table ABI are backend-neutral. All of them are computed once, in
Rust, alongside the `interface/ptir` crate. A driver receives exactly two
things:

- a serialized region plan per stage signature: the ordered list of regions,
  each either an op DAG to generate code for or a library call;
- a lane table per fire group: the per-lane runtime data (Section 6.5).

CUDA consumes plans through NVRTC codegen; Metal consumes the same plans
through its own codegen. Backend-specific scheduling and library
implementations are expected; a second IR is not. This keeps the driver
general and prevents the compiler from existing twice in two languages and a
third time for Metal.

### Decision 2: User IR is composable SSA; Rust recognizes library semantics

No composite annotations and no driver-side DAG pattern matcher. SDK convenience helpers
such as softmax, masked argmax, entropy, and scalar gather lower to ordinary
general-purpose SSA. Their call-site names do not enter canonical bytes,
program hashes, signatures, plans, or backend dispatch.

When a normalized SSA subgraph exactly implements a replaceable semantic
library contract, the Rust compiler recognizes its typed dataflow and emits an
internal library region. Exact nucleus sampling is the initial recognized
composite. Existing general opcodes such as `TopK`, `SortDesc`, scans, and
`MatMul` identify their own library boundaries directly. Recognition affects
only performance: a miss executes the generic SSA unchanged. Drivers consume
the serialized plan and never recover SDK intent.

Workload policy must not enter the core IR. Beam selection is `TopK` plus
ordinary index arithmetic and gathers, not `BeamSelect`. Beam ancestry is
expressed with general row-membership SSA, not a `BeamAncestryMask` opcode.

### Decision 3: Stage signatures treat step-varying extents as symbolic

Dimensions in the compiler IR are `Static(n)` or `Sym(id)`. Truly static
extents (vocabulary size, declared channel shapes) are baked into the
signature. Extents that vary per fire (kv length, page counts, row counts,
token counts; see Section 3.7) enter the signature only as symbolic
structure; their values flow through the lane table at dispatch time.

Schedule buckets (for example, row-count ranges that select between the
one-CTA and hierarchical templates) are a separate key layered on top of the
signature for compiled-executable caching. They never affect semantics or
grouping identity.

Without this split, mask and descriptor stages would produce a new signature
every token, destroying both grouping and the compile cache.

### Decision 4: Production is generated fused execution only

Every accepted stage executes the Rust-owned fused partition over grouped
lanes. Generic regions are generated code; semantic islands use stock
libraries. Compilation occurs at registration, before any instance can fire.

- Full first-party op coverage is a prerequisite for cutover.
- A deterministic unsupported/codegen failure rejects registration with an
  exact reason.
- A transient compiler/cache/resource failure is retryable and is never
  negative-cached as unsupported.
- There is no fire-time compile, handwritten Tier 0, per-op fallback,
  singleton production path, or success-shaped degradation.
- The generated singleton partition is test-only and may be compiled by
  differential tests, never selected by production dispatch.

This guarantees the production cost floor is
`unique ready signatures x necessary fused/library regions`; no failure mode
silently returns to `signatures x ops`.

### Decision 5: The numeric equivalence contract is fixed before any generated code ships

Floating-point reductions are not associative, so "same results" must be
defined, not assumed. The contract:

- Every reduction op has a canonical reduction schedule whose accumulation
  order is a function of the static shape only (fixed lane-tree width, fixed
  stride pattern), never of launch configuration. Generated singleton and
  fused kernels implement the same written schedule. The Rust reference
  interpreter adopts the same canonical
  schedules, so float accumulation over identical inputs is
  bitwise-comparable across implementations.
- Within one backend, fused generated code matches the test-only generated
  singleton partition bitwise for identical inputs.
- Across backends (CUDA vs Metal vs reference interpreter), parity is
  semantic: tie-breaking rules, top-p inclusion and cutoff rules, NaN
  handling, and the integer hashing path of keyed RNG match bitwise;
  transcendental functions (exp, log) match within a defined tolerance,
  because their implementations legitimately differ per backend.
- Moving the bf16-to-f32 conversion point (Section 7, gather elimination) is
  a change under this contract and must be specified as such: today the cast
  happens once at gather; direct consumption upcasts in registers.

This is what keeps differential tests deterministic instead of flaky, given
that sampled tokens are decided by floating-point comparisons.

### Decision 6: Readiness rides the existing ticket system; RS lanes never join failure-capable groups

No new readiness DAG scheduler. Group formation is: at retirement time,
partition the fires whose ticket reservations validated into
(stage signature, schedule bucket) groups, and execute each group. Channel
ordering across instances is already enforced by ticket expectations and the
pipeline FIFO; a lane whose expected epoch is not yet committed is simply not
ready and does not join the group.

Corollary for recurrent state: RS-carrying lanes are retry-ineligible
(Section 3.9), so they may only join groups whose readiness is fully
determined before launch. Under the current submit-time ticket model this
holds by construction; it is stated as an invariant (Section 10) so that
runahead and late-prepare work cannot silently violate it.

### Decision 7: Both backends share one generated-only production policy

CUDA and Metal both execute only fused generated regions and stock libraries
in production. The region partitioner still emits fused and singleton
partitions from one normalized stage, but singleton is compiled only by
tests as an on-device differential baseline. It is not a coverage fallback.

Compilation starts at program registration, which always precedes the first
fire. Executables persist in a bounded on-disk cache keyed by the standard
cache identity. Registration completes only when every fused/library region
is available; deterministic failure rejects with a reason, while transient
failure remains retryable.

Execution model consequences, all enabled by unified memory:

- channel cells move into shared-storage buffers, so the same memory serves
  host orchestration and device kernels with no mirror machinery;
- generated kernels perform readiness predication, direct channel sinks,
  and grouped commit on device, symmetric with the CUDA contract (6.8);
- the CPU retains only encode-time orchestration (descriptor and geometry
  resolution, scheduling), the same host-side role it has on CUDA;
- PTIR regions are encoded into the same command buffer as the forward,
  consuming bf16 logits directly, so no CPU code touches vocabulary-sized
  data.

Parity is defined against the Rust reference interpreter and the test-only
generated singleton partition, never against legacy behavior. Handwritten
CUDA Tier 0 and the Metal CPU interpreter are deleted. The end state has one
semantic implementation (Rust) and backend code generated from one shared
plan.

## 5. North-Star Execution Model

### 5.1 Core principle

The target unit of execution is:

> One grouped launch per fused region of each unique ready stage signature,
> with explicit library cuts and pass-atomic effects.

It is not:

- one launch per PTIR operation;
- one launch per PTIR instance;
- necessarily one launch per complete epilogue.

An epilogue that contains an exact top-p library island may optimally require
two or three regions. A simple masked argmax epilogue should require one.

### 5.2 Example

Given four pipelines:

```text
program a: prologue A, epilogue E1
program b: prologue B, epilogue E2
program c: prologue C, epilogue E2
program d: prologue D, epilogue E1
```

the desired post-forward execution is:

```text
forward [a, b, c, d]

E1 lanes [a, d]
E2 lanes [b, c]

publish [a, b, c, d]
```

Each lane supplies runtime data through the lane table (Section 6.5); the
compiled code and schedule are shared by all lanes in the group. The same
machinery applies to pre-forward stages (mask and descriptor preparation),
not only epilogues.

### 5.3 Common target topologies

| Workload | North-star post-forward topology |
| --- | --- |
| Greedy or masked argmax | 1 grouped fused kernel |
| Gumbel-max watermarking | 1 grouped fused kernel |
| Sampling primitives | 1 grouped row kernel, multiple reductions/outputs |
| Chat exact top-p | nucleus library region + fused state/geometry region |
| Beam search | candidate transform + selection library region + state update |
| Speculative verification | grouped multi-row regions; host accept/reject boundary |

Library implementations may internally launch more than one CUDA kernel. The
compiler counts and optimizes semantic regions rather than pretending a
global sort is one physical launch.

## 6. Required Features

### 6.1 Canonical stage signatures

A stage signature must encode everything that changes generated code or
semantics:

- stage kind;
- normalized operation DAG;
- input and output dtypes, ranks, and static dimensions;
- symbolic-extent structure (which dims are symbolic, and their roles);
- intrinsic and port schema;
- reduction and scan semantics;
- channel take/peek/put effect schema;
- recognized semantic library modes;
- compile-time constants used for specialization;
- required library operations and semantic modes;
- backend-visible behavior such as exact versus approximate sampling.

It must exclude instance-specific data:

- channel IDs and addresses;
- logits pointers and row offsets;
- current channel epochs;
- RNG values;
- symbolic extent values;
- runtime scalars passed as arguments rather than specialized.

Whole-program hashes remain the registration identity. Stage signatures serve
code reuse, grouping, and compiled-kernel caching. Signature computation must
be deterministic: normalization (6.2) canonicalizes operand order for
commutative ops and value numbering before hashing.

### 6.2 Typed SSA normalization

Before partitioning, each bound stage is normalized into the compiler IR:

- dead-code elimination;
- common-subexpression elimination;
- constant folding;
- algebraic simplification;
- reshape/view elimination;
- broadcast lowering to indexed loads;
- `iota` lowering to generated indices;
- redundant descriptor construction elimination;
- invariant hoisting where channel and pass semantics permit it.

Generic normalization may run in any order because no source-range metadata
needs to survive it. Semantic recognition runs over normalized typed dataflow,
while the normalized IR retains source-to-PTIR mappings for diagnostics and
test-only singleton differential testing.

### 6.3 Semantic operations and composable helpers

SDK helpers for softmax/log-softmax, Gumbel-max, masked argmax, entropy, scalar
gather, and beam state updates expand into general SSA. The compiler optimizes
their typed dataflow without knowing the helper name.

Exact nucleus sampling remains ordinary softmax, cumulative-mass mask, keyed
Gumbel, and argmax SSA in the wire program. Rust may replace the exact graph
with an internal semantic library region. `TopK`, pivot-threshold selection,
sort, scans, and matrix multiplication are already explicit general opcodes.

Structured causal, window, and sink attention masks remain reusable semantic
geometry operations. Workload-specific ancestry policy is composed from
general tensor operations and uses the dense fallback until a general sparse
ownership descriptor ABI exists.

### 6.4 Value-domain analysis

Classify values by execution domain: scalar; per-row; per-vocabulary-element;
generated index; pool or mask element; page descriptor; library result;
effect token.

Fusion decisions are domain-aware. A single kernel may produce outputs from
several domains using guarded writes: all threads write vocabulary or mask
elements; selected threads write page descriptors; lane leaders write token,
position, RNG, and length scalars. This mixed-domain multi-output capability
is essential for real inferlets.

### 6.5 Lane-table ABI

The lane table is the single grouped-dispatch ABI consumed by generated
regions and stock libraries. Per lane it carries:

- logits base pointer and row span;
- channel cell addresses and expected epochs;
- scalar and descriptor inputs;
- symbolic extent values (kv length, page count, row count);
- RNG state location;
- pending output cell addresses;
- per-lane commit slot.

Requirements:

- heterogeneous runtime addresses and row spans; no contiguity assumption
  over instance state;
- the first implementation may require exact static-shape equality within a
  group; ragged groups (per-lane row offsets/counts, inactive-lane masking)
  follow in Phase 4;
- stable layout, versioned with the plan format, so CUDA graph capture can
  later update it through graph-compatible buffers.

### 6.6 Stage-local fusion

The compiler forms maximal legal regions bounded by:

- explicit library operations;
- operations without generated/library coverage (registration rejection,
  Decision 4);
- required global synchronization;
- incompatible schedules;
- effect ordering that cannot be represented inside a region;
- resource limits such as registers, shared memory, and occupancy.

Fusion must support arbitrary elementwise DAGs, multiple reductions in one
stage, reduction results consumed by later elementwise work, multiple
outputs, scalar and tensor gathers, masked and guarded stores, and multiple
logical sweeps within one CTA using barriers.

The default is maximal safe fusion, with a small cost model splitting a
region only when required for correctness or clearly beneficial for resource
usage.

An operation that cannot be represented by generated code or an explicit
library is a registration error. It never selects another production tier.

### 6.7 Multi-output direct channel sinks

Compiled kernels write results directly into reserved pending channel cells
whenever possible, removing scratch materialization for terminal values,
separate device-to-device channel copies, and the associated launch and
bandwidth overhead.

Output pointers are provided only after the corresponding channel capacity is
reserved and validated. Failed or retrying passes must not expose partial
results.

### 6.8 Grouped pass-atomic commit

Fusion and grouping preserve the existing channel contract:

- takes observe the expected committed cell;
- puts target pending cells;
- no output becomes visible before the pass succeeds;
- all effects for one instance commit atomically;
- retry leaves externally visible channel state unchanged;
- repeated puts retain the defined last-put behavior.

Commit is itself grouped: one finalization kernel per group with per-lane
guards replaces the current per-instance `k_commit_bump<<<1,1>>>`. For
one-CTA-per-instance bodies the kernel may publish its own commit record;
multi-CTA work requires the grouped finalization kernel or another explicit
global synchronization mechanism.

Grouped execution does not imply grouped fate: one lane may fail readiness or
retry without corrupting successful lanes, subject to the RS invariant
(Decision 6).

### 6.9 Reduction schedule templates

Two templates cover the corpus initially:

One CTA per row: elementwise transforms; max, sum, argmax; softmax and
log-softmax; masked and Gumbel argmax; entropy; scalar gather; mixed-domain
state writes.

Hierarchical row reduction: when a vocabulary row is too large or there are
too few rows to occupy the GPU:

```text
partial row reductions -> final reduction or selection -> optional fused
consumer/state update
```

The schedule selector uses static shape, grouped lane count, backend limits,
and operation class through a lightweight decision table, not autotuning.
Both templates implement the canonical reduction schedules of Decision 5, so
schedule choice never changes results within a backend.

### 6.10 Library islands

First-class compiler boundaries with a semantic ABI:

```text
NucleusSample(logits, top_p, rng_state) -> token  // internal compiler ABI
TopK(scores, k) -> values, indices
```

Temperature scaling is ordinary SSA applied before the recognized nucleus
subgraph.

plus matmul and model-owned projections. A library island defines exact
numerical and tie-breaking semantics, supported shapes and dtypes, scratch
requirements, stream behavior, output layout, whether producers or consumers
may fuse into it, and CUDA/Metal parity requirements.

Exact top-p is one semantic sampler, not something split at an arbitrary
internal pivot operation, so its selection algorithm can change without
changing PTIR-visible behavior.

### 6.11 Production compilation and caching

Cache identity:

```text
backend, device architecture, compiler version,
stage signature, shape/schedule bucket, semantic mode
```

Lifecycle behavior:

- lookup compiled executable; compile every miss during registration;
- publish the program id only after every fused/library executable is ready;
- reject unsupported regions at registration with source-mapped diagnostics;
- negative-cache only deterministic compilation failures;
- surface transient compiler, disk, memory, and cache-capacity failures as
  retryable registration errors;
- bound executable and code-memory usage;
- persistent on-disk cache under the same identity (binary archives on
  Metal), so serving restarts do not repay warm-up jitter;
- expose cache and compile metrics;
- invalidate safely when compiler or ABI versions change.

The system must never silently substitute approximate behavior for an exact
PTIR operation.

### 6.12 Parity testing

The parity contract implements Decision 5 and explicitly covers:

- argmax tie-breaking, including lower-index preference;
- NaN handling;
- reduction identities, empty cases, and canonical accumulation order;
- exact top-p inclusion and cutoff rules;
- pivot and TopK tie behavior;
- keyed RNG mapping from key, counter, and element index, and explicit
  counter advancement;
- gather and scatter edge cases;
- integer overflow behavior where defined;
- channel take, peek, put, retry, and commit semantics.

Test matrix:

- fused CUDA vs test-only singleton CUDA: bitwise, on device;
- fused Metal vs test-only singleton Metal: bitwise, on device;
- generated code vs the Rust reference interpreter: bitwise for
  integer/selection/RNG-hashing paths and for float accumulation under the
  canonical reduction schedule (the reference implements the same
  schedule); tolerance only for transcendentals;
- randomized differential tests with adversarial ties, NaNs, empty masks,
  extreme temperatures, and boundary top-p values.

Metal PTIR pipelines must be compiled with fast math disabled (the Metal
default is enabled and silently breaks NaN, mask, and argmax semantics);
this is enforced by a compile-options gate, not convention.

### 6.13 Observability

Per stage signature, expose: ready-instance count; grouped lane count; chosen
schedule; fused and library region counts; physical launch counts;
compile/cache state; scratch and shared-memory usage; reasons for fusion
cuts; registration rejection reasons; reasons an instance could not join a
group; retry and commit outcomes.

A debug mode prints a normalized stage DAG and its region partition without
requiring CUDA execution.

## 7. Memory-Traffic Features Beyond Launch Count

### 7.1 Remove the sampled-logits gather where possible

The current path gathers sampled bf16 rows and casts them into a separate
FP32 buffer before PTIR dispatch. A grouped sampler should eventually consume
model output directly using row indices and strides, removing two kernels,
one full sampled-row write, one full sampled-row read, and an intermediate
buffer. The conversion-point change is specified under the numeric contract
(Decision 5). This follows correct grouped dispatch; it does not block it.

### 7.2 Ragged and multi-row stages

The corpus includes `[vocab]` single-row sampling, `[B, vocab]` beam and
consensus work, `[k+1, vocab]` speculative verification, and varying
sampled-row counts per instance. Ragged grouping is the natural extension of
symbolic extents into the lane table: per-lane row offset and count, padded
or inactive rows, and row-to-instance attribution for commit.

### 7.3 Structured attention masks and geometry

Dense causal, sliding-window, and sink masks are structured values.
Materializing dense booleans only to repack them for an attention backend
wastes bandwidth and can force solo execution. The long-term representation
is semantic ops (Decision 2 applied to masks):

```text
CausalMask(kv_len)
SlidingWindowMask(kv_len, window)
SinkWindowMask(kv_len, sink, window)
```

flowing directly into the attention backend where possible, with arbitrary
dense masks as a supported fallback. Beam ancestry remains a general
row-membership SSA composition; no workload-specific opcode enters PTIR.

### 7.4 CUDA graph integration

Once grouped generated stages are production-ready, graph topology should be
keyed by the actual compiled stage sequence and schedule buckets, which is
when the currently unwired `program_set_hash` hook gets populated. Pointer
indirection tables must remain stable or be updated through graph-compatible
buffers. Graph integration is a later optimization and must not dictate the
initial ABI.

## 8. Workload Capability Matrix

| Inferlet or family | Core compiler requirement | Expected hard cut |
| --- | --- | --- |
| `sampling-primitives` | multiple reductions, multi-output row kernel | none |
| `greenlist-watermarking` | select/bias, keyed RNG, Gumbel-max, scalar state sinks | host-produced green mask |
| `json-schema-constrained-decoding` | masked argmax, empty-mask policy, scalar state sinks | host grammar engine |
| `mirostat-v2-sampling` | softmax, pivot threshold, Gumbel-max, scalar gather | TopK for rank floor; host `mu` update |
| `chat-completion` | generic nucleus SSA, mixed-domain state and geometry outputs | compiler-recognized nucleus library |
| `attention-sink` | structured sink/window mask, state update | attention backend |
| `sliding-window-attention` | structured window mask, state update | attention backend |
| `beam-search` | `[B,V]` transforms, flatten selection, parent gather, many outputs | global TopK/beam select |
| `consensus-decoding` | batched rows and multi-output reductions | host consensus policy where applicable |
| `mtp-speculative-decoding` | multi-row logits and verify-window reductions | host accept/reject |
| `cacheback-speculative-decoding` | multi-row sampling and descriptor updates | host accept/reject and cache control |
| `contrastive-decoding` | per-model epilogue fusion | host-mediated boundary between model passes |
| `prefix-tree-kv-cache` | row kernels and descriptor updates | WorkingSet fork/COW runtime |

The compiler does not fuse through host policy, model boundaries, WorkingSet
ownership operations, or attention-library calls without an explicit semantic
ABI.

## 9. Implementation Roadmap

### Phase 0: Contracts and measurement (no behavior change)

Deliverables:

- exact normalized-dataflow recognition for semantic library contracts;
  generic SDK helpers remain ordinary SSA and do not alter canonical identity;
- normalization passes and `StageSignature` with the symbolic-extent model,
  implemented in Rust alongside `interface/ptir`;
- the numeric contract written down: canonical reduction schedules defined
  independently of launch configuration; within-backend bitwise and
  cross-backend semantic tiers defined;
- lane-table and region-plan format drafts;
- instrumentation: per-fire launch counts, scratch bytes, channel-copy
  bytes, and post-forward critical-path time (CPU dispatch plus GPU epilogue
  span); baseline traces for representative inferlets.

Exit criteria:

- identical stages in different complete programs receive identical
  signatures; semantically or shape-incompatible stages receive different
  signatures;
- signatures are stable across decode steps as kv length and page counts
  grow;
- recognized and unrecognized forms execute identically under registration
  validation;
- baselines are recorded; no production execution behavior changes.

### Phase 1: Full generated coverage (test-only singleton)

Scope:

- lane-table ABI implementation;
- generated singleton emitter for every first-party op, effect, intrinsic,
  symbolic extent, and dtype;
- stock library ABI coverage for TopK, exact nucleus, scans, matmul, and
  second-party boundaries;
- device readiness, direct pending-cell sinks, and atomic commit;
- test harness only: no production dispatcher may select singleton.

Exit criteria:

- every valid first-party golden compiles as singleton on CUDA and Metal;
- singleton output matches the Rust reference under the numeric contract;
- malformed/unsupported programs reject at registration;
- singleton symbols are absent from production dispatch and metrics.

### Phase 2: Generated fused regions (only production path)

Scope:

- Rust region partitioner over the normalized IR, bounded by explicit library
  operations;
- CUDA NVRTC codegen consuming region plans and the Phase 1 lane ABI;
- one-CTA-per-row template; elementwise DAGs; multiple reductions under the
  canonical schedules; multiple and mixed-domain outputs; direct channel
  sinks;
- registration-time production compile cache (bounded memory + persistent
  disk), transient retry, and deterministic negative caching;
- grouped execution of each ready signature over fused/library regions;
- no compile-on-fire or alternative execution path.

Target inferlets: `sampling-primitives`, `greenlist-watermarking`,
`json-schema-constrained-decoding`, and masked-argmax portions of others.

Exit criteria:

- one generated body launch per signature group for target epilogues, plus
  grouped commit only where required;
- fused output bitwise-matches test-only singleton on the same backend;
- unsupported stages reject before an instance can be created;
- first fire has no compilation work or fallback latency cliff.

### Phase 3: Semantic samplers and chat decode

Scope:

- generic fusion of softmax/log-softmax, Gumbel-max, entropy, and scalar
  gather SSA;
- exact nucleus-sampling library ABI and implementation;
- fused sampler consumer and state/geometry update region.

Target inferlets: `chat-completion`, Mirostat v2 argmax-floor path, simple
attention-window state updates.

Exit criteria:

- chat decode post-forward PTIR is reduced from roughly two dozen launches
  to a nucleus library region, one fused state/geometry region, and grouped
  commit;
- no full-probability intermediate crosses the nucleus boundary unless
  required by a PTIR output;
- exact top-p and RNG parity hold on adversarial tests.

### Phase 4: Ragged, multi-row, and selection workloads

Scope:

- ragged lane ABI (per-lane row offsets/counts, inactive-lane masking);
- `[B,V]` and `[k+1,V]` row execution; hierarchical reduction template;
- exact TopK library ABI;
- general parent-index decomposition, gathers, and row-membership ancestry/state
  updates;
- per-instance completion from multi-row groups.

Target inferlets: `beam-search`, `consensus-decoding`,
`mtp-speculative-decoding`, `cacheback-speculative-decoding`, Mirostat v2
rank-floor path.

Exit criteria:

- multi-row instances group without losing instance attribution;
- global selection uses explicit library regions;
- host accept/reject and model boundaries remain explicit and correct;
- group commit handles partial readiness and retry safely.

### Phase 5: Structured geometry and memory traffic

Scope:

- semantic causal/window/sink descriptor ops in the SDK and container;
- attention-backend descriptor consumption; dense-mask fallback;
- invariant descriptor hoisting; elimination of redundant `Pages` and
  `PageIndptr` materialization;
- direct bf16 sampled-logits access where backend layout permits, specified
  under the numeric contract.

Exit criteria:

- structured masks need no dense materialization and repacking;
- eligible structured-mask requests are no longer forced solo solely because
  of mask representation;
- the sampled-logits gather is removed for supported layouts;
- attention and PTIR parity remain exact.

### Phase 6: Graphs and tuning (CUDA)

Scope:

- CUDA graph capture keyed by compiled stage sequences and schedule buckets
  (populating `program_set_hash`); indirection through graph-compatible
  buffers;
- stream overlap for independent signature groups;
- lightweight cost-model refinement; bounded background compilation and
  cache management;
- persistent or cooperative schedules only where measurements justify them.

Exit criteria:

- steady-state execution avoids CPU launch overhead for stable batch shapes;
- compiler and cache resource usage remain bounded and observable.

### Phase 7: Generated-only cutover and Tier 0 deletion

Scope:

- make fused/library compilation mandatory during registration on CUDA and
  Metal;
- move singleton compilation behind test-only build APIs with no production
  symbol or dispatch route;
- delete handwritten CUDA Tier 0 runners, kernels, launchers, grouped per-op
  execution, compile-miss fallback state, fallback metrics, and fallback
  environment switches;
- delete transitional interpreter/fallback tests and replace them with
  Rust-reference versus generated-singleton versus fused differential tests;
- reject any program whose fused partition lacks complete generated/library
  coverage before allocating a native instance.

Exit criteria:

- repository production sources contain no Tier 0 dispatcher, fallback branch,
  or singleton selection;
- every first-party golden and migrated inferlet registers and executes through
  fused/library production code on both backends;
- injected compile misses, transient failures, and deterministic unsupported
  programs never reach fire-time execution;
- target launch topology is `unique ready signatures x fused/library regions`
  at B=1/2/4/8, including staged prologue/attention/epilogue programs;
- Rust-reference, test-only singleton, and fused outputs satisfy the numeric
  contract; CUDA ASan/Compute Sanitizer and Metal full suites are clean.

### Metal track

The Metal track is not a port of the CUDA phases; it is a different
projection of the same Rust-owned plan artifacts (Decisions 1 and 7). It
runs in parallel with the CUDA phases and starts as soon as the Phase 0
plan format exists. The CPU interpreter is deleted before cutover; generated
singleton remains test-only, and production starts only when fused M2/M3
coverage is complete.

The forward-side prerequisites (paged multi-request batching, async
completion) continue under `metal_ptir_plan.md` Phases 1-3; that plan's
Phase 4 (a handwritten MSL port of the Tier 0 kernels) is superseded by
this track.

#### M0: Substrate (parallel with CUDA Phase 0)

Scope:

- channel cells move to shared-storage buffers; the interpreter operates on
  them unchanged (unified memory makes this a relocation, not a redesign);
- keyed RNG constants and the canonical reduction tree implemented and
  unit-verified in MSL (simdgroup width 32 matches the tree width);
- the fast-math compile-options gate;
- instrumentation: CPU epilogue time, bf16-to-f32 conversion time, and
  forward-wait time per step.

Exit criteria:

- RNG and reduction golden vectors match the Rust reference bitwise on
  device;
- channel relocation changes no observable behavior.

#### M1: Singleton codegen over the full operation set (test-only)

Scope:

- MSL emitter consuming region plans in singleton partition; per-op-class
  templates, one-CTA-per-row schedule first;
- registration-time compilation, persistent executable cache, one-time
  block on a true miss, registration-time rejection for deterministic
  failures;
- readiness predication, direct channel sinks, and grouped commit in
  generated code; lane ABI in place with lane count one;
- host-side differential harness against the Rust reference.

Exit criteria:

- full PTIR op coverage through generated singleton regions;
- parity against the Rust reference under the numeric contract across the
  golden and adversarial suites;
- the CPU interpreter is deleted;
- singleton has no production dispatch entry point.

#### M2: Fused regions on the forward's command buffer

Scope:

- fused-partition execution using the CUDA Phase 2 partitioner output;
- PTIR regions encoded into the same command buffer as the forward, after
  the LM head;
- direct bf16 logits consumption in generated kernels, removing the CPU
  per-element conversion loop;
- fused vs singleton on-device bitwise differential in CI.
- fused executables are the only programs published by registration.

Exit criteria:

- zero CPU-side vocabulary-sized work per decode step;
- fused output bitwise-matches singleton output on device;
- measured reduction in post-forward critical-path time per step.

#### M3: Grouping and libraries (after Metal multi-request batching)

Scope:

- grouped dispatch over lanes (the ABI is already in place from M1);
- MSL library islands (exact nucleus sampling and TopK)
  implementing the same semantic ABIs as CUDA Phases 3 and 4;
- ragged lanes as needed by the multi-row workloads.

Exit criteria:

- launch scaling with unique ready signatures, matching the CUDA guarantee;
- library semantics pass the shared cross-backend suite under the two-tier
  parity contract.

## 10. Correctness Invariants

The following are non-negotiable:

1. A fused generated region must be observationally equivalent to the Rust
   reference and test-only generated singleton under Decision 5; within one
   backend this means bitwise.
2. Grouping must not change channel dependency order.
3. One instance's failure or retry must not expose partial writes or poison
   successful instances.
4. An RS-carrying lane must never join a group whose readiness can fail
   after launch.
5. No failure mode may select a per-op or singleton production path.
   Unsupported or unavailable fused execution rejects registration.
6. Runtime values, including symbolic extent values, must never be captured
   as compile-time constants in a stage signature.
7. Exact operations must not silently use approximate algorithms.
8. Production has no semantic fallback tier.
9. Compilation failure must be explicit and diagnosable at registration,
   never at fire time.
10. Core opcodes must remain general and composable; SDK helper names and
    workload policy must not enter canonical bytes or backend dispatch.
11. Host and WorkingSet boundaries must remain boundaries until an explicit
    contract makes them safe to cross.
12. Generated CUDA and Metal must follow the same tie, NaN, RNG, and
    selection semantics as the reference implementation.

## 11. Performance Success Metrics

### 11.1 Primary metrics

- post-forward critical-path time per fire (CPU dispatch plus GPU epilogue
  span), at batch sizes 1, 2, 4, 8, and larger serving buckets;
- PTIR body launches per fire;
- commit/finalization launches per fire;
- grouped lanes per launch;
- scratch bytes materialized;
- channel-copy bytes;
- compiled cache hit rate and registration compile/rejection rate.

Launch count is a proxy; post-forward PTIR overlaps the sampling stream, so
launch reductions must be confirmed on the critical path.

### 11.2 Global targets

- N same-signature instances execute one grouped body per region, not N;
- a mixed-signature batch executes one grouped body per unique ready
  signature per region;
- per-workload target topologies are those of Section 5.3 and the matrix in
  Section 8;
- launch count scales with unique signatures and fused/library regions;
- on Metal, zero CPU-side vocabulary-sized work per decode step once M2
  lands.

### 11.3 Secondary metrics

- end-to-end inter-token latency and tokens per second;
- GPU occupancy and achieved bandwidth;
- compile latency, warm-up behavior, and executable cache memory;
- regression against the test-only singleton oracle for small or unusual
  workloads.

No optimization is accepted solely because it reduces launch count if it
materially worsens end-to-end latency or resource usage.

## 12. First Vertical Slice

The same workload proves each axis once, in isolation:

```text
Phase 1 slice (generated coverage):
  bound epilogue -> singleton region plan -> generated test executable ->
  direct sinks -> atomic commit -> parity vs Rust

Phase 2 slice (fusion):
  same workload -> normalized SSA -> region plan -> one-CTA row fusion ->
  multi-output direct channel stores -> per-lane atomic commit ->
  bitwise parity vs test-only singleton

Metal M1 slice (test oracle):
  same workload -> singleton region plan -> generated MSL ->
  shared-storage channel sinks -> device readiness and commit ->
  parity vs the Rust reference; never production
```

Use masked argmax or Gumbel-max as the workload. It exercises logits input, a
vocabulary-sized elementwise DAG, a reduction, keyed RNG, scalar token
output, loop-carried scalar channel state, grouping across different
instances, direct sinks, and commit.

After these slices are correct, add softmax and exact nucleus sampling rather
than broadening immediately to every PTIR operation.

## 13. Non-Goals

The initial optimizing compiler does not attempt:

- a user-facing scheduling DSL;
- driver-side or heuristic recovery of composites (exact normalized
  recognition is Rust-owned);
- a new readiness or DAG scheduler beside the existing ticket system;
- a handwritten MSL port of the Tier 0 kernels;
- keeping the Metal CPU interpreter as a permanent production tier;
- arbitrary dynamic-rank tensor compilation;
- general polyhedral optimization;
- fusion across unrelated model forwards;
- speculative fusion through host accept/reject logic;
- implicit fusion through WorkingSet fork/COW operations;
- a custom replacement for mature matrix multiplication libraries;
- approximate sampling under an exact PTIR operation;
- persistent kernels before grouped static launches are measured and proven.

## 14. Final North-Star Statement

PTIR should evolve from an operation-by-operation correctness executor into a
stage compiler and grouped execution runtime:

> For every readiness level of a batched forward pass, identify equivalent
> stage programs by canonical signature, execute each unique signature over
> all compatible instances using the minimum legal set of fused kernels and
> semantic library calls, write directly to pending channel state, and
> publish each instance atomically with exact, contract-defined
> cross-backend semantics.

The plan is computed once in Rust; drivers execute plans. Generic SDK helpers
remain ordinary SSA, while Rust alone recognizes exact semantic library
regions. Reusable structured geometry remains explicit and general.
Production on both backends consists only of grouped fused regions and stock
libraries. Singleton generation is a test oracle; handwritten Tier 0 and
interpreters are absent. This changes the dominant cost from
`instances x PTIR ops` to `unique stage signatures x necessary regions`
without a slower production escape hatch.
