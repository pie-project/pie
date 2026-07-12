# Runahead North Star

Status: active plan (2026-07-11, rev 2 after two adversarial review passes).
Supersedes the runahead notes in `direct_ffi_new_plan.md` where they
conflict. Companion to `kv_refact.md` (flattened-table model, which this
plan completes) and `cpp-refact.md` (driver layout, unchanged here).

Rev 2 changes against rev 1, all correctness-driven:

- Channel ordering across retries is preserved by **per-fire sequence
  tickets**. Availability projection is deleted; sequence reservation is
  kept and its role inverted (device-checked expectation instead of
  host-finalized result).
- **RS in-place continuation is a confirmed retry blocker**
  (`runtime/engine/src/pipeline/fire/rs.rs` classifies private state as
  in-place; see the `first_fire_resets_then_continues_in_place` test).
  RS-carrying passes are excluded from retry until RS state is versioned.
- I1 is restated: logical effects publish on commit; pre-commit physical
  writes must be private/versioned or provably retry-idempotent.
- Retried attempts **re-issue their pre-launch copy plans** (KV CoW
  preserves, RS fork copies) so every attempt starts from the committed
  snapshot.
- Dummy rows are demoted to an optional late phase; **narrower batches are
  the default** for straggler waves. A missing pipeline has no fire, so
  there is nothing to retry; padding and retrying are different concepts.
- Late allocation is **dispatch-time host preparation** owned by a
  runtime-side preparation service, not the driver and not the device.
- RETRY gets an **escalation policy** and RETRY rows count as quorum wave
  participation, so permanent causes surface and retrying pipelines are not
  demoted as stragglers.

This document is the implementation plan for the four runahead first
principles. Everything else in the engine exists to support them; where the
current implementation conflicts with a principle, the implementation
changes, not the principle.

---

## 1. First principles

**P1: Runahead submission.** An inferlet can submit its current and future
forward passes ahead of time. Dependencies between future passes are
resolved through channels (which is why inputs are channels). The driver can
enqueue batch n+1 onto the GPU before batch n finishes, with no host
round-trip (vLLM-style asynchronous scheduling).

**P2: Quorum batching.** The batch scheduler assumes the K active pipelines
continuously submit future passes. When K submissions arrive they are packed
into a batch and handed to the driver immediately, regardless of whether the
previous batch has completed.

**P3: Asynchronous host coordination.** Work that inherently needs the CPU
(e.g. constrained-decoding logit masks) is delivered through `channel.put`,
which is asynchronous and never serializes CPU against GPU. The put's
deadline is the moment the consuming pass executes, not the moment the fire
is submitted.

**P4: Invalidate and retry.** If a quorum deadline passes with members
missing, the batch fires without them. If a fire becomes meaningless at
execution time (a required channel is empty when the pass needs it, a
predecessor did not commit, its translation was not ready), that fire
commits nothing and is retried in a later batch. Retry is cheap because a
retried fire re-executes the same step into the same physical resources.

## 2. Unifying invariants

**I1: Commit-bound logical effects.** A fire's validity is decided at
execution time, on the device. Logical effects (channel ring movement,
output publication, KV/RS mapping publication) are published only on
commit. Any pre-commit *physical* write (KV slots, RS slabs written by the
forward itself) must target private, versioned speculative resources or be
provably retry-idempotent. The host schedules and observes; it does not
validate.

**I2: Trusted runahead visibility.** Host bookkeeping trusts its own
runahead: state decided at prepare time is visible to the same pipeline's
subsequent prepares immediately, not after execution proves it. Speculative
state is scoped to the pipeline that created it; other pipelines, the CAS
prefix index, and reclaim accounting see committed state only.

**I3: Retry identity.** A retried fire re-executes the same step, with the
same inputs, into the same physical resources, under the same channel
sequence tickets. Two mechanical requirements make this true rather than
aspirational:

- Every attempt re-issues the fire's pre-launch copy plans (KV CoW
  preserves, RS fork copies). Sources are committed state, untouched by the
  failed attempt, so re-copies restore the snapshot.
- Every physical write the forward performs must be an absolute
  (last-writer-wins) write into a fire-private destination. KV slot writes
  satisfy this. In-place RS continuation does not (a re-executed recurrence
  applies twice), which is why RS-carrying passes are retry-ineligible
  until RS is versioned (Section 4.2).

Given I3, anything prepared against a pending fire's effects remains valid
across its retries. Only a hard failure (poison) invalidates downstream
state, and a hard failure kills the whole pipeline, so there is nothing
downstream left to be wrong. I3 is what makes I2 safe: the store's current
"publish only at finalize" policy exists to protect readers from aborted
state being replaced by different content, and retry identity removes that
possibility for benign failures while poison removes the reader for hard
ones.

## 3. Audit: where the implementation stands

| Principle | Status | Blocking defects |
|---|---|---|
| P1 | Scheduler and driver are ready (non-blocking submit, enqueue-only host path, 2 batches in flight). Defeated by the runtime KV gate. | KV drain loop `runtime/engine/src/pipeline/fire.rs:337` awaits the previous fire's GPU completion on every same-pass decode submit (token-granular; the in-code "page boundary" comment does not match the implementation). Device-geometry fires additionally force a full stream drain (`resolve_descriptors`, `driver/cuda/src/pipeline/dispatch.cu`). Runahead depth hardcoded to 2; `PIE_SCHED_MAX_IN_FLIGHT` documented and set by tests but never read (`runtime/engine/src/scheduler/quorum.rs:47`). |
| P2 | Implemented (`WaitAllPolicy`: all-ready quorum, 500us cold hold, 10ms straggler deadline, miss-limit demotion). | `WaveDecision::Fire { missing }` is discarded at `runtime/engine/src/scheduler/worker.rs:775`; a deadline fire launches a narrower batch, which is acceptable (Section 4.5), but wave accounting for retried rows does not exist yet. |
| P3 | Put side done: `channel.put` writes the pinned SPSC ring directly, release-publishes the tail, never touches the scheduler (`runtime/engine/src/pipeline/channel.rs`). Guest `take`/`read` are already live-cursor loops that race the oldest in-flight op (`runtime/engine/src/inferlet/host/forward.rs`), so the guest-side wait discipline needs no redesign. | Consume side is eager: `validate_channel_budget` rejects empty writer rings at submit (`dispatch.cu:170`), and `pull_writer_inputs` is a host-enqueued memcpy that reads ring state at enqueue time (`program_runtime.hpp:202`). The value's deadline is submit, not execution. Runahead makes this worse: with depth D, the mask for step n must be ready before step n-D completes. |
| P4 | Not implemented. | Terminal outcomes are binary; a non-committing pass poisons every host-visible channel and permanently fails the pass. Host ring cursors advance optimistically with no reconciliation. `PendingFire` retains no payload; `finalize_fire` commits or aborts unconditionally on completion. RS private state continues in place, so naive re-execution corrupts it. |

The KV drain loop deserves one paragraph because it is the single worst
defect. `submit_pass` compares the pass's optimistic cursor against
`committed_token_len()` (`runtime/engine/src/store/kv.rs:571`), which counts
only finalized slot hashes. The cursor is always ahead by the in-flight
fire's tokens, so the condition is true on every runahead decode submit, at
token granularity, regardless of page boundaries. The loop then pops the
previous `PendingFire` and awaits its GPU completion. Combined with
wait-all quorum, every pipeline's next submit waits for its previous fire,
so wave n+1 cannot assemble until batch n completes: a GPU bubble every
step, engine-wide.

## 4. Target architecture

### 4.1 KV plane: WS-relative wire, dispatch-time translation (Option 2)

The decided end state. Physical page ids disappear from the submission path
entirely; the per-fire translation segment becomes the only place physical
ids exist. This completes the `kv_refact.md` flattened-table direction (the
translation-segment overlay in `dispatch.cu` is the seed of it).

- **Wire geometry is WorkingSet-relative.** The launch plan carries
  WS-relative page indexes and write slots, derived by pure cursor
  arithmetic. Submit reads *no predecessor pending state*. Committed-only
  reads remain legal at submit: the prefix-cache probe (`match_prefix`)
  reads the committed CAS and trims the launch, and must keep running
  before wave-size accounting since the trim changes wire geometry.
- **Allocation deadline moves from submit to dispatch.** Physical pages for
  a fire must exist when its translation segment is built, at batch
  composition on the host ("dispatch-time"; the device never allocates).
  Allocation stays lazy: pre-allocation is rejected because it wastes pool
  under K pipelines and cannot anticipate CoW or prefix-cache adoption.
- **A late-prepare transaction owner.** Moving allocation to dispatch is
  not a formatting change: logical reservation, in-place/CoW/fresh
  classification, physical allocation, copy-plan construction, hash
  metadata, `KvTxn` ownership, OutOfPages policy, and the commit/abort
  lifecycle move together. They land in a preparation service at the
  runtime/scheduler boundary. The driver does not gain Rust transaction
  ownership; it stays a consumer of finished translation segments.
- **Translation reads the pending overlay.** The store exposes, per working
  set, the composed view `committed mapping ⊕ pending txns in FIFO order`,
  including pending hash-chain state (`KvPreparedWrite` carries none today;
  this is real new state, not bookkeeping). Translation segments build from
  that view. The overlay is pipeline-scoped (I2). Overlay entry lifecycle:
  added at prepare, merged into the table at commit, removed at abort;
  pass drop drains the FIFO in order, so entries retire in order.
- **Unready translation is not an error.** A fire whose translation cannot
  be completed because a predecessor has not prepared is a RETRY row
  (Section 4.4). OutOfPages is different: the preparation leaves the hot
  rotation as BLOCKED, the contention ladder owns its reservation-backed
  wait, and a grant or terminal error nudges it back into preparation.
  Blocked time does not consume `PIE_FIRE_RETRY_MAX`.
- **Admission bound.** Pending fires consume real pages before they
  commit. Runahead depth bounds this today (depth x pages per fire);
  deeper depths require an explicit allocation credit so pending fires
  cannot over-commit the pool.
- **Retry identity for KV (I3).** A retried fire writes the same WS slots
  through the same translation into the same physical pages, after
  re-issuing its CoW preserve copies. Pending mappings are stable across
  retries; downstream translations survive. Nothing unwinds on benign
  invalidation.
- **Hard failure unwinds with epochs.** On poison, pending physical pages
  return through epoch-deferred recycling (the `recycle_after_epoch`
  mechanism already used for cache roots) so pages are not reallocated
  while a downstream in-flight fire still references them.
- **Hash-chain integrity.** Canonical slot-hash chains chain from pending
  commits (the values are host-known at prepare and stable under I3), or
  demote to opaque while runahead is outstanding. The CAS prefix index
  only ever indexes committed pages; both options preserve
  "hash == content".
- **Drain loop deleted.** `fire.rs:337-346` and the token-granular
  `committed_token_len` gate go away (flag-gated in Phase 1). The
  context-coverage check inside `kv::prepare` reads the overlay view; the
  cross-pass cursor floor (`fire.rs:324`) reads the overlay for
  same-pipeline handoff and committed state otherwise.

### 4.2 Speculative physical state: what the forward writes before commit

The tier-0 commit predicate runs after the forward, and the forward has
already written KV slots and RS slabs by then. I1 requires those writes to
be retry-idempotent or private-versioned. Case analysis:

- **KV slot writes: idempotent.** A slot write is an absolute write of the
  same content (same input tokens, same position) on every attempt, into
  pages reserved for this fire. Safe under re-execution, provided the CoW
  preserve copies are re-issued per attempt (otherwise preserved slots
  would carry the failed attempt's residue on pages that also hold
  preserved committed content).
- **RS in-place continuation: NOT idempotent (confirmed).**
  `pipeline/fire/rs.rs` classifies private recurrent state as in-place
  continuation in the same physical slab. Re-executing the forward applies
  the recurrence twice: `state2 = F(F(state0, x), x)`. Therefore:
  - Near term: **RS-carrying passes are retry-ineligible.** They keep
    eager channel validation and fail-hard semantics (any non-commit is
    FAILED). The scheduler must know pass retryability; it is a static
    property of the program (does it bind an RS working set).
  - Long term (separate track): versioned RS. Every fire's RS update
    becomes CoW-continue into a distinct destination slab with the fork
    copy re-issued per attempt, restoring retry eligibility. The current
    shared-fork CoW path already has the copy-plan machinery; the change
    extends it to private state.
  - Rejected for now: replaying only the epilogue from parked logits. It
    breaks pass atomicity (KV/RS would commit while the pass retries) and
    adds a logits-parking lifetime problem. Revisit only if retried
    forwards show up in profiles.
- **Copy re-issue rule (general).** A requeued attempt re-issues every
  pre-launch `copy_d2d` plan attached to the fire (KV CoW preserves, RS
  fork copies; both are issued at submit today, `fire.rs:475`,
  `fire.rs:512`). Sources are committed state, unchanged by the failed
  attempt, so re-copies are idempotent and restore the attempt-0 snapshot.

### 4.3 Channel plane: sequence tickets, device-owned actuals, lazy pull

The channel accounting splits into two pieces with different owners. The
**sequence reservation** (which ring entry belongs to which fire) is
assigned by the host at submit and is immutable. The **actual state**
(head, tail, poison words) is owned by the device and observed by the
host. Availability is not checked anywhere before execution.

- **Tickets.** At submit, each channel binding of a fire receives its
  expected ring position: consumers get `expected_head` (the next
  unassigned consume sequence for that ring, monotonic per channel),
  producers get `expected_tail` likewise. Tickets are assigned once and
  survive retries (I3). They are the former precomputed target epochs with
  the role inverted: an expectation the device verifies at commit, not a
  result the host finalizes. The assignment plumbing survives; the
  host-side finalize-by-target plumbing goes.
- **Commit preconditions (device-checked).** For every consumed binding:
  `actual_head == expected_head` (every predecessor consumed its own
  entry) and `actual_tail > expected_head` (this fire's entry has
  arrived). For every published binding: `actual_tail == expected_tail`
  and reader capacity is available. Any precondition failure means the
  pass does not commit and the fire is RETRY. This closes the reordering
  hole: if fire N retried, its entry arriving late cannot be stolen by
  fire N+1, whose `expected_head` no longer matches; the cascade is
  ordered by construction, for shared writer streams as well as
  producer-consumer chains.
- **Backpressure is a RETRY trigger, not a validation error.** The
  reader-capacity half of the deleted `validate_channel_budget` reappears
  as the publish precondition above: no output capacity at commit means no
  commit.
- **Device-side pull-and-validate.** `pull_writer_inputs` (host-enqueued
  memcpy reading ring state at enqueue time) is replaced by a kernel that
  runs stream-ordered immediately before the tier-0 pass: system-scope
  acquire read of the pinned tail word, ticket precondition check, copy of
  the entry into the device cell (including the packed-bool unpack that
  the host staging path does today), and a validity predicate feeding the
  existing predicated-commit machinery (`tier0_runner.hpp`). It **never**
  blocks the stream: a late mask costs one retried row, not a stall of
  every batch behind it. Because the tier-0 pass is the post-forward
  epilogue, "a channel used only in the epilogue is checked only at the
  epilogue" holds by construction. The eager pull path is deleted; the
  seed path (`channel::from` initial values applied at bind) stays.
- **Output publication.** Destinations are deterministic given tickets, so
  the host still enqueues the D2H output copies targeting the ticket cell;
  the tail word release-publish is predicated on the commit flag
  read-back. An uncommitted attempt may copy garbage into its ticket cell;
  the unpublished tail keeps it invisible and the retry overwrites it.
- **Memory-ordering contract.** The pinned rings are UVA device-accessible
  as allocated (no `cudaHostAllocMapped` rework needed on the supported
  64-bit Linux targets). What is required is the explicit contract on the
  head/tail/poison words: device system-scope atomics (or
  `__threadfence_system` before the releasing store), host
  `atomic_ref` acquire loads. State it once, assert it in review.
- **Callback reads actuals.** `notify_runtime_callback` stops finalizing
  precomputed targets and derives notifications from the words the pass
  actually produced. Guest `take`/`read` already loop on a live cursor and
  race the oldest in-flight op, so the only guest-side change is
  classification: a RETRY completion drains without being treated as
  failure and without popping the logical fire.

### 4.4 Outcome plane: three-state completion and the LogicalFire machine

Terminal outcomes become COMMITTED / RETRY / FAILED. A new ABI terminal
outcome (`PIE_TERMINAL_OUTCOME_RETRY`) is added, distinct from the existing
`Invalid` (malformed/stale submission, still a synchronous reject). The
change spans the ABI constant and validation, the Rust `TerminalOutcome`,
the dummy driver, the CUDA callback, completion plumbing, and scheduler and
pipeline finalization; it is an end-to-end state-machine change, not a
constant.

| Outcome | Trigger | Channel effects | KV/RS effects | Scheduler action |
|---|---|---|---|---|
| COMMITTED | all commit preconditions pass | consumes at `expected_head`, publishes at `expected_tail` (device-side, predicated) | txn finalizes, mapping publishes | retire fire, guest FIFO drains |
| RETRY | required entry absent; ticket mismatch (predecessor retried); no output capacity; non-memory translation dependency unready | none (rings untouched) | none (pending mapping stays, pages retained) | requeue the same LogicalFire; re-issue copy plans; count as wave participation |
| BLOCKED | KV allocation awaits a contention grant or requester restore | none (rings untouched) | no prepared txn; concrete grant is reserved for one process/request | remove from retry rotation; nudge and reinsert on grant/restore/error |
| FAILED | `pass.ok == false`, driver error, launch exception, escalation threshold | poison all host-visible channels | txn aborts, epoch-deferred page recycle | fail the pass, poison readers, pipeline terminates |

The runtime-side identity that makes RETRY coherent:

```
LogicalFire (identity stable across attempts)
  fire_id            immutable
  channel tickets    expected positions per binding, assigned at submit
  logical geometry   WS-relative, cursor-derived
  KV/RS txns         retained across attempts, finalized exactly once
  copy plans         re-issued per attempt
  attempts           each with its own driver completion
  retry_count        escalation input
```

Rules: the guest-visible completion resolves only on COMMITTED or FAILED. A
RETRY attempt neither finalizes the txns nor pops the pipeline FIFO; the
fire stays at the front and re-enters the next wave with the same identity
and tickets. `drain_settled` classifies RETRY attempts as progress, not
poison.

**Escalation.** Blind next-wave retry is the default: the wave fires anyway
for the other pipelines, and per-batch instance dedup
(`worker.rs:989`, a load-bearing rule) staggers a retrying chain to one row
per wave, so the marginal cost is that row's forward. But permanent causes
must surface: today a missing put is a synchronous submit error the guest
sees immediately; it must not become a silent infinite retry. Policy hook
on `retry_count`: warn at N, surface FAILED at M or on a classified
non-transient cause (OutOfPages with no reclaim progress). Gating a requeue
on put-arrival (the runtime observes every put) is an optional refinement
against always-late producers. RETRY rows count as quorum wave
participation so a retrying pipeline is never miss-counted toward straggler
demotion.

### 4.5 Scheduler plane: narrower batches, requeue, dummy rows deferred

- **Straggler waves fire narrower.** This is current behavior and it is
  correct: P4's intent is that missing members must not block the batch,
  and a narrower batch satisfies it on the dynamic-shape single-GPU path.
  `WaveDecision::Fire { missing }` needs wave-accounting wiring, not row
  fabrication.
- **Dummy rows are deferred** until graph-shape stability needs them (TP
  graph lattice, graph replay). They are padding, not retryable work: a
  missing pipeline has no payload, no txns, no terminal cell. When built,
  they must be zero-work rows or write only to dedicated scratch KV/RS
  storage. Page 0 is not acceptable as a write target: the existing page-0
  precedent covers masked-out *reads*; a dummy row's forward *writes* KV,
  and page 0 is a real owned page.
- **Requeue** is Section 4.4's LogicalFire machinery; the scheduler keeps
  payloads until COMMITTED/FAILED and re-enters RETRY fires into the next
  wave.
- **Depth configuration.** Wire `PIE_SCHED_MAX_IN_FLIGHT`
  (`quorum.rs:47` returns the hardcoded default today; tests already set
  the variable and silently run depth 2). Depths above 2 stay locked until
  the retry machinery and the admission bound (Section 4.1) exist.

### 4.6 Driver plane: what changes, what stays dumb

Gains: the pull-and-validate kernel with ticket preconditions, device-owned
ring words, predicated output publication, the RETRY outcome. Losses (all
deletions): availability validation (`validate_channel_budget`,
`writer_inputs_available`), availability projections
(`reserved_head_`/`reserved_tail_`, `project_fire_success`), the
finalize-by-precomputed-target plumbing. Unchanged: ticket assignment
arrives from the host with the launch (the driver checks, never assigns);
the predicated-commit machinery (it is the foundation); the single
execution stream (back-to-back enqueue satisfies P1; concurrent batch
execution is a non-goal); the seed path; and the principle that policy
(retry, escalation, pacing, wave shape) lives in the runtime while the
driver reports what happened.

Device-geometry fires keep host-side geometry resolution for now, but a
wave member whose geometry channels are empty at composition is excluded
from that wave (straggler treatment) instead of rejecting the launch. Note
this is a compose-path restructure (today one member's miss throws for the
whole batch), not a one-line failure-mode change. Eliminating the
`resolve_descriptors` full-stream drain by resolving geometry without host
readback is a separate long-term track; it does not block P1 for
wire-geometry programs.

## 5. Phases

Each phase lands green. Phase 1 is the value milestone (runahead decode
works); phases 2-3 make P3/P4 real; phases 4-5 finish the end state.

### Phase 0: truthful knobs
- Wire `PIE_SCHED_MAX_IN_FLIGHT` in `quorum.rs`; assert the configured
  depth in the runahead test. Keep the effective cap at 2 until Phase 3
  lands.
- Exit: tests that set the variable test what they claim.

### Phase 1: KV visibility (pending overlay, drain removal behind a flag)
- Per-WS pending-txn overlay (FIFO order, pipeline-scoped) including
  pending hash-chain state; `kv::prepare` context projection,
  `build_translation`, and the cursor floor read it.
- Remove the drain loop behind a flag; flip after overlay/committed
  equivalence verification (identical projections, translations, and
  chains across the existing prefix-cache and CoW suites).
- Hard-fail unwind through epoch-deferred recycling.
- Safety argument for landing before retry exists: submit-time channel
  validation is untouched and RETRY does not exist, so fires cannot
  reorder; the only abort is poison, which kills every downstream reader
  in the pipeline.
- Exit: same-pass decode submit(n+1) never awaits fire n's completion
  (assert via `PIE_FIRE_TIMING`); wave n+1 assembles while batch n is on
  the GPU; prefix-cache and CoW suites green with the flag on.

### Phase 2: outcome plumbing and the LogicalFire machine (test-triggered)
- `PIE_TERMINAL_OUTCOME_RETRY` end to end: ABI + validation, Rust
  `TerminalOutcome`, dummy driver, CUDA callback (commit flag 0 with no
  error maps to RETRY: publish, consume, and poison nothing), completion
  plumbing, scheduler requeue.
- LogicalFire: retained payload and txns, ticket assignment (recorded, not
  yet device-enforced), per-attempt completions, FIFO-front retention,
  copy-plan re-issue per attempt, `drain_settled` classification.
- RS-carrying passes marked retry-ineligible (static program property).
- Escalation counters, wave-participation accounting for RETRY rows.
- Trigger via a test hook forcing commit = 0 (no real lazy trigger exists
  yet). Verify at depth 1, then depth 2: rings and KV/RS state
  bit-identical across attempts; the retried attempt commits identically
  to a never-failed run.

### Phase 3: channel plane (device-authoritative, lazy, ticket-enforced)
- Delete submit-time writer validation and availability projections; keep
  ticket assignment.
- Pull-and-validate kernel: system-scope acquire tail read, ticket
  preconditions, pinned-to-device cell copy with packed-bool unpack,
  validity predicate into predicated commit; never blocks the stream.
- Output side: capacity and ticket preconditions at commit; host D2H
  copies target ticket cells; tail release-publish predicated on the
  commit read-back. Callback derives notifications from device actuals.
- Memory-ordering contract stated and reviewed (system-scope
  atomics/fences on the pinned words).
- Exit: constrained-decoding e2e with the put landing after submit and
  before the epilogue (overlap realized); a withheld put yields RETRY then
  commits on the wave after the put; the ticket-ordering test passes (fire
  N retries, its entry then arrives, fire N+1 must RETRY rather than
  consume it); logit-mask channels verified not to ride the dense-AttnMask
  solo-batch rule (`fire.rs:294`); real errors still poison.

### Phase 4: WS-relative wire and the dispatch-time preparation service
- Preparation service at the runtime/scheduler boundary owns logical
  reservation, classification, allocation, copy plans, hash metadata,
  `KvTxn` lifecycle, and OutOfPages policy at dispatch time. The driver
  consumes finished translation segments.
- Submission carries WS-relative geometry only; unready translation maps
  to RETRY; OutOfPages routes through the contention ladder and becomes an
  event-driven BLOCKED preparation until a concrete grant or terminal error.
- Allocation credit bounds pending-fire pool usage before deeper depths
  unlock.
- `match_prefix` stays a submit-time committed-only read; the trim
  precedes wave-size accounting.
- Exit: submit reads no predecessor pending state (committed-only reads
  remain); a fire prepared before its predecessor's allocation launches
  correctly through the service.

### Phase 5: optional and long-term
- Versioned RS (restores retry eligibility for hybrid/linear-attention
  models); until then RS passes stay fail-hard.
- Dummy rows only if graph-shape stability requires them, under the
  Section 4.5 safety requirements.
- Device-geometry per-member exclusion (compose restructure), then
  device-resolved geometry (kill the `resolve_descriptors` stream drain).
- Deeper depths (>2) with the admission bound; input double-buffering so
  batch n+1's H2D overlaps batch n's compute; revisit the per-instance
  `publish_done` ping-pong (`dispatch.cu:494`) and the single-slot commit
  snapshot once retry exists.

## 6. Verification focus

- **Overlap, not just correctness:** timing assertions (`PIE_FIRE_TIMING`)
  that submit(n+1) completes while fire n is in flight, and that GPU idle
  between consecutive waves is bounded by composition cost, not completion
  round-trips.
- **Overlay equivalence:** with the drain flag off vs on, projections,
  translations, and hash chains are identical across the prefix-cache and
  CoW suites.
- **Retry identity:** withheld put, late put, unready translation; each
  yields RETRY with device rings and KV/RS mappings bit-identical to the
  pre-fire state (including re-issued copy plans), then commits
  identically on retry.
- **Ticket ordering:** the reordering scenario (N retries, N's entry
  arrives, N+1 executes first) must end with N+1 in RETRY and N's retry
  consuming its own entry. Also: two fires in one batch sharing a channel
  where the first retries; notifications reflect device actuals with no
  off-by-one.
- **RS exclusion:** an RS-carrying pass never takes the RETRY path; its
  non-commit is FAILED.
- **Escalation:** a guest that never puts sees a diagnostic and eventually
  FAILED, not a silent livelock; a retrying pipeline is never demoted as a
  straggler.
- **Integrity under runahead:** the CAS never indexes pending content;
  canonical chains produced under runahead match the chains the same
  tokens produce synchronously.
- **Poison still works:** hard failures terminate the pipeline and free
  pages only after downstream in-flight fires drain.

## 7. Non-goals

- Concurrent multi-stream batch execution (back-to-back on one stream
  satisfies P1; revisit only if profiling shows composition gaps).
- Upfront or speculative physical page allocation (lazy allocation is
  correct; visibility was the problem, and dispatch-time preparation
  removes even that dependency from submit).
- A device-side allocator. The device checks preconditions and owns ring
  actuals; it never allocates or assigns sequence positions.
- Epilogue-only replay from parked logits (breaks pass atomicity; revisit
  only with profile evidence).
- Guest-visible API changes: the inferlet contract (submit runahead, await
  on take/read, put anytime) is already the right one; this plan makes the
  engine honor it.
