# Direct Channel Plane Refactor

## Status

**Implemented (2026-07-10).** Phases 0-3 are landed: scheduler nudge +
callback contract + CoW-on-FIFO (Phase 0), direct reader wake + word-gated
visibility + finalize decoupling (Phase 1), ring puts + driver pull +
availability check + ABI v2 (Phase 2), gate tests + grep gates (Phase 3).

§14 gate coverage: 1 (ABI fields deleted; dummy/CUDA pull tests), 2/3
(`parked_reader_wakes_straight_from_the_driver_callback`,
`extern_export_flows_into_importing_instance`), 4/5
(`writer_ring_backpressure_wakes_after_a_consuming_fire`), 6
(`completion_retirement_is_event_driven` + the §16.2 backstop wake-class
counter asserting zero), 7 (`parked_reader_wakes_into_poisoned_not_empty`),
8 (frame.rs register-then-recheck tests), 9 (`test_ptir_dispatch_race`:
96 same-instance run-ahead fire pairs + registry growth + instance churn,
TSAN-instrumented run on an RTX 4090 with zero reports, CUDA-runtime
suppressions only), grep gates clean. Gate 10 (pin-float bound) is
covered by construction (`drain_settled` at submit entry + the forward_txn
suite); a full guest-driven test waits on the inferlet SDK restoration.

Follow-up hardening sweep (same day): batch-level channel budgeting — one
launch's members sharing a channel are validated against the AGGREGATE of
their planned ring consumes and reader publishes (`validate_channel_budget`),
closing the per-instance blind spot on both the CUDA and dummy drivers;
`publish_reader_mirror` and the second reader cursor deleted (the released
tail word is the only gate; `latest_reader_value` reads it directly); dead
carrier code removed (`PtirProgramSubmission`, `bind_seeds_first_fire`,
`read_channel_ids`); nvcc excluded from the sccache auto-launcher; orphaned
sampling-ir test targets removed and `pack_dense_mask` updated to the
[TOTAL_Q, STRIDE] signature.

Validated: pie-engine 209 lib tests + integration suites (remaining failures
are the pre-existing guest-SDK breakage, identical at the pre-change
baseline); dummy/worker/abi/waker/ptir suites; the full driver/cuda ctest
suite (30/30, including dispatch-race and golden-exec) on an RTX 4090; Metal
stub test.

Second sweep (same day): the deferred micro-optimizations landed — bool cells
pack straight into the pinned ring (`pack_bool_into`, no intermediate
allocation), reader cells decode by move instead of a second copy
(`decode_reader_cell` takes ownership), and a prebuilt single-request batch
moves its `LaunchPlan` into the submission instead of cloning it;
`arm_completion_nudge` was assessed and left alone (one registration per
scheduler block is already minimal, a cache only adds a lost-wake hazard).
Metal gained its Phase-3 real-execution increment: `launch` executes
channel-plane PTIR programs on a host interpreter
(driver/metal/src/ptir/host_interp.hpp, a C++ mirror of the canonical
interface/ptir interp over the shared pure-host decode headers under
driver/cuda/src/ptir) and publishes per §4.3/§4.4 — writer-ring pull with
batch-aggregated availability, seed credit, reader tail/poison publication,
per-channel wakes, terminals, and the batch notify last. Intrinsic-,
host-input-, per-layer-, and kernel-call programs still reject UNSUPPORTED
until the Metal forward is wired. The stub test was rewritten to the
execution contract (put→launch→take, availability rejection, poison
settlement, seed credit; ctest green) and the worker links with the embedded
driver. The guest inferlet SDK breakage is being closed by migrating the
stale guests to `inferlet::ptir` (the SDK itself was never broken — the old
`sampling`/`emit`/classic-ForwardPass surface was deliberately deleted);
migrated so far: generate, lowlevel-chat, specverify, mtpverify (+
no_context WIT-path fix, a dummy-driver `LaunchObserver` seam restoring
mock fire observation, and wasip2 targets for ptir guests — the wasip3 std
pulls a `wasi:random` rc version wasmtime does not link).

Successor plan to [direct_ffi.md](direct_ffi.md) and [direct_ffi_fix.md](direct_ffi_fix.md)
for the channel data plane and the wake paths. The direct FFI transport
migration those documents drove is done and is not reopened here. This plan
rewrites what still contradicts the target architecture:

- Host channel puts are staged host-side and shipped inside launch
  descriptors instead of being written directly into shared channel memory.
- Channel reads wake through the batch completion, the scheduler's 5 ms
  poll, and the pipeline FIFO instead of through the channel's own wait slot.
- The completion callback mutates driver instance/registry state concurrently
  with the scheduler thread without synchronization.

Settled decisions D1-D5 of direct_ffi_fix.md (exact terminal outcomes, atomic
batch acceptance, global channel endpoints, pipeline failure domains, stable
typed pools) remain normative and are not restated. This plan supersedes
direct_ffi_fix.md §1.7 (channel-specific waiting) and is the concrete
realization of direct_ffi.md §4.4 (channels are the only output surface) and
§7.4 (channel head/tail changes use the same wait-slot mechanism).

## 1. Goal

Three invariants define the final shape:

1. **Submit is direct and nothing polls.** Inferlet submissions flow through
   the per-driver batching scheduler into a direct FFI call. The driver
   receives direct calls; the runtime receives completion callbacks that wake
   the exact waiter. No component discovers progress by timeout.

2. **A host put is a shared-memory write.** `channel.put` writes the cell
   into the channel's pinned ring and release-publishes the tail word.
   Puts are independent of pipelines and submissions. The ring holds
   `capacity + 1` cells; the spare slot distinguishes full from empty and
   holds the not-yet-committed producer cell.

3. **Driver output wakes the channel directly.** During a forward pass, PTIR
   programs update device channel state freely. At pass completion the driver
   publishes committed reader cells into the pinned mirror, release-publishes
   each channel's tail word, and notifies that channel's reader wait slot.
   `channel.take().await` parks on the channel wait slot and wakes without
   scheduler involvement and without draining a pipeline FIFO.

```text
inferlet put ──────────────► pinned writer ring ──► driver pull ──► device cell
inferlet submit ──► scheduler batch ──► direct FFI launch ──► GPU
GPU epilogue ChanPut ──► DMA to pinned mirror ──► tail word ──► notify(reader_wait_id)
                                                              └► inferlet take().await wakes
batch terminal cells ──► notify(batch wait_id) ──► scheduler nudge ──► retire txns
```

## 2. What Stays

- One queue: the scheduler thread exclusively owns the `NativeDriver` and is
  the ordering point for launches and typed control operations
  (direct_ffi.md §4.1).
- Typed operations, terminal cells, atomic batch acceptance, the epoch
  discipline, and the generation-tagged `WakerTable`.
- The pinned mirror plus four-word (`head`, `tail`, `poison`, `closed`)
  channel endpoint layout of `PieChannelEndpointBinding`. This plan
  reinterprets it symmetrically; it does not change it.
- The publication order contract of direct_ffi.md §7.1: channel state, then
  member terminal entries, then release fence, then the batch notify.
- Runtime-side reader capacity reservations at submit
  (`reserve_reader_capacity`) and driver-side `can_publish` validation.
- The same-pipeline constraint (B3 FIFO invariant) for fires that share
  channels. It remains an ordering rule for submissions; it stops being the
  wait mechanism for values.

## 3. Current State vs Target

| Flow | Current | Target |
|---|---|---|
| Host put | Staged in `ChannelCell::staged` (host heap), shipped per fire as `PieLaunchDesc::ptir_host_put_values`, driver `host_feed` copies at fire time | Written into the pinned writer ring at put time; driver pulls ring entries stream-ordered before the pass |
| Reader visibility | Mirror tail gated through `publish_reader_mirror` at fire finalize | `take` reads the tail word directly; visibility is the release-published word |
| Reader wake | Batch `wait_id` only; channel `reader_wait_id`/`writer_wait_id` ignored by CUDA and Metal | Callback notifies `reader_wait_id` with the new tail per touched channel |
| take/read progress | Drains the pipeline FIFO, awaits `InstanceCompletion`, which resolves only after scheduler retire | Parks on the channel wait slot; wakes straight from the driver callback |
| Scheduler retire | `recv_timeout(5ms)` poll; completion wake is lost (no registered waiter) | Nudge waker: completion wake enqueues a scheduler message |
| Callback thread | Mutates `Impl::instances`, registry vectors, `fire_seq`, `apply_fire_result` unlocked against the scheduler thread | Callback touches only precomputed pinned word pointers, terminal cells, and `notify` |
| Cross-instance channels | A task that did not submit the producing fire cannot wait (`Empty` error) | Any task parks on the channel wait slot; extern import/export works |

## 4. Channel Plane Specification

### 4.1 Ring layout, both directions

Every host-visible channel endpoint is one SPSC ring over the existing
binding:

```text
words[0] = head    consumer progress, monotonic u64
words[1] = tail    producer progress, monotonic u64
words[2] = poison  nonzero = failed epoch
words[3] = closed  nonzero = endpoint closed
cell slot for sequence s = s % cap1,   cap1 = capacity + 1
```

- **Reader channel** (device produces, host consumes): the driver writes
  mirror cells and `tail`; the host `take` reads cells and advances `head`.
  This is exactly today's reader path; only the wake and the visibility gate
  change.
- **Writer channel** (host produces, device consumes): the host writes mirror
  cells and `tail`; the driver pulls entries into device cells and publishes
  `head` after the consuming fire commits. This direction is new; the mirror
  and words are already allocated for writer channels and currently unused
  ([channel_registry.hpp](driver/cuda/src/ptir/channel_registry.hpp) allocates
  them for every role).

Word updates are release stores; word reads on the opposite side are acquire
loads. Cell bytes are written before the tail store that publishes them.

### 4.2 Host put (direct write)

```text
put(bytes):
  head = acquire_load(words[head])
  if tail_local - head >= capacity: return Full
  write wire bytes (bool packed) at mirror[tail_local % cap1]
  release_store(words[tail], tail_local + 1); tail_local += 1
```

- `tail_local` is the runtime-side monotonic producer cursor (today's
  `writer_tail` in [ptir_channel_store.rs](runtime/engine/src/ptir/ptir_channel_store.rs)).
- `Full` remains a synchronous error. The driver notifies `writer_wait_id`
  when `head` advances (§4.4), so an awaitable `wait-writable` can be added to
  the WIT surface later without further driver changes. It is not required by
  this plan.
- **Pre-endpoint staging.** A guest may put before the channel is bound into
  any forward pass, and the native endpoint registers lazily on first bind.
  Until the endpoint exists, puts stage host-side exactly as today; the
  runtime flushes the staged FIFO into the ring when `attach_endpoint` runs.
  After the flush, puts write directly. Steady state has no staging.

### 4.3 Driver pull (writer channels)

At launch preparation, on the scheduler thread, for each host-writer channel
bound by a launched instance:

1. `tail = acquire_load(words[tail])`.
2. For every sequence in `consumed_ .. tail` not yet pulled, enqueue an H2D
   copy from `mirror[s % cap1]` into the device cell ring at the matching
   index and set the device full bit, on the execution stream, so the copies
   are stream-ordered before the pass. Advance the driver-side `consumed_`
   cursor (scheduler-thread state; today's `reserved_head_`).
3. `schedule_host_consume`/`finalize_host_consume` keep their exact roles:
   the pull reserves the head target, the completion callback publishes
   `words[head]` on success or `words[poison]` on failure.
4. After `words[head]` is published, notify `writer_wait_id` with the new
   head (space became available).

**Availability check replaces payload validation.** Today a fire cannot miss
its input because the value rides the descriptor. In the ring model a
missing put is a guest ordering bug (put happens-before submit in guest
program order, and the put is immediately visible). `validate_launch`
therefore checks, per launched instance, that every host-writer channel its
trace `ChanTake`s has `tail - consumed_ >= takes_per_fire`, and rejects the
launch synchronously with `PIE_STATUS_INVALID_ARGUMENT` otherwise. Device-side
non-commit poison stops being the discovery path for missing inputs.

`validate_host_puts`, `feed_host_puts`, and the descriptor-driven `host_feed`
path are deleted.

### 4.4 Reader publish and per-channel wake

The device-to-host publication pipeline is unchanged: predicted committed
cells DMA into the pinned mirror on the copy stream, the conditional consume
kernel and the commit-flag D2H follow, then `cudaLaunchHostFunc` runs the
completion callback. The callback's obligations become, in order:

1. For each touched reader channel: `finalize_host_publish` release-stores
   `words[tail]` (or `words[poison]` on failure), then
   `notify(reader_wait_id, new_tail)`.
2. For each touched writer channel: `finalize_host_consume` release-stores
   `words[head]` (or poison), then `notify(writer_wait_id, new_head)`.
3. For each launch member: release-store the terminal cell.
4. Last: `notify(batch_wait_id, target_epoch)` exactly once.

One callback function, as in direct_ffi.md §7.4; the wait id identifies the
event class on the Rust side. Channel notifies use the channel's tail/head
value as the epoch, which is monotonic and composes with the existing
epoch-filtered `publish`: a waiter registered with `observed = last seen
tail` is woken only by genuinely new data.

### 4.5 take and read

```text
take():
  loop:
    tail   = acquire_load(words[tail]);  check poison/closed
    if copied < tail: copy + decode mirror cells, advance copied
    if a decoded cell exists: advance head word (take only); return it
    register waker on reader_wait_id with observed = tail
    re-check (register-then-recheck); park
```

- No pipeline FIFO involvement. Any task, including one that never submitted
  a fire, can wait on any channel it holds. Extern import channels work.
- `publish_reader_mirror`'s role as a visibility gate is deleted; the word is
  the gate. Poison and closed words keep surfacing as `ChannelError`.
- `read` is the same loop without the head advance.
- The host `head` advance on take needs no notify to the driver: the driver
  reads the head word on the scheduler thread when it validates
  `can_publish` for the next launch.

### 4.6 Seeds

Seed values stay in `PieInstanceDesc::seed_values` (cold path, one-shot,
covers device-private seeded channels that have no host role). A seeded
Writer channel's seed is naturally a pre-bind put and flushes with §4.2's
staging rule. No launch-descriptor involvement remains.

## 5. Wake Plane

### 5.1 Scheduler nudge (kill the 5 ms poll)

The scheduler currently holds each in-flight batch/control `Completion` and
polls `check()` under `recv_timeout(5ms)`
([scheduler.rs](runtime/engine/src/inference/scheduler.rs)); the driver's
notify publishes to a slot with no registered waiter, so the wake is lost and
retirement waits for the timeout.

Replace polling with a nudge waker:

- Add `SchedulerItem::Nudge` (empty payload). Implement `std::task::Wake` for
  a small struct holding a cloned scheduler `Sender`; `wake()` sends `Nudge`
  (send failure is ignored: a closed queue means the scheduler is gone).
- After dispatching a launch or control operation, and after any retire pass
  that leaves work in flight, the scheduler registers that waker on the front
  in-flight completion's wait slot with the observed epoch
  (register-then-recheck via the existing `WakerTable::register`).
- The main wait becomes a plain `recv` (or a long hang-backstop timeout of
  several hundred ms guarded by a counter metric; it must never be the
  steady-state wake path).

Retirement latency is then bounded by the callback, not the poll. This also
makes guest wake prompt end to end: `InstanceCompletion` still resolves on
the scheduler thread from the member terminal cells (D1/D2 unchanged), but
the scheduler now runs immediately after the callback.

### 5.2 Wake classes after this plan

| Wait slot | Registered by | Published by | Epoch |
|---|---|---|---|
| Batch completion | Scheduler nudge waker | Driver callback | Scheduler-assigned target epoch |
| Control completion | Scheduler nudge waker (and any guest awaiting the `Completion`) | Driver callback | Target epoch |
| Instance completion | Guest (`take` no longer needs it for values; error paths and `await`s on the submission still use it) | Scheduler retire (`resolve_from_terminal`) | 1 |
| Channel reader | Guest `take`/`read` | Driver callback | Channel tail |
| Channel writer | Future `wait-writable`; unused waiters are free | Driver callback | Channel head |
| Instance pacing | Unchanged | Unchanged | Unchanged |

## 6. Fire Finalization Decoupled from take

`take` currently doubles as the KV/RS transaction settlement point because
`finalize_fire` needs the guest's wasmtime resource table. Values move to
channel waits; settlement stays on the guest task but becomes opportunistic:

- **submit** and **take/read entry**: non-blockingly drain the pipeline FIFO
  head while its completion `check()` is `Some` (already-resolved fires
  only). This bounds pin float by run-ahead depth exactly as today, without
  making take block on it.
- **pipeline.close / drop**: blocking drain, unchanged, so arena pins never
  outlive the pipeline.
- **Failure surfacing**: unchanged in substance. The driver poisons channel
  words; take/read error on poison immediately (they no longer wait for the
  fire to be drained first). The drained fire marks the pass and pipeline
  failure domain (`fail_pass`, `PipelineFailure`) as today.

`Channel::fires` stops being the wait mechanism. Keep the channel-to-pipeline
wiring solely as the same-pipeline constraint check
(`wire_channels_to_pipeline` validation); the plan does not change the B3
ordering rule.

## 7. Callback Thread Contract

This section is mandatory before per-channel notifies are added; it fixes an
existing data-race family.

The completion callback (`notify_runtime_callback`,
[ptir_dispatch.cu](driver/cuda/src/ptir/ptir_dispatch.cu)) currently runs on a
CUDA host-func thread and touches `Impl::instances` (map lookup),
`BoundInstance::fire_seq`, `PtirInstance::apply_fire_result` (registry ring
bookkeeping vectors), and registry vector indexing, all of which the
scheduler thread concurrently mutates (same-instance run-ahead launches,
`register_channel` including `grow()` reallocation, `close_instance`).

New contract, normative:

1. The callback may dereference only pointers precomputed at enqueue time on
   the scheduler thread: pinned word pointers (`std::uint64_t*` per touched
   channel), terminal cell pointers, the pinned commit-flag pointer, and the
   runtime callback table. Pinned allocations are per-slot stable, so these
   pointers survive registry growth and instance churn.
2. `NotifyContext::FinalizeEntry` carries those raw pointers plus the wait
   ids and target head/tail values. The callback performs word stores,
   terminal stores, and `notify` calls. Nothing else.
3. Ring bookkeeping that today runs in the callback
   (`apply_fire_result`, host head/tail mirror sync, poison enumeration over
   `bound.trace->channels`) moves to the scheduler thread: applied when the
   next operation touching that instance is prepared, and unconditionally at
   `close_instance`. The poison word set is precomputed at enqueue (the set
   of host-visible channels is known then), so failure poisoning stays in
   the callback as plain word stores.
4. Any remaining shared mutable state gets one `std::mutex` on
   `PtirDispatch::Impl`, taken by scheduler-thread entry points and never by
   the callback.
5. `fire_seq` and the commit flag readback move behind the same rule:
   the callback reads the precomputed commit-flag pointer only.

## 8. ABI Changes (version 2)

- Delete from `PieLaunchDesc`: `ptir_host_put_values`, `host_put_indptr`,
  and `PieChannelValueDescSlice` uses tied to launches.
- `PieChannelDesc::reader_wait_id` / `writer_wait_id` become mandatory: every
  native driver must store them and notify per §4.4. Dummy already does;
  CUDA and Metal must.
- `PieInstanceDesc` (seeds) and `PieChannelEndpointBinding` are unchanged.
- Bump `PIE_DRIVER_ABI_VERSION` to 2. Regenerate
  [pie_driver_abi.h](interface/driver/include/pie_driver_abi.h) with cbindgen.

Driver word-ownership table, normative:

| Word | Reader channel | Writer channel |
|---|---|---|
| head | host (take) | driver (callback, after commit) |
| tail | driver (callback) | host (put) |
| poison | driver | driver |
| closed | driver (ordered close) | driver (ordered close) |

## 9. Runtime Changes

- [ptir_channel_store.rs](runtime/engine/src/ptir/ptir_channel_store.rs):
  `put` writes the ring per §4.2 (keep pre-endpoint staging plus the flush at
  `attach_endpoint`); delete `snapshot_host_puts`, `commit_host_puts`, and
  the Writer use of `staged`; `take`/`read` read words directly per §4.5;
  reduce `publish_reader_mirror` to poison/closed checks or delete it.
- New `ChannelWait` future (or extend `poll_wait_slot` reuse): parks on
  `reader_wait_id`/`writer_wait_id` with register-then-recheck against the
  ring words.
- [ptir_host.rs](runtime/engine/src/ptir/ptir_host.rs): submit stops
  collecting `host_puts`; `take`/`read` use `ChannelWait` and the
  opportunistic drain of §6; `LaunchSubmission` and `PendingFire` lose the
  host-put fields.
- [frame.rs](runtime/engine/src/driver/frame.rs): `LaunchSubmission` and
  `LaunchDescBorrow` lose host-put lowering.
- [scheduler.rs](runtime/engine/src/inference/scheduler.rs):
  - Nudge waker per §5.1.
  - Bug fix: `dispatch_launch_batch` must pop and reject a front launch whose
    instance is unknown instead of breaking with it still queued (today this
    head-of-line blocks the whole driver queue permanently).
- [ptir_host.rs](runtime/engine/src/ptir/ptir_host.rs) CoW copies: the
  `Completion`s returned by `copy_d2d` at submit (KV CoW, RS CoW) must ride
  the pipeline FIFO as `PendingOp::Move` like `copy_into` already does, so an
  asynchronous copy failure poisons the pipeline failure domain instead of
  vanishing into a warn log.
- [completion.rs](runtime/engine/src/driver/completion.rs): expose a helper
  to register an external waker on a pending `Completion`'s slot (for the
  scheduler nudge).

## 10. Driver Changes

- [channel_registry.hpp](driver/cuda/src/ptir/channel_registry.hpp):
  - Store `reader_wait_id`/`writer_wait_id` per slot at `register_endpoint`.
  - Add the writer pull (§4.3): consumed cursor per slot, H2D from mirror to
    device cell plus full bit on a given stream. Delete the descriptor-driven
    `host_feed` path (keep `seed_cell`).
  - Expose stable word/cell pointer accessors for callback precompute.
- [ptir_dispatch.cu](driver/cuda/src/ptir/ptir_dispatch.cu):
  - Launch prep: pull writer rings before `fire_async`; availability check in
    `validate_launch` per §4.3; delete `read_channel_values` and the host-put
    SoA plumbing.
  - `NotifyContext`: precomputed pointers and wait ids per §7; callback emits
    per-channel notifies then terminals then the batch notify per §4.4.
  - Move `apply_fire_result` and ring-mirror sync to scheduler-thread
    pre-op/close; add the `Impl` mutex.
- [entry.cpp](driver/cuda/src/entry.cpp): `LaunchScratch` loses the host-put
  SoA; launch validation drops host-put checks.
- [dummy](driver/dummy/src/lib.rs): align the word/pull semantics exactly so
  dummy-backed tests exercise the same contract the CUDA driver implements
  (today dummy notifies channel slots but CUDA does not; the asymmetry hides
  regressions).
- Metal: implements the same §4/§8 contract. Real execution remains gated on
  direct_ffi_fix.md Phase 3; this plan defines what its completion handler
  must publish and notify.

## 11. Deletion Inventory

Delete after Phase 2:

```text
PieLaunchDesc.ptir_host_put_values / host_put_indptr (ABI + borrow lowering)
runtime snapshot_host_puts / commit_host_puts / Writer staging steady-state
runtime publish_reader_mirror visibility gating
take/read pipeline-FIFO wait loop (value waiting via PendingFires)
driver LaunchScratch host-put SoA (ptir_host_put_channels/blob/lens/indptr)
driver read_channel_values / validate_host_puts / feed_host_puts launch path
driver host_feed (descriptor-bytes variant)
```

Do not delete: the pipeline FIFO itself (txn settlement and ordering), seed
descriptors, reader capacity reservations.

## 12. Ordering and Memory Model (normative)

- Put: cell bytes are written before the release store of `tail`; the driver
  acquire-loads `tail` on the scheduler thread; pulls are stream-ordered
  before the pass that consumes them.
- Reader publish: DMA precedes the callback in copy-stream order; the
  callback release-stores `tail` before notifying; the guest acquire-loads
  `tail` before reading mirror cells; every waiter re-checks after
  registering (the existing per-slot mutex in
  [table.rs](runtime/waker/src/table.rs) linearizes registration against
  publication).
- Terminal cells and the batch notify keep direct_ffi.md §7.1 order, with
  §4.4 inserting the per-channel word stores and notifies before the
  terminals.
- Epochs: reader slot epoch = tail value; writer slot epoch = head value;
  batch/control epochs unchanged. All are monotonic and compose with the
  epoch-filtered `WakerTable::publish`.

## 13. Implementation Order

**Phase 0, hygiene, no ABI change.**
Scheduler nudge waker (§5.1); callback contract (§7: pointer precompute,
bookkeeping relocation, `Impl` mutex); stale-front-launch pop
(§9 scheduler bug); CoW completions onto the pipeline FIFO (§9). Each is
independently landable and independently testable.

**Phase 1, reader direct wake, no ABI change.**
CUDA stores channel wait ids and the callback notifies them (§4.4); runtime
`take`/`read` park on the channel slot and read words directly (§4.5);
finalization decoupling (§6). The launch descriptor still carries host puts
during this phase.

**Phase 2, writer direct puts, ABI v2.**
Ring put (§4.2), driver pull and availability check (§4.3), descriptor
payload deletion (§8, §11), regenerate the header, dummy parity.

**Phase 3, closure.**
Extern cross-inferlet channel tests, writer backpressure wake, Metal
contract, deletion inventory grep gates, TSAN gate.

## 14. Validation Gates

Behavioral tests:

1. Put-before-submit: a put is visible to the next fire with no
   host-put payload in the launch descriptor (assert descriptor slices are
   empty on the driver side).
2. Direct reader wake: a guest parked in `take` wakes from the driver
   callback with the scheduler thread deliberately stalled (inject a slow
   scheduler op); the value arrives anyway.
3. Cross-instance extern channel: instance A's fire fills a channel; a task
   that never submitted wakes from `take` on it.
4. Writer backpressure: puts up to capacity succeed, the next returns Full;
   after a consuming fire commits, `writer_wait_id` is notified and a put
   succeeds.
5. Missing put: a launch whose trace takes from an empty writer ring is
   rejected synchronously; no epoch is consumed, no poison is published.
6. Retirement is event-driven: with a single pipeline and one fire, the
   nudge (not the timeout) retires the batch. Assert via a scheduler
   counter: steady-state retires with zero timeout-path wakeups.
7. Poison ordering: a failed fire poisons words before the channel notify;
   a parked `take` wakes into `Poisoned`, not `Empty`.
8. Register-then-recheck race: publish tail between a waiter's check and its
   registration; the waiter must not park forever (existing lost-wakeup test
   extended to channel slots).
9. Same-instance run-ahead under the new callback contract passes TSAN
   (fire N callback concurrent with fire N+1 prep, plus concurrent
   `register_channel` growth and `close_instance` of an unrelated instance).
10. Pin-float bound: with take never called, submits drain resolved fires and
    arena pins stay bounded by run-ahead depth.

Grep gates (zero results outside migration documents):

```text
rg 'ptir_host_put_values|host_put_indptr' interface runtime worker driver
rg 'snapshot_host_puts|commit_host_puts' runtime
rg 'validate_host_puts|feed_host_puts|read_channel_values' driver
rg 'recv_timeout' runtime/engine/src/inference/scheduler.rs   # backstop only, with counter
```

## 15. Non-goals

- No new WIT surface beyond the changed semantics of `take`/`read`
  (an awaitable `wait-writable`/`put-async` is enabled by §4.4 but not
  required here).
- No change to batching policy (WaitAll), run-ahead depth, KV/RS transaction
  semantics, typed memory operations, or `driver/transport`.
- No second result API: channels remain the only driver-to-runtime value
  surface (direct_ffi.md §4.4 stands).
- No reopening of the transport migration: request/response, shmem, and
  service-loop paths stay deleted.

## 16. Definition of Done

1. All Phase 0-3 tests in §14 pass; grep gates are clean.
2. A single-pipeline decode loop performs put, submit, and take with: zero
   scheduler timeout wakeups, zero launch-descriptor value bytes, and the
   take wake originating from the driver callback (verified by wake-class
   counters on the waker table).
3. TSAN-clean scheduler/callback concurrency under same-instance run-ahead.
4. Dummy, CUDA, and (contract-level) Metal publish and notify identically.
