# KV Contention and Active Preemption Implementation Plan

## Status

This document describes the gap between the current runtime and the target KV
contention behavior, then proposes an implementation plan for closing that gap.
It reflects the repository state as of July 2026. Revised 2026-07-11 after a
code-verified review: unified waiter/restore grant order, two-domain keystone
definition, await-boundary safe points, explicit RS scope (3.17), and
capability-check corrections.

The intended end state is:

1. KV pool exhaustion does not immediately fail an otherwise valid inferlet.
2. The runtime first removes reclaimable cache-only state.
3. If more space is required, it preempts younger inferlets at safe points.
4. Preemption copies private resident KV pages to host swap storage and returns
   their device slots to the KV pool.
5. Allocation and restore are ordered by the inferlets' original submission
   order, not by the time they happened to block or finish suspending.
6. When space becomes available, including after another inferlet terminates,
   the oldest eligible inferlet is restored first.
7. Suspend and restore are transparent to the inferlet: logical WorkingSet
   indexes, committed token state, prefix identity, and generated output remain
   correct.
8. The system cannot deadlock by parking a process while it still holds the
   pages needed to make progress, and it cannot thrash by restoring work that
   must immediately be evicted again.

## 1. North-Star Semantics

### 1.1 Fairness clock

Every inferlet receives a monotonic `submit_seq` when the runtime accepts and
registers the process. This is the authoritative FCFS clock.

The initial implementation may use the existing process registration order as
`submit_seq`, because `process::spawn` registers the process before its WASM task
runs. The name should nevertheless be made explicit: restore order must be
defined in terms of original submission order, not suspension time.

The ordering rules are:

- Victim selection: preempt the youngest eligible running inferlet first
  (`max(submit_seq)`).
- Grant priority: newly available pages go to the oldest entitled request
  first (`min(submit_seq)`), comparing blocked allocation requests and
  restore requests in one order. A younger allocation waiter never outranks
  an older suspended inferlet's restore (section 3.16 describes the
  inversion this rule prevents).
- The progress keystone is the oldest non-terminated inferlet, in whatever
  state. It is never selected as a victim and is exempt from requester
  self-suspend. A process that is merely the oldest running inferlet while
  an older peer sits suspended receives no exemption: it must still yield
  so the older peer can restore.
- A process cannot bypass an older waiter merely because it races and observes
  free pages first.

"Oldest" therefore has two domains, and they must not be conflated:

- Exemption domain (victim pick, requester self-suspend): all non-terminated
  processes, including suspended ones.
- Exhaustion-clock domain: processes that can currently issue allocation
  requests. A suspended keystone issues none, so the exhaustion watchdog must
  also cover a restore head that stays continuously unsatisfiable
  (section 8).

If a single inferlet can create multiple simultaneous allocation requests, the
runtime must either:

- maintain one aggregate contention request per process; or
- order requests by `(submit_seq, request_seq)`.

The first option is preferred for the initial implementation because it avoids
one run-ahead inferlet occupying several positions in the global wait queue.

### 1.2 Safe preemption

An inferlet may suspend only at a runtime-controlled safe point:

- no new fire may be submitted after the suspend request is observed;
- all already submitted fires that can read or write the affected WorkingSets
  must have retired;
- no prepared KV transaction may remain uncommitted;
- no driver copy may still refer to pages that will be recycled;
- no store or ResourceTable lock may be held while awaiting a copy, park, or
  restore.

The safe point freezes the WASM task, not just an individual forward pass. A
process can own several WorkingSets through create, fork, and slice, so
preemption must operate over the process's complete KV residency set.

Safe points must also be reachable from long host awaits. A victim blocked in
a channel receive, a fire-result await, or any other external-event wait may
never issue another host call; the park request must be able to interrupt
these awaits (race the await against the park notification). A process with
no in-flight fires that is idle in a host await is already quiescent; it is
the cheapest possible victim, not an unreachable one. Without an
await-boundary seam, a park request against an idle process is never honored
and its pages stay stranded until it wakes on its own.

### 1.3 Transparent residency

The inferlet continues to address KV by `WorkingSetPageIndex`. Suspension may
change the physical GPU page assigned to a logical page, but only while the
process is frozen and no fire can observe the mapping.

The following state must survive suspension:

- WorkingSet identity and logical extent;
- trie topology and shared-prefix relationships;
- token-slot hashes, page hashes, and chain state;
- committed token length;
- pending logical reservations;
- canonical prefix-cache identity;
- process, pipeline, and forward-pass state;
- channel contents and guest-visible cursors.

The runtime must not implement suspension by dropping and recreating a
WorkingSet. Doing so would lose or incorrectly reconstruct mapping, hash, and
sharing metadata.

### 1.4 Reclaim scope

Only pages exclusively reachable from the suspended process may be returned to
the device pool.

- Shared prefix pages still used by a running process remain resident.
- Cache-root-only references are removed by contention rung 1 before selecting
  a victim.
- A victim may release its reference to a shared page without freeing the
  physical page.
- Private pages protected by an in-flight snapshot or fire lease are
  temporarily ineligible and produce a grace-deferred outcome.

The existing `KvStore::exclusive_footprint` is a sizing primitive, not a
complete suspend plan. The actual plan must identify the exact private physical
pages and the metadata locations that will be changed.

Two scope limits are explicit in the first version:

- Pages shared only among suspended processes stay resident. Exclusive
  reachability is computed against all anchors, so a prefix shared by two
  suspended victims is freed by neither. A shared-subtree rung (evict a
  subtree once every anchor on it is itself suspended or cache-only) is a
  later extension.
- Suspension covers KV pages only. A process holding RS (recurrent-state)
  slots keeps them resident on the device while parked: RS folded state is
  written in place, cannot be reconstructed mid-sequence, and the driver's
  `copy_state` is device-to-device only, so there is no host destination for
  it (section 3.17). Leaving RS resident is safe because the frozen process
  performs no in-place writes, but the suspend plan must account for it when
  reporting freed capacity, and RS-pool contention itself has no reclaim
  rung in v1.

### 1.5 Restore admission

Restore is allowed only when all pages required by the restore can be allocated
without evicting another process.

- Allocation is all-or-nothing.
- Restore never causes preemption.
- Blocked allocation requests and restores compete in one `submit_seq` order:
  the older entry receives the next grant. Victims are picked youngest-first,
  so waiters still usually outrank restores in practice; the unified order
  exists so an old suspended inferlet cannot be starved by a stream of
  younger waiters (section 3.16).
- A restore head deferred by the utilization pause yields its turn to younger
  entries (the pause is voluntary); a restore head that simply does not fit
  holds its FCFS slot.
- Among restores, the smallest `submit_seq` wins.
- The existing utilization pause remains an anti-thrash gate.
- Aging may override the utilization pause, but never the "restore must fit"
  rule.

## 2. Current Implementation

### 2.1 Components that are already live

The following pieces are implemented and connected:

#### Process registration and teardown

- `inferlet::process::spawn` registers each process with the global
  `ContentionOrchestrator`.
- the process termination funnel unregisters the process;
- unregister removes waiters and restore entries and wakes a parked task being
  torn down.

#### KV allocation failure

- `KvStore::reserve` is logical-only;
- `KvStore::prepare_write` performs all-or-nothing physical allocation;
- pool exhaustion remains a typed `KvStoreError::OutOfPages`;
- the normal host-geometry fire path converts this into deferred scheduler
  preparation and retries;
- the device-geometry path has a direct async contention loop.

#### Zero-cost reclaim

- released canonical prefixes can be retained as bounded cache roots;
- `KvStore::drop_unused_cache_leases` drops cache-only roots under pressure;
- this happens before victim selection.

#### Passive wait-for-free behavior

With `PIE_KV_CONTENTION=preempt`, the default `KvPoolBackend`:

- reads real free and total page counts;
- parks allocation requests;
- wakes them when a fire, discard, WorkingSet drop, or process exit makes pages
  available;
- does not preempt a running process.

This is useful backpressure, but it is not active preemption.

#### Orchestrator state machine

`store/reclaim.rs` already defines:

- process registration order;
- running, suspending, park-requested, suspended, and restoring states;
- victim selection;
- waiter parking and notification;
- a restore queue;
- anti-thrash restore gating and aging;
- an exhaustion timeout;
- wait-all scheduler leave notifications;
- contention counters.

#### KV lifetime safety

- prepared writes pin their terminal snapshots;
- freed physical IDs are recycled only after completion epochs retire;
- `KvFireLease` delays WorkingSet release while fires are active;
- store locks are not held across driver awaits.

### 2.2 Driver primitives that already exist

CUDA already provides most of the physical copy data plane:

- `SwapPool` allocates pinned host storage with `cudaMallocHost`;
- the host layout mirrors every KV buffer for every model layer;
- D2H, H2D, D2D, and H2H page copies exist;
- copies run on a dedicated non-blocking stream;
- the driver ABI exposes KV copies by source and destination memory domain;
- Rust scheduler trampolines exist as `copy_d2h`, `copy_h2d`, `copy_d2d`, and
  `copy_h2h`;
- `copy_state` exists for RS slots but is device-to-device only; the pinned
  swap pool backs KV pages exclusively.

These primitives are not yet called by the runtime preemption path.

### 2.3 Existing configuration

- `PIE_KV_CONTENTION=preempt` installs the contention orchestrator.
- `PIE_KV_PREEMPT_ACTIVE=1` selects `SelfSuspendBackend`.
- `scheduler.restore_pause_at_utilization` is passed to the orchestrator.
- `PIE_FIRE_RETRY_MAX` bounds scheduler preparation retries.
- comments describe KV exhaustion and restore-aging environment controls, but
  the current implementation hard-codes the corresponding durations.

## 3. Current Gaps

### 3.1 Active preemption is not wired

`SelfSuspendBackend::suspend` returns `SuspendOutcome::Requested`, causing the
orchestrator to mark a victim `ParkRequested`. No live execution path checks
`ContentionOrchestrator::should_park`, so the victim never:

1. reaches a suspend safe point;
2. saves KV state;
3. calls `report_suspended`;
4. parks;
5. restores after being released.

`decline_park`, `report_suspended`, and `park_until_restored` exist but have no
call sites at all; not even tests exercise them.

The comment near `inferlet/host/grammar.rs` describing a self-suspend cycle is
not an implementation.

### 3.2 No process-level residency inventory

The orchestrator tracks a `ProcessId`, while KV state is held through
process-local WASM resources. A process may own multiple related WorkingSets,
and the orchestrator cannot safely reach into another process's ResourceTable.

There is currently no process-level registry that can answer:

- which KV WorkingSets belong to this process;
- which ones are live, shared, pending, or already suspended;
- which physical pages are unique across all of those WorkingSets;
- whether all associated fires have retired;
- how many pages can actually be freed.

Without this inventory, active preemption can save one WorkingSet while leaving
other process-owned pages resident, double-count shared pages, or miss a
WorkingSet entirely.

### 3.3 No runtime host-slot allocator

The CUDA driver owns a pinned host `SwapPool`, but the Rust runtime has no typed
host-slot free list and no ownership records.

`arena_cpu_pages` is collected during bootstrap and currently discarded. The
runtime therefore cannot:

- reserve host slots before D2H;
- prevent two suspended pages from using the same slot;
- associate a swapped page with its host slot;
- release slots after restore or process termination;
- report host swap capacity and utilization;
- reject or choose a cold fallback when host swap is full.

### 3.4 KV metadata cannot represent non-resident pages

`KvPageTable` currently stores resident `PhysicalKvPageId` values directly in
owned trie nodes. There is no state for a committed logical page whose contents
are in a host slot rather than a GPU page.

Returning a physical ID to the free pool while leaving that ID in a suspended
WorkingSet would be incorrect: another inferlet could reuse and overwrite it.

The initial implementation therefore needs an explicit page residency state,
for example:

```rust
enum KvPageBacking {
    Resident(PhysicalKvPageId),
    Swapped(HostKvSlotId),
}
```

Flattening a runnable WorkingSet must require every visible page to be
`Resident`. A suspended WorkingSet may contain `Swapped` pages but cannot be
bound to a launch.

This is a deliberate near-term relaxation of the "physical page IDs never
change" design goal. Logical indexes remain stable; physical IDs may be
reassigned only while the process is frozen. Preserving stable physical IDs
across eviction would require CUDA VMM or equivalent sparse backing and is a
larger follow-up.

### 3.5 No suspend or restore transactions

The runtime lacks APIs equivalent to:

- `classify_for_suspend`;
- `prepare_suspend`;
- `commit_suspend`;
- `abort_suspend`;
- `prepare_restore`;
- `commit_restore`;
- `abort_restore`.

A correct transaction must span store metadata and asynchronous driver copies
without holding locks:

```text
lock store
  validate safe point
  reserve host slots
  capture device IDs and metadata locations
unlock store

await D2H

lock store
  mark pages swapped
  recycle device IDs at a safe epoch
unlock store
```

Restore requires the inverse sequence:

```text
lock store
  allocate all required device IDs
  capture host-slot to device-page copy plan
unlock store

await H2D

lock store
  publish new resident IDs atomically
  invalidate and version flattened tables
  release host slots
unlock store
```

Every failure path needs an explicit rollback. A failed D2H must leave the
resident mapping valid. A failed H2D must leave the swapped mapping and host
slots valid.

### 3.6 Restore order is not original submission order

The current `restore_queue` is populated when suspension completes and is
drained FIFO. This orders restores by suspension completion, not by process
submission.

Example:

```text
submit order: A, B, C
victim order: C, then B
current restore queue: C, B
required restore order: B, C
```

The restore structure must be keyed by `submit_seq`, such as a `BTreeSet` or a
priority queue with reversed ordering. Suspension completion order must not
change fairness.

### 3.7 Waiter wakeup is advisory, not a strict allocation grant

The current waiter queue notifies a FIFO prefix whose cumulative need fits the
observed free pool. A notified waiter then races to allocate. A new requester
can also observe `free >= need` and allocate before an older waiter consumes
its pages.

This violates strict FCFS and can starve an older process.

The orchestrator needs reservation-backed grants:

- free pages are assigned to a specific request in FCFS order;
- the request receives an `AllocationGrant`;
- `prepare_write` consumes that grant;
- cancellation or allocation mismatch returns the reservation;
- ungranted callers cannot bypass older waiters.

The physical pool should remain non-blocking. The orchestrator owns waiting and
fairness; the pool owns IDs.

### 3.8 The requester self-suspend deadlock breaker is unused

`acquire_or_self_suspend` and `Acquired::SelfSuspendFirst` exist, but live fire
paths call `acquire`, which always passes `holds_reclaimable=false`.

Consequently, a process can park while still holding private pages, even when
freeing its own pages is required for global progress.

The fire path must:

1. calculate the process's reclaimable footprint;
2. call `acquire_or_self_suspend`;
3. execute the same suspend cycle when `SelfSuspendFirst` is returned;
4. retry only after restore.

The oldest process remains exempt from requester self-suspend so it can act as
the completion keystone.

### 3.9 Deep run-ahead is not gated during contention

The orchestrator exposes `contended()` and `fire_retired()`, but the fire path
does not use them.

A process may continue submitting while its own settled-but-undrained fires
hold pins or leases. This can prevent the pages needed by a waiter from
becoming reclaimable.

When contention is active, a lane must stop deepening and drain its own fire
queue until the relevant leases retire.

### 3.10 Scheduler retry is polling-shaped

The normal fire path launches an async `acquire` task and repeatedly returns
`LaunchPreparationError::Retry`. The scheduler rotates the prepare item and
counts retries up to `PIE_FIRE_RETRY_MAX`.

Long-lived memory contention should be represented as a blocked preparation,
not thousands of retries:

- a contention wait must not consume the ordinary transient retry budget;
- the scheduler should requeue the item when its grant or terminal error is
  ready;
- cancellation must cancel the waiter;
- one process should not create duplicate background acquire tasks. The
  current guard is an `AtomicBool` created per fire preparation, so a process
  with several in-flight fires spawns one acquire task per fire and occupies
  several waiter slots; the per-process aggregation of section 1.1 is
  required, not optional.

### 3.11 Restore wakeup and restore completion are conflated

The current parked-process drain path sets the process state to `Running` and
wakes it because no real H2D work exists yet. Once restore is implemented, that
ordering would make the process runnable before its KV pages are resident.

The protocol must be split:

1. the orchestrator selects the oldest suspended process and grants restore
   capacity;
2. the process transitions to `Restoring` and wakes;
3. the process performs H2D and atomically publishes resident mappings;
4. the process reports restore completion;
5. only then does the orchestrator transition it to `Running`.

`park_until_restored` should therefore be replaced or redefined as
`park_until_restore_granted`, returning a restore grant rather than implying
that restoration has completed.

### 3.12 Incomplete scheduler lifecycle coupling

Suspend emits a scheduler leave notification, correctly removing a frozen
pipeline from wait-all quorum. Join notifications are currently no-ops, relying
on implicit rejoin at the next request.

Implicit rejoin can remain the policy, but it needs explicit tests for:

- a process suspended between waves;
- a process suspended during deferred preparation;
- requester self-suspend;
- cancellation while parked;
- restored work re-entering wait-all without being prematurely demoted.

### 3.13 Platform and topology limitations

The initial active path can only target single-GPU CUDA:

- CUDA has a host-pinned SwapPool and cross-domain copies.
- CUDA `copy_kv` currently rejects tensor parallelism greater than one.
- Metal supports same-domain page copies only and has no host-pinned swap pool.
- the runtime contention singleton is wired around the current single-model,
  driver-0 assumptions.
- the capability surface has no field describing supported copy domains, and
  the only proxy, `swap_pool_size`, is unreliable: the Metal driver reports a
  nonzero value while allocating no host-pinned pool.

These constraints must be explicit capability checks, not late copy failures.

### 3.14 Observability is incomplete

Existing counters show waiter, suspend, and restore counts, but production
diagnosis also needs:

- current waiter count and ordered waiter list;
- current suspended count and oldest suspended `submit_seq`;
- device pages free, resident, pending recycle, and reserved by grants;
- host swap slots free and used;
- pages and bytes moved D2H/H2D;
- suspend and restore copy latency;
- grace-deferred victim count;
- victim selection count by process;
- restore blocked by waiters, capacity, or utilization;
- cancelled waits and rolled-back transactions;
- exhaustion timeout count;
- per-process resident and swapped page counts.

The worker currently reports constant zero values for KV pressure and inflight
load. This does not block local preemption, but it prevents meaningful routing
and cluster-level visibility.

### 3.15 Test coverage does not prove the north star

There are tests for passive wait-for-free behavior and an ignored CUDA
contention harness. There is no deterministic test that proves all of:

- a running victim performs D2H;
- its GPU IDs become allocatable;
- an older waiter consumes the freed capacity first;
- termination creates restore headroom;
- suspended inferlets restore by original submission order;
- H2D restores exact KV contents;
- logical mappings and generated output remain correct;
- no page or host slot leaks after success, failure, or cancellation.

### 3.16 Waiter priority over restores can starve or wedge the oldest process

The current drain gives parked allocation waiters strict priority over
restores and skips the restore phase entirely while any waiter is parked. The
orchestrator also computes "oldest" over all registered processes, but only an
`acquire` caller can arm the exhaustion clock.

Combined, these produce an inversion. Suppose A (seq 1) terminates while B
(seq 2) is suspended and C (seq 3) is parked on allocation. B is now the
oldest non-terminated process.

- B's restore is structurally blocked for as long as C, or any later waiter,
  stays parked, even though B is older. A sustained stream of younger waiters
  starves B indefinitely, violating north-star rule 6.
- The exhaustion clock never arms: B issues no allocation requests, and C is
  not the oldest registered process, so its deadline resets forever. The
  system can wedge silently with no loud failure.
- If C were instead treated as the keystone (oldest running), the
  self-suspend exemption would prevent C from yielding its own pages, which
  is exactly the action that would let B restore.

The fixes are the two-domain keystone definition and the unified grant order
of section 1.1, plus the restore-head exhaustion watchdog of section 8.

### 3.17 RS residency is outside the current plan

Models with recurrent state hold RS slots in a separate typed store. Three
facts constrain preemption:

- RS folded state is written in place when uniquely owned, and there is no
  rollback across a committed fold; it cannot be recomputed from a midpoint,
  so drop-and-replay is not even a theoretical fallback for it.
- The driver's `copy_state` is device-to-device only, and the pinned swap
  pool mirrors KV buffers, not RS slots. There is no host destination for RS.
- A frozen process performs no in-place RS writes, so suspending only its KV
  while RS stays resident is safe.

The v1 policy is therefore: suspend evacuates KV pages only; RS slots remain
resident and keep occupying the RS pool while the process is parked; the
process residency inventory must track RS ownership so freed-capacity
reporting and any future RS ladder have accurate inputs. Contention on the RS
pool itself has no reclaim rung in v1 and fails through the exhaustion
policy. RS host backing and domain-parameterized state copies are a Phase 9
item.

## 4. Proposed Architecture

### 4.1 Separate orchestration from residency operations

Keep `ContentionOrchestrator` responsible for:

- process ordering;
- victim selection;
- allocation waiter ordering;
- restore eligibility;
- state transitions;
- grants and notifications.

Keep `KvStore` responsible for:

- trie reachability and sharing;
- resident and swapped page metadata;
- device and host slot ownership;
- suspend and restore transactions;
- epoch-safe page recycling;
- flattened-table publication.

Keep the process task responsible for:

- reaching a safe point in its own ResourceTable;
- enumerating its WorkingSets;
- draining its fires;
- issuing and awaiting D2H/H2D operations;
- parking and resuming its WASM continuation.

Keep the driver responsible for:

- copying opaque KV pages between named device and host slots;
- reporting completion or failure;
- exposing swap capacity and supported memory domains.

### 4.2 Per-pool contention registry

Replace the eventual single global contention object with a registry keyed by:

```text
(model_id, driver_id, resource_kind)
```

The first implementation may keep a single CUDA KV pool, but new APIs should
carry the key so multi-driver support does not require redesigning every state
record.

### 4.3 Process KV residency registry

Add a process-owned residency registry independent of WASM resource handles.
WorkingSet creation, fork, slice, and release update it.

Suggested responsibilities:

```rust
struct ProcessKvResidency {
    process_id: ProcessId,
    submit_seq: u64,
    working_sets: HashSet<WorkingSetId>,
    state: ProcessResidencyState,
}

enum ProcessResidencyState {
    Running,
    Quiescing,
    Suspending,
    Suspended,
    Restoring,
    Terminating,
}
```

The registry must not own a second copy of trie metadata. It tracks membership
and lifecycle only; the store remains authoritative.

### 4.4 Typed host swap pool

Add `HostKvSlotId` and a `Pool<HostKvSlotId>` or a dedicated host-slot pool.

Requirements:

- capacity comes from the driver/bootstrap configuration;
- host slots are allocated all-or-nothing for a suspend plan;
- pending D2H slots are distinct from committed swapped slots;
- slots are released on restore, process termination, or suspend rollback;
- capacity must match the driver's allocated SwapPool;
- debug checks reject an out-of-range host slot before entering the driver.

### 4.5 Explicit residency metadata

Refactor owned KV page metadata to distinguish resident and swapped backing.
One possible representation is:

```rust
enum KvPageBacking {
    Resident(PhysicalKvPageId),
    Swapped(HostKvSlotId),
}

struct OwnedKvPage {
    backing: KvPageBacking,
    token_hashes: Vec<Option<Hash256>>,
    page_hash: Option<Hash256>,
}
```

The actual representation may retain parallel vectors for scan efficiency, but
the invariants must be equivalent.

Important invariants:

- a runnable WorkingSet flattens only resident pages;
- a swapped page never exposes its former GPU ID;
- a GPU ID is recycled only after D2H completion and all prior users retire;
- a host slot is released only after H2D completion and metadata publication;
- shared-node rewrites cannot affect a running process;
- mapping version increments whenever a logical-to-physical value changes.

### 4.6 Suspend transaction

Add a typed transaction with an explicit state:

```rust
struct KvSuspendTxn {
    process_id: ProcessId,
    working_sets: Vec<WorkingSetId>,
    pages: Vec<SuspendPage>,
    captured_epoch: u64,
}

struct SuspendPage {
    location: TriePageLocation,
    gpu_id: PhysicalKvPageId,
    host_slot: HostKvSlotId,
}
```

`prepare_suspend` must:

1. verify process state and WorkingSet ownership;
2. verify no pending writes or active fire leases;
3. compute the union of private resident pages;
4. reject or defer pinned pages;
5. reserve all host slots;
6. pin metadata locations against mutation;
7. return a D2H copy plan.

After D2H succeeds, `commit_suspend` must:

1. atomically replace resident backing with host slots;
2. invalidate affected flattened mappings;
3. unpin transaction metadata;
4. recycle GPU IDs through the completion-epoch mechanism;
5. return the number of pages freed now versus pending retirement.

`abort_suspend` must release host slots and transaction pins while preserving
the resident mappings.

### 4.7 Restore transaction

`prepare_restore` must:

1. select every swapped page belonging exclusively to the process;
2. allocate all required GPU IDs under an FCFS restore grant;
3. pin metadata locations;
4. return an H2D copy plan.

After H2D succeeds, `commit_restore` must:

1. atomically replace host-slot backing with new GPU IDs;
2. refresh and version affected flattened tables;
3. release host slots;
4. mark the process runnable;
5. unpin metadata.

`abort_restore` must return allocated GPU IDs safely while leaving host-backed
metadata intact.

### 4.8 Allocation grants

Introduce a reservation object that ties orchestration to physical allocation:

```rust
struct AllocationGrant {
    process_id: ProcessId,
    request_id: u64,
    pages: u32,
    pool: PoolKey,
}
```

Two implementation options are possible:

1. Reserve actual physical IDs when issuing the grant.
2. Reserve a page count in the orchestrator and require `try_alloc_n_granted`
   to consume the reservation.

Reserving actual IDs is simpler and provides stronger correctness. The grant
must be returned automatically on drop if it is not consumed.

Restore grants and forward-allocation grants should use the same reservation
mechanism and compete in one `submit_seq` order (section 1.5); a
pause-deferred restore yields its turn, a non-fitting one holds its slot.

### 4.9 Ordered process sets

Replace time-of-event FIFO structures with ordering keyed by `submit_seq`:

```text
running victims: greatest eligible submit_seq first
grant queue (allocation waiters and restores, merged): smallest submit_seq
  first; a pause-deferred restore yields its turn to the next entry
```

Use stable request IDs to remove cancelled entries without relying on process
identity alone.

### 4.10 Safe-point helper

Implement a real process-side helper, conceptually:

```rust
async fn honor_kv_preemption(ctx: &mut ProcessCtx) -> Result<()> {
    if !orchestrator.should_park(ctx.process_id()) {
        return Ok(());
    }

    quiesce_new_submissions(ctx)?;
    drain_all_process_fires(ctx).await?;

    match prepare_and_copy_suspend(ctx).await? {
        SuspendResult::NothingReclaimable => {
            orchestrator.decline_park(ctx.process_id());
        }
        SuspendResult::Suspended { freed_pages } => {
            orchestrator.report_suspended(ctx.process_id(), freed_pages);
            let grant = orchestrator
                .park_until_restore_granted(ctx.process_id())
                .await?;
            match restore_process_kv(ctx, grant).await {
                Ok(()) => orchestrator.report_restored(ctx.process_id())?,
                Err(error) => {
                    orchestrator.report_restore_failed(ctx.process_id(), &error);
                    return Err(error);
                }
            }
        }
    }

    Ok(())
}
```

The helper must run at a central host-call boundary that every active inference
loop reaches. Adding it independently to only one fire variant is insufficient.

The initial safe points should include:

- before submitting a normal host-geometry fire;
- before granting device-geometry pages;
- before other host calls that can continue a long-running generation loop;
- inside long-blocking host awaits (channel receive, fire-result waits),
  raced against the park notification, so an idle process honors a park
  request instead of stranding its pages (section 1.2).

Longer term, host-call dispatch should provide one common preemption prologue.

### 4.11 Contention-aware run-ahead gate

Before accepting another fire from a process while contention exists:

1. drain already settled fires;
2. if relevant fires remain in flight, wait for `fire_retired`;
3. re-check contention and park requests;
4. do not deepen the process until its pins no longer block progress.

Use the enable-check-await pattern around `Notify` to avoid lost wakeups.

### 4.12 Event-driven scheduler preparation

Add a blocked preparation outcome rather than reusing transient retry:

```rust
enum LaunchPreparationError {
    Blocked(ContentionWait),
    Retry(String),
    Failed(String),
}
```

The scheduler should:

- remove a blocked preparation from the hot retry rotation;
- reinsert it when its wait future yields a grant or hard error;
- preserve process and request ordering;
- avoid counting blocked duration against `PIE_FIRE_RETRY_MAX`;
- reject and release the grant on shutdown or cancellation.

This replaces the retry-with-escalation shape that `runahead-plan.md`
prescribes for the same dispatch-time seam; update that document alongside
this change so the two plans do not diverge on one code path. Aborting a
blocked preparation is safe for RS-bearing fires: RS prepare computes targets
without mutating the mapping, only committed fires are retry-ineligible, and
quiesce already drains those.

## 5. Implementation Phases

### Phase 0: Freeze semantics and invariants

Deliverables:

- name and persist `submit_seq`;
- define allocation, victim, and restore ordering;
- fix the two "oldest" domains: exemption over all non-terminated processes,
  exhaustion arming over requesting processes plus the restore-head watchdog
  (sections 1.1, 3.16);
- adopt the unified waiter/restore grant order and its anti-thrash
  interaction (section 1.5);
- adopt the v1 RS policy: KV-only suspend, RS slots stay resident, no RS
  reclaim rung (section 3.17);
- add a driver capability field for supported copy domains, and stop treating
  `swap_pool_size` as that signal (Metal reports it nonzero with no pool);
- decide that the first active implementation is CUDA, single GPU, one model;
- approve remap-on-restore as the near-term residency model;
- document that VMM-backed stable physical IDs are a later optimization;
- define behavior when host swap is exhausted;
- define whether oversized requests fail immediately or may use cold replay.

Recommended initial policy:

- no cold replay in the first active version;
- if host swap cannot hold a complete suspend transaction, mark that victim
  unsupported for the current acquisition and try another;
- if no action can satisfy the oldest request, use the existing fail-loud
  exhaustion path.

Exit criteria:

- unit-testable ordering rules are written as code-level invariants;
- no ambiguity remains between submit order and suspension order.

### Phase 1: Make fairness reservation-backed

Deliverables:

- replace FIFO restore completion order with `submit_seq` ordering;
- order allocation waiters by `submit_seq`;
- deduplicate or aggregate process allocation waits;
- add `AllocationGrant`;
- prevent new callers from bypassing existing waiters;
- make cancellation and unregister return grants;
- add queue and grant metrics.

Tests:

- A, B, C submit in order; C and B suspend; B restores before C;
- an older suspended process outranks a younger allocation waiter for the
  next grant;
- a new D request cannot steal pages granted to B;
- a cancelled B returns its grant and C proceeds;
- multiple fire requests from B do not occupy multiple process-priority slots;
- a large head waiter enforces the chosen strict-FCFS policy.

Exit criteria:

- fairness is deterministic under forced races;
- no wakeup depends on a caller winning an unreserved pool race.

### Phase 2: Add host swap ownership and residency metadata

Deliverables:

- introduce `HostKvSlotId`;
- register host swap capacity at bootstrap;
- add the host-slot allocator;
- represent `Resident` and `Swapped` page backing;
- reject flatten/launch for nonresident WorkingSets;
- add process residency inventory (including RS slot ownership);
- fix the Metal capability report (zero swap capacity, or gate it on the new
  copy-domain field) so bootstrap checks cannot pass against a pool that does
  not exist;
- add consistency checks between runtime and driver swap capacity.

Tests:

- host slots allocate and free all-or-nothing;
- shared pages are counted once;
- swapped pages expose no stale GPU ID;
- resident-only flattening remains unchanged;
- process teardown releases both GPU and host resources.

Exit criteria:

- store metadata can safely describe a frozen, nonresident WorkingSet without
  losing trie or hash state.

### Phase 3: Implement suspend transactions and D2H

Deliverables:

- `prepare_suspend`, `commit_suspend`, and `abort_suspend`;
- exact private-page classification across all process WorkingSets;
- grace deferral for pending or in-flight pages;
- call the existing scheduler `copy_d2h` trampoline;
- await copy completion before recycling device IDs;
- report the actual immediately reclaimable page count.

Tests:

- D2H success transitions pages to `Swapped`;
- D2H failure rolls back without changing visible mappings;
- pinned pages return grace-deferred;
- forked and sliced WorkingSets preserve sharing;
- a process with several WorkingSets is suspended atomically;
- device IDs are not reused before copy and retirement complete.

Exit criteria:

- a synthetic process can be suspended and its GPU pages become safely
  allocatable while its logical KV state remains intact.

### Phase 4: Wire process safe points and real victim suspension

Deliverables:

- central `honor_kv_preemption` helper;
- `should_park` call at live host-call boundaries;
- park-request interruption of long host awaits (the await-boundary safe
  point of section 1.2);
- fire drain and quiesce logic;
- `decline_park` for zero-yield or grace-deferred cases;
- `report_suspended` after committed D2H;
- split `park_until_restored` into restore-grant wakeup and restore-completion
  reporting;
- wait-all leave behavior verified;
- use `acquire_or_self_suspend` from both fire paths.

Tests:

- a selected victim observes a park request;
- it submits no new fire after quiescing;
- it leaves wait-all while parked;
- requester self-suspend breaks the page-holder deadlock;
- the oldest process never self-yields;
- cancellation at every state reaches the single teardown funnel.

Exit criteria:

- `PIE_KV_PREEMPT_ACTIVE=1` produces real `suspends > 0` backed by D2H and
  reclaimed GPU IDs, not merely waiter parking;
- the legacy direct-release drain path is disabled for D2H-suspended
  processes (no wake without a restore transaction), and the flag stays
  development-only until Phase 5 lands;
- the `restores` counter increments at restore completion, not at wake.

### Phase 5: Implement restore transactions and H2D

Deliverables:

- `prepare_restore`, `commit_restore`, and `abort_restore`;
- FCFS restore selection by `submit_seq`;
- all-or-nothing GPU allocation via grants;
- `park_until_restore_granted`, `report_restored`, and
  `report_restore_failed`;
- call and await the existing `copy_h2d` trampoline;
- atomically publish remapped GPU IDs;
- wire the flat-table version consumer end to end: the fire path currently
  discards the version returned by `flat_table`, and after a restore the
  remapped IDs must provably reach every launch translation and any
  device-side cache;
- release host slots only after successful publication;
- mark the process runnable only after H2D and mapping publication complete;
- explicit restore error policy.

Recommended restore error policy:

- transient lack of capacity leaves the process suspended and requeues it;
- driver copy failure retries with a bounded policy;
- repeated or non-retryable driver failure terminates the process with an
  explicit restore error while releasing all owned resources.

Tests:

- termination of A creates space and restores B before C;
- H2D failure leaves B safely swapped and retryable;
- restored flattened mappings contain only the newly allocated GPU IDs;
- generated tokens match a non-preempted reference;
- host slots return to baseline after restore.

Exit criteria:

- a preempted process resumes from the same logical state and completes
  correctly.

### Phase 6: Gate run-ahead and make scheduler waiting event-driven

Deliverables:

- use `contended()` and `fire_retired()` to stop deepening;
- ensure every lane drains its own settled fires under pressure;
- add blocked preparation state;
- remove contention waits from the transient retry budget;
- cancel blocked waits and grants on scheduler shutdown;
- prevent duplicate acquire tasks per process.

Tests:

- sustained run-ahead cannot pin the entire pool indefinitely;
- no preparation exceeds retry limits merely because it waited for memory;
- a stopped scheduler returns every outstanding reservation;
- readiness and wait-all behavior remain correct.

Exit criteria:

- long contention periods are event-driven and bounded in CPU usage;
- pins and pending transactions cannot silently prevent progress.

### Phase 7: Observability and worker load reporting

Deliverables:

- export queue, grant, swap, D2H, H2D, and latency metrics;
- structured tracing for every state transition;
- include pool and contention state in model status;
- compute `kv_pressure_bucket` from actual resident usage, pending grants,
  waiters, and swap pressure;
- report real inflight process or sequence counts;
- add one diagnostic dump of ordered waiters and suspended processes.

Exit criteria:

- an operator can explain why a process is waiting, suspended, or not restoring
  without attaching a debugger.

### Phase 8: End-to-end hardening

Deliverables:

- deterministic dummy-driver swap support for host CI;
- non-ignored runtime integration tests;
- CUDA over-capacity test with forced small pools;
- failure injection for D2H, H2D, cancellation, and driver shutdown;
- long-running stress test with fork, slice, prefix sharing, and run-ahead;
- leak assertions for GPU IDs, pending recycle IDs, host slots, waiters, and
  grants.

Required scenarios:

1. Two processes each need more than half the pool.
2. Three processes suspend in reverse submit order and restore in submit order.
3. The oldest process completes and its termination unlocks the next restore.
4. A victim is grace-blocked by an in-flight fire, then suspends after retire.
5. The requester must self-suspend to break a deadlock.
6. Shared canonical prefixes remain correct across suspend and restore.
7. D2H fails once and rolls back.
8. H2D fails once and retries without losing host data.
9. A parked process is cancelled.
10. A request exceeds total device capacity and fails loudly.
11. An older suspended process restores ahead of a younger parked waiter.
12. A victim idle in a host await honors a park request.
13. An RS-holding process suspends its KV, keeps RS resident, and resumes
    with correct recurrent state.

Exit criteria:

- all correctness tests pass without relying on timing;
- stress tests show no deadlock, livelock, starvation, or resource leak;
- restored output is equivalent to the accepted non-preempted reference.

### Phase 9: Platform expansion

After the single-GPU CUDA path is stable:

- add tensor-parallel swap plans covering every rank;
- make suspend and restore atomic across ranks;
- add RS host backing and domain-parameterized `copy_state`, lifting the v1
  KV-only suspend scope and giving the RS pool its own ladder;
- on Metal unified memory, evaluate whether copies are needed at all: device
  pages are host memory, so suspend may reduce to residency bookkeeping
  rather than data movement;
- replace the singleton with per-model/per-driver orchestrators;
- evaluate CUDA VMM or Metal sparse mapping to preserve stable physical virtual
  IDs while changing backing;
- add cold replay when host swap is full;
- consider remote KV backing only after local correctness is proven.

## 6. Detailed State Machines

### 6.1 Process state

```text
Running
  |
  | victim selected
  v
ParkRequested
  |
  | process reaches safe point
  v
Quiescing
  |
  | all fires retired, suspend plan prepared
  v
Suspending
  |
  | D2H + metadata commit
  v
Suspended
  |
  | oldest eligible restore receives grant
  v
Restoring
  |
  | H2D + mapping publication
  v
Running
```

Failure and cancellation transitions:

- `ParkRequested -> Running` through `decline_park`;
- `Suspending -> Running` through suspend rollback;
- `Restoring -> Suspended` through restore rollback;
- every state may transition to `Terminating`;
- termination owns final cleanup and queue removal.

### 6.2 Suspend page state

```text
Resident
  |
  | host slot reserved, metadata pinned
  v
SuspendPrepared
  |
  | D2H success
  v
Swapped

SuspendPrepared -- D2H failure --> Resident
```

The GPU page is not recyclable in `SuspendPrepared`.

### 6.3 Restore page state

```text
Swapped
  |
  | GPU ID granted, metadata pinned
  v
RestorePrepared
  |
  | H2D success + metadata publication
  v
Resident

RestorePrepared -- H2D failure --> Swapped
```

The host slot is not reusable in `RestorePrepared`.

## 7. Locking and Ordering Rules

The implementation must preserve these rules:

1. Never await while holding:
   - `ContentionOrchestrator::inner`;
   - a `KvStore` mutex;
   - a process ResourceTable borrow;
   - a scheduler queue lock.
2. Never call a backend operation while holding the orchestrator lock.
3. Reserve resources before starting a copy.
4. Publish metadata only after copy completion.
5. Recycle the source resource only after publication and epoch safety.
6. Acquire locks in one documented order when more than one synchronous lock is
   unavoidable.
7. Do not invoke `on_blocks_freed` while holding a store lock.
8. A suspend or restore transaction has one owner and one terminal outcome.
9. Drop-based rollback is allowed only if it is explicit, testable, and cannot
   silently ignore a driver operation still in flight.

## 8. Failure Policy

### Impossible allocation

If one forward allocation requires more pages than total device capacity, fail
immediately with `ContentionError::Impossible`.

### Continuous exhaustion

If the oldest progress keystone remains continuously unsatisfiable and no
victim or reclaim action can change the condition, fail loudly after the
configured exhaustion deadline.

The watchdog must arm from two places: an `acquire` caller that is the oldest
requesting process, and a restore head that is the oldest non-terminated
process and stays continuously unsatisfiable. A suspended keystone issues no
allocation requests, so an acquire-side clock alone can wedge silently
(section 3.16).

The documented `PIE_KV_EXHAUSTION_MS` must either be implemented or removed from
comments. The same applies to restore-aging configuration.

### Host swap exhaustion

For the first version:

- do not partially suspend a process;
- try another eligible victim;
- if no victim can fit in host swap, park or fail through the normal exhaustion
  policy;
- emit a distinct metric and diagnostic reason.

### Copy failures

- D2H failure: rollback to resident and continue or select another victim.
- H2D failure: remain swapped and retry with a bounded policy.
- repeated hard failure: terminate only the affected process, release its host
  slots, and wake other waiters.

### Process cancellation

Cancellation must:

- remove allocation and restore requests;
- return unconsumed grants;
- abort or await any in-progress copy safely;
- release resident and host-backed pages;
- wake the next eligible waiter or restore candidate.

## 9. Rollout Strategy

### Stage 1: Development-only active path

- require `PIE_KV_CONTENTION=preempt`;
- require `PIE_KV_PREEMPT_ACTIVE=1`;
- require CUDA, TP=1, model 0, driver 0;
- fail bootstrap if required driver capabilities are absent.

### Stage 2: Opt-in production experiment

- expose complete metrics;
- cap host swap size;
- retain the passive backend as fallback;
- add a kill switch for active preemption;
- compare throughput, latency, swap traffic, and error rate.

### Stage 3: Default local contention policy

Make preemption the default only after:

- deterministic CI coverage exists;
- CUDA stress tests pass;
- no resource leaks are observed;
- restore ordering is proven;
- failure injection is clean;
- passive wait-only mode remains available for diagnosis.

## 10. Definition of Done

The north star is implemented when all of the following are true:

- pool exhaustion selects and requests a real victim;
- the victim reaches a safe point and performs D2H;
- an idle victim blocked in a host await honors a park request;
- copied GPU pages are safely returned to the pool;
- older blocked inferlets receive allocation grants before younger ones;
- suspended inferlets are selected for restore by original `submit_seq`;
- an older suspended inferlet is never starved by younger allocation waiters;
- free space from process termination triggers the next eligible restore;
- H2D completes before the process becomes runnable;
- restored logical mappings, hashes, and generated output are correct;
- shared prefixes are neither duplicated incorrectly nor freed while live;
- run-ahead cannot indefinitely pin reclaimable pages;
- cancellation and copy failures leak no GPU IDs, host slots, grants, or queue
  entries;
- contention waiting is event-driven rather than retry polling;
- production metrics expose every wait, suspend, copy, restore, and failure
  transition;
- the behavior is covered by deterministic unit and integration tests.

## 11. Recommended Immediate Work Order

The shortest safe path to visible progress is:

1. Fix ordering (one `submit_seq` grant order across waiters and restores,
   the two-domain keystone definition) and add reservation-backed allocation
   grants.
2. Add runtime host-slot ownership and resident/swapped metadata.
3. Implement transactional D2H suspend over a synthetic process inventory.
4. Wire the process safe point and `should_park`.
5. Switch live allocation paths to `acquire_or_self_suspend`.
6. Implement transactional H2D restore ordered by `submit_seq`.
7. Gate run-ahead and replace contention retries with blocked preparation.
8. Add deterministic CI tests and CUDA stress validation.
9. Add observability and real worker pressure reporting.
10. Expand beyond single-GPU CUDA only after the core invariants are proven.

The critical path is not the CUDA copy kernel or driver ABI. Those primitives
largely exist. The critical path is the runtime bridge: process-safe quiescence,
host-slot ownership, nonresident KV metadata, transactional suspend/restore,
and reservation-backed FCFS scheduling.
