# KV contention orchestrator: resolved seam and remaining limits

## Status

The previously disconnected KV contention seam is connected in the current
working tree.

Physical KV demand is now computed before fire leases, RS transactions, KV
transactions, completion cells, or channel tickets are acquired. Contended
requests wait in the orchestrator while holding no fire-preparation pins.
Concrete allocation grants are then consumed exactly once by grant-aware KV
preparation APIs.

The active-preemption fleet and zero-host-swap kill policy both pass repeatedly.
These changes are not committed yet.

## Original failure

`WorkingSet::reserve` is logical. Physical pages are materialized later during
forward preparation, so overcommit is intentional.

Previously, all physical allocation paths performed:

1. direct pool allocation;
2. idle cache-root reclaim;
3. one retry; and
4. `OutOfPages`.

No production path called `ContentionOrchestrator::acquire*`, despite the
orchestrator, victim selection, host swap, restore queue, and exhaustion policy
already existing.

The default host-only fleet test therefore failed mid-generation instead of
letting processes take turns.

## Implemented preparation boundary

Fire preparation now has the intended three phases.

### A. Compute demand

Demand computation holds the KV lock only for mapping inspection. It allocates
nothing and opens no transaction.

For ordinary host/decode-envelope fires, demand is:

```text
declaration-realization COW pages
+ missing physical backing through the writable containment bound
```

For explicit/device-geometry fires, demand is the fresh/COW requirement for the
author-supplied write indexes.

The relevant APIs are:

- `KvStore::backing_demand`
- `KvStore::write_demand`
- `fire::kv::realize_declaration_demand`
- `fire::kv::prepare_explicit_demand`

The writable page span is a conservative host-known bound. It is safe but can
reserve more than the exact device-derived write set. That is an intentional
simplicity tradeoff, not a claim that containment and exact demand are the same.

### B. Acquire once

If demand is nonzero and contention mode is enabled, the fire calls the
orchestrator before it owns:

- a `KvFireLease`;
- RS transactions;
- a KV transaction;
- a completion cell; or
- channel tickets.

An allocation waiter temporarily leaves quorum accounting using allocation-wait
semantics: already-accepted scheduler work drains untracked, while a future
request implicitly rejoins. It is not terminated and its accepted fire is not
cancelled.

Unmet allocation waiters have strict priority over restores. A victim cannot
immediately consume through restore the pages it just freed for a waiter.

### C. Build and commit

After a concrete grant exists:

1. acquire the working-set fire lease;
2. prepare RS state;
3. consume the KV reservation through grant-aware preparation;
4. build translation and launch state;
5. reserve channel tickets;
6. submit.

If RS preparation fails, the still-owned grant drops back to the pool. If KV
installation fails, every unused reserved page is returned and any opened KV/RS
transaction is aborted.

Grant-aware APIs are:

- `KvStore::ensure_backed_reserved`
- `KvStore::prepare_write_reserved`
- `fire::kv::realize_declaration_reserved`
- `fire::kv::prepare_explicit_reserved`

Each consumes only its required prefix of a caller-owned reservation. Surplus
remains owned and is returned exactly once.

## Victim and wait-set policy

The FCFS-oldest process remains the completion keystone and is never a victim.

Victims are selected youngest-first, but cooperative suspension is currently
restricted to processes already waiting at the allocation boundary. This is the
point where the implementation can prove they hold no new preparation pins.

This restriction is deliberate. Arbitrary running-process park requests were
tested and remained vulnerable to a last-host-boundary notification race.
Allocation-boundary victims passed the fleet repeatedly and preserve liveness.
A requester with no eligible victim waits for the protected oldest process to
make natural progress.

The state transitions are:

```text
Running
  -> allocation waiter
  -> selected victim / ParkRequested
  -> drain already-accepted FIFO work
  -> process-level Suspend leave
  -> freeze
  -> D2H
  -> Suspended
  -> restore grant
  -> H2D
  -> Running
  -> implicit scheduler rejoin
```

Important ordering fixes:

- accepted FIFO work drains before process-level suspend/freeze;
- waiter notification occurs after `ParkRequested` is published;
- stale park requests are cancelled when no unmet allocation waiter remains;
- restore is considered only after unmet allocation waiters are served.

Fallback polling while contended is capped at 50 ms. Normal notification-driven
progress remains immediate; the poll prevents a lost notification from becoming
a one-second latency step.

## Restore policy

Restore is attempted whenever pages are returned:

- fire finalization;
- working-set/process teardown;
- cache-root reclaim;
- dropped or surplus allocation grants.

Restore proceeds only when:

1. there is no unmet allocation waiter;
2. the oldest eligible suspended process fits without evicting;
3. utilization is below the restore pause threshold, unless aging overrides the
   pause.

Pipelines rejoin only after H2D and mapping publication complete.

## Host swap exhaustion

Active preemption now allows a zero-sized host swap pool.

If a quiesced victim cannot reserve enough host swap slots:

1. record host-swap exhaustion;
2. terminate that victim with an explicit error;
3. wait for native teardown to release its device pages;
4. drain the allocation queue.

Kill does not reuse pages still reachable by native work. The victim is first
drained/quiesced; termination removes guest cooperation from the remaining
teardown but does not bypass GPU lifetime rules.

No replay path is used. Device-derived token/beam/speculative write history is
not reconstructible from host-visible outputs, and RS models make suffix replay
insufficient.

## Cancellation and close

- Dropping an acquisition future removes its waiter and returns any grant.
- Process termination unregisters its allocation/restore entries.
- Pipeline close does not cancel an already-accepted preparation waiter; close
  semantics still require submitted/preparing work to settle.
- Stale victim requests are returned to `Running` once pressure is satisfied.

## Diagnostics

`ContentionDiagnostics` now includes every registered process and its state:

- running
- suspending
- park-requested
- quiescing
- suspended
- restoring

It also exposes current park-request count and existing allocation/restore,
copy, rollback, exhaustion, and pool counters.

## Validation

Repeated successfully:

```bash
for run in 1 2 3; do
  cargo test -p pie-engine --test contention \
    active_preemption_swaps_and_restores_an_over_capacity_fleet --quiet -j1
done

for run in 1 2 3; do
  cargo test -p pie-engine --test contention_host_full --quiet -j1
done
```

The normal fleet verifies:

- all eight processes complete;
- suspension and restoration engage;
- D2H/H2D counts balance;
- queues and suspended state drain;
- host slots and device pages return.

The host-full fleet verifies:

- a zero host swap pool is accepted;
- at least one victim reports the host-swap kill reason;
- another process completes;
- no fleet wedge remains;
- all device pages return.

KV store tests also verify exact demand and reserved-page consumption.

The real CUDA harness now supports process-isolated legacy-roomy,
preempt-roomy, and preempt-constrained profiles. On an RTX 4090 with
Qwen3-0.6B, fleet 8, and 48 output tokens per lane:

- preempt-roomy versus legacy-roomy had median throughput `0.982x` and p95
  latency `1.018x` across four runs;
- an 8-page cap had median throughput `0.429x` and p95 latency `2.334x` versus
  preempt-roomy across four runs;
- a 12-page cap measured throughput `0.525x` and p95 latency `1.906x`.

The severe-pressure cost is not confined to the allocation step. A suspended
pipeline remains outside the runnable quorum until its grant and restore
complete; copy time is visible in diagnostics, while narrower runnable batches
and waiting account for most of the measured wall-time penalty.

## Scheduler identity

Process lifecycle identity and pipeline quorum identity are now separate.
Each `PipelineScope` has a stable scheduler ID:

- pipeline close and allocation wait remove only that scope from quorum;
- process suspend/terminate removes every scope owned by the process;
- multiple allocation waiters from sibling scopes retain independent queue
  entries under the process's FCFS priority;
- completed bind placeholders end before the first concrete scoped fire.

Focused policy, orchestrator, and worker tests cover sibling isolation and
process-wide removal.

## Remaining limitations

1. **RS contention is not integrated with this KV grant.** RS preparation runs
   after KV acquisition and can still fail independently with `OutOfSlots`.
   Existing preemption also keeps RS state resident while KV is swapped.
2. **Victims are safe-boundary-only.** Killing or suspending an arbitrary
   running process after a bounded grace period is not implemented.
3. **Containment-bound reservation is conservative.** A future optimization may
   compute a tighter host-known demand without reintroducing device-to-host
   geometry inference.

These are explicit follow-ups; they are not hidden behind the attention API,
mask lowering, or CUDA graph code.
