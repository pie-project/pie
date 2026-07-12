# Runahead Implementation: Review Follow-ups

Scope: commit `d58baaf5` (runahead implementation, plan rev 2 in
`runahead-plan.md`) plus the uncommitted `worker.rs` shutdown hunk.
Method: 8 independent finder angles, 39 candidates, every survivor
adversarially verified against the code (CONFIRMED unless marked otherwise).
Line numbers refer to the tree at `d58baaf5`.

Overall: the plan's core machinery (tickets, RETRY, pending overlay, RS
exclusion, escalation, wave participation) is implemented and its happy path
verified correct. Every defect below is an edge: error paths, rebind,
requeue ordering, teardown lifecycles, and backend conformance. Fix order:
P0 items are all of the same class (a transient or normal event escalates
into permanent pipeline death) and should land before anything else.

---

## P0: pipeline-killer correctness

### P0-1. Ticket reservation leaks on submit error paths
`runtime/engine/src/pipeline/fire.rs:346` (reserve) vs `:597` (rollback)

`reserve_channel_tickets` mutates the shared `ChannelCell` counters at
fire.rs:346, but only the synchronous submit-failure path (:597) calls
`rollback_channel_tickets`. Every early return between them leaks the
reservation: `claim_pipeline_scope` failure (:369-373), `fire_lease` failure
(:374-377), `rs::prepare` error returning `Ok(Err(..))` (:498-500), and the
`?` on resource fetches (:366, :469). The same pattern exists on the
device-geometry path (:1153-1170).

Failure: a transient `rs::prepare` OutOfPages leaves `device_reserved_head`
one ahead of any fire that will ever exist. Every subsequent fire on that
channel fails the device precondition (`expected_head != actual_head`),
RETRYs to `max_fire_retries` (1024), and the pipeline dies FAILED. A
transient, expected-under-runahead error becomes permanent death.

Fix: make the reservation an RAII guard (`TicketReservation` that rolls
back on `Drop`, disarmed on successful enqueue). Do not try to enumerate
error paths by hand; new early returns must be leak-free by construction.

### P0-2. Cross-pass cursor floor reads the overlay before it exists
`runtime/engine/src/pipeline/fire.rs:393` (floor) vs `:516-534` (deferred prepare)

The floor reads `visible_token_len` (committed + pending overlay) at SUBMIT
time, but host-geometry `kv::prepare`, which registers the overlay entry
via `track_pending_write` (fire/kv.rs:353), is deferred into the scheduler's
`LaunchPreparation` closure and runs at DISPATCH time. A fresh pass starts
at `committed_tokens: 0` and nothing serializes prepare across passes
(`fire_lease` is a non-exclusive counting lease).

Failure: guest runahead submits decode pass2 while prefill pass1's fires
are queued but unprepared. pass2's floor misses pass1's tokens, and the
too-low `committed_tokens` is captured into pass2's deferred prepare
closure (:529). pass2 prepares against a shorter context or appends over
pass1's slots: silent wrong KV geometry or a context-projection Fatal.

Fix: split extent reservation from mapping registration. At submit (in
submit order) record the WS-level reserved token extent (pure cursor
arithmetic, no allocation); `visible_token_len` includes reserved extents;
page mappings continue to register at prepare. Alternatively floor against
a per-WS submitted-cursor maintained at submit. Either way the invariant
is: any later submit on the same WS sees every earlier submit's extent
immediately (plan I2).

### P0-3. Channel re-bind resets ticket counters against a persistent ring
`runtime/engine/src/pipeline/channel.rs:272-274`

`bind()` zeroes `device_reserved_head/tail` whenever `attachments` is empty
(cold re-attach), but the driver-side ring endpoint persists across
detach/re-attach: forward.rs:307-309 reuses the existing endpoint,
`detach()` (:302-305) never clears it, and `close_native` never calls
`close_channel`. The deleted `reader_reserved_tail = max(visible_tail)`
resync was not re-established for the new counters.

Failure: instance A commits k fires on a host-visible channel and closes.
Instance B re-attaches the same cell: counters restart at 0 while the
device head word is k. B's first fire ships `expected_head = 0`, the device
check (0 != k) fails every attempt, perpetual RETRY, escalated FAILED, even
though the put is present.

Fix: on bind, when an endpoint already exists, initialize
`device_reserved_head/tail` from the endpoint's live head/tail words; keep
the 0/seeded initialization only when a fresh ring is registered.

### P0-4. OutOfPages requeue reorders same-WS prepares
`runtime/engine/src/scheduler/worker.rs:921`

A retryable dispatch preparation is requeued with
`pending.push_back(QueuedItem::Prepare(request))`; the dispatch loop only
inspects `pending.front()` with no per-pipeline ordering guard, so a later
fire of the same pipeline/working set prepares first.

Failure: pipeline submits fire A then B on one WS. A's `kv::prepare` hits
OutOfPages and goes to the back; B prepares first. B's cursor already
counts A's pages but A's overlay is unregistered, so
`context_pages.len() < valid_pages` trips `KvError::Fatal`
(fire/kv.rs:295-300) and the pipeline dies FAILED under transient memory
pressure. (Note: the `track_pending_write` seq debug_assert does NOT catch
this; seq is assigned in prepare order, so it stays monotonic. The Fatal is
the real manifestation.)

Fix: requeue at the FRONT (the `queue_attempt_front` helper exists for
exactly this ordering), or requeue per-pipeline so only other pipelines'
items can overtake. Head-of-line blocking within one pipeline is correct:
the overlay invariant is per-WS submit order.

---

## P1: correctness

### P1-5. One device-geometry member's miss poisons the whole batch
`driver/cuda/src/pipeline/dispatch.cu:793` -> `compose.cpp:162-164` -> `context.cpp:1148-1152`

`resolve_descriptors` still runs the eager `writer_inputs_available` check
per member; on failure compose throws before composition and
`settle_failed_launch` iterates ALL instances, poisoning every host-visible
channel and publishing FAILED to every terminal cell. Under quorum
batching, pipeline X's late geometry put permanently kills unrelated
pipelines Y and Z in the same wave.

Fix (plan §4.6 near-term): gate at composition in the runtime; a
device-geometry member whose geometry channels are unready is excluded
from the wave (straggler treatment, requeued as RETRY) so the driver never
sees it. Interim mitigation if the compose restructure is deferred: scope
`settle_failed_launch` to the offending instance instead of the batch.

### P1-6. Completion-lease acquire races close; pacing wait-id freed while live
`runtime/engine/src/driver/instance.rs:63-76`

`acquire_completion_lease` pushes and `fetch_add`s `active_leases` with no
`close_requested` recheck. Its twin, `KvLifecycle::acquire_fire_lease`
(store/kv/working_set.rs:96-102), re-loads the close flag after the
increment and backs out; instance.rs never received that hardening. Nothing
serializes the guest-thread `reserve_completion` (fire.rs:331) against the
worker-thread close (worker.rs:1130, :1369, :1399-1416); the
`in_flight != 0` guard (:1123) does not cover the reserve-before-dispatch
window.

Failure: `maybe_finalize` observes `active_leases == 0` before the
increment lands, sweeps and frees the wait id, then the lease goes live; a
freed-and-reused waker slot delivers wakes to the wrong waiter.

Fix: copy the working_set.rs pattern (post-increment recheck + back-out).
Better: extract one shared lease-counted-finalize primitive both files use
(see also P3-2), so hardening lands in one place.

### P1-7. pipeline_drop blocks on a retrying fire, holding pins
`runtime/engine/src/pipeline/fire.rs:759-778` (drop), `:740-757` (close), `:873` (await)

Drop/close loop `pop_front` + `finalize_op(..).await`; `finalize_fire`
awaits the `WorkItemCompletion`, and RETRY keeps it Pending
(completion.rs:606-609 resets the terminal without resolving). The only
fast-fails are scheduler shutdown and the 1024-retry cap; dropping the
pipeline does not stop requeueing.

Failure: the producing peer exits so the put never arrives; the guest drops
the consumer pipeline; the drop stalls for up to ~1024 full-batch rounds
with the fire's KvTxn/ws_guard pinning pool pages.

Fix: drop-time cancellation. Dropping/closing the pipeline marks its
in-flight LogicalFires cancelled (a flag on the shared completion); the
scheduler's requeue point (worker.rs:1315) sees the mark, stops requeueing,
and rejects the completion so the drop's await resolves immediately.

### P1-8. Terminal-cell quarantine is an unbounded leak
`runtime/engine/src/driver/completion.rs:62-68` (alloc), `:95` (push-only pool)

Every distinct work item Box-allocates a fresh `OwnedTerminalCell`; Drop
pushes it into the global quarantine SegQueue, which nothing ever pops. The
quarantine is deliberate (ABA safety, comment :53-55) and retries reuse the
same cell via `terminal.reset()`, but there is neither reuse nor a bound:
memory grows with total work items for the life of the process.

Fix: epoch-deferred recycling. A quarantined cell becomes reusable once its
settlement is confirmed retired (the same epoch discipline as
`recycle_after_epoch` on KV pages); pop from the pool in
`OwnedTerminalCell::new` once the head entry's epoch has passed.

### P1-9. Metal backend silently ignores the runahead contract
`driver/metal/src/context.cpp:542` (outcomes), `:462-471` (eager reject)

`run_launch_job` initializes outcomes to SUCCESS and only ever assigns
FAILED; no `PIE_TERMINAL_OUTCOME_RETRY`, no reads of
`channel_expected_head/tail`, `channel_ticket_indptr`, or `retry_eligible`
anywhere in driver/metal; the deleted submit-time availability projection
survives as a synchronous reject. The runtime ships the contract
unconditionally (abi.rs:245) and its requeue/escalation machinery waits on
RETRYs that never come. There is no capability gate, so the divergence is
silent.

Fix: either implement the contract on Metal (RETRY outcome + ticket
preconditions in the launch job) or add a driver capability flag reported
at init; a non-conformant backend keeps eager-validation semantics
EXPLICITLY (runtime skips ticket emission, restores submit-time validation
for that backend, and errors clearly on programs that need lazy puts).
Silent divergence is the bug; pick one.

### P1-10. Channel-RETRY escalation is count-only; permanent causes burn 1024 batches
`runtime/engine/src/scheduler/worker.rs:1325` (cap), `:75-87` (thresholds)

`retire_ready_launches` fast-fails only on shutdown (:1317) and RS
ineligibility (:1321). A retry-eligible fire whose put never arrives
re-enters every wave: one warn at 32, FAILED only past 1024. The plan
(§4.4) requires a classified non-transient fast-fail; it exists only for
the OutOfPages preparation path (:913-927).

Fix: classify the blocking cause. The runtime observes channel endpoints:
if the blocked channel's writer endpoint is closed/never-registered (the
producer is gone), fail immediately; otherwise optionally gate the requeue
on put-arrival instead of blind-requeueing. Keep the count cap as the
backstop, not the primary mechanism.

---

## P2: latent hazards (not currently reachable or currently converging, fix before they bite)

### P2-1. RS-ineligibility policy is duplicated in the CUDA driver
`driver/cuda/src/pipeline/dispatch.cu:113-114` vs `runtime/engine/src/scheduler/worker.rs:1321-1324`

CUDA computes `failed = poison || (!committed && !retry_eligible)`, coercing
an RS non-commit to FAILED in the driver; the dummy driver returns RETRY
unconditionally and lets the runtime coerce. Both currently converge to
guest failure, but the policy lives in two layers and two backends emit
different terminal outcomes for the same event (diverging diagnostics and
escalation accounting; when versioned RS lands, the driver must be edited
too or eligible work fails silently).

Fix: delete the `retry_eligible` term from the CUDA classification; the
driver always reports RETRY on benign non-commit; worker.rs:1321 stays the
single policy point. Drop `retry_eligible` from the wire once nothing
device-side reads it.

### P2-2. force-retry test hook bypasses the real predicate and getenvs per fire
`driver/cuda/src/pipeline/dispatch.cu:56` (global), `:570-578` (hook)

`std::getenv("PIE_CUDA_FORCE_RETRY_ONCE")` is called uncached inside the
per-program fire loop, and the hook `cudaMemsetAsync`s the commit flag to 0
AFTER `k_pull_validate` already ran the real ticket check. Tests using it
prove only "commit==0 => RETRY", never the device ticket predicate; the
"once" is process-global so a second in-process test cannot re-trigger and
the affected pipeline is nondeterministic.

Fix: cache the env read once at `Impl` construction. Replace the memset
with a fault that exercises the real path (e.g. offset the target fire's
`expected_head` by one so `k_pull_validate` itself refuses), and make it
per-launch like the dummy driver's `retry_launches_remaining` option.

### P2-3. Ticket CSR under-supply passes host ABI validation
`interface/driver/src/local.rs:948` (validate_csr), `:1241-1245` (presence guard)

`validate_csr` enforces monotonic and `last <= values_len` but not
`last == values_len`, and the presence guard never checks the final indptr
offset equals `channel_expected_head.len()`. CUDA and dummy catch the
per-program count mismatch later; Metal ignores tickets entirely, so on
that backend a malformed CSR mis-attributes silently.

Fix: enforce `last == values_len` for the ticket CSR (and
`channel_expected_head.len() == channel_expected_tail.len()` == final
offset) at the boundary, where the error is attributable.

### P2-4. Silent driver-side ticket synthesis fallback
`driver/cuda/src/pipeline/dispatch.cu:174-214`

When the batch-level `supplied` shape check fails, `build_channel_tickets`
silently synthesizes expected head/tail from the LIVE ring words, making
the device precondition trivially true for the whole batch. Verified
unreachable today (scheduler/batch.rs:126, :145-167 always emits a
shape-correct CSR), but it is exactly the deleted availability projection
living on as dead code, and it converts a future composer bug into a silent
ordering-guarantee loss.

Fix: replace the fallback with a launch rejection (loud error). The plan's
rule is "the driver checks, never assigns" (§4.6).

### P2-5. Batch grouping rules duplicated between peek and dispatch, already diverging
`runtime/engine/src/scheduler/worker.rs:797-847` (peek) vs `:1160-1222` (dispatch)

Same-instance dedup, mask-solo, and capacity are implemented twice (the
comment at :789-796 admits they must be kept in sync). They already
diverge: dispatch tracks `batch_has_prebuilt` (dead via `let _ =` at
:1215), peek omits prebuilt entirely, and the capacity checks go through
two different code paths (raw limits vs `BatchAccumulator`). Peek feeds
`decide_wave_at`; drift makes quorum size disagree with dispatched
geometry, mis-counting stragglers and retries.

Fix: extract one shared grouping predicate/simulation consumed by both;
delete the dead prebuilt tracking.

---

## P3: cleanup (verified duplication / hot-path waste)

Per-token hot path unless noted. Each is independently landable.

1. **Ring-word layout literals** — `dispatch.cu:91-99` and ~10 sites in
   `channel_registry.hpp` (171, 303, 306, 347, 351, 353, 361, 369, 418,
   438) hardcode `head=0, tail=1, poison=2, closed=3`. Define shared named
   constants; the two finalize paths must not be able to disagree.
2. **WakerTable teardown triple** — `instance.rs:80-110` and
   `channel.rs:161-165` both hand-roll sweep + per-id deregister + free.
   One shared teardown helper (also the natural home for the P1-6 lease
   primitive).
3. **Mask bit-packing reimplementation** — `driver/abi.rs:66-77` expands
   `RunMask` via `to_vec()` then repacks LSB-first u32; use
   `runtime/grammar/src/bitmask.rs` (`bitmask_size`/`set_bit`) or pack
   straight from the runs, keeping the compressed form.
4. **Pending-overlay scan duplication** — `kv.rs:686-708` `pending_target`
   and `pending_page` are identical reverse double-scans differing only in
   field; `visible_flat_table`/`visible_chain_state`/`visible_token_len`
   re-encode the same newest-first rule. One generic
   `latest_pending(ws, index, accessor)` helper.
5. **Stored derivable flag** — `worker.rs:124-137` caches `retry_eligible`
   from the retained request's RS fields; derive on demand from `request`
   so the rule cannot diverge from the payload.
6. **Quorum double bookkeeping** (PLAUSIBLE) — `quorum.rs:92-93`
   `retry_participating` + per-pipeline `in_flight` mirror scheduler queue
   state for the demotion filter (:223); drift leaves a pipeline
   permanently un-demotable. Derive from the queues or centralize the
   increments.
7. **queue_attempt_back/front near-duplicates** — `worker.rs:765-787`
   encode copy-before-launch twice with inverted push order; one helper
   with an end parameter.
8. **max_in_flight indirection** — `quorum.rs:46-48` is a pass-through to
   `configured_max_in_flight()` with no other caller; fold it.
9. **find_publish_ticket linear scan** — `dispatch.cu:256-265`, called per
   output per fire (:602-605): O(outputs x tickets). Build a slot index
   when the ticket vector is assembled.
10. **channel_accesses per fire** — `fire.rs:178-201` rescans every
    stage/op/port per submit (:322) to rebuild a static mask; compute once
    at bind and store on the bound program.
11. **visible_flat_table clones per token** — `kv.rs:584-603` clones the
    whole committed table per `prepare_write`; `fire.rs:1054-1056`
    materializes it just for `.len()` inside the device-geometry retry
    loop. Add `visible_len()`/`visible_page_at()`; reserve the clone for
    the CoW-tail slice that needs it.
12. **build_channel_tickets rebuild per launch** — `dispatch.cu:561-562`
    re-queries static per-instance fields every fire; cache a per-instance
    ticket template at bind, patch expected_head/tail per fire (preserve
    the `apply_sequence_ticket` side effect at :233).
13. **Double allocation in ticket reservation** — `fire.rs:346-348`
    (and :1153-1155) unzips into two Vecs then clones both into `req`,
    keeping the originals only for the rare rollback; move into `req` and
    borrow back on the error path. (Subsumed if P0-1's RAII guard owns the
    values.)

---

## Appendix: claims investigated and REFUTED (do not re-flag)

- **Peek-port channels lack arrival checks** — covered independently of
  tickets: `requires_channel_input` (program_runtime.hpp:293) marks any
  host-visible writer channel, dispatch.cu:230 sets `kTicketRequireInput`,
  and `k_pull_validate` clears `pass_commit` when `tail <= head`
  (channels.hpp:194-206). A late mask yields RETRY, not a garbage commit.
- **visible_flat_table page-0 gap fill as a write target** — writes cannot
  reach the fills: `prepare_write` enforces fresh contiguity (kv.rs:357-364)
  and `privately_writable` errors `Unwritten` for unmapped indices
  (page_table.rs:413-418); fills remain read-only masked candidates.
- **Device-geometry fresh boundary over pending state** — a prior fire's
  hard failure forces the current fire's `success = false` via the
  pipeline-shared failure domain and FIFO finalization (fire.rs:874-891),
  so the misclassification cannot commit.
- **Driver ticket-synthesis fallback as a live hole** — unreachable today
  (batch.rs always ships a shape-correct CSR); kept as P2-4 to convert to
  an error.
- **Uncommitted worker.rs shutdown hunk** — benign: `stopping` is monotonic
  (set at worker.rs:694/607/660, never cleared), `reject()` settles the
  completion so guest-side awaits return promptly, and the un-rolled-back
  tickets leak only into process teardown.
