# KV contention: intent and redesign

This document is about **why**, not **how**. The implementation hints at the end
are deliberately thin — the shape of the solution follows from the reasoning, and
anyone who understands the reasoning will produce a better plan than a checklist
would.

## What the system promises

KV-cache contention is handled by **preempt/restore, not admission**. That is a
standing directive, already recorded in `runtime/engine/src/store/reclaim.rs`:
`max_concurrent_processes` is a large physical safety cap only. A process is
never refused up front for fear it might need pages later.

This promise is what makes `WorkingSet::reserve` *logical*. Reserving does not
allocate device pages; physical demand only materializes at forward preparation.
The pool is therefore **intentionally overcommitted**, and the reclaim ladder is
what makes overcommit safe.

## Where it actually breaks

The ladder has five rungs. Only rung 0 is connected.

| Rung | Intent | Wired? |
|---|---|---|
| 0 | Drop cache-root leases nothing live reaches — costs no work | **yes** |
| 1 | Victim loop: suspend the YOUNGEST running process, protecting the FCFS-oldest | no |
| 2 | Unified grant order: blocked allocations and suspended restores compete by `submit_seq` | no |
| 3 | Restore-on-free: the oldest eligible entry proceeds | no |
| 4 | Exhaustion endgame: fail loud rather than wedge | no |

`ensure_host_backing` ([fire.rs](runtime/engine/src/pipeline/fire.rs)) attempts an
allocation, drops idle cache leases, retries once, and surfaces `OutOfPages`. That
retry *is* rung 0. Everything above it — the victim loop, the FCFS grant order,
restore, the endgame — is implemented, unit-tested, and **unreachable from the
fire path**, because nothing in production ever calls
`ContentionOrchestrator::acquire*`. The same is true of the second allocation
seam, `prepare_explicit_kv`.

So the failure is not "a feature is missing." It is: **we built the machine and
never connected the throttle.** Under overcommit — the regime the design exists
to serve — processes die with `OutOfPages` instead of taking turns.

This is a liveness/correctness gap, not a performance one, and it is worse than
it looks because `reserve` is logical: the guest gets no early signal. It reserves
successfully, runs for a while, and fails mid-generation at a moment that depends
on which peers happen to be co-scheduled.

## Why the obvious fix is wrong

The obvious fix is to `await orchestrator.acquire(...)` where `OutOfPages` is
discovered, then retry. This was tried twice and reverted, and understanding why
is the whole point of this document.

At the moment exhaustion is discovered, the preparation already holds a
working-set fire lease, open RS transactions, and an open KV transaction. Those
hold **pins**. And pinned pages are precisely what the victim loop cannot
reclaim — `SuspendOutcome::DeferredByGrace` exists to say so.

Three consequences follow, and they are all fatal:

1. **The requester waits while holding what it is waiting for.** It pins its own
   pages and then blocks. In the corner where the requester is itself the only
   viable source of pages, this cannot resolve.
2. **The requester cannot self-suspend.** `Acquired::SelfSuspendFirst` is a
   protocol — suspend, report, park, restore, retry — and it requires the process
   to reach a state where its pages are unpinned. A process sitting inside fire
   preparation is not in that state, so `prepare_suspend` declines and the
   protocol never completes. The second experiment's 56-suspend/56-restore thrash
   with a blocked receiver is exactly this shape: `SelfSuspendFirst` was treated
   as a retry hint rather than a protocol.
3. **There is nowhere to put the answer.** An `AllocationGrant` is a *concrete
   reservation* — physical page ids already removed from the pool, owed back
   exactly once. `ensure_backed` allocates from the pool itself; it has no
   variant that accepts pre-reserved ids.

Underneath all three is one structural fact: **resource acquisition is
interleaved with work, so the failure that requires waiting is discovered late,
after irreversible state exists.** Rollback is hand-copied at four separate error
sites, which is the same fact viewed from a different angle — there is no single
object whose disposal undoes an attempt, so "abandon and retry" is not
expressible.

Bolting `acquire()` onto the leaf therefore does not add a feature; it adds a
second, competing lifecycle around fire preparation.

## Decisions, and why

### Contention is resolved by preemption, not admission

Unchanged, and restated because it constrains everything else: we do not refuse
work up front. Admission control would make the problem trivial and the product
worse.

### The victim is the youngest; the oldest is protected

Already the policy, and worth preserving deliberately. The FCFS-oldest process is
the *completion keystone*: never a victim. Protecting the first-comer guarantees
someone always finishes, which is what prevents the fleet from livelocking under
sustained pressure. Restore runs oldest-first by original `submit_seq`, so the
order in which work is given back mirrors the order it was promised.

### Drop-to-replay is rejected

The tempting answer to "host memory is also full" is to discard KV — it is a
cache, recomputable from tokens — and replay on resume. **This is architecturally
excluded, not merely difficult.**

The host does not know what produced the KV. Only *seeded* channel values are
host-known; every other channel is device-derived and reads as unknown to the
host, with the driver filling those ports itself. That is not an oversight — it
is the reason run-ahead is fast. Tokens generated in an epilogue are fed back on
device without a host round-trip, and adding provenance tracking means adding
that round-trip back, destroying the property we built the design around.

Cooperation does not save it either. Even an inferlet that emits every output
token gives an incomplete ledger: speculative decoding writes KV for draft tokens
that are then rejected and never surface as output, and beam search prunes beams
whose page layout is the guest's own fork/prune computation. **Output tokens are
not the write history**, and replay needs the write history.

For hybrid/linear-attention models the cost compounds: recurrent state is a fold
over the entire sequence, so nothing can be recomputed from a suffix.

### When host memory is also exhausted, kill

Given replay is out, the remaining options were compress-then-spill (quantize on
D2H, or add a tier below host) and kill. **We choose kill, for simplicity.**

The reasoning is proportion, not principle. Host swap capacity is a configured
quantity and host RAM normally dwarfs device memory, so host exhaustion is a far
rarer regime than device exhaustion. Spending design budget on a graceful answer
to the rare case, while the common case is still unwired, is the wrong order.
Kill is honest, bounded, and returns both device pages and host slots at once. If
host-full ever becomes a regime we actually live in, compress-then-spill is the
upgrade path — and it inherits the one property that matters: like the existing
D2H stash, it moves **opaque bytes** and needs no provenance.

### Self-suspend leaves the critical path

This is the payoff of choosing kill, and it is the single largest simplification
available.

Self-suspend exists to avoid losing the requester's work when no other process
can yield. If killing is permitted in that corner, the ladder can suspend
cooperative victims and kill uncooperative ones, and **the requester never has to
suspend itself**. Killing a victim requires no cooperation from it: no waiting for
a host-call boundary, no waiting for pins to drop. The hardest distributed state
machine in the design stops being on the critical path.

Note this is a decision about *policy*, not about the plumbing below. It removes
the worst obstacle; it does not by itself make the wiring easy.

## The redesign, and what it buys

The plumbing difficulty is not the orchestrator. It is *where* the retry boundary
lives. So change that, rather than working around it.

Split fire preparation into three phases:

- **A — compute demand.** Pure. Determine every resource this fire needs. No
  locks, no transactions, no pins.
- **B — acquire once.** A single atomic acquisition, and the *only* point where
  resource failure or waiting can occur. This is the one place that talks to the
  orchestrator.
- **C — build and commit.** With resources already held, construct the fire.
  Resource exhaustion is impossible here by construction.

The entire value of this structure is one property:

> **At the moment we wait, we hold nothing.**

Every obstacle above dissolves as a consequence rather than being solved
individually:

- The self-deadlock disappears — no pins are held while waiting.
- Self-suspend becomes *possible* again, so keeping it or dropping it in favor of
  kill becomes a free policy choice rather than a forced one.
- "Rebuild, don't resume" needs no machinery: phase A is pure, so retrying is
  just calling it again.
- The four hand-copied unwind sites collapse into the disposal of one owned
  object.
- Grant installation stops being a missing API and becomes the natural output of
  phase B — the reservation *is* the grant.
- The channel-ticket LIFO discipline is preserved for free, because nothing is
  partially unwound across an await.

And the fire path's entire contact with contention shrinks to a single
acquisition call. The ladder — idle lease, victim loop, kill, endgame — lives
behind it and the fire path never learns it exists.

## Why this does not cost speed

- The common case gets **shorter**, not longer. Today's allocation seam has a
  two-attempt structure baked in (try, reclaim, retry). Afterwards it is a demand
  computation plus one acquisition, with fewer lock round-trips.
- Demand computation is span arithmetic and a frontier read — it does not scale
  with page count.
- Run-ahead is untouched: preparation is still per-fire and still pipelined. The
  single acquisition point is already serialized by the KV lock today.
- Holding a reservation across the build costs no more residency than the
  transaction pins already cost.

## Why this is possible now

Phase A can only be pure if demand is computable on the host. For device-derived
geometry the host does not know which slots will be written — which would have
sunk this design before.

The attention refactor removes that obstacle. Because the author now declares
`readable-pages` and `writable-pages` explicitly — *per-fire containment
declarations*, in the WIT's own words — the declared writable span is a host-known
**upper bound** on what the fire can touch, even when the geometry itself is
device-resolved. Reserving to the declared bound is exactly what a containment
declaration means.

This is worth stating plainly: **the refactor did not cause this problem, and it
is what makes the clean fix available.** Making geometry and page demand explicit
is precisely what turns demand into something the host can compute up front.

## What we are explicitly not doing

- **Not** admission control.
- **Not** drop-to-replay, now or later.
- **Not** compress-then-spill in this increment (it is the upgrade path if
  host-full ever becomes a real regime).
- **Not** hiding any of this inside the public attention API, geometry, mask
  lowering, or CUDA graph code. This is a lifecycle change and belongs in its own
  workstream.

## Properties that must survive

These are the invariants the design exists to protect. An implementation that
violates one of them has not solved the problem regardless of what the tests say.

- A physical page is in exactly one of: the free pool, a reservation, a live
  mapping, a suspend/restore transfer, or a finalized fire transaction.
- A reservation is installed once or returned once.
- No await while holding a store, orchestrator, resource-table, or channel lock.
- No process is suspended while holding an unabortable preparation.
- FCFS ordering for both allocation and restore remains authoritative.
- Cancellation at any await point — guest termination, pipeline close, preemption
  — removes the waiter and releases anything unconsumed.
- Exhaustion fails loud only after no victim, no kill, and no self-transition can
  make progress.
- Already-submitted fires keep their close/drain semantics.

## Hints

Deliberately brief.

- The rollback boundary comes first. Nothing else is safe to attempt before an
  attempt can be discarded cleanly, and that refactor is most of the work.
- Both allocation seams need the same treatment; fixing one and leaving the other
  reproduces the wedge somewhere less obvious.
- `SuspendOutcome` currently cannot express "host swap is globally exhausted"; it
  collapses into the local "nothing to reclaim here," so the orchestrator walks
  every victim — quiescing each one, which is not free — before giving up. A
  global-abandon signal is a small change with disproportionate benefit, and is
  worth making regardless of everything else here.
- Wire victim-based acquisition before anything that can kill, so fairness and
  cancellation are validated while the blast radius is still small.
- The fleet test `active_preemption_swaps_and_restores_an_over_capacity_fleet` is
  host-only and is **not** `#[ignore]`d, so it is already in the default test run.
  Establish whether it currently fails before planning around it — the starting
  point may be "a red test we are carrying," not "a disabled test to re-enable."
  It should pass repeatedly, not once.
