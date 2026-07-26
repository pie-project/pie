# Project Kilimanjaro: KV Contention, Rebuilt — Implementation Plan

Date: 2026-07-24

Status: Implementation plan. Supersedes the direction-only draft of the same
date; the philosophy from that draft is compressed into §1–§2.

Execution strategy (operator directive, 2026-07-24): **build-first**. The
entire structural rewrite lands upfront in dependency order, followed by a
polish pass; correctness burn-down and performance proof are trailing stages,
not per-step gates. See §7.

Baseline: `tasks/contention/agents/alpha` @ `45592f9ee`. Line anchors below
refer to this tree; symbol names accompany every anchor because lines will
drift.

Historical companions (removed from the tree at `ef6cad91f` "Remove obsolete
planning documents"; readable via `git show ef6cad91f^:<name>`):

- `contention-plan.md` — original intent; the A/B/C (demand / acquire once /
  build) argument and the drop-to-replay exclusion.
- `KV_CONTENTION_ORCHESTRATOR_GAP.md` — implementation report of the current,
  working seam and its named limitations.
- `next-goal.md` — admission / scalability (M5), the layer above this one.

---

## 1. Thesis (settled)

Contention management is a **liveness/correctness mechanism first**. Under
intentional overcommit its non-negotiable job is: never wedge, never corrupt,
always make progress. Scheduling intelligence — who runs at all — is decided
one layer up, by admission (`next-goal.md` M5 / Vesuvius). The two layers must
stay separate; a clever contention policy is intelligence leaking into the
safety net.

**But dumb policy does not mean a slow machine.** Sustained throughput under
heavy contention is a product goal in its own right. Being the safety net does
not excuse dead time: wasted copies, polling stalls, idle pages beyond the
guarantee's bounded price, and slow suspend/restore turnaround are defects
here, not acceptable costs. The layer never gets *smarter* to get faster — it
gets *leaner*. §6 states the cost model and what "fast under pressure" means
precisely.

**FCFS-by-spawn is the policy, chosen for drainage, not fairness.** Monotone
arrival-age is the only cheap key under which every waiter's rank improves as
time passes, so livelock and starvation are excluded by construction: protect
the oldest as the completion keystone → it finishes → the next-oldest becomes
the keystone. Every key that optimizes a metric (shortest-job,
memory-weighted, LRU) fails this property under sustained arrival — and
optimizes something that is admission's job anyway. FCFS is also the
throughput-right degradation under severe overcommit: it runs the oldest
subset to completion instead of time-slicing everyone into thrash — the
classic working-set shape.

**This project is a subtraction, not an expansion.** The single behavioral gap
(hog reclaim, §4/B6) is small. The subsystem is rebuilt because it grew three
parallel lifecycles, hand-copied its own rollback across dozens of sites, and
disguises "the feature turned off" as a backend.

## 2. Decisions carried forward (and why they hold)

Inherited from `contention-plan.md`, restated because they bound everything:

- **Preempt/restore, not admission-refusal.** `WorkingSet::reserve` is logical
  (verified: `KvPageTable::reserve` touches `page_len` only —
  `store/kv/page_table.rs:348`); overcommit is intentional; the reclaim ladder
  makes it safe. No process is refused up front.
- **Drop-to-replay is excluded, permanently.** Output tokens are not the write
  history: speculative drafts and pruned beams write KV that never surfaces,
  and recurrent state is a fold over the whole sequence. KV is opaque bytes —
  movable, never reconstructible.
- **When host swap is also full, kill.** Chosen on proportion, not principle:
  host RAM dwarfs device memory, so host-full is rare; kill is honest, bounded,
  and returns device pages and host slots at once. Kill also embodies the
  performance philosophy of §6: never wait unboundedly on cooperation — and
  D7 extends the same rung to a second extreme regime, the global progress
  deadline. Compress-then-spill remains the named upgrade path (it also moves
  opaque bytes, needing no provenance).

## 3. Decisions settled *here* (so the build doesn't relitigate them)

**D1 — One queue, head-first-claim.** Allocation waiters and restore entries
order by one spawn clock. The head (oldest unmet entry, of either kind) has
first claim on every freed page and accumulates until its demand fits; younger
entries are served only from surplus beyond the head's accumulation. This one
rule subsumes today's three (waiters-strictly-before-restores, the
restore-utilization pause, the aging override): drainage is provable because
the head's demand is finite and freed pages flow to it monotonically, and
victim-restore thrash is impossible because a victim is by construction
younger than the waiter it yielded to, so its restore entry queues behind.
Head-of-line cost is bounded (pages accumulate only under contention, only up
to one process's demand) and is the price of the guarantee.

**D2 — The safe point is a state, not a place.** Eligibility to suspend is
"holds no pins" — no fire preparation in flight. A process parked in an idle
await satisfies this *permanently*; the fire prologue is merely where a
running process passes through it. Consequence: idle-honor stops being a third
lifecycle and becomes an observation, and victim eligibility naturally covers
every pin-free process (D4 then bounds the candidate set by age).

**D3 — Self-suspend is unified, not eliminated.** This explicitly reverses
`contention-plan.md`'s "self-suspend leaves the critical path" (which chose
kill so the requester never suspends itself). That decision was made when the
requester waited while holding pins; the A/B/C split removed the pins, so
self-suspend is no longer the hardest distributed state machine — it is a park
at a pure point. A requester that must yield is simply a process at the safe
point; victim and requester become the same code at the same boundary.

**D4 — Hog reclaim without timers: victims younger than the head, honored at
the next safe point.** Two parts.

*Scope — the net-progress rule.* A victim must be **younger than the current
queue head**, the process the reclaim serves. Suspending anything older
cannot make net progress: the elder's restore entry would immediately become
the new, older head, with demand equal to exactly the pages just freed, plus
two copies of pure waste. The head never suspends itself for its own request
for the same reason. This rule subsumes the keystone exemption (the
globally-oldest process is never younger than any head, so it is never a
victim — a theorem now, not a rule) and gives the requester's self-suspend
(D3) its condition for free: a requester yields only while an older head
exists.

*Liveness — one clock, no per-victim bookkeeping.* Widening victim
eligibility was tried before and reverted for a last-host-boundary
notification race (per the GAP doc). An earlier draft of this plan answered
with per-victim grace windows, withdrawal, and re-selection; that machinery
is deleted as unnecessary bookkeeping. What actually makes park requests safe
under D2/B5: idle/blocked holders are pin-free and suspendable *without
cooperation*; running processes pass a safe point at every fire prologue and
at entry to every idle await, so requests are honored at the next safe point;
selection is not one-at-a-time — while one candidate is slow, others can be
tapped; and stale requests are withdrawn by **pressure relief, not timers**
(the `begin_quiesce` commit point re-checks that an older unmet head still
exists). The residue — a guest spinning in pure compute while holding pages,
a hung fire, a victim that dies mid-park — either frees its pages by
terminating or is exactly what the **single system-wide progress deadline**
catches: one clock, reset by any progress toward the head (pages entering its
accumulation; any suspend, restore, or completion), whose breach escalates to
D7. No per-victim timeout exists anywhere.

**D5 — RS residency stays, as a named assumption.** Suspension keeps
recurrent-state slots resident today; this project unifies RS *acquisition*
into the grant (a fire never half-succeeds) but does not swap RS residency.
Assumption on record: the RS pool is sized so that RS-slot contention does not
occur in practice. Revisit only on evidence.

**D6 — Cost-aware victim selection, inside the FCFS invariant.** The original
philosophy reserved exactly two freedoms, and this is one of them: victim
*selection* may be cost-aware as long as service order stays spawn-clock.
This plan wires it: within D4's candidate set (pin-free, younger than the
head), prefer idle/blocked holders over running ones, and prefer the smallest
footprint that covers the demand; tie-break youngest first. The rationale is
performance, not fairness: suspension is whole-process, so an oversized
victim over-evicts — a 40-page victim for a 4-page demand is 36 pages of
useless D2H plus a lane's width lost. Smallest cover minimizes wasted copies
and width loss. Selecting a non-youngest victim within the set is fine;
drainage is unaffected because restore order remains the spawn clock. Lands
with hog reclaim (B6) — same selection code.

**D7 — The escalation ladder ends in kill, and the ladder points down in
age.** Work is preserved in every regime where preservation is possible:
reclaim idle leases, then suspend young victims to host swap — the norm.
Destruction happens in exactly two extreme regimes: **host swap cannot hold
the victim**, or **the progress deadline expires**. Kill is runtime-level —
the same terminate/abort path the host-full kill uses today
(`inferlet::process::terminate`, quiesce-first, GPU lifetime respected,
transactions unwound by the B3 guards) — never an OS `SIGKILL`, never a
bypass of resource cleanup. B3's RAII is precisely what makes
kill-at-any-await structurally clean: kill is just cancellation, and the
cancellation invariant already releases everything unconsumed. The kill
target follows the same age rule as suspension: youngest-necessary first —
prefer an already-selected victim that never reached its safe point (provably
uncooperative), else the youngest whose footprint covers the residual demand;
never the head, never its elders. Fail-loud survives only as the final rung,
when no younger work exists — the lone process growing past the entire pool
fails *itself*. Note this **reverses the inherited endgame**, which failed
the oldest requester's allocation (the starving elder ate the `OutOfPages`);
the old module doc names the alternative it didn't build — "victim-the-hog",
`reclaim.rs:35-36` — and this plan builds it, now that clean abort is
structural.

## 4. What is actually wrong today (with receipts)

Almost none of it is policy. One behavioral hole, plus structure.

**The behavioral gap: hog reclaim.** Victims are restricted to processes
already waiting at the allocation boundary; a young process holding a large
resident footprint while making steady progress is never a victim, so active
preemption redistributes pages among the simultaneously starving. Restriction
to remove (under D4's mechanism). This is also a *throughput* hole: a starving
elder's resident pages are dead memory while it waits, and its lane is stalled.

**The structural rot**, verified in this tree:

- **Three lifecycles where there should be one.**
  (a) victim-at-allocation-boundary — `acquire_or_self_suspend_live_for_pipeline`,
  `store/reclaim.rs:1059-1293`; (b) requester self-suspend — the
  `SelfSuspendFirst` protocol, `reclaim.rs:294-308`, retry loop in
  `pipeline/fire.rs:223-250`; (c) idle-honor — `honor_idle`,
  `inferlet/process/preemption.rs:366-426`. Three `Notify` populations
  (`reclaim.rs:291, 318, 335`), and a lost-wakeup dance copied five times in
  `preemption.rs`.
- **Rollback hand-copied where one owned object belongs.** The grant's Drop
  safety is deliberately disarmed mid-build by `into_pages()`
  (`reclaim.rs:202-205, 227-229`); after that point: 5 manual
  `release_device_reservation` sites (`fire.rs:267, 276, 279, 328, 332`,
  none panic-safe), 11 `abort_rs_transactions` sites (RS transactions have no
  Drop rollback at all — stated at `fire.rs:1606-1607`), 6
  `kv::finalize(.., false)` sites, and 8 DevGeo put-back blocks caused by
  holding `DevGeo` out of the resource table across the contention await
  (`fire.rs:1811-1816` → await at `:1844`).
- **"Passive" is the feature wearing a costume.** `KvPoolBackend::suspend`
  returns a constant `Unsupported` (`reclaim.rs:2149-2153`);
  `SelfSuspendBackend` differs by exactly one constant (`Requested`,
  `:2200-2205`); `restore()` is a no-op in both. The `ReclaimBackend` trait is
  a one-bit policy flag masquerading as a seam — the real physical work lives
  in `preemption.rs`.
- **Cached derived state, hand-balanced.** Three mirrored `AtomicUsize`
  counters (`reclaim.rs:420-422`) with 12 conditional mutation sites, guarded
  by `debug_assert` (`remove_count`, `:492-498`).
- **Notification is not trusted.** Both blocking waits are
  `select!{notified(), sleep(20–50ms)}` (`:1286-1290`, `:956-960`);
  correctness rests on the poll, whose interval is derived from the unrelated
  exhaustion deadline. Under heavy contention these poll steps are transition
  latency, paid repeatedly.
- **Duplicated judgment.** Three different "oldest" predicates
  (`is_fcfs_oldest :699`, `is_oldest_requester :1465`, inline `min_by_key`
  `:1140/:1555`); two exhaustion clocks (`:1272-1283`, `:1554-1590`);
  `ProcState` transitions spread over 12 sites with inconsistent guards
  (`report_restore_failed` sets `Suspended` unguarded, `:995-998`).
- **Dead code obscuring the design.** ~390 lines of `#[cfg(any())]` legacy
  that no longer compiles (`reclaim.rs:1695-2084`, 15% of the file); the
  documented deadlock-breaker gate (`:1249-1268`) has never fired in
  production (`fire.rs:238` passes `holds_reclaimable: false`); `acquire()`
  ends in `unreachable!("self suspend disabled")` (`:1014`); assorted
  zero-caller accessors.
- **Config smuggled through env, and a global singleton.** Five knobs
  (`PIE_KV_CONTENTION`, `PIE_KV_PREEMPT_ACTIVE`, `PIE_KV_RESTORE_AGING_MS` —
  read *inside* `drain()` at `:1549` —, `PIE_KV_EXHAUSTION_MS`,
  `PIE_KV_RESTORE_RETRIES`), each behind its own read-once; `static
  CONTENTION: OnceLock` (`:2252`); single-driver baked in (backends hardwired
  to driver 0; `preemption.rs` hardcodes `(0, 0)`; bootstrap validates
  exactly-one-driver at `bootstrap.rs:524-545`).
- **Process-side sprawl.** 32 hand-placed preemption prologues (26 `honor` +
  6 `contention_gate`) across 6 files with no structural guarantee a new WIT
  method gets one; `suspend_restore` is one 160-line function ending in
  `unreachable!()` (`preemption.rs:563-722`); 9 `decline_park` bail-out sites;
  zero unit tests in the file.

**The assets the build stands on** (equally verified):

- The call seam is genuinely narrow: the fire path reaches the orchestrator
  through one function (`acquire_kv_pages`, `fire.rs:223-250`) at two call
  sites; installation is one `OnceLock` set at `bootstrap.rs:354`. A
  replacement core can be built beside and swapped without touching `fire.rs`.
- The store layer is already grant-native: the demand/reserved split exists
  end-to-end, consumes exact prefixes, and a short grant is a loud
  `GrantMismatch` (`store/kv.rs:480-483, 689-694`) — never a silent pool
  fallback.
- The FCFS clock (`submit_seq`) never leaves the module (minted in
  `register()`, `reclaim.rs:708-722`) — it can be replaced freely.
- `tests/contention.rs` runs unignored in the default suite, hard-gates
  engagement (`suspends > 0`, `d2h == h2d`) *and* asserts 100% pool/host-slot
  restoration — a built-in page-leak detector. 14 mock-backed unit tests pin
  the policy core. `tests/contention_host_full.rs` pins the kill policy.

**Where the net has holes** (this dictates Stage S's coverage work): no
failure-injection coverage for the reserved wrappers (`fire.rs:252-335`) or
any of the ~18 post-acquire early returns; the device-geometry contention path
is host-untested (covered only by `#[ignore]`d 4090 suites); `preemption.rs`
has zero unit tests; the quantitative A/B
(`bin/pie/tests/cuda_contention.rs:63-77`) is 4090-only and manual.

## 5. The shape we move toward

One sentence: **one owned grant, one FCFS queue, one safe point, and a state
machine the type system enforces.**

- **One owned grant (RAII).** Acquired once (KV pages and RS slots together),
  lends pages by exact prefix during the build, committed once or dropped
  once; its disposal *is* the rollback. Pages return to the pool only through
  a destructor or a commit — never through a hand-written error path.
- **One safe point** (D2/D3). Every fire begins at it, holding nothing,
  because demand computation is pure; idle waits sit at it permanently.
  Victim, requester, and idle paths are the same code.
- **One FCFS queue, one typed state machine** (D1). Head-first-claim;
  transitions consume the prior state so illegal transitions are
  unrepresentable; derived counts are computed at one chokepoint, not
  mirrored at twelve.
- **Intelligence lives above.** The contention layer never learns that
  admission exists.

## 6. Performance under heavy contention: the cost model

Measured today (GAP doc, RTX 4090, Qwen3-0.6B): the mechanism's uncontended
tax is ~2% (preempt-roomy vs legacy-roomy: 0.982x throughput, 1.018x p95).
Under severe pressure the picture is different: an 8-page cap runs at 0.429x
of roomy, a 12-page cap at 0.525x — and the measured attribution is that
**narrowed runnable batches and waiting dominate**; copy time is visible but
secondary. A suspended pipeline is outside the runnable quorum until its grant
and restore complete, and every such absence narrows the wave the GPU runs.

**The ceiling framing.** Under overcommit, the number of lanes that physically
fit sets a hard throughput ceiling; part of 0.429x is the regime's physics,
not a defect. This layer's performance obligation is to **hug that ceiling**:

- **Zero wasted copies** — no thrash (D1 makes it structurally impossible),
  no over-eviction (D6 smallest-cover selection).
- **Near-zero dead time between transitions** — no polling steps (the 20–50ms
  backstops die with the new core), restore turnaround (pages freed →
  pipeline rejoined) tracked as a first-class diagnostic and minimized,
  head-restore prefetch (reserve the queue head's grant early; overlap its
  H2D with ongoing compute) so copy latency hides behind work.
- **Bounded idle pages** — only the head's accumulation (D1), never more than
  one process's demand, only under contention.
- **No unbounded waits on cooperation** — safe points are dense (every fire
  prologue, every idle-await entry) and the single progress deadline (D4/D7)
  bounds every wedge. Kill is an extreme-rung tool, not a latency tool: its
  real cost (lost work re-run) is itself contended-throughput damage, so the
  deadline is sized for "the fleet is wedged," not "this victim is slow."

**The metric is gap-to-ceiling**, not raw ratios: ceiling ≈ fitting width ×
per-lane throughput at that width (measured from capped-but-uncontended
runs); improvements are reported as gap closure. Raw floors still hold as
regression gates (§10).

**The boundary.** Recovering *width* under overcommit — running fewer, fatter
cohorts so contention rarely fires — is admission's job (M5), not this
layer's. This layer's obligation ends at near-zero dead time between
physics-limited transitions. It gets leaner, never smarter.

## 7. The route: build-first

Per operator directive, the route optimizes for implementation speed and bold
refactoring. Structure lands upfront; verification trails. Concretely:

- **The build order still follows real dependencies** — ownership before the
  state machine (so "half-prepared fire" states never need to exist), core
  before process side (so `preemption.rs` adapts once, to the final
  protocol), the behavior change last.
- **In-flight discipline is deliberately light**: every commit compiles;
  unit tests are written only where they are the cheapest way to think (the
  new core's queue and state machine — the 14 ported behaviors are the
  design's spec); fleets may be run opportunistically but are not gates
  mid-build. Fleet-red states are acceptable between build steps.
- **What this trades**: per-step ×3 fleet gates and fine bisect granularity,
  in exchange for speed and the freedom to cut across the old structure
  instead of tiptoeing through it. The mitigation is commit coherence (each
  build step is one coherent, compiling commit) and the end-state arbiters:
  the fleet tests' page-leak and engagement asserts, and §8's invariants.

### Stage B — Build (the whole structural rewrite, upfront)

**B1 — Burn the dead code.** Delete: the `#[cfg(any())]` legacy block
(`reclaim.rs:1695-2084`); the test-only `acquire()`/`acquire_or_self_suspend()`
pair (`:1007-1026`); `ReclaimableProbe` (`:224`) +
`acquire_or_self_suspend_live` (`:1044-1057`) + the never-fired step-6 gate
(`:1249-1268`); `allocation_requires_grant`, `is_running`, `is_suspended`;
`notify_pipeline_join` and its 4 call sites; `LeaveKind::Terminate`;
`DevicePageReservation::len`. Pure red diff.

**B2 — One config, read once.** `ContentionConfig { mode: Off | Preempt,
restore_aging, exhaustion, restore_retries }`, parsed from env at the
bootstrap edge only; the five per-knob read-once statics die;
`PIE_KV_PREEMPT_ACTIVE` dies as a concept. Tests inject config explicitly.
Operator-facing env names keep working; only where they are read changes.

**B3 — One owned grant.** Kill `into_pages()`; the grant guard stays armed
through the build (prefix lending, explicit `commit()`, Drop returns
everything unconsumed). Abort-on-drop guards for RS and KV transactions.
DevGeo no longer held out of the resource table across the await; scope-claim
order symmetrized. One build path per fire shape: a pool-direct grant
provider for mode=Off returns the same grant type (running rung 0
internally), deleting the non-grant fallbacks and their two-attempt
structure. Stale demand → bounded recompute-and-reacquire instead of
`GrantMismatch` death. `rs_demand` as a pure phase-A function.

**B4 — The new core.** Keep the module name (`store::reclaim` is the
contract); rebuild internals (`reclaim/{queue, state, grant}.rs`): one queue
under D1; typed consuming state machine; computed counts (one packed summary
word, one writer site — the lock-free hot path survives); one Notify story,
register → re-check → await, both fallback polls deleted; one oldest
predicate; one exhaustion clock; the backend trait dies (pool operations
become a minimal pool port — one production impl, one test impl); acquire
takes `{kv_pages, rs_slots}`; singleton → registry-owned per (model, driver).
Plus the performance plumbing from §6: restore-turnaround timestamps
(freed → rejoined) and head-grant pre-reservation as the prefetch hook.
Swap at `bootstrap.rs:338-358`; delete the old core in the same series.

**B5 — One safe point.** One `yield_point(ctx)` replaces `honor` +
`honor_idle` + `contention_gate` and the five lost-wakeup copies;
`SelfSuspendFirst` dies (enum, `fire.rs:223-250` retry loop, `:1111-1113`
exit); the 32 prologues route through a dispatch trunk where one exists,
one-liners where not; `suspend_restore` splits into suspend/restore halves;
`decline_park` bail-outs become a guard; hardcoded `(0, 0)` becomes
parameters.

**B6 — Hog reclaim + victim selection + the endgame.** Eligibility widens to
any pin-free process younger than the head (D4's net-progress rule; the
keystone exemption becomes a theorem); the `begin_quiesce` commit point gains
the age/pressure re-check that withdraws stale park requests; D6's
smallest-cover / prefer-idle heuristic lands in the same selection code; the
two exhaustion clocks collapse into the single progress deadline, whose
breach drives D7's kill through the existing host-full terminate path, with
fail-loud demoted to the final no-younger-work rung.

### Stage P — Polish

The elegance pass, while design intent is fresh: a simplify/altitude sweep
over everything Stage B touched; naming; module-level doc comments stating
each contract (the new queue rule, the state machine, the guard protocol);
leftover knobs folded into config or deleted; diagnostics derived from
computed state.

### Stage S — Stabilize

Now the burn-down: run the fleets (`tests/contention.rs`,
`tests/contention_host_full.rs`) and fix until green ×3 — their page-leak and
engagement asserts are the arbiters. Then write the coverage the surgery was
always going to need, and make it green ×3: failure-injection tests for the
guard contracts (grant fully returned on RS failure / KV-install failure,
surplus returned exactly once); a host-runnable device-geometry contention
fleet; the hog fleet (young, steadily-progressing hog with a large footprint
+ a starving elder → the elder completes); a progress-deadline fleet (an
uncooperative pure-compute holder + a starving elder → the holder is killed
through the abort path, the elder completes, every page returns). Invariants
become assertions where cheap. `preemption.rs` gets its first protocol-level
unit tests here.

### Stage V — Prove performance

All measurement happens here, against the pre-rewrite ref (this baseline
commit, via git) with the same harness:

- Contended profiles first — they are the goal: 8-page and 12-page caps,
  ceiling model computed, gap-to-ceiling reported. Floors: ≥ 0.429x and
  ≥ 0.525x (no regression), with improvement expected from poll removal,
  thrash elimination, and D6.
- Roomy A/B: ≥ the standing 0.982x / ≤ 1.018x p95. Ordinary c0/256 ≥ 99% of
  its baseline.
- Restore-turnaround distribution from the new diagnostic; traces show no
  20–50ms poll steps.
- Tune where the numbers say to: head-restore prefetch / H2D-compute overlap
  is the named first lever.

## 8. Invariants that must survive

An implementation that violates one of these has not solved the problem,
whatever the tests say:

- A physical page is in exactly one of: free pool, reservation, live mapping,
  suspend/restore transfer, or finalized fire transaction.
- A reservation is installed once or returned once — and it returns through a
  destructor or a commit, never through a hand-written error path.
- No await while holding a store, orchestrator, resource-table, or channel
  lock.
- No process is suspended while holding an unabortable preparation.
- FCFS ordering is authoritative for both allocation and restore, under
  head-first-claim: younger entries consume only surplus beyond the head's
  accumulation.
- Only work younger than the head is ever sacrificed for it — by suspension
  or by kill. The head and its elders are never disturbed for a younger
  process.
- Work is preserved wherever preservation is possible: destruction requires
  host-swap exhaustion or a progress-deadline breach, and outright request
  failure requires that no younger work remains.
- Cancellation at any await point removes the waiter and releases anything
  unconsumed.
- Exhaustion fails loud only after no victim, no kill, and no self-transition
  can make progress.
- Already-submitted fires keep their close/drain semantics.

## 9. What we are explicitly not doing

- **Not** admission control in this layer — that is the layer above, by
  design.
- **Not** drop-to-replay, now or ever.
- **Not** compress-then-spill in this increment — the named upgrade path, not
  this work.
- **Not** a smarter contention policy. FCFS, deliberately dumb; D6 tunes
  *cost*, never *order of service*.
- **Not** kill as a latency tool — destruction only at the two extreme rungs
  (host swap short, progress deadline breached), always through the runtime
  abort path, never `SIGKILL`.
- **Not** a rewrite of the physical page/copy primitives — a lifecycle and
  ownership change, not a storage change.
- **Not** RS residency swap (D5) and **not** multi-driver support — only the
  removal of single-driver from the structure.

## 10. Acceptance gates

Applied at Stage S (correctness) and Stage V (performance) — not per build
step:

- Correctness: both fleets ×3 consecutive green, including the 100%
  pool/host-slot restoration asserts (page-leak detector) and the engagement
  hard-gates (anti-vacuous guard); the new failure-injection,
  device-geometry, and hog fleets, same cadence.
- Performance (contended first — it is the goal): 8-page ≥ 0.429x, 12-page
  ≥ 0.525x vs the pre-rewrite ref, gap-to-ceiling reported and expected to
  close; roomy ≥ 0.982x / p95 ≤ 1.018x; c0/256 ≥ 99%; no poll steps in
  contended traces.

---

The mountain is climbed alpine-style this time: one committed push to the
summit ridge — build and polish in a single ascent — with the fixed ropes
(the fleet arbiters, the invariants) checked on the descent. Everything still
bends toward making the bottom layer do less, prove more, and waste nothing
while it does it.
