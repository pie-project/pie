# Project Rainer: Frame-Boundary Residency Planning

Part I below is the v1 design as landed (+ the 2026-07-25 revisions);
**Part II (§11–§17) is Rainer v2 — the single-writer boundary pass**,
the redesign the phase-2 measurements point at.

Date: 2026-07-25 (perf standing updated 2026-07-26)
Status: **v1 + E5/E6 IMPLEMENTED, contended 0.96x vLLM** — after the
CONTENTION_FOLLOWUP.md §17 mechanism fixes (lease-quiescent victim
preference; batched swap copies; standalone-copy completion decoupled from
the compute stream): h2h 13.39–13.42K = 0.963x, press ~11.0–11.5K ≈ 0.79x
(restore-rotation economics remain, §17.1), roomy 17.24K parity. The
2026-07-25 baseline below stands as written —
`runtime/engine/src/planner.rs` + `planner/{grant,exec}.rs`; the reclaim
ladder, safe points, park protocol, and per-fire grant gate are deleted.
Final bench verdict (§15): roomy 17.16K (parity), h2h contended
12.87–13.01K ×6 = 0.93x vLLM (13.94K, from 0.42x), pressure 11.7–11.9K
with 256/256 ×2. The 0.42x gap was NOT page-supply latency but the E5
omission: v1 ran `plan()` inline on every guest task, and the idle-reclaim
scans + eviction quoting under the global KV lock serialized the fleet's
turnaround (55.7% GPU idle → 5.9% after the single-owner drain +
idle-reclaim latch + parking_lot on the two storm locks + the
lane-resurrection leave fix). See `CONTENTION_FOLLOWUP.md` §14–§15 for the
record; §2's K-frame lookahead paragraph was revised 2026-07-25
(declaration-based, no extrapolation) and its batched-grant component
remains the candidate for the residual 7%.
Supersedes: the reactive reclaim ladder (Kilimanjaro B5/B6) and the §6
E-mechanism proposal in `CONTENTION_FOLLOWUP.md`. Evidence for every claim
below lives in that file (§§1–5, 12, 13).

---

## 1. Thesis

Today's contention management preempts a *running* process with its
cooperation: park request → safe point → quiesce → copy → free, a negotiation
hand-wired across five subsystems. Every defect found in the 2026-07-25
investigation lived at those seams (zero-page victim livelock, seal key-space
wedge, frame⇄copy⇄resize deadlock, intermittent thrash), and the negotiation's
front door — every fire routed through one global grant gate — cost **−17%
roomy throughput** by collapsing the Venus run-ahead overlap (bisected to B2,
`CONTENTION_FOLLOWUP.md` §13).

Rainer replaces the negotiation with a decision:

> **At each frame boundary, a planner computes the next frame's resident set —
> the FCFS-by-spawn prefix whose cumulative working sets fit total capacity.
> If you are not in the set, you are not in the frame.**

This is only possible because Venus already made execution frame-native and
lockstep: a global point where *every* process is quiescent arrives for free
every ~7–16 ms. The current design fights that synchrony with asynchronous
preemption; Rainer exploits it.

## 2. The planner

One pure function per boundary, over two ledgers:

```
inputs:   FCFS registry (spawn order — the only priority key)
          residency ledger (per-process: pages held, lane positions)
          committed pool geometry (total pages, free pages)
          driver-advertised growable headroom

resident set = longest FCFS prefix with  Σ(held + delta) ≤ total capacity
```

- **Demand is declared, never predicted** (revised 2026-07-25; the original
  text claimed the delta "K frames ahead" was exact arithmetic on
  `position % 16` — that smuggles in the assumption that lanes keep
  decoding, which is a *prediction*: an inferlet is an arbitrary program
  and may prefill 10K tokens, fork, idle, or exit next frame). What is
  true unconditionally: whatever an inferlet does must be **declared
  before it executes** — every fire carries its exact page demand at
  submission, and run-ahead means declaration precedes execution by at
  least a frame (roomy's gather overlap is the measured proof of that
  distance). The planner acts on the declared pipeline only — the queue
  of submitted-but-unexecuted fires — never on extrapolated positions.
  For a decode fleet this recovers the same numbers the arithmetic gave
  (steady state at 128 lanes: ~8 pages/frame), as facts instead of
  projections.
- **Held pages are assets, not demand.** `free == 0` does not evict anyone
  whose lanes cross no page boundary this frame. The prefix is computed
  against *total* capacity, never against the free count.
- Fires never block on allocation mid-flight: the frame's pages are granted
  in one batched operation at plan time. There is **no per-fire gate** — not
  even an atomic check. The B2 toll is deleted structurally, not bypassed.

## 3. Capacity waterfall (elastic pool interaction)

Deficit is funded in cost order:

```
free pages  →  pool grow (VMM map)  →  evict youngest
zero cost      no copies, cheap        copies; last resort
```

- **Grow is rung zero of reclaim.** Eviction handles only the deficit that
  survives headroom exhaustion.
- **Plan only against committed geometry.** A grow is an async control; when
  it retires, the ledger's total rises and the *next* boundary plans richer.
  Never plan against memory not yet held — the same discipline as grant RAII.
- Headroom is a number the driver advertises; competition for device memory
  stays the driver/allocator's concern. The planner requests
  `min(residual deficit, headroom)`.
- **Shrink is asymmetric**: considered only at completion events, never paced
  per frame (the resize-per-reclaim-step mistake cost 45x once already).

## 4. Eviction and restore

**Eviction** — sized and aimed, never negotiated:

- Exactly `⌈residual deficit / footprint⌉` processes, from the youngest edge
  of the prefix. No park requests, no consent protocol, no victim scoring:
  an evictee is between frames and therefore already quiescent.
- D2H copies go to the transfer channel at the boundary and overlap the next
  frame's execution. Freed pages land at copy retirement, so evictions fund
  *future* frames; funding starts **at declaration time** — the moment the
  unmet demand is submitted — so the copy latency hides behind execution
  already in flight. "Start as early as knowledge exists" leaves no timing
  constant to tune. (Facts again — not a heuristic.)

**Restore** — gated by the accumulation invariant (kept from Kilimanjaro):

- The oldest evictee is the *restore head*. Freed pages accumulate to it;
  younger residents may consume only the surplus beyond its accumulation.
- It boards again when accumulation covers its swapped set (revised
  2026-07-25: the original added "plus the fleet delta over its transfer
  horizon" — a prediction of fleet demand, dead for the same reason as
  K-frame extrapolation). No horizon term is needed: demand arriving
  while its H2D is in flight is younger by FCFS and waits its turn.
  Restores therefore arrive one at a time, in FCFS order, funded by
  completion-sized releases.

**Thundering herd is unrepresentable**: evictions are deficit-sized from one
edge (over-freeing cannot happen), restores are accumulation-gated (mass
re-entry cannot happen), and membership is monotone between completions
(footprints only grow until a process finishes). The reactive design produced
a measured 900-cycle suspend/restore herd in 1 of 5 runs; the planner has no
state in which that behavior can be expressed.

## 5. What dies, what remains

| dies | why it is unnecessary |
|---|---|
| safe-point machinery (B5) | the boundary is everyone's safe point |
| park request / decline / ParkGuard / escalation ladder | membership is a pure function; nothing is negotiated |
| victim selection + `ReclaimQuote` scans | the planner's ledger already knows who holds what |
| 5-state `ProcState` machine | in-set / out-of-set (+ transfer in flight) |
| per-fire grant gate (`serialize_under_contention` on the hot path) | pages are granted per frame at plan time |
| copy⇄frame⇄resize queue-order rules | transfers own their channel; membership changes only at boundaries |
| mid-frame leave/join seams (the seal-bug habitat) | membership is planner-owned and boundary-only |

| remains | role |
|---|---|
| FCFS registry | the one priority key (spawn order) |
| residency ledger | event-updated pages-held / lane positions |
| boundary planner | one pure function |
| swap engine | batched D2H/H2D on a dedicated channel |
| hog endgame | a process whose own working set exceeds total capacity: host-stream or fail loud |
| grant RAII / page-ownership invariants (B1–B4 substrate) | unchanged foundation |

## 6. Guarantees

- **Head-first becomes a theorem.** The FCFS prefix contains the oldest
  process by definition → the head always runs → completes → the next-oldest
  becomes head. Drainage no longer depends on a ladder behaving.
- **Deadlock excluded**: no process waits for pages mid-frame, so the
  circular-wait ingredient does not exist.
- **Thrash excluded**: §4 above.
- **No knobs.** Every quantity in the design is a ledger value, a measured
  latency, or exact arithmetic.
- **Layering preserved.** This is not admission control: M5/Vesuvius decides
  who enters the system; Rainer decides which already-admitted processes are
  resident *this frame* — the same authority today's suspend/restore ladder
  exercises, moved to the boundary. FCFS-prefix keeps the policy deliberately
  dumb (no metric optimization).

## 7. Edge cases

- **Prefill bursts**: newcomers are youngest; they wait out-of-set (holding
  nothing) until the prefix reaches them. They cannot displace elders.
- **Idle processes / long host awaits**: idle is trivially quiescent —
  evictable at any boundary. Simpler than today's park-signal races.
- **Undersized pool** (`total < head's working set`): hog endgame, not a
  planning case. Fail loud after grow is exhausted.
- **Frame-external work** (riders, control fires): unchanged; they carry no
  planned KV growth.
- **Multi-model/driver**: one planner instance per (model, driver) pool, as
  the ledgers already are.

## 8. Expected performance

Reference points (RTX 4090, Qwen3-0.6B, measured 2026-07-25):

- Roomy: pre-rewrite pie was vLLM parity (17,240 vs 17,187 tok/s @conc128;
  19,397 vs 19,376 @conc256). Rainer restores this structurally (no per-fire
  gate at all) — better than the interim futex-fast-path patch.
- Contended (2.1x oversubscribed, 256 req): the fixed reactive engine reaches
  8,000–9,090 tok/s = 0.80–0.91x of an ideal-admission ceiling that *itself*
  pays the B2 toll. Rainer's ceiling is the true one, and boundary-planned
  eviction removes the 1-in-5 thrash mode entirely.

## 9. Open questions

1. Run-ahead (depth 2) across a membership change: frame N+1's set is planned
   while frame N executes — the plan must key off frame N's *post* state, not
   its retirement. Believed fine (delta arithmetic is position-based), needs
   a precise statement.
2. Restore H2D interleaving with the boarding frame: join at the first
   boundary after copy retirement (simple), or overlap-join mid-gather
   (faster, more seams). Start simple.
3. Grow quantum sizing under fragmentation of the VMM reserve — driver-side
   question; planner just consumes the advertised headroom.
4. Stage V's recorded roomy A/B floor (≥0.982x) contradicted the measured
   −17%; post-mortem the old measurement before trusting any A/B harness for
   Rainer acceptance.

## 10. Migration sketch

1. Ledgers first: residency ledger + committed-geometry ledger, event-updated,
   read-only shadow mode (planner computes and *logs* the set; ladder still
   rules). Divergence between plan and reality is the acceptance metric.
2. Cut fires over to plan-time batched grants (deletes the per-fire gate;
   roomy parity should return here — gate on the roomy A/B).
3. Flip eviction/restore to the planner; delete the ladder, safe points, and
   the ProcState machine.
4. Delete the interim scheduler accommodations that exist only for mid-frame
   membership churn (out-of-band copy sweep stays — transfers keep their own
   channel by design).

Existing regression tests and the 2026-07-25 bench harness
(`CONTENTION_FOLLOWUP.md` §9, scratchpad runners) carry over as the
acceptance suite: roomy parity, 256-req pressure 5/5 completion, no thrash
signature (suspend/restore cycles bounded by arithmetic, not luck).

---

# Part II — Rainer v2: The Boundary Pass

Date: 2026-07-25 (after the phase-2 measurement round)
Status: **DEFERRED as a performance project** — the precondition probes
(`CONTENTION_FOLLOWUP.md` §16, same day) measured v2's two headline
purchases away: worker ingest has no >2 ms passes even at 128-fire
bursts; KV-lock hold utilization is 4.4% (batching has little to
amortize); and the §15.1 regime cliff is CLOSED post-parking_lot
(forced-coherent depth-1 runs −5.8% of depth-2, identical fat-wave
shape). Lazy chunked quoting (the one targeted fix the probes indicated)
landed with no measurable throughput change — the residual ~7% to vLLM
is resident-set size (capacity physics) plus small per-row overheads,
not a lock or scheduling story. v2's remaining value is the *seam-class*
simplification of §14 (membership pushed, not inferred) — a robustness
refactor to take incrementally, starting with the choke-point leave
cleanup, not a rewrite justified by throughput. v1 + E5/E6 (0.93x
contended, roomy parity, 9/9 clean) is the baseline of record.

## 11. What the measurements taught

Every number is from the instrumented 2026-07-25 rounds (§15):

- **L1 — The bottleneck was never the GPU, the pages, or the planner.**
  Per-row GPU cost is identical in fast and slow regimes (~89 µs/row);
  the evict chain is p50 5 ms; park→serve p50 0 ms. The bottleneck was
  guest turnaround touching shared state: inline `plan()` (idle-reclaim
  scans + eviction quoting under the global KV lock), 4–6 KV-lock hits
  per fire build, a planner-lock hit per gated WIT call. Fleet turnaround
  serialized at ~0.4 ms/guest → 55.7% GPU idle behind a wait-all seal.
- **L2 — Every wedge lived in distributed state inference.** The lane-
  resurrection race (a pre-fence fire arriving after the evictor's leave
  resurrects an awaited lane; §15.2) is the archetype: frame membership
  was *inferred* from racing leave/arrival events, enforceable only by a
  convention spread across five park sites. The same shape produced the
  seal key-space bug and the elder/cascade/starvation patch pile.
- **L3 — Fleet phase is a first-class variable.** Two stable regimes,
  35% apart (coherent herd 8.4K vs de-cohered cohorts 12.9K), selected
  by accidents of lock timing (§15.1 A/B). Nothing in the design chooses
  the regime; performance currently rests on a stable accident.
- **L4 — What survived contact:** FCFS-prefix membership as a pure
  function, declaration-based demand (never extrapolation — §2 revision),
  the boundary as a free quiescent point, no-knobs/no-timers liveness,
  the swap engine, E6 as an event rule.

## 12. Thesis

v1 applied "membership, not negotiation" to *residency* but left its
execution distributed: guests acquired per fire, parked in a waiter
queue, and membership was inferred from leave/arrival events. v2 applies
E5 to everything:

> **One boundary pass per frame is the only writer of shared truth —
> membership, page allocation, eviction, restore. Everything else is a
> lock-free declaration queue in and a result queue out.**

## 13. Architecture

**Guest side (zero shared state):**

- A fire submission is a *declaration*: exact `Demand` attached, queued.
  No acquire, no park, no planner or KV-lock contact. Turnaround is pure
  WASM + submit (~50 µs) — nothing for a wake-herd to collide on.
- Results arrive on the guest's own channel as today. The residency gate
  stays one atomic (evicted processes must not touch pooled state).

**The boundary pass (once per frame, one task, the single writer):**

1. **Collect** — the retired frame's releases (finalized pages,
   completions) and the declarations queued since the last boundary.
2. **Membership** — resident set = longest FCFS prefix whose
   Σ(held + declared) fits total capacity. Explicit, published, owned.
3. **Fund** — allocate every member fire's pages in ONE batched KV-lock
   acquisition and attach grants to the queued fires. Post evictions for
   the residual deficit: victims are out-of-prefix, therefore absent
   from the next frame, therefore structurally quiescent at the current
   frame's retirement — D2H overlaps the next frame's execution. Board
   the restore head when the accumulation ledger covers it.
4. **Publish** — push the member set to the frame policy: awaited ⟺
   member. Lane membership is delivered state, not inferred state.

An unfunded (out-of-set) fire simply stays queued; its lane is not a
member, the seal does not wait for it, and no park/wake machinery exists.

## 14. What dies, what remains (relative to shipped v1)

| dies | why it is unnecessary |
|---|---|
| per-fire `acquire`, waiter queue, park/serve/collect, `WaitRegistration` | an unfunded fire waits in its own queue; "parking" is not a concept |
| elder bypass, serve cascade, accumulation reservation dance | derivatives of implicit membership; accumulation becomes a ledger number |
| the leave contract + its five park-site re-posts (§15.2 fix) | membership is pushed by the boundary pass; the resurrection race is unrepresentable |
| lease/fence Dekker negotiation, Yield-at-acquire, settle-tail paths | eviction never touches a member of the executing or next frame; frame structure IS the quiescence. The fire lease demotes to a debug assertion |
| the starvation predicate's head-selection gymnastics | the boundary pass holds a global per-frame view; Impossible/Hog/Starved become simple assertions in one place |
| the E5 drain task as a separate entity | it *becomes* the boundary pass |

| remains | role |
|---|---|
| FCFS registry | the single priority key |
| swap engine (D2H/H2D channel) | unchanged |
| E6 | one line of membership hysteresis: a restored member stays a member until one fire completes |
| hog endgame, residency gate (one atomic) | unchanged |
| declaration-based demand (§2 as revised) | the only information source |

## 15. Guarantees and performance model

- Liveness: head-first is still a theorem; every endgame predicate is a
  boundary-pass assertion over a consistent global snapshot. No timers,
  no knobs, no heuristics in any liveness path (the §6 criterion).
- **Regime bistability is eliminated, not won:** with turnaround at
  ~50 µs, both fleet phases are gapless, so the system settles into
  coherent full-width waves — 128 rows at ~58 µs/row versus today's
  de-cohered 48 rows at ~89 µs/row. At the measured 5.9% idle this
  prices above vLLM (13.9K) rather than 7% below it, as a structural
  property instead of a stable accident (L3).
- Roomy is untouched: with nobody out-of-set the boundary pass is
  bookkeeping plus one batched no-op.
- End state (optional): the KV store itself becomes boundary-owned —
  the global KV lock disappears entirely.

## 16. Migration from the shipped tree

The 0.93x tree is already half the adapter. Four steps, each
independently benchmarked (roomy A/B + h2h ×6 + press ×2 gates):

1. Ledger + boundary-batched grants: member fires stop calling
   `acquire`; the drain task starts running at boundaries.
2. Membership push to the frame policy (replaces leave/arrival
   inference; deletes the five re-post sites).
3. Delete the waiter queue and park machinery; the drain task is
   renamed to what it now is — the boundary pass.
4. Demote lease/fence to assertions.

## 17. Open questions

1. Boundary-pass latency budget: bookkeeping + one batched allocation
   must stay well under a frame (~7 ms); expected µs-scale, must be
   measured with the §15 same-clock markers from day one.
2. The membership-push protocol is the one new seam (L2 says seams are
   where bugs live): keep it a whole-set publication per boundary, never
   a delta stream.
3. k > 1 frames, multi-pipeline processes, riders/control fires: enumerate
   against the membership-push rule before implementation.
4. The one-off driver `fixed-decode compose` fail-stop (§15.2) predates
   the wedge fix and is unexplained; re-check under v2's simpler
   membership before attributing it anywhere.
