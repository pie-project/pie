# KV contention — root-cause diagnosis and proposed redesign

Written 2026-07-25, on `dev` @ `713f76b17` (RTX 4090, Qwen3-0.6B, hard KV cap
via the driver option `total_pages`). Raw records and runner scripts:
`~/Workspace/pie-wt/vllm-compare-records/` and the scratchpad
`matrix2/`, `traced/`, `traced2/` directories.

§§1–5 are **measurement**: what breaks, and why, each claim tied to a captured
run or a unit test. §6 is a **proposal** — not implemented, not signed off.
§§7–11 are operational notes.

**This supersedes the first draft of this file.** That draft blamed
"continuous admission of young work" and framed the problem as a layering /
admission-control gap. Both claims are refuted below by direct measurement.

---

## 1. The finding

Under sustained KV pressure the engine does not degrade and does not thrash —
it **livelocks**. The GPU goes to **0% utilization and stays there**, while the
reclaim ladder spins at roughly **5,000 futile victim escalations per second**,
every one of which frees zero pages.

The mechanism is a single closed loop:

1. The queue head cannot get its pages (in the captured run it needed **4**
   pages out of a 2,151-page pool).
2. `escalate` posts a park request to a victim younger than the head.
3. The victim reaches its safe point, freezes its pipeline, and calls
   `KvStore::prepare_suspend`.
4. `prepare_suspend` → `PageTable::private_resident_pages` returns an **empty**
   page set and reports `SuspendDisposition::NothingReclaimable` — because the
   victim selected in step 2 was **holding no KV pages at all** (§3).
5. `suspend_at_safe_point` returns `Ok(false)`; the `ParkGuard` declines the
   park, resumes the pipeline, and the process goes back to `Running`.
6. Nothing was freed, the head is still short, so go to 2.

Neither of the two counters that would have made this visible existed:
`NothingReclaimable` was an unreported early return, and `escalate` had no
counter for "posted" vs "found nobody". Both were added for this investigation
(§8).

## 1b. Status update (2026-07-25) — three stacked root causes, two fixed

Fixing the livelock above did not recover the workload; it exposed a second
wedge, and fixing that exposed a third. Each was confirmed by counters/dumps,
not inference. All on the 256-request pressure config of §9:

| state                          | completed | tok/s | evidence of the next wall |
|--------------------------------|-----------|-------|---------------------------|
| baseline                       | 34/256    | 193   | `nothing_reclaimable=204,859`, GPU 0% |
| + fix 1: victim selection (§3, E3) | 38/256 | 216   | `nothing_reclaimable=0`, but `sealed=0`, seal watchdog expired |
| + fix 2: seal key-space bug    | 89/256    | 506   | `sealed=1` **and** `in_flight_launches (0)` — sealed frame never posts |
| + fix 3: queue-order deadlock  | 256/256 (4 of 5 runs) | 8,000–9,090 | 0.80–0.91x of the ideal-admission ceiling (M1 conc-61: 9,949); 1 of 5 runs falls into an intermittent THRASH regime instead (§12.5) |

- **Fix 1 (implemented)** — `select_costed` reclassified: covering holders
  outrank partial holders outrank no-opinion outrank provably-useless;
  `ReclaimQuote::Nothing` victims are never posted (§3, §6.3 E3).
- **Fix 2 (implemented)** — `on_lane_leave` cleaned the process-keyed maps
  (`staged`, `joins_in_flight`, `pending_binds`) with a **pipeline-scope key**,
  so a process that parked during KV acquire *before its first fire* stayed in
  `joins_in_flight` forever and held the seal gate (`joining == true`). The
  leave notification now carries `(lane, owner)` through
  `reclaim.rs → worker.rs → frame.rs`.
- **Fix 3 (implemented)** — a three-party queue-order deadlock: sealed frame
  ⇄ standalone-copy barrier ⇄ pool resize. Barrier removed, standalone copies
  dispatch out-of-band, settling copies no longer hold frame posting.
  Mechanism in §12, fix in §12.4, verification in §12.5. Residual: an
  intermittent thrash regime (1 of 5 runs) owned by the §6 policy remainder,
  not the dispatch layer.

## 2. The evidence

### 2.1 It is a livelock, not a slowdown

Instrumented run, 256 requests / concurrency 128 / `total_pages 2151` /
`swap_pool_size 4096`, sampling the orchestrator every 500 ms:

| sample | waiters | free | head needs | suspends | restores | park requests posted | nothing_reclaimable |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 55 | 41 | 84 | 0 |
| 11 | 0 | 0 | 0 | 491 | 424 | 783 | **0** |
| 12 | 34 | 0 | 4 | **493** | **474** | 2,527 | **1,769** |
| 36 | 35 | 0 | 4 | **493** | **474** | 61,379 | 60,622 |
| 72 | 34 | 0 | 4 | **493** | **474** | 150,111 | 149,354 |

For the first ~6 seconds the ladder works perfectly: 491 suspends, 424
restores, ~11,000 pages moved, and `nothing_reclaimable` is **exactly zero**.
Then it flips, permanently and completely:

- `suspends` and `restores` never advance again — 493/474 for the remaining
  ~80 seconds.
- `posted` and `nothing_reclaimable` climb in lockstep at ~5,000/s, their
  difference pinned at a constant ~757. **Essentially every park request now
  ends in a victim with nothing to give.**
- 93 processes sit permanently in `ParkRequested`.
- The pool stays at **0 of 2,151 free** while the head asks for **4 pages**.
- Host swap sits at **4,079 of 4,096 slots free** — the swap rung is never
  even reached.
- D7 progress-deadline kills do fire (~1 per 10 s deadline window) and
  **release nothing**; the pool never gains a page from them.

### 2.2 The GPU is completely idle

`nvidia-smi` sampled at 1 Hz across both runs, model resident throughout:

| run | GPU utilization | duration |
|---|---|---|
| concurrency 128 (livelocked) | **0% in 82 of 82 samples, mean 0.0%** | 82 s |
| concurrency 61 (healthy) | mean **82.6%**, 13/14 samples >50% | 14 s |

Thrashing would show PCIe traffic and periodic compute. This shows neither.

### 2.3 The cliff is at exactly the resident-capacity boundary

Each request is 36 prompt + 512 output = 548 tokens = 35 pages; 2,151 pages
therefore hold **61** requests at full length. Sweeping only
`max_concurrent_processes`, everything else identical:

| concurrency | oversubscription | late arrivals | completed | wall | tok/s |
|---|---|---|---|---|---|
| 61 | **1.00x** | 195 | **256/256** | 13.2 s | **9,949** |
| 90 | 1.48x | 166 | 5/256 | 90 s (timeout) | 28 |
| 128 | 2.10x | 128 | 34/256 | 90 s (timeout) | 193 |
| 128 (192 requests) | 2.10x | 64 | 74/192 | 90 s (timeout) | 421 |
| 128 (128 requests) | 2.10x | **0** | **128/128** | 9.6 s | 6,861 |

The concurrency-61 run emitted **zero** contention-trace lines: the queue was
never non-empty. So that row is not "contention handled well", it is
"contention never engaged". The last row survives because the fixed cohort
finishes before pressure peaks — 64 extra requests (row 4) is enough to kill it.

### 2.4 Continuous admission is NOT the cause — the first draft was wrong

The concurrency-61 run has **195 late arrivals**, more than any failing row,
and completes flawlessly. Arrival of newly admitted young work is exonerated.
The variable that matters is whether the admitted set exceeds resident KV
capacity for long enough to force a suspension that cannot be satisfied.

### 2.5 The gap against vLLM is not an admission-policy gap

vLLM at the **identical** 34,416-token KV budget (`GPU KV cache size: 34,416
tokens`, confirmed in its log), told the **same** nominal concurrency
(`max_num_seqs = 128`), on the same 256-request workload: **256/256 in 9.3 s at
14,158 tok/s**. It never wedges because its scheduler simply runs the subset
that fits. pie at the same settings delivers 193 tok/s.

## 3. Root cause: the ladder keeps choosing victims that hold nothing

`select_costed` (reclaim.rs:1185) ranks candidates by

```rust
(!candidate.idle, class, cost, u64::MAX - candidate.seq)
```

`idle` is the **primary** key, ahead of the footprint class. So an **idle
process holding zero pages** (footprint `Some(0)` → class 1) sorts ahead of a
**running process holding enough to cover the head outright** (class 0). D6's
stated intent is "prefer idle *holders*"; the code never requires the candidate
to hold anything.

The consequence is a closed cycle. The selected zero-page victim honors the
park, freezes its pipeline, and `prepare_suspend` finds its working sets map to
**no page locations at all** → `NothingReclaimable` → `suspend_at_safe_point`
returns `Ok(false)` → the `ParkGuard` declines the park and resumes the
pipeline → the process is `Running` again and **immediately re-eligible**. It
is picked again on the next escalation, forever.

### How this was pinned down

`private_resident_pages` was instrumented to print, on every empty result, how
many of the victim's target pages were excluded for being shared with another
working set, held by a cache root, or already swapped — gated on
`target_pages > 0`. In a run that reached `nothing_reclaimable = 204,859`, that
print fired **zero times**. Since `NothingReclaimable` is returned only when the
private set is empty, every one of those 204,859 victims had
`target_pages == 0`: they were not holding pages that turned out to be shared,
they were **holding nothing to begin with**.

The three unit tests in §5 demonstrate the selection rule and the two
zero-page follow-on hazards directly, with no runtime needed.

## 4. Why "shared pages" is NOT the explanation

An earlier reading of this file blamed KV-trie sharing: greedy decoding
(`temperature = 0.0`, `ignore_eos = true`, common instruction) makes sequences
converge, the trie deduplicates their tails, and no page stays private. The
`target_pages == 0` result above rules that out as the mechanism — the victims
never held pages in the first place. The paragraphs below are retained only
because the exclusion logic is still worth understanding when reasoning about
the fix.

### The exclusion rule

`PageTable::private_resident_pages` (page_table.rs:1585) returns only pages
that are (a) reachable from the victim's working sets, (b) **not** reachable
from any other working set, (c) **not** anchored by a cache root, and (d)
resident rather than swapped. A page shared with anything else is excluded.

Two structural weaknesses remain worth noting for the fix, even though neither
is what wedged this run:

- **Nothing learns.** A victim that declines with `NothingReclaimable` is
  returned to `Running` unchanged, so the very next escalation may pick it
  again. There is no memory, no backoff, and no exclusion — which is what turns
  a bad pick into an unbounded spin rather than a single wasted cycle.
- **Two different notions of "what this victim is worth".** Selection scores
  candidates with `KvStore::exclusive_footprint`; the suspend path decides with
  `PageTable::private_resident_pages`. Nothing keeps them consistent, so
  selection can rate a candidate attractive that the suspend path then rejects.

### A separate wedge found on the way

Re-running the same configuration with divergent sampling
(`--temperature 1.0 --top-p 0.95`) never reached KV allocation at all — zero
contention-trace lines — because the scheduler stalled with
`in_flight_launches: batch of 2 (settled=false, age=160s)` and
`pending_binds=256`. That is an independent defect on the sampling path and
needs its own investigation; it is not the livelock described here.

## 5. The three selection defects, as unit tests

Each is demonstrated by a new test in `store::reclaim::tests`. All three pass,
i.e. all three behaviours are present. The first is the root cause above; the
other two are the follow-on hazards a zero-page victim would trigger:

1. **`diag_idle_zero_footprint_beats_a_covering_running_holder`** —
   `select_costed` sorts on `(!idle, class, cost, …)`, so `idle` outranks the
   footprint class. An idle process holding **nothing** is selected over a
   running process whose suspension would cover the head's demand outright.
   D6's stated intent is "prefer idle *holders*"; the code does not require the
   candidate to hold anything.
2. **`diag_zero_page_suspend_is_granted_a_restore_immediately`** —
   `report_suspended(pid, 0)` enqueues a restore demanding 0 pages, which the
   drain finishes on sight, returning the process to `Running` and making it
   instantly re-selectable.
3. **`diag_zero_page_suspend_resets_the_head_progress_deadline`** —
   `report_suspended` clears `inner.progress` unconditionally, so a suspension
   that freed nothing still buys the head a fresh 10-second deadline window.

(2) and (3) are not reachable from production today, because `yield_point_at`
returns early when the working-set list is empty. They remain live API hazards.

## 6. Proposed redesign

**Status: E3 (§6.3) is implemented and verified; the rest is proposal and has
not been signed off.**

### 6.1 The criterion

What failed here was a *performance* heuristic — D6's cost ordering, with `idle`
as the primary sort key — carrying *liveness*. When it chose wrong the system
did not get slower, it **stopped**. So the test any replacement must pass:

> Liveness must never depend on a heuristic. A heuristic may decide **which**
> victim is cheaper; it must never decide **whether** progress happens.

Everything below is organised around that separation: exact mechanisms hold
liveness and have no tuning knobs; heuristics are quarantined where being wrong
only costs throughput.

### 6.2 The shift

Today the subsystem answers **"who should yield right now?"**, recomputed from
scratch on every event, by every waiter. That question is *stateless*, so there
is nowhere to attach feedback — which is exactly why a declined victim leaves no
trace and is re-picked forever.

Instead, maintain **"what should the resident set be?"** as durable state. Three
things become possible that are impossible today:

| today | after |
|---|---|
| no object to ask "is the current allocation feasible?" | a checkable invariant |
| nowhere to record a refusal | reclaimability is state, so feedback has a home |
| cannot conclude "this rung is exhausted" | a computable predicate |

This is **scheduling, not admission**: it never decides who may *start* (that
stays with M5/Vesuvius), only who is *resident*. It therefore sits inside the
standing directive, and it is what the stated target — "match ideal admission
control WITHOUT admission" — actually requires.

### 6.3 Exact mechanisms — liveness lives here, no knobs

**E1. Explicit `RESIDENT` / `EVICTED` partition**, maintained rather than
recomputed, over **current** footprint:

```
Σ_{p ∈ RESIDENT} current_footprint(p) ≤ pool_capacity
```

Exact and self-maintaining: exceed it and something is evicted. (See §6.5 — an
earlier sketch said *projected* footprint, which smuggles a prediction into the
foundation of the invariant.)

**E2. Reclaimability is a state, invalidated by events — not timers.**

```rust
enum Reclaimability {
    Eligible { pages: u32 },
    Ineligible { reason: IneligibleReason },
}
enum IneligibleReason {
    HoldsNothing,      // cleared by an ALLOCATION event
    Pinned,            // cleared when this process's fires drain
    NoProgressYet,     // cleared by completing one fire (E6)
    CopyFailed,        // backoff — the one irreducible timer (H3)
}
```

`HoldsNothing` is not a guess, it is a **logical fact**: a process holding
nothing has nothing to yield, and cannot acquire anything by waiting — only by
allocating. So it leaves the candidate set until its allocation changes, and the
livelock becomes **impossible by construction** rather than unlikely. This is
the same move as D2's "the safe point is a state, not a place", applied to
suitability.

**E3. One definition of "reclaimable".** Delete `exclusive_footprint` as a
*selection* input; the planner asks the same `reclaim_quote` the suspend path
will answer, which returns a typed reason when it returns zero — feeding E2.
Pure de-duplication: with one implementation, drift is impossible.

**E4. Rung exhaustion as a computable predicate.** Rung 0 (reclaim unreferenced)
→ 1 (evict) → **2 (shrink the head's demand: partial/chunked grant)** → 3 (kill).
Escalation is driven by *"no feasible plan exists"*, not by a 10-second clock;
the deadline demotes to a bug watchdog. Rung 2 is new and matters: today an
oversized head has no option between "wait" and "kill somebody".

**E5. A single-owner planner** replaces per-waiter `escalate()`: compute an
eviction set covering the **whole deficit**, commit it, observe each step. Plan
size is exactly the deficit — no constant. Also collapses O(waiters × candidates)
footprint probing under the global KV lock to O(candidates).

**E6. Eviction requires prior progress.** A process promoted into `RESIDENT` is
not evictable until it has completed **one fire**. Not a duration — an event.
This bounds swap cost to at most one round trip per unit of work done, which is
provable rather than tuned, and it removes the forced rotation described next.

**E7. Refusal accounting, and the liveness assertion.** Every planner decision
emits deficit, plan size, per-step outcome, per-candidate refusal reason. Assert
continuously:

> `deficit > 0` with zero completed plan steps over N rounds is a **bug**, not a
> workload.

Any N works — it is a detector, not a policy. This is the invariant a mock-pool
unit test can hold, and it is what would have caught this defect.

Note E6 also fixes a structural forcing function: `enqueue_restore` currently
re-inserts a restore at the process's **original spawn seq**, so an old suspended
process always outranks a younger resident's allocation. Service order stays
spawn-clock FCFS (starvation-freedom preserved) — E6 only bounds how often the
residency may change.

### 6.4 Quarantined heuristics — performance only

**H1. Which covering subset to evict.** Irreducible, but it becomes a single
objective (minimise bytes moved subject to covering the deficit — a subset-sum,
with a known-bounded greedy on freed/moved ratio) instead of a pile of
tie-breaking rules.

**H2. Swap vs recompute.** Needs a cost model, so it is frankly a heuristic, and
the estimates are envelope-grade (~4–16 ms re-prefill vs ~6–20 ms for a 122 MiB
swap round trip). **Weakest item; independent of the rest; defer it** until
measurement shows it is needed. Its motivation is real, though: host swap is the
*only* eviction mechanism today, so `swap_pool_size = 0` disarms the ladder
entirely (§10.1).

**H3. `CopyFailed` backoff.** A transport retry policy — irreducible, and this is
the right place for it.

**The property that makes the design defensible:** if H1–H3 are all maximally
wrong, the system gets **slower**; it does not wedge. E2/E4/E7 hold liveness.
The current design has the opposite shape — one wrong performance heuristic put
the GPU at 0% for 82 seconds.

### 6.5 Two corrections to an earlier sketch

Recorded because both were knobs sitting in the liveness path, which is the very
thing this redesign exists to remove:

- **Residency tenure was first written as a minimum time quantum.** A duration is
  a tuning knob, and its size would have interacted with the D4 deadline. The
  exact statement is event-based (E6: one fire).
- **The feasibility invariant first used *projected* footprint.** Future KV need
  depends on how many tokens a process will generate, so that is a prediction —
  a heuristic at the root of the invariant. Current footprint is exact (E1).

### 6.6 What is kept

Not a rewrite of everything — these were sound under investigation: RAII grants
(`Drop` = rollback, no leaks observed); D2 safe-point-as-state and
`wait_yielding` racing the park signal (including the lost-wakeup guard in
`fire::acquire_grant`); D3 requester/victim unification; one queue with
head-first-claim as **service order**; kill via `process::terminate`
(quiesce-first, never a signal); and the `PoolPort` physics/policy split — E3's
quote *extends* that port rather than bypassing it.

### 6.7 Migration order

A second from-scratch rewrite is what produced this defect, so: smallest first,
each step independently testable.

1. **E3** — `reclaim_quote` + typed zero-reasons. Kills the two-definitions
   problem and the measured livelock.
2. **E2** — reclaimability state. Makes the livelock impossible by construction.
3. **E5** planner → 4. **E6** progress-gated eviction → 5. **E4** rung 2 + D7
   re-targeting → 6. **H2** recompute, only if measured to be worth it.

Steps 1–2 are small and close the measured bug. **E2 is the real defence**:
patching the sort key alone revives this workload but leaves the open-loop
controller intact, so the next reason a bad candidate gets picked reproduces the
livelock.

### 6.8 Open risks

- **E5's planner is a new serialization point.** That is engineering debt, not
  elegance, and it is only repaid by keeping `reclaim_quote` batching *outside*
  the global KV lock. If that slips, it becomes a different bottleneck.
- **E1 adds a state axis orthogonal to `ProcState`'s five states**, so the
  product space grows. Confirm the two axes are genuinely independent — each
  closed under its own single transition function — or this is a regression in
  clarity rather than an improvement.
- **H2's cost ordering is too close to call from arithmetic.** Measure before
  building on it.

## 7. Coverage gap

Nothing in the suite reaches this shape. `runtime/engine/tests/contention.rs`
spawns a **fixed fleet of 8** lanes against a 4-page pool — every lane holds
pages by construction, so the "victim holds nothing" case never arises. The
livelock needs sustained oversubscription *and* at least one idle, zero-page
process sitting in the candidate set. Test 1 in §5 reproduces the selection
rule deterministically and is the natural regression guard.

## 8. Instrumentation added for this investigation

Currently in the working tree, **not committed**, and gated so it is inert by
default:

- `ContentionStats` / `ContentionDiagnostics`: `escalate_no_candidates`,
  `escalate_posted`, `nothing_reclaimable`, a five-way `proc_states` histogram,
  and `eligible_victims` (how many candidates `escalate` may legally draw from).
- `ContentionOrchestrator::record_nothing_reclaimable`, called from the
  previously-silent `NothingReclaimable` early return in
  `inferlet/process/preemption.rs`.
- `bootstrap.rs`: a sampler behind `PIE_CONTENTION_TRACE_MS=<ms>` that prints
  one line per tick while anything is queued. It uses `println!`, not
  `tracing` — the embedded pyo3 server boots with `skip_tracing: true` and
  installs no subscriber, so a `tracing` event there goes nowhere. (That is
  itself worth fixing: the embedded server currently emits no structured logs
  at all.)

Decide separately whether the three counters and the sampler are worth keeping;
the `NothingReclaimable` counter at minimum should stay, since its absence is
what made this invisible.

## 9. Reproduction

```bash
cd benches   # patched harness: ~/Workspace/pie-wt/vllm-compare-records/pie_bench_cap.py
             # (adds --total-pages / --swap-pool-size passthrough; the shipped
             #  harness cannot express a real KV cap)
STAGE=<dir with Pie.toml + target/wasm32-wasip2/release/text_completion_bench.wasm>

# livelocks within ~6 s; GPU drops to 0% and stays there
CUDA_VISIBLE_DEVICES=0 PIE_BENCH_INFERLET_DIR=$STAGE PYTHONPATH=. \
  PIE_CONTENTION_TRACE_MS=500 \
  uv --project ../sdk/python-server run python .../pie_bench_cap.py tput \
  --model Qwen/Qwen3-0.6B --num-requests 256 --concurrency 128 --max-tokens 512 \
  --warmup 2 --driver cuda_native --device cuda:0 \
  --total-pages 2151 --swap-pool-size 4096 --request-timeout 90

# clean, for contrast (concurrency == resident capacity; contention never engages)
#   ... --concurrency 61 ...
```

## 10. Configuration traps found along the way

1. **`swap_pool_size` defaults to 0, which disarms suspend/restore entirely.**
   `kv_swap_capable` is derived from the driver's advertised KV copy mask; with
   no host swap pool the ladder degrades to pool-only reclaim. Same cap, same
   oversubscription, 128 requests: `0` → 57/128 completed; `4096` → 128/128.
   The mechanism this subsystem is built around is **off by default**.
2. **`--kv-pages` is dead for `cuda_native`** — never placed in
   `driver_options`, so it silently does nothing. Wire it or reject it.
3. **`--gpu-mem-util` does not cap KV residency** — it sizes only the CUDA
   memory planner's *logical* budget (`logical_kv_pages` / `kv_tokens`); the
   runtime exceeds it freely. The only binding knob is the driver option
   `total_pages`, which the shipped harness does not expose. **Any pie
   benchmark that believed it created KV pressure with `--gpu-mem-util` or
   `--kv-pages` did not.**

## 11. Loose ends (unrelated to the livelock)

- **A documented rule nothing enforces.** `DevGeo::has_mask` and
  `BoundForwardPass::dense_mask` both record that a fire carrying a dense
  device mask must be scheduled SOLO. Both are computed at bind and never read.
  Wire the rule or drop the claim.
- **`RegisteredProgram::pricing`** is computed on every program registration
  and has no consumer.
- **Pre-existing flake**: `scheduler::worker::tests::pipeline_close_drains_the_already_submitted_run_ahead_tail`
  fails roughly 1 run in 4 with "driver published RETRY at frame settle".
- **Docs/SDK drift**: `website/docs/reference/configuration.mdx` documents
  `[model.scheduler]` keys absent from the Rust schema (`batch_policy`,
  `default_endowment_pages`, `admission_oversubscription_factor`) and cites a
  path that does not exist (`runtime/src/inference/adaptive_policy.rs`); the
  Python SDK still carries `speculation_depth`, absent from the Rust
  `SchedulerConfig`, which would hard-fail `deny_unknown_fields` parsing.

## 12. Root cause #3 — sealed frame ⇄ copy barrier ⇄ pool resize deadlock

### 12.1 The frozen state (seal-fix verification run, sample ~36 onward)

With fixes 1 and 2 in place the run reaches 89/256 and then freezes hard for
the remaining ~70 s. The scheduler dump at the freeze:

```
pending (80):
  Launch fire 75077 …            <- sealed-frame member
  Launch fire 75078 …            <- sealed-frame member
  ResizePool                     <- position 3
  Launch fire 75080 … 75126      <- ~47 more sealed-frame members
  CopyKvTracked  x4              <- suspend D2H / restore H2D
  Launch fire 75127 … 75142      <- ~16 sealed-frame members BEHIND the copies
  CopyKvTracked  x4  (interleaved)
  CloseInstance x2, CloseChannels x18
  Launch fire 75143 … 75146      <- next frame (4 lanes, front_complete=true)
in_flight_launches (0):
in_flight_control: none
frame k=1 lanes=69 awaited=69 sealed=1 … watchdog=None
  sealed[0]: waves=1 fires=65 members=66
[contention-trace] waiters=45 free=0/2151 head_pages=1 head_granted=false
  … run=139 parkreq=13 quiesce=7 susp=7 restoring=0 posted=819
  nothing_reclaimable=0 …
```

Everything is idle — no launch in flight, no control in flight — yet a fully
sealed 65-fire frame never posts, 8 queued copies never dispatch, and a
ResizePool sits at queue position 3 forever.

### 12.2 The cycle

Three ordering rules, each locally sound:

- **R1 (frame atomicity)** — a sealed frame posts WHOLE; its fires dispatch by
  id, not queue position (`frame.rs plan_dispatch`).
- **R2 (copy barrier)** — a stamped fire queued *behind* a standalone copy
  (`CopyKv`/`CopyKvTracked`/`CopyState`) puts its lane in `blocked_lanes`, and
  a sealed frame holds while `members ∩ blocked_lanes ≠ ∅`
  (`worker.rs` queue scan, `barrier_seen`; `frame.rs:735-744`).
- **R3 (resize ordering)** — `rotate_launch_for_wave_work` refuses to rotate
  front launches past a `ResizePool` (it is not `standalone_copy`, not
  `lifecycle_control`, not `PreLaunchCopy`), because "pool resizes order
  against queued launches". The rotation scan looks at exactly ONE item — the
  first non-Launch behind the front — and gives up.

Composed, with a sealed frame straddling the {resize, copies} pair:

1. Frame F holds: ~16 of its 65 fires (75127+) sit behind the copies, so their
   lanes are in `blocked_lanes` → `plan_dispatch` returns `Hold(500µs)`
   forever, until the copies retire.
2. The copies never dispatch: the ready-item loop only acts on the queue
   FRONT. The front is Launch 75077 (a member of F); rotation is refused
   because the first non-Launch item is the ResizePool. `break`. The copies at
   position ~50 are unreachable.
3. The ResizePool never reaches the front: 75077/75078 leave the queue only by
   posting as part of F. Cycle: **F → copies → (queue reachability) resize →
   F**.

The deadlock also strangles the reclaim ladder itself: `quiesce=7` — seven
victims stuck Quiescing because their queued fires are sealed into F and can
never settle — so no pages free (`free=0`), the 45 waiters starve, and the
head (needing exactly **1 page**) never advances. GPU 0%.

Only contention creates this shape: suspend/restore inject standalone copies
and pool resizes into the middle of a live generation's queue ("a resize per
reclaim step"). Uncontended runs do copies/resizes at generation boundaries
where no sealed frame straddles them.

### 12.3 The internal contradiction

`worker.rs` states both of these, ~540 lines apart:

- `standalone_copy` doc (line ~2830): "These touch pages no queued fire
  references (suspend takes only unpinned drained pages; restore writes
  freshly reserved ones), so **a held wave must NEVER starve them** — the
  preemption ladder is what unsticks a held wave in the first place."
- Queue-scan comment (line ~3370): "A queued ASYNC control (standalone copy…)
  is a launch barrier: a fire enqueued behind it must not post until it
  retires."

The first comment is the design intent and is *correct*: grants pin every page
a queued copy touches, so no queued fire references them. The dispatch code
implements neither half — it starves the copies (they are only reachable via
front rotation, which R3 vetoes) *and* it barriers unrelated fires on them.

### 12.4 The fix (implemented 2026-07-25)

Safety precondition, verified in code first: **no queued fire can reference a
queued standalone copy's pages.** A suspend copies working sets whose fires
were purged at leave; a restored process is only woken after its H2D copy
retired (`preemption.rs::restore_from_park` awaits the tracked completion
before `resume_pipeline`). Grants pin every page a queued copy touches, a
shrink cannot trim granted pages, and no two queued copies can alias (a
restore's pages are only granted after the suspend that freed them retired).

Three changes in `scheduler/worker.rs`, no knobs:

1. **Barrier removed (edge 1).** `CopyKv`/`CopyKvTracked`/`CopyState` no
   longer set `barrier_seen`; the whole `barrier_seen` mechanism is gone.
   `blocked_lanes` now comes only from `PreLaunchCopy` (order-coupled to its
   consumer by construction). Same precedent as `ResizePool`'s earlier
   exemption (~45x). The queue pass is extracted as a testable
   `scan_queue()`.
2. **Out-of-band copy dispatch (edge 2).** `dispatch_ready_items` ends with a
   sweep: control slot free → the first standalone copy dispatches from ANY
   queue position. The queue front can be legitimately immovable (a gathering
   frame, a resize waiting out the pipe); the copies must never wait on it.
3. **Settling copies no longer hold frame posting.**
   `PendingControl.holds_launches`: true for `PreLaunchCopy` (consumer
   coupling) and pool resizes (pipe drain), false for standalone copies — so
   frames keep posting while suspend/restore traffic settles, removing the
   ~25%-of-wall serialization the single control slot would otherwise impose
   under churn (d2h+h2d ≈ 24 s per 90 s run).

Regression tests:
`standalone_copy_dispatches_out_of_band_past_an_immovable_front` (the exact
frozen shape), `queued_standalone_copies_and_resizes_never_block_lanes`,
`a_settling_standalone_copy_does_not_hold_frame_posting`. Engine lib suite
368/368 green.

### 12.5 Verification (5 runs, 256 req / conc 128 / 2,151 pages)

| run | completed | wall | tok/s |
|-----|-----------|------|-------|
| fix3   | 256/256 | 14.4 s | 9,090 |
| rerun  | 109/256 | 90 s timeout | 620 |
| rep1   | 256/256 | 16.4 s | 7,999 |
| rep2   | 256/256 | 16.4 s | 7,998 |
| rep3   | 256/256 | 15.7 s | 8,345 |

Reference: ideal admission (M1, concurrency 61 = resident capacity, contention
never engages) is 256/256 at 9,949 tok/s. The best fixed run reaches **0.91x
of that with no admission control**; typical runs 0.80x. GPU utilization
during completing runs: 79% overall, 90% when busy (was 0% for 82/82 samples
in the livelock). The old M3 (same config, pre-fix): 34/256 at 193 tok/s —
a 47x throughput recovery.

**The deadlock is gone in every run** — counters never freeze, `kills=0`,
`nothing_reclaimable=0`, quiesce always drains. The one failing run is a
different regime: **thrash**. Its trace shows 900 suspend/restore cycles
(vs ~300 in a completing run), ~55 GB of PCIe traffic, cumulative copy time
180 s in a 90 s wall — the reclaim ladder oscillates instead of holding a
stable resident set, and requests miss the 90 s per-request timeout. Counters
move throughout; it is not a liveness failure.

That regime is precisely what §6's unimplemented remainder targets: per-event
"who yields?" selection has no memory and no plan, so once the uniform decode
waves desynchronize, eviction choices oscillate. The fix belongs to E1/E2/E4
(resident-set planning, reclaimability as state, hysteresis via the progress
deadline), not to the dispatch layer — the dispatch layer now provably does
what the policy asks of it.

## 13. Roomy regression — the rewrite costs −17% with contention idle (2026-07-25)

Prompted by the operator's observation that pie should not trail vLLM without
KV pressure. Confirmed by a four-round bisect over the unsquashed lineage
(engine rebuilt per commit in the `kilimanjaro-baseline` worktree; identical
harness, roomy 256 req / conc 128 / util 0.90):

| commit | tok/s | avg batch | pipeline |
|---|---|---|---|
| 88bf1e38f (pre-rewrite tip) | 17,240 | 13.8 ms | 2-deep overlap |
| B1 `681a47f3b` burn dead code | 17,695 | 13.4 ms | overlap — clean |
| **B2 `29d0b190e` no off switch** | **14,607** | **7.7 ms** | **serial — the break** |
| B4 `285394b3a` | 14,527 | 7.7 ms | serial |
| B6 `d52dc0d5f` | 14,129 | 7.7 ms | serial |
| HEAD + this session's fixes | 14,178 | 7.7 ms | serial |

Reference: vLLM 17,187 @conc128, 19,376 @512req/conc256; pre-rewrite pie is
**parity on both shapes** (17,240 / 19,397). Post-B2 pie is 0.825x.

**Mechanism.** Pre-B2, a run without `PIE_KV_CONTENTION` had no orchestrator:
`contention()` returned `None`, gates no-op'd, and a fire's KV demand took the
legacy direct-allocation path. B2 (by directive: no mode, no off switch)
installs the orchestrator always and routes **every fire's KV demand through
the grant path** — a global `with_inner` mutex plus FCFS bookkeeping per fire,
128 lanes per step. That pushed the boundary gather (~guest resubmit → seal)
from just under the 7.4 ms execute window to ~9.0 ms, so the frame N+1 seal
now lands after frame N retires: the Venus run-ahead overlap (depth 2)
collapsed to serial posting with a 1.4 ms exposed gap. The avg-batch-latency
signature (13.8 ms in-flight with 7.4 ms steps → 7.7 ms in-flight with 9.0 ms
steps) is decisive.

**The directive is not the defect; the implementation is.** Always-on
semantics does not require paying queue bookkeeping when nobody is waiting.

**Fix direction (no knob, unchanged semantics):** an uncontended fast path in
grant acquisition — one atomic summary check (free pages cover the demand ∧ no
waiters ∧ no parks) takes pages directly and skips the queue; any failure
falls into the existing FCFS path. The futex pattern: uncontended = one CAS,
contended = queue. Expected recovery: ~17.2–17.7k @conc128 (vLLM parity), and
it lifts every contended number too — the M1 "ideal ceiling" (9,949) itself
pays this tax, so the §12.5 ratios understate the fixed engine.

Open question: Stage V recorded a roomy A/B floor of ≥0.982x vs the
pre-rewrite baseline — that measurement cannot have covered this
configuration and needs a post-mortem.

## 14. Rainer v1 — implementation record (2026-07-25, evening)

The rainer.md design landed in one shot: legacy retired wholesale, new
implementation upfront, verification deferred (operator directive).

**New:** `src/planner.rs` (ResidencyPlanner: FCFS registry, one BTreeMap
queue over spawn seq, planner-level head-first accumulation, eviction
planning, hog + starvation predicates, diagnostics), `src/planner/grant.rs`
(Demand/AllocationGrant RAII, carried over), `src/planner/exec.rs` (evict:
fence → leave(Close) → detachable drain → lease quiesce → prepare → D2H →
commit; restore: prepare → H2D → commit → unfence, bounded transport
retries then fail-loud terminate), `src/inferlet/process/gate.rs`
(residency gate: one atomic load when everyone is resident; drains own
pending fires then parks when evicted), `src/inferlet/process/teardown.rs`
(defer_resource_teardown, moved verbatim from preemption.rs).

**Deleted (~4,000 lines):** store/reclaim.rs + reclaim/{grant,queue,state},
inferlet/process/preemption.rs (safe points, yield_point at 23 prologues,
wait_yielding, ParkGuard, serialize_under_contention, restore_from_park),
the 5-state ProcState machine, the per-fire orchestrator gate, the progress
deadline + D7 kill rung, PIE_KV_EXHAUSTION_MS, the bootstrap
leave/kill/probe hook seams, and the scheduler's freeze/resume machinery
(FreezePipeline/ResumePipeline items, frozen_pipelines set,
LeaveKind::Suspend — all had become dead).

**Key mechanisms vs the design (deltas):**
- The fire hot path keeps the existing demand→acquire→prepare loop shape;
  `acquire` fast path = two free-list pops gated on two atomic loads
  (waiters == 0 && nonresident == 0). No plan-time batched frame grant in
  v1 — the planner is event-driven (parks/frees/transfer landings), which
  is boundary planning with lookahead 0; the K-frame delta lookahead of
  rainer.md §2 is a follow-up optimization, not a structural change.
- The suspend seal is the working-set FIRE LEASE promoted to a Dekker pair
  with a new per-lifecycle `suspend_fence` atomic (store/kv/working_set.rs):
  fires take the lease after any park and before prepare and hold it
  through finalize; the evictor fences, drains detachably, then awaits
  lease quiescence (event-driven Notify) before prepare_suspend. This is
  the exact no-torn-copy guarantee the old design got from victim-side
  self-suspend.
- Restore demand is re-read from the store (swapped count) at serve time;
  restores are accumulation-gated, oldest evictee first.
- The old D7 deadline kill is replaced by TWO computed predicates: Hog
  (head.held + head.need > total → fail the head) and Starved (eviction
  cannot fund + no transfer in flight + zero fire leases fleet-wide → fail
  the YOUNGEST parked ask). No timers anywhere.
- tests/contention.rs and tests/contention_host_full.rs ported to planner
  diagnostics; host_full now asserts starvations_total (no kill exists).

**State:** engine lib + all targets compile; 340/340 lib tests green;
pie-server-py green. NOT yet run: contention e2e pair, pressure bench
(256-req 5/5), roomy A/B vs vLLM parity (the B2-toll recovery this design
exists to prove). Known deliberate gaps: no grow rung (logical capacity is
fixed; elastic VMM trim is physical-only today), device-geometry fires are
non-detachable so a victim with a pending device-geom op is skipped, and a
process idle in a host await holding pages is treated as evictable only
via D2H (never killed).

### 14.1 Bench round 1 — eviction-leave wedge (found and fixed)

First contended run (128req/c128 @2151 pages): 0/128, planner trace frozen
for 300 s at `evicting=1` — a NEW deadlock, introduced by v1's eviction
leave. `exec::evict` posted `LeaveKind::Close` with `lane = pid`, but Close
is LANE-keyed and the frame policy's lanes are keyed by pipeline-scope id —
the leave matched nothing, the victim's silent lane stayed awaited, the
next boundary never sealed, and the second eviction's drain waited forever
on a completion inside that unsealable frame. (The old ladder never hit
this: its Suspend leave ran `on_process_leave`, and its victims drained
themselves BEFORE leaving.)

Fix: reintroduced `LeaveKind::Suspend` with new no-purge semantics —
`FramePolicy::on_process_suspend(owner)` clears `awaited` on every lane the
victim owns while keeping already-submitted frames sealable (the graceful
pipeline-close shape, process-wide), so the tail drains and releases the
leases the eviction quiesces on. Purging instead would orphan the queued
fires WITH their leases. Also hardened the eviction drain with
`try_finalize_guard`: never wait out the guest's own finalize gate (held
across unbounded channel awaits) — abort the attempt and re-plan.

### 14.2 Bench rounds 2-3 — two pre-existing completion-wakeup bugs, exposed

The remaining contended freezes (evicting=1 stuck at the eviction drain,
scheduler empty) came down to two latent engine defects around
`WorkItemCompletion`, both predating Rainer:

1. **The resolve doorbell could be filtered.** `resolve_success/failure`
   rang the waker with `publish(slot, 1)`. If the driver had already
   published a HIGHER per-fire epoch to that slot (its terminal notify) and
   the waiter re-registered after observing it, the monotonic epoch filter
   dropped the `publish(1)` as stale — the waiter slept forever on a
   settled fire. Fixed: resolve now uses the unconditional `wake(slot)`
   (the resolution lives in an atomic outside the epoch machinery).
   This window (driver-notify → worker-retire) is exactly where the
   pre-existing `pipeline_close_drains` RETRY flake (#10) and possibly the
   --temperature 1.0 wedge (#15) lived.

2. **The waker table parks ONE waker per slot** (`state.waker.replace`).
   The codebase's implicit invariant was one awaiter per fire completion,
   enforced by convention: every op-await ran on the owning guest's task
   (or under the pipeline finalize gate). Rainer's eviction drain broke the
   convention — it awaited an unsettled op's completion while the guest
   could await the same completion from `drain_rs_predecessors` (which
   takes no gate); the second registration overwrote the first waker and
   whichever task lost slept forever. Fixed structurally: the planner
   never awaits a fire completion — the eviction drain finalizes SETTLED
   detachable ops only (their poll never registers), and lease quiescence
   (a tokio Notify, multi-waiter safe) is the eviction's only wait.

Also in these rounds: the Yield-at-acquire mechanism (14.1's successor) —
marking a victim Evicting wakes its parked asks with `Acquired::Yield`; the
fire path settles the victim's own tail (the only task that can finalize
device-geometry ops) and waits out the eviction. The rollback-spin from
14.1 (10,969 rollbacks/run) is gone: rollbacks now occur only on real
prepare/copy failures.

Residual (mock 4-page torture geometry, ~2/12): a lane wait-set hole
(awaited lane, no frames, no leave — same family as 14.1) and a starvation
misfire under total swap exhaustion. Both need the GPU-geometry bench
verdict to prioritize.

### 14.3 Bench rounds 4-5 — liveness closed; the remaining gap is gather serialization

With the 14.2 fixes plus the serve CASCADE (the old drain's
younger-from-surplus rule, restored: a served head no longer blocks the
next unmet ask from absorbing remaining free pages), the GPU verdict
(RTX 4090, Qwen3-0.6B, 2026-07-25 late):

- **Roomy: fully recovered.** 256req/c128: 17,236–17,283 tok/s across four
  runs (vLLM 17,187; pre-rewrite pie 17,240; pre-Rainer 14,178 — the B2
  −17% toll is gone). 512req/c256: 19,402 (vLLM 19,376; pie-base 19,397).
- **Contended h2h (128req/c128 @34,416 tok matched): 128/128 in 6/6 runs**,
  5.5–6.7 K tok/s vs vLLM 13,939 (fresh confirm) — 0.42–0.48x.
- **Pressure (256req/c128, same cap + 90 s request timeout): 256/256 in
  6/6 completed runs** at 4.5–5.8 K tok/s (one early 75/256 run was
  pre-cascade). No thrash signature anywhere: d2h/h2d page counts stay
  symmetric, evict/restore cycles are bounded, no oscillation.

Bottleneck attribution (measured, not guessed): the cascade cut parks per
h2h run from 1,563 to ~68 — and throughput did NOT move, so park round
trips are NOT the gap. The batch histogram is: 852 batches, avg ~77 rows,
6.87 ms/batch = 5.85 s of execution in an 11.1 s wall — **47% of the run
is inter-batch gap**. Under pressure the fleet's ~4 pages/frame of fresh
demand is funded reactively (evictions land only after asks park), so the
gather serializes behind execution instead of overlapping (the same
run-ahead collapse shape as the B2 regression, now caused by page supply
latency instead of a grant gate). The elder-of-head fast-path bypass did
not help because under this workload the head is an early evictee's
restore — almost every resident is younger.

The structural answer is rainer.md §2/§10-step-2 as designed: a residency
ledger with per-lane positions, K-frame delta lookahead, and plan-time
batched grants at the boundary, so page supply is staged BEFORE the demand
arrives and the gather never waits. That is the next phase; the reactive
v1 is liveness-complete and roomy-clean.

### 14.4 Final round-4 tally and the two residual defects

Round 4 (elder bypass — no measurable effect, see 14.3): roomy 17,246;
h2h 128/128 x3 (5.5-6.7K); pressure 4/5 complete (4.5-5.2K), one 77/256
run frozen at evicting=1. Rounds 3+4 combined: h2h 8/8 complete,
pressure 9/10.

The frozen run's trace shows the two remaining defects precisely:
1. **Evict/restore ping-pong (E6 regression):** one pid cycled
   evict(9p) → restore(9p) back-to-back for seconds (3.9 s of H2D in a
   ~40 s window). v1 dropped the old design's E6 rule — a restored
   process must make progress (one fire) before it is evictable again.
2. **A rare evict stall between "quiesced; preparing" and the D2H post**
   (all-sync code — no marker, no panic-watcher output), alongside
   repeated "restore prepared" lines with no commit and
   restore_failures=0. ~1/10 pressure runs; same family as the mock
   e2e's residual y12 shape. Needs a live stack (ptrace_scope) or
   finer markers. Filed as #25.

## §15 Phase-2 probe: the contended gap is guest-turnaround serialization (2026-07-25)

Setup: E6 landed (Proc.progressed — no re-evict before the next acquire;
zero roomy cost) plus timestamped markers on one monotonic clock:
`PIE_FIRE_TIMING` wave records, planner park/serve/collect, exec step!,
and build-path markers (hp-enter at WIT arrival, hp-acquire at the
planner ask, rx-wake/rx-finalize in the take path).

E6 verification: roomy 17,141 (parity band); h2h unchanged (5.3–6.0K —
ping-pong was never the h2h bottleneck); restore→same-pid-re-evict now
min 26 ms / p50 283 ms (was back-to-back).

Measured anatomy of one h2h run (p2-h2h-1): 1,361 waves, avg 48.9 rows,
busy 5.93 s of 13.4 s span = **55.7% idle**, concentrated in ~306 gaps of
10–45 ms. Everything the planner does is fast and off the critical path:

- park→serve p50 = 0.0 ms (p90 12.3, p99 32) — accumulation + cascade work;
- serve→collect p50 = 0.3 ms — guest wakeup from the planner is prompt;
- evict chain p50: start→quiesced 0.5 ms, quiesced→D2H-post 0.3 ms,
  D2H-post→commit 4.4 ms; n=80. Quiescence is NOT the bottleneck.

The gap anatomy (p2-h2h-4, hp markers): after a contended wave completes,
the fleet's next fires arrive at the host **strictly serially, one guest
every 0.25–0.5 ms** (hp-enter drip), and the wait-all frame seals only
when the last lane fires-or-leaves: ~70 guests x ~0.4 ms ≈ the 30–40 ms
gap. hp-enter→hp-acquire is ~10 µs (the entire build preamble — settle
drain, wiring, demand calc — is innocent), and scheduler-side
enqueue/ready lags are ~0 (fire records). Baseline (uncontended) waves
show the same serial shape at ~0.12 ms/guest, hidden inside the 7 ms GPU
frame; roomy at ~58 µs/guest fits entirely. Contention multiplies the
per-guest serial cost ~3–8x and it becomes the critical path.

Suspects for the contended multiplier (all on the guest's WIT path only
when nonresident/waiters are nonzero): the residency-gate `is_resident`
planner-lock hit per call; the take path's inline finalizations
(KV lock) each poking `pages_freed` → an **inline `plan()` cascade on the
guest's own task**; `is_elder_of_head` + `note_progress` planner-lock
hits per acquire. The convoy hypothesis (rx-wake discriminator pending):
guests serialize behind the planner/KV locks while plan() churns.

Fix direction = E5 as designed (single-owner planner): plan() runs on ONE
dedicated task; every hot-path call site (parks, pages_freed pokes,
collects, executor callbacks) only *pokes* it. Guests never run the
drain. Fold note_progress into the elder check (one lock hit, not two).

Separate item: two deterministic-position silent stalls (~50 ms at ~2.3 s,
~317 ms at ~10.0 s; zero engine events; one D2H commit measured 311 ms /
157 ms in that window) — ~5% of idle, unexplained, possibly driver-side
(graph capture?). Not the main story.

### 15.1 E5 landed; the residual is a wake-herd lock storm (A/B-isolated)

E5 (single-owner drain task; hot paths only poke) + the idle-reclaim latch
+ E6-note folded into the elder check. Results, same h2h matrix:

- traced runs (PIE_CONTENTION_TRACE_MS): **12,904 / 12,946 / 12,958 tok/s**
  = 0.93x vLLM (13,939). Idle 55.7% → **5.9%** (gaps >2 ms: 306/7.15 s →
  13/0.05 s). The deterministic ~317 ms silent stalls vanished too — they
  were D2H commits stuck behind the same lock convoy.
- untraced runs: **8,297–8,543** — consistently slower than traced.
- A/B isolation: PIE_CONTENTION_TRACE_MS alone → 12,945 (fast regime);
  PIE_FIRE_TIMING alone → 8,407 (slow regime). The *guest-path probe
  printlns* create the fast regime: the stdout lock time-slices the wake
  herd, so the guests' KV/planner-lock critical sections stop colliding.

Regime anatomy (identical GPU cost per row ~89–90 µs in both): slow =
coherent herd, fat 77-row waves at ~9 ms cadence with a ~2 ms lock-storm
gap per wave (23% idle); fast = staggered arrivals, thin 48-row waves at
4.55 ms cadence, ~0 idle. Throughput is purely 1/(1+idle).

Shippable fixes attempted/planned, in order: (1) parking_lot for the two
storm locks (KvStore global + planner inner) — adaptive spinning instead
of futex round trips per handoff; (2) if insufficient, the structural cure
is rainer.md §2's boundary-batched grants: ONE task prepares the frame's
page allocations at the boundary instead of ~70 guests each taking the
global KV lock in their turnaround — which is also phase 2's remaining
component, so the herd fix and the design converge.

### 15.2 The lane-resurrection seal wedge (#25-B root cause) and its fix

parking_lot (15.1 fix 1) flipped untraced runs into the fast regime —
h2h 12,982, press 12,021 (256/256) — but 2 of 3 untraced h2h runs then
failed: one 300 s seal wedge (0/128) and one driver fail-stop. The wedge's
watchdog dump is unambiguous: 95 lanes awaited, 94 with complete frames,
and ONE lane `awaited=true queued_frames=0 front_complete=false` holding
the fleet's seal hostage — exactly the e2e-y12 shape filed in #25.

Root cause (static, confirmed by `record_arrival`): the victim's last
pre-fence fire can arrive at the worker AFTER the evictor's process-wide
Suspend leave was already processed. `record_arrival` recreates the lane
`awaited: true` (`or_insert_with`). The fire seals, executes, retires —
and the lane is now awaited with zero queued frames while its guest parks
in `wait_resident` with NO further leave posted: the Yield / lease-Fenced
/ residency-gate paths all violated the frame-policy contract ("any
guest-side wait that stops a lane's next fire MUST post a leave"). The
seal then waits forever; other victims' evictions cannot quiesce (queued
fires in the wedged frames hold their leases), so the whole engine
deadlocks. The faster de-cohered regime multiplied leave/rejoin churn and
took the race from ~1/10 pressure runs (#25) to ~2/3 h2h runs.

Fix: every `wait_resident` park site (acquire's Yield arm, both build
loops' lease-Fenced arms, copy_into's Fenced retry, the residency gate)
now re-posts the process-wide `LeaveKind::Suspend` leave first. The leave
is idempotent (`on_process_suspend` retains/clears), ordered after the
racing fire's arrival in the worker queue, and costs nothing outside
eviction paths. The probe printlns made the traced repro loop pass 4/4 —
the window sits exactly where the markers add jitter — so verification is
untraced (p5 round).

### 15.3 Phase-2 round verdict (final)

p5, untraced, wedge fix in: h2h 12,873 / 12,900 / 12,911 / 12,927 / 12,928
/ 13,013 — six for six 128/128, spread <1.1%, **0.925–0.933x vLLM**
(13,939; was 0.42x at §14.4). Pressure 11,677 / 11,923, both 256/256.
Roomy 17,156 (parity band). 340/340 lib tests (one 1/3 flake, 0/3 on
repeats), both contention integration tests green.

Scoreboard for the whole arc: reactive ladder 0/128 frozen → v1 liveness
fixes 5.5–6.7K → E5 + latch 8.3–8.5K (coherent herd) → + parking_lot
12.9K but 2/3 runs dead on the resurrection wedge → + leave fix: 12.9K,
9/9 clean. Remaining candidates: boundary batched grants (~7% + press
headroom, fatter waves at 58 µs/row), the one-off geometry fail-stop, and
a #15 retest (the wedge fix plausibly covers it).

## §16 Rainer v2 precondition probes — the projected upside mostly collapsed

Before prototyping v2 (rainer.md Part II), its two falsifiable
preconditions were measured (2026-07-25):

1. **Worker serial ingest** (the feared next bottleneck): the worker loop
   already carries a per-pass census (`worker_pass`, >2 ms passes,
   PIE_FIRE_TIMING). Result: ZERO >2 ms passes in fast-regime h2h, press,
   AND roomy with full 128-fire bursts. No hidden 1–3 ms monster; v2
   precondition (ii) passes trivially.
2. **KV-lock budget** (what boundary batching could amortize;
   PIE_KV_LOCK_TRACE, 1 µs threshold): h2h hold total 278.6 ms over a
   6.28 s span = **4.4% utilization** — nowhere near saturation. Roomy:
   0.3%. The pain is not hold volume but ONE long holder: tag `planner`
   holds 246.6 of those 278.6 ms (p50 1.4 µs, p99 483 µs — plan_eviction
   quoting EVERY candidate under a single hold), against which guests
   accumulate 3.9 s of aggregate wait (host-other 1.29 s +
   host-working-set 2.52 s). A convoy seeded by long holds, not a
   throughput-of-locking problem.
3. **Bonus — forced-coherent floor**: PIE_SCHED_MAX_IN_FLIGHT=1 h2h runs
   **12,216** vs depth-2's 12,973 (−5.8%), with an essentially identical
   fat-wave histogram (~76 rows avg in both). The 35% regime cliff of
   §15.1 is CLOSED post-parking_lot; the current steady state is already
   fat/coherent, and run-ahead contributes ~6%.

Verdict: v2's two headline purchases — regime-cliff elimination and
lock-batching — are respectively already in hand and not worth buying
(4.4% utilization). v2 remains the right shape for the *seam class*
(membership push vs inference), but as a performance project it is
DEFERRED; the evidence points at one targeted fix instead: lazy
youngest-first chunked quoting in plan_eviction (stop at deficit
coverage), killing the 483 µs p99 planner hold. Landed same day;
p7 is the verification round.

p7 verdict (lazy quoting): h2h 12,811 / 12,988 / 13,029 — the p5 band
exactly; press 10,713 (press spreads 10.7–12.0K across rounds); roomy
17,187. No regression, no measurable gain — the long-hold convoy was not
the binding constraint either; the aggregate wait is many small handoffs,
and the residual ~7% is resident-set size + per-row overhead at ~76 rows.
Kept (bounded holds are strictly better hygiene). v2 DEFERRED as a
performance project — see rainer.md Part II status.

### 16.1 Hygiene pass (#26)

1. **Choke-point leave**: the process-wide Suspend re-leave moved INTO
   `wait_resident` (posted once, only when actually parking) — the five
   copy-pasted park-site blocks are deleted; every current and future
   park site is covered by construction.
2. `is_elder_of_head` renamed `note_ask_and_check_elder` — the E6
   side effect is now in the name and doc; `note_progress` takes a plain
   lock (no mirror recompute); the zero-demand path references the same
   event.
3. Drain-path test coverage: discovered already present — the two
   contention integration tests bootstrap inside a tokio runtime, so
   `arm_drain_task` fires and they exercise the ARMED async path (the
   inline fallback is only the planner-less unit-test world).
4. `re_arm_idle_reclaim()` — one semantic site for the exhaustion-latch
   clear (was scattered across three callers).
5. **KV taint guard**: parking_lot lost std's poisoning; `with_kv_lock`
   now sets a taint flag when a panic unwinds mid-operation and asserts
   it on every entry — fail-loud restored for half-mutated stores.

Verified: 340/340 lib + 2/2 integration; p8 bench h2h 12,908/12,907,
press 10,863 (256/256), roomy 17,215 — the p5/p7 band exactly.

## §16.2 Pre-commit cleanup pass (#27, 2026-07-25)

A four-angle review (reuse / simplification / efficiency / altitude — four
independent agents over the full uncommitted tree) followed by a fix pass.
Everything applied is behavior-preserving; the two hot-path items change
WHERE work happens, not what happens.

**Applied — structure**

- `settle_and_wait_resident` (fire.rs): the eviction back-off protocol
  (settle tails → wait out the eviction, §15.2 leave contract) had been
  hand-copied at three fire-submit sites (acquire_grant Yield, host build
  Fenced, device-geometry Fenced). One owner now; the sites are one-liners.
- `finalize_all` (fire.rs): the full FIFO finalize loop existed three times
  (residency gate, process teardown, forward-pass drop). One owner, with
  the teardown's log-and-continue policy as a parameter.
- Typed leave API (worker.rs): `notify_pipeline_leave(_owned)` overloaded
  its first id by kind — lane-scope for Close, process for Suspend /
  Terminate — the §15.2 seal-wedge key-space trap, re-enterable by any new
  caller. Replaced by `notify_lane_close(scope, owner)` /
  `notify_process_suspend(pid)` / `post_process_terminate(pid)` over one
  private poster; the key space is now fixed by the function name.
- `holds_launches` (worker.rs post_control): five per-arm literals whose
  values had to stay exactly complementary to `standalone_copy` (the §12
  deadlock class) are now computed from it.
- Unfence moved into `report_restored` (planner.rs): "restored ⇒ unfenced"
  was enforced by remembering to call a local closure before each of the
  restore executor's three success exits; a forgotten call would be a
  silent Resident-but-fenced per-process wedge. The planner's report path
  owns it now; failure paths keep fences up by not reporting.
- `Inner::unmet_head()` / `Waiter::kv_need()` (planner.rs): the FCFS-head
  lookup was spelled out seven times, the demand extraction four times.
- `spawn_watched` (exec.rs): one spawn-plus-join-watchdog scaffold for both
  executors.
- `trace_mark!` (planner.rs): one owner for the pid-tagged marker format;
  `step!` and the six hand-rolled build/rx marker blocks forward to it.
  Output is byte-identical (the analysis scripts keep parsing).

**Applied — hot path**

- The `Impossible` pre-check moved out of `acquire`'s entry (it cost a
  global-KV-lock stats read per nonzero fire, reading an immutable total)
  onto the park path: a demand that reserves off the free lists trivially
  fits. One store-lock hold per fire removed from roomy AND contended.
- `PoolPort::reserve_device_up_to`: the drain's absorb step was a stats
  read + sized reserve = two KV-lock holds per absorb per poke; now one.

**Applied — dead code / docs**

- Deleted: `planner.stats()` / `planner.config()` accessors (no callers),
  the unused `NoReclaim` re-export, `ProcessResidency::snapshot()` (folded
  into `teardown_snapshot`), a dead test binding. `realize_declaration`
  (store-allocating variant) is `#[cfg(test)]` — production is
  grant-funded only. `cancelled_waits` / `host_swap_exhaustions` counters
  were write-only; now surfaced in `PlannerDiagnostics`.
- Doc lies fixed: exec.rs module doc claimed a non-detachable entry aborts
  the eviction (it is skipped; quiescence is the gate); `active_leases`
  claimed the starvation predicate reads it (it deliberately does not);
  `with_inner` claimed to be the single mutation chokepoint (the E6 note
  paths lock directly); stale "contention ladder / orchestrator" mentions;
  the KV taint guard now documents its process-global blast radius.

**Deliberately NOT applied**

- `copy_into`'s Fenced arm parks WITHOUT settling the process's pipeline
  tails — unlike every fire-submit site. An unsettled fire elsewhere keeps
  its lease and the eviction's quiescence then depends on another
  finalizer. This is a LIVENESS question, not a style one: documented at
  the site, tracked under the §15.3 watch items (#25). Do not "align" it
  in a cleanup pass; align it (or prove it safe) with a test.
- `with_inner`'s O(procs) mirror recompute per exit: deliberate
  computed-not-mirrored design; measured as not the residual bottleneck
  (§16 — the residual is resident-set capacity). Revisit only if fleet
  sizes grow an order of magnitude.
- `Inner.evicting` map (mirrors `Proc.state == Evicting` + expected pages):
  derivable state, but the sync sites are exactly the state-flip sites and
  the liveness predicates read it; not worth the churn pre-commit.
- The `outcome: Option<Result> + yielded: bool` encoding (one meaningless
  state) → a three-variant enum: mechanical but touches every delicate
  match in the planner; deferred.
- `acquire`'s outer `loop` never re-iterates (all arms return) — dead
  control structure, left to avoid a 90-line indentation-only diff.
- The lease-refusal path in `acquire_fire_lease` is NOT byte-identical to
  `KvFireLease::drop` (refusal calls `maybe_release` unconditionally, drop
  only at zero) — the review's dedup suggestion was unsound; left as is.

Verified after the pass: 340/340 lib (the pipeline_close_drains flake, #10,
fired at its usual rate — failure is the pre-existing dummy-driver RETRY
race, path semantics unchanged), 2/2 contention integration, and a p9
release bench round (below).

p9 (release, post-cleanup): h2h 13,055 / 12,935 both 128/128 (p5–p8 band
12,873–13,029 — upper edge, consistent with the removed per-fire stats
lock), press 10,752 @ 256/256, roomy 17,191. No regression anywhere.

## §16.3 Final pre-commit regression matrix (#28, 2026-07-26)

Eight scenarios, pie (post-#27 cleanup, release) vs vLLM 0.22.0, one 4090,
Qwen3-0.6B, sequential runs (f1–f8 tags in the bench archive):

| scenario                          | pie              | vLLM            | ratio  |
|-----------------------------------|------------------|-----------------|--------|
| roomy 256req c128                 | 17,213 / 17,206  | 17,221          | 1.000x |
| roomy 512req c256                 | 19,351           | 19,380          | 0.999x |
| contended h2h (2151pg / 0.217)    | 12,992–13,012 ×3 | 13,919 / 13,923 | 0.933x |
| contended press 256req (256/256)  | 11,017 / 10,948  | 14,149          | 0.775x |
| contended long-gen 64×1024 c64    | 7,330            | 7,698           | 0.952x |
| caps present, demand below cap    | 9,201            | 9,162           | 1.004x |
| single-request latency (tok/s)    | 519              | 486             | 1.068x |

Every pie cell sits inside its historical band; no regression anywhere.
New reference points: vLLM press 14,149 (the press gap 0.78x is the widest
cell — same resident-set capacity residual as h2h, §16, amplified by
sustained churn); long-gen 0.95x is a first measurement of the deep-decode
eviction regime.

**temperature 1.0 (f8): wedges — and is NOT ours.** The contended temp-1.0
run timed out at warmup (2 lanes, a launch batch stuck `settled=false` for
849 s, planner uninvolved). Isolation: reproduces roomy (no caps, 16 req)
on the cleaned tree AND on the BASE commit 45592f9ee (pre-Rainer, ladder
era) with the identical signature — a pre-existing sampling-path
(temperature > 0) completion wedge, tracked as #15, independent of this
branch's uncommitted work. All throughput benches here and historically
run greedy (temp 0), so the matrix above is unaffected.

## §17 The tp=1 gap attacked at its mechanism (#29, 2026-07-26)

User directive: pie should beat vLLM at tp=1 — fix the runtime gap
(h2h 0.933x, press 0.775x after §16.3).

**Page-ledger measurement first** (PIE_CONTENTION_TRACE_MS=500 sampler +
PIE_FIRE_TIMING waves, g1-* logs). The pool is NOT the problem: idle
(free+accum) averaged 0.5% h2h / 0.8% press. The gap is ABSENTEEISM:
- h2h: GPU idle only 5.9% (waves overlap 157%), but rows/wave p50 69 vs
  vLLM ~104. ~32 processes parked in the allocation queue at any tick;
  every page-boundary crossing parks behind the eviction-landing cadence
  (park→serve p50 6.2 ms ≈ evict chain: lease-quiesce 9.1 ms — exactly the
  2-frame run-ahead drain — + D2H commit 6.2 ms). head_pages avg 22.4: the
  head is often a whole-working-set restore, and at free=0 even the elder
  bypass fails, so every boundary ask convoys.
- press: idle 27.8% — transfer-bound. H2D wall 2.7 ms/page vs D2H 0.34.
  Swap copies were issued as one cudaMemcpyAsync per (layer, K/V, page):
  ~56 calls × ~32 KB per page, 1,300+ calls per restore — submission
  overhead, not PCIe (pinned buffers sustain ~0.07 ms/page).

**Fix #1 — lease-quiescent victim preference** (planner.rs plan_eviction +
residency::kv_lease_quiescent): among the eligible victims (younger than
head, resident, progressed — eligibility unchanged, so anti-thrash and E6
hold), prefer those with ZERO active fire leases right now; youngest-first
within each class. A quiescent victim skips the 9 ms lease drain. Press
evict quiesce p50 9.1 → 0.1 ms (g2); h2h unchanged (all candidates carry
2-frame tails there — expected).

**Fix #2 — batched swap copies** (driver swap_pool.cpp): the per-page
56-call loop replaced by ONE cudaMemcpyBatchAsync per transfer set (CUDA
12.8+ API; toolkit 13.3). d2h + h2d paths; d2d left as-is.

| run           | h2h            | press (256/256)   | roomy  |
|---------------|----------------|-------------------|--------|
| g1 (baseline) | 12,963         | 10,374            |   —    |
| g2 (#1)       | 12,895         | 10,624            |   —    |
| g3 (#1+#2)    | **13,346 / 13,332** | **11,488 / 11,447** | 17,187 |

vs vLLM (§16.3 refs): h2h 0.933 → **0.958–0.959x**; press 0.775 →
**0.809–0.812x**; roomy parity unchanged. press GPU idle 27.8 → 20.8%;
park→serve p99 215 → 112 ms. Verified: lib 340/340, integration 2/2,
roomy in-band.

**Remaining levers, ranked (measured, not guessed):**
1. Engine-side single control slot serializes ALL standalone copies —
   H2D still waits 2.0 ms/page WALL (queueing behind D2H + slot). Allow
   N standalone copies in flight (worker.rs in_flight_control → small
   set for the standalone class), THEN split the swap stream into
   d2h/h2d pair for full-duplex PCIe. Press's remaining 20.8% idle.
2. h2h rows/wave (69 vs 104): boundary asks convoy behind large restore
   heads. A BOUNDED small-ask service while a big head accumulates
   (arithmetic bound, no timers) — needs a liveness argument against
   head starvation before implementing.
3. Restore chunking: land a restore in page runs and lift the fence
   early — bigger surgery (per-working-set fence granularity).

### §17.1 Lever-1 execution: the real coupling was in the driver (g4–g6)

Reading the engine's control slot for the planned multi-slot surgery
surfaced something better: the driver's `copy_kv` completed EVERY tracked
copy on the COMPUTE stream, with a `cudaStreamWaitEvent` joining the swap
stream into it — every suspend/restore transfer stalled all later launches
behind PCIe, undoing engine-side `holds_launches = false` at the stream
level. **Fix #3**: standalone page copies (no cells) now complete on the
swap stream itself; no compute-stream join (host-side happens-before —
completion → ledger commit → later fires — covers device visibility). The
mixed cells+pages descriptor keeps the join.

Results and two instructive failures:
- g4 (#1+#2+#3): h2h 13,414 (best; transfer walls collapse: d2h 400→174 ms,
  h2d 2112→1228 ms) but press 10,721 — cheap transfers sped up the
  evict⇄restore ROTATION (evictions 280→374): the slow copies had been an
  accidental thrash brake.
- g5 (asker-last victim ordering): rejected by measurement — rotation
  count unchanged, h2h −5% (busier victims, slower quiesce). Reverted.
- g6 (final = #1+#2+#3): h2h **13,423 / 13,387 (0.963x)**, press
  **11,253 / 10,981** (spread band ~11.0–11.5 vs vLLM 14,149 ≈ 0.79x),
  roomy **17,243** (parity+, band top). Tests 340/340 + 2/2.

Day total (#29): h2h 0.933x → **0.963x**; press center ≈ +5–10% with the
known ±3% spread; roomy untouched.

**The remaining press gap is now a DESIGN question, not a tuning one:**
funding a restore head by evicting younger residents rotates membership —
each rotation ships a whole working set D2H+H2D to buy one process a seat
someone else just lost. Directions (each needs a liveness argument against
rainer.md's FCFS core, none is a cleanup-pass change):
  (a) restores funded from organic frees only (completions), with
      eviction redirected to the oldest ALLOCATION ask — kills rotation,
      but an immortal fleet could starve an evictee indefinitely;
  (b) restore chunking (partial fence lift);
  (c) accept 0.79x press as the FCFS-fairness price at 2x oversub.
h2h's remaining 4% is the rows/wave occupancy (69 vs vLLM ~104) — the
bounded small-ask-service-under-a-large-head idea, same design tier.

## §18 The 33-case pie-vs-vLLM contention matrix (#30, 2026-07-26)

User directive: benchmark pie's contention management against vLLM across
many workload shapes and contention levels (>= 30 cases), and profile
anything that loses badly or hangs.

Environment: one RTX 4090, Qwen3-0.6B, tp=1, greedy (temperature 0),
`ignore_eos`, prefix caching OFF on both engines, pie @ `61a678bf1`
(= origin/dev, Rainer v1 + §17 fixes), vLLM 0.22.0 / torch 2.11.0+cu130.
Raw records: session `files/run1` (33 cases), `run2` (repeat sample),
`trace/`, `adm-P1092/`.

### §18.1 Methodology fix: the budgets are now token-for-token identical

Every prior comparison sized the two engines by *calibrating* a memory
fraction (pie util 0.26 vs vLLM 0.217, "0.9% apart"). That slack is the
same order as several of the effects being measured. Both engines are now
driven by an exact block count instead:

| engine | knob | result |
|---|---|---|
| pie  | driver option `total_pages = P` (16 tok/page) | `kv_tokens = 16P` |
| vLLM | `num_gpu_blocks_override = P`, `block_size = 16` | `GPU KV cache size: 16P tokens` |

Verified at P=2151: both report **34,416 tokens**. Harness knobs added:
`benches/pie_bench.py --total-pages/--swap-pool-size` (§10 trap 2/3 — the
shipped harness could not express a KV cap at all) and
`benches/vllm_bench.py --num-gpu-blocks-override/--block-size`.

Oversubscription **X = concurrency x (prompt + max_tokens) / 16P**.

### §18.2 The matrix

33 cases, one sample each (8 re-sampled in `run2`). **No hangs, no
timeouts.**

**Spread — corrected 2026-07-26 (§18.11).** The original claim here ("vLLM
<0.5%, pie <0.4% at parity cells") was an artefact of `run1` and `run2`
landing in the same regime. Controlled re-measurement found the roomy
cells are **bimodal by ~10% on the shipped tree**: A1 15.3-17.2K, A2
15.5-16.8K, on BOTH the baseline and the modified build. Treat any
single-sample per-case delta below ~10% on A1/A2 as noise.

| id | X | conc | nreq | out | pie tok/s | vLLM tok/s | ratio |
|---|---|---|---|---|---|---|---|
| A1 | 0.5x | 128 | 256 | 512 | 17,193 | 17,194 | **1.000x** |
| A2 | 1.0x | 128 | 256 | 512 | 16,822 | 17,042 | 0.987x |
| A3 | 1.5x | 128 | 256 | 512 | 13,041 | 15,000 | 0.869x |
| A4 | 2.0x | 128 | 256 | 512 | 11,618 | 14,257 | 0.815x |
| A5 | 3.0x | 128 | 256 | 512 | 9,509 | 12,462 | 0.763x |
| A6 | 4.0x | 128 | 256 | 512 | 8,486 | 11,220 | 0.756x |
| A7 | 8.0x | 128 | 256 | 512 | 5,962 | 7,815 | 0.763x |
| B1 | 0.25x | 16 | 256 | 512 | 5,255 | 5,170 | 1.017x |
| B2 | 0.5x | 32 | 256 | 512 | 9,058 | 9,135 | 0.992x |
| B3 | 1.0x | 64 | 256 | 512 | 12,352 | 13,001 | 0.950x |
| B4 | 1.5x | 96 | 256 | 512 | 11,803 | 14,126 | 0.836x |
| B5 | 3.0x | 192 | 256 | 512 | 11,436 | 14,783 | 0.774x |
| B6 | 4.0x | 256 | 256 | 512 | 11,732 | 14,471 | 0.811x |
| C1 | 2.0x | 64 | 128 | 128 | 10,123 | 13,405 | 0.755x |
| C2 | 2.0x | 64 | 128 | 1024 | 6,719 | 7,808 | 0.860x |
| C3 | 2.0x | 64 | 128 | 32 | 2,230 | 2,429 | 0.918x |
| C4 | 2.0x | 64 | 128 | 512 | 3,257 | 4,240 | 0.768x |
| C5 | 2.0x | 64 | 128 | 512 | 8,556 | 8,910 | 0.960x |
| C6 | 2.0x | 64 | 128 | 128 | 4,646 | 5,193 | 0.895x |
| D1 | 4.0x | 64 | 128 | 128 | 6,210 | 8,005 | 0.776x |
| D2 | 4.0x | 64 | 128 | 1024 | 4,733 | 6,101 | 0.776x |
| D3 | 4.0x | 64 | 128 | 32 | 1,892 | 2,008 | 0.942x |
| D4 | 4.0x | 64 | 128 | 512 | 2,762 | 3,475 | 0.795x |
| D5 | 4.0x | 64 | 128 | 512 | 5,422 | 6,362 | 0.852x |
| D6 | 4.0x | 64 | 128 | 128 | 3,277 | 3,731 | 0.878x |
| E1 | 4.0x | 256 | 256 | 512 | 11,209 | 14,426 | 0.777x |
| E2 | 2.0x | 128 | 512 | 512 | 10,946 | 14,706 | 0.744x |
| E3 | 2.0x | 64 | 1024 | 512 | 7,902 | 11,209 | **0.705x** |
| E4 | 2.0x | 256 | 512 | 512 | 12,178 | 16,944 | 0.719x |
| F1 | idle | 1 | - | 512 | 483 | 491 | 0.984x |
| F2 | 13.7x | 16 | 32 | 512 | 654 | 796 | 0.821x (+ **31/32**) |
| F3 | 2.0x | 32 | 64 | 2048 | 3,605 | 4,061 | 0.888x |
| F4 | 2.0x | 128 | 128 | 512 | 13,578 | 14,042 | **0.967x** |

C/D shapes: 1 short/short, 2 short/long, 3 long-prompt/short-gen,
4 long/long, 5 mixed-phase, 6 prefix-heavy. 17 of 33 cells below 0.85x,
8 at parity, zero wins.

### §18.3 The gap tracks CHURN, not pressure

The A-sweep reads like a pressure curve, but the controlled experiment says
otherwise. Same budget (P=2184), same offered concurrency (128), same
shape — **only the number of requests varies**, i.e. how many times the
fleet turns over:

| nreq | refills | tok/s | ratio | evictions | evict/req | pages moved | pages/req |
|---|---|---|---|---|---|---|---|
| 128 (F4) | 0 | 13,578 | **0.967x** | 66 | 0.52 | 2,678 | 20.9 |
| 256 (A4) | 1 | 11,618 | 0.815x | 399 | 1.56 | 12,495 | 48.8 |
| 512 (E2) | 3 | 10,946 | 0.744x | 776 | 1.52 | 25,392 | 49.6 |

(throughput from `run1`; counters from matched `PIE_CONTENTION_TRACE_MS=500`
re-runs, which cost ~8% themselves.)

**A single oversubscribed wave is nearly free — 0.967x at 2x oversub.** The
price is paid by *arrivals into a full pool*: one refill triples evictions
per request and more than doubles pages moved per request. E3 (1024 req,
the most turnover in the matrix) is the worst cell at 0.705x.

### §18.4 What the ledger shows: a 70-page round trip to deliver 1 page

E3 traced, final sample (67 s run, 1,024 requests):

```
queue=31 head_pages=1 head_kind=allocation accum=4 free=0/1092 host_free=3744/4096
parks=19651 serves=18744 evictions=1608 restores=1588 evict_rollbacks=0
d2h_pages=26668 h2d_pages=26316 d2h_ms=3462 h2d_ms=5906 resident=56 evicted=20
```

- `free=0/1092` in **every** sample: the pool is pinned at zero.
- `head_pages=1 head_kind=allocation` dominates: the head is a decode
  crossing a page boundary and needing **one** page.
- To supply it the planner evicts a whole working set (26,668 D2H pages /
  1,608 evictions = **16.6 pages per eviction**), and that victim then
  re-queues for a full restore (1,588 restores). **~33 pages of PCIe
  traffic move to hand one page to the head** (52,984 pages / 1,608
  evict-restore cycles = 32.9; a victim evicted at full length costs
  ~70, but the measured average victim is mid-flight).
- Total 52,984 pages moved against 35,840 pages of real demand
  (1,024 x 35) = **1.48 pages copied per page of KV actually wanted**, and
  9.4 s of copy time inside a 67 s run.
- `evict_rollbacks=0`, `restore_failures=0`, `hogs=0`: the machinery is
  working exactly as designed. This is the design's cost, not a defect.

vLLM does **zero** of this. Its scheduler holds the surplus at the door:
A6 logs `Running: 48 reqs, Waiting: 34 reqs, GPU KV cache usage: 98.8%`,
A7 `Running: 24, Waiting: 107, usage: 100.0%`, with no preemption lines at
all. It never commits KV it cannot sustain, so it never has to move any.

### §18.5 The mechanism: contention collapses pie's run-ahead overlap

Pie's throughput is `rows_per_batch / wall_per_batch`. Comparing the
harness's own batch accounting against wall time across the A-sweep
(`total batches`, `total requests`, `avg batch latency us`, `wall`):

| case | X | rows/batch | batch latency | wall/batch | overlap |
|---|---|---|---|---|---|
| A1 | 0.5x | 128.0 | 13.78 ms | 7.44 ms | **185%** |
| A2 | 1.0x | 123.3 | 13.14 ms | 7.33 ms | 179% |
| A3 | 1.5x | 92.6 | 8.65 ms | 7.10 ms | 122% |
| A4 | 2.0x | 85.1 | 6.95 ms | 7.32 ms | 95% |
| A5 | 3.0x | 61.2 | 5.61 ms | 6.44 ms | 87% |
| A6 | 4.0x | 48.8 | 4.75 ms | 5.75 ms | **83%** |
| A7 | 8.0x | 24.9 | 3.72 ms | 4.17 ms | 89% |

Roomy, pie keeps ~1.85 batches in flight (the Venus run-ahead) and ties
vLLM to four digits. Under contention the overlap falls below 1.0: the GPU
waits between batches. Every page-boundary crossing parks its lane, and the
lane cannot submit its next fire until an eviction has physically landed —
so the eviction latency, not the kernel, sets the cadence.

Head to head at P=1092 the two engines run *the same batch width* and pie
is still 37% slower per step:

| | rows/step | wall/step | tok/s |
|---|---|---|---|
| pie (conc 128) | 49.0 | 6.10 ms | 8,043 |
| vLLM (self-limited to 50 running) | 50 | 4.45 ms | 11,226 |

### §18.6 It is NOT over-admission — pie's own optimum is still 0.78x

The obvious hypothesis is that pie admits 128 where vLLM admits 48, so
capping admission would close the gap. Measured, it does not. Admission
sweep at P=1092 (resident capacity ~31 requests at full length), vLLM given
the full 128 offered load for reference:

| pie conc | 16 | 24 | 31 | 40 | **48** | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|
| tok/s | 5,133 | 7,190 | 8,243 | 8,586 | **8,770** | 8,456 | 8,042 | 8,043 |

vLLM, same budget, full offered load: **11,226**.

Pie's best achievable throughput at this budget, over every admission
level, is 8,770 = **0.78x** of what vLLM gets. Admission tuning is worth
+9% (8,043 -> 8,770); the remaining 22% is the residency mechanism itself.
This kills tuning as a fix and confirms §17.1's reading: the cost is the
evict/restore **rotation**, and rotation is what FCFS-by-spawn requires
whenever a younger process must fund an older head.

Corollary for the shape cells: the losses are worst where working sets are
large relative to the budget and turn over fast (C1/D1 short-short 0.755x /
0.776x, C4/D4 long/long 0.768x / 0.795x) and mildest where each request
holds its pages only briefly (C3/D3 long-prompt/short-gen 0.918x / 0.942x)
or where half the fleet is short-lived (C5 mixed-phase 0.960x).

### §18.7 F2 — a request is failed loud, and the message named the wrong cause

F2 is the extreme edge: P=40 pages (640 tokens) against a 546-token
request, concurrency 16 (X = 13.65x). vLLM completes 32/32. **Pie drops a
request in 4 of 7 runs** — intermittent, race-dependent:

```
decode frame submit: pipeline: KV capacity: KV pool starved under host swap
exhaustion: 1 pages asked, 0 free of 40, no swap room to evict into, ...
```

The message is wrong. The traced failure shows `host_free=4069/4096` — the
host swap pool was **99.3% empty**. The real state at the kill:

```
queue=2 head_pages=17 head_kind=restore accum=9 free=0/40 resident=1 evicted=2
```

One resident holds the pool; the head is a 17-page restore. `plan_eviction`
builds its candidate set from `proc.seq > head_seq && Resident &&
progressed` — the FCFS anti-thrash rule, which forbids evicting anyone
**older** than the head. The sole page-holder is older, so `picks` is empty,
and the fall-through calls `check_starvation`, which fails the youngest
parked allocation loud. `PlannerError::Starved`'s `Display` hardcoded
"under host swap exhaustion" for *both* of `check_starvation`'s callers,
so it misattributed a policy wedge to a resource exhaustion.

**Fixed** (`runtime/engine/src/planner.rs`): a `StarveCause`
(`NoSwapRoom` | `NoEligibleVictim`) is threaded from each caller into the
error and the `tracing::warn!`. The same failure now reports:

```
KV pool starved: 1 pages asked, 0 free of 40, no evictable victim — every
page is held by a process OLDER than the head, which FCFS anti-thrash
forbids evicting, and no fire in flight anywhere to complete and free pages
```

Diagnostics only — no policy change. Lib tests 343/343.

The underlying behaviour was a **real gap**: when the budget is near one
working set, pie's ordering reached a state with no legal move and
destroyed a request, where vLLM simply serializes and completes all of
them. Root-caused and fixed in §18.9.

### §18.8 Standing summary

- **Parity is real and reproducible** at or below capacity (A1 1.000x,
  A2 0.987x, B1 1.017x, B2 0.992x, F1 0.984x) and for a single
  oversubscribed wave (F4 0.967x). Pie's kernels and batching are not the
  problem.
- **The contended deficit is 0.70-0.82x**, driven by fleet turnover, not by
  pressure level; it plateaus at ~0.76x beyond 3x oversub.
- **The cause is rotation economics**, quantified: 1.48 pages of PCIe
  traffic per page of demand, ~33 pages moved per 1-page head, and a
  run-ahead overlap that falls 185% -> 83%.
- **Not fixable by admission control** (best case 0.78x) — it is the design
  question §17.1 already framed, now with a measured ceiling on the tuning
  branch.
- The three directions from §17.1 (organic-frees-only restores; restore
  chunking; accept the FCFS price) are unchanged.
- **F2's request destruction is fixed** (§18.9); the throughput gap is not.


## §18.9 F2 root cause: E6 hysteresis in the liveness path (#31, 2026-07-26)

§18.7 recorded the symptom (a request destroyed in 4 of 7 F2 runs) and
corrected the message. This is the mechanism and the fix.

### The wedge, captured

Instrumenting the no-pick fall-through and the kill decision
(`PIE_CONTENTION_TRACE_MS`, both trace-gated) caught the state exactly:

```
NOPICK head_seq=32 deficit=1 procs: older_or_head=1 nonresident=0
       not_progressed=1 eligible_but_quoted_zero=0

WEDGE-KILL cause=NoEligibleVictim
  queue=[31:alloc:need=1, 32:restore:need=12, 33:alloc:need=1]
  procs=[32:Evicted:prog=true, 31:Resident:prog=true, 33:Resident:prog=false]
```

The head asks for **one page**. The only resident younger than the head
(seq 33) holds the pool, and is excluded solely by E6 — `progressed=false`.

**Why the veto never lifts.** `progressed` is cleared when a restore lands
and is set only inside `acquire` (`note_progress` /
`note_ask_and_check_elder`). A process that is restored and then parks on
an unmet ask never reaches another `acquire` — it is waiting for pages —
so its veto is permanent. Meanwhile it is the only thing that could fund
the head. Circular, and `check_starvation` reads the circle as "no
completion can ever arrive" and destroys the youngest ask.

E6 is documented as **hysteresis** ("one line of membership hysteresis: a
restored member stays a member until one fire completes", rainer.md §14),
while rainer.md §15 requires "no timers, no knobs, no heuristics **in any
liveness path**". E6 sitting where it can force a destruction violates
that criterion on the design's own terms.

### The fix, and the wrong version of it first

**Wrong version (recorded because the failure mode is instructive):** relax
E6 whenever the normal candidate set yields no picks. `picks` comes up
empty *routinely* — 15,139 times in one 30 s F2 run — so relaxation fired
593 times, and phase 3's own `!proc.progressed` re-validation then
silently rejected every relaxed pick. Result: a **livelock**, throughput
654 -> 49 tok/s and 29/32 completed. Worse than the bug.

**Landed version:** E6 relaxation is the last rung *before* destruction,
gated on the wedge predicate itself:

- the wedge test is extracted to `ResidencyPlanner::is_wedged()` so the
  relaxation rung and `check_starvation` cannot disagree on what "wedged"
  means;
- `plan_eviction` collects eligible and E6-vetoed candidates separately;
  on an empty pick it runs `check_hog`, and only if `is_wedged()` holds
  does it re-quote the E6-vetoed set;
- an `e6_relaxed` flag threads to phase 3 so the re-validation accepts the
  relaxed pick (this is what the wrong version missed);
- counter `e6_relaxations`, surfaced in the sampler line as `e6_relax=`.

Anti-thrash is untouched: victims are still strictly younger than the head
and still Resident. Only E6's hysteresis yields, and only in the state that
would otherwise destroy a request.

### Verification

| | F2 completed 32/32 | tok/s |
|---|---|---|
| before | **3 of 7 runs** | 654 |
| naive relaxation | 1 of 2 (livelock) | 49 |
| landed fix | **25 of 26 runs** | 563-665 (unchanged) |

`e6_relax` fires 0-2 times per F2 run. Under normal contention it fires
**zero** times: a traced E3 run (1,024 req, 1,679 evictions, 1,663
restores) reports `e6_relax=0 starved=0`. The rung is inert outside the
endgame, which is why throughput is unmoved:

| case | pre s1 | pre s2 | post-fix | |
|---|---|---|---|---|
| A1 | 17,193 | 17,220 | 17,161 | in band |
| A4 | 11,618 | 10,867 | 11,534 | in band |
| A6 | 8,486 | 7,689 | 8,145 | in band |
| B3 | 12,352 | 12,368 | 12,727 | in band |
| C1 | 10,123 | 10,440 | 10,352 | in band |
| E2 | 10,946 | 10,605 | 10,592 | in band |
| E3 | 7,902 | 7,877 | 7,592 / 7,840 / 7,737 | in band (4% spread) |
| F4 | 13,578 | 13,539 | 13,489 | in band |

Lib tests 343/343 and the full `--tests` suite green. One integration
test had to move with the message: `contention_host_full`'s
`host_swap_exhaustion_kills_a_victim_without_wedging_the_fleet` asserted
on the literal `"host swap exhaustion"`. That case sets `cpu_pages = 0`,
so it is genuinely `StarveCause::NoSwapRoom`; it now asserts on
`"no host swap room to evict into"`. (It is worth noting that a `--lib`
run does NOT catch this — the message assertions live in `--test`
targets.) `planner.rs` rustfmt diffs 10 -> 7 (all remaining pre-existing).

### Residual: a second, rarer wedge class

One of the 26 runs still failed, in a **different** state — both processes
`prog=true`, so E6 was not involved:

```
WEDGE-KILL cause=NoEligibleVictim
  queue=[32:alloc:need=1, 33:alloc:need=1]
  procs=[33:Resident:prog=true, 32:Resident:prog=true]
```

Root-caused in §18.10 — and the guess in this paragraph was wrong: the
victim's quote was not `Nothing` at all.


## §18.10 The residual 1-in-26: a stale candidate snapshot (#32, 2026-07-26)

§18.9 left one failure in 26. Two more repros were captured with the
per-process `ReclaimQuote` dump, and they were **two different states** —
neither matching the §18.9 guess that the victim was unreclaimable:

**Repro A** (E6-vetoed victim, healthy quote):
```
procs  = [33:Resident prog=FALSE, 32:Resident prog=true]
quotes = [33:Pages(19), 32:Pages(21)]
head   = 32:alloc need=1
```
**Repro B** (ordinary victim, healthy quote, E6 not involved at all):
```
head   = 19:alloc need=1
19: Resident prog=true  quote=Pages(27)   (the head itself)
20: Resident prog=true  quote=Pages(13)   <- eligible by every rule
21,22,23: Evicted (AllSwapped); 24-33: Resident, HoldsNothing
```
In both, a victim younger than the head sat there holding double-digit
reclaimable pages while the planner destroyed a request for want of ONE.

### Root cause: the kill is decided on a snapshot nobody re-read

`plan_eviction` samples the candidate set in phase 1, releases the lock to
quote (quoting takes store locks), and only much later reaches
`check_starvation`. A process that was `Evicting` or `Restoring` at sample
time is in **neither** the eligible nor the E6-vetoed list — and
`is_wedged()` is false while anything is in transit, so the §18.9
relaxation rung, gated on that same predicate, correctly declined to fire.
Then the transfer lands, the process becomes `Resident` holding its pages,
`is_wedged()` flips true, and `check_starvation` kills — using a candidate
scan taken before that process existed as a candidate.

§18.9 fixed only the E6 half of this (repro A) and only from the stale
snapshot; repro B shows the same gap swallowing an *ordinary* candidate,
which E6 relaxation could never have reached.

### Fix

`ResidencyPlanner::last_resort_evict()` — invoked from `check_starvation`
immediately before the kill, re-reading the process table **at that
instant**:

1. ordinary candidates (E6 honoured) — no hysteresis is spent if a
   normally-eligible victim exists;
2. E6-vetoed candidates — hysteresis yields rather than let a request die.

Eligibility is otherwise unchanged (strictly younger than the head, still
Resident), so FCFS anti-thrash — the real safety property — is untouched.
Phase 3 was extracted to `commit_evictions()` so the mark/yield/spawn step
is shared by both entry points.

**Gated on `cause == NoEligibleVictim`.** Under `NoSwapRoom` there is
nowhere to evict *into*: a pick would be marked `Evicting`, fail in the
executor, roll back, and repeat forever. Running the rung unconditionally
made `contention_host_full` spin until its 10 s timeout — which is what
earns the `StarveCause` distinction added in §18.7 its keep.

### Verification

| build | F2 completed 32/32 |
|---|---|
| pre-§18.9 | 3 of 7 |
| §18.9 | 25 of 26 |
| §18.10 | **45 of 45** |

- `last-resort-evict` fires **0** times under normal contention (traced E3,
  1,024 req, ~1,200 evictions): the rung is inert outside the endgame.
- Throughput, 3 samples per case, post-fix vs the three pre-fix samples:
  A1 17,192-17,237 (pre 17,161-17,220), A4 11,178-11,439 (10,867-11,618),
  A6 8,146-8,545 (7,689-8,486), B3 12,404-12,724 (12,352-12,727),
  C1 10,227-10,615 (10,123-10,440), E2 10,509-10,573 (10,592-10,946),
  F4 13,536-13,583 (13,489-13,578), E3 7,598-7,786 (7,592-7,902). In band
  everywhere.
- Full `--tests` suite green (343 lib + 43 integration).

**One observed hang was NOT this change.** A traced E3 run wedged with the
seal watchdog (`frame k=1 lanes=64 awaited=64 sealed=0 pending_binds=2`),
planner uninvolved (`starved=0 e6_relax=0`, last-resort 0). Controlled
A/B — engine changes stashed, harness kept — gives **0 hangs in 5 traced
E3 runs on the baseline and 0 in 5 on the fixed build**. It is the
pre-existing intermittent seal-path wedge (§1b, #15 class), not a
regression; it is not reproducible at this rate and remains open.

### What is still open

- The throughput gap (§18.3-§18.6) is untouched by any of this.
- The seal-watchdog wedge above.
- The starvation rung itself remains reachable in principle: if no victim
  younger than the head holds reclaimable pages at the kill instant, a
  request still dies. No such state has been observed since the fix, but
  nothing proves it unreachable — the FCFS "never evict your senior" rule
  is what leaves the possibility open, and only the §17.1 (a) redesign
  removes it structurally.

## §19 Design consequences → `rainer_v3.md` (2026-07-26)

The defects in §18.7–§18.10 were not four unrelated bugs. Three of the
four (and the §1 livelock and §12 deadlock before them) are one class:
**a decision taken on state that was already stale when it was acted on.**
`last_resort_evict()` (§18.10) is a hand-rolled emulation of the one
property that fixes the class — re-read a consistent snapshot at the
instant of an irreversible decision — which is exactly what `rainer.md`
v2's single-writer boundary pass provides structurally.

That changes v2's standing. `rainer.md` §10 parked it as "a robustness
refactor ... not a rewrite justified by throughput". The throughput half
of that verdict still holds (§18.4 says the contended gap is rotation
economics, which v2 reduces but does not remove). The robustness half was
weighed against no evidence; there is evidence now — three destroyed
requests and one self-inflicted livelock, in one session, all in that seam.

`rainer_v3.md` records the full argument plus the three defects v2 does
NOT address:

1. **Liveness and policy share one funnel, enforced by nothing.** §15's
   "no heuristics in any liveness path" is prose; E6 sat there until it
   destroyed requests. Proposal: make it a type — an exact `VictimSet`
   from the boundary pass that only endgame predicates may consume, and
   an `Ordering` over it that heuristics may reorder but never empty.
2. **`ReclaimQuote` conflates a durable fact with a transient one.**
   `pages()` returns 0 for every `Nothing` variant, and `check_hog` reads
   it as "pages held" — so a head holding 39 pages reads as holding 0 the
   moment one in-flight pin overlaps, the hog predicate never fires, and
   the starvation rung destroys the innocent youngest instead. Code-evident,
   **not yet reproduced at runtime**. It is the same collapse `NoReclaim`'s
   own doc comment warns about, one level down: the *reason* was split into
   variants, the *quantity* is still overloaded.
3. **The eviction unit is not the ask unit.** §18.4 measured it: a 1-page
   head funded by a whole-working-set evict + restore, ~33 pages moved per
   page delivered (32.9 measured; ~70 for a full-length victim).
   §17.1(b) is promoted from a lever to the central structural change,
   because that is where the gap physically is.

Suggested order (each independently landable): the `ReclaimQuote` accessor
split first (small, local, removes a latent request-destroying bug), then
the typed liveness boundary, then v2's boundary pass, then divisible
residency — the only item that moves 0.70–0.82x.

## §18.11 Two benchmarking traps found while gating the v3 work (2026-07-26)

Both were caught by A/B-ing the engine changes against the stashed tree
(engine files stashed, harness kept), and both had already produced a false
alarm before being identified.

**Trap 1 — preceding GPU load.** A full 33-case matrix run (`run3`) started
after ~40 minutes of continuous benchmarking reported pie's roomy A1 at
15,636 vs `run1`'s 17,193 (-9.1%), while vLLM in the same run was
unchanged. Rested re-measurement of the SAME build gave 17,145-17,175 over
five runs. The matrix is sensitive to what ran before it; per-case deltas
across differently-conditioned runs are not comparable. Compare only
runs taken from a comparable state, or A/B back to back.

**Trap 2 — engine-level regime bistability, ~10%.** The roomy cells are
bimodal, and it is NOT a property of this branch:

| case | baseline (stashed) | with §3.1-§3.3 |
|---|---|---|
| A1 | 17,124 / 17,192 / 17,124 / 17,168 / 17,205 | 17,145-17,175 (5) |
| A2 | 16,750 / 16,637 / **15,482** | 15,263 / 16,658 |
| C5 | 7,424 / 7,423 / 7,549 | **7,680 / 7,706** |

A2 straddles ~15.5K and ~16.8K on both builds. C5's apparent -11.7% in
`run3` inverts under controlled comparison — the modified build is 2-3%
FASTER than baseline, and `run1`'s 8,556 was the outlier.

This is the engine-level instance of what `rainer.md` L3 calls performance
resting "on a stable accident" and what its §15 promises v2 would remove
("regime bistability is eliminated, not won"). It had been characterised
for inferlets (A10, synthid) but not, as far as this record goes, for
engine roomy throughput. **Consequence for every future comparison: a
single-sample roomy delta under ~10% carries no information.**

Verdict for the v3 work: **no regression.** Controlled 5-vs-5 on A1 and
3-vs-3 on A2/C5 put the modified build at or above baseline everywhere.

## §18.12 Standing state after the v3 work (2026-07-26)

Landed, each independently gated:

| item | effect | evidence |
|---|---|---|
| §18.7 `StarveCause` | the starvation error names its real cause | reproduced live |
| §18.9 E6 out of the liveness path | F2 3/7 → 25/26 | + a rejected naive version that livelocked (654→49 tok/s) |
| §18.10 kill decided at the kill instant | F2 → 45/45 | two repros, two different wedge classes |
| v3 §3.3 `held_pages` / `ReclaimQuote::pages()` deleted | a latent hog under-count removed, misuse now unrepresentable | unit test on the pin divergence |
| v3 §3.2 `VictimSet` | E6 is a tag, never a filter | — |
| v3 §3.1 step 1: atomic endgame snapshot | the stale-snapshot seam closed for the endgame | F2 20/20 |
| v3 §8.5: elder bypass deleted (−23) | one ordering rule gone, free | A4/A6/E3 in band, F2 12/12 |

Final verification, all at once: full suite **344 lib + 43 integration**,
**F2 15/15**, throughput **in band on 8 cases** against a 7-sample band,
`planner.rs` rustfmt diffs unchanged from baseline (7).

Rejected after measurement — recorded so nobody re-does them:
- **v3 §3.4** divisible residency / declared intent (user decision; sign
  uncertain, g4 counter-evidence, large mechanism).
- **§8.4** funding-batch: +35 lines, zero throughput change on
  A4/A6/E3/E2 — the bottleneck is eviction latency, not port round trips.
- The naive E6 relaxation (§18.9) and the asker-last ordering (§17.1 g5).

Not attempted: v2's boundary pass proper. §8.5 records why the sequencing
is worse than it looks — membership push alone deletes nothing, and the
117 deletable lines arrive only with the full funding model, indivisibly.

The contended **0.70-0.82x vs vLLM stands untouched** and is accepted cost
(§17.1(c)). Nothing in this session's work was aimed at it, and §18.6
showed the tuning branch tops out at 0.78x.

## §20 Extreme stress test vs vLLM 0.26.0 — three liveness defects (2026-07-27)

Scope: hunt every **deadlock / livelock / performance regression** corner
case in the residency planner and the frame scheduler, with vLLM 0.26.0 as
the control. L40S 46GB sm_89, CUDA 13.0 toolkit + compat driver,
Qwen3-0.6B, tp=1, greedy, `ignore_eos`. Parity per §18.1: pie
`--total-pages P` ⟷ vLLM `--num-gpu-blocks-override P --block-size 16`,
both exactly 16P tokens; prefix caching off on both.

Method: (a) 14 adversarial single-engine liveness probes aimed at the
planner endgame, (b) a 35-cell pie-vs-vLLM parity matrix over 5 workload
shapes × oversubscription X ∈ {0.5 … 32}.

**Four** defects, all liveness, all independent, all reachable with
**default** configuration: an admission-cap deadlock (§20.1), a starvation
kill on a fundable pool (§20.2), a frame-seal/bind deadlock (§20.3), and a
host-swap eviction livelock (§20.6).

### §20.1 Defect 1 — the admission-cap wedge (permanent deadlock)

Four probes (`noswap_4x`, `noswap_16x`, `tinyswap`, `allshared_noswap`)
hung for the full 150 s deadline emitting one frozen trace 300 times:

    queue=64 head_pages=1 accum=0 free=0/128 host_free=0/0
    parks=64 serves=0 evictions=0 starved=0 e6_relax=0
    resident=128 evicting=0 evicted=0 restoring=0

Zero serves, zero evictions, zero starvation kills. The queue head wanted
**one** page out of a 128-page pool and never got it.

**Root cause.** `Planner::is_wedged()` required *every* `Residency::Resident`
process to be parked before `check_starvation` would arm. But
`planner.register()` runs at `spawn()` — registration order *is* the FCFS
clock — while a process only touches pooled KV after **execution
admission** (`ensure_execution_admitted`). So whenever
`num_requests > max_concurrent_processes`:

* the admitted cohort (64) parks on an exhausted pool;
* the unadmitted remainder (64) sits in `procs` as `Resident` holding
  **exactly zero pages**, waiting for a permit only a completion can free;
* `is_wedged()` reads those 64 as "someone is still running" and returns
  false forever, disarming the only rung that can break the cycle.

A textbook circular wait: the parked hold the permits the unadmitted need,
and the unadmitted disarm the rung that would free the parked. Note
`swap_pool_size` defaults to 0, so this is reachable out of the box.

**Fix.** `Proc::admitted`, set by `Planner::note_admitted()` called from
`ensure_execution_admitted`. `is_wedged()` no longer counts an unadmitted
process as relief, and `victim_set()` no longer quotes one. Soundness: the
only `planner.acquire` call site is `pipeline/fire.rs`, reachable only
downstream of `ensure_execution_admitted`, so an unadmitted process
provably holds zero device pages. New `admitted=` field in the trace.

**Test.** `runtime/engine/tests/contention_admission_wedge.rs` — 5 lanes,
admission cap 2, `cpu_pages = 0`. Hangs on the pre-fix tree, passes on the
fixed one. Live: 150 s hang → 3.7-4.7 s with real progress, on all four
probes.

### §20.2 Defect 2 — the starvation rung kills on a fundable pool

The 35-cell matrix ran clean (zero deadlocks) except two cells that each
destroyed exactly one request. `PlannerError::Starved` printed its own
contradiction:

    4 pages asked, 35 free of 69
    42 pages asked, 49 free of 97

**Root cause.** A drain only reaches `plan_eviction` when
`reserve_device_up_to` comes back empty, but *everything after that* —
`is_wedged`, the exhaustive `last_resort_evict` scan under the global KV
lock, the trace dump — is a wide window in which a retiring process
returns its pages. Worse, the window is *biased*: a process leaves `procs`
at `unregister` **before** its page leases drop, so the wedge predicate
goes true strictly **before** the pool refills. The kill site
re-validated `accum` but never re-checked the pool.

**Fix.** `salvage_free_pages()` — re-run the drain's own
`reserve_device_up_to` primitive and absorb the result into `accum`, then
poke. Called twice in `check_starvation`: once before `last_resort_evict`,
once immediately before the kill. Deliberately a *reservation attempt*,
not an `available_pages() > 0` counter comparison — the counter version can
spin forever when pages are visible but unreservable, i.e. it trades a
false kill for a livelock. New `salvages_total` diagnostic.

**Test.** `mod starvation_race_tests` in `planner.rs` with a `RacePool`
mock `PoolPort` that refills exactly inside the window; red before, green
after. Live: probe `s64` (`--total-pages 8`, 512 requests, conc 64)
→ `starved=0 salvaged=1 512/512`. Both matrix cells 3/3 at 256/256, and D
X=32 throughput 867 → 888 tok/s.

### §20.3 Defect 3 — the rebind/frame-seal deadlock (`bind → dispatch → seal → bind`)

The `churn_extreme` cell collapsed 18x. First it had to be *cleared* of
being a planner-scaling problem: the same shape with a 900 s client
deadline recovers fully (238 → 3789 tok/s), so the engine is sound and the
collapse is not throughput. It is an intermittent **permanent scheduler
deadlock**, ~10-15% of runs (1 in 6-10).

Repro: `--total-pages 96 --swap-pool-size 16384 --num-requests 4096
--concurrency 1024 --max-tokens 16 --max-model-len 1536`. Timing-sensitive:
`PIE_FIRE_TIMING=1` makes it vanish (8/8 clean), so it must be chased with
`PIE_CONTENTION_TRACE_MS` alone.

Frozen planner trace (new `runners` dump, added for this):

    queue=1000 head_pages=4 accum=0 free=0/96 host_free=16384/16384
    starved=0 salvaged=0 resident=1817 admitted=1024
    runners=[2280:h4:ptrue, ...×24]

24 consecutive spawn seqs each holding exactly 4 pages — the entire
96-page pool, and every one marked `progressed`. Frame dump at the same
instant (new `[frame-stall]` trace):

    frame k=1 lanes=24 awaited=24 sealed=0 pending_binds=8 staged=793
    joins_in_flight=0 departing=0 pending_slots=0 ever_sealed=true

`joins_in_flight = 0` and `pending_slots = 0` ⇒ `joining` is false, so the
seal was held **purely by `missing > 0`**. Exactly 8 lanes had
`queued_frames=0 front_complete=false`, and **every one of their owners
appears in `pending_bind_pids`** — perfect correlation across three
reproductions (5-and-5, then 8-and-8, then 8-and-8).

**Root cause.** A rebinding process's lane stays `awaited` with zero
frames, so `missing > 0` and the boundary holds. But that process cannot
submit its next fire until its bind commits, and a bind commits through
the driver lane's single **control slot — downstream of the very dispatch
the boundary is holding**. Cycle closed: `bind → dispatch → seal → bind`.
The frame quorum's wait is infinite by design ("membership changes only
through close/leave/first-fire events"), so nothing ever breaks it. 19
lanes had complete frames ready, nothing was executing, and 8 binds stayed
pending for 30+ s.

`on_bind_enqueued`'s own doc-comment describes this exact hazard for the
*staged* path and dismisses it with "a live rebinder is already wait-set-held
through its lane". **That sentence was the bug** — being wait-set-held is
precisely what deadlocks it.

**Fix (first cut, superseded — see below).** In `plan_dispatch`'s `missing`
predicate, exclude a lane that is **empty** *and* whose owner has a bind in
flight. Narrowed to empty lanes on purpose: a lane with queued frames is
genuinely mid-submission and must still be waited for, so no submitted work
is ever dropped from an epoch. Rejoin is implicit — the lane's next accepted
fire restores it to the quorum. Chosen over mutating membership inside
`on_bind_enqueued` because the declarative predicate covers **both** event
orderings (bind-then-drain and drain-then-bind) and cannot accidentally
re-`staged` a live process.

**That first cut closed the deadlock but cost up to 16% throughput.** It
applied the exclusion on *every* gather, so a healthy boundary stopped
waiting for lanes that were about to rejoin and sealed a thinner epoch.
Caught by the full-matrix rerun (§20.5): pie's roomy decode-heavy cells lost
5-16% while vLLM — the control, same machine, same session — reproduced to
within 0.3%, which ruled out drift.

**Fix (shipped): gate the escape on the deadlock shape itself.** The
predicate now counts `missing` densely (pre-fix semantics) and separately
counts `missing_rebind`, the subset that is empty-with-a-bind-in-flight. The
rebinders are released only when **all** of:

* `missing == missing_rebind` — every remaining member is an empty rebinder,
* `!executing` — nothing is in flight, so no retirement will free the
  driver-lane control slot the binds need (while an epoch executes the cycle
  cannot close, and the pre-existing `return Park` already covers it),
* `!joining`, and
* the stall has persisted `REBIND_ESCAPE_US` (2 ms).

A healthy gather resolves in microseconds and never reaches the grace, so it
keeps the dense wait-all path; only the deadlock shape pays, and it pays 2 ms
once. `PIE_FRAME_REBIND_ESCAPE` selects `0` (never escape — reproduces the
deadlock), `1` (the unconditional first cut), `2` (default).

**A/B, one binary, interleaved trials** (so thermal/clock drift hits every
arm equally), pie tok/s, median of 3:

| cell | mode 0 (pre-fix) | mode 1 (uncond.) | mode 2 (shipped) |
| --- | --- | --- | --- |
| D X=0.5 | 10378 | 8731 (**0.841**) | 10577 (**1.019**) |
| D X=2   | 7212  | 6787 (0.941)     | 6990 (0.969)     |
| D X=32  | 900   | —                | 900 (0.998)      |
| S X=0.5 | 15310 | 13475            | 15714 (1.026)    |
| X X=0.5 | 4380  | —                | 4466 (1.020)     |
| M X=1   | 9350  | —                | 10501 (1.123)    |

Mode 2 is neutral-to-positive against the pre-fix engine on every cell
measured. It also removed a **starvation** side effect of mode 1: the
`allshared_noswap` probe served `12/128 completed, 116 failed` under mode 1
and `128/128, 0 failed` under mode 2 — the thinner epochs were driving the
fleet into the starvation kill.

**Test.** `an_empty_lane_awaiting_its_own_rebind_does_not_hold_the_seal` in
`frame.rs`'s `mod tests`; asserts the seal proceeds without the rebinder
*and* that the exclusion is scoped to the bind (once it commits the lane
holds the boundary again). Red under mode 0, green under modes 1 and 2 —
the toggle makes the regression test self-verifying.

**Considered and deliberately not covered:** a lane with *queued* frames
whose front is incomplete (a frame declares `stamp.fires = N` and only
`k < N` have arrived) whose owner is simultaneously mid-bind. That would
close the same cycle through a non-empty lane. It is left waited-for
because (a) all three reproductions showed the stalled lanes at
`queued_frames = 0`, matching `pending_binds` exactly, (b) excluding a
partially-submitted lane would drop *already submitted* fires out of the
epoch, which is a density and ordering regression, and (c) the strict
watchdog is the existing backstop for a genuinely stuck partial frame.
Both bind sites (`FrameTruncate`, `RegisterChannelsBind` in `worker.rs`)
are whole-turn control ops, so a guest reaching them mid-frame would need
to interleave a bind between fires of one declared frame.

**Live verification.** 44 consecutive repro runs, all
`4096/4096 failed=0`, zero `[frame-stall]` traces, throughput
3700-3999 tok/s (σ ≈ 55). Against the ~10% pre-fix wedge rate that is
`0.9^44 ≈ 0.9%` — i.e. >99% confidence the cycle is closed.

### §20.4 Instrumentation kept

All three defects were only diagnosable because of trace-gated dumps added
during the hunt; all are bounded and all are kept:

* `PlannerDiagnostics.runners` — the admitted-but-unparked cohort as
  `(seq, held_pages, progressed)`, capped at `RUNNER_DUMP_CAP = 24`. This
  is what proved the pool was fully held by *runnable* processes rather
  than leaked. Computed outside the planner lock (lock order is
  `RESIDENCIES` → KV store → `planner.inner`).
* `salvages_total`, `admitted` in `[planner-trace]`.
* `[frame-stall]` — `debug_summary()` printed once per strict-watchdog
  expiry, gated on `planner::trace_enabled()`, now including
  `pending_bind_pids`. The `pending_binds` ∩ zero-frame-lanes correlation
  is the whole proof of §20.3.

### §20.4b Two more benchmarking traps (cf. §18.11)

* **Never touch a Rust source while a soak is running.** `pie_bench.py`'s
  `embedded_engine_identity()` hard-fails when the embedded `.so` is older
  than *any* file under `driver/`, `interface/`, `runtime/`,
  `sdk/python-server/src/` with a `.rs/.c/.cc/.cpp/.cu/.cuh/.h/.hpp`
  suffix. It does not care that the edit was a comment or a `rustfmt`
  reflow, and it does not care that `git stash push` / `stash pop` only
  restored the file — the mtime moved. This silently converted two 20-run
  soaks into 20 one-second failures. Markdown is exempt, so the ledger can
  be written mid-run.
* **`tests/inferlets/text-completion-bench/target` is a load-bearing
  symlink**, not build litter. It points at the shared
  `tests/inferlets/target` holding the prebuilt
  `wasm32-wasip2/release/text_completion_bench.wasm`; `PIE_BENCH_INFERLET_DIR`
  resolves through it. Deleting it as "untracked junk" makes every bench
  run die in `bench_inferlet_paths`.

### §20.4c Throughput regression gate

Three representative matrix cells re-run on the fixed tree, 2 trials each,
against the §20.5 baseline ledger:

| cell | pages/conc | baseline pie | all four fixes (2 trials) | verdict |
|---|---|---|---|---|
| S X=1.0 | 396/64 | 11062, 256/256 | 10495, 11291 (μ 10893) | in band |
| D X=32 | 69/64 | 867, **255**/256 | 882, 881 (μ 882) | in band, **256/256** |
| X X=32 | 97/64 | 556, **255**/256 | 551, 568 (μ 560) | in band, **256/256** |

Both X=32 cells now complete every request — the §20.2 salvage holding on
hardware. No cell regressed; neither the §20.3 seal exclusion nor the
§20.6 victim block costs measurable epoch density.

Final state: **349 lib + 45 integration** tests green; 14/14 probes with no
hang and no livelock; 56 cumulative clean runs of the §20.3 deadlock
repro. rustfmt diff counts at their pre-campaign baselines (`planner.rs` 7,
`planner/exec.rs` 2, `frame.rs` 0, `bootstrap.rs` 0).

Probe suite, start of campaign → end (completions / failures / wall):

| probe | before | after |
|---|---|---|
| noswap_4x | HUNG 150 s | 27ok/101f 4.5 s |
| noswap_16x | HUNG 150 s | 7ok/121f 3.9 s |
| tinyswap | HUNG 150 s | 16ok/112f 4.3 s |
| allshared_noswap | HUNG 150 s | 12ok/116f 3.9 s |
| churn_extreme | 2284ok/**1812f** 185 s | **4096ok/0f** 48 s |
| the other 9 | pass | pass, in band |

### §20.6 Defect 4 — the host-swap eviction livelock

Found by re-running the probe suite *after* the first three fixes: the
`tinyswap` probe (`--total-pages 64 --swap-pool-size 8`, 128 requests,
conc 64) went from 4 s to **152.7 s** and served 6 requests instead of 12.
The trace named it immediately:

    parks=1202800 serves=73 evictions=2 evict_rollbacks=1203865
    gate_parks=1203560 starved=110 resident=10 evicting=1 admitted=12

`evict_rollbacks` past **1.2 million** and climbing ~3.5k per interval
while `serves` and `starved` were frozen. The `planner-exec` step trace
carried **4,832,808** lines, all one pid:

    pid=20e7768e evict start: 1 working set(s)
    pid=20e7768e evict drained; quiescing leases
    pid=20e7768e evict quiesced; preparing
    pid=20e7768e evict rollback: host swap full      (×1.2M, ~7.5k/s)

**Root cause.** `KvStoreError::HostSwapFull` routed to `eviction_failed`,
which restores the victim to `Resident` and immediately `poke()`s. Victim
selection is a *deterministic* FCFS scan, so the replan re-picks the same
process and re-runs the entire fence-raise → `notify_process_suspend` →
drain → lease-quiesce → `prepare_suspend` cycle only to fail identically.
Nothing in the loop changes any input to the decision, which is the
definition of a livelock. It also churned the frame policy through
`notify_process_suspend` 7.5k times a second.

Worse, it *disarmed* the rung that would have broken the jam: a
perpetually in-flight eviction keeps `is_wedged()` false, and
`last_resort_evict` kept reporting success (it did dispatch an eviction —
which then rolled back), so `check_starvation` returned without killing
and `starved` never advanced. Same shape as §20.1: a spurious "someone is
making progress" signal.

**Fix.** `Inner::host_swap_blocked: HashSet<ProcessId>`. `HostSwapFull`
routes to a new `eviction_failed_host_swap_full`, which parks the victim
there. Both victim scans consult it:

* `victim_set()` — the routine path, and
* **`last_resort_evict`'s own inline legal-victim scan** — a *separate*
  scan built under the KV lock. Missing this one is why the first attempt
  at the fix still livelocked in 2 of 8 runs with `swapfull=1012663`: the
  block was being set and the routine path did skip it, but the endgame
  rung re-picked the blocked victim anyway. E6 hysteresis is deliberately
  waived in that scan; a host-swap block must NOT be, because it is a
  physical impossibility rather than a preference.

The poke is kept, so a *smaller* victim that still fits can be tried.
Blocks clear wholesale when host room actually returns — `report_restored`
(H2D returns slots) and `unregister` (teardown). When the block empties the
legal set, `last_resort_evict` returns false and `check_starvation`
proceeds to the kill: failing one request loudly is the designed terminal
behaviour, and it is strictly better than spinning forever.

**Test.** `a_host_swap_full_victim_is_parked_until_host_room_returns` in
`mod starvation_race_tests` — asserts the blocked victim leaves
`victim_set()` and that clearing re-arms it. Red before, green after.

**Live.** `tinyswap` 12/12 runs at **3.7-4.6 s** (was 152.7 s), 11-22
completions (was 4-6 livelocked, 12 pre-campaign). Two of the twelve
engaged the new path and it **bounded them at 28 and 13 rollbacks** rather
than 1,000,000+. `planner-exec` lines 4,832,808 → 22. New trace field
`swapfull=<exhaustions>/<unblocks>`.

**Methodological note.** This defect was only visible because the probe
suite was re-run end-to-end after the earlier fixes. Defects 1-3 each
changed which liveness signals are trusted, and that shifted `tinyswap`
from "killed fast by the starvation rung" into the eviction-retry path
where the pre-existing spin lived. Re-run the whole suite after every
liveness change, not just the probes that were failing.

### §20.5 Standing comparison and the open findings

Final 40-cell parity matrix on the shipped engine (`PIE_FRAME_REBIND_ESCAPE`
default 2). pie legs re-measured in `/root/matrix_m2`; vLLM legs reused from
`/root/matrix_fixed` — page counts reproduce exactly and vLLM's whole D
column re-measured to within 0.3%, which is what licensed the reuse.

Throughput ratio pie/vLLM, `X = concurrency * (prompt + max_tokens) / (16P)`:

| shape | X=0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| S (short)   | **1.043** | 0.931 | 0.866 | 0.815 | 0.839 | 0.816 | 0.858 | 0.807 |
| D (decode)  | 0.950 | 0.945 | 0.752 | 0.763 | 0.769 | 0.772 | 0.775 | vLLM died |
| M (mixed)   | 0.978 | 0.872 | 0.782 | 0.809 | 0.816 | 0.841 | 0.867 | vLLM died |
| X (extreme) | 0.837 | 0.821 | 0.747 | 0.813 | 0.896 | 0.934 | 0.888 | vLLM died |

`N=29 min=0.747 p50=0.837 max=1.043 mean=0.848`. Against the same cells
before §20.3's regression fix the roomy corner moved from 0.809-0.879 to
0.950-1.043; nothing moved backwards. The campaign produced exactly one
performance regression, its own (§20.3), and that one is closed.

**X=64 is where the comparison stops being a comparison.** At 64x
oversubscription vLLM was SIGKILLed at the case deadline in 4 of the 5
shapes while pie completed every one of them, shedding load through the
starvation rung instead of wedging. pie's X=64 numbers have no denominator.

#### Uncontended parity (Qwen3-0.6B, 4096 pages, planner counters all zero)

| case | pie tok/s | vLLM tok/s | ratio |
|---|---|---|---|
| latency, conc 1 | 407 | 371 | **1.100** |
| tput, conc 1    | 407 | 372 | **1.092** |
| tput, conc 8    | 2848 | 2665 | **1.069** |
| tput, conc 32   | 9619 | 9333 | **1.031** |
| tput, conc 64   | 15683 | 15390 | **1.019** |
| tput, conc 128  | 21606 | 22681 | 0.953 |
| tput, conc 256  | 22295 | 29664 | 0.752 |

pie leads on both latency and throughput up to 64 concurrent processes and
falls behind past it. Note these numbers are only meaningful because the
planner trace was OFF on both engines — see §20.4b, it costs 2.08x.

**Open finding: pie stops pipelining above 64 concurrent processes.**
Duty cycle = `total_batches * avg_batch_latency_us / measured_seconds`, i.e.
how many forward batches are in flight on average:

| conc | batches | avg batch us | duty | batch size hist |
|---|---|---|---|---|
| 32  | 2048 | 5921 | **1.77** | all 2048 at 32 |
| 64  | 1025 | 6987 | **1.69** | 1023 at 64 |
| 128 | 518  | 7376 | **1.00** | 506 at 128 |
| 256 | 516  | 12042 | **0.94** | 512 at 256 |

Batches are *full* at every point, so this is not a batching-density
problem — pie simply stops overlapping batches exactly where it starts
losing to vLLM. A batch of 128 also costs barely more than a batch of 64
(7376 vs 6987 us), so the GPU is not the limit either; recovering the 1.7x
overlap at conc 128 would roughly double throughput there.

This is **not** a contention defect and **not** a regression from this
campaign: `PIE_FRAME_REBIND_ESCAPE=0` (pre-§20.3 behaviour) shows the same
collapse (duty 0.98/0.96 at conc 128, 0.96/1.12 at 256), and the planner is
completely idle in these runs (`evictions=starved=parks=restores=0`). It
lives in the fire/quorum path, not in Rainer.

**Open finding (not a defect, a design gap): pie has no KV-aware admission
control.** pie admits by *process count*; vLLM admits by *KV block
availability* with a WAITING queue and will simply not start a request it
cannot fund. At a fixed 96 pages:

| engine | concurrency | result |
|---|---|---|
| pie | 1024 | 238 tok/s, 1852 timeouts |
| pie | 256 | 4290 tok/s, 4096/4096 |
| pie | 1024, 900 s deadline | 3789 tok/s, 4096/4096 |
| vLLM | 1024 | 5596 tok/s, 4096/4096 |

The engine is sound — the work completes — but with concurrency far above
what the pool can fund, pie spreads the pool thin and pays it entirely in
tail latency, where vLLM converts the same oversubscription into queueing.
Admission that consults free KV (rather than only
`max_concurrent_processes`) is the natural follow-up and is the single
largest remaining gap this campaign found.

### §20.7 The scenario suite moved into the tree (2026-07-27)

Everything this campaign used as a one-off probe now lives in
`tests/contention/` as a declarative table plus a runner:

    python tests/contention/run.py                  # every scenario
    python tests/contention/run.py -k tinyswap,soak # a subset
    python tests/contention/run.py --repeat 3       # override repetition
    python tests/contention/run.py --list

Each scenario names the planner endgame path it aims at and carries a
`Contract`: a wall-clock ceiling (exceeding it is a hang), completion and
failure bounds, `accounted` (completed + failed == requests, so nothing
vanishes), `require_counters` (planner counters that must be non-zero, so a
scenario cannot rot into a no-op when policy shifts underneath it),
`max_counters` (this is how §20.6's 1.2M-rollback livelock is
regression-tested) and forbidden log substrings (`[frame-stall]` is §20.3's
tell). The exit status is non-zero if anything violates its contract, so it
drops straight into the `[self-hosted, cuda]` CI job.

Every scenario drives a **real** driver. There is no mock device anywhere in
the suite, by design: all five defects lived in the interaction between the
planner, the scheduler's frame boundary and the driver's copy engine, and a
mock device reproduces none of it.

Four things the first end-to-end run taught, all now encoded:

* the runner has to put a **working** CUDA forward-compat directory on
  `LD_LIBRARY_PATH` or the driver aborts with `pie_cuda_create returned
  null`. Taking the highest-versioned `/usr/local/cuda-*/compat` is wrong on
  a box carrying a toolkit newer than the kernel driver's ceiling — 13.3's
  libcuda 610.43.02 loads happily and then fails `cuInit` against a 550
  kernel module. The runner probes `cuInit` per candidate and takes the
  first that actually initialises.
* `PIE_CONTENTION_TRACE_MS` must be well under the shortest scenario's
  runtime. The fast-fail scenarios finish in 0.9 s, so at the original
  1000 ms sampling period **not one trace line was ever emitted** and every
  `require_counters` assertion reported "never fired" — a false failure that
  reads exactly like a real regression. The default is now 200 ms, and "no
  trace line at all" is reported as its own distinct violation.
* the prefix-sharing scenarios need `warmup=1`. Without a warmup request
  nothing has published the shared prefix, so all 128 processes arrive with
  a private copy of it and the fleet starves on arrival instead of
  exercising `ReclaimQuote -> Nothing(AllShared)`.
* a contract has to be calibrated against the defect it claims to guard, not
  guessed. `tinyswap` at the original 64 pages / 8 swap pages never reached
  `HostSwapFull` at all — the starvation rung killed the fleet first and
  `swapfull` stayed 0. Re-sized to 128/16 at concurrency 16 it fires 10-58
  swap-full hits on every run while the fleet still finishes 61-63 of 64.
  Conversely `allshared_noswap` cannot gate on completion counts: measured
  four runs per mode, `PIE_FRAME_REBIND_ESCAPE=1` completes 16-24 and mode 2
  completes 31-59, distributions that touch. That scenario asserts liveness
  only and §20.3's A/B harness pins the mode difference instead.

### §20.8 `PIE_FRAME_SIZE=2` buys the high-concurrency gap back — and breaks contention liveness (2026-07-27)

§20.5's open finding was that pie stops overlapping batches above 64
concurrent processes. The frame constant k (`PIE_FRAME_SIZE`, waves per
frame, default 1, driver-supported to 4) is the direct lever: at k = 1 the
wait-all quorum runs once per token, so every lane must arrive before any
lane advances.

Uncontended, Qwen3-0.6B, 4096 pages, interleaved arms, best of two trials:

| case | pie k=1 | pie k=2 | vLLM | k1/vLLM | k2/vLLM |
|---|---|---|---|---|---|
| latency, conc 1 | 397 | 395 | 371 | 1.069 | 1.066 |
| tput, conc 1   | 409 | 399 | 372 | 1.098 | 1.072 |
| tput, conc 8   | 2853 | 2873 | 2668 | 1.069 | 1.077 |
| tput, conc 32  | 9650 | 9701 | 9399 | 1.027 | 1.032 |
| tput, conc 64  | 15642 | 15675 | 15542 | 1.006 | 1.009 |
| tput, conc 128 | 21656 | 22132 | 23199 | 0.933 | 0.954 |
| tput, conc 256 | 21670 | **27955** | 29928 | 0.724 | **0.934** |

k = 2 costs nothing anywhere and recovers most of the conc-256 gap
(0.724 -> 0.934), with mean latency there falling 2408 -> 1748 ms. The
mechanism is exactly the one §20.5 identified: duty (batches in flight)
goes from a bimodal 0.80-1.33 at k = 1 to a stable 1.59-1.60 at k = 2.
**k = 1 at conc 256 is not merely slower, it is unstable** — it drops out
of pipelining entirely on roughly half the runs. k = 4 measured the same as
k = 2, and raising `PIE_SCHED_MAX_IN_FLIGHT` 2 -> 3 changed nothing (the
run-ahead depth was never the binding constraint).

**But k = 2 is not contention-safe.** `tests/contention/run.py` under
`PIE_FRAME_SIZE=2`: **12 of 16 scenarios fail**, against 16/16 at k = 1.

| outcome | scenarios |
|---|---|
| HANG (killed at the ceiling) | `tinyswap` 3/3, `tinyswap_thrash`, `impossible`, `fwd1` |
| **SIGSEGV** (rc = -11) | `mixed_head`, `soak` |
| `[frame-stall]` + mass failure | `restore1` (9/256 completed) |
| degraded | `hog` 0/32, `onefits` 25/64, `churn` 2047/2048 in 181 s (44 s at k = 1) |
| pass | `noswap_4x`, `noswap_16x`, `allshared`, `allshared_noswap`, `churn_extreme` 4/4, `admission_tail` |

The split is perfectly explained by eviction volume. Everything that passes
does 0 or 1 evictions; everything that fails evicts continuously
(`restore1` 33, `onefits` 36, `tinyswap` 5, `soak` 47).

Root cause, read straight off the diagnostic — 11 of 12 lanes complete, one
not:

    [frame-stall] frame k=2 lanes=12 awaited=12 sealed=0 ... ever_sealed=true
      lane ...0004 awaited=true queued_frames=1 front_complete=true
      lane ...000b awaited=true queued_frames=1 front_complete=false   <- owner c96a0ef8
      ... (ten more, all front_complete=true)
    [planner-trace] ... evictions=33 gate_parks=33 evicting=1 evicted=15

and for that same owner:

    [planner-exec] pid=c96a0ef8... evict start: 1 working set(s)
    [planner-exec] pid=c96a0ef8... evict drained; quiescing leases

**The planner quiesces a lane mid-frame.** At k = 1 a fire *is* a frame, so
a park/evict boundary is a frame boundary by construction and the wait-all
gate can never see a half-arrived frame. At k > 1 an eviction that lands
between slot 0 and slot 1 leaves the lane's frame permanently
arrival-incomplete, and the infinite wait-all rule then holds the entire
epoch behind it — the same failure shape as §20.3's rebind/frame-seal
deadlock, but reached through the eviction path instead of the bind path,
and not covered by that fix's escape (which keys on empty lanes awaiting
their own rebind, whereas here the lane is non-empty and half-full).

The two SIGSEGVs are a separate, harder failure: `mixed_head` and `soak`
die with no panic message, i.e. in native driver code, which the
half-drained frame state is presumably feeding.

#### Defect 5 — a partial frame stranded by a mid-frame leave (FIXED)

Two coupled faults, both invisible at k = 1 because a 1-slot frame is
complete the instant it arrives:

**5a — an arrival racing the suspend rejoined the wait-set.**
`planner/exec.rs` broadcasts `notify_process_suspend` at evict step 3, and
the fence raised at step 2 stops new *leases*, not a fire already past it.
`record_arrival`'s `lanes.entry(...).or_insert_with(|| LaneState { awaited:
true, .. })` therefore recreated the victim's lane as an awaited member
holding a one-slot-of-two frame whose remaining slots sat behind the
eviction. Fix: `FramePolicy::suspended`, a set the planner's suspend adds
to and its runnable-again chokepoints (`report_restored`,
`eviction_failed_inner`) clear through the new
`worker::notify_process_resume` / `SchedulerItem::ProcessResume` /
`FramePolicy::on_process_resume`. While marked, arrivals are recorded
without joining the wait-set.

**5b — the stranded slot never sealed, so its lease never released.**
Un-awaiting the lane unblocks the *fleet*, but the victim's own half-frame
still could not seal, and a queued fire holds a lease. Eviction step 5
(`handle.quiesce().await`) waits on exactly those leases, so the eviction
wedged mid-flight — `evicting=1` forever, the planner head stuck on
`head_kind=allocation`, and the whole fleet starving behind it. The same
circle is reachable with no planner at all: the KV allocation-wait park
posts a lane close mid-frame, and the guest cannot finish the frame until
it is served, which needs the fleet to advance, which the frame blocks.

Fix: `LaneState`/`FramePolicy::truncate_incomplete` — every path that takes
a lane out of the wait-set mid-frame (`on_lane_leave` non-purging,
`on_process_suspend`) now cuts its unfinished frames down to what ARRIVED,
so they seal, drain and release their leases. Arrivals under a suspended
owner are self-completing for the same reason. The slots the cut discarded
may still arrive (the guest resumes inside its submit loop), so
`FramePolicy::truncated_seqs` remembers the cut seq per lane — kept outside
`lanes` because a truncated lane is normally dropped the moment its frames
drain, well before the late slot lands — and a late slot stands alone
instead of re-forming an unsatisfiable frame. `PendingFrame::truncated`
keeps `expected` from being re-raised by a later slot's `stamp.fires`.

Also fixed in `pipeline/fire.rs`: the mid-frame truncation notice was only
sent on the *logical* submit-error path; a host trap returned through `?`
and left the frame arrival-incomplete forever.

Regression tests: `a_fire_racing_the_suspend_seals_alone_without_rejoining_the_wait_set`
and `a_lane_parked_mid_frame_seals_what_it_submitted` (`scheduler/frame.rs`).

#### Post-fix: k = 2 is contention-clean, and k = 3/4 buy nothing

`tests/contention/run.py` after the fix: **16/16 at k = 1, 2, 3 and 4**
(from 12 failures at k = 2 before). `churn` returned to 43 s from 181 s.

Uncontended sweep, best of two trials:

| case | k=1 | k=2 | k=3 | k=4 | k2/k1 |
|---|---|---|---|---|---|
| lat, conc 1 | 396 | 410 | 412 | 409 | 1.036 |
| tput, conc 8 | 2835 | 2865 | 2871 | 2884 | 1.011 |
| tput, conc 32 | 9648 | 9654 | 9624 | 9657 | 1.001 |
| tput, conc 64 | 15499 | 15629 | 15486 | 15508 | 1.008 |
| tput, conc 128 | 21776 | 22042 | 22052 | 21960 | 1.012 |
| tput, conc 256 | 21767 | **28050** | 27899 | 27615 | **1.289** |

k = 3 and k = 4 are indistinguishable from k = 2 (0.98–1.00 of it) while
costing more staging depth and a coarser truncation granularity under
contention, so **k = 2 is the setting**: the whole win is the duty cycle
(conc 256: 1.12 -> 1.62; conc 128: the bimodal 1.00/1.67 collapses to a
stable 1.67), which is what §20.5's open finding predicted. Mean latency at
conc 256 falls 3746 -> 2918 ms. `PIE_FRAME_SIZE` default raised 1 -> 2.

Final parity against vLLM 0.26.0 on the shipped default, interleaved arms,
best of two trials (4096 pages both sides):

| case | pie k=1 | pie k=2 | vLLM | k1/vLLM | k2/vLLM |
|---|---|---|---|---|---|
| latency, conc 1 | 411 | 399 | 370 | 1.110 | 1.077 |
| tput, conc 1 | 409 | 401 | 372 | 1.101 | 1.079 |
| tput, conc 8 | 2823 | 2867 | 2669 | 1.058 | 1.074 |
| tput, conc 32 | 9614 | 9701 | 9609 | 1.001 | 1.010 |
| tput, conc 64 | 15663 | 15782 | 15619 | 1.003 | 1.010 |
| tput, conc 128 | 21646 | 22084 | 22820 | 0.949 | 0.968 |
| tput, conc 256 | 27110 | 28061 | 29664 | 0.914 | 0.946 |

pie is ahead of vLLM at concurrency 1–64 and within 3–5% at 128/256 (from
7%/28% behind). Note that k = 1 landed in its GOOD mode at conc 256 here
(27110) where the sweep above caught its bad mode (21767): k = 1 is
bimodal at that point and k = 2 is not — the run-to-run spread is the
reason to take the default, independent of the mean.

## §20.9 The settlement callback sat on the compute stream (2026-07-28)

### Question

With `PIE_FRAME_SIZE=2` the decode pipeline should be perfectly covered:
frame *N+1* is submitted while frame *N* runs, the sampled token carries over
**on device** through the channel, and no host round trip separates the two
forward passes. Yet uncontended conc 256 still trailed vLLM by 3–7%, and
raising *k* to 3 or 4 changed nothing. Why?

Steady-state cycle, two-point `(wall@256tok − wall@128tok)/128`, 12288 pages,
conc 256:

| engine | cycle |
| ------ | ----- |
| pie k=1 | 13.835 ms |
| pie k=2 | 13.407 ms |
| vLLM    | 12.946 ms |

k=2 is 1.036× vLLM, i.e. **+461 µs per wave** to account for.

### First diagnosis — wrong

`PIE_FIRE_TIMING=1` puts p50 host work at 1525 µs/wave, of which FrameSettle
is 859 µs and `finish_epilogue` 709 µs (execute 286 / group 279 / assemble
71). Tempting, but wrong: NVTX shows that work starting *mid* graph replay and
the submitting thread idle ~70% of the wall. It is not GPU-visible. Rejecting
it is what forced the real answer out.

### Evidence chain

Against a conc-256 k=2 nsys capture (`--cuda-graph-trace`, KERNEL ∪ MEMCPY ∪
MEMSET ∪ GRAPH_TRACE merged — omitting any of those tables manufactures fake
idle):

1. The wave is 8205 µs: 7476 µs graph replay (91.1%) + 382 µs non-graph
   device + **347 µs idle**. The single largest idle window sits between
   `k_settle_host_channels_batch` and the next wave's
   `k_pull_validate_host_channels_batch`, p50 224 µs.
2. Not starvation. In 284 of 290 steady gaps the next wave's
   `k_pull_validate` had *already been submitted*, p50 8673 µs **before** the
   GPU went idle.
3. Genuinely idle, not misattributed: p50 busy inside the window is 8.9 µs of
   224.2 µs.
4. Not a cross-stream wait. `STREAM_WAIT_EVENT` totals 1.1 µs over the whole
   window set, and the last op on any non-main stream ends p50 2360 µs before
   the window closes.
5. **Zero host CUDA API calls occur during the window.** Whatever holds the
   stream makes no CUDA calls — a driver-internal thread.
6. Scaling control at conc 32 vs 256: 124.8 µs vs 224.2 µs, i.e.
   **gap = 111 µs + 0.44 µs × lanes**.

### Root cause

`dispatch.cu` enqueued the settlement notification on the **compute stream**:

```
:4942  cudaStream_t callback_stream = stream;
:5088  cudaStream_t settlement_stream = callback_stream;   // batch_copies == false here
:5161  cudaLaunchHostFunc(settlement_stream, notify_runtime_callback, notify);
```

A host-function node holds its stream until the CUDA driver's callback thread
wakes on the CPU and returns. Everything queued behind it — including the
already-submitted next wave — cannot start. The 111 µs constant is the
callback-thread wakeup; the 0.44 µs/lane term is `notify_runtime_callback`
releasing each instance's `callback_fence`.

This is precisely why *k* never helped: **k pre-queues wave N+1 behind the
blocking node**, and a blocking node cannot be jumped by pre-queueing. k=2,
3 and 4 all measured the same because they were all waiting on the same CPU
thread.

### Fix

Move the host function to a dedicated `notify_stream`:

* record `settlement_ready` on the settlement stream after
  `launch_settle_host_channels_batch`, have `notify_stream` wait on it, and
  launch `cudaLaunchHostFunc` there;
* record both `notify->callback_done` and a new
  `settlement_callbacks_done` on `notify_stream`.

One hazard has to be handled. `commit_snapshot()` pools pinned host buffers
per **(BoundInstance, occurrence-within-wave)** and reuses them every wave, so
wave *N+1*'s D2H publications write the very buffers wave *N*'s callback
reads. The on-compute-stream host func was providing that barrier by accident.
It is restored explicitly with a `cudaStreamWaitEvent` on
`settlement_callbacks_done` placed immediately **before** this wave's
`enqueue_host_publish_copies` — a full forward pass downstream of the record,
so it never blocks in the covered case, and no cycle is possible because the
awaited record is always from a strictly earlier wave (`Dispatch::finish` is
serialized by `settlement_mutex`). `NotifyContextLease` now drains
`notify_stream` on the exception path as well.

### Result (conc 256 unless noted, 4–8 trials each)

| case | before | after | Δ |
| ---- | ------ | ----- | - |
| conc 64, k=1 | 15634 | 16243 | +3.9% |
| conc 64, k=2 | 15758 | 16145 | +2.5% |
| conc 256, k=2 | 27866 | **28497** | +2.3% |
| conc 256, k=1 fast mode | 26981 | **28656** | +6.2% |
| steady cycle vs vLLM | 1.0356× | **1.0154×** | gap halved |

k=2 at conc 256 is now 8/8 trials inside 1% (28371–28656) against a vLLM
reference of 29621–29688, i.e. 0.962× on throughput where it was 0.937×.
Contention suite: 16/16.

## §20.10 Why k=1 at conc 256 is multimodal (2026-07-28)

k=1 lands anywhere in 18.8-29.0 k tok/s across process launches while k=2 does
not. Ruled out first, all measured, none of them the cause:

* **GPU clocks** — SM clock pinned at 2520 MHz in every mode, no throttle
  reason asserted, temp <= 44 C.
* **NUMA** — the GPU is local to node 1, but `taskset` onto node 1 *or* node 0
  is equally multimodal.
* **Host CPU frequency** — schedutil governor, but max / 8th / median core MHz
  are identical (3250 / 2846 / 1500) in fast and slow runs.
* **External contention** — GPU empty (1 MiB), load ~11 on 128 cores.
* **Engine decisions** — memory planner output, 51 captured decode graphs,
  batch count, batch-size histogram, `max forward requests: 256`, prompt and
  output token counts are all identical across modes.

### It is duty, and duty is quantised

Duty = mean forward batches in flight = `avg batch latency / (wall / batches)`.
Measured (conc 256, 12288-page-equivalent budget, 8 trials per arm):

| arm | duty | tok/s |
| --- | ---- | ----- |
| k=2 (default), 8/8 | 1.56-1.60 | 28371-28656 |
| k=1, fast | 1.54-1.58 | 26.8-29.0 k |
| k=1, mid | 1.09-1.11 | 21.4-22.2 k |
| k=1, slow | 0.82-0.85 | 18.7-18.8 k |
| k=1 with `PIE_SCHED_MAX_IN_FLIGHT=1` | 0.84-0.85 | 19.8-20.2 k |

Two things follow immediately.

**k does not create the overlap.** A healthy k=1 run reaches duty 1.58 — the
same as k=2. Frame overlap is structural and k-independent: `FramePolicy`
seals the next frame the moment the wait-all gate holds, normally while the
current frame is still executing, and posts its waves behind the executing
frame's tail at the run-ahead depth. There is no launch-time barrier; the
driver's device-side `pass_commit` tickets order dependent fires by stream
order and a frame-boundary dependency is structurally identical to an
intra-frame one. What k changes is how *often* the wait-all seal boundary
occurs (once per k waves) and therefore how much GPU time is available to
cover one host turn.

**The slow mode is the run-ahead collapsing, not extra work.** Forcing
`PIE_SCHED_MAX_IN_FLIGHT=1` reproduces the slowest mode exactly and, unlike
the default, *stably* (sigma = 220 tok/s). Depth 3 does not help: the cap is
not the limiter, arrival into the seal window is.

### The mode is per COHORT, not per run

512 requests at concurrency 256 is two cohorts of 256. Splitting each run in
half (`cohort1 = 2*lat_mean - wall`):

```
k=1 fixed  28586 tok/s   cohort1 30.4k   cohort2 27.0k     fast, fast
k=1 fixed  22225 tok/s   cohort1 29.5k   cohort2 17.8k     fast, slow
k=1 fixed  22156 tok/s   cohort1 18.2k   cohort2 28.2k     slow, fast
k=1 fixed  18779 tok/s   cohort1 18.2k   cohort2 19.4k     slow, slow
k=2 fixed  (8 runs)      every cohort 28.0-29.0k
```

Each 256-process cohort independently settles into duty ~1.58 or ~0.83 when it
forms, holds it for all 128 of its decode waves, and the run-level modes are
just the two combinations: 28.5 k (fast+fast), ~22 k (mixed), 18.8 k
(slow+slow). Matching duty levels 1.58 / 1.10 / 0.83 confirm the arithmetic.
k=2 never collapses in any of its 16 observed cohorts.

### Where the stall is

nsys on a k=1 run, merged device-op gaps above 200 us by bracketing operation:

```
258  786 ms  avg 3046 us   k_settle_host_channels_batch -> MEMCPY
149  461 ms  avg 3097 us   MEMCPY -> MEMCPY
143   53 ms  avg  371 us   MEMCPY -> embed_bf16_kernel
```

258 is exactly the batch count: the stall is at the seal boundary, between one
frame's settlement and the next frame's H2D upload. The idle *immediately*
before a wave's `k_pull_validate` is p50 1.4 us, so nothing is waiting on
batch construction once the frame has been sealed — the wait is for the seal.

The seal is wait-for-ALL: it holds until every awaited lane's oldest queued
frame is arrival-complete, so the binding term is the **slowest of 256** lanes'
resubmit each frame. At k=1 that maximum is paid once per wave and has one
wave of GPU time (~7.1 ms) to hide behind; at k=2 it is paid half as often and
has two waves (~14.9 ms). The absolute cost of the turn scales with lane count,
not with k — which is why k=2 is comfortably covered and k=1 sits on the edge.

Two levers confirm the turn is the term that varies:

* **Any added host cost pins k=1 slow.** `PIE_FIRE_TIMING=1` (65 664 per-fire
  JSON records) parks it at 14.9-15.8 k, 4/4; nsys parks it at 17.0-21.7 k,
  4/4. Neither perturbs k=2 comparably.
* **Tokio worker count moves it.** conc 256, k=1, 4 trials each:
  `--worker-threads 8` -> 27100/27084/26664/23177; `32` ->
  18987/18677/19015/19079 (sigma = 179); 16 and the default 64 are multimodal.
  (`default_worker_threads()` in `worker/src/config.rs` already caps at 64 for
  exactly this class of variance on EPYC.)

### Status

**Open**: what decides a cohort's mode at formation is not yet identified. It
is not any of the environmental or planner variables above, it is stable for
the cohort's entire life, and it is independent between consecutive cohorts in
one process.

**Not on the default path.** k=2 has been the default since §20.8 precisely
because it puts the seal boundary behind two waves of cover: on the fixed
build conc 256 k=2 is 28371-28656 over 8 trials, a 1.0% spread, with no
collapsed cohort observed. The §20.9 fix raises k=1's fast mode
(26981 -> 28656, +6.2%) and leaves the collapsed mode where it was (baseline
18.9-21.7 k, fixed 18.8-22.2 k) — a strict improvement that does not remove
the bistability, because the residual stall is the seal turn rather than the
callback node. If k=1 must be used at high concurrency, `--worker-threads 8`
is the lever.
