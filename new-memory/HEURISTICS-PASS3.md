# Heuristic numbers, pass 3

Tree read: origin/memory-redesign 451233bfd (paths under `crates/`). Two things are read on
another commit and say so: the wgpu rings on frames-laid c23a6a821, and commit history.
Read-only: nothing was built, run, or checked out. No replacement number is proposed anywhere.
Model figures are arithmetic from the catalog packages under `models/`, not from a run of the
chunk plan; where the plan's own pick would be needed, the text says so.

Contents: 1 Paces. 2 Sizes that decide memory or device objects. 3 KEEP_CACHE to PLEX surface.
Appendix: batching numbers with no memory in them. Not verified.

---------------------------------------------------------------------------------------------

## 1. Paces

The governor counts in ticks of `TICK` = 250 ms (`runtime/src/governor.rs:475`), so every
number below is a duration only through the poll that the ruling already replaces with its
event. Today's values in time: PATIENCE 0.5 s, FIRST_HOLD 1 s, LONGEST_HOLD 64 s, SETTLED 16 s,
IDLE_AFTER 30 s, KEEP_CACHE 10 min.

### 1.1 The five governor paces

| Pace | Site | Goes under | Why |
|---|---|---|---|
| PATIENCE (governor use) | `governor.rs:132`, read `:381` | `governor_backoff` | It is the "stays congested" in the key's own sentence |
| FIRST_HOLD, LONGEST_HOLD | `governor.rs:252-253` | `governor_backoff` | They are how long a yield lasts before the process takes memory back |
| SETTLED | `governor.rs:257` | `governor_backoff` | It is when a yield is forgotten |
| IDLE_AFTER | `governor.rs:261` | `governor_release_idle` | It is when the release the key turns on begins |

None of the four goes to PLEX. KEEP_CACHE alone does (section 3).

**PATIENCE, governor use.**
- Who decides today: the runtime, as a constant. `Governor::tick` backs the target off by
  `tolerance.backoff` once `congested_ticks >= PATIENCE` (`governor.rs:378-386`); Critical
  skips it and goes to the floor at once (`:374-377`).
- What reads it: only that arm, and only in a process that yields. A process with
  `backoff = 1.0` reads every tick as calm (`:364-369`), so the constant is dead there.
- What the key would have to state: `governor_backoff` is documented as "The share of its
  memory this process keeps each time the system around it stays congested"
  (`worker/src/config.rs:935-938`). "Stays congested" is exactly what PATIENCE defines and the
  key does not say. The key has to state the stretch that counts as staying congested, or the
  pressure reading has to arrive already as a stretch (a system pressure event carries its own
  window where the threshold is stated), in which case the backoff acts once per event and
  there is nothing left to count.
- What disappears from the runtime: the constant and `congested_ticks`.
- Not PLEX: PLEX has no reading of the system around the process. A worker policy reads tier
  bytes of this worker only (`runtime/src/policy/service.rs:1007-1037`).
- The rung walk's use of the same constant (`runtime/src/store/shared/target.rs:140`) is the
  lost-wake cover of pass 2 row K4b and goes with its event, not here.

**FIRST_HOLD, LONGEST_HOLD.**
- Who decides today: the runtime. After a backoff the hold is quadrupled and clamped between
  the two (`governor.rs:395-402`); while `held < hold` the target does not rise (`:425-429`).
- What reads them: `backed_off`, `calm`, and `can_rise` (`:434-436`). `can_rise` is the
  wider consequence: `asked` serves a waiter by raising the target only when it holds
  (`:441-447`). While the process is yielding or holding, a waiter is not served by a rise and
  ends as tier pressure, so these two numbers also decide how long a waiter's shortfall is
  answered by eviction instead of by growth.
- Live only when the process yields: `backed_off` is reached only from the Congested and
  Critical arms, which a process with `backoff = 1.0` never enters.
- What the key would have to state: after yielding, how long the process waits before it
  takes memory back, and how that wait grows when the probe is pushed back again. The
  quadrupling and the halving (`:397`, `:407`) are part of the same statement and are bare
  factors today.
- What disappears: both constants and the two factors; `hold` stays as state.

**SETTLED.**
- Who decides today: the runtime. A hold halves and `yielding` ends after
  `SETTLED x (hold / FIRST_HOLD)` calm ticks (`governor.rs:404-410`).
- What reads it: `calm` only; through `yielding` it also gates `can_rise`.
- What the key would have to state: how long a calm has to last before a past push-back
  stops counting. It is the same schedule as the holds and has no meaning apart from them.
- What disappears: the constant.

**IDLE_AFTER.**
- Who decides today: the runtime. `idle()` is `idle_ticks >= IDLE_AFTER` (`governor.rs:353-355`);
  a cut happens only when `tolerance.release_idle && idle()` (`:419-423`).
- What reads it: that gate, and `Steer` reports the transition as `Turn.idled`
  (`runtime/src/governor/run.rs:141-144`). No reader of `Turn.idled` or `Turn.yielding` was
  found on 451233bfd: the only caller of `steer.turn` in serving discards the returned `Turn`
  (`governor/run.rs:166-168`, `:185`).
- What the key would have to state: `governor_release_idle` is a bool documented as "Gives
  back, a little at a time, the memory demand stopped asking for" (`config.rs:939-941`). It
  has to state after how long without rising live demand the release begins. "A little at a
  time" is CUT_EVERY and CUT_SHARE (`governor.rs:264-265`, pass 2 row R4a) and belongs to the
  same statement.
- What disappears: the constant; `idle_ticks` stays only as the counter the key's statement
  is checked against.
- Not PLEX: what a release-idle cut takes first is room nothing holds (the target stands
  above `wanted`, free chunks). No working set and no cache entry is involved, so a retention
  policy has no subject to key. That is the difference from KEEP_CACHE, which decides when
  cached kv stops counting as wanted, and that is retention.
- One consequence to carry into section 3: once kept = live + cache keyed above the floor,
  IDLE_AFTER is also the lifetime of cache at the floor in a process with release-idle on.
  Today that lifetime is KEEP_CACHE.

### 1.2 ENGINE_LANE_HOT_US (stays; owner: runtime)

Site: `runtime/src/scheduler/lane.rs:333-336`, 1 s native, 0 on wasm32; the lane spins on its
queue for that long after the last request, then parks.

Where the number came from: commit bdc52ec75 (2026-07-19, "Make generated
pivot_threshold(cummass_le) run in parallel", no body). Before it the spin window was
`quorum::wave_window()` (not a derived value: see 1.2a). The commit replaced that with
`DRIVER_LANE_HOT_US = 1_000_000` and the comment that the lane "parks only after true
idleness (between requests / at shutdown). This is a lane wake optimization, independent of
the fire policy." Renamed in 8d8f449f0 (2026-08-28); the wasm32 zero came in 9cb4d315b
(2026-09-20, web support).

Measurement beside it: none in the repository. No figure in those three commit messages, and
nothing about the lane spin in `new-memory/*.md`.

What it trades: one core busy for up to the window after the last request, against the wake
latency of a parked lane thread on the next request.

What measurement would set it: the distribution of the gap between one request's end and the
next post on a serving lane (per backend, at token cadence and between turns), beside the
park-to-run wake latency of the lane thread on the same machine, and the core time the spin
costs at each candidate window. The number is set where a longer window stops removing wakes.

### 1.2a What `wave_window()` was, and whether it can be restored

Read with `git show` / `git log` only. Paths before the commit are under `runtime/engine/src/`.

**Correction first.** Section 1.2 called `quorum::wave_window()` a value the scheduler
derives. It was not derived. It returned a constant with an environment override and a floor.
"Derived" is the old lane comment's own word ("the scheduler's own derived wave-window
constant", `scheduler/worker.rs:868-869` at bdc52ec75^). So there is no derived form to
restore; restoring it as it was restores a tuned number.

**Inputs and formula** (`scheduler/quorum.rs` at bdc52ec75^):
- `WAVE_WINDOW_US = 6_000` (`:63`), replaced by `PIE_SCHED_WAVE_WINDOW_US` when set and
  parseable (`:74-78`, `:89`).
- Floor `cold_hold()`: `COLD_HOLD_US = 2_000` (`:38`), replaced by `PIE_SCHED_COLD_HOLD_US`
  when set (`:40-51`).
- `wave_window() = max(requested, cold_hold())`, read once per process (`:86-102`); a request
  below the floor is clamped with a warning.
- Nothing measured at run time enters it. The one run-time estimate nearby, an EWMA of the
  wave retire-to-retire cadence, was added to the window for straggler demotion only
  (`:183-190`); the lane spin did not read it.

**Values recorded.**
- The constant's own measurement (`quorum.rs:54-63`): dense-wave cadence about 3 ms on the
  reference RTX 4090 decode workload; at 2 ms, 915 waves instead of about 520 and 35% less
  throughput; dense throughout at 6 ms; regressing again by 12 ms. This measures the
  scheduler's barrier, not the lane spin.
- The test `wave_window_floors_at_cold_hold` (`quorum.rs:915-937`): unset or unparseable
  gives 6000 us; anything under 2000 us gives the 2000 us floor; 12000 us passes through.
- The lane's own record (`worker.rs:860-869`): a parked lane "measurably broke run-ahead
  pipelining (81 % to ~30 % of transitions enqueued ahead on the parked variant)", and "at
  wave cadence the spin window always covers the next post". This supports spinning against
  parking, and gives the condition a window has to meet (cover the next post). It gives no
  figure for a window length. That comment is not in the tree at 3e4e78300.

**Callers at bdc52ec75^.**
- `scheduler/quorum.rs:671`: `WaitAllPolicy`, straggler demotion. A gathering wave held the
  barrier for the window, then fired narrow without the absentees.
- `scheduler/worker.rs:870` and `:900`: the lane's spin before parking.
- Tests only: `quorum.rs:1152`, `:1187`, `:1203`, `:1304`, `:1319`, `:1378`, `:1425`, `:1448`.

**Why it was dropped.** bdc52ec75 deleted the function's module, not only the lane's use.
- The commit: 284 files, +38607 / -14201, empty body, a direct commit on the line (parent
  6aa2d550e, child 0aaa9ba2a which is PR #487); no PR number or merge of its own.
- It deletes `scheduler/quorum.rs` (1790 lines) and adds `scheduler/frame.rs` (1466 lines).
  Its note says so: "every deployment ... runs `FramePolicy` (scheduler/quorum.rs deleted)"
  (`vesuvius-phase2.md:240-243` at bdc52ec75), under the rule "quorum infinite wait-all
  (watchdog 1 s, report-only)" (`vesuvius-phase1-report.md:13`).
- So straggler demotion went, and the wave window with it. The frame policy waits without
  bound; its one clock is `STRICT_WATCHDOG_US = 1_000_000`, report-only (`scheduler/frame.rs:132`
  at bdc52ec75). The lane spin lost the function it borrowed and got a constant in its place.
- The new constant has the watchdog's value, one second. No link is stated; the comment says
  the spin is "independent of the fire policy". From 6 ms to 1 s is a factor of 167.
- No figure for the one second: not in the commit, not in the notes it adds (no mention of
  the lane's spin or parking in any `*.md` at bdc52ec75), not in the bodies around it
  (6aa2d550e empty; e2201dadb and 0aaa9ba2a are CUDA model work).
- `git log -S wave_window` finds two code commits only: 6aa2d550e (2026-07-19, 2190 files,
  which brings the scheduler in) and bdc52ec75 (same day). The window's own history (the
  runs its comments cite) is not in this repository.
- `git log -S LANE_HOT_US`: bdc52ec75 introduces it; later hits are moves and the wasm32 zero
  (9cb4d315b). No commit changes the native value.

**At 3e4e78300: do the inputs still exist?** None does.

| Input at bdc52ec75^ | At 3e4e78300 | What stands in its place |
|---|---|---|
| `WAVE_WINDOW_US`, `wave_window()` | gone | nothing: the seal waits without bound |
| `PIE_SCHED_WAVE_WINDOW_US` | gone (no `PIE_SCHED_` variable is read) | nothing |
| `COLD_HOLD_US`, `cold_hold()` | gone (last in the tree at 0e2ee887a^, 2026-08-02) | nothing |
| cadence EWMA | gone | no cadence is estimated anywhere in the scheduler |
| `scheduler/quorum.rs` | gone | `crates/runtime/src/scheduler/frame.rs`; its one clock is `STRICT_WATCHDOG_US` (`:57`) |

The derived form cannot be restored as it was, for two reasons: the function and the policy
it belonged to are gone, and what it returned was a constant tuned to a 3 ms cadence on one
card. What the scheduler knows today that a derived window could read: it stamps each retire
and records the idle gap to the next dispatch as a statistic
(`crates/runtime/src/scheduler/frame.rs:1074-1083`, `stats.rs:38`), and it declares an
inter-fire time (`scheduler/probe.rs:5`, exported at `server/handler.rs:104-105`) for which no
writer was found. The condition the old comment states, that the window covers the next
post, is a statement about that gap. Nothing in the lane reads it today.

**GAP_BUDGET (9d90f9d51): no derived predecessor.** Confirmed. The commit (parent 244401d23,
9 files, +203 / -11, no body, a direct commit) adds the constant and the `Instant::now() >=
until` test to a loop that existed. Before it, a gap ran a waiting raise's steps with no time
bound, until the raise was armed or a step did not step; the removed comment gives the
reason ("every gap a step is left out costs it far more fires than the step costs the
others"). So the predecessor was unbounded, not a derived value and not another number. The
two older `git log -S GAP_BUDGET` hits (8d8f449f0, fb5325c5a) are an unrelated test constant,
`GAP_BUDGET: f32 = 0.25` in `crates/engine-cuda/tests/two_frames_in_flight.rs:94`, added and
then deleted. One thing beside it is derived: 1fbedd22d (2026-11-13) makes a busy lane owe
each step's time back to its requests, so arming takes at most half the lane's time
(`crates/runtime/src/scheduler/arming.rs:259-262`, `:344-345`, `:371` at 3e4e78300). The
20 ms stands on top of that share and is unchanged at 3e4e78300 (`arming.rs:206`).

Not verified for 1.2a: the runs the old comments cite (they are outside this repository);
which commit removed the lane's measurement comment; whether the idle-gap statistic is
recorded in every build or only with the probes on; whether the 1 s was chosen to match the
watchdog.

### 1.3 GAP_BUDGET (stays; owner: runtime)

Site: `runtime/src/scheduler/arming.rs:206`, 20 ms; used at `runtime/src/scheduler/lane.rs:452-466`:
in a gap between two requests a waiting raise runs steps while the budget lasts.

Where the number came from: commit 9d90f9d51 (2026-11-13, "runtime: no state chunk opens in
the span the engine holds back, and a waiting raise steps within a gap's budget while its
pass fires at the rungs armed", no body). Its doc at introduction: "the requests behind it
wait no longer than this for steps of a few milliseconds, and one step for longer ones."

Measurement beside it: none in the repository. "Steps of a few milliseconds" is stated, not
recorded; no bench figure in the commit or in `new-memory/*.md`.

What it trades: the wait added to a request that arrives while a gap's steps run (at most
the budget, or one step if a step is longer), against how many gaps a raise takes to finish,
during which passes fire at a narrower rung.

What measurement would set it: per backend, the duration of each raise step (so "a few
milliseconds" is a recorded distribution), the queue wait of requests that arrive inside a
gap, and the number of gaps and the wall time a raise takes to complete at each candidate
budget, with the token rate while it is incomplete.

### 1.4 PIPELINE_CACHE_SETTLE (owner: runtime, Vulkan engine)

Site: `engine-vulkan/src/device/ctx.rs:450`, 2 s; `settle()` writes the pipeline cache once no
build has happened for that long (`:494-504`). It came with commit 02b7fbbf7 (2026-10-11,
"vulkan: one pipeline cache for kernels and guest stages, written soon after a build (#896)").
The commit carries a measurement, but of writing the cache early at all, not of the two
seconds: "On a Galaxy S26, the first answer of a fresh process with top_p, top_k and a
presence penalty comes in 0.21 s, 0.92 s for 23 tokens, after the last process was killed
with SIGKILL (was 1.84 s and 4.05 s)". No figure for the value itself was found. It trades
rewrites of the cache file during a burst of builds against how long a finished build is
exposed to loss when the process is killed. What would set it: on the same class of device,
the gaps between consecutive builds inside one burst (a load, a sampler's first fire) and the
time one write takes; the value sits just past the longest gap inside a burst.

---------------------------------------------------------------------------------------------

## 2. Sizes that decide memory or device objects

Terms. Arms: frames in flight, the configured runahead depth (`engine/src/runahead.rs:2-4`;
`Runahead::F1` is the Vulkan tune default, `engine-vulkan/src/tune.rs:99`; at most
`MAX_FRAMES`, `runahead.rs:16`). Rungs: the fire rungs a load fits. Classes: the structural
classes present in a rung. Dispatches: the dispatches of one recording. The binding form is
the one agreed: a binding is a whole pool buffer; a tensor is an offset in a per-fire params
uniform; Vulkan records once per (arm, rung, structural class); wgpu bind groups are
(pipeline, distinct buffers), a fixed set made at load; params sit in a per-arm params buffer
at a fixed offset per dispatch.

Figures used from the background: the arming note measures rungs 1, 6, 8 and dispatches
54, 91, 43 on qwen3-micro-text, qwen35-tiny, gemma4-e4b-mini-l1, and infers 7, 90, 120
recordings per arm when every class mix is its own recording
(the arming session's recordings note, not in the repo). Those inferred counts (7, 90, 120)
were withdrawn after this pass was written, once the mask classes were resolved from the
programs; the form ruled since is the superset one, a single recording per (arm, rung) made at
load, under which no cap on kept shapes has anything to evict. Where 90 and 120 appear below
they are the withdrawn figures and carry no argument.
The probe reads 759 built SPIR-V kernels: the largest push block is 92 bytes, a kernel has at
most 12 buffers of its own, and with one u32 offset per own buffer the largest params block
is 116 bytes, 144 with an offset and an extent (computed from
the pool probe's push-block listing, not in the repo).

### 2.1 Summary

| Constant | Bounds | From laid things | Binding form |
|---|---|---|---|
| SHAPES_KEPT | recordings kept | arms x rungs x classes | removes it |
| SETS_PER_BATCH (and SETS_PER_POOL, BUFFERS_PER_POOL) | descriptor sets taken per call, per pool | one set per (layout, distinct buffers) | removes all three |
| SPARE_FENCES | idle fences kept | one fence per live frame | removes the cap for recordings; guest frames remain |
| HAZARD_CAP | spans tracked before a forced barrier | dispatches x bindings per dispatch | removes the reason for it |
| widen SLOTS | deferred widen configs alive | arms x deferred passes per step | removes it |
| BIND_CACHE_CAP | wgpu bind groups cached | pipelines x distinct buffers | removes it |
| UNIFORM_RING | uniform bytes of kept groups | arms x dispatches x aligned block | replaces it by that expression |
| SCRATCH_RING | bytes one frame stages | staged extents of the widest laid frame | removes the view half; copies remain |
| PRICED_FROM, PRICED_MOVES, PRICED_FROM_MOST | pipelines built to price the rest | none | removes the need only if every class is built at load |
| NODE_COPIES | indexed copies of a node start | none | not touched |

### 2.2 Each one

**SHAPES_KEPT** (`engine-vulkan/src/record.rs:74`, 64).
- Bounds: recordings in the recorder; past it the least recently used is dropped
  (`record.rs:143-158`).
- From laid things: arms x rungs x classes present. With one superset recording per rung it
  is arms x rungs, which is 1, 6 and 8 per arm on the three miniatures.
- Binding form: removes it outright. The key becomes (arm, rung, structural class), each
  recorded once, so the count is laid, and the eviction, the use clock and the constant go.
- Where the derivation is thin: today's key also hashes the descriptor, the readout bytes,
  the slot tables, caches, bindings, windows and each lane's reads
  (`engine-vulkan/src/serve.rs:372-397`). Each of those must become data in the params or part
  of the structural class. The note's count with every class mix recorded apart, 90 and 120
  per arm, is already past 64, so under that reading today's cap would evict live recordings.
  Recordings are also forgotten whole when segments or expert stages change
  (`serve.rs:155-163`); that stays an event, not a size.

**SETS_PER_BATCH, SETS_PER_POOL, BUFFERS_PER_POOL** (`engine-vulkan/src/device/ctx.rs:1283`,
`:1281`, `:1284`; 64, 1024, 8192).
- Bounds: a frame takes 64 descriptor sets of a layout at a time and keeps the unused ones
  for later dispatches of that layout (`ctx.rs:1398-1460`); a pool holds 1024 sets and 8192
  buffer descriptors. Up to 63 sets per layout per frame are allocated and never used.
- From laid things: a set is needed per (layout, distinct buffers), not per dispatch.
- Binding form: removes all three outright. A set names whole buffers and never an offset,
  so the sets are a fixed set written at load: the layouts built times the distinct buffer
  tuples the laid frames name, with one more factor of arms if the params buffer is per arm
  and bound without a dynamic offset. The pool is then sized exactly from that set.
- Where it does not hold: guest stage frames encoded live keep taking sets per frame until
  their bindings are a load-time set too.

**SPARE_FENCES** (`engine-vulkan/src/device/ctx.rs:157`, 32).
- Bounds: fences kept for reuse when a frame drops; past it a returned fence is destroyed
  (`ctx.rs:327-342`). A frame takes one at creation (`:1111`, `lend_fence` `:309-325`) and
  returns it at drop (`:1277`). It does not bound live fences.
- From laid things: one fence per live frame. Live frames are the kept recordings plus the
  one-time frames in flight.
- Binding form: recordings are a laid set that is never evicted, so their fences are taken
  once and returned only when everything is forgotten. The spare list then holds exactly the
  fences of frames not alive, and no cap is needed: its size is bounded by the peak number of
  live frames, arms x rungs x classes plus the one-time frames.
- Where no derivation holds: the one-time frames. A deferred guest pass is a frame of its own
  per instance per step (`engine-vulkan/src/program.rs:820-857`), and instances are registered
  while serving, so their peak is not laid at load. The only laid bound is lanes fitted x arms.

**HAZARD_CAP** (`engine-vulkan/src/device/ctx.rs:1308`, 96).
- Bounds: read and write spans tracked since the last barrier. Past 96 a barrier is owed
  whatever the spans are (`ctx.rs:1320-1323`), so the constant decides barriers baked into a
  recording, not memory. It exists because each dispatch is compared against every tracked
  span.
- From laid things: at most dispatches x bindings per dispatch spans can be tracked in one
  recording; with at most 12 own buffers a kernel, that is the dispatch count times a small
  constant.
- Binding form: removes the reason for the cap. Barriers are baked once from laid extents at
  recording, not on every fire, so the comparison cost is paid once per (arm, rung, class) and
  can track every span. The constant goes; the barrier count becomes what the laid extents
  require.
- Where it remains: frames encoded live on every fire (guest passes) still pay the comparison
  per fire. For them the bound is the dispatches of that pass, which is known when the
  program is registered, not at load.

**widen SLOTS** (`engine-vulkan/src/guest/widen.rs:23`, 16; SLOT_BYTES 256, `:26`).
- Bounds: config slots of the readout widen, "one per widen still on the queue"
  (`widen.rs:14-15`). The slots are a ring that wraps without a check (`:141-144`); a frame
  holds at most one widen (`:117-125`).
- From laid things: widens recorded and not yet executed, which is arms x deferred guest
  passes per step.
- Binding form: removes it outright. A dispatch's params sit at a fixed offset in the per-arm
  params buffer, so the widen's config is its dispatch's block there and the ring is gone.
- SLOT_BYTES is not a heuristic: it is the largest storage-buffer offset alignment a device
  may ask, and the device's own limit is the derived value.
- Where no derivation holds without the form: deferred passes per step depend on instances
  registered while serving. If more than SLOTS widens are queued before the first runs, the
  first runs with a later one's offsets; whether serving can reach that was not verified.

**BIND_CACHE_CAP** (`engine-wgpu/src/device/pipelines.rs:174`, 16 x 1024; cleared whole at the
cap, `:250`).
- Bounds: wgpu bind groups cached. Today's key is (pipeline, views with offsets and sizes,
  scalar bytes), so the set grows with every distinct offset and scalar.
- From laid things: pipelines x distinct buffer tuples.
- Binding form: removes it outright. The key is (pipeline, distinct buffers), a fixed set
  made at load; nothing is cached, cleared or capped.

**UNIFORM_RING** (frames-laid c23a6a821 `engine-wgpu/src/device/ctx.rs:63`, 4 MiB; on
451233bfd the growing `UNIFORM_CHUNK` `ctx.rs:61` and `UNIFORM_ARENA_CHUNK` `pipelines.rs:175`).
- Bounds: uniform blocks of the bind groups kept. On frames-laid it is already tied by its
  doc to the cap above: "every bind group the cache keeps at a 256-byte block"
  (`ctx.rs:61-62`), and 16 x 1024 x 256 bytes is the 4 MiB written.
- From laid things: arms x dispatches of the widest laid recording x the aligned block, the
  block being the largest params block rounded up to the device's uniform offset alignment.
  The probe puts the largest params block under one 256-byte block on every kernel built.
- Binding form: replaces the constant by that expression, one params buffer per arm. The
  ring, its laps and the wrap that drops every kept group (`pipelines.rs:241-262` on
  frames-laid) go.

**SCRATCH_RING** (frames-laid c23a6a821 `ctx.rs:67`, 8 MiB; on 451233bfd `SCRATCH_CHUNK`
`ctx.rs:63`).
- Bounds: bytes one frame stages. Two users: views at an offset the device cannot bind
  (`engine-wgpu/src/encode.rs:346-402`) and copies whose two ends are one buffer, in pieces of
  half the ring (frames-laid `ctx.rs:945`). A frame still encoding keeps every block it took,
  so a frame that stages more than the ring is refused (`Ring::take`, frames-laid
  `ctx.rs:303-346`).
- From laid things: the staged extents of the widest laid frame, summed over its dispatches,
  per arm. The size-only walk lays every workspace, so those extents are known at load.
- Binding form: removes the view half outright, since a binding is a whole buffer at offset
  zero and no view is staged for alignment. It does not remove the copy half: with one pool
  buffer every pool-to-pool copy has both ends in one buffer, which is the case that stages.
  That half goes only if copies become kernels, as the arming note says they must.
- Where no derivation holds: a read at an offset that is not a whole word
  (`encode.rs:358-383`) is staged because kernels index words; it stays unless every laid
  offset is word-aligned.

**PRICED_FROM, PRICED_MOVES, PRICED_FROM_MOST** (`engine-gpu/src/serve.rs:4066`, `:4069`,
`:4073`; 16, 3, 64).
- Bounds: how many pipelines a load builds to measure what the driver takes per pipeline,
  before the pool is fit (`price_pipelines`, `serve.rs:4097-4148`). The price times the
  pipelines left unbuilt is reserved as `outside`, so the three decide both device objects
  built at load and bytes held back from the pool.
- Derivation attempt: no derivation found. They are sample sizes for a quantity the driver
  does not report: the size of the block its shader heap grows by. Candidates considered:
  the driver's block size (no interface reports it); the census length (already used for the
  stride, `:4107`, and it says how many there are, not how many must be read); the granularity
  of the device's memory budget reading (it is what the reading moves by, not how often a
  build moves it).
- Binding form: removes the need only under one condition. If every (arm, rung, class) is
  recorded at load, every pipeline a forward can name is built before the pool is fit, the
  unbuilt count is zero for forwards, and `outside` is read whole across those builds, with
  nothing estimated. If recordings are made on first use, pipelines are still unbuilt at fit
  time and a price is still needed. Guest stages are built when a program is registered and
  are outside either case.

**NODE_COPIES** (`runtime/src/store/kv/page_table.rs:217`, 4).
- Bounds: per first-page hash, how many node copies the prefix index keeps
  (`index_node_start`, `page_table.rs:611-626`); the oldest is dropped from the index, not
  from the table. `sealed_run` scans the copies and takes the one that runs furthest
  (`:498-512`). It decides no device object and holds no pages. It decides which sealed copy
  a new prompt can find, and so whether a prefix is reused or computed again.
- Derivation attempt: no derivation found, and nothing laid bounds it. Copies arise when
  pages are sealed apart, once per working set that seals the same first page on its own.
  Candidates considered: two (one copy on the device, one off it, which is what the comment
  names), but which copy runs furthest depends on the keys asked, so no fixed count is
  sufficient; lanes fitted (a copy outlives its lane as cache).
- What holds without a number: the index entry is already removed with its node
  (`page_table.rs:1171-1178`), so an uncapped index is bounded by live nodes, which the pool
  bounds. The cap then only limits the scan in `sealed_run`, which is linear in copies.
- Binding form: does not touch it.

### 2.3 TARGET_CHUNK

A design parameter of the pool. No formula is proposed.

What it is: the size the chunk plan aims near. 4 MiB on Vulkan and wgpu
(`engine-gpu/src/store/layout.rs:79`, through `engine-gpu/src/chunks.rs:22-41`), 2 MiB on
Metal (`engine-metal/src/store/layout.rs:67`), 16 MiB on XLA
(`engine-xla/src/store/layout.rs:41`) and in the server CUDA profile
(`poem-compiler/src/chunk/profile.rs:90-91`). The plan weighs whole multiples of the kv list
slot between half and twice the target (`profile.rs:168-176`,
`poem-compiler/src/chunk/mod.rs:806-838`) and takes the one that wastes least, nearest the
target among equals (`mod.rs:941-950`). A chunk is never smaller than one list slot, so a
large page overrides the target.

The kv list slot is one page of every layer (`engine-gpu/src/store/layout.rs:7-12`). Catalog
shapes: a layer's kv is `[tokens, 2, kv_heads, head_dim]` (`models/qwen3_dense.poem:102`,
`models/qwen_3.poem:537`), in bf16 for the deployments listed.

| Model (catalog dims) | kv bytes a token | Page of 16 tokens | Page of 32 tokens |
|---|---|---|---|
| qwen3-0.6b, 28 layers (`qwen3_dense.poem:82`) | 112 KiB | 1.75 MiB | 3.5 MiB |
| qwen3-8b, 36 layers (`qwen3_dense.poem:83`) | 144 KiB | 2.25 MiB | 4.5 MiB |
| llama33-70b, 80 layers (`llama_3.poem:78`) | 320 KiB | 5 MiB | 10 MiB |
| qwen35-d9b, 8 attention layers of 32 (`qwen_3.poem:426`, `:524`) | 32 KiB | 0.5 MiB | 1 MiB |

Chunk sizes the plan may weigh, from those slots and each backend's range:

| Model | Vulkan, wgpu (target 4 MiB, page 16) | Metal (target 2 MiB, page 32) | Server CUDA (target 16 MiB, page 16) |
|---|---|---|---|
| qwen3-0.6b | 3.5, 5.25, 7 MiB (2 to 4 pages) | 3.5 MiB (1 page) | 8.75 to 31.5 MiB (5 to 18 pages) |
| qwen3-8b | 2.25, 4.5, 6.75 MiB (1 to 3 pages) | 4.5 MiB (1 page, past twice the target) | 9 to 31.5 MiB (4 to 14 pages) |
| llama33-70b | 5 MiB only (1 page) | 10 MiB (1 page, five times the target) | 10 to 30 MiB (2 to 6 pages) |

What the chunk size trades:

- Lending granularity. A rung lends and takes back whole chunks, so one chunk is the
  smallest amount that moves. On Vulkan that is 2 to 4 pages of qwen3-0.6b (32 to 64 tokens
  of one lane) and exactly one page of llama33-70b (16 tokens). A boundary between the state
  side and the activation span rounds to a chunk, so up to one chunk per boundary is neither.
- Evacuation cost of one chunk. To hand a chunk back its live slots are copied out: at most
  the chunk's bytes (3.5 to 7 MiB for qwen3-0.6b on Vulkan, 5 MiB for llama33-70b) and one
  entry rewritten in the logical-to-physical table, since held slot refs are logical and are
  not renamed (`new-memory/PLAN-P7.md:29`). A larger chunk costs more per evacuation and is
  less often empty, because any one live slot pins it.
- Table size and chunk count. One table entry and one set of counters per chunk. Per GiB of
  pool: 455 chunks at 2.25 MiB, 293 at 3.5 MiB, 205 at 5 MiB, 64 at 16 MiB. A smaller chunk
  means more entries to walk when the pool is ordered or checked.
- Waste. Only whole slots fit a chunk. The list slot divides it exactly by construction; the
  fixed records, the rs list and weight slots do not, and their tails are what the plan
  weighs (`mod.rs:916-953`).

Constants tied to it:
- The range, half to twice (`profile.rs:171-175`); `MOST_CANDIDATES` 128 sizes weighed and
  `WEIGHT_TAIL_SLACK` 0.01 (`mod.rs:105`, `:86`); the tie slack `(state / 10_000).max(4 KiB)`
  (`mod.rs:944`); `RS_SHARES` 4 (`mod.rs:99`).
- Device limits: a chunk is never larger than one binding or one segment (`profile.rs:153-157`);
  on Vulkan and wgpu both are the device's largest buffer (`engine-gpu/src/serve.rs:785-792`,
  `chunks.rs:25-27`), and a pool is at most four segments (`chunks.rs:15-17`).
- Alignment: `row_align` and `segment_grain` 256 (`chunks.rs:26`, `:28`); the TPU tile
  (`profile.rs:132`, `engine-xla/src/store/layout.rs:37-39`) and `MOST_CHUNK` 256 MiB there
  (`layout.rs:43`).
- Growth: CUDA maps in 2 MiB pages and 32 MiB units with 256 handles an arena
  (`engine-cuda/src/device/elastic.rs:4`, `:7-9`); the runtime commits 64 MiB ahead
  (`runtime/src/store/shared.rs:73`).
- `MIN_PIECE` 1 MiB (`mod.rs:51`): the least piece of a cut record, which has to fit a chunk.

### 2.4 Page sizes

Design parameters. No formula is proposed.

**kv page.** 16 tokens on Vulkan, wgpu and CUDA (`engine/src/load.rs:38`,
`worker/src/backend.rs:134`), 32 on Metal (`backend.rs:86`). Where a backend states none the
plan takes the fewest tokens, a multiple of `TOKEN_GRAIN` 16, at which every layer's rows run
`MIN_RUN` 32 KiB (`poem-compiler/src/chunk/mod.rs:54`, `:57`, `:700-716`, `:1086-1110`). The
operator key is removed: "a kv page is the compiler's chunk plan"
(`worker/src/config/removed.rs:141-142`).

What it trades, with the table of 2.3:

- Fork copy cost. The first write after a fork to a page two working sets read copies that
  page (`runtime/src/pipeline/fire/kv.rs:14-27`). One page is 1.75 MiB for qwen3-0.6b,
  2.25 MiB for qwen3-8b, 5 MiB for llama33-70b and 0.5 MiB for qwen35-d9b at 16 tokens, and
  twice that on Metal at 32. The token written is 112, 144, 320 and 32 KiB of it.
- Tail waste per lane. A lane's last page is partly empty: up to one page less one token
  (1.64 MiB for qwen3-0.6b, 4.69 MiB for llama33-70b at 16 tokens), about half a page on
  average (0.875 MiB and 2.5 MiB). A hundred lanes of qwen3-0.6b hold about 87 MiB of empty
  tail at 16 tokens and 175 MiB at 32.
- Page-table size. One u32 ref a page a lane (`new-memory/PLAN-P7.md:29`): a lane of 10240
  tokens is 640 refs at 16 tokens and 320 at 32; a lane of 32768 tokens is 2048 and 1024.
  Halving the page doubles the table a fire uploads and the refs a fork or an eviction walks.
- Sharing granularity. A prefix is shared, cached and evicted by whole pages, so a shorter
  page reuses more of a prompt that differs late and a longer one fewer.

Constants tied to it: `TOKEN_GRAIN` and `MIN_RUN` above; the chunk, which is a whole number
of pages and never smaller than one (so on Metal the page, not `TARGET_CHUNK`, sets the chunk
for qwen3-8b and llama33-70b); the page ceiling `u32::MAX` (`engine/src/fit.rs:88`); the pages
moved at a time by a snapshot and by an offloaded prefill, 64 and 32
(`runtime/src/handle/snapshot.rs:37`, `runtime/src/offload/prefill.rs:77`).

**State page (the fixed record and its pieces).** A working set's recurrent state is one
record, cut into pieces no smaller than `MIN_PIECE` where it outgrows a chunk
(`poem-compiler/src/chunk/mod.rs:49-51`; `engine-gpu/src/store/layout.rs:13-15`). Catalog
shapes for a gated-delta layer: `delta_state [v_heads, k_dim, v_dim]` and
`conv_state [conv_kernel, qkv]` (`models/qwen_3.poem:553-556`). Vulkan, wgpu and Metal keep
state rows in f32 (`engine-gpu/src/chunks.rs:36`, `profile.rs:121`).

| Model | Recurrent layers | A layer, f32 | Record, f32 | Record, bf16 |
|---|---|---|---|---|
| qwen35-d0.8b (`qwen_3.poem:334`) | 18 of 24 | 1.09 MiB | 19.7 MiB | 9.8 MiB |
| qwen35-d9b (`qwen_3.poem:426`) | 24 of 32 | 2.13 MiB | 51 MiB | 25.5 MiB |

- Fork copy cost. A recurrence writes every layer's state on every token, so the first token
  after a fork copies the whole record whatever the piece size: 19.7 MiB and 51 MiB in f32.
  The piece size does not change the bytes copied; it changes how many slots the copy takes
  and whether they are adjacent.
- Tail waste. A piece must fit a chunk, and a chunk holds whole pieces only. A layer of
  qwen35-d9b is 2.13 MiB in f32, so a chunk near the 4 MiB target holds one layer and leaves
  most of the rest unless the plan picks a size the layers fill; that fit is what the plan
  weighs and was not computed here.
- Table size. Pieces per record times working sets is the count of fixed slots the runtime
  tracks and moves on a suspend.
- Tied constants: `MIN_PIECE`; `FIXED_SLOT_FLOOR` 1 KiB (`mod.rs:82`); `FIXED_SETS_PER_WS` 1
  and `COW_SETS_PER_LANE` 1, the copy-on-write reserve of one record per lane of the widest
  fire (`mod.rs:61-78`); `MOST_CUTS` 64 (`mod.rs:102`); `SLOT_SKEW` 1 KiB (`mod.rs:1062`).

**Others, not computed.**
- rs-list slot: at most `MOST_RS_TOKENS` 16 unfolded tokens, a power of two (`mod.rs:89`),
  weighed with `EXPECTED_UNFOLDED` 4 (`mod.rs:95`). It trades a fork's copy of a shared tail
  slot against slots per lane.
- Window-list slot: `T_w` tokens, chosen per window by the plan
  (`engine-gpu/src/store/layout.rs:18-21`); `WINDOWS` 4 lists at most (`layout.rs:61`).
- `INSTANCE_SLOT` 4 KiB and `CHANNEL_SLOT` 1 KiB (`layout.rs:56`, `:58`), stated as
  structural; an item past a slot takes a run of adjacent slots.

---------------------------------------------------------------------------------------------

## 3. KEEP_CACHE to PLEX: the surface change

Readable alone. Nothing here is built yet.

**Today.** The governor holds the pool's target at everything wanted, cache included, until
live demand has not risen for `KEEP_CACHE` ticks (ten minutes); then it keeps only what is
live (`runtime/src/governor.rs:269`, `kept()` `:451-457`). With `governor_release_idle` on,
the target is cut toward live, and a pool that cannot shrink because cache holds its chunks
raises device tier pressure (`runtime/src/governor/run.rs:99-128`,
`governor/steered.rs:291-330`). The ten minutes is a retention decision made by the runtime.

What a worker policy has (`package plex:policy@1.0.0`):
- Reads `candidates(kind)`, `attr(of, name)` (`crates/policy/wit/io.wit:8`, `:29`). Writes
  `set-alarm`, `install-retention(list<retain>)`, `discard(working-set)` (`:49`, `:64`, `:67`).
- Edges `timer`, `retained`, `discarded`, `tier-pressure` (`crates/policy/wit/types.wit:94`,
  `:110`, `:111`, `:121`). SDK: `worker::Policy::{on_tier_pressure, on_retained, on_discarded,
  on_timer, plan}` (`crates/policy/src/lib.rs:158-181`).
- Worker attrs `{device,host}-band`, `-short-bytes`, `-{model,outside,total,state,live,wanted,
  act}-bytes` (`runtime/src/policy/service.rs:1007-1037`). Working-set attrs `tenant`,
  `session`, `tokens`, `kv-bytes`, `last-use-ms` (`:1186-1224`).
- Cached sets with an owner are already `working-set` candidates (`service.rs:1138-1142`).
  This corrects pass 2 (A.4), which said idle cache is not offered.
- One shortfall per tier, summed over a planner's waiters and the governor's own cuts
  (`runtime/src/policy/pressure.rs:182-201`, `:242-271`). A policy cannot tell them apart.
- Default responder, when no worker policy subscribes to `tier-pressure`
  (`service.rs:363-430`): discards working sets until the shortfall is covered, lowest key
  first, then least recently used, never one keyed `PINNED` (`pressure.rs:416`, `:429-452`),
  never a cached set whose owner still runs (`service.rs:1309-1313`).

**What changes.**
1. The governor keeps live plus cache keyed above the floor. `KEEP_CACHE` and its clock go.
   Cache at the floor (key 0: every key when no policy is installed) no longer holds the
   target up.
2. The governor's cut drops floor cache itself, through the drop the planner already uses
   (`reclaim_droppable`, `runtime/src/store/kv.rs:752-757`), and raises nothing for it.
3. Idle cache is offered as candidates in full. Missing today: entries with no owner
   (`store/kv.rs:689-710`), the recurrent-state cache, and how much of a cached set sits on
   the host (`service.rs:1086-1091`, "not told").
4. A release-idle reading apart from a waiter's shortfall: the bytes the governor would give
   back if keyed cache let go, beside `device-short-bytes`, which then means waiters only.
   Open: whether a backoff under system pressure is a third reading or stays in the shortfall.
5. `last-use-ms` stays as it is. A policy ages its keys on its own `timer` edge, as
   keep-session already does (`crates/builtins/policies/keep-session/src/lib.rs:84-112`).
6. Default responder: on a waiter's shortfall, as today. On a release-idle reading it
   discards nothing (floor cache is already gone; the rest a policy keyed). It holds no clock.

**An existing policy that is not updated.**
- None installed: idle cache is dropped when release-idle cuts begin (after `IDLE_AFTER`),
  not ten minutes later. With release-idle off, nothing changes.
- Keys cache above the floor, does not answer `tier-pressure` (keep-session, keep-handles):
  today the default responder discards that cache once the ten minutes pass under
  release-idle. After the change it stays until the policy lowers the key or discards it, or
  a waiter is short. The policy becomes the only thing that ages it.
- Answers `tier-pressure`: the governor's idle release no longer shows in
  `device-short-bytes`; a policy that let idle cache go on it must read the new reading.
- No WIT item is removed or changes type. The additions are attrs and candidates.

**Runtime sites the build touches.**
- `runtime/src/governor.rs`: `KEEP_CACHE` `:269`, `kept()` `:451-457`, `cut()` `:468-471`;
  `Demand` gains cache keyed above the floor (`runtime/src/store/slots.rs:1136-1144`).
- `governor/run.rs:99-128`, `governor/steered.rs:291-330`: the cut calls the floor drop
  before it reports, and reports release-idle apart from a shortfall.
- `runtime/src/store/kv.rs`: `reclaim_droppable` `:752-757`, floor test `:997`,
  `cached_working_sets` `:689-710`.
- `reclaim_idle_once` (`runtime/src/planner/residency_planner.rs:342-349`, port
  `planner/pool.rs:256-263`); callers today `planner/acquire.rs:159`, `planner/plan.rs:149`,
  `:189`. The governor becomes a fourth.
- `runtime/src/policy/pressure.rs:182-271` (a gauge per origin);
  `runtime/src/policy/service.rs:1007-1037`, `:1062-1100`, `:387-430` (attr, candidates,
  responder).

---------------------------------------------------------------------------------------------

## Appendix: batching numbers with no memory in them

For the question put to the user as outside the memory redesign. One line each.

- `CLOSE_CHANNEL_BATCH_MAX` = 512 (`runtime/src/scheduler/worker.rs:1921`): channel ids
  carried by one close-channels item on the scheduler queue.
- `LAUNCH_RUN_BEFORE_CONTROL` = 2 and `CONTROL_RUN_MAX` = 32
  (`runtime/src/scheduler/lane.rs:211`, `:213`): how many launches a lane takes in a row
  before it looks at control messages, and how many control messages in a row before it
  looks at launches.
- `SYNCS` = 8 (`runtime/src/handle/snapshot.rs:38`): a snapshot write is synced to disk in
  that many batches.
- `OUTPUT_BUFFER_CAP` = 4096 (`runtime/src/inferlet/process.rs:708`): output events a process
  keeps buffered; past it the oldest is dropped (`:800-805`). It is host memory of a few
  thousand small events per process, not pool memory.

---------------------------------------------------------------------------------------------

## Not verified

1. Nothing was built or run. Every model figure is arithmetic from catalog shapes and the
   profiles' constants. Which chunk size the plan actually picks for each model, the pieces
   it cuts a record into, and any row in a kv slot other than the kv rows (side planes, rs
   rows) were not computed.
2. The Metal figures assume the Metal recipe takes its page (32) as the list slot's tokens;
   the wiring from `METAL_PAGE_SIZE` into the Metal chunk profile was not traced. The CUDA
   figures use the `SERVER_CUDA` profile as written; what the CUDA engine overrides in it was
   not traced.
3. No measurement was found for ENGINE_LANE_HOT_US or GAP_BUDGET in commit messages or
   `new-memory/*.md`. Benchmarks outside the repository (scratch directories, team notes)
   were not searched.
4. `Turn.idled` and `Turn.yielding` have no reader that I found on 451233bfd; a reader
   through a path other than `steer.turn`'s return value was not excluded.
5. The structural classes of a rung are taken from the arming note's inference (2^C - 1 class
   mixes); the note itself marks C as inferred. Whether the classes can be enumerated at load
   was not checked in code.
6. Whether serving can queue more than 16 deferred widens before the first runs
   (widen SLOTS wrapping) was not verified.
7. That the fork copy of a kv page is the whole slot on every engine was read at the runtime
   (`pipeline/fire/kv.rs`), not in each engine's copy.
8. "A recurrence writes every layer's state on every token" is taken from the model shapes,
   not from a trace of writes.
9. The wgpu ring constants are read on c23a6a821 (frames-laid), which is not in 451233bfd;
   line numbers for them are on that commit.
10. The chunk-count and table figures assume one table entry per chunk; the entry's size
    was not read.
11. The guest `settle` calls in `engine-vulkan/src/program.rs` and `guest/session.rs` were
    not checked to be the pipeline cache's `settle`; only `engine-vulkan/src/api.rs:113` was.
12. Section 3's "what an existing policy sees" is read from the code paths, not from a run
    with keep-session or keep-handles installed.
