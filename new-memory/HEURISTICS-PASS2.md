# Pass 2: every number into WAITS / SHARES / SIZES, lent chunks, the join hold

Read-only. Every `file:line` is under `crates/` on origin/memory-redesign `451233bfd`
(read from a `git archive` export; nothing in `/root/pie` was checked out, built or run).
The Pie Leader's rulings (1) to (7) are taken as given. The withdrawn MI300X 8.1 ms figure is
not used anywhere below.

Status words used in the tables:

- **derivable now**: the replacement is a value the load or the runtime already holds.
- **needs a measurement (which)**: the replacement exists only after that measurement.
- **policy input (where)**: no derivation; it is a decision, and this is where it belongs.
- **contract**: a sandbox, protocol or platform limit; where it is stated is named.
- **leftover (ruling n)**: stays this PR by the Pie Leader's ruling.
- **no derivation found; candidates: ...**: said plainly where that is the case.

Lines that moved since pass 1 (cec91bb4b to 451233bfd): `PASS_ASKS_FROM` `runtime/src/store/shared.rs:241`,
`DEFAULT_FRAME_SIZE` `runtime/src/scheduler.rs:305`, `MAX_INFLIGHT_UPLOADS` `runtime/src/server.rs:232`,
`NODE_COPIES` `runtime/src/store/kv/page_table.rs:217`, `AGAIN` `runtime/src/policy/pressure.rs:232`,
second count-freshness site `runtime/src/store/kv.rs:2351`, backstop `runtime/src/scheduler/worker.rs:1334-1335`,
`CLOSE_CHANNEL_BATCH_MAX` `runtime/src/scheduler/worker.rs:1921`. All other pass-1 lines still match.

Paths in the tables are shortened: `rt/` = `runtime/src/`.

---

## A. The inventory, regrouped

### A.1 WAITS (a time someone waits)

Kinds: **deadline** = a failure deadline reported to the caller, not a retry. **cover** = a clock
that stands in for a wake that is missing or can be lost (a bug to name). **poll** = a clock that
re-reads something that could announce itself. **debounce / pace** = a clock with no event behind
it at all (a decision).

| Row | Number, where | Kind; the event waited for; who signals it | Replacement; what it touches; status |
|---|---|---|---|
| K4a | governor `PATIENCE = 2` `rt/governor.rs:132`, used `:381` | debounce of a polled reading ("One congested tick is a single reading of a noisy signal", `:129-131`). Event: the OS saying pressure is sustained. Nobody signals it; the loop polls (`rt/governor/psi.rs:26`, `:52`; macOS `sysctlbyname` `rt/governor/macos.rs:24`) | Act on the reading: Linux PSI `avg10` is already the kernel's own 10 s average, so a second 250 ms sample debounces nothing the kernel has not. Goes away entirely with R6's event source. Touches `rt/governor.rs`. **needs a measurement** (how often one congested reading is followed by a calm one, Linux and macOS) |
| K4b | the same `PATIENCE` read by the rung walk `rt/store/shared/target.rs:12`, `:128-140` | cover. Event: the last ask for the wide rung leaving. The signal is missing: `RungWant::drop` only removes the want and wakes nobody (`rt/store/shared.rs:205-211`); the walk hears of it at the next governor tick | Rule 3 (ruling 7): the rung comes down when the ask leaves, with the wake on the leaving ask; the patience read is deleted. Touches `rt/store/shared.rs`, `rt/store/shared/target.rs`. **derivable now** (DERIVATIONS section 3) |
| R1 | `FIRST_HOLD = 4`, `LONGEST_HOLD = 256`, hold x4 on a push-back `rt/governor.rs:252-253`, `:397` | pace: calm ticks before the target probes up again. Event: "the system stopped pushing back". Nobody signals a release of pressure; rises already have an event path (`Governor::asked`, `:441-447`) | No derivation found. Candidates: (a) no hold: rise only on `asked` while the kernel's average reads calm (the kernel's window is then the only clock, and it is the kernel's); (b) the hold becomes the operator's. **policy input (`governor_backoff`**, today only the multiplicative factor, `worker/src/config.rs:938`) unless (a) is measured not to oscillate on a shared host |
| R2 | `SETTLED = 64` `rt/governor.rs:257` (used in `calm`, `:405-430`) | pace: calm ticks before a lengthened hold halves | Falls with R1: no hold, nothing to settle. Same status as R1 |
| R3 | `IDLE_AFTER = 120` (30 s) `rt/governor.rs:261` | pace: how long unused committed room stands before the target starts down. Event: `demand.live` falling with no waiter; it exists (the pool's demand, `rt/store/shared.rs:1828-1835`) but "will it be asked again soon" has no event | **policy input (`governor_release_idle`**: today a bool, `worker/src/config.rs:941`; the idle time is the operator's statement of how long a shared host lends its room). Candidate derivation: release at once when recommitting costs less than holding; **needs a measurement** (a segment's map cost on the lane, the quantity `COMMIT_AHEAD_BYTES` is documented against, `rt/store/shared.rs:73`) |
| R4a | `CUT_EVERY = 4` `rt/governor.rs:264`, used `:419-423` | pace of idle cuts | One cut to `max(kept, floor)` when idle begins. The pool walker already paces the release segment by segment off the fire path and stops at any retarget (`rt/store/shared/target.rs:377-405`, `:500-512`), so the governor's pacing duplicates it. Touches `rt/governor.rs`. **derivable now** |
| R5 | `KEEP_CACHE = 2400` (10 min) `rt/governor.rs:269`, read `:449-457` | a retention clock in the runtime | Goes to PLEX (ruling 6). See A.4 |
| R6 | governor `TICK = 250 ms` `rt/governor.rs:475`, `interval(TICK)` `rt/governor/run.rs:157` | poll of OS memory pressure. The event sources exist in the platforms and are not used: Linux PSI triggers (a threshold written to `/proc/pressure/memory`, then waited on), macOS memory-pressure dispatch source. Today both are read on the clock (`rt/governor/psi.rs:26`, `:52`; `rt/governor/macos.rs:24`, `:93`) | Wait on the platform's event; the tick's other consumers (`pool.ticked()`, `rt/store/shared/target.rs:518-523`; weight cache `rt/store/weights.rs:993`) go event-driven with rule 3. A PSI trigger takes its own stall threshold and window, which are R9's numbers moved, not removed. Touches `rt/governor/{run,psi,macos}.rs`. **derivable now** for the wake; thresholds per R9 |
| R12 | count freshness `max(1 s, took * 50)` `rt/store/kv.rs:1291-1293`, `:2351`; `RECLAIMABLE_FRESH = 1 s` `rt/planner/pool.rs:146` (used `:268`) | poll: a cached count is trusted for a time. Event: the cache or its retention keys changing. Partly signalled already: `host_cached.set(None)` at `rt/store/kv.rs:1161`, `:1268` | Invalidate on change, no clock. Touches `rt/store/kv.rs`, `rt/planner/pool.rs`. **derivable now**. Not verified: which mutations (entry going idle, a policy writing keys) still lack the invalidation |
| R15 | `SEAT_WAIT = 10 s` `rt/pipeline/fire.rs:2188`; `FIRE_PROGRESS = SEAT_WAIT` `rt/planner/in_flight.rs:30` | cover for a wait-for cycle the runtime cannot see. On this ref `SEAT_WAIT` has **no use but `FIRE_PROGRESS`** (grep of all crates); its doc ("Also how long a planner waiter stands short ... before it fails", `fire.rs:2186-2187`) is stale: the short waiter fails on an event, the policy's answer of nothing (`rt/planner/residency_planner.rs:250-254`), not on this clock. What the 10 s still decides: how long a process's unsettled fires count as progress (`rt/planner/proc.rs:67`, `in_flight.rs:79`) before `fleet_stalled` may count it parked (`rt/planner/inner.rs:353-372`). Event waited for: the fire settling, which pokes the planner (`in_flight.rs:8-9`). The case covered: a fire waiting on its guest's host-written input while that guest waits on the parked lane (`in_flight.rs:18-21`) | Record the wait-for edge itself (this fire awaits input from a guest blocked on a parked lane) and let `firing()` read it; then the stall is an event and the error the caller gets (`PlannerError::Short`, `residency_planner.rs:267-271`) needs no clock. Touches `rt/planner/{in_flight,proc,inner}.rs` and wherever a fire is known to await host input. **derivable now in principle**; not verified where "awaits a host-written input" is known. Until then it is a **deadline** and should be named as one |
| R21a | `JOIN_HOLD_US = 2000` `rt/scheduler/frame.rs:60`, hold at `:1233-1243` | give-up bound on a join. Event: the joining lane's first fire arriving (`record_arrival`, `frame.rs:334-337`) or the join being forgotten (`forget_staged`, `:660`); both arrive as messages on the worker channel, which wakes the park (`rt/scheduler/worker.rs:1343-1362`) | Wake is already an event; only the bound is a number. Candidate: one measured step time of the standing batch (waiting longer than a step costs more than running without the joiner). Touches `rt/scheduler/frame.rs`. **needs a measurement** (step time, per engine, taken at run time by the worker). See part C |
| R21b | `GATHER_POLL_US = 500` `frame.rs:59`, used `:1214-1229`, `:1233-1243` | poll: re-plans every 500 us while waiting for missing lanes or a join, although arrivals already wake the park | `Hold(time to the deadline)` instead of a re-poll step. Touches `rt/scheduler/frame.rs`. **derivable now** |
| R21c | `Hold(500 us)` for a blocked lane `frame.rs:1053-1066` | poll. Event: the pipeline's queued pre-launch copy leaving the worker's own queue (`blocked_lanes`, `rt/scheduler/worker.rs:2190-2194`, `:2176-2180`) | Re-plan when that copy is dispatched (the worker dispatches it itself), no timed hold. Touches `rt/scheduler/{frame,worker}.rs`. **derivable now**. Not verified: whether the copy's own dispatch waits on a control slot settling (if so that settle is the event, and it is already in the park's select) |
| R21d | `STRICT_WATCHDOG_US = 1 s` `frame.rs:57`, `:1220-1223`; `HOLD_SAYS_AFTER = 5 s` `frame.rs:27` | neither fails nor retries anything: the watchdog bounds one hold and is re-armed on expiry (`:1223`); `HOLD_SAYS_AFTER` only paces a log line (`:1196`) | With R21b gone the hold is "until the missing lane arrives or leaves its quorum" and the watchdog has nothing to bound. Diagnostic pacing stays as diagnostics. **derivable now** |
| R22 | backstop `250 ms` `rt/scheduler/worker.rs:1334-1335` | **cover for a missed wake; the code says so**: on timeout it checks whether the front launch or a control had already settled, counts `BACKSTOP_RETIREMENTS` and warns "completion retired by the backstop poll, not the nudge" (`:1463-1472`). The wake that can be lost is the completion nudge into the park's select (`:1343-1362`) | Find and fix the lost nudge, then park with no bound but `wait_hint`. Touches `rt/scheduler/worker.rs`, `rt/scheduler/lane.rs`. **needs a measurement** (`BACKSTOP_RETIREMENTS` over a soak on each backend: nonzero names the bug, zero licenses deleting the backstop). The stall dump at 10 s + 60 s per dump (`:1473-1480`) only logs |
| R24a | `ENGINE_LANE_HOT_US` = 1 s native, 0 on wasm32 `rt/scheduler/lane.rs:333-336`, spin `:371-378` | not a wait for an event: the lane spins after its last work, then blocks on a select over launches, controls and `arming::room_wake` (`:380-392`). The events exist; the spin trades a core for wake latency | Block at once, or keep the spin as an operator's choice. **needs a measurement** (decode step latency with and without the spin, each backend; the no-perf-regression gate decides). If kept: **policy input** (an operator key; none exists) |
| R25 | `GAP_BUDGET = 20 ms` `rt/scheduler/arming.rs:206`, used `rt/scheduler/lane.rs:452-466` | pace: lane time a gap gives a waiting raise's steps. Event that should end it: a request arriving on `launch_rx` / `control_rx` | Run steps while both queues are empty, checking between steps; the unit is one arming step. Touches `rt/scheduler/lane.rs`. **derivable now**; **needs a measurement** of the longest single arming step (it is the delay an arriving request can then see) |
| R26 | `AGAIN = 100 ms` `rt/policy/pressure.rs:232`; stale re-report in `raise` `:238-272`; waiter loop `rt/planner/acquire.rs:218-229`; restore `:273-288` | cover. Events that exist: the policy's answer (`pressure::answered()`), the waiter's own notify, the gauge changing. The case the clock covers: a discard was answered, freed too little, and nothing else changes, so nobody raises again | Raise again when a discard settles with the shortfall still standing (the event), not on a clock. Touches `rt/policy/{pressure,service}.rs`, `rt/planner/acquire.rs`. **derivable now**. Not verified: where the default responder's discards (`rt/policy/service.rs:413-`) report completion |
| R27a | `DEFAULT_DEADLINE_MS = 50`, `MAX_DEADLINE_MS = 1000` `rt/policy/manifest.rs:9-10` (validated `:125`, error `:91`); policy epoch `TICK = 1 ms` `rt/policy/program.rs:31` (`:360`, `:381`) | **deadline**: a policy program that overruns has its writes discarded (`policy/wit/io.wit:41`) | **contract**: the manifest's `deadline-ms` field. The WIT states the semantics but **no number**; the range and default are stated only in `manifest.rs`. To be honest as a contract they belong in the WIT or the manifest schema's documentation. The 1 ms tick is the deadline's resolution (the unit of the field) |
| R28a | `FORK_WAIT = 250 ms` `rt/handle/mod.rs:25` | **deadline** reported to the caller as `in_use` ("its pages are still moving"). The wait itself is event-driven: `moves_ended()` (`rt/handle/held.rs:212-242`, `:244-`) | The wake needs nothing. For the bound: refuse at once and let the guest retry on its own event, or take the guest's stated deadline (precedent: `submit-deadline-us`, `inferlet/wit/model.wit:211`). Touches `rt/handle/held.rs`, the handle WIT. **policy input** (the guest's, through the WIT) |
| R32a | `LEASE_TTL = 60 s` `rt/offload/prefill.rs:660` (sweep `:682-688`); `UNTAKEN_TTL = 60 s` `rt/offload/stream.rs:24` (`:128-134`); sweep period `min / 2` `rt/offload.rs:40` | liveness timeouts for a peer that died. Event: the asker's link closing | Release on link close; the TTL then covers only a partition with no close, which is a protocol timeout. **contract** of the offload protocol, not found stated in any `.wit`. Not verified: where a link close is delivered to the executor |
| R32b | `BLOB_GRACE = 60 s` `rt/inferlet/program/repository.rs:32` (`:641`, `:694`) | **cover for a missing hold**: "so one an install is still writing, or reusing, is not taken from under it". An install in flight does not pin the blobs it will name; the clock stands in for that pin | An open install holds a reference on its blobs; removal reclaims only unreferenced ones. Touches `rt/inferlet/program/repository.rs`. **derivable now** |
| G7a | `FENCE_TIMEOUT_NS = 120 s` `engine-vulkan/src/device/ctx.rs:306` (`:287`, `:1251`, `:1631`) | **deadline** on a device wait, reported as a fault. A hung device raises no event (device loss does, as an error code) | Stays a stated deadline; no derivation exists. **policy input** (an engine profile key would be its home; none exists) |
| G7b | `PIPELINE_CACHE_SETTLE = 2 s` `engine-vulkan/src/device/ctx.rs:450`, `:494-504` | debounce: a changed cache waits for more builds before it is written. Event: the load's (or a raise's) builds finishing | Write when arming completes and at shutdown. Touches `engine-vulkan/src/device/ctx.rs` and its `settle()` callers (not traced). **derivable now** |
| G9a | `Poller::WAIT = 500 ms` `engine-wgpu/src/device/host.rs:168`, used `:188-194` | bound on one blocking `device.poll(Wait)`; on timeout the thread kicks itself and polls again. Event: the submission completing, which the poll already waits for. The timeout has one other effect: `stop` is looked at only between polls (`:178-185`), so it bounds how long a stop waits behind a blocking poll | `timeout: None`: the poll returns when the submission does, and a stop then waits for that submission, which is work in flight anyway. Touches `engine-wgpu/src/device/host.rs`. **derivable now** |

Also seen, log pacing only (decide nothing): `ADMIT_SAYS_AFTER` `rt/planner/residency_planner.rs:74`.

**Summary of the waits.** Failure deadlines reported to the caller: R27a (policy deadline),
R28a (`FORK_WAIT`), G7a (fence timeout); R15 behaves as one today. Clocks covering a missing or
lost wake (bugs to name): K4b (the leaving ask wakes nobody), R22 (the backstop, by the code's own
counter), R26 (a discard that frees too little raises nothing), R32b (an install does not pin its
blobs), R15 (a fire-waits-on-guest-waits-on-parked-lane cycle is not recorded). Polls with an
event already available: R6, R12, R21b, R21c, G9a. The rest are paces or debounces: decisions.

### A.2 SHARES (a fraction or ratio)

| Row | Number, where | Replacement; what it touches; status |
|---|---|---|
| K1a | `DEVICE_SHARE = 0.9` `engine/src/load.rs:60`; CUDA `total / 10` in `safety_floor_bytes` `engine-cuda/src/device/elastic.rs:27-35` | The measured `outside` term in the ledger: capacity = what the device reports less what was measured outside the pool. The doc already says provisional "until that `outside` term is measured into the ledger (REDESIGN 2.2)". Touches `engine/src/load.rs`, each engine's accounting. **needs a measurement** (driver and context bytes outside the pool at load, per backend) |
| R4b | `CUT_SHARE = 8` `rt/governor.rs:265`, used `:468-471` | Goes with R4a: the whole gap is released, paced by the walker. **derivable now** |
| R7b | `THRASHING = 0.5` `rt/governor.rs:484` | An OS-signal threshold with no source. With R6 on macOS's own pressure events the refault ratio is not computed at all. Otherwise **policy input (`governor_backoff`)**; **needs a measurement** to state a default |
| R9 | PSI `CONGESTED = 5.0`, `CRITICAL = 10.0` `rt/governor/psi.rs:11`, `:13` | Same: they become the PSI trigger thresholds of R6. **policy input (`governor_backoff`)**; **needs a measurement** for defaults. (`CRITICAL: u32 = 4` in `rt/governor/macos.rs:14` is the OS's level code: **contract**, the platform's) |
| R11 | reclaim step `capacity / 64` times `(elapsed / 4 ms).clamp(1, 8)` `rt/store/kv.rs:770-774` | Free the waiter's shortfall, which the caller knows; nothing is sized by a share of the pool or by how long the walk took. Touches `rt/store/kv.rs`, `rt/planner/pool.rs:256-263`. **derivable now** |
| R17 | `RESERVE_SHARE = 32` `rt/planner/inner.rs:19`, used `:128`, `:132` | `seated x growth` exactly (`WINDOW_GROWTH` slots per seated lane, `:118-127`), held back only up to what the pool has beyond the head waiter's quote. Touches `rt/planner/inner.rs`. **derivable now** |
| R18 | `HOST_RESERVE_DIVISOR = 8` `rt/planner/residency_planner.rs:22`, used `:373`, `:437`, `rt/planner/eviction.rs:207` | What a suspend can actually need: the largest working set now resident on the device (the planner holds every seated process's quote). Touches the three sites. **derivable now** as a candidate; if a margin beyond it is wanted that margin is **policy input (PLEX)** |
| R19 | to-end spread `rt/planner/to_end.rs`: fold `(3a + b) / 4` `:38-40`, `:79`; "an eighth more lanes" `:~172`; `life * 2` `:~183`; eight groups `:~191`; env `PIE_TO_END_OVER` `:118-122` | **No derivation found.** Candidates: the fold becomes the exact mean over the last `seated` completions; the three seating ratios are seating policy and go where `seat_to_end` / `seat_spread` already live (E6) or to PLEX; the env override is deleted (no dual paths). **policy input (PLEX seating, or the chip profile)** |
| new | pressure bands: Critical entered at `short x 4 >= total`, left below `short x 8 >= total` `rt/policy/pressure.rs:84-104` | Not in pass 1. The bands shape what the policy is told (`types.wit:120` "crossed a pressure band"); the WIT states no number. Candidates: report the shortfall in bytes and let the policy band it. **policy input (PLEX)** |
| E3b | bodies allowance `room / 4` `engine-cuda/src/store.rs:~135-137` | The bodies weighed per instantiation (merged in `585097298`): the allowance is the sum of what the load's decode bodies weigh, not a share of the room. Touches `engine-cuda/src/store.rs`. **derivable now** |
| E-wsp | `engine.working_set_percent = 50` `engine-cuda/src/chip.rs:72` | DERIVATIONS section 1; rules 1 + 2 inside fire-fit's solve (ruling 7). **derivable now** |
| G1 | fire room `/ 2` `engine-gpu/src/serve.rs:711` | The same solve on Vulkan / wgpu (backend parity). **derivable now** |

### A.3 SIZES (a count or bytes)

Runtime.

| Row | Number, where | Replacement; what it touches; status |
|---|---|---|
| K1b | Metal `DRIVER_FLOOR = 64 MiB` `engine-metal/src/store/accounting.rs:5`; CUDA `SAFETY_FLOOR_BYTES = 128 MiB` `engine-cuda/src/device/elastic.rs:10` | The measured `outside` term, as K1a. The Metal doc calls the floor "the user's decision, every backend": if it stands as a decision it is **policy input (an operator key; none of the seven is it)**; otherwise **needs a measurement** (as K1a) |
| K2 | `PASS_ASKS_FROM = 16` `rt/store/shared.rs:241` (`pass_asks`, `:249-251`); `RUNG_WANT_FROM` `rt/pipeline/fire.rs:731` | A pass asks when its rows exceed what the least rung fires (`rungs().first()`), or its fire is cut by the cap (already the second term). Touches `rt/store/shared.rs`, `rt/pipeline/fire.rs`. **derivable now** |
| K3 | `UNFOLDED_ROWS = 16` `engine/src/fire.rs:590` | The model's MTP head count, or a draft depth the inferlet declares; the doc names both. Touches `engine/src/fire.rs`, CUDA `store/rs.rs`. **derivable now** (head count); the declared depth is a manifest addition |
| R7a | `MIN_EVICTED = 2000` pages a tick `rt/governor.rs:483` | Goes with R7b |
| R8 | `GROWTH_STEP = 512 MiB` `rt/governor/run.rs:17`, used in `rise` `rt/governor.rs:459-466` | Rise to what is asked: `wanted - target` (the demand the pool reports), which the walker then commits segment by segment. Touches `rt/governor.rs`, `rt/governor/run.rs`. **derivable now** |
| R10 | `COMMIT_AHEAD_BYTES = 64 MiB` `rt/store/shared.rs:73`, used `:1816-1821` | One rung step's chunks (what the next raise will ask for), or one segment. Touches `rt/store/shared.rs`. **derivable now** |
| R13 | `SEATS_UNBOUNDED = 1 << 16` `rt/store/chunks.rs:525`, used `:562-566` | The fitted lanes. **derivable now** |
| R14 | `PIECES_AHEAD = 2` `rt/pipeline/fire.rs:736`, used `:3259` | The frame dispatch depth, which it is already maxed with. **derivable now** |
| R16 | `WINDOW_GROWTH = 2` `rt/planner/pool.rs:8`, used `:235` | Structural: the straddled slot and the next boundary's. Name it as structure, not a tunable. **derivable now** (it is a count of two named things) |
| R20 | `DEFAULT_FRAME_SIZE = 2` `rt/scheduler.rs:305` (`0 =>` at `:296`); `DEFAULT_DISPATCH_DEPTH = 2` `rt/scheduler/frame.rs:10` | Depth is structural (one frame executing, one queued behind it; REDESIGN section 3). Frame size has an operator key, `[runtime] frame_size`, so its default is **policy input (that key)**; no derivation found for the default |
| R23 | `CLOSE_CHANNEL_BATCH_MAX = 512` `rt/scheduler/worker.rs:1921-1923` | **No derivation found.** Candidates: every pending close in one item (the item carries a list); or the engine's declared channel count |
| R24b | `LAUNCH_RUN_BEFORE_CONTROL = 2`, `CONTROL_RUN_MAX = 32` `rt/scheduler/lane.rs:211`, `:213` (used `:216-217`, `:226`; no doc) | **No derivation found.** Candidates: launches before control = the frame dispatch depth (also 2; not verified to be the same quantity); controls = drain until a launch is queued (an event, `launch_rx` non-empty) |
| R27b | policy `MEMORY_BYTES = 64 MiB` `rt/policy/program.rs:32` (`:443`), `MAX_ENTRIES = 1 << 16` `:33` (`:68`, `:96`, `:590`, `:616`) | **contract**: the policy sandbox's limits. Stated only in `program.rs`; no `.wit` under `policy/wit` or `inferlet/wit` carries a numeric quota. They belong in the WIT's documentation or the manifest |
| R28b | `MAX_HANDLES = 4096` `rt/handle/mod.rs:29` (refusal `rt/handle/table.rs:372-373`); `MAX_MEDIA_BYTES = 8 GiB` `rt/handle/mod.rs:33` (`table.rs:121`) | Media: "Nothing moves these off the host, so this is all that bounds them", so they are host bytes and count against `budget_host`. **derivable now**. Handles: a guest quota, **contract**, stated only in code and the `quota` error, not in the WIT |
| R29 | snapshot `CHUNK_PAGES = 64` `rt/handle/snapshot.rs:37`, `SYNCS = 8` `:38` (`:726`); offload `CHUNK_PAGES = 32` `rt/offload/prefill.rs:77` | Snapshot chunk: the host slots of the move tier free at the time (it passes through them, `:35-36`). **derivable now**. `SYNCS`: **no derivation found**; candidates: one sync at the end, or one per chunk. Offload chunk: what the link carries while the next is read; **needs a measurement** (link throughput), or **policy input (PLEX**, which already sets link bandwidth shares, `policy/wit/io.wit:114`) |
| R30 | `UNCAPPED_PREWARM_PROCESSES = 64` `rt/inferlet/process.rs:226` (`:305`); `STAGED_COHORTS = 1` `:308` (`:311`); `OUTPUT_BUFFER_CAP = 4096` `:708` (`:801`) | Prewarm: the fitted lanes. **derivable now**. Staged cohorts: structural (the one gate that holds below the floor). Output cap: **no derivation found**; candidates: back-pressure from the consumer, or a guest quota stated in the WIT |
| R31 | `MAX_ROLLBACK_TOKENS = 64` `rt/inferlet/host/grammar.rs:81` (`:96`); `TOKENIZER_OFFLOAD_THRESHOLD = 64` and the `x 4` bytes per token `rt/inferlet/host/tokenizer.rs:17`, `:21`, `:29` | Rollback: guest-visible depth, **contract** if stated to guests (not in the WIT); candidate: the declared draft depth of K3. Offload threshold: a time trade; **needs a measurement** (tokenize cost per token against the hop to a blocking thread) |
| R32c | `REPORT_SLICE = 1024` `rt/offload/directory.rs:58` | Yield the lock when a fire is waiting on it, not every N entries. Touches `rt/offload/directory.rs`. **derivable now** if the lock can report a waiter (not verified which lock type it is) |
| R33 | `DEFAULT_CAPACITY = 256` `rt/pipeline/program.rs:102` (`:258`); `MAX_INFLIGHT_UPLOADS = 16` `rt/server.rs:232` (`rt/server/handler.rs:400-404`); `CORE_RESOURCES_PER_COMPONENT = 16` `rt/bootstrap.rs:779` (`:842-852`); `NODE_COPIES = 4` `rt/store/kv/page_table.rs:217` (`:612-626`) | Program cache: bound by bytes against `budget_host`, not a count; **derivable now**. Uploads: client-visible quota ("Too many uploads in flight"), **contract**, stated only in code. Core resources: wasmtime's pooling configuration, **contract** with the embedder. Node copies: **no derivation found**; candidates: the tier count plus one, or no cap (`retain` already drops dead copies) |

CUDA / ROCm, Metal, XLA, shared engine code.

| Row | Number, where | Replacement; what it touches; status |
|---|---|---|
| E1 | `engine.fire_lanes = 256` `engine-cuda/src/chip.rs:146` | **leftover (ruling 2)**, listed as such |
| E2 | `engine.forward_tokens` `engine-cuda/src/chip.rs:115` and profiles | Removed once L4, L40S and MI300X are re-measured (ruling 1): the scheduler decides fire width, memory only the ladder. **needs a measurement** (those three) |
| E3a | `BODIES_FLOOR_BYTES = 512 MiB` `engine-cuda/src/store.rs:126` | As E3b: the weighed decode bodies. **derivable now** |
| E4 | `TOKENS_FLOOR = 1024` `engine-cuda/src/serve/load.rs:25`, used `:283`, `:679` | The least rung the ladder declares (`:283` already takes `max_lanes.max(TOKENS_FLOOR)`); per ruling 1 the halving prices the rs replay scratch and a fit leaving zero state goes back into it. **derivable now** |
| Metal | `MIN_FITTED_FORWARD_TOKENS = 512` `engine-metal/src/api.rs:187`, used `:526` | The same least rung, the same solve (parity). **derivable now** |
| E5 | `CUDA_PAGE_SIZE = 16` `worker/src/backend.rs:134`; `METAL_PAGE_SIZE = 32` `:86` | CUDA's is "the planner's page"; Metal "has no planner to derive a page from". Replacement: the same planner derivation on Metal. Not verified what the planner derives the page from, so: **no derivation found here**; candidate as stated |
| E6 | chip keys `engine-cuda/src/chip.rs:108-147` | `walk_fire_rows`, `window_spare`: removed by REDESIGN 2.5. `fire_rows`, `prefill_lanes`: fire width, go with E2 (ruling 1). `lattice_steps`: the ladder, from the fit. `seat_to_end`, `seat_spread`, `seated_first`: seating policy, **policy input (PLEX or the profile)**, with R19 |
| G2 | `DEFAULT_STATE_SLOTS = 256` `engine-gpu/src/api.rs:272` (`:446`), `engine-metal/src/api.rs:189` (`:450`), `engine-xla/src/api.rs:245` (`:440`) | The fitted lanes. **derivable now** |
| patches | `DERIVED_PATCH_CEILING = 4096` `engine-cuda/src/api.rs:436` (`:453`), `engine-metal/src/api.rs:205`, `engine-xla/src/api.rs:259`, `engine-gpu/src/api.rs:288`; `PATCH_LATTICE_FLOOR = 64` `poem-compiler/src/budget.rs:174`, used in all four `api.rs` | Ceiling: what the model's vision tower declares (its position count times the images a prompt may carry). Floor: one image at the model's least grid. Both are load facts if the trace carries the vision config. **derivable now** as a candidate; not verified where the vision config reaches `bake_budgets` |
| G4 | `TARGET_CHUNK` 4 MiB `engine-gpu/src/store/layout.rs:79`, 2 MiB `engine-metal/src/store/layout.rs:67`, 16 MiB `engine-xla/src/store/layout.rs:41`; `INSTANCE_SLOT = 4 KiB` `:56`, `CHANNEL_SLOT = 1 KiB` `:58`; `WINDOWS = 4` `:61` | Slots: the doc says a measured instance's tables fit one; size them from that measure. **derivable now**. `WINDOWS`: ruling 4, a declared range. **derivable now**. Chunk: **no derivation found** beyond REDESIGN section 3 calling it structural; candidates: the least multiple of the device's map granule that holds the largest slot the load lays (CUDA has the granules: `MAP_UNIT_BYTES`, `LOGICAL_PAGE_BYTES`, `engine-cuda/src/device/elastic.rs:7-9`) |

Vulkan / wgpu (`engine-gpu`, `engine-vulkan`, `engine-wgpu`).

| Row | Number, where | Replacement; what it touches; status |
|---|---|---|
| G3a | `READOUT_ROWS_PER_LANE = 8` `engine-gpu/src/serve.rs:3703` (`:3830-3834`) | `poem_compiler::arena::readouts_per_lane(trace)`, which `bake_budgets` already uses (`engine-gpu/src/api.rs:~283`). **derivable now** |
| G3b | `PRICED_FROM = 16` `serve.rs:4066`, `PRICED_MOVES = 3` `:4069`, `PRICED_FROM_MOST = 64` `:4073` | Parameters of a measuring procedure (how many pipelines are built to read one's price). **No derivation found**; candidate: stop when two successive readings agree, an event of the measurement itself |
| G5a | `RING = 2` `engine-gpu/src/experts.rs:14` (`:470-471`, `:543`) | Candidate: the frame dispatch depth (R20). Not verified to be the same quantity |
| G5b | `MAX_ROUTES = 1 << 18` `experts.rs:16` (`:479-480`, refusal `:515`) | The most routes one fire of the top rung produces (rows times routed layers times experts chosen per row). **derivable now** as a candidate; the per-fire formula was not verified |
| G5c | `COPY_THREADS = 16` `experts.rs:18`; `FILL_THREADS = 16` `engine-gpu/src/store/rows.rs:48`; `GROUPS_PER_CORE = 4` `engine-gpu/src/guest/session.rs:15` (`:63-68`) | Threads: the host's reported parallelism. **derivable now**. Groups per core: a chip-measured decision; its home is the chip profile. **needs a measurement** (per chip) |
| G5d | `UPLOAD = 64 MiB` `engine-gpu/src/store/experts.rs:54`, `store/rows.rs:45`; `STAGE_BYTES = 32 MiB` `engine-gpu/src/store.rs:1588`; Vulkan `STAGING_BYTES = 256 MiB` `engine-vulkan/src/device/ctx.rs:147` (`:297`); wgpu `PREAD_CHUNK = 256 MiB` `engine-wgpu/src/device/alloc.rs:16` (`:468`) | The largest single item the load lays through that path (largest expert row block, largest spill batch, largest tensor), bounded by the device's reported buffer limit, which both engines already read (`engine-wgpu/src/device/alloc.rs:40`; `engine-vulkan/src/device/ctx.rs:697-699`). **derivable now**; where throughput per chunk matters, **needs a measurement** |
| G6 | `UNREPORTED_CORES = 16` `engine-vulkan/src/chip.rs:53` (`:83`), `engine-wgpu/src/chip.rs:30` (`:93`) | **No derivation found.** Candidates: a short scaling probe at load; or a chip profile entry for devices that report nothing. **needs a measurement** either way |
| G7c | `SPARE_FENCES = 32` `engine-vulkan/src/device/ctx.rs:157` (`:335`) | **No derivation found**; candidate: no cap on a free list (it holds only what was once needed). Use at `:335` read only as a cap |
| G7d | `SETS_PER_POOL = 1024` `:1281` (`:1433`), `SETS_PER_BATCH = 64` `:1283` (`:1412`), `BUFFERS_PER_POOL = x 8` `:1284` | The sets and buffers the recorded shapes bind, known from the recording. **derivable now** as a candidate. Not verified: the device's own allocation limits are not among the limits I found the code reading (`max_compute_work_group_*`, `max_storage_buffer_range`, `max_buffer_size`, `timestamp_period`) |
| G7e | `HAZARD_CAP = 96` `:1308` (`:1314-1321`) | Past the cap a barrier is owed regardless: a bound on a scan. **No derivation found**; candidates: exact tracking with no cap, or the cap from a measured scan cost |
| G7f | `TIMED_DISPATCHES = 8192` `:1238` (`:1149-1160`); wgpu `TIMING_QUERIES = 4096` `engine-wgpu/src/device/ctx.rs:65` (`:83`, `:780`) | Diagnostic only (the Vulkan pool exists under `encode::profiling()`, `:1142`); size from the recording's dispatch count. **derivable now** |
| G8a | Vulkan `SHAPES_KEPT = 64` `engine-vulkan/src/record.rs:74` (evicts `:146-152`) | The shapes the load declares (the ladder's rungs and lane buckets, the declared window range of ruling 4): keep all of them and nothing is evicted. **derivable now** |
| G8b | `KEEP_BYTES = 64 KiB` `engine-vulkan/src/guest/run.rs:324` | The doc gives the quantity: a vocabulary-wide cell. The model's vocabulary times the cell's element size. **derivable now** as a candidate (not verified that the session has the vocabulary at hand) |
| G8c | widen `SLOTS = 16` `engine-vulkan/src/guest/widen.rs:23`, `SLOT_BYTES = 256` `:26` | `SLOT_BYTES`: **contract**, the platform's: 256 is the largest storage-buffer offset alignment the Vulkan specification lets a device require (stated from the specification, not from the repo; the doc says "any device's"). The device also reports its own value. `SLOTS`: **no derivation found**; use not read |
| G9b | wgpu `TAB_PREFILL_WORK = 1 << 19` `engine-wgpu/src/tuning.rs:202` (`:273`, wasm32 only) | A measured platform limit (the doc's M2 Max figures: 95 ms dispatches ended, 48 ms not). Replacement: time a dispatch at load and bound the work to stay under the limit. **needs a measurement** (the platform's limit itself is not published) |
| G9c | wgpu `UNIFORM_CHUNK = 1 MiB` `engine-wgpu/src/device/ctx.rs:61` (`:242`, `:762`), `UNIFORM_ARENA_CHUNK = 1 MiB` `engine-wgpu/src/device/pipelines.rs:175` (`:261-265`), `SCRATCH_CHUNK = 8 MiB` `ctx.rs:63` (`:258-280`, `:800`) | One fire's uniforms (known from the recording) times the frames in flight; the scratch from the size-only walk that already lays every workspace (`585097298`). **derivable now** |
| G9d | wgpu `BIND_CACHE_CAP = 16 * 1024` `pipelines.rs:174` (clears all at `:250`) | The bind groups of the declared shapes, kept per shape with the recording as G8a. **derivable now** |
| G9e | wgpu `OUTBOX_FLOOR = 256` `engine-wgpu/src/guest/session.rs:31` (`:568-576`) | The guest's declared output cell. **derivable now** as a candidate; not verified whether the buffer grows past the floor |

Seen while reading and not in the pass-1 inventory (not classified in depth): CUDA
`LOGICAL_PAGE_BYTES = 2 MiB`, `MAP_UNIT_BYTES = 32 MiB`, `HANDLES_PER_ARENA = 256`
(`engine-cuda/src/device/elastic.rs:7-9`; the first two look like driver granules, a contract with
the device); CUDA `STAGE_BYTES = 16 MiB` (`engine-cuda/src/comm.rs:166`; as G5d); wgpu tactic
literals `decode_threads: 256, tile_rows: 16, tile_lanes: 16` (`engine-wgpu/src/tuning.rs:262-270`;
chip-tuned, as E6).

### A.4 `KEEP_CACHE`

**What the runtime does at that timer today.** Nothing is dropped by the timer itself.
`idle_ticks` counts calm ticks in which `demand.live` did not rise (`rt/governor.rs:412-417`,
reset on any pressure `:370-372`). At `idle_ticks >= KEEP_CACHE` (2400 ticks of 250 ms, 10 min),
`kept()` stops returning `demand.wanted` and returns `demand.live` (`rt/governor.rs:449-457`),
that is, the target no longer stays up for what the cache holds (`wanted` = live + cached +
waiting, `live` = without the idle cache; `rt/store/slots.rs:1136-1144`). Only where
`release_idle` is set and the pool is idle does anything follow: `cut(kept)` lowers the target
toward `max(kept, floor)` every `CUT_EVERY` ticks by `gap / CUT_SHARE` (`:419-423`, `:468-471`).
The pool walk then cannot release segments that still hold cache (`SegmentMove::NoRoom`,
`rt/store/shared.rs:1757`, `:1761-1763`; `rt/store/shared/target.rs:402-405`), `retarget` returns
`Retarget::NoRoom` (`target.rs:500-512`), the governor's loop turns that into
`Signal::TierPressure` (`rt/governor/run.rs:117-126`), and it reaches the PLEX gauge as tier
pressure from the governor's own source (`rt/governor/steered.rs:~288-305`). A policy subscribed
to `tier-pressure` answers it; otherwise the default responder does (`rt/policy/service.rs:387-430`,
lowest retention key first, then least recently used). On a dedicated pool (`release_idle` false,
the default, `worker/src/config.rs:953`) the timer changes nothing at all.

So the timer is a retention decision made by the runtime's clock: after ten minutes it stops
counting cache as worth keeping room for.

**What PLEX needs instead.** No clock in the runtime. `kept()` becomes: live, plus idle cache
keyed above the floor. Cache at or below `FLOOR` (`rt/policy/retention.rs:13-23`) is never a
reason to hold the target up: it is the runtime's to drop when its room is wanted (`may_drop`,
`rt/policy/pressure.rs:291-293`). When keyed cache stops being worth its room is PLEX's call,
and PLEX already has what it needs to make it on its own clock: the `timer` edge, delivered
whether or not it is listed (`rt/policy/manifest.rs:13-26`, `:132`), and `last_use_ms` on each
candidate (`rt/policy/pressure.rs:~419-427`). It ages cache out by lowering its keys. Two things
are missing for it to do that well:

1. A way to tell a release-idle reading from a waiter's shortfall. Both arrive as `tier-pressure`;
   the governor's source differs internally (`steered.rs:~288-293`) but the event's attributes
   were not seen to say "this is the pool giving idle room back, nobody is waiting".
2. Idle cache as candidates. The default responder's candidates are working sets only
   (`rt/policy/service.rs:1065-1067`, `SubjectKind::WorkingSet`).

**If PLEX keys nothing.** All idle cache sits at `FLOOR`. By the keyed-idle-demotion rule it is
dropped when its room is wanted and never copied down: `prepare_demote` skips whatever is not
above the floor (`rt/store/kv.rs:975-1000`, the skip at `:997`), and `reclaim_droppable` drops
exactly the `may_drop` entries (`rt/store/kv.rs:753-757`; rs checkpoints the same,
`rt/store/rs.rs:924-959`).

**A gap this exposes.** That drop is wired only to a planner waiter's want: `reclaim_idle` is
called from `reclaim_idle_once` (`rt/planner/residency_planner.rs:342-349`), whose three callers
are waiter paths (`rt/planner/acquire.rs:159`, `rt/planner/plan.rs:149`, `:189`). Nothing under
`rt/governor*`, `rt/store/shared*` or `rt/policy/service.rs` calls a reclaim. So when the room is
wanted by the governor's release (no waiter), floor-keyed idle cache is not dropped by that rule
today; the cut goes to tier pressure and the default responder, which looks at working sets.
With `KEEP_CACHE` gone, the governor's cut must itself count floor-keyed cache as releasable
and call the droppable reclaim for the segments it gives back. Minimal change: `kept()` takes the
above-floor cached bytes from the pool's demand (a third field beside `wanted` and `live`), and
the release path calls `reclaim_idle` before reporting `NoRoom`.

---

## B. Are lent chunks taken last?

**New chunks: yes. Slots in partly filled chunks: no.** (`rt/store/slots.rs`.)

State opens chunks lowest physical address first (module doc `:10-12`, `:20-27`). The openable
set is everything below `state_limit() = total - act` (`:756-758`, `:590-596`); lowering the rung
reopens the chunks between the old limit and the new (`set_act` `:802-808`, `reopen` `:624-630`),
and those lent chunks are the highest openable physical ids. A new chunk is
`self.open.first()` (`open_chunk`, `:1512-1515`; `Bits::first` is the lowest set bit, `:295-305`).
So while any unlent chunk is free, a newly opened chunk is unlent; lent chunks are opened last.

A slot, though, goes to a partly filled chunk of its shape before any chunk is opened:
`match self.shapes[shape].partial.first() { Some(&(_, _, logical)) => logical, None => self.open_chunk(shape) }`
(`acquire`, `:1167-1199`, the match at `:1190-1193`; runs the same, `:1234-1240`). `partial` is
ordered `(Reverse(used), physical, logical)` (`:365-367`): fullest first, then lowest physical.
Once a lent chunk has been opened and is partly filled, new slots of that shape go into it
ahead of a less full unlent chunk, and ahead of opening a fresh unlent chunk. With both an
unlent and a lent chunk partly filled, the fuller one wins whichever it is; physical address
only breaks ties.

Nothing moves state out of lent chunks between raises (no proactive compaction found); a raise
evacuates what stands in the new span to the lowest free chunk below (`plan_rung`, `:813-838`,
`move_chunk` `:915-938`) or reports `NoRoom`.

Local `sc/np-rung-lends` `9b3a3d30f` does not differ here: its `slots.rs` diff is six added lines
(`has_waiters`); `chunks.rs` and `pool.rs` are unchanged.

**Minimal change.** Put a lent bit at the front of the `partial` key:
`(lent, Reverse(used), physical, logical)` with `lent = physical >= total - act_chunks(top rung)`.
The boundary is the top rung's span start, fixed for the load, so the bit never changes for a
chunk in place. It is computed where the key is built (`reposition`, `:450-460`) and where a
chunk changes physical address (`move_chunk`, `:915-938`); the `check` invariant (from `:1923`; the `partial` assertions near `:1950-1987`)
asserts the same key. Every unlent partly filled chunk of a shape is then filled before any lent
one. A fresh unlent chunk would still be opened only after the lent partial chunks are full; if
rule 3 needs that case too, `acquire` must prefer `open_chunk` over a lent partial while
`open.first()` is below the boundary (two more lines at `:1190-1193`).

---

## C. Is a not-full step held under a wide cap?

**No. Fullness never causes a wait, and `JOIN_HOLD_US` does not read the cap.**
(`rt/scheduler/`.)

The step limit reaches the worker as `stepped = min(limit, fire_cap).max(1)`
(`worker.rs:1240-1243`) and is used in two places: a launch whose tokens exceed it stays queued
(`worker.rs:2406-2414`), and `build_frame_submission` (`worker.rs:2459`; `batch.rs:255-`) splits
an already sealed wave into step groups. In `batch.rs`, `StepGroup::push` returns "full"
(`:173-176`); its one caller uses that only to route the rest of the wave into the next group
(`closed = grouping.push(...)`, `:277`). The frame policy is built with the unstepped limits
(`FramePolicy::new`, `worker.rs:~1173-1180`), which it reads only in `seal()`
(`frame.rs:696`, `:703`, `:826-835`).

The holds a plan can return (`frame.rs`), none of which reads the token limit:

- blocked lane: `Hold(500 us)` when a front member's pipeline has a pre-launch copy queued ahead (`:1053-1066`);
- missing awaited lanes: `Park` if a frame is executing, else `Hold` to the watchdog deadline in `GATHER_POLL_US` steps (`:1214-1229`);
- join hold: `if !executing && self.is_joining() && self.have_seal_candidate()`, hold until `now + JOIN_HOLD_US` in `GATHER_POLL_US` steps (`:1233-1243`).

Trace, 64 lanes each decoding one token, all arrived:

- limit 8192: one wave sealed; one step group; no `push` returns full; one step dispatched; no hold.
- limit 64: one wave sealed; the 64th `push` returns full with nothing left to route; one step dispatched; no hold.

The two are identical. `JOIN_HOLD_US` applies only when nothing is executing and a join is in
flight (a pid in `joins_in_flight`, inserted `:493`, `:513`, removed on its first arrival `:337`
or `forget_staged` `:660`) while a seal candidate exists; the cap plays no part. The step limit
changes behaviour only when the lanes exceed it (two step groups, back to back, still no wait)
or one launch's tokens exceed it (held at `worker.rs:2406-2414` until the cap rises, an event).

---

## Not verified

- Where a fire is known to be waiting on a host-written input (R15's replacement depends on it).
- Which cache mutations lack the count invalidation (R12).
- Whether a pre-launch copy's dispatch waits on a control slot settling (R21c).
- Why the completion nudge can be missed (R22); only that the code counts it.
- Where the default responder's discards report completion (R26), and whether a `tier-pressure`
  event's attributes distinguish the governor's release from a waiter's shortfall (A.4).
- Where a link close reaches an offload executor (R32a); the lock type behind `REPORT_SLICE` (R32c).
- What the planner derives the page from (E5); where the vision config reaches `bake_budgets` (patches).
- The uses of `SPARE_FENCES`, widen `SLOTS` and `OUTBOX_FLOOR` beyond the lines cited.
- No numeric quota was found in any `.wit` under `policy/wit` or `inferlet/wit`; every "contract"
  number above is stated in code only, except the Vulkan alignment, which is the specification's.
