# Rung raise: cost of moving state vs fire time saved — what is measured today

Read-only study. Refs, cited as `path:line @TAG`:

| Tag | Ref | Note |
|---|---|---|
| T | `3e4e78300` | memory-redesign tip |
| O | `origin/sc/np-rung-offload` (89c900041) | merge-base with T: f8d9104ac |
| F | `origin/sc/np-flight-term` (0c951f2a4) | merge-base with T: cec91bb4b |
| L | `sc/np-rung-lends` (c853968bc) | merge-base with T: 10d708421 |

None of O, F, L is an ancestor of T, and L is not in O. Paths are under `crates/`.
There is no `engine-rocm` crate (ROCm builds from `engine-cuda`); Vulkan and wgpu share `engine-gpu`.

## Short answer

- **Move side:** the planner keeps cumulative pages and microseconds per tier direction. Nothing turns them into a rate, nothing counts bytes, and no decision reads them. Device-to-device chunk evacuation (what a raise does today) is not timed at all.
- **Fire side:** nothing is keyed by rung. The runtime has the last fire's latency and cumulative sums; engines time bodies only under diagnostics, to stderr.
- **Both halves of the comparison would have to be built.** Raw material exists for the move half (planner counters) and for a crude fire half (latency, lanes and tokens are all in scope where a fire retires, but are only summed).
- **Where it would sit:** on O, the offload block of `RaiseFunding` (`runtime/src/pipeline/fire.rs:841-856 @O`) together with `ResidencyPlanner::offload_for_raise` (`runtime/src/planner/eviction.rs:616-707 @O`). See Q3.

## Q1. Move rates

### Between tiers (device/host/disk): counted, log-only

- `PlannerStats` (`runtime/src/planner/stats.rs:4-34 @T`) holds cumulative `AtomicU64`s:

| Direction | Count | Time |
|---|---|---|
| device to host, kv | `d2h_pages` (:22) | `d2h_copy_us` (:32) |
| host to device, kv | `h2d_pages` (:23) | `h2d_copy_us` (:33) |
| to disk | `disk_write_pages` (:24) | `disk_write_us` (:28) |
| from disk | `disk_read_pages` (:27) | `disk_read_us` (:29) |
| host to disk spill | `host_disk_pages` (:26) | none |
| recurrent-state slots | `d2h_rs_slots`, `h2d_rs_slots` (:30-31) | none |

- Units are pages (slots) and microseconds. There are no bytes, no rate, no moving average. A grep for `ewma` over `crates` at T finds nothing.
- Written by `record_tier_copy(&TierMove, elapsed)` (`runtime/src/planner/report.rs:194-220 @T`). Time is recorded for kv only; rs/role slots are count-only (:207-208).
- Call sites: spill (`runtime/src/planner/exec.rs:132-138 @T`), `MoveGuard::copy` (:384-407), persist (:830-838).
- The clock in `MoveGuard::copy` starts at submit (:388) and is read after `completion.wait()` (:395-402). It therefore includes time queued on the engine lane, and overlaps when several copies are submitted together. It is not pure copy time.
- Evict phase times (`drained`, `quiesced`, `prepared_at`, `copied`) are locals printed in a ptrace line (`exec.rs:598-607 @T`). Restore total is a ptrace line (:789-793).
- Readers: `planner/diagnostics.rs:175-185 @T` (`*_total`) and from there only `runtime/src/bootstrap.rs:557-582, 618-630 @T`, the `[planner-trace]` print gated by `PIE_CONTENTION_TRACE_MS`. No decision reads them.
- O adds per-process `host_pages`/`disk_pages` and phase milliseconds to an info log, "planner: a process is moved to the host" (`runtime/src/planner/exec.rs:552-559, 617-627 @O`). Log only.

### Device-to-device chunk evacuation: not measured

- `carry_raise` calls `mover.set_rung` then `mover.evacuate` with no clock (`runtime/src/store/shared/programs.rs:346-379 @T`). The raise log carries only `moved = evacuation.moves.len()` (`programs.rs:332-336`, `runtime/src/store/shared/target.rs:266-271 @T`).
- On the lane, `PoolOp::Rung` and `PoolOp::Evacuate` (`runtime/src/scheduler/lane.rs:966-968 @T`) run as Control work. `LaneCharge` adds them to `lane_control_us`, `lane_control_n`, `lane_control_max_us` (`lane.rs:42-55, 495 @T`) together with every other control item. They cannot be separated, and only the server status reads them (`runtime/src/server/handler.rs:293-307 @T`).

### Engine side: nothing

- The contract's `set_rung`, `evacuate`, `copy_slots` return `Result<()>` and report no size or time (`engine/src/engine.rs:251-285 @T`).
- `crates/engine/src` has no timing code at all (grep at T).
- The engines' `Instant` uses are for other things: expert rebalance (`engine-cuda/src/ep/rebalance.rs:104, 179-183 @T`), Metal load prefault and resident-window copy under `tier-trace` (`engine-metal/src/mapping.rs:342-391`, `engine-metal/src/device/alloc.rs:125-161 @T`), tuning, and arming.
- Metal `copy_slots` commits asynchronously (`engine-metal/src/serve.rs:1735-1744 @T`); `evacuate` drains and commits synchronously (:1697-1712).

## Q2. Fire time per rung

No table of fire time by rung exists in `runtime`, `engine`, any engine crate or `profile`.

- **Per fire (runtime).** `record_fire_stats(stats, latency, batch_size, total_tokens)` (`runtime/src/scheduler/stats.rs:51-87 @T`) is called when a batch retires with `retired.started.elapsed()`, `retired.batch_size`, `retired.total_tokens` (`runtime/src/scheduler/worker.rs:2537-2542 @T`; `started` set at :2470).
  - Stored: `last_batch_latency_us` (overwritten), `cumulative_latency_us`, `total_batches`, `total_tokens_processed`, and a batch-size histogram (`stats.rs:24-35`). Per scheduler, not per rung or per token count.
  - Readers: PLEX worker attribute `step-ms` via `last_batch_us()` (`runtime/src/policy/service.rs:1231-1233`, `runtime/src/scheduler.rs:463-477 @T`), and the status handler (`server/handler.rs:92-97 @T`). The rung code reads none of it.
- **Lane time by kind.** `lane_launch_us/_n`, `lane_prefill_us/_n` are cumulative sums and counts (`scheduler/lane.rs:56-63, 496-497 @T`). Status only.
- **`engine_fire_us`** (`scheduler/lane.rs:529`, `scheduler/probe.rs:83 @T`) accumulates only with the `profile-fire` feature (`probe.rs:92-113`; `runtime/Cargo.toml:147 @T`).
- **Arming (runtime).** `Arming::step` times one `engine.arm_step` and uses it only for `owed_until` (`runtime/src/scheduler/arming.rs:336-371 @T`). It is capture time, not replay time, and is not stored.
- **`engine::live`** holds `armed`, `outside`, `held`, `tallies` and no times (`engine/src/live.rs:25-35 @T`). `armed_through()` (:110-112) is the only arming fact the rung reads (`runtime/src/store/shared.rs:273-275 @T`).
- **Arming (CUDA).** Under diagnostic `arm-bench`, 50 timed replays per armed body are printed as `us/fire` to stderr (`engine-cuda/src/serve/arming.rs:1835-1915 @T`). `arm-trace` prints warm+capture and golden milliseconds (:1813-1825, :1941-1954). Neither is stored. The on-disk bodies record holds bytes and body counts only (`engine-cuda/src/serve/measured.rs:11-33 @T`).
- **`crates/profile`.** `pie tune` times one 512-token prompt and 16 decode steps at 1, 4 and 8 lanes (`profile/src/search.rs:9-15, 27-31 @T`) to accept or reject a tuned profile (`Check`, :435-461, :489-510). It runs offline, on a fixed shape, not per rung.
- **F** adds `GuestTerm`, a table of bytes by lanes (`engine/src/program/term.rs:104-112 @F`). It carries no time. Its only timing is a doc figure for the table's build cost (:92-97).

## Q3. Where the rung is decided

### At T

- **Who asks.**
  - A prompt pass: `prefill_want`, `pass_asks`, `want_rows`, `raise_now` (`runtime/src/pipeline/fire.rs:3120-3134 @T`), then the widest piece (:3148-3167) and the process hold (:3171-3175).
  - The scheduler: lanes queued and in flight (`scheduler/worker.rs:1239-1240 @T`).
  - The walker: `walk_rows = wanted_rows(engine).max(live_lanes())` (`store/shared/target.rs:111-113 @T`).
- **Goal.** `rung_goal(pool, engine, rows) = pool.armed_rung_for(rows, armed_through)` (`store/shared.rs:273-275 @T`), clipped to `fit_rung()` in `RungWalk::step` (`shared.rs:377-382`; `fit_rung`/`fit_cap` at `store/shared/programs.rs:133-163 @T`).
- **Inputs to a raise.** Rows asked, widest armed rung, the fire terms' fit, `act_chunks(goal)` against `act_reserved()` (`shared.rs:397-398 @T`), and `plan_raise(goal)` (:424-430). No time or rate enters.
- **Patience** applies to lowering only: `PATIENCE = 2` governor ticks (`runtime/src/governor.rs:132`, `target.rs:132-145 @T`).
- **What makes a raise wait or be refused.**
  - The rung is not armed: funded through `arming::unfunded` (`fire.rs:795 @T`).
  - Not enough free chunks below the span: `plan_rung` answers `NoRoom` (`runtime/src/store/slots.rs:813-839 @T`), the walker sets `raise_blocked` until `room_freed` (`target.rs:156-187, 293-297 @T`).
  - The engine refuses the carve or the copy: the raise is taken back (`programs.rs:359-375 @T`).
  - `raise_now` raises at once only when nothing must move (`shared.rs:282-318 @T`).
  - `RaiseFunding` asks the planner for the lacking kv pages as a waiter (`fire.rs:745 ff`, `raise_short` at `shared.rs:1515-1530 @T`); the planner frees room by rank, then `take_raise` runs (`programs.rs:309-338 @T`).

### What O changes

- **Goal** widens to the widest armed rung of the same chunk class: `widest_alike` (`store/shared.rs:289-292, 1707-1717 @O`).
- **Scheduler ask** becomes lanes plus prompt rows summed (`scheduler/worker.rs:540-551, 1251-1255 @O`).
- **New offload step** in `RaiseFunding`, before the planner ask (`pipeline/fire.rs:841-856 @O`):
  - `asked = rows.max(wanted_rows(engine))`.
  - `pool.raise_offload(rung_goal(asked))` gives `RaiseOffload { spc, chunks }` (`shared.rs:352-365, 1552-1564 @O`), from `rung_unplaced`: the span's state chunks beyond the free chunks below (`store/slots.rs:856-865 @O`).
  - `planner.offload_for_raise(&keep, &offload)`, with `keep` = asking processes plus the carrier.
- **What `offload_for_raise` moves** (`planner/eviction.rs:616-707 @O`):
  - Every admitted, resident, non-holder-or-idle, unblocked process not in `keep` that has any kv page in those chunks (:629-636, :659).
  - Each is suspended whole: `Reclaim.pages` is all pages of its working sets, plus its rs units (:651-667). The amount moved is the owners' whole state, not just the unplaced chunks.
  - Order is map iteration order; rank is ignored ("whatever their rank", :607-615).
  - Destination is the ordinary suspend's choice, host first then disk (`store/kv.rs:2763-2765 @T`).
  - Bound: only the chunk set and `keep`. No cost test.
- **Restore.** No new code in the diff. The doc comment says processes come back "as room frees (the rung lowered or pages let go)" (`eviction.rs:611-613 @O`).

### L and F

- **L** leaves raises as they are. A waiter short of slots lowers the rung at once, through holds, without patience (`goal_rows` `store/shared/target.rs:129-148`, `forced … || short` :172, `fired_rows` `store/shared.rs:361 @L`). New slots go to lent chunks last (`store/slots.rs:737-755, 1254-1314 @L`).
- **F** changes what fits (bytes), not when to raise.

### Where a cost-vs-saving comparison would sit

1. **`RaiseFunding`'s offload block, `pipeline/fire.rs:841-856 @O`.** This is the only place an off-device move is started for a raise.
   - In scope: `rows` (the pass's ask, passed as `wanted`, `fire.rs:3184-3189 @T`), `asked`, `engine`, `pid`, `shared` (so `fire_cap`, the goal rung, `act_chunks`, chunk bytes), `planner`, the `RaiseOffload` chunk set.
   - Not in scope, would have to be passed in: the pass's total rows and its cut (`rows_total`, `cuts`, `cut_at` live at the call site, `fire.rs:3109-3146 @T`); the pages that will actually move (computed inside `offload_for_raise`); the tier each page lands on (known only at `prepare_suspend`, `exec.rs:552-559 @O`); any move rate; any fire time per rung (none exists).
2. **`ResidencyPlanner::offload_for_raise`, `planner/eviction.rs:616-707 @O`.** The cost half fits here.
   - In scope: the picked processes with their page counts (:645-672), `self.stats` (the Q1 counters), `self.bytes_of` (as used at `eviction.rs:183 @T`), host and disk free via `swap_stats` (`eviction.rs:165-176 @T`).
   - Would have to be passed in: the time the raise saves (rows, current and goal rung, fire times).
3. **`RungWalk::step`'s raise branch, `store/shared.rs:424-430 @T`**, called from `SharedGrow::walk_rung` (`target.rs:146-163 @T`). This is where a raise whose state fits below is decided, so it is the place for the device-to-device case.
   - In scope: `pool`, `goal`, `fire_cap`, the `RungEvacuation` (moves count), chunk bytes; engine id and `walk_rows` in the caller.
   - Nothing about time or rates is in scope.

## Q4. Which tiers exist

- **Stated by the engine** in `ChunkFacts.host_chunks` and `disk_chunks` (`engine/src/chunk.rs:80-91 @T`). The runtime reads them into `Tiers { host: Option<Shared>, disk_slots }` (`runtime/src/store/shared.rs:569-605 @T`). The planner is suspend-capable when either is non-zero (`runtime/src/bootstrap.rs:449-455, 522 @T`) and reads free and total per tier through `host_stats`/`disk_stats` (`runtime/src/planner/pool.rs:213-222 @T`).
- **Budgets** come from config: `budget_host`, `budget_disk`, `disk_path` (`worker/src/config.rs:970-973 @T`) into `Residency.host_kv_budget`, `disk_kv_budget`, `disk_kv_dir` (`engine/src/load.rs:109-121 @T`).
- **Per engine:**

| Engine | Host tier | Disk tier |
|---|---|---|
| CUDA | Seated by default when unstated, within free host memory (`engine-cuda/src/serve.rs:525-571`; facts at `engine-cuda/src/store.rs:3472-3473, 3645-3658 @T`) | From the seated slot file (same lines) |
| Metal | Only where a budget is stated: "The device pool already is host memory … suspended slots otherwise go straight to disk" (`engine-metal/src/api.rs:550-554`; facts `engine-metal/src/store/pools.rs:953-965 @T`) | Yes (`api.rs:555-562`) |
| engine-gpu (Vulkan, wgpu) | `host_kv_budget.map_or(0, …)`: none unless stated (`engine-gpu/src/serve.rs:694, 1239-1240 @T`) | Only with a directory, dropped if an unstated budget fails to seat (`serve.rs:1371-1398 @T`) |
| wgpu in a browser | As engine-gpu | Slot files in OPFS; "a tab it refuses serves with no disk tier" (`engine/src/disk/opfs.rs:1-3, 13-17`, `engine/src/disk.rs:1-4 @T`) |

  So a browser tab can have neither tier, but it is not always so.
- **A raise with nowhere to move state, at T.** `plan_rung` answers `NoRoom`, the walker waits (`slots.rs:827-832`, `target.rs:180-187 @T`). `RaiseFunding` asks the planner, whose eviction plan with `!suspend_capable() || swap_free == 0` notes the shortage and runs `check_starvation(NoSwapRoom)` (`planner/eviction.rs:200-205 @T`). That path salvages, serves from hoard, or fails or restarts a victim (`planner/starvation.rs:80-130 @T`). The raise waits or is refused (`fire.rs:902-907 @O`, same code at T); fires stay cut to the rung in hand.
- **Same case on O.** `offload_for_raise` does not test for a tier before beginning evictions. A suspend with no room answers `HostSwapFull` (`store/kv.rs:2391-2397 @T`), the process is marked `host_swap_blocked` (`planner/exec.rs:510-513 @T`) and is skipped the next time (`eviction.rs:634 @O`). The call then answers 0 and the code falls to the planner ask by rank, as the comment says (`fire.rs:839-840, 857-865 @O`).

## Q5. Fires are cut

- **Scheduler step:** `stepped.max_forward_tokens = min(cap)` with `cap = fire_cap(engine)` (`scheduler/worker.rs:1241-1243 @T`).
- **Prompt pass:** `prefill_share` is bounded by `fire_cap` (`pipeline/fire.rs:695-709 @T`). `pass_pieces(…, prefill_share(…), …)` (:3136-3144, :1117-1169) cuts through `host_cuts_from` (:958-1007).
- **The cap itself** is set by the walker's Cap step, logged "fires cut to the activation rung" (`store/shared/target.rs:200-218 @T`).
- **Cost in fires.** The share loop cuts while `rows - from > share`, page-aligned (`fire.rs:984-994 @T`), so about `ceil(rows / share)` pieces. With several lanes the share is divided by lanes and floored to a page (:1146). While a raise is pending, `PIECES_AHEAD = 2` pieces run ahead and a lone lane's rest is cut again at the raised cap (:736, :3259-3306).
- **Fixed vs per-token time: not measured anywhere.** Each retiring fire has latency, lanes and tokens in scope together (`scheduler/worker.rs:2537-2542 @T`), but they are stored as last value and sums (`scheduler/stats.rs:57-86 @T`). Only global means are derivable (latency per batch, tokens per batch). CUDA's `arm-bench` `us/fire` per body is diagnostic stderr only (`engine-cuda/src/serve/arming.rs:1881-1913 @T`).

## Not verified

- Line numbers for `planner/stats.rs`, `report.rs`, `exec.rs`, `diagnostics.rs`, `scheduler/dispatch.rs`, and the T-side `shared.rs`, `slots.rs`, `target.rs`, `fire.rs:745-1169` were read earlier in this study and not re-read at the end.
- What the planner's copy clock really covers per engine. Metal's `copy_slots` commits asynchronously; whether `completion.wait()` waits for the copy to land was not traced. The same doubt applies to `retired.started.elapsed()` as device time when several frames are in flight, and to lane time on engines that settle asynchronously.
- Whether `disk_write_us`/`disk_read_us` cover the same window as the copy clock.
- How much a whole-process suspend really moves: `prepare_suspend` takes `movable_resident_pages` (`store/kv.rs:2381 @T`); pages shared with kept processes may stay. Not traced.
- The restore after an offload on O (order, who returns first, whether a returning process can land back in lent chunks and be offloaded again). Only the doc comment was read; no test was read (`store/shared/tests.rs` on O, +78 lines, unread).
- The no-tier behaviour on O is inferred from the code paths, not from a test or a run.
- ROCm tier seating was assumed to be engine-cuda's; not checked separately.
- Whether a browser boot ever states `host_kv_budget`; the OPFS refusal was read from doc comments only.
- Vulkan's encode/submit/fence profiling credits (`engine-vulkan/src/encode.rs:208-225`, `engine-vulkan/src/device/ctx.rs:1734-1907 @T`) were seen in a grep, not read.
- Whether `pie tune` timings are written anywhere a serving process could read.
- `engine-xla` and `engine-cerebras` were not studied.
- L's `slots.rs` changes were read through grep hits, not in full; F's `programs.rs` changes likewise.
- Nothing was built or run.
