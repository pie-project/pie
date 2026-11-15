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
- What the planner's copy clock really covers per engine: now traced, see "Metal and the copy clock" below. The same doubt applies to `retired.started.elapsed()` as device time when several frames are in flight, and to lane time on engines that settle asynchronously.
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

## Metal and the copy clock

Follow-up to the unverified item above. Ref: `3e4e78300` (T) only, every citation below is `@T`. Read with `git show` / `git grep`; nothing checked out, built or run.

### Short answer

- **The runtime clock stops when the engine call returns, on every engine.** The lane turns the verb's `Ok(())` into a completion that is already ready. Nothing awaits an engine-side landing.
- **On Metal the tier moves and the evacuation are complete when the call returns.** Disk and host moves are host memcpy plus synchronous pread/pwrite; evacuation commits with `waitUntilCompleted`. Only a same-device `copy_slots` returns on enqueue, and that is not a planner tier move.
- **So the asynchronous commit does not distort Metal's tier clock.** What the clock adds to the copy is lane queue time and `drain()`, which waits for every frame in flight.
- **Metal is not the odd one.** CUDA/ROCm return on enqueue for device↔host and for evacuation; Vulkan/wgpu for evacuation; XLA for every write to the device. Each engine has at least one direction where a clock around the call measures an enqueue.
- **No engine reports bytes or time back.** No existing report has a field for either.

### How the runtime clock ends (all engines)

- Start: `Instant::now()` before `copy_tier_tracked` (`runtime/src/planner/exec.rs:388-389`). Every copy of a move is submitted before any is awaited (:386-393), so a later copy's window includes the earlier copies of the same move.
- The lane runs the verb and wraps its result: `verbs::settled(engine.copy_slots(&plan))` (`runtime/src/scheduler/lane.rs:942-952`), where `settled` maps `Ok(())` to `SubmissionCompletion::ready()` (`runtime/src/engine.rs:59-63`).
- The scheduler worker stores it as `ControlSlotState::Ready` (`scheduler/worker.rs:2722-2734`) and resolves the tracked completion on its next retire pass (:2562-2578).
- Stop: after `completion.wait()` (`planner/exec.rs:395-402`), then `record_tier_copy` (`planner/report.rs:194-220`).
- `settles_asynchronously` governs fire frames only (`lane.rs:727-771`); the copy path does not consult it.
- `set_rung` and `evacuate` are `PoolOp`s answered on a response channel (`lane.rs:960-968`); no clock is on them.

### 1. Metal: return of the call vs completion of the copy

| Move | Path | The call's return is | Completion known at |
|---|---|---|---|
| device → disk | `copy_slots` → `drain()` → `Pools::spill(to_disk)` | the copy, done: per slot `pool.read` into the stage then `file.write` | the return |
| disk → device | same, `spill(!to_disk)` | the copy, done: `file.read` then `pool.write` | the return |
| device ↔ host (budget stated) | `copy_slots` → `drain()` → `Pools::park` | the copy, done: one host memcpy a slot | the return |
| evacuation (device → device chunks) | `evacuate` → `drain()` → blits → `frame.commit()` | the copy, done: `waitUntilCompleted` | the return |
| same-device slot copy | `copy_slots` → blits → `commit_async(None)` | an enqueue | the next `drain()` |

- Entry: `engine-metal/src/serve.rs:1735-1744` (`copy_slots`), :1697-1712 (`evacuate`).
- Tier branches encode nothing into the frame: `engine-metal/src/store/pools.rs:1899-1912`; `spill` :1924-1968 (loop :1950-1966); `park` :983-1005. The frame committed afterwards (`serve.rs:1742`) is empty for them.
- Slot file IO is `write_all_at` / `read_exact_at` (`engine/src/disk.rs:70-78`), opened with `F_NOCACHE` on Apple (`engine/src/disk/native.rs:167-177`). No fsync on the path; `SlotFile::sync` exists (`disk.rs:59-62`) and `spill` does not call it.
- `commit()` waits: `engine-metal/src/device/ctx.rs:772-786`. `commit_async` returns a `Pending` after committing (:822-866).
- The async handle is `self.grafted` (`serve.rs:301`), waited only in `drain()` (:2496-2503), which first harvests every frame in flight (`harvest_one` → `pending.wait()`, :2566-2570).
- `drain()` is called by the next copy, evacuation, tier read/write and others (`serve.rs:1454, 1583, 1650, 1680, 1689, 1701, 1724, 1739, 1751, 1761, 2179, 2524, 3823`). A fire does not wait for it; it is ordered behind it on the queue.
- Host tier: seated only where a budget is stated (`engine-metal/src/api.rs:550-554`); otherwise suspended slots go straight to disk.
- `Engine::copy_kv` on Metal refuses any tier side (`api.rs:967-979`), so tier moves reach Metal only through `copy_slots`.

### 2. What `PlannerStats`' microseconds measure on Metal today

- **`disk_write_us`** (device → disk): lane queue wait, plus earlier copies of the same move, plus `drain()` (every frame in flight lands), plus memcpy and uncached pwrite per slot, plus the enqueue of an empty command buffer, plus the worker's retire and wake.
- **`disk_read_us`** (disk → device): the same with pread and a memcpy into the pool.
- **`d2h_copy_us` / `h2d_copy_us`** (host tier, only with a budget): the same with one memcpy a slot. The first touch of a tier chunk allocates and zero-fills the whole chunk (`store/pools.rs:613-619`).
- **rs and role slots**: counted, no time (`planner/report.rs:207-208`).
- **Evacuation and `set_rung`**: not on this clock.
- **Same-device slot copies**: never a `TierMove` (`scheduler/dispatch.rs:161-185` always has a tier side), so the one asynchronous commit never enters `PlannerStats`.

### 3. Where a clock would start and stop on Metal; a report to carry it

- **Disk and host moves.** The bytes move on the lane thread inside `pools.copy_slots` (`serve.rs:1741`), after `drain()` returns (:1739). A host clock from :1739's return to :1741's return covers the copy alone. No command buffer carries these bytes, so `GPUStartTime` / `GPUEndTime` do not apply. IO completion is the return of pwrite / pread.
- **Evacuation.** The blits are in the frame committed at `serve.rs:1704`. `Frame::commit_timed` already returns `GPUEndTime - GPUStartTime` for a frame (`device/ctx.rs:794-812`); the wall alternative is :1702-1704, after the drain at :1701.
- **Same-device slot copy.** The `Pending` in `self.grafted` has `gpu_span_us()` (`device/ctx.rs:928-939`), readable once `grafted.wait()` returns in `drain()` (`serve.rs:2500-2502`); today the handle is dropped there unread. `commit_async` also takes a completed handler (`ctx.rs:829-855`), which today carries only an error string.
- **Timestamps in use today.** `GPUStartTime` / `GPUEndTime` appear only at `device/ctx.rs:811-812, 936-937`. Consumers: the `device-done … gpu_us` fire trace (`serve.rs:2574-2581`), kernel profiling (`encode.rs:125-126`), `tune.rs:1227`. `kernelStartTime` / `kernelEndTime`: no use anywhere in `engine-metal` (grep, no hit).
- **Existing engine → runtime reports.** None has a bytes or a time field:
  - the verbs answer `Result<()>` (`engine/src/engine.rs:251, 259, 282`);
  - `CompletionSink` is per fire step, `(StepDone, StepOutcome)` (`engine.rs:17-42`);
  - `weight_uses()` is a polled answer of read counts (`engine.rs:299-306`, `engine/src/chunk/weight.rs:112-121`);
  - `engine::live::Live` is the one record an engine writes and the runtime reads without asking the lane: `armed`, `outside`, `held`, four event tallies (`engine/src/live.rs:20-35, 43-56`). CUDA and XLA keep one (`engine-cuda/src/serve/arming.rs:1571`, `serve/load.rs:1343`; `engine-xla/src/api.rs:158, 404`). Metal, Vulkan and wgpu do not override `live()`; the default builds a fresh `Live::armed()` per call (`engine.rs:75-77`).

### 4. The other engines

**CUDA.** The same three answers differ by direction.
- Device ↔ host returns on enqueue: the copy rides the swap stream, a `landed` event is kept in `self.swaps`, and the bytes reach the tier on a landing thread (`engine-cuda/src/store.rs:3857-3907`, `store/host.rs:288-382, 437-440`). A later fire naming those pages waits through `fence_swaps` (`store.rs:4202-4220`); `settle_swaps` is the host wait (:4223-4228).
- Device → disk is complete on return: per slot `copy_any` to the stage, `ctx::sync(stream)`, `file.write` (`store.rs:4015-4018`). The sync also waits for every fire on the stream.
- Disk → device: `file.read` then an enqueued `copy_any`; the last slot is left landing (`store.rs:4010-4013, 4020-4022`).
- Evacuation returns on enqueue: `copy_batch_d2d(stream, …)` then the table (`store/chunks.rs:720-749`, `store.rs:2369-2379`, `serve/stats.rs:146-150`).
- So `d2h_copy_us` / `h2d_copy_us` measure an enqueue on CUDA, and `disk_write_us` includes a stream sync. No timing around these paths was found in `engine-cuda`.

**ROCm.** The same crate and paths: `cudaMemcpyAsync` → `hipMemcpyAsync`, `cudaStreamSynchronize` → `hipStreamSynchronize` (`engine-cuda/src/device/hip.rs:340-349, 434-437`). Same answers as CUDA as far as read.

**Vulkan / wgpu** (`engine-gpu`).
- Tier moves are complete on return: `drain()` then `frame.commit()` (`engine-gpu/src/serve.rs:1433-1449`), and `commit` is submit plus wait on both (`engine-vulkan/src/device/ctx.rs:1708-1714`, `engine-wgpu/src/device/ctx.rs:964-970`). Host copies are blits in that frame for mapped host chunks, or host reads for heap ones (`engine-gpu/src/store.rs:1266-1300`).
- Same-device slot copies return on enqueue (`serve.rs:1443-1444`).
- Evacuation returns on enqueue (`serve.rs:1687`), unless weight caches exist: then `commit()` and `drain()` first (:1680-1685).
- Both have `commit_timed`, a host clock around submit and wait (`engine-vulkan/src/device/ctx.rs:1716-1724`, `engine-wgpu/src/device/ctx.rs:972-980`). No device timestamps seen.

**XLA.**
- Device → host or disk is complete on return: `copy_tier` reads every source slot into a host buffer first, the device through a raw readback (`engine-xla/src/serve.rs:1211-1279`, `store.rs:1591-1613`).
- Host or disk → device returns on enqueue: `write_slots` → `exec::maintain`, which runs with `wait = false` (`store.rs:1618-1645, 1698`, `exec.rs:65-72`); `Device::run` waits only when asked or profiled (`device.rs:495-504`).
- Evacuation is the same maintenance path (`store.rs:1481, 1489-1511`), and the pool is `Growth::Fixed`: "the runtime does not ask it of a `Growth::Fixed` pool" (`api.rs:789-793`, `store.rs:970`).

**Verdict.** Every engine needs an engine-side clock for at least one direction. Metal needs it least for tier moves (they are synchronous) and has device timestamps ready for the evacuation.

### 5. A probe move at load

- **The runtime's own quiet point** is the lane's idle hook: `next_request(…, || arming.step(&mut engine, true, !landed_instances.is_empty()))` (`runtime/src/scheduler/lane.rs:478-487`). It runs with no request on hand, and the last argument says whether frames are still in flight.
- **"Beside arming" exists only on CUDA/ROCm.** `Arming::step` runs only in `State::Arming` (`scheduler/arming.rs:329-331`), entered only when the engine's `Live` is not fully armed (:275-279). Only `engine-cuda` overrides `arm_step` (`engine-cuda/src/api.rs:561-567`, `serve/arming.rs:1571-1579`). On Metal, Vulkan, wgpu and XLA the lane's arming is `Done` from construction and never calls the engine.
- **CUDA/ROCm.** Each `arm_step` ends with `device.synchronize()` (`serve/arming.rs:1617`), so nothing is in flight when it returns. Inside `load`, load-time arming (`staged.finish()`, `api.rs:693`, `serve/load.rs:1640-1641`) runs before the tiers are seated (`api.rs:696-718`); the in-load point with tiers seated is `api.rs:718-725`.
- **Metal.** In `load`, after `seat_tier` and `seat_disk_kv` (`engine-metal/src/api.rs:554-562`) and before the facts are read (:573-583): no frame has been submitted.
- **Vulkan / wgpu.** In `load`, after `seat_disk_kv` (`engine-gpu/src/api.rs:474-481`) and before the facts (:489-490).
- **XLA.** In `load`, after `seat_tiers` (`engine-xla/src/api.rs:479-490`) and before `tiered_facts` (:491). The way back to the device is an enqueue there too (section 4).
- **What the arming lease lends** is free kv list slots and record slots (`engine/src/live.rs:157-170`), not a chunk and not tier slots.
- **A tier move is by slot of a shape, not by chunk** (`engine-metal/src/store/pools.rs:1899-1912`). Metal's slot files hold kv list and fixed record slots only (:1925-1933); CUDA's holds kv list slots only (`engine-cuda/src/store.rs:3839-3846`).
- **A cold probe pays first-touch costs.** Metal allocates and zero-fills a host tier chunk on its first write (`store/pools.rs:613-619`). CUDA pins host kv pages a chunk at a time on first touch (`engine-cuda/src/store.rs:4031-4037`).

### Not verified (this section)

- Whether the device slots a probe would name are backed at the in-load point. Metal's `spill` and `park` call `pool.admit` on the slot's chunk (`store/pools.rs:988-989, 1938-1941`); the runtime commits segments later through `PoolOp::CommitSegments` (`lane.rs:963`). `admit` and the segments committed at load were not read.
- Who allocates tier slot refs for a probe. The runtime's stores own them; that side was not traced for a move with no process.
- `engine-gpu`'s `spill` body (`store.rs:1456`) was not read; "complete on return" for its disk moves rests on the `drain()` and `commit()` around it.
- wgpu's `Pending::wait` in a browser (whether it can block) was not read; only the native poll was seen (`engine-wgpu/src/device/host.rs:188`).
- CUDA's `copy_batch` / `copy_any` bodies were not read; "enqueue" rests on their doc comments and on the events recorded after them.
- CUDA's `HostTier` landing thread: whether its ring-to-tier memcpy is counted anywhere was not checked.
- ROCm was read only as the HIP mapping of the two calls; no ROCm-specific tier code was looked for.
- XLA's `read_raw` was assumed to block (it returns bytes); `synchronous()` (`engine-xla/src/api.rs:571-573`) was not read.
- The worker's retire latency between the lane's reply and `completion.wait()` returning was not sized.
- Whether `drain()` on Metal is usually short at a tier move (how many frames are in flight then) cannot be read from code.
- Nothing was built or run; no timing was observed.
