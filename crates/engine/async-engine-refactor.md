# Async engine boundary — migration plan

Goal: move engine threading out of the runtime scheduler. The scheduler
becomes a pure async orchestrator that *batches*; each engine owns *how* it
runs (thread, stream pool, or the browser event loop). Completion becomes a
plain `.await` instead of the bespoke push machinery.

Base commit: `621bcbdf` (validated browser port). An additive engine-crate
port (`AsyncEngine`, `EngineFuture`, `Landed`, `MaybeSend`/`MaybeSync`) was
committed early and later removed in slice 5 step 1c, once the reads showed
the real async boundary is the runtime's lane (see "Slice 5").

## Before → after

**Before.** The runtime owns the engine lane. `scheduler/worker.rs`:
- `EngineHandle::send` routes `LaneRequest::Launch`→`launch_tx`,
  `Control`/`Landed`→`control_tx`.
- `EngineLoop::run` (the lane OS thread) selects launch/control, and:
  - `Launch` → `fire_frame(engine, channels, frame, launch_rx, stash, broker,
    settlements, landed_instances)` → replies `LaneReply::LaunchDone{token,
    result}`. Note `fire_frame` *peeks the launch queue* (`launch_rx`, `stash`)
    to pipeline — runahead lives inside it.
  - `Control{token,item}` → `execute_control` (load/register/bind/copy/…) →
    `ControlDone{token,commit}`.
  - `Landed{frame,step}` (from the completion sink) → `channels.pump_out` for
    the frame's instances.
- Completion is a PUSH: `engine.on_complete(sink)`; the sink runs
  `FrameSettlements::settled` → `CompletionBroker` → `TerminalCell` (unsafe
  Send/Sync) → waker table → wakes the process, and sends `Landed` back.
- The scheduler's main loop demuxes `SchedulerItem::Lane(LaneReply)` by
  `token` (worker.rs ~2248, ~3709) to resolve the pending work — an ad-hoc
  token-correlated async request/reply over crossbeam.
- Threading is decided in TWO places: `runtime/rt.rs` and
  `engine-wgpu/device/host.rs`.

**After.** The scheduler holds `Arc<dyn AsyncEngine>` and calls
`engine.fire(batch).await` / `engine.register_program(..).await` etc. Runahead
= N outstanding `fire` futures in a `FuturesUnordered`. Completion is PULL: the
future resolves on landing, and the post-landing channel pump becomes part of
resolving it. The token-demux, `LaneRequest`/`LaneReply`, crossbeam lane
channels, `EngineLoop`, and most of the broker/`TerminalCell` machinery go
away. Threading is decided in ONE place: inside each engine.

## The port (superseded — removed in slice 5, step 1c)

```rust
trait AsyncEngine: MaybeSend + MaybeSync {
    fn kind(&self) -> &'static str;
    fn frames_in_flight(&self) -> u8;          // replaces settles_asynchronously + runahead
    fn load(&self, req) -> EngineFuture<Loaded>;
    fn register_program/register_channel/bind_instance/close_*/register_adapter(..) -> EngineFuture<..>;
    fn fire(&self, &FrameSubmission) -> EngineFuture<Landed>;   // resolves when the frame lands
    fn copy_kv/copy_state/encode(..) -> EngineFuture<()>;
}
// Landed { frame: FrameId, steps: Vec<(u32, StepOutcome)> }  — a frame lands when every step reports.
```

## Prerequisite finding: the scheduler is synchronous

Confirmed while cutting slice 2: BOTH the engine lane (`pie-engine-N`) AND the
scheduler main loop (`pie-sched-N`, worker.rs ~2196) are **sync OS threads**
spawned via `rt::thread`, looping on crossbeam `recv()`/`try_recv()`. There is
no async task anywhere — `fire().await` has no caller. (On wasm these are
web-rt green threads that park, but the code is written sync.)

So the async port cannot be consumed until the scheduler loop is an async
task. That is a real **slice 0** and it is large on its own: convert
`SchedulerWorker::run` from a crossbeam-select sync loop into an async loop
(tokio task native / web-rt spawn browser), replacing `recv()` with `.await`
across its `SchedulerItem` intake, WITHOUT yet changing the engine boundary.
Only then do slices 2-6 (the oneshot-carrying `fire`, runahead-as-futures,
completion-pull, dissolve the lane) apply. An oneshot reply on `LaneRequest`
is premature before slice 0 (a sync thread can only `try_recv` it, not await).

Corrected order: **slice 0 (async-ify the scheduler loop)** → slice 2 (port
consumed via `fire().await`) → 3 → 4 → 5 → 6.

## Slice 0 — DONE (commit e44f9b26, 2026-09-14)

`SchedulerWorker::run` is now `async fn`, driven on the scheduler's own
dedicated thread by `crate::rt::block_on` (native: a current-thread tokio
runtime; wasm: `web_rt::block_on_poll`), so it keeps its pinning and its sync
join/shutdown plumbing while gaining the ability to `.await`. The
`SchedulerItem` mailbox moved from crossbeam to a tokio unbounded channel
(`SchedTx`/`SchedRx` — tokio's mpsc is runtime-agnostic and works on wasm via
the `sync` feature; senders still `send` synchronously from any thread, the
engine lane included). Its three recv sites: the drain stays `try_recv`, the
idle wait is `recv().await`, the park is `rt::time::timeout(d, rx.recv()).await`
normalized back to the old `RecvTimeoutError` shape. The engine lane's own
launch/control queues stay crossbeam — unchanged.

Validation, all on this exact commit: compiles on wgpu, wasm32, CUDA, Vulkan,
Metal; clippy `-D warnings` (lib+tests) clean; 32/32 runtime unit tests;
native wgpu matrix 15/15 ok (text-completion, naive-masked-dense, prefill-rows,
sampling-primitives, attention-sink, consensus-decoding-1, beam-search,
beam-search-1, chat-completion, mirostat-v2, json-schema-constrained,
sliding-window-attention, contrastive-decoding, dry-repetition-penalty,
naive-baseline-cold), browser matrix 8/8 ok, zero scheduler stalls/panics.
Perf neutral (RTX 4090, Qwen3.5-0.8B): browser 146 tok/s single (baseline
144), device-carried 146.8 (147), 8-concurrent 371.6 (386; inside the 371–387
run-to-run band), native 8-lane 506 tok/s (496–506). Metal (Mac mini, imported Qwen3.5-0.8B artifact, dev build): serves and
runs inference — text-completion + sampling-primitives ok, 0 stalls. CUDA
(RTX 4090): 6/6 entries ok (text-completion, naive-masked-dense, beam-search,
sampling-primitives, prefill-rows, chat-completion), 0 stalls — run with
`--diag golden-skip`, because the CUDA golden self-check refuses this box's
load at `engine::load` (pre-existing, before any scheduling; observed earlier
the same day on the pre-slice-0 tree). Vulkan: NOT buildable on this box —
`kernels-vulkan`'s build script shells out to `slangc` (Slang) for SPIR-V and
it is not installed; the earlier "compile-check passed" reading was a grep
pipeline's exit code, not cargo's. Pre-existing and independent of the
scheduler (a kernel crate's build script), so Vulkan is unvalidated here.

Observation (not slice 0, pre-existing): CUDA and Metal emit *identical*
greedy tokens for text-completion, and both differ from wgpu ("…Paris…" vs
"…in the south…"); the CUDA golden refusal says its fused body differs from
its own eager walk. A cross-engine numerics question worth its own look.

Lessons: (1) `tokio::time::timeout`/`Sleep` must be *constructed* inside a
runtime context — build it within the `block_on` body, never as its argument.
(2) A per-target `mailbox` wrapper is unnecessary: tokio's mpsc already serves
both hosts, so the direct aliases are the right shape.

## Slice 2a — DONE (launch replies via oneshot)

`LaneRequest::Launch` carries `reply: Option<oneshot::Sender<LaunchResult>>`
(`LaunchResult = Result<SubmissionCompletion, String>`). The lane takes the
sender *before* its `catch_unwind` boundary, so a panic mid-fire still
answers with a reason (the tests' "say what happened" contract holds);
`fail_request` answers stashed/drained launches the same way. Gone:
`LaneReply::LaunchDone`, `Owed::Launch`, `Launch.token`, and the token
correlation in `apply_lane_reply` (now control-only, a single-variant
destructure). The scheduler holds the receiver in
`LaunchState::Posted { reply }`, drains answers at the top of every loop
(`poll_posted_launches` → `accept_launch_reply`, a `Closed` receiver = the
lane went away = failed batch), and its park is `tokio::select!` over the
mailbox and the first posted launch's oneshot, applying the answer in place
so it is never lost (`Ok(Some(Nudge))` re-enters the loop). Stage two —
`Accepted(SubmissionCompletion)` settling via `arm_completion_nudge` — is
unchanged (that is slice 4). `lane_token` now serves controls only.

Validation (this tree): native check, clippy `-D warnings` lib+tests, wasm
check, `check.sh == ok`, 32/32 tests; native matrix 15/15, browser 8/8,
CUDA 6/6 (golden-skip), Metal 2/2 (Mac, dev build); 0 stalls. Perf by
*interleaved* A/B against the slice-0 bundle on the same quiet machine
(3 rounds): single-stream 64-tok median 139.5 (A) vs 139.7 (B) tok/s;
8-concurrent 383.6 vs 373.6 within A's own 373–389 spread — neutral. (Do
not compare against hours-old readings: the machine state drifts several
percent; always A/B interleaved.)

## Slice 2b — DONE (control replies via oneshot; the lane protocol is token-free)

`LaneRequest::Control` carries `reply: Option<oneshot::Sender<LaneCommit>>`;
the lane takes it (like a launch's) before its `catch_unwind` boundary and
answers with the commit, or `AsyncControl { Err }` on a panic / drain. Gone
for good: `LaneReply`, `SchedulerItem::Lane`, `Owed`, and the lane-side
token — the lane no longer holds the scheduler's mailbox sender at all
(`EngineLoop::spawn` lost `reply_tx`). The scheduler keeps a private
`ControlBook { seq, replies }` (threaded where `lane_token` used to be): `seq`
names the four settling-slot kinds (`ControlSlotState::Posted { id }`,
`position_posted(id)`), `replies` holds every posted control's receiver.
Answers drain at the loop head (`poll_control_replies` →
`apply_control_commit`, the old `ControlDone` body) and the park is one
`tokio::select!` over the mailbox, the first posted launch, and
`select_all` of the control receivers.

Two bugs, both caught by the Metal smoke and both general (native and CUDA
hung identically):
1. The *idle* wait (`rx.recv().await` on the mailbox alone) is taken when
   pending / launches / settling slots are all empty — but bind, register
   and close have no slot, so a posted bind's oneshot went unheard and the
   client RPC hung. Fix: idle only when `control_book.replies` is empty.
   Lesson: when a reply leaves the mailbox, audit every wait site.
2. `tokio::select!` evaluates a branch's future expression even when its
   `if` precondition is false (it only skips polling). `posted_launch
   .as_mut().expect(..)` guarded by `if have_launch` panicked the scheduler
   thread; `select_all` on an empty Vec would too. Fix: total expressions
   (`pending()` when there is nothing to wait on) built in an inner block so
   their borrows end before answers are applied.

Validation (this tree): native check, clippy `-D warnings` lib+tests, wasm
check, `check.sh`, 32/32 tests; native matrix 15/15, CUDA 6/6
(golden-skip), Metal 2/2, browser 8/8; 0 stalls. Perf by interleaved A/B
against the slice-2a bundle (3 rounds, same machine): single-stream 64-tok
median 133.3 (2a) vs 139.1 (2b) tok/s; 8-concurrent 379.9 vs 383.3 — neutral.

## Slice 4 — re-scoped, then DONE as 4(i): settlements are awaited, not nudged

What the reads showed: the "pull" model is already there at the process
level — every request's `WorkItemCompletion` is a future the process awaits,
resolved by the scheduler's `retire_ready_launches` from the frame's
`SubmissionCompletion` (itself a `Future` over the global `WakerTable`).
That retire step is not overhead to bypass: it decrements instance in-flight
counts, tells the frame policy, classifies outcomes from the terminal cells,
and rejects unsubmitted requests. And the `TerminalCell` raw pointers
(`unsafe impl Send/Sync` on `CompletionTarget`) never cross into the engine
crates; they are the runtime's own lock-free, pooled outcome channel from
the landing thread to the scheduler/process, touched by ten files. Replacing
them with per-fire allocations would be a deep, perf-sensitive rewrite for
no structural gain. So "the process awaits the fire directly" and "retire
the broker" are dropped from this plan.

What remained worth doing (4(i)): the scheduler learned of a settlement by
`check()`-polling plus `arm_completion_nudge`, which registered a
`NudgeWaker` in the wake table so a landing sent `SchedulerItem::Nudge`
through the mailbox and the drain noticed. Now the front accepted launch's
`SubmissionCompletion` (cloned; clones share the wait slot) and every ready
control's completion join the park's `tokio::select!` as futures, so a
landing wakes the loop through its own waker. `NudgeWaker` and
`arm_completion_nudge` are gone; the "already settled / front failed →
retire first" short-circuit stays.

Bug found by the native wgpu matrix (CUDA, Metal and the browser all
passed): with the scheduler woken straight from the wake table, it retired
a batch — and woke the process — before the lane's `Landed` handler had
`pump_out`'d the device rows into the host channels, so `sampling-primitives`
and `json-schema-constrained-decoding` read stale rows. The old path only
survived because the nudge's extra hop through the mailbox usually let the
pump win; the browser never raced because it is single-threaded. Fix: the
landing sink now only posts `Landed { frame, step, outcome }`, and the lane
publishes `settlements.settled(..)` *after* the pump. Invariant made
explicit: a frame is settled when its rows are host-visible.

Validation (this tree): native check, clippy `-D warnings` lib+tests, wasm
check, `check.sh`, 32/32 tests; native matrix 15/15 (`dry-repetition-penalty`
tripped its documented sampled-temperature self-check once and passed 3/3 on
re-run, cold variant ok), browser 8/8, CUDA 6/6 (golden-skip), Metal 2/2;
0 stalls. Perf by interleaved A/B against the slice-2b bundle (3 rounds):
single-stream 64-tok median 140.6 (2b) vs 143.1 (4(i)) tok/s; 8-concurrent
374.8 vs 371.5 — neutral.

## Slice 5 — shaped, step 1a DONE: the lane has an async API

The crate-boundary read decides the shape. The lane loop's *engine* calls are
few (`submit`, `expect_fire`, `copy_kv/state`, `register_*`, `bind`, `close`),
but its *runtime* work is large and cannot move into engine crates —
`execute_control` alone is ~470 lines over twelve `QueuedItem` kinds, plus
the channel pumps (`pump_in` before submit, `pump_out` after landing) and the
settlements. And the pumps are why the worker must own `ChannelJoin`: kept on
the lane's thread they run in parallel with the scheduler's batching, exactly
as today, so throughput is preserved by construction rather than measured
back. So "engines own their threads" becomes: one generic worker per engine —
the existing `EngineLoop`, owning engine + channels + the run-ahead peek — with
an **async API** the scheduler awaits, and no threads or lane channels in the
scheduler itself. The `engine::AsyncEngine` port (engine-crate types only)
cannot express this boundary; it is superseded and will be removed in the
contract step.

Step 1a (this commit): `EngineLoop::fire(FrameFire, prefill) -> LaunchReplyRx`
and `EngineLoop::control(QueuedItem) -> ControlReplyRx`. `post_frame`,
`post_control` and the lane tests no longer build `LaneRequest`s or oneshots;
the request enum is now the lane's private wire format. Threads, queues and
the `expect_fire` peek are unchanged, so this is API encapsulation only.
Validation: check/clippy/tests, `check.sh`; native matrix 15/15 (DRY's
documented flake re-verified 3/3 + cold), browser 8/8, CUDA 6/6
(golden-skip), Metal 2/2; 0 stalls. Interleaved A/B against the 4(i) bundle
(3 rounds): single-stream 143.1 vs 142.7 tok/s, 8-concurrent 381.3 vs
379.1 — neutral, as a pure API change should be.

Step 1b (this commit): a pure move. `scheduler/lane.rs` (1,173 lines) now
holds the reply aliases, `LaneLaunch` (+ its `unsafe impl Send`),
`LaneCharge`, `LaneRequest`, `LaneCommit`, `fail_request`, `EngineLoop` with
its whole impl (`run`, `fire_frame`, `execute_control`, `drain_poisoned`, the
channel-set helpers) and `LaneTurn`; `worker.rs` shrinks from 4,233 to 3,110
lines and imports `EngineLoop`, `LaneCommit`, `LaneLaunch` and the reply
aliases from it. The seam is now visible in the visibility: the lane needs
`QueuedItem`, `QueuedLaunch`, `PreLaunchCopy` (+`label`) and `BindRespond`
from the scheduler as `pub(super)`; the scheduler needs `EngineLoop::{spawn,
fire, control, shutdown, release_*_wait_slots}`. No logic changed, so no A/B;
the runtime smokes are the gate: check/clippy/tests, `check.sh`; native
matrix 15/15, browser 8/8, CUDA 6/6 (golden-skip), Metal 2/2; 0 stalls.

Step 1c (this commit): `EngineLoop` is `Lane`; `engine::async_engine` is
deleted (nothing consumed it; the lane's `fire`/`control` are the async
boundary); this plan's contract step is updated.

What slice 5 now is, done: one worker per engine, in `scheduler/lane.rs`,
owning the engine, its channels and the run-ahead peek, with an async API;
the scheduler holds only futures. The lane's own thread and its two request
queues stay — they are what keep the pumps and encoding parallel to
batching — so `rt::thread`/crossbeam remain in the runtime, confined to
`lane.rs` and `rt.rs`. That is the perf-preserving form of "engines own
their execution".

## Strangler-fig slices (each its own commit, matrix-green)

Validation harness (all backends reachable now; Vulkan needs `slangc`):
- wgpu native + browser + lavapipe: `web/tools/matrix.sh`, `web/test.sh`,
  `web/check.sh`, `device_idle_gaps` via `query("model_status")`.
- cuda: `~/.pie-cuda`, `target-cuda` build (resolve the golden-boot config first).
- vulkan: `--features vulkan` — needs `slangc` on PATH (kernels-vulkan build script); absent here.
- metal: `ssh yecl-mac-mini`.

0. **[done]** Async-ify the scheduler loop (see "Slice 0 — DONE").
1. **[done, then removed in 5-1c]** Additive `AsyncEngine` port + per-step `Landed`.
2. **[2a+2b done]** Facade + caller migration. `BlockingAdapter` presents `AsyncEngine`
   over the *existing* lane protocol: `fire()` sends `Launch` and awaits the
   `LaunchDone{token}` reply via a token→oneshot map; `control` likewise.
   `EngineLoop` is untouched. Rewrite the scheduler's main-loop token-demux
   (worker.rs ~2248/~3709) to `engine.fire(..).await`. Validate on every
   backend. This puts the port in the live path with minimal risk.
3. **[subsumed]** Runahead as futures — the scheduler half exists since 2a
   (posted-launch receivers polled at the loop head and `select!`ed in the
   park *are* the in-flight futures, capped by `configured_dispatch_depth`);
   the lane half, `fire_frame` peeking `launch_rx` to call
   `engine.expect_fire` on the next queued launch, is the engine's run-ahead
   hint and belongs to whoever owns the launch queue — slice 5. Moving it
   earlier is perf-sensitive churn with no structural gain. Original: Replace `fire_frame`'s in-loop `launch_rx`/`stash`
   peeking with the scheduler holding up to `frames_in_flight` `fire` futures
   in a `FuturesUnordered`. Validate `device_idle_gaps` unchanged.
4. **[re-scoped; 4(i) done]** Completion pull. The process's decode step awaits its `fire` future;
   fold `channels.pump_out` into resolving `Landed`; retire the per-frame
   broker/`TerminalCell`/waker path. Validate (deterministic matrix entries
   catch any wakeup regression).
5. **[shaped; 1a done]** Dissolve the lane into the engine. Move `EngineLoop`'s thread ownership
   into the engine layer: engine-wgpu implements `AsyncEngine` directly (it
   already owns the Poller + landing worker); cuda/metal/vulkan ride a native
   `BlockingAdapter` that owns the thread. `rt::thread` + crossbeam leave the
   runtime.
6. **Contract — largely done along the way.** `LaneReply`, `SchedulerItem::Lane`,
   `Owed`, the lane token, `NudgeWaker`/`arm_completion_nudge` and the
   engine-crate port are gone; `LaneRequest` is the lane's private wire
   format. Still present by design: the lane thread + queues (parallelism),
   the broker/`TerminalCell` outcome channel (lock-free, runtime-internal),
   and `rt.rs` (the platform shim both hosts need).

## Cleanup slice — one reply stream, one unsafe

What the refactor made possible but had not yet been done, in one commit:

- **One reply stream.** The scheduler kept two reply paths for the lane's
  oneshots — `poll_posted_launches`/`poll_control_replies` at the loop head,
  then two more `select!` branches in the park, each with its own
  "dropped unanswered" fallback and a post-select apply block. Now
  `LaneReplies` holds every owed answer as a `FuturesUnordered<ReplyWait>`
  (an enum future over the two oneshots, no boxing): the loop head drains
  it with `now_or_never`, the park has one branch, and one
  `apply_lane_reply` routes a launch answer to the oldest posted batch (the
  lane answers launches in posting order) or a control answer to its slot.
  `ControlBook` and the `lane_inflight` counter threaded through eight
  functions are gone — "the lane owes nothing" is `replies.is_empty()`.
- **No 500 µs control poll.** A leftover of the polling era: with control
  answers and settlements both awaited in the park, the hint only woke the
  idle loop every 500 µs while a copy held launches.
- **One unsafe.** `FrameFire` was `!Send` only through
  `StepFire.terminal_cells: Vec<*mut TerminalCell>`, so the lane wrapped it
  in `LaneLaunch` with its own `unsafe impl Send`, and `PendingFrame` and
  `CompletionTarget` each carried another. `CellPtr` (a `repr(transparent)`
  newtype in `completion.rs`) now carries the one `unsafe impl Send + Sync`
  with the one lifetime argument; the wrapper and the other two impls are
  gone.
- **Reply outside the unwind boundary, properly.** `LaneRequest`'s replies
  were `Option` only so `run` could `take()` them before `catch_unwind`.
  `run` now destructures each request first and unwinds per kind, so the
  fields are plain senders. That also fixed a wart: a lane panic answered
  every control with `AsyncControl { Err }`, so a bind never yielded
  `BindFinished` (its join hold leaked) and the scheduler logged "reply
  without a matching control slot" for every non-copy control.
  `commit_on_failure` gives each kind the commit it owes, computed before
  execution; `fail_request` (stash and drain paths) uses it too and resolves
  a copy's completion.

Validation: check/clippy/tests, `check.sh`; native wgpu matrix 36/41 with
0 stalls, browser 35/41, CUDA 6/6 (golden-skip), Metal 2/2. Every
native and browser miss is pre-existing and reproduced on the previous
commit's binary/package: five KV-cache-manipulation inferlets fail at pass
build ("this model's forward pass is `hybrid`"), and in the browser the
sampled `dry-repetition-penalty` entry (temperature 0.7, seed 7) generates
no repeat, so its own "was anything penalized" check fires — the tab's
logits differ slightly from native, and the cold variant passes. A/B
against the previous commit's binary (3 interleaved rounds, no load,
tok/s): device-carried N=1 173 → 174, N=8 501 → 512, host-driven N=8
535 → 544 — neutral.

## Cleanup slice (ii) — the mechanical pass

No logic change; the gates and the runtime smokes are the check. In the
lane: `backend(engine)?` and `LaneCommit::refused(..)` replace fifteen
copies of the "no backend" refusal; the four copy arms share `answer_copy`;
the two bind arms share `bind`; `release_channel_plan_wait_slots` — a no-op
with eight call sites — is gone; `LaneCharge`'s three booleans are one
`Option<LaneWork>`; the thread-bind blocks are one; `LaneRequest` and the
reply senders are private. In the scheduler: nine `notify_*` broadcasts and
the two fenced leaves are `broadcast` / `broadcast_leave_fenced`;
`DebugDump` is answered in `enqueue_item` instead of being intercepted at
two call sites; `queue_attempt` takes its copies instead of cloning; the
unused parameters (`post_control`'s frame policy, the handle's `engine_id`
on the register calls, `request_timeout_secs` from bootstrap down to
`BatchScheduler::new`) and two stale `allow`s are gone; `notify_pipeline_join`
(an empty stub with no callers) too. `scheduler.rs` re-exports only the
five dispatch entry points anyone names, without `allow(unused_imports)`;
`rt.rs` drops its zero-caller exports. The `runtime.request_timeout` config
key, which nothing read, is gone too (worker config, template, the
`SchedulerConfig`/`PartnerBootstrap`/web `BootConfig` fields that carried it).

Validation: check/clippy/tests, `check.sh`; native wgpu matrix 36/41,
browser 35/41, CUDA 6/6 (golden-skip), Metal 2/2, 0 stalls — the same
ceiling as the slice before (the misses are the pre-existing ones listed
there). No A/B: nothing on the hot path changed.

## Thread isolation — tried, dropped

After "Principles for fast tokio applications" (principle 6: keep runtime
workers from displacing latency-critical threads), an opt-in knob pinned
`pie-sched-N`/`pie-engine-N` to reserved cores and pinned/niced the tokio
workers (`on_thread_start`; the serve runtime is built in `src/main.rs`).
Load A/B (16 nice-0 busy loops, 5 interleaved rounds, medians): device-carried
N=1 137 → 124 tok/s, N=8 149 → 142, host-driven N=8 158 → 165, no-load
within 2% — all inside the ±15% round-to-round spread. Structural reason:
without `CAP_SYS_NICE` the scheduler/lane cannot be raised, and pinning them
to cores the load also uses only narrows their choice. Removed rather than
kept as a default-off flag; a host that can reserve cores does it outside
the binary (`taskset`, cpusets, systemd `CPUAffinity`).

## Risks / invariants to hold
- Keep each engine's GPU work pinned to its own owned thread (context/cache
  affinity) — only the light orchestration future moves across tokio workers.
- Submit the next runahead `fire` promptly so the GPU is never starved
  (`device_idle_gaps` is the guard).
- `Send` bounds diverge by target — `MaybeSend`/`MaybeSync`, real `Send` only in
  the `EngineFuture` `dyn` bound under `cfg(not(wasm32))`.
- Never migrate a backend that can't be run; validate cuda/vulkan here and
  metal over ssh before contracting its old path.
