# Vesuvius Phase 2 — Driver Multi-Step Execution (working notes)

Date: 2026-07-22
Branch: `tasks/ptir-fusion/agents/golf`, on top of `ebf5066f` (Phase 1
wait-all frames complete; k>1 gated behind `PIE_FRAME_SIZE`).
Spec: `vesuvius-project.md` §12 Phase 2 — "the composed wave program runs k
steps enqueued ahead with device-resolved advance; per-wave publication with
frame-boundary settlement and completion; per-wave graph-class selection.
Gates: settle/completion host lines divided by ~k; 2,048×32 ≥ 25k tok/s;
Phase 1 correctness gates unchanged."

## What the driver already has (exploration synthesis)

The Phase 1 attribution ("the driver polls predecessor publication and can
only RETRY, so dependent waves cannot overlap") turned out to be true only
for one of three geometry classes. The load-bearing facts:

- **One submission = one wave = one `PieLaunchDesc`** over the frozen C ABI
  (`interface/driver/include/pie_driver_abi.h`), executed synchronously on
  the driver lane thread: `launch_impl` (`context.cpp:1861`) →
  `handle_fire_batch` (`compose.cpp:343`) = begin → resolve_descriptors →
  compose → forward (graph replay for pure decode) → finish (epilogue
  sampling + publication + `cudaLaunchHostFunc` completion callback).
- **Three geometry classes**, ACK'd at bind (`forward.rs:753-759`,
  `PIE_GEOMETRY_CLASS_*`): `Host` (host-derivable, geometry rides the
  wire), `DecodeEnvelope` (canonical 7-port single-token decode; host
  supplies a shape template, device supplies values), `DeviceGeometry`
  (Track B, beam-shaped `[B, P>1]` Pages — full device trace).
- **The template path** (`try_device_composed_template`,
  `dispatch.cu:5086-5261`): a batch whose non-Host lanes are all canonical
  DecodeEnvelope launches with **zero host observation of device state** —
  placeholder wire geometry, `enqueue_fixed_decode` /
  `enqueue_decode_envelopes` compose tokens/positions/kv-CSR on device from
  channel cells (`compose_fixed_decode`, `dispatch.cu:441-620`), the
  attention plan comes from the working-set envelope upper bound. Only the
  fallback path (Track B lanes, non-canonical envelopes) does the blocking
  cross-stream D2H readiness poll that RETRYs
  (`dispatch.cu:5498`).
- **Device-side ordering already exists**: `k_pull_validate_host_channels_batch`
  checks each channel ticket's `expected_head/expected_tail` against the
  live ring words at kernel-run time and clears the lane's `pass_commit` on
  mismatch (`channels.hpp:221-299`) — an uncommitted predecessor makes the
  successor self-invalidate on device (no publish, KV writes confined to
  its prepare-bound pages, terminal RETRY at settlement). This is exactly
  the §3.1 "trust the dataflow" gate, and the k=1 quorum path already runs
  dependent device-composed fires at depth 2–3 through it.
- Per-instance `CommitSnapshot` occurrences, pinned staging pools
  (`kUploadStagingDepth = kSchedulerMaxInFlight + 1`), and the settlement
  arena pool (8) are all sized for the existing depth-3 run-ahead.
- **Settlement cost lines** (`Dispatch::finish`, `dispatch.cu:3847-4230`):
  `settlement_enqueue_us` = the whole finish (commit-bump batch, per-lane
  publish assembly, D2H mirror copies, settle kernel, host-func enqueue);
  completion latency = stream-drain to `notify_runtime_callback`
  (`dispatch.cu:1514`), which writes terminal cells, wakes channel waiters
  (`runtime.notify` per endpoint), and returns the arena.
- **Prepare lease** (`prepare_launch`, `context.cpp:1745`): per-wave
  physical KV/state commit against the elastic pool; `AdmissionWatermark`
  (`worker.rs:701-753`) already skips repeat covered launches.

## Why Phase 1 frames serialized anyway

`FramePolicy::plan_dispatch` gated a frame's next wave on
`outstanding == 0` — a self-imposed engine-side barrier (one wave per frame
in flight), while the k=1 quorum path freely posts dependent fires at
`configured_max_in_flight()`. The per-wave host gap of ~0.4–1 ms × (k−1)
per epoch is the retire→plan→dispatch round trip that this gate forces.

## M1 — intra-frame wave chaining (landed this revision)

`PIE_FRAME_CHAIN` (static, default 1, clamp 1..=3 to match the
scheduler/driver run-ahead depth): up to that many consecutive waves of one
sealed frame may be posted-unretired at once. Depth 1 is byte-identical to
Phase 1. Mechanics (`scheduler/frame.rs`):

- `SealedFrame.wave_outstanding: Vec<usize>` tracks posted-unretired fires
  per wave; the open gate becomes `inflight_waves() < chain_depth`.
- Makeup replay is now **wave-ordered**: only the oldest makeup wave
  dispatches, and only once every earlier wave has drained — required
  because a RETRY mid-chain leaves later-wave fires of the same lane in
  flight (they bounce on the stale rings and replay in order behind it).
- No driver change: chained waves ride the template path; the device
  ticket gate orders them on-stream. A wave that would host-observe
  (fallback class) simply RETRYs and replays — correctness by the same
  makeup machinery, throughput unchanged for those waves.

Probe (CUDA, RTX 4090, k=16): chaining alone lifted 2048×32 16.6k →
18.4k (depth 3) and 16×1900 to +3% over k=1, with exact 4/4 greedy token
parity and zero retry storms (chained retries were *fewer* than the k=1
baseline's normal retry load). But instrumentation showed the dominant
residual was elsewhere — see M1b/M1c.

## M1b — fleet-scope strict rounds (cohort re-merge)

Instrumented widths exposed a second, bigger leak: **phase-offset
cohorts**. Busy-lane exclusion sealed a straggler cohort (bootstrap or a
late herd) into its own epochs, and the wait-all gate then waited for busy
lanes' *submissions* but sealed *without* them — so cohorts never
re-merged (c0/256 ran as a permanent 230/26 split, median wave width 29 vs
k=1's 256). That deviates from the operator's principle (wait for ALL
pipelines). Fix: a NEW round seals only after the executing round fully
drains; mid-round capacity partitions (lane-disjoint by construction)
still pipeline. Measured: c0/256 22.2k → 26.4–26.7k (−2% vs k=1),
512×512 19.27k (k=1 parity), 64×1536 7.32k (parity), 16×1900 3.69k (+3%).

## M1c — boundary gather hold

On the churn shape (2048×32) strict rounds alone at chain=1 gave 20.9k
(+26% over Phase 1), but chaining *lowered* it: faster epochs shrink the
wall-clock window in which the replacement herd (spawned on the dying
generation's completions) can bind before the boundary seal, so epochs
get narrower — slow epochs were accidentally buying width. Fix:
`PIE_FRAME_GATHER_US` — when members departed during the drained round,
hold the next seal a fixed window so the herd's binds land in the same
epoch (the bootstrap cold hold generalized to membership-shrink
boundaries; a static deployment constant, no runtime adaptation).

## M1d — the chained-churn crater: a driver template-path unsoundness

With merge+gather landed, chain=3 still cratered 2048×32 (12.7–16.9k,
thousands of per-fire RETRYs) while chain=1 was clean. Diagnostics
(bounded retry-lane endpoint dumps + an envelope-kill probe) showed the
retried lanes' ring state matched their tickets exactly — the kill was
`compose_decode_envelopes`' containment check firing with
`source_page_count = 1`: the **placeholder geometry** from
`try_device_composed_template`. The shortcut accepted MIXED batches
(Host-class chunk lanes + canonical envelope decodes), returned 1-page
placeholder spans, and set `device_composed = false` — which routes to
`enqueue_decode_envelopes`, the variant that TRUSTS host page spans. Every
envelope lane past its first page then containment-killed, cascading down
its chained successors (ring never advances → ticket mismatch → RETRY),
and the wave-ordered makeups replayed the epoch at ~half throughput.
Chain=1 never tripped it (its mixed batches happened to resolve through
the descriptor fallback). Fix: the template shortcut now applies only to
fully device-composed batches (`all_decode_envelopes`); mixed batches
always resolve real geometry through the fallback — which at wave 0 of a
strict round observes only drained state and never retries.

## M1e — geometry-homogeneous frame batches; where chaining stands

After the template fix, chained churn runs are correctness-clean (zero
retries, exact parity) but still slow: the mixed wave-0's descriptor
fallback synchronizes the compute stream. The frame dispatch path now
splits each wave's launches by geometry class (wire chunks vs
device-resolved decodes), so no frame batch ever takes the fallback —
yet chain=3 on 2048×32 stays ~14.5k vs chain=1's 21.4k with identical
width and zero retries. The remaining chain>1 × churn interaction
(round-partitioned epochs + deep chaining slow the stream even when
nothing bounces) is unexplained and parked as an M2 diagnosis item.

**Resulting deployment guidance (static constants, the page-size
pattern):** stable decode fleets deploy `PIE_FRAME_CHAIN=3` — c0/256
26.5k (−3% vs k=1), 512×512 19.3k (parity), 64×1536 7.3k (parity),
16×1900 3.69k (+3%). Churny short-output fleets deploy `PIE_FRAME_CHAIN=1
PIE_FRAME_GATHER_US=20000` — 2048×32 21.4–22.0k (+29–32% over Phase 1's
16.6k, −12% vs k=1's 24.5k).

## Scoped next milestones

- **M2 — frame-boundary settlement**: keep per-wave publication (mirror
  copies + doorbell words + channel-waiter wakes — interior tokens must
  stay host-visible one step after production for the reactive SDK loop),
  but batch the heavy settlement (terminal cells, scheduler retirement
  bookkeeping, `cuda_settled` reporting) once per frame tail. Divides the
  `settlement_enqueue`/completion host lines by ~k. Driver: split
  `notify_runtime_callback` into a light per-wave wake and a per-group
  settle; engine: retire the frame as a unit in `retire_ready_launches`.
  Needs a chain-group marker on `PieLaunchDesc` (new ABI field; both sides
  build from one tree; remote driver keeps per-wave semantics).
- **M3 — frame-level prepare**: one admission lease per sealed frame
  (union high-water of its k waves) at first dispatch instead of one per
  wave; the driver lease machinery (`prepare_launch`/`launch_prepared`)
  already supports held leases. Removes k−1 elastic-pool commits per frame
  and the per-wave watermark checks.
- **M4 — varlen graph class**: capture chunk-carrying waves as a
  num_tokens-bucketed variant on the same `ForwardGraphCache` (the key
  already has the field; needs `fixed_split_size` +
  `cudaGraphExecKernelNodeSetParams` per `forward.hpp:78-83`). Independent
  of M1–M3; only worth it if chunk-wave launch overhead shows up after
  they land.

## Validation of record (final battery, k=16 chain=3 gather=20 ms)

- Engine unit tests 373/373 (three new frame tests: chain depth, wave-
  ordered makeups, straggler re-merge; plus the gather-hold test).
- CUDA greedy oracles t ∈ {31, 32, 33, 256}: exact token parity vs the
  stored k=1 baseline dumps, 4/4.
- 2048-request fleets: 3 consecutive completions at chain=3 and 3+ at
  chain=1+gather (every probe/instr run completed; zero failures).
- Zero retried fires in the instrumented post-fix runs on both shapes;
  zero `frame_wait_watchdog` reports.
- k=1 quorum path: c0/256 27.3k (band 27.2–27.3k ✓). 2048×32 moved
  24.5k → 23.7–23.9k (−3%) — and the instrumented run shows why this is
  the fix, not a regression: k=1's habitual **3,588 retried fires per
  run dropped to 0** (172 → 165 waves). The template bug was silently
  containment-killing lanes in k=1's mixed waves all along, masked as
  "normal" retries; post-fix those waves take the descriptor fallback,
  whose compute-stream sync costs slightly more than the retry waste
  did. Porting the frame path's geometry-homogeneous batching to the
  quorum path (removing the sync) is the follow-up that should put k=1
  above its old band.

| Shape (k=16) | Phase 1 (wait-all) | Phase 2 best | config | k=1 |
|---|---:|---:|---|---:|
| 2048×32 c512 | 16.1–17.0k | **21.4–22.0k** | chain1+gather20 | 24.5k |
| c0/256 | 22.1k | **26.4–26.7k** | chain3 | 27.3k |
| 512×512 | 16.8k | **19.3k** | chain3 | 19.3k |
| 64×1536 | 6.9k | **7.3k** | chain3 | 7.3k |
| 16×1900 | 3.51k | **3.69k** | chain3 | 3.57k |

## Known-inherent residuals (not Phase 2 targets)

- Frame-boundary membership: a fresh bind joins at the next epoch (~k
  waves), capping width under churn (~225/512 on 2048×32). Fades as
  epochs get cheap; deliberate phase-offset fleet partitioning is the
  documented fallback if it still binds after M2/M3.
- One host gap per frame boundary (launch-time trust: frame f+1 launches
  after f completes) — by design; amortized k×.

## Overlap rework (operator directive — landed, commit 05d155b2)

The launch-time barrier is gone: FramePolicy seals early (the moment the
wait-all gate holds, during the prior frame's execution), consumes every
ready lane's front frame atomically (one frame per lane per boundary),
posts all waves in global seal×slot order at the run-ahead depth, and
replays RETRYs globally wave-ordered. Busy-lane exclusion, strict rounds,
`PIE_FRAME_CHAIN`, and `PIE_FRAME_GATHER_US` are deleted; k is the only
frame constant (keep it minimal — 2–3 typical). Every M1b/M1c/M1d width
pathology this document chronicles is subsumed: early wait-all sealing
re-merges stragglers by construction (a seal never excludes a busy lane).

Gates: units 372/372; oracles exact 18/18 (k∈{2,3} × t∈{1..256}); 3
consecutive 2048-fleets; zero retried fires; zero watchdog reports;
median wave width 512. Results (vs same-session k=1): 2048×32 k2
22.7–24.8k / k3 23.7k ≈ k1 23.7k (Phase 1's −33% erased); c0/256
27.34–27.36k > k1 27.20k; 512×512 19.4–19.6k, 64×1536 7.36k, 16×1900
3.70k — all ≥ k1 bands. Next (directed): unify the scheduler paths —
route k=1 through FramePolicy and delete WaitAllPolicy.

## Scheduler-path unification (operator directive — landed)

The k==1 → legacy `WaitAllPolicy` fork is gone: every deployment,
including the default `PIE_FRAME_SIZE=1`, runs `FramePolicy`
(scheduler/quorum.rs deleted). A 1-slot frame IS a wave: at k=1 the
worker synthesizes a per-fire stamp at admission (lane = the pipeline
scope, seq = the globally monotonic fire id) — synthesis happens only on
the accept path, so stamp coverage is exactly the old wait-set membership
and rejected fires never touch the policy. Untracked/prebuilt fires stay
unstamped riders. Folded into frame.rs: `configured_max_in_flight`
(PIE_SCHED_MAX_IN_FLIGHT), the PIE_SCHED_COLD_HOLD_US lever, the
structurally-full cold-hold bypass, and the quorum wave/cold-hold stats
counters (rider batches count too). Deleted outright: readiness credits,
`credit_published`, `quorum_generation`/`quorum_pid` (RV-20/W1 identity
generations), `PipelineEpoch`/`WaveDecision`, `departed_pipelines`, the
RETRY credit re-arm, the PreLaunchCopy retire-time credit handoff, the
preview/decide two-phase, and the lane `Release` path — frames key
everything by fire id plus explicit leave events, so the whole
credit-accounting class has no successor.

Three semantics were deliberately ported/repaired in the process:

- **Async-control launch barrier**: frame dispatch picks launches by id,
  so a fire enqueued behind a queued standalone copy / pool resize could
  overtake it (caught by `launch_then_control_then_launch_preserves_
  fifo_order` the first time the k=1 tests ran the frame path; latent at
  k>1 too). The dispatch scan now stops candidate eligibility at the
  first queued async control — quorum's composer rule.
- **Shutdown drain**: the frame path used to reject queued fires at
  stopping; quorum drained them. Stopping now bypasses the wait-all gate
  and posts every accepted fire in queue order (per-lane ticket order is
  queue order); only retrying fires reject at retirement.
- **Post-terminate ghost lanes**: a stamped straggler rejected after its
  process terminated no longer records a rejected-arrival — the record
  would resurrect the purged lane as a permanently-missing member (fixes
  a latent k>1 infinite hold).

Gates: units 351/351 (24 quorum tests superseded by frame analogs; the
WaitAllPolicy-specific credit tests rewritten as frame drop tests);
oracles 15/15 token-exact (k=1 vs pre-unification k=1 dumps at t∈{1,2,
15,16,17,31,32,33,256}; k=2/3 spot checks at t∈{16,32,256}); 3
consecutive k=1 2048-fleets; zero retried fires (k1 168 waves, k2 188
waves), zero watchdog. Perf (4090): k=1 2048×32 24.8–26.4k (median
25.7k) vs pre-unification 23.7k — the geometry-split batching now
applying at k=1 recovers the template-fix −3% and beats the historical
24.5k band; k=1 c0/256 27.5k (was 27.2k), 512×512 19.6k; k=2 sanity
23.0k / 27.4k in band. Cross-check vs pre-Vesuvius: charlie's canonical
c0 shape (256 req, unlimited concurrency, 128 tok) measures 32.6k on the
charlie build and 32.7k on this build same-day — zero cumulative
regression across Phase 1 + Phase 2 + overlap + unification. (The
34.27k in charlie's elastic-memory report does not reproduce on today's
environment with either build; its delta is environmental.)

## Attribution re-profile (post-unification, 2026-07-22)

Question (operator fork): is the remaining −20% on 2048×32 c512 (k=1
25.7k vs vLLM 32.2k) the host starving the GPU (→ M2/M3 first) or the
GPU busy but trapped in launch gaps (→ M4 first)?

Method: nsys 2026.1.3, `-t cuda -s none --cuda-graph-trace=graph`,
capture window = the measured section via the bench's
`--cuda-profiler-capture` (cudaProfilerApi range), on k=1 and k=2
2048×32; plus PIE_FIRE_TIMING aggregation of the unify-suite instr logs
(unprofiled, representative). Probe effect: profiled k=1 ran 23.2k
(−10%, structure valid); profiled k=2 collapsed to 8.5k — nsys's per-API
overhead cascades in the boundary machinery (251 gaps ≥ 1 ms vs k=1's
17), itself evidence of the k>1 path's host-latency sensitivity; k=2
conclusions below use the unprofiled instr log. Raw artifacts:
session scratchpad `attr/` (nsys-rep, sqlite, analysis scripts).

**Verdict: host-starved — GPU compute is busy only 61.7%** of the
measured window (graph replays 124 × p50 8.6 ms = 1,066 ms + individual
kernels Σ662 ms of a 2.80 s window; all-engine-activity 63.7%). M4 is
ruled out as the next move: the launch fan-out it targets is negligible
today — cudaLaunchKernel Σ31 ms host across 15.3k calls, sub-100 µs
compute gaps ≈ 80 ms ≈ 3.5% of wall. The idle decomposes three ways:

1. **Generation-boundary lifecycle storms — the #1 cost, ~31% of steady
   wall, outside every scoped milestone.** Three no-wave-in-flight
   windows of 171–270 ms (engine view; GPU fully silent 75–109 ms in
   each). All 2048 wasm processes spawn+instantiate up front (53.6 ms
   total, ~5-way parallel, NOT part of the stalls); the stall is the
   c512 admission cohort turning over: per boundary, 512 dying lanes
   post 9,216 `close_channel` + 1,024 `close_instance` controls
   (~60 ms of driver-lane occupancy) plus a ~3,000-call cudaFree storm
   (8–16 ms), and 512 fresh lanes post 1,024 `register_channels_bind`
   controls (94–133 ms occupancy). Controls execute one-at-a-time on
   the driver lane, launches FIFO-barrier behind queued controls, and
   the wait-all seal holds until the whole herd's first submissions
   land — `driver_bind` p50 is 79 ms of QUEUEING (the work itself is
   ~0.13 ms/control). Control occupancy alone accounts for 65–91% of
   each stall; run total 666 ms ≈ 29% of wall. Cross-check that this
   is the whole vLLM gap: charlie-canonical c0 (256 req, unlimited
   concurrency — no cohort turnover) runs 32.7k ≈ vLLM's 32.2k on the
   same hardware; the 2048×32 deficit is churn-boundary-shaped, and
   vLLM pays no per-request lifecycle.
2. **Per-wave submit/settle host lines — ~15% of wall, exactly M2+M3's
   target.** `driver_submit` Σ360 ms (p50 2.85 ms/wave), of which
   `settlement_enqueue` Σ222 ms = 65% of driver-lane host time
   (`finish_settle_prep` 121 ms and `finish_epilogue` 101 ms inside
   it); engine `retire_settle` Σ167 ms (7.3%). The lane thread is not
   saturated (duty 14.8%) — the cost lands as boundary cadence: the
   100 µs–1 ms compute-gap class (n≈4.0k, Σ659 ms ≈ 24% of wall,
   partially covered by upload staging copies) is these lines' GPU
   shadow.
3. **k=2 today changes nothing**: unprofiled coverage 74.0%, the same
   boundary stalls (n=20, Σ643 ms), `settlement_enqueue` Σ217 ms — NOT
   ÷k (188 waves vs k=1's 168, settlement still per-wave). Confirms
   M2+M3 is precisely the missing ÷k; until they land, k>1 has no
   mechanism by which to beat k=1.

Decision per the fork: **M2+M3 next** (host-starved side); M4 deferred
until chunk-wave launch overhead surfaces after M2/M3. Escalated to the
operator (observed, NOT directed): the lifecycle storm (item 1) is
twice M2+M3's direct target and no scoped milestone touches it.
Candidate mitigations to weigh: drain dead-lane `close_*` storms
asynchronously off the boundary path (they need not precede the next
seal), batch the 2-posts-per-bind registration into cohort-sized posts,
and/or relax the launch-vs-control FIFO barrier for controls belonging
to lanes outside the executing wait-set.
**Operator ruling (2026-07-22): M2+M3 first; the lifecycle storm is
addressed AFTER M2+M3 land.**

## M2 entry gate — §6.2 validation-completeness audit (passed)

Full report with the exhaustive per-class evidence table:
`vesuvius-m2-entry-audit.md`. Verdict: **§6.2 requirement 1 holds at
8184a5f1 — no strict GATE-BLOCKER.** Load-bearing facts M2's design rests
on: driver stream settlement (`notify_runtime_callback`) can only produce
per-lane SUCCESS or RETRY (`entry.poison` is never set anywhere — per-lane
FAILED at settlement is unreachable); every FAILED terminal is batch-scoped
and written synchronously in `launch_impl`'s catch blocks at post time;
every guest-recoverable per-lane class (geometry, staged cells, channel
capacity, page binding, RS, masks) is caught at guest submit or engine
admission. Three flagged constraints feed M2/M3's design:

1. **Makeup/RETRY handling is per-wave-retirement-driven** — under
   frame-tail retirement, RETRY discovery + budget clocks stretch ≤ k×;
   decide: keep commit-mirror-based per-wave retry visibility (the mirror
   already rides the kept publication copies) or rescale budgets.
   → resolved by the retry-auto-flush rule in the M2 design below.
2. **Device-only ring capacity has no §5 arm** — enforced only by the
   device publish-capacity gate as a mid-frame RETRY loop; prove vacuous
   from ticket pre-reservation or add a fourth `validate_frame` arm
   before M2 lands. → CLOSED: `validate_frame` now walks device-only
   unseeded rings in slot order and rejects a frame whose reserved
   backlog (`device_ring_backlog`) plus in-order net growth exceeds the
   declared capacity, mirroring the device gate
   (`tail-head < cap1-1 + same_fire_consume`, cap1 = capacity+1).
   Consumes not yet reserved by any accepted fire grant no relief
   (static, timing-independent — the S5 "2k-1" philosophy); seeded
   descriptor channels exempt.
3. **Per-wave prepare is a mid-frame admission point** — exactly the gap
   M3 closes (one lease per sealed frame at first dispatch).

## M2+M3 design (directive: M2+M3 first, lifecycle storm after)

Ground truth from driver/engine reconnaissance (paths:
`driver/cuda/src/pipeline/dispatch.cu`, `context.cpp`,
`pipeline/channels.hpp`, `runahead.hpp`; engine `scheduler/worker.rs`,
`scheduler/frame.rs`, `driver/completion.rs`, `driver/abi.rs`,
`interface/driver/src/local.rs`): `Dispatch::finish` enqueues per wave —
epilogue kernels, commit-bump batch, per-lane publish assembly, D2H
mirror copies (payload cells + doorbell words + the 4-byte `commit_host`
mirror), settle kernel, `publications_done` event (next wave's `begin`
waits it), `cudaLaunchHostFunc(notify_runtime_callback)`. The callback
latches per-lane `committed = *commit_host != 0` → SUCCESS/RETRY, wakes
endpoint waiters, writes terminal cells, fires the batch completion
notify, emits `cuda_settled`, releases instance-close fences, returns
the settlement arena (pool of 8; acquire can block the lane).
`PieLaunchDesc` is generated from `interface/driver/src/local.rs`
(`PIE_DRIVER_ABI_VERSION = 12`) via pie-driver-abi-cbindgen; `reserved0`
is validated must-be-zero. The remote path re-builds submissions from
`RemoteLaunch` (no desc field crosses the wire) → stays per-wave
automatically. Leases (`prepare_launch`/`launch_prepared`) are keyed by
id, validated `target_bytes[i] <=` component-wise, consumed by erasure
at the first `launch_prepared`, retired at a post-enqueue stream point;
lease floors guard elastic-pool trim. Depth gate today:
`in_flight_launches.len() >= configured_max_in_flight()` (2..=3);
driver `kSchedulerMaxInFlight=3`, `kUploadStagingDepth=4`.

**M2 — group-deferred settlement (what actually moves).** Per-wave
publication is untouched: epilogue, commit-bump, D2H copies, doorbells,
settle kernel, `publications_done`, endpoint wakes, AND the per-wave
outcome latch + terminal-cell writes (commit words are per-wave-volatile
— the next wave's `k_pull_validate` rewrites the same instance snapshot,
so outcomes MUST latch per wave; terminals are cheap host writes and
keep engine-visible truth fresh). Deferred to the group tail, per wave a
small record {completion (wait_id, epoch), `cuda_settled` payload,
instance-close fences}: the batch completion notify, `cuda_settled`
emission, fence release. Arenas return per-wave (unchanged pressure).

Drain protocol (stream- and order-agnostic, monotonic counters under a
tiny accumulator mutex touched only by host-func callbacks, the lane's
synchronous settle paths, and FlushSettle): every `finish` assigns a
wave seq and increments `expected`; every callback appends its record
and increments `arrived`; a defer=0 callback, ANY latched RETRY, a
synchronous settle (L1 whole-batch RETRY / L2 whole-batch FAILED /
launch reject), or an engine FlushSettle **arms** the drain with
`drain_up_to = expected`; whoever brings `arrived >= drain_up_to`
drains all records with seq <= drain_up_to in seq order. Rules:
- **retry-auto-flush**: a wave whose latched outcomes include any RETRY
  arms an immediate drain — RETRY discovery latency stays exactly
  today's, and the makeup gate can never deadlock against a deferred
  tail (audit §5.1 resolved).
- k=1, riders, makeups, shutdown-drain batches always post defer=0; a
  defer=0 callback with an empty accumulator takes today's settle path
  byte-for-byte (k=1 unchanged).
- Engine flush points (spurious flushes are harmless — they only settle
  early): stopping; frame truncate; lane leave/terminate; prepare
  IMPOSSIBLE/Err rejects; drops that empty a planned wave; plan Park
  while a deferred group is open.

Engine: `PieLaunchDesc.reserved0` → `settle_defer: u32` (0|1), ABI
version 12→13, validator relaxed, header + C++ regenerated from the one
tree. FramePolicy's dispatch plan gains the defer decision (wave j of a
sealed frame defers iff a later non-empty wave of that frame remains);
the worker tracks `deferred_open` and posts `LaneRequest::FlushSettle`
at the flush points. Depth gate becomes group-aware: count distinct
settle groups (defer=0 batches each count alone) so a k-wave group can
always post its tail (R1); staging beyond `kUploadStagingDepth` merely
re-serializes uploads. `retire_ready_launches` then pops the whole
group in one pass (one retire→plan→dispatch round trip per frame
instead of per wave — the measured 0.4–1 ms × (k−1) cadence line).

**M3 — frame-level prepare.** At the first dispatch of a sealed frame
the worker computes the frame-union demand (component-wise max of
`AdmissionWatermark::demand` over ALL the frame's queued fires — the
fields are high-water marks, so union = max) and runs ONE
`driver_lane.prepare`; the lease is held: a `launch_prepared` whose desc
says `settle_defer=1` validates against the lease WITHOUT consuming it;
the tail (defer=0) consumes it and schedules today's stream-point
retirement, so lease floors protect the elastic pool for the whole
frame (R14). FlushSettle also retires a held lease (truncation).
Later waves skip prepare entirely — no per-wave demand scan, no
per-wave lease-less `commit_cuda_arena_targets_atomically`. Frame-union
EXHAUSTED/IMPOSSIBLE now surfaces before wave 0 posts — §6.1's
"reservation covers all k steps" made literal, closing audit §5.2.
TP>1: prepare stays UNSUPPORTED → unchanged. Watermark `covers()` keeps
skipping redundant prepares across frames.

Gates for the pair: engine units green; k=1 oracles byte-exact vs
current dumps; k∈{2,3} oracles exact; 3 consecutive 2048-fleets; zero
watchdog; `PIE_FIRE_FORCE_RETRY` battery exercising retry-auto-flush;
then the decisive k=2/3 vs k=1 re-measurement (k>1 must now win).

## M2+M3 measured outcome (landed as substrate; deferral OFF by default)

Commits: d32c574f (attribution + audit + cleanup), 01ad4ba0 (ABI v13 +
driver protocol + §5 device-ring arm), 31965f0a (engine wiring),
c95f7e96 (lease-hold reverted pending frame leases), 9a3a6394 (gate
v3), then the `PIE_FRAME_SETTLE_DEFER` lever (default 0).

**Correctness: everything green with deferral ON.** Oracles 11/11
token-exact (k=1 byte-exact vs pre-M2 dumps at t∈{1,2,16,32,256};
k∈{2,3} exact at t∈{16,32,256}); 2048-fleets complete at k∈{1,2,3};
forced-RETRY battery (`PIE_CUDA_FORCE_RETRY_ONCE=1`) completes 64/64 at
every k with retried waves replaying through the auto-flush — no hangs;
settled-conservation holds (cuda_submit == cuda_settled: 168/168 k1,
188/188 k2, 200/200 k3); zero watchdog; zero organic retries.

**Performance: deferral is a net k>1 LOSS on this driver — the M2
premise did not survive measurement.** With deferral on, k2 c0/256
dropped 27.4k → 22.4–22.5k and k2 2048×32 23.0k → 20.2–20.9k under BOTH
gate variants tried:

- deep posting (group-counted gate, 6–9 batches in flight): the lane
  thread serializes on upload-staging slots (`kUploadStagingDepth = 4`;
  `h2d_prepare` Σ74→320 ms) and on the prior wave's `publications_done`
  wait inside `begin` (`begin_pull_validate` Σ5→180 ms) — the driver's
  own backpressure caps useful run-ahead at ~3 batches regardless;
- shallow posting (historical cap + open-group exception, gate v3):
  completion resolution is frame-granular, so the posting window
  advances in k-wave steps and the stream drains at every frame tail.

The root cause is structural, not a tuning miss: §6.2 forces per-wave
publication to stay, and the audit showed publication IS almost all of
`settlement_enqueue` — what deferral can legally move (terminal cells,
completion notify, `cuda_settled`, fences) is microseconds per wave,
while coupling retirement to the frame quantum costs the pipeline
percent-level throughput. The same holds for M3's target (`h2d_prepare`
Σ74 ms ≈ 3% at k2 before any regression). Post-unification, per-wave
settle/prepare host lines are simply no longer where k>1's deficit
lives; the attribution's item 1 (generation-boundary lifecycle storms,
~31% of wall — operator-queued AFTER M2/M3) and item 2's GPU shadow are.

**Disposition**: the full M2 machinery lands and stays correct — ABI
v13 `settle_defer`, the driver's park/drain/arm protocol with
retry-auto-flush, `pie_cuda_flush_settlement`, engine defer marking,
group-tail-safe gate — behind `PIE_FRAME_SETTLE_DEFER=1` (default 0 =
byte-identical pre-M2 behavior everywhere; the bit is advisory, so
non-CUDA and remote drivers are unaffected either way). It becomes
worth re-enabling only after the driver decouples its staging/
publication backpressure from the lane thread. M3's engine half
(frame-union lease) is NOT wired: with per-wave prepare already
watermark-elided and the commit cost at ~3%, it cannot pay before the
lifecycle storm is addressed; the driver-side hold semantics are
designed (held lease consumed by the tail, `release_launch` on
truncation) and documented above for when it is. The §5 device-ring
arm and the §6.2 audit stand regardless — they gate any future
frame-scoped settlement, and the audit's Q1 fact (stream settlement is
{SUCCESS, RETRY} only) plus the retry-auto-flush rule are what make the
parked machinery safe to ship dormant.

## Lifecycle storm, round 1 (operator directive — landed)

Target (attribution item 1): 512-lane generation boundaries stalled
171–270 ms ×3–4 per 2048×32 run (~31% of steady wall), ending 1–3 ms
after the LAST `register_channels_bind` control finished — 1,024
register controls serialized on the driver lane behind and between
~10.2k teardown controls, with `driver_bind` p50 79 ms of pure
queueing. Four coordinated changes:

1. **Close coalescing** (`QueuedItem::CloseChannels`): consecutive
   channel closes ride one lane round trip (per-id driver calls
   inside; bounded at 512/batch) — a cohort's ~9.2k close posts become
   ~0.5–2.5k.
2. **Binds overtake queued closes** (`queue_bind_control`): safe
   because a bind and a queued close always target different
   instances/channels (ids never reused; a close only posts after its
   own bind committed; a cell is only planned for registration while
   it has no endpoint).
3. **Boundary close hold** — the decisive one: while any bind is in
   assembly (`FramePolicy::has_pending_binds`, the same signal that
   holds the seal), teardown closes rotate WITHOUT dispatching and
   WITHOUT claiming progress (re-checked on the bind-completed wake);
   they drain during the next generation's execution. Queue
   reordering alone was insufficient — closes ARRIVE interleaved with
   binds and each worker pass flushes its controls into the lane FIFO
   (measured: 74.5 ms of close occupancy still inside the register
   span). Shutdown never holds.
4. **Driver registry sized for close-lag overlap**: slot reserve ×32 →
   ×48 (both generations' slots briefly live at once; a mid-ramp
   `grow()` costs a device-wide sync) and the inactive-storage byte
   caps load-scaled (`cap_slots_ × 8 KiB` vs the fixed 64 MiB cliff
   that fed the boundary cudaFree storm).

Measured (4090, k=1 2048×32 c512): stalls 853 → 653 ms; close
occupancy inside the register span 32–46 ms → **0.0 ms in every
stall**; first-register delay 76–88 ms → 5–26 ms; throughput
25.5–26.4k → **26.8–27.5k (median 27.4k, ≈85% of vLLM's 32.2k)**; k2
2048×32 23.0–23.7 → 24.3–24.9k; c0/256 27.4k and 512×512 19.2k
unchanged; oracles exact; zero watchdog; zero registry grows.

### Round 2 attempt: bind-ahead (two-stage admission) — machinery landed dormant

Directive: fundamental redesign, layer 1 — split admission into BIND
(driver bring-up: `ensure_bind_admitted` at `forward-pass.program` /
working-set creation, pool = execution limit + `PIE_BIND_AHEAD`) and
EXECUTION (the existing gate at submit), so the next cohort binds while
the current one executes. Three measured rounds:

- bind-ahead + binds still holding the seal: k2 2048×32 **25.8k (best
  ever)**, k1 26.1–26.5 (below round 1's 26.8–27.5);
- gating the pending-bind seal hold on execution admission (a
  bind-ahead process fires a generation later): k2 CRATERED to
  9.8–10.7k — the bind hold was doing boundary GATHERING work, and
  removing it fragments k>1 epochs (the M1b pathology);
- replacing it with an imminent-join hold (registered at
  exec-admission, cleared on first arrival): did NOT restore k2
  (10.7k) — and the timeline instrumentation showed why nothing helps
  k1 either: **exec admission completes at +5–28 ms into the boundary
  (prebinding works, admission waits p50 530 ms measured), yet waves
  start only at ~+170 ms — the boundary is bound by ENGINE-THREAD herd
  serialization**: 512 guests' submit paths plus the next cohort's 512
  bring-ups contend on the engine's task execution, ~150 ms per cohort
  regardless of where the driver-lane register storm sits.

Disposition: seal-hold gating and the imminent-join hold REVERTED
(every bind holds the seal again); the two-stage admission machinery
stays with `PIE_BIND_AHEAD=0` default (bind pool == execution pool ==
historical behavior; restoration verified — k1 26.2–27.1, k2 24.9,
c0/256 27.3–27.4, oracles exact). `holds_seal` still rides the bind
RPC for a future gathering mechanism that survives measurement.

**Round 3: worker-pass probe — the engine-thread hypothesis REFUTED,
the boundary named precisely.** A `worker_pass` fire-timing probe
(mailbox/retire/dispatch timing per pass, emitted >2 ms) shows: inside
a 146–199 ms boundary stall the worker's slow passes cover only
13–57 ms — the scheduler thread is mostly IDLE, not compute-bound. The
mailbox flood (10–12k items: closes + the herd's submits) drains at
~4 µs/item by +17–54 ms; every fire is queued by then. What the
scheduler waits on for the remaining ~120 ms is the SEAL: the NEXT
cohort's 1,024 binds sit in `pending` (`pending_len ≈ 1022`), each
holding the seal (`pending_binds`, the unconditional hold restored
after ba2/ba3), and they complete only as fast as the driver lane
grinds their register controls; the first wave posts 1–3 ms after the
last bind completes (dispatch pass at +156–202 ms observed directly).
The tokio runtime is multi-threaded (min(cores,64), same
`build_runtime` for the embedded engine) — guest-side herd work was
never the limiter; the earlier ~150 ms inference conflated ba3's
still-unexplained residual with engine work. (ba3's own +28→+170 ms
block with the hold gated remains UNRESOLVED — its k2 crater and this
residual should be re-diagnosed together if that path is revisited.)

Two candidate remedies, in order of confidence:
1. **Make the binds cheaper** (directly shortens the seal hold):
   register cohort batching (framing ~26 µs/control), a driver batch
   verb, registry reuse keying under close-lag (`cell_grew` 29.4k),
   `bind_instance` instance_us (~47 µs). Every µs off the 1,024-control
   grind is a µs off the boundary.
2. **Buffer next-generation binds across the boundary** (the precise
   fix): a `holds_seal=false` bind (process not exec-admitted) gets
   BUFFERED worker-side — neither seal-holding nor lane-occupying —
   until the boundary's first seal fires, then enqueued to grind during
   execution. Semantically the ba2/ba3 intent done without enqueueing;
   requires first explaining ba3's k2 crater and residual, so it lands
   only behind fresh instrumentation.

Remaining boundary structure (~140–175 ms per stall, now almost pure
register work): per control (p50 120–140 µs × 1,024) — ~59 µs channel
registrations (9× ~6.6 µs: init/mirror/words), ~47 µs
`bind_instance` (`instance_us`), ~26 µs FFI framing + entry hooks;
plus 5–26 ms herd bring-up before the first register. Known trade
recorded: the close hold defers slot retirement past the same
boundary's registrations, so `cell_grew` rose 19–20k → 29.4k per run
(~42 ms of fresh-alloc cost partially offsetting the win; steady-state
one-cohort-lag reuse should cover most of it and only partially does —
the registry's reuse keying is the follow-up). Candidate next levers,
by measured size: batched/cohort register controls (framing ~27
ms/boundary), registry reuse keying (~40 ms/run), `bind_instance`
instance_us (~48 ms/boundary), herd bring-up window.

## Bind-cheap round 1 (operator directive "bind 최적화" — landed)

Directive: attack remedy (1) — make the 1,024-bind boundary grind
cheaper. Instrumented profiling (ls1 log) named the payers per
`register_channels_bind` control (occupancy p50 104 µs / mean 113 µs,
Σ462 ms/run): `bind_instance` driver time 57 µs — dominated by
per-instance device allocations — plus ~48 µs engine-side loop, with
the 9 per-channel registrations' driver time ≈ 0 when slot storage is
reused.

Three driver-side changes landed (no engine changes, no ABI change):

1. **Consolidated baked-list buffer + pool** (`tier0_runner.hpp`).
   `Tier0Runner` used to pay ~8 cudaMalloc + synchronous-cudaMemcpy
   pairs per instance (`d_commit_` in the ctor; desc/per-stage/commit
   slot lists in `bake_static_lists` via `upload_slots`). All of it now
   packs into ONE device buffer per instance — commit word at offset 0,
   every list an offset — acquired from a size-keyed global
   `BakedBufferPool` (a cohort re-binds the same trace, so every
   acquire after the first cohort is an exact-size hit; pooled memory
   is deliberately never freed at exit). Upload is a single
   `cudaMemcpyAsync` on the registry's initialization stream from a
   persistent host staging member.
2. **No host sync on the bind path** (`channel_registry.hpp`,
   `program_runtime.hpp`, `dispatch.cu`). The `PtirInstance` ctor's
   `settle_seed_copies()` — flush + `cudaStreamSynchronize` per bind —
   is gone. The registry now stamps an `init_done_` event
   (`publish_initialization`), and every wave orders after it via
   `order_after_initialization(stream)` in `Dispatch::begin` (all
   production composition funnels there; `run_pass`/
   `launch_pass_async` carry the same edge for the standalone paths).
   Settling survives only where a FREE could race in-flight
   initialization-stream work: `release_slot_storage` and reused-slot
   growth in `init_slot`.
3. **Close-path fix — the O(N²) scan and the zombie-slot leak**
   (`dispatch.cu::close_channel`). Every close ran
   `any_of(instances)` × `find(channel_ids)` — ~9k comparisons per
   close at cohort teardown (36.9k closes/run) — and passed
   `defer_if_attached` INVERTED: a close racing its instance's teardown
   hard-failed ("still has instance attachments", 8–15k per run), the
   engine dropped it, and the slot NEVER retired. The leaked zombies
   starved the retained-storage pool, which is why steady-state
   registrations kept growing (`cell_grew` 19k, worsening to 31.7k once
   binds got faster). Now closes always defer-if-attached: the
   registry's per-slot refcount is exact and `release()` retires
   pending-close slots at zero. Errors 15k→0; close batches Σ362 ms →
   Σ5.8 ms; `cell_grew` steady-state blocks 9.2k/7.5k → ~2k.

Measured (bc3, RTX 4090, same battery): `cuda_bind` total 57 µs →
**12 µs**; control occupancy mean 113 → 91 µs. Throughput: k1 2048×32
**28.5–29.1k tok/s (median 28.8k, ≈89% of vLLM 32.2k;** round-1 band
was 26.8–27.5k**)**; k2 2048×32 **28.2k** (was 24.3–24.9k — k2's
denser boundaries benefit MORE); k3 27.6k; c0/256 k1 27.5k / k2 27.6k;
512×512 19.5k. Oracles t=32/t=256 byte-EXACT; watchdog 0.

Bench-harness note for the record: the 512×512 shape runs at
concurrency 256 (`--num-requests 512 --concurrency 256`). At c512 the
workload does not physically fit — its tail needs Σ ≈ 11.5k KV pages
against the 10,756-page elastic budget and prepare correctly reports
Impossible; an early bc1 suite ran that wrong shape and the "failure"
was the suite's, not the build's.

Remaining boundary cost after this round, by size: the engine-side
register/bind loop (~70–78 µs/control — 9 separate per-channel FFI
calls with per-call marshalling/validation, HashSet bookkeeping, and
LaneCommit plumbing), then the residual ~2k fresh allocations per
boundary from closes whose deferred retirement lands after the next
cohort's registrations. Next lever if the grind is attacked again: an
engine-side/ABI batch register verb.

## Bind-cheap round 2 (directive "1, 2, 3 전부 진행" — landed)

The three follow-up levers from round 1, resolved:

1. **Engine-side register/bind loop**: instrumented first
   (`engine_bind_breakdown` event: set/program/bind µs, permanent under
   PIE_FIRE_TIMING). Verdict: the "~70 µs engine residual" was a
   mis-attribution — most of it was the GROWTH allocation cost hiding
   inside `register_channel_set` (blocks 0/1 grow 9 slots × ~7.5 µs of
   cudaMalloc/cudaHostAlloc per control). With pooling (below), the
   real engine numbers are set 11.3 µs (9 channels ≈ 1.3 µs each),
   program ≈ 0 (engine-side program-id cache), bind 13.6 µs (11.4
   driver). A batch register verb's remaining upside is ~6–8
   ms/boundary — parked as not worth the ABI churn now.
2. **PIE_BIND_AHEAD re-sweep** (64/256/512 on the round-1 build):
   neutral to −0.7k on every setting, k1 and k2 alike. With binds this
   cheap the boundary grind is ~29 ms; pre-binding's per-wave seal
   interference at k=1 cancels what it saves. Default stays 0; the
   dormant machinery remains for a future where binds are expensive
   again.
3. **Slab-pool slot storage** (`SmallBlockPool`, the round's win): all
   per-slot storage (cells, staging, mirrors, word blocks) now
   recycles through registry-owned exact-size slab pools; slabs return
   to CUDA only at teardown. Registration performs NO CUDA allocation
   calls — including the cold ramp, which used to pay ~18.4k fresh
   allocation trios across blocks 0/1. This also deleted the two
   remaining settle sites on the retire path (release_slot_storage,
   reused-slot growth): recycled blocks stay inside the
   initialization-stream FIFO / `init_done_` ordering, so no
   free-under-copy hazard exists.

Measured (bc4): `register_channels_bind` occupancy mean 91 → **28 µs**
(Σ462 → Σ115 ms/run); growth driver time 7.5 → 0.2 µs; the boundary
grind ~120 → **~29 ms**. Throughput: k1 2048×32 **28.8–32.1k tok/s**
(instrumented run **32.5k** — at the vLLM 32.2k reference); k2
**31.5k**; k3 **31.1k**; c0/256 28.0k (was 27.4–27.6k); 512×512
19.3k. Oracles byte-EXACT; golden step-exec 69/69; engine units
351/351; watchdog 0. The k>1 shapes gained the most — their denser
boundaries were the most grind-bound — and k2/k3 now sit ABOVE every
pre-round k1 number.

Boundary accounting after both rounds: what was a 140–175 ms wait-all
stall per cohort turnover is now ~30–40 ms (grind ~29 ms + bring-up).
The remaining host gap to ideal is spread across the per-wave path
again, not concentrated in the boundary.

## Full re-profile toward +10% over vLLM (directive "다시 철저히 재프로파일"; target ≈35.4k)

Instrumented k1 2048x32 (32.5k) + nsys graph-mode (31.6k under probe,
representative) + nsys node-mode 1024x32 (32.4k, negligible probe).
Scripts in session scratchpad `reprof/` (wave_attr / submit_fields /
census / hole_census / boundary_anatomy / bind_micro; sqlites kept).

**Wall census (timed window ~2.015 s for 65,536 output tokens):**
- Steady decode: 124 waves x 512 fires, period p50 9.33 ms -> internal
  ~54.9k tok/s. The forward pass is ONE CUDA graph per wave, p50
  8.33 ms / p90 9.38 ms (155 launches, 1,127 ms total). Compute-only
  GPU idle across the whole run is just 12.1% (352.7 ms) and 322.8 ms
  of it is five >=1 ms holes: the four generation boundaries + startup
  edge. Steady-state micro-idle is ~30 ms TOTAL — the control plane's
  run-ahead (fires submitted ~2 waves ahead; dispatch starts ~4.5-5.3 ms
  before the previous wave's native_complete; the same two wave slots
  alternate) keeps the graph stream contiguous. Steady decode is
  GPU-BOUND; the earlier "remaining host gap is in the per-wave path"
  reading was wrong — the host path (driver submit host_total 2.53 ms =
  settlement_enqueue 1.39 + finish_epilogue 0.91 + begin 0.39) is fully
  hidden by pipelining.
- Node-mode kernel split: decode attention (BatchDecodeWithPagedKVCache)
  95 us/layer -> 2.7 ms/wave, near KV-bandwidth-bound; decode GEMMs
  ~3.5 ms/wave at ~60% MFU; glue (rmsnorm 3 us x 28x2, rope/qkv, swiglu)
  ~1 ms/wave; settlement/pull-validate/commit kernels ~14 ms per RUN
  (negligible). Kernel-side headroom is modest (glue fusion maybe
  5-10% of the graph), NOT the road to +10%.
- Prefill: prompt = 37 tokens; each generation prefills as 3 waves
  (221+221+70 fires, ~19k tokens) in ~130 ms, discrete kernel launches
  (~395/wave, not graphed) with big 507-us GEMMs; roughly
  compute-honest, minor upside.
- Generation-boundary holes: 45.7 / 77.4 / 51.0 / 80.6 ms (census;
  nsys sees the same four at 62.9-92.4 under probe) = ~255 ms, i.e.,
  ~12.5% of wall. THE dominant recoverable loss.

**Hole anatomy (hole_census + boundary_anatomy):** ALL 2048 processes
spawn at t~0 (the bench client pre-submits everything; concurrency is
enforced ENGINE-side by the admission chain). Ladder per boundary:
[~11-15 ms exit cascade: last settle -> guests finish -> closes ->
permits release] -> [prewarm admit + instantiate, instant] ->
[guest bring-up + bind-queue lag, up to ~20 ms jitter] -> [bind grind:
1,024 register_channels_bind controls x 27-32 us = 27-35 ms serial on
the driver lane] -> [512 exec admits + first fires + seal]. Then the
first prefill wave dispatches.

Why nothing overlaps the previous generation (and why the BIND_AHEAD
sweeps were neutral): MAX_PREWARM_PROCESSES=64, and the prewarm permit
is released only AFTER the bind permit is acquired — so the 64
next-cohort processes instantiate early, park on bind admission
(pool = 512 execution-tied permits, freed at exit) STILL HOLDING their
prewarm permits, and the conveyor clogs. Extra bind permits alone
don't fix it because pending binds hold the executing round's seal
(wait-all rule), so early binds just move the stall from the boundary
into mid-generation seals — measured neutral, exactly as ba5 found.
prewarm_wait mean 659 ms vs bind_admission_wait mean 45.9 ms: the
upstream gate is prewarm, not bind.

**Bind micro-profile (operator: "bind에 더 먹을 게 있다"):** per
control occupancy is now 29.8 us mean = engine channel-set 12.1 us +
bind RPC 14.4 us (driver: instance build 11.0 us; decode/topology/
event ~0) + framing 3.1 us. Slab pools hold: alloc/init/staging/
mirror/words all ~0 (sums 0-1.8 ms over 36,900 registrations).
Per boundary: 30.5 ms occupancy total. A batch register verb would now
save only the 3.1 us/ctl framing (~3 ms/boundary — DOWN from the
earlier 6-8 ms estimate; stays parked). The operator's hunch is right
but the meat is WHERE binds run, not what they cost: the entire 30 ms
grind + 15 ms exit cascade + 20 ms bring-up jitter sits inside a GPU
hole only because admission pins it there.

**Ranked path to >=35.4k (need wall <=1.851 s, i.e., -165 ms):**
1. L1 — overlap the cohort turnover (worth ~-200 ms; sufficient
   alone): (a) release the prewarm permit BEFORE parking on bind
   admission (+ possibly raise MAX_PREWARM), so the whole next cohort
   instantiates + binds during the previous generation's decode;
   (b) bind-ahead permits on by default (pool = limit + limit);
   (c) seal-hold scoping: pending binds hold the executing round's
   seal ONLY for exec-admitted processes; a parked (bind-ahead)
   process cannot fire, so its binds must not gate anyone. The unused
   holds_seal field on the bind RPC is the plumbing. The boundary
   still needs a hold while exec admission floods the new cohort in
   (else the first wave seals narrow — the M1b fragmentation trap):
   transfer the hold from "pending bind" to "staged cohort: bound but
   awaiting exec admission", released per process at first-fire
   enqueue. Guards: ba3's k2 crater came from un-scoped release;
   exec-admitted binds must keep holding. Verify k2/k3 + oracle + the
   409-ms-hole shapes (c0256, 512x512).
2. L2 (stretch, ~-30-60 ms): fuse decode glue (rmsnorm x56/graph,
   rope/qkv, swiglu) into neighbors; attention and GEMMs are already
   near their respective roofs.
3. Non-levers, measured: settlement machinery (14 ms GPU + hidden host),
   cuMemUnmap/graph churn (startup/shutdown only), batch register verb
   (~3 ms/boundary), prefill compute (honest), BIND_AHEAD alone
   (neutral without seal scoping).

## Staged cohorts + join-gathered seals (directive "한번 해보자" — landed)

The L1 lever from the re-profile, implemented WITHOUT heuristics as an
invariant correction (operator pushback on the earmark/knob framing was
right). One principle: **the execution cap governs execution; the seal
waits for joins that are actually imminent.** Engine-only; driver
untouched.

Mechanics (frame.rs / worker.rs / process.rs / preemption.rs):
- Prewarm permits release BEFORE parking on bind admission (a parked
  holder used to clog the 64-slot conveyor, pinning the whole ladder to
  the boundary).
- Bind pool = 2x the execution limit — executing cohort + ONE staged
  cohort, plain double-buffering. `PIE_BIND_AHEAD` env DELETED.
- "Pending binds hold the seal" DELETED wholesale (a live rebinder is
  already wait-set-held through its lane; an unadmitted process cannot
  fire). Replaced by imminent-join tracking in FramePolicy:
  `staged` (bind accepted, no first fire yet), `pending_slots` (free
  execution slots, event-sourced: seeded with the pool capacity at
  bootstrap, +1 per release — mailed BEFORE the permit drops — and
  saturating -1 per admission; EVERY capped admission notifies), and
  `joins_in_flight` (admitted-but-unfired, identity-paired by process
  id). Seal holds while `joins_in_flight` is non-empty or a free slot
  has a staged taker. holds_seal deleted from the bind RPC.
- The retiring teardown announces the slot before dropping the permit;
  ensure_execution_admitted notifies after acquiring. First fire (keyed
  by OWNER process id, not stamp.lane) promotes staged->live; every
  leave path forgets staged/joining processes, so a dead successor can
  never wedge a seal.

Three bugs found by the suite on the way (each now a unit test):
1. Anonymous earmark counting deadlocked the 1x1 oracle (warmup B
   consumed A's slot before the release event landed; the earmark then
   waited on the unadmittable real request). -> identity pairing.
2. Id-space mismatch: promotion removed `stamp.lane` (pipeline id) from
   process-id-keyed sets -> joins never cleared. -> promote by owner.
3. Semaphore permit laundering: released permits blend into the free
   pool, so parked-only notification left a phantom positive balance
   that held the FIRST generation's every seal (~all requests timed
   out). -> every admission notifies + capacity-seeded balance.
   Capacity seeding also restored the initial-fleet gather: without it
   the first epoch sealed 24-wide and the lead-less lanes paced every
   gen-1 seal at the full commit roundtrip (+4.7 ms/wave, ~140 ms).

Result (suite v5, 4090): oracles t=32/256 byte-EXACT, units 355/355,
watchdog 0, zero failed requests on every shape. k1 2048x32
**32.58/32.80/32.63k** (median 32.63k, instr run 33.59k) vs vLLM 32.2k
— consistently ABOVE the reference with the old 3-4k run-to-run
variance collapsed to ~0.2k. k2 32.06k, k3 32.26k (both best-ever,
epoch gathering intact — no ba3 crater). c0256 28.0k/27.9k, 512x512
19.3k (unchanged; staged working sets don't disturb the KV arena).

Boundary anatomy after: the register grind is GONE from the hole
(controls now 0.8-5.7 ms per boundary, binds dispatch during the prior
generation exactly as designed). Remaining holes ~189 ms: initial
gather 49 ms + three turnovers 56/52/32 ms that are now
**exit-cascade-limited**: nothing re-admits until the last wave's
settlement publishes (~12 ms, matches finish_to_settle), then 512
teardown->permit->admit->fire chains trickle over ~30 ms. That cascade
(and the driver's 12 ms settle publication) is the next lever toward
the +10% target (~35.4k needs holes <=90 ms total); after it, decode
glue fusion (~30-60 ms) is the stretch.

## Exit cascade, round 1 (directive "다음 지렛대로 진행. 10% 갭 달성해야 함")

Instrumented the exit path per process (`guest_main_returned`,
`process_drop`, `process_teardown` events, PIE_FIRE_TIMING-gated) and
attributed the ~30-40 ms boundary release trickle exactly:

- Guest unwinding is NOT the cascade: all 512 guests return, drop their
  ctx (scratch rm ~2 us) and reach the teardown task by **+6 ms** after
  the last decode wave completes. `finalize` is ~0.
- The cascade WAS `notify_process_terminate`'s ack: the scheduler
  worker's mailbox drain ran `try_recv` until EMPTY, and the next-next
  cohort's bring-up flood (~14k items/boundary: bind + channel-register
  + close traffic) kept it non-empty for ~47 ms. Leave acks — and with
  them permit releases, admissions, first fires, and the seal — were
  hostage to the flood's tail, FIFO-ordered behind it.

Two landed invariant fixes (`worker.rs`, `frame.rs`):

1. **Epoch drain**: a worker pass consumes only the items queued when it
   began (`rx.len()` snapshot), then always runs retire + dispatch. The
   flood spills to later passes and overlaps prefill execution; the
   seal-opening events reach the policy at pass cadence. Acks fell from
   p50 17-27 ms to 4-13 ms.
2. **`releases_in_flight`** (+ `slotted` set): the live drain had been
   ACCIDENTALLY load-bearing — with seal checks now exposed between
   passes, the window between a slot holder's Terminate leave and its
   teardown's Released broadcast read `pending_slots == 0` and the seal
   closed on a partial cohort (ragged 464/48 split, run variance blew
   up to ~2k, one run at 30.5k). The completed invariant: a departing
   slot holder's release is IMMINENT — Terminate leave of a slotted pid
   arms the counter (idempotent per pid; the exit funnel sends two
   leaves), Released retires it, and the seal holds while
   `releases_in_flight > 0` with a staged successor. Every big wave is
   512-wide again. Permit lifecycle audited: set only at
   `admit_execution`, taken only at ctx Drop -> capped teardown, so an
   armed release always lands (suspend never touches it).

Negative result, reverted after measurement (run g): making the
teardown's terminate leave fire-and-forget. Per-producer FIFO does
preserve every ordering (leave < Released < successor's consume/fire),
and releases indeed all landed by +15 ms — but k1 REGRESSED ~3k tok/s
(28.9-30.6k): the awaited ack is boundary FLOW CONTROL. Pacing each
retirement behind a worker pass keeps the successor admissions — and
the next-next cohort's bind/close flood their permit drops unleash —
arriving in small epochs BEHIND the successors' first fires. Without
it the whole flood lands at once and the first prefill slipped from
+33 ms to +117 ms behind queued bring-up controls. Documented at the
call site; `notify_process_terminate` stays awaited.

Result (suite f, 4090): oracles t=32/256 byte-EXACT, units 357/357,
watchdog 0. k1 2048x32 32.36/32.75/33.26k (instr 33.25k), k2 32.57k,
k3 32.46k, c0256 28.0/27.9k, 512x512 19.30k. Boundary holes (first
prefill dispatch after last decode): 67.9 / 33.2 / 27.0 ms (v5:
56.3/52.3/31.5) — turnovers 2 and 3 now near the floor (release tail
~24-27 ms + 3-6 ms dispatch), boundary 0 still fat: its ack tail (p50
26 ms vs 4 ms) sits under the heaviest bring-up flood while the tail
half of the 2048-herd is still instantiating.

Next levers toward 35.4k: (1) shrink the flood itself — the parked
batch-register verb is now the lever with teeth (18 channel-register
control items per process x 512 collapse ~9x; it is what the ack tail
and boundary 0 are made of); (2) initial gather 49 ms (prewarm
conveyor); (3) decode glue fusion (~30-60 ms, stretch).

## Exit cascade, round 2: the leave tax (continuation of "10% 갭")

The mailbox census (per-variant count+time in the epoch drain, permanent
under PIE_FIRE_TIMING) named the surviving boundary cost precisely:
`PipelineLeave` items at **31-37 us each** — 1,200+ per boundary (two
Terminate leaves per exit: the process actor's fire-and-forget plus the
teardown's awaited one) summing to 35-47 ms, dwarfing everything else
(binds 7.6 us, closes ~0-4 us, launches 0.6 us).

Two mechanical fixes in `worker.rs`:
1. `reject_pipeline_queued` rebuilt the whole pending deque (moving
   every ~700 queued items) on EVERY leave — a pure no-op for natural
   completion. Now a one-field scan decides; only a real purge (external
   terminate with queued fires) pays the rebuild.
2. A duplicate Terminate for an already-terminated pid skips the whole
   arm and just sends the ack the teardown awaits (first leave did all
   the work; every step is idempotent-no-op for the second).

Effect: leave items 31-37 us -> **0.8-1.3 us**; boundary mailbox sums
35-60 ms -> 0-4 ms; releases all land by +15-18 ms; boundary holes
(last decode complete -> first prefill dispatch) 67.9/33.2/27.0 ->
**19.0/21.7/17.1 ms** — the exit cascade is now: guest exits by +6,
releases by +15, seal opens and prefill dispatches by +17-22.

Result (suite m, 4090): oracles t=32/256 byte-EXACT, units 357/357,
watchdog 0, zero failed requests. k1 2048x32 **35.07/34.62/35.10k
(median 35.07k = vLLM +8.9%)**, instr 34.67k. k2 **34.45k**, k3
33.66k, k2-c0256 28.03k — all best-ever. c0256 28.02k, 512x512 19.31k
unchanged. Remaining to +10% (35.42k): ~330 tok/s ≈ ~19 ms of wall —
the cold gather (~30-49 ms at fleet bring-up) is the next and last
boundary lever; decode glue fusion the stretch beyond it.

## Review round (operator directive: "hack/heuristic 금지, opus 리뷰")

An Opus reviewer audited ddfeae3b/e5808b8f/b0359aa4 against the
"invariants, not heuristics" bar. Verdict: the staged/departing scheme,
epoch drain, and leave-tax fixes are sound (explicitly verified,
including the fast-scan superset claim and multi-driver balance); three
findings required work, all landed:

1. The awaited terminate ack had been documented as "boundary flow
   control" — a timing claim. Investigation showed BOTH prior stories
   were wrong (launches dispatch by id, not queue position, so the
   run-g "flood ahead of launches" mechanism cannot be it). The ack's
   real structural role: a REFERENCE FENCE — after it resolves, every
   scheduler has purged the pid's queued work and cancelled its
   protected in-flight control, so teardown finalization and the
   resource drop run with no scheduler-side reference to the recycled
   pages. Reframed in code; measured numbers and run letters removed
   from load-bearing comments. The bind queue-insertion rule was also
   rewritten as a priority invariant (execution > bring-up > teardown;
   a queued launch never depends on a queued bind, since a fire exists
   only after its own bind completed).
2. "An armed release always lands" was false on the no-runtime
   `mem::forget` path. `releases_in_flight` (a counter) became
   `departing` (a ProcessId set, fully identity-paired):
   `ExecutionSlotReleased(pid)` resolves its own holder,
   `ExecutionSlotForfeited(pid)` resolves a departure whose permit
   leaked WITHOUT crediting the balance (the pool shrank; the
   accounting agrees). New regression test.
3. `terminated_processes` grew unbounded. The teardown now broadcasts
   `ProcessQuiesced(pid)` strictly last (it is the process's final
   producer), which retires the tombstone. Terminate-leave sources cut
   from three to two (the actor's duplicate deleted; the free-fn's
   early leave is guarded on registry delivery so a terminate aimed at
   a quiesced pid cannot mint an unretired tombstone).

Also: STAGED_COHORTS named constant replaces the bare 2x bind-pool
multiplier; host_reader scan note taken for the bind round.

Result (suite r, 4090): oracles EXACT, units 358/358, watchdog 0.
k1 2048x32 35.20/35.26/35.16k (median 35.20k = vLLM +9.3%), instr
34.92k; k2 34.44k, k3 34.14k, c0256 28.09k, 512x512 19.68k — every
shape at or above its previous best. Target re-set by the operator:
+15% (37.03k), runtime-only (kernels off-limits). Re-profile ledger:
GPU-idle pool ~144 ms = cold ~67 (bind grind ~50: 1,024+ rcb at 47.7us
cold vs 24 steady, instance build 20us) + boundaries 19.0/21.7/17.1 +
post-prefill tails ~8-10 + stream-dry 8.9. Next levers: template/clone
bind (driver host), boundary teardown/admission overlap, run-ahead
depth, prefill chunk unification.

## +15% runtime-only round 1: three measured negatives + the honest pool

Directive: target +15% (37.03k), runtime-only, kernels off-limits.

Measured this round (each one bench-verified, none landed):
- **Tier0 bake template** (trace-pure bake image computed at program
  registration, bind only remaps dense->slot): built, golden 69/69,
  instrumented run — instance_us cold UNCHANGED at ~21us. The bind
  cost is channel-VIEW registry binding (per-channel slot wiring),
  not the trace traversal. Reverted (dual path for a within-noise
  win fails the maintainability bar). Knowledge kept here.
- **Run-ahead depth 3** (PIE_SCHED_MAX_IN_FLIGHT=3): 34.81k vs 35.30k
  at depth 2, same build, same session — N9's depth-2 conclusion
  holds in the 35k regime. Default stays 2.
- **Steady-state epilogue**: GRAPH_TRACE decomposition — inter-graph
  gap p50 0.64 ms/wave, 73% of it REAL epilogue kernels (fused
  sampling 250us + settle 84us + pull_validate 55us + commit/
  readiness); true micro-idle ~23 ms/run. Steady is closed under the
  kernels-off-limits constraint.

Honest runtime-only pool at 35.2k (wall 1.862s): cold ~59 ms (bind
grind ~35-40: rcb 1024+ x 47-57us serialized on the lane; client
spawn ramp ~17: spawn #512 lands +17.5ms) + boundaries 3x17.5 = 52 ms
(current floor: guest exit +6, releases +9, seal+dispatch +3) +
post-prefill tails ~10 + stream-dry 9 (the price of depth 2 winning).
Two design rounds remain, both awaiting direction:
1. **Settlement-coupled slot release** ("final fire"): the guest
   declares its last submission (WIT addition); at that fire's
   settlement the engine releases the execution slot and drops the
   lane from the wait-set — the successor admits while the
   predecessor's guest unwinds. Boundaries 17.5 -> ~7-9 ms (-27 ms).
   KV pages still free at teardown; a tight-shape successor parks in
   the existing prepare-retry (allocation-credit) path, so overlap is
   safe-by-retry. Requires a WIT/API surface change.
2. **Cold bind surgery** (driver host): bulk channel-view binding /
   registry wiring for the 1k-bind cohort (-15 to -25 ms).
Ceiling estimate if BOTH land plus tails: ~36.3-36.6k = +13-14%.
+15% additionally needs the client spawn ramp and prefill-chunk
unification (planner capacity trade against the KV pool) to land —
possible only if everything does.

## Cold bind surgery (B) round 1: baked lists join the registry slabs

Directive: (A) rejected (no WIT change), (B) approved. Measured entry
point: `cuda_bind.instance_us` cold 21us vs steady 11us — the delta is
the BakedBufferPool's per-instance cudaMalloc on every cold-cohort
miss (the pool starts empty; 1,024 first-touch acquires each pay a
lane-thread malloc), exactly the class of cost the SmallBlockPool
slabs already removed for cell/mirror/word storage.

Change: with a shared registry bound, the tier-0 baked lists now come
from the registry's device slab pool (`acquire_device_block`); the
runner records the owner for release. The standalone/test path keeps
the process-wide size-keyed pool. One 1 MB slab now serves ~2k baked
buffers where the cold ramp paid ~1k cudaMallocs.

Measured: instance_us cold 21 -> 15 (p90 32 -> 20); cold hole 59.4 ->
53.4 ms. Steady unchanged. Well under the (B) estimate of -15..25 ms:
device init batching already exists (flush_pending_initializations),
per-register host cost is ~1.5us with no single fat item left — the
remaining cold rcb ~50us is genuinely distributed (instance build 15
+ 9 registers + framing/reply), so further cuts would be micro-trims
with poor complexity/benefit under the no-hacks bar.

Result (suite s, 4090): oracles EXACT, units 358/358, golden 69/69 x4
(one anomalous partial run traced to environmental interference — 17
of 69 executed; deterministic 69/69 on every controlled rerun),
watchdog 0. k1 35.54/35.01/34.92 (median 35.01, band-equal with the
r suite's 35.20 — the -6 ms cold trim is inside run noise on k1);
k2 34.69k, k3 34.33k, k2-c0256 28.04k, 512x512 19.77k, instr-k1
35.04k, instr-k2 34.47k — every one a new best (denser shapes see
the cold trim more).
