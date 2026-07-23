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
