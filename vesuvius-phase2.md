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
