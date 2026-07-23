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
