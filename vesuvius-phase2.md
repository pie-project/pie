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

## Known-inherent residuals (not Phase 2 targets)

- Frame-boundary membership: a fresh bind joins at the next epoch (~k
  waves), capping width under churn (~225/512 on 2048×32). Fades as
  epochs get cheap; deliberate phase-offset fleet partitioning is the
  documented fallback if it still binds after M2/M3.
- One host gap per frame boundary (launch-time trust: frame f+1 launches
  after f completes) — by design; amortized k×.
