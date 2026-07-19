# Vesuvius Phase 0/1 Report — Frame-Based Submission

Date: 2026-07-22
Branch: `tasks/ptir-fusion/agents/golf`
Baseline: `73e48f05` (Phase 0 measured there)
Implementation commits: `2253892a`, `f164dac1`, `1afc992d` + the wait-all
frame quorum rework (this revision)
Spec: the Project Vesuvius frame-based submission design (2026-07-22)

> **Revision note.** The first landed frame scheduler sealed epochs
> take-who's-there (arrival-complete lanes only, nothing waited). The
> operator corrected the principle: the scheduler's fundamental rule stays
> **quorum infinite wait-all (watchdog 1 s, report-only)** — what changes at
> k > 1 is only the unit being awaited: a FRAME per lane instead of a wave.
> The frame policy was reworked accordingly and re-measured; the
> take-who's-there numbers below are kept as the superseded reference.

## Phase 0 — re-baseline at the implementation baseline

RTX 4090, Qwen3-0.6B, CUDA TP1, chat-templated ~34-token prompts, greedy,
prefix caching off, `--ignore-eos --unique-prompts`. The spec's Section 1
numbers predate recent scheduler work and have moved substantially:

| Shape | Spec §1 figure | Phase 0 re-measure | vLLM reference |
|---|---:|---:|---:|
| 2048×32, active 512 | 19.08k tok/s | 21.8k cold, 24.6–25.2k warm (median ~24.7k) | **32.2k** (vLLM 0.16.0, n=3; spec said 30.81k) |
| 512×512, c256 | 18.51k | 19.13–19.48k | — |
| 64×1536 | 7.34k | 7.20–7.33k | — |
| 16×1900 | 3.72k | 3.57–3.58k | — |
| 512 req × 256 tok, c256 ("c0/256" here) | — | 27.17–27.30k | — |

Consequences:
- The Phase 2 "≥ 25k tok/s" gate is effectively met at baseline; the real
  target is vLLM parity at ~32.2k (current gap ≈ −23%).
- All Phase 1 comparisons below are against these re-measured numbers.

Raw logs/JSON for every run live in the session scratchpad, not the repo.

## Phase 1 — runtime-only frames: what landed

- **WIT**: `model.frame-size()` (k; env `PIE_FRAME_SIZE`, default 1, clamp
  1..=64 — a static deployment constant, the page-size pattern) and
  `model.max-embed-length()` (C; the driver's structural per-launch token
  cap). `forward-pass.submit` replaced by the interface-level
  `submit(on, slots)` frame call (exactly k ordered slots, slot i executes
  in wave i, `none` = no-op). Canonical + SDK + bakery mirrors are
  byte-identical.
- **SDK**: `pass.submit(on)` remains as single-slot-frame sugar — every
  existing inferlet compiles unchanged; `submit_frame(on, slots)`,
  `frame_size()`, `max_embed_length()` are the frame-aware surface.
- **Host validation (§5, enforced at k > 1)**: per-host-writer-channel
  classification into *staged* (ring-sequence supply ≥ frame consumption),
  *device-advanced* (program publishes an advance), or *latest-value*
  (read-only bound, one committed cell, `set` any time); static
  reader-capacity overflow prevention (worst-case ring occupancy ≤
  capacity, with the 2k−1 sizing rule in the error message). k = 1 skips
  all of it and is byte-identical to the legacy per-pass path.
- **Scheduler** (`runtime/engine/src/scheduler/frame.rs`): `FramePolicy` —
  the wait-all quorum lifted to frame granularity. The seal HOLDS until
  every awaited lane's oldest queued frame is arrival-complete (an idle
  lane between frames is a missing member); the wait is infinite by
  principle, with a 1 s watchdog that only REPORTS
  (`frame_wait_watchdog`), mirroring the per-wave quorum's
  `strict_wait_watchdog`. Membership changes only through explicit
  events: a lane joins on its first stamped fire, a pending bind holds
  the seal unconditionally (bind events are routed to the frame policy),
  graceful close releases the lane immediately while its accepted frames
  drain, terminate purges. Capacity overflow partitions the epoch under
  the quorum's round rule (`round_served`: a served lane is not
  re-awaited until the round closes; partitions are lane-disjoint and
  pipeline). Deterministic first-fit admission against per-wave
  row/token budgets; RETRY fires re-dispatch as makeups gating their
  frame's advance; frame truncation notices keep partially-submitted
  frames sealable. Controls, untracked riders, retries, and the k = 1
  quorum path reuse the existing machinery untouched.
- **Bench inferlet**: frame-aware reactive loop (first frame = prefill
  chunk + k−1 decode slots; each later all-decode frame submits after its
  predecessor's first token; out-ring capacity 2k−1).

## Correctness gates — all green (re-run after the wait-all rework)

| Gate | Result |
|---|---|
| Engine unit tests | 369/369 |
| Dummy e2e at k=1 and k=4 | 10/10 both |
| Worker / canary / boot-smoke / dummy-driver suites | green |
| CUDA serial oracle, k=1 vs baseline engine, t ∈ {1,2,15,16,17,31,32,33,256} | exact token parity |
| CUDA serial oracle, k=16 vs baseline, same t set | exact token parity |
| 2048-request fleets at k=16 (3 consecutive) | all complete, 0 failures |
| k=1 default-path regression (2048×32 / c0-256) | 24.5k / 27.3k — in the Phase 0 band |
| `frame_wait_watchdog` reports during the instrumented fleet | 0 (no stalled gathers) |

## Phase 1 performance — wait-all frames recover most of the frame tax

| Shape | k=1 (quorum, baseline) | k=16 take-who's-there (superseded) | k=16 WAIT-ALL frames (landed) | Δ landed vs k=1 |
|---|---:|---:|---:|---:|
| 2048×32 c512 | 24.5–25.2k | 13.1–13.7k | 16.1–17.0k | ≈ −33% |
| c0/256 (512×256, c256) | 27.3k | 16.4–16.8k | 22.1k | ≈ −19% |
| 512×512 c256 | 19.3k | 13.1k | 16.8k | ≈ −13% |
| 64×1536 | 7.3k | 5.5k | 6.9k | ≈ −6% |
| 16×1900 | 3.57k | 2.24k | 3.51k | ≈ −2% |

Restoring the wait-all principle recovered +23–57% over take-who's-there
on every shape; stable narrow fleets are now at or near k = 1 parity. The
default deployment stays k = 1 (zero regression); k > 1 remains an
explicit opt-in (`PIE_FRAME_SIZE`) — the Phase 2 development vehicle.

### Attribution (instrumented, 2048×32 at k=16)

Wave-width evidence for the rework (same shape, same instrumentation):

| Scheduler | waves | mean width | median | max |
|---|---:|---:|---:|---:|
| k=16 take-who's-there | 562 | 117 | 36 | 437 |
| k=16 wait-all frames | 292 | 225 | 227 | 291 |

Wait-all halves the wave count and doubles/uniformizes the width (median ≈
mean; the take-who's-there median of 36 shows most of its waves were
near-empty stragglers). Zero `frame_wait_watchdog` reports: the fleet
keeps up with the gather; nothing ever stalls a second.

The remaining −33% on 2048×32 has two parts, both Phase 2's target:

1. **Data-dependent waves cannot overlap on the per-wave path.** At launch
   staging the CUDA driver polls each lane's predecessor-publication
   snapshot with a cross-stream d2h copy
   (`dispatch.cu`: "ptir prologue or channel readiness did not commit");
   a successor posted while its predecessor executes can only RETRY, never
   wait. Feeding frame waves at depth 2 produced a fleet-wide retry storm
   (~6k tok/s). With per-frame retirement gating (the landed design), a
   frame's waves serialize with a ~0.4–1 ms host gap per wave — k gaps per
   epoch that device-side multi-step execution deletes wholesale.
2. **Frame-boundary membership vs per-wave membership under churn.** The
   k = 1 barrier admits a freshly bound lane at the very next wave
   (~1 ms); a k = 16 epoch admits it only at the next boundary (~16
   waves), so on the churny 2048×32 shape the epoch width equilibrates at
   ~225 of 512 slots vs ~450 for the per-wave barrier. This is inherent
   to frame granularity, not to the wait-all rule; it fades as epochs get
   cheap (Phase 2's O(1) host interaction per epoch) and on stable fleets
   (64×1536 and 16×1900 are already at −6%/−2%).

Historical note (superseded experiments, kept for the record): eager
sealing (≤2 epochs whenever candidates exist) and half-phase sealing both
collapse into a stable fast-narrow-epoch attractor (mean width 12, ~5.7k
tok/s) because with take-who's-there membership, epoch duration is itself
the arrival-gather window. The wait-all rule removes that failure mode by
construction — width is set by membership, not by arrival timing.

### Design consequences recorded for Phase 2

- **Granularity today**: only the quorum gather/seal is frame-level.
  Driver `prepare` (physical KV admission lease) and commit/settlement
  (publication, retirement) still run per WAVE — the per-wave driver path
  is the Phase 1 contract. Frame-level prepare (declarative frame-delta
  KV admission at seal, one lease per frame) and frame-boundary
  completion/settlement are the Phase 2 driver deliverables.
- Phase 2's device-side multi-step execution removes both residual costs
  at once: in-frame steps advance on device (no per-wave completion round
  trip, no cross-stream readiness poll), and boundary-quantized admission
  stops hurting because per-epoch host interaction is O(1) instead of
  O(k).
- Phase 2 entry points identified during this work:
  `driver/cuda/src/pipeline/dispatch.cu` (staging readiness → per-step
  device advance + per-wave publication/doorbell),
  `driver/cuda/src/context.cpp` (frame completion),
  `driver/cuda/src/batch/forward_graph.hpp` (per-wave graph classes),
  `driver/cuda/src/runahead.hpp` (staging depth constants shared with the
  scheduler).
- If churn workloads still under-fill epochs after Phase 2, the remaining
  lever is deliberate fleet partitioning into phase-offset epoch groups
  (deterministic split, e.g. by lane id); rejected for Phase 1 because it
  halves wave density on the per-wave path.

## Notes and deviations

- §5 staged-class validation rejects the `Channel::writer` late-put
  run-ahead pattern at k > 1 by design (§9: stage before submit).
  `direct-channel-e2e` pre-stages on frame deployments and keeps late-put
  coverage at k = 1.
- Physical-KV admission remains at the existing layers (contention
  orchestrator at preparation + per-launch driver prepare lease); seal
  admission is row/token arithmetic. Declarative frame-delta admission
  lands with Phase 2's eager frame binding.
- The CLI `--driver dummy` bench path fails config parsing on this branch
  independently of these changes (reproduced at the pristine baseline).
- `sdk/python-server` uv cache-keys were missing the runtime/driver/WIT
  trees, so engine edits could serve a stale cached wheel (and copying the
  .so into the source tree masks the bench's mtime staleness guard); the
  cache keys are fixed in this branch.
